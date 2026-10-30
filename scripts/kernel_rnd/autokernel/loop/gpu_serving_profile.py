#!/usr/bin/env python3
"""GPU SERVING profile as a planner input (G2, AK long-context audit 2026-10-04 §4.2).

Before this, a selected GPU serving target printed "selected GPU serving profile
unavailable" and the planner chose kernels blind; the only GPU profile was a
`llama-bench -p/-n` rocprofv3 pass at KV depth 0 (`hotspots.profile`). This module
folds the working approach of `tmp/rocprof-longctx-20261004/rocprof_longctx.py` into the
loop:

* **The server runs under the side-loaded rocprofv3 for its whole life.** ROCm 6.2's
  rocprofv3 has no collection-delay or period option, so the trace covers load to
  exit and measurement WINDOWS are cut out of it afterwards by timestamp marks. Each
  mark records wall + CLOCK_BOOTTIME + CLOCK_MONOTONIC at one instant; the trace clock
  is chosen by (a) every dispatch inside [launch, exited] and (b) a deliberate idle
  FIDUCIAL gap whose end must coincide with its mark. The wrapper exec()s the server,
  so the PID we start IS the server, and the trace is written on a graceful SIGTERM
  exit -- hence the long TERM grace (a SIGKILL loses the whole trace).
* **Host threads on the GPU lane**: `numactl --membind=3 -- taskset -c 184-191`
  (SMT siblings of cores 88-95, NUMA node 3). The caller holds the `mi210_0` device
  claim (the loop holds it for a GPU run's whole life) and the orchestrator's gpu-quiet
  lock EXCLUSIVE (`run._gpu_quiet_measurement_window`) around this call.
* **Per-kernel tables with registers, spills and occupancy** (like
  `tmp/lb1-profile-20261003/`). rocprofv3 6.2's kernel trace carries no VGPR columns,
  so register/spill/LDS figures come from the AMDGPU code-object metadata of the
  launched build's own `libggml-hip.so` (static, CPU-only: no extra GPU pass), joined
  by demangled kernel name; dynamic LDS and scratch come from the trace row itself.
  Occupancy is a gfx90a estimate (VGPR/AGPR unified file, SGPR file, LDS), labelled so.
* **One anchor launch per anchor change**, cached under the store by anchor commit,
  launch execution digest and request digest (`cache_key`).
* **Launch-weighted kernel order (AKX-ALL-29.2a, operator ruling 2026-10-20)**: each
  window records the tokens it processed (prefill: prompt tokens; decode: decoded
  tokens apportioned to the window at each request's own rate), so every kernel row
  carries `launches_per_token` and `ns_per_token` = launches per token x mean time at
  that window's shape. `launch_weighted` sums `ns_per_token` over the observed windows;
  it ORDERS the kernels the planner is shown (`hotspot_rows`) and the candidates the
  serialized tail times first (`kernel_weight`). It never gates: a keep still needs a
  measured end-to-end win. A profile without token counts (older cache) falls back to
  the primary window's device-time order, labelled so.
* **Windows**: `prefill`, `short_decode`, `concurrent_decode` from the target's own
  frozen requests; `long_decode` and `prefill_at_depth` through `LongContextHook`, the
  seam for the long-context surface (a separate branch builds it). A window that
  cannot be measured is recorded SKIPPED with its reason, never dropped.

Process discipline: only the PID this module started is signalled (TERM, then KILL
after the grace), and its death is verified.
"""
from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import struct
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
import urllib.error
import urllib.request

SCHEMA = "epyc.autokernel.gpu_serving_profile.v1"
STORE_DIR = "gpu-serving-profiles"
GPU_HOST_CPU_LIST = "184-191"
GPU_HOST_NUMA_NODE = "3"
GPU_HOST_PREFIX = ("numactl", f"--membind={GPU_HOST_NUMA_NODE}", "--",
                   "taskset", "-c", GPU_HOST_CPU_LIST)
TRACE_BASENAME = "serving"
ROCPROF_FLAGS = ("--kernel-trace", "--memory-copy-trace", "--stats",
                 "--output-format", "csv")
FIDUCIAL_S = 6.0
TERM_GRACE_S = 600.0
WINDOWS = ("prefill", "short_decode", "concurrent_decode", "long_decode", "prefill_at_depth")
LONG_CONTEXT_WINDOWS = ("long_decode", "prefill_at_depth")
TABLE_ROWS = 16
# gfx90a (CDNA2) per-SIMD limits for the occupancy ESTIMATE.
GFX90A_MAX_WAVES_PER_SIMD = 8
GFX90A_VGPR_FILE = 512          # unified arch+acc VGPRs per lane
GFX90A_VGPR_GRANULE = 8
GFX90A_SGPR_FILE = 800
GFX90A_SGPR_GRANULE = 16
GFX90A_LDS_PER_CU = 65536
GFX90A_SIMDS_PER_CU = 4
WAVE = 64


from .hotspots import ProfileFailed as _HotspotProfileFailed


class ProfileFailed(_HotspotProfileFailed):
    """The serving profile could not be produced; the planner is told so."""


class LongContextHook(Protocol):
    """Seam for the long-context surface (KV filled once, restored per launch).

    `identity` joins the cache key. `window_requests(window, launch)` returns the
    request bodies for `long_decode` / `prefill_at_depth`, or None to skip that
    window with `skip_reason(window)`.
    """
    identity: str

    def window_requests(self, window: str, launch: Any) -> Sequence[bytes] | None: ...

    def skip_reason(self, window: str) -> str: ...


@dataclass(frozen=True)
class WindowPlan:
    name: str
    bodies: tuple[bytes, ...]
    #: "prefill": send -> done (n_predict 1, no prompt cache); "decode": first token ->
    #: done for one request, last first-token -> first done for concurrent requests.
    kind: str


# ------------------------------------------------------------------ requests / plan

def _mutate(body: bytes, **fields: Any) -> bytes:
    row = json.loads(body)
    if not isinstance(row, dict):
        raise ProfileFailed("frozen request body is not a JSON object")
    row.update(fields)
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode()


def plan_windows(frozen_requests: Sequence[tuple[str, bytes]], *, np: int,
                 hook: LongContextHook | None = None, launch: Any = None
                 ) -> tuple[list[WindowPlan], dict[str, str]]:
    """(measurable windows, {skipped window: reason}) from the target's own requests."""
    if not frozen_requests:
        raise ProfileFailed("no frozen requests to profile")
    first = frozen_requests[0][1]
    plans = [WindowPlan("prefill", (_mutate(first, stream=True, n_predict=1,
                                            cache_prompt=False),), "prefill"),
             WindowPlan("short_decode", (_mutate(first, stream=True),), "decode")]
    skipped: dict[str, str] = {}
    if np > 1 and len(frozen_requests) > 1:
        plans.append(WindowPlan("concurrent_decode", tuple(
            _mutate(body, stream=True) for _pid, body in frozen_requests[:np]), "decode"))
    else:
        skipped["concurrent_decode"] = f"recipe serves np={np}; no concurrent slots"
    for name in LONG_CONTEXT_WINDOWS:
        bodies = None if hook is None else hook.window_requests(name, launch)
        if not bodies:
            skipped[name] = ("no long-context surface installed (LongContextHook)"
                             if hook is None else hook.skip_reason(name))
            continue
        kind = "prefill" if name == "prefill_at_depth" else "decode"
        plans.append(WindowPlan(name, tuple(_mutate(b, stream=True) for b in bodies), kind))
    return plans, skipped


def request_digest(plans: Sequence[WindowPlan]) -> str:
    return hashlib.sha256(json.dumps(
        [[p.name, p.kind, [hashlib.sha256(b).hexdigest() for b in p.bodies]] for p in plans],
        separators=(",", ":")).encode()).hexdigest()


def cache_key(*, anchor_commit: str, execution_digest: str, request_digest_: str,
              hook_identity: str | None = None) -> str:
    return hashlib.sha256(json.dumps({
        "schema": SCHEMA, "anchor_commit": anchor_commit,
        "execution_digest": execution_digest, "request_digest": request_digest_,
        "windows": list(WINDOWS), "long_context": hook_identity,
        "host_lane": list(GPU_HOST_PREFIX)}, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ launch command

def _set_flag(argv: list[str], flag: str, value: str) -> list[str]:
    out = list(argv)
    if flag in out:
        out[out.index(flag) + 1] = value
    else:
        out += [flag, value]
    return out


def profile_command(command_argv: Sequence[str], launch_env: Mapping[str, str], *,
                    port: int, trace_dir: Path, rocprof: str,
                    profiler_env: Callable[[Path, str], dict]) -> tuple[list[str], dict]:
    """The production argv on a scratch port, on the GPU host lane, under rocprofv3.

    `command_argv` is the target's resolved server command WITHOUT any topology prefix;
    the GPU lane prefix replaces whatever the serving launch carried. The env is the
    launch env with the profiler's loader overlay (the build's own dir first).
    """
    argv = [str(item) for item in command_argv]
    if not argv or Path(argv[0]).name != "llama-server":
        raise ProfileFailed(f"profile command must start at llama-server, got {argv[:1]}")
    if port == 8083:
        raise ProfileFailed("refusing production port 8083 for a profile launch")
    argv = _set_flag(_set_flag(argv, "--port", str(port)), "--host", "127.0.0.1")
    overlay = profiler_env(Path(argv[0]), rocprof)
    env = dict(launch_env)
    for key in ("LD_LIBRARY_PATH", "ROCPROFILER_METRICS_PATH", "ROCP_METRICS_PATH"):
        if key in overlay:
            env[key] = overlay[key]
    env.pop("LD_PRELOAD", None)
    env.setdefault("PATH", os.environ.get("PATH", "/usr/bin:/bin"))
    command = [*GPU_HOST_PREFIX, rocprof, *ROCPROF_FLAGS, "-d", str(trace_dir),
               "-o", TRACE_BASENAME, "--", *argv]
    return command, env


# ------------------------------------------------------------------ trace parsing

@dataclass(frozen=True)
class Dispatch:
    start: int
    end: int
    name: str
    private_segment: int
    group_segment: int
    workgroup: int


def normalize_name(name: str) -> str:
    """Kernel identity shared by trace rows and code-object metadata."""
    text = name.strip().strip('"')
    if text.startswith("void "):
        text = text[5:]
    depth = 0
    for index, char in enumerate(text):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char == "(" and depth == 0:
            text = text[:index]
            break
    return text.replace(".kd", "").strip()


def _int(row: Mapping, *names: str, default: int = 0) -> int:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                return int(float(value))
            except ValueError:
                continue
    return default


def parse_trace(csv_text: str) -> list[Dispatch]:
    rows = []
    for row in csv.DictReader(io.StringIO(csv_text)):
        name = (row.get("Kernel_Name") or row.get("kernel_name") or "").strip()
        start, end = _int(row, "Start_Timestamp", "start_ns"), _int(row, "End_Timestamp", "end_ns")
        if not name or end <= start:
            continue
        workgroup = (_int(row, "Workgroup_Size_X", default=1) * _int(row, "Workgroup_Size_Y", default=1)
                     * _int(row, "Workgroup_Size_Z", default=1))
        rows.append(Dispatch(start, end, name, _int(row, "Private_Segment_Size"),
                             _int(row, "Group_Segment_Size"), workgroup))
    rows.sort(key=lambda item: item.start)
    return rows


def _offsets(marks: Sequence[Mapping]) -> dict[str, float]:
    """trace_ns = wall_s * 1e9 + offset, per candidate clock (median over marks)."""
    out = {}
    for clock, key in (("boot", "boot_ns"), ("mono", "mono_ns")):
        values = [m[key] - m["wall"] * 1e9 for m in marks if key in m]
        if values:
            out[clock] = statistics.median(values)
    return out


def _idle_gap_ends(rows: Sequence[Dispatch], min_gap_s: float) -> list[int]:
    ends, high = [], None
    for row in rows:
        if high is not None and row.start - high >= min_gap_s * 1e9:
            ends.append(row.start)
        high = row.end if high is None else max(high, row.end)
    return ends


def choose_clock(rows: Sequence[Dispatch], marks: Sequence[Mapping]) -> dict:
    """Pick the trace clock: in-bounds AND the fiducial idle gap ends at its mark."""
    by_name = {m["name"]: m for m in marks}
    offsets = _offsets(marks)
    gap_ends = _idle_gap_ends(rows, max(1.0, FIDUCIAL_S - 2.0)) if rows else []
    record: dict[str, Any] = {"offsets_ns": offsets, "candidates": {}}
    launch, exited = by_name.get("launch"), by_name.get("exited") or by_name.get("teardown")
    fid = by_name.get("fid_end")
    for clock, offset in offsets.items():
        cand: dict[str, Any] = {}
        if rows and launch and exited:
            first = (rows[0].start - offset) / 1e9
            last = (max(r.end for r in rows) - offset) / 1e9
            cand["in_bounds"] = first >= launch["wall"] - 1.0 and last <= exited["wall"] + 1.0
        if fid and gap_ends:
            target = fid["wall"] * 1e9 + offset
            nearest = min(gap_ends, key=lambda end: abs(end - target))
            cand["fiducial_ms"] = round((nearest - target) / 1e6, 2)
            cand["fiducial_ok"] = -500.0 <= cand["fiducial_ms"] <= 3000.0
        record["candidates"][clock] = cand
        if cand.get("in_bounds") and cand.get("fiducial_ok"):
            record.update(method=f"clock:{clock}", offset_ns=offset)
            return record
    if fid and gap_ends:
        nearest = min(gap_ends, key=lambda end: abs(end - fid["wall"] * 1e9
                                                    - offsets.get("boot", 0.0)))
        record.update(method="gap-match", offset_ns=nearest - fid["wall"] * 1e9)
        return record
    record.update(method="UNALIGNED", offset_ns=None)
    return record


def cut(rows: Sequence[Dispatch], offset_ns: float, start_wall: float,
        end_wall: float) -> list[Dispatch]:
    """Dispatches whose START lies in [start, end] (wall seconds)."""
    starts = [row.start for row in rows]
    lo = bisect.bisect_left(starts, start_wall * 1e9 + offset_ns)
    hi = bisect.bisect_right(starts, end_wall * 1e9 + offset_ns)
    return list(rows[lo:hi])


def _busy_ns(rows: Sequence[Dispatch]) -> int:
    busy, current = 0, None
    for row in sorted(rows, key=lambda item: item.start):
        if current is None or row.start > current[1]:
            if current is not None:
                busy += current[1] - current[0]
            current = [row.start, row.end]
        else:
            current[1] = max(current[1], row.end)
    if current is not None:
        busy += current[1] - current[0]
    return busy


# ------------------------------------------------------------------ resources / occupancy

def occupancy(*, vgpr: int | None, agpr: int | None = 0, sgpr: int | None = None,
              lds_bytes: int = 0, workgroup: int = 0) -> dict:
    """gfx90a waves-per-SIMD ESTIMATE and its limiting resource."""
    limits: dict[str, int] = {"max": GFX90A_MAX_WAVES_PER_SIMD}
    if vgpr is not None:
        total = ((int(vgpr) + 3) // 4) * 4 + int(agpr or 0)
        total = max(GFX90A_VGPR_GRANULE,
                    -(-total // GFX90A_VGPR_GRANULE) * GFX90A_VGPR_GRANULE)
        limits["vgpr"] = GFX90A_VGPR_FILE // total
    if sgpr:
        alloc = -(-int(sgpr) // GFX90A_SGPR_GRANULE) * GFX90A_SGPR_GRANULE
        limits["sgpr"] = GFX90A_SGPR_FILE // alloc
    if lds_bytes > 0 and workgroup > 0:
        groups = GFX90A_LDS_PER_CU // int(lds_bytes)
        waves_per_group = -(-int(workgroup) // WAVE)
        limits["lds"] = (groups * waves_per_group) // GFX90A_SIMDS_PER_CU
    waves = min(limits.values())
    limiter = min((name for name in limits), key=lambda name: (limits[name], name != "max"))
    return {"waves_per_simd": max(0, waves), "limiter": limiter, "estimate": True}


_BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"


def _msgpack(data: bytes, i: int = 0):
    """Minimal msgpack decoder for the AMDGPU metadata note (maps/arrays/str/int/bool)."""
    c = data[i]
    i += 1
    if c <= 0x7F:
        return c, i
    if c >= 0xE0:
        return c - 256, i
    if 0x80 <= c <= 0x8F or c in (0xDE, 0xDF):
        if c <= 0x8F:
            n = c & 0x0F
        else:
            width = 2 if c == 0xDE else 4
            n, i = int.from_bytes(data[i:i + width], "big"), i + width
        out = {}
        for _ in range(n):
            key, i = _msgpack(data, i)
            out[key], i = _msgpack(data, i)
        return out, i
    if 0x90 <= c <= 0x9F or c in (0xDC, 0xDD):
        if c <= 0x9F:
            n = c & 0x0F
        else:
            width = 2 if c == 0xDC else 4
            n, i = int.from_bytes(data[i:i + width], "big"), i + width
        items = []
        for _ in range(n):
            item, i = _msgpack(data, i)
            items.append(item)
        return items, i
    if 0xA0 <= c <= 0xBF or c in (0xD9, 0xDA, 0xDB):
        if c <= 0xBF:
            n = c & 0x1F
        else:
            width = {0xD9: 1, 0xDA: 2, 0xDB: 4}[c]
            n, i = int.from_bytes(data[i:i + width], "big"), i + width
        return data[i:i + n].decode("utf-8", "replace"), i + n
    if c in (0xC4, 0xC5, 0xC6):
        width = {0xC4: 1, 0xC5: 2, 0xC6: 4}[c]
        n, i = int.from_bytes(data[i:i + width], "big"), i + width
        return data[i:i + n], i + n
    if c == 0xC0:
        return None, i
    if c in (0xC2, 0xC3):
        return c == 0xC3, i
    if c == 0xCA:
        return struct.unpack_from(">f", data, i)[0], i + 4
    if c == 0xCB:
        return struct.unpack_from(">d", data, i)[0], i + 8
    if 0xCC <= c <= 0xCF:
        width = 1 << (c - 0xCC)
        return int.from_bytes(data[i:i + width], "big"), i + width
    if 0xD0 <= c <= 0xD3:
        width = 1 << (c - 0xD0)
        return int.from_bytes(data[i:i + width], "big", signed=True), i + width
    raise ValueError(f"unsupported msgpack tag 0x{c:02x}")


def _elf_amdgpu_metadata(elf: bytes):
    if elf[:4] != b"\x7fELF" or elf[4] != 2:
        return
    shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    shentsize, shnum = struct.unpack_from("<HH", elf, 0x3A)
    for index in range(shnum):
        header = shoff + index * shentsize
        if struct.unpack_from("<I", elf, header + 4)[0] != 7:  # SHT_NOTE
            continue
        offset, size = struct.unpack_from("<QQ", elf, header + 0x18)
        cursor = offset
        while cursor + 12 <= offset + size:
            namesz, descsz, kind = struct.unpack_from("<III", elf, cursor)
            cursor += 12
            name = elf[cursor:cursor + namesz]
            cursor += (namesz + 3) & ~3
            desc = elf[cursor:cursor + descsz]
            cursor += (descsz + 3) & ~3
            if name.rstrip(b"\0") == b"AMDGPU" and kind == 32:  # NT_AMDGPU_METADATA
                yield _msgpack(desc)[0]


def code_object_kernels(fatbin: bytes, *, target: str = "gfx90a") -> list[dict]:
    """Kernel metadata from every offload bundle for `target` in a .hip_fatbin blob."""
    kernels: list[dict] = []
    position = 0
    while True:
        start = fatbin.find(_BUNDLE_MAGIC, position)
        if start < 0:
            return kernels
        count = struct.unpack_from("<Q", fatbin, start + 24)[0]
        cursor = start + 32
        for _ in range(count):
            offset, size, triple_len = struct.unpack_from("<QQQ", fatbin, cursor)
            cursor += 24
            triple = fatbin[cursor:cursor + triple_len]
            cursor += triple_len
            if size and target.encode() in triple:
                for metadata in _elf_amdgpu_metadata(fatbin[start + offset:start + offset + size]):
                    kernels.extend(metadata.get("amdhsa.kernels") or ())
        position = start + len(_BUNDLE_MAGIC)


def _hip_fatbin(library: Path) -> bytes:
    """The `.hip_fatbin` section of a HIP shared library (section headers, ELF64)."""
    data = Path(library).read_bytes()
    if data[:4] != b"\x7fELF":
        raise ProfileFailed(f"{library} is not an ELF object")
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    strtab_off = struct.unpack_from("<Q", data, shoff + shstrndx * shentsize + 0x18)[0]
    for index in range(shnum):
        header = shoff + index * shentsize
        name_off = struct.unpack_from("<I", data, header)[0]
        end = data.index(b"\0", strtab_off + name_off)
        if data[strtab_off + name_off:end] == b".hip_fatbin":
            offset, size = struct.unpack_from("<QQ", data, header + 0x18)
            return data[offset:offset + size]
    raise ProfileFailed(f"{library} carries no .hip_fatbin section")


def _demangle(names: Sequence[str]) -> list[str]:
    tool = shutil.which("c++filt")
    if not tool or not names:
        return list(names)
    done = subprocess.run([tool], input="\n".join(names), capture_output=True, text=True,
                          timeout=120)
    out = done.stdout.splitlines()
    return out if done.returncode == 0 and len(out) == len(names) else list(names)


def kernel_resources(library: Path, *, demangle: Callable[[Sequence[str]], list[str]] = _demangle
                     ) -> dict[str, dict]:
    """{normalized kernel name: registers/spills/scratch/LDS} for one build's HIP lib."""
    kernels = code_object_kernels(_hip_fatbin(Path(library).resolve()))
    names = demangle([str(k.get(".name", "")) for k in kernels])
    table: dict[str, dict] = {}
    for kernel, name in zip(kernels, names):
        table.setdefault(normalize_name(name), {
            "vgpr": kernel.get(".vgpr_count"), "agpr": kernel.get(".agpr_count", 0),
            "sgpr": kernel.get(".sgpr_count"),
            "vgpr_spill": kernel.get(".vgpr_spill_count", 0),
            "sgpr_spill": kernel.get(".sgpr_spill_count", 0),
            "scratch_bytes": kernel.get(".private_segment_fixed_size", 0),
            "lds_static_bytes": kernel.get(".group_segment_fixed_size", 0),
            "max_workgroup": kernel.get(".max_flat_workgroup_size")})
    return table


def kernel_table(rows: Sequence[Dispatch], resources: Mapping[str, Mapping], *,
                 limit: int = TABLE_ROWS, tokens: float | None = None) -> list[dict]:
    grouped: dict[str, list[Dispatch]] = {}
    for row in rows:
        grouped.setdefault(normalize_name(row.name), []).append(row)
    total = sum(row.end - row.start for row in rows) or 1
    table = []
    for name, items in grouped.items():
        ns = sum(item.end - item.start for item in items)
        res = dict(resources.get(name) or {})
        lds = max(max(item.group_segment for item in items), int(res.get("lds_static_bytes") or 0))
        scratch = max(max(item.private_segment for item in items), int(res.get("scratch_bytes") or 0))
        workgroup = max(item.workgroup for item in items)
        occ = occupancy(vgpr=res.get("vgpr"), agpr=res.get("agpr"), sgpr=res.get("sgpr"),
                        lds_bytes=lds, workgroup=workgroup)
        table.append({
            "kernel": name, "calls": len(items), "total_ns": ns,
            "share": round(ns / total, 6), "mean_us": round(ns / len(items) / 1e3, 3),
            "vgpr": res.get("vgpr"), "agpr": res.get("agpr"), "sgpr": res.get("sgpr"),
            "vgpr_spill": res.get("vgpr_spill"), "sgpr_spill": res.get("sgpr_spill"),
            "scratch_bytes": scratch, "lds_bytes": lds, "workgroup": workgroup,
            "waves_per_simd_est": occ["waves_per_simd"] if res else None,
            "occupancy_limiter": occ["limiter"] if res else None,
            "resources": "code_object" if res else "absent",
            # AKX-ALL-29.2a: launches per token x mean time at this window's shape.
            "launches_per_token": (round(len(items) / tokens, 4) if tokens else None),
            "ns_per_token": (round(ns / tokens, 1) if tokens else None)})
    table.sort(key=lambda item: -item["total_ns"])
    return table[:limit]


def analyze(rows: Sequence[Dispatch], marks: Sequence[Mapping], windows: Mapping[str, Mapping],
            resources: Mapping[str, Mapping]) -> dict:
    """Window tables from one whole-lifetime trace (pure)."""
    clock = choose_clock(rows, marks)
    if clock.get("offset_ns") is None:
        raise ProfileFailed("trace clock could not be aligned to the window marks")
    out = {}
    for name, span in windows.items():
        picked = cut(rows, clock["offset_ns"], span["start"], span["end"])
        length_ns = max(1, int((span["end"] - span["start"]) * 1e9))
        tokens = span.get("tokens")
        tokens = float(tokens) if isinstance(tokens, (int, float)) and tokens > 0 else None
        out[name] = {"status": "observed" if picked else "empty",
                     "window_s": round(length_ns / 1e9, 3), "dispatches": len(picked),
                     "kernel_ns": sum(r.end - r.start for r in picked),
                     "busy_fraction": round(_busy_ns(picked) / length_ns, 4),
                     "tokens": round(tokens, 2) if tokens else None,
                     "kernels": kernel_table(picked, resources, tokens=tokens)}
    return {"clock": clock, "windows": out}


# ------------------------------------------------------------------ lifecycle

def _mark(marks: list, name: str, **extra) -> dict:
    row = {"name": name, "wall": time.time(), "mono_ns": time.monotonic_ns(),
           "boot_ns": time.clock_gettime_ns(time.CLOCK_BOOTTIME), **extra}
    marks.append(row)
    return row


def _stream(url: str, body: bytes, timeout_s: float) -> dict:
    """POST a streaming completion; return {sent, first, done} wall times + error."""
    times: dict[str, Any] = {"sent": time.time(), "first": None, "done": None, "error": None,
                             "tokens_predicted": None, "tokens_evaluated": None}
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if times["first"] is None and event.get("content") and "prompt_progress" not in event:
                    times["first"] = time.time()
                if event.get("stop"):
                    # AKX-ALL-29.2a: the server's own counts (final event, else timings).
                    timings = event.get("timings") or {}
                    for key, alt in (("tokens_predicted", "predicted_n"),
                                     ("tokens_evaluated", "prompt_n")):
                        value = event.get(key, timings.get(alt))
                        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                            times[key] = value
                    break
    except (OSError, ValueError, urllib.error.URLError) as exc:
        times["error"] = f"{type(exc).__name__}: {exc}"[:256]
    times["done"] = time.time()
    return times


def _wait_healthy(port: int, proc, timeout_s: float) -> float:
    started = time.time()
    while time.time() - started < timeout_s:
        if proc.poll() is not None:
            raise ProfileFailed(f"profiled server exited {proc.returncode} during load")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return time.time() - started
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    raise ProfileFailed("profiled server not healthy within the boot timeout")


def _stop_own(proc, grace_s: float) -> str:
    """TERM our own child, wait for the trace flush, escalate, verify death."""
    if proc.poll() is not None:
        return f"exited {proc.returncode}"
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(grace_s)
        outcome = "terminated"
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(30)
        outcome = "killed (trace likely lost)"
    if proc.poll() is None:
        raise ProfileFailed(f"profiled server {proc.pid} survived SIGKILL")
    return outcome


def _window_span(plan: WindowPlan, results: Sequence[Mapping]) -> dict | None:
    if any(r.get("error") for r in results):
        return None
    if plan.kind == "prefill":
        span = {"start": min(r["sent"] for r in results), "end": max(r["done"] for r in results)}
        return {**span, "tokens": _prefill_tokens(results)}
    firsts = [r["first"] for r in results]
    if None in firsts:
        return None
    start, end = max(firsts), min(r["done"] for r in results)
    if end <= start:
        return None
    return {"start": start, "end": end, "tokens": _decode_tokens(results, start, end)}


def _prefill_tokens(results: Sequence[Mapping]) -> int | None:
    """Prompt tokens the window processed; None unless EVERY request reported its count."""
    counts = [r.get("tokens_evaluated") for r in results]
    if not counts or any(not isinstance(c, int) or c <= 0 for c in counts):
        return None
    return sum(counts)


def _decode_tokens(results: Sequence[Mapping], start: float, end: float) -> float | None:
    """Decoded tokens inside [start, end]: each request's tokens after its first, spread
    uniformly over its own first->done interval and clipped to the window (one request:
    the whole interval). None unless every request reported its count."""
    total = 0.0
    for r in results:
        predicted, first, done = r.get("tokens_predicted"), r.get("first"), r.get("done")
        if not isinstance(predicted, int) or first is None or done is None or done <= first:
            return None
        overlap = max(0.0, min(end, done) - max(start, first))
        total += max(0, predicted - 1) * overlap / (done - first)
    return round(total, 2) if total > 0 else None


def run(*, command_argv: Sequence[str], launch_env: Mapping[str, str], port: int,
        plans: Sequence[WindowPlan], out_dir: Path, rocprof: str,
        profiler_env: Callable[[Path, str], dict], resources: Mapping[str, Mapping],
        popen: Callable = subprocess.Popen, stream: Callable = _stream,
        wait_healthy: Callable = _wait_healthy, sleep: Callable = time.sleep,
        sampler_factory: Callable | None = None, boot_timeout_s: float = 600.0,
        request_timeout_s: float = 1800.0, term_grace_s: float = TERM_GRACE_S) -> dict:
    """One profiled anchor launch; returns the analysed record (also written to disk)."""
    from . import hip_launch_proof, residency
    binary = Path(command_argv[0])
    try:
        with binary.open("rb") as handle:
            is_elf = handle.read(4) == b"\x7fELF"
    except OSError:
        is_elf = False
    if not is_elf or not (binary.parent / "libggml-hip.so").exists():
        raise ProfileFailed(f"{binary} is not a HIP llama-server build (ELF + libggml-hip.so); "
                            "nothing to profile")
    out_dir = Path(out_dir)
    trace_dir = out_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    command, env = profile_command(command_argv, launch_env, port=port, trace_dir=trace_dir,
                                   rocprof=rocprof, profiler_env=profiler_env)
    marks: list[dict] = []
    spans: dict[str, dict] = {}
    failures: dict[str, str] = {}
    sampler = (sampler_factory or residency.Sampler)()
    base = f"http://127.0.0.1:{port}"
    proc = None
    teardown = "not_started"
    maps = None
    with open(out_dir / "server.log", "wb") as log, sampler:
        try:
            _mark(marks, "launch")
            proc = popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
            if callable(getattr(sampler, "watch_pid", None)):
                sampler.watch_pid(proc.pid)
            load_s = wait_healthy(port, proc, boot_timeout_s)
            _mark(marks, "healthy", load_s=round(load_s, 2))
            maps = hip_launch_proof.mapped_ggml(proc.pid, Path(command_argv[0]).parent)
            try:
                attached = "librocprofiler-sdk-tool" in Path(f"/proc/{proc.pid}/maps").read_text()
            except OSError:
                attached = None
            if attached is False:
                raise ProfileFailed("rocprofiler-sdk tool is not mapped into the server")
            # Warm the graph/caches outside every window, then the idle fiducial.
            stream(f"{base}/completion", plans[0].bodies[0], request_timeout_s)
            _mark(marks, "fid_start")
            sleep(FIDUCIAL_S)
            _mark(marks, "fid_end")
            for plan in plans:
                results: list[dict] = [{} for _ in plan.bodies]

                def one(index: int, body: bytes) -> None:
                    results[index] = stream(f"{base}/completion", body, request_timeout_s)
                threads = [threading.Thread(target=one, args=(i, b), daemon=True)
                           for i, b in enumerate(plan.bodies)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                span = _window_span(plan, results)
                if span is None:
                    failures[plan.name] = "; ".join(
                        str(r.get("error") or "no token streamed") for r in results)[:512]
                else:
                    spans[plan.name] = span
        finally:
            if proc is not None:
                _mark(marks, "teardown")
                teardown = _stop_own(proc, term_grace_s)
                _mark(marks, "exited")
    traces = sorted(trace_dir.rglob("*kernel_trace.csv"))
    if not traces:
        raise ProfileFailed(f"rocprofv3 wrote no kernel trace (teardown: {teardown})")
    rows = [row for path in traces for row in parse_trace(path.read_text(encoding="utf-8"))]
    analysed = analyze(rows, marks, spans, resources)
    for name, reason in failures.items():
        analysed["windows"][name] = {"status": "failed", "reason": reason}
    proof = dict(sampler.proof)
    residency_record = {**proof, "covers_request_phase": bool(spans)}
    hip_proof = hip_launch_proof.fold(
        residency_record, maps=maps,
        link=hip_launch_proof.linkage(Path(command_argv[0]), env.get("LD_LIBRARY_PATH")))
    if hip_proof["status"] == hip_launch_proof.REFUTED:
        raise ProfileFailed(f"HIP launch refuted for the profiled server (legs "
                            f"{hip_proof['legs']}); the trace is not a GPU profile of this build")
    analysed.update(teardown=teardown, marks=marks, spans=spans, trace_files=[str(p) for p in traces],
                    command=command, residency=proof, hip_proof=hip_proof)
    return analysed


# ------------------------------------------------------------------ cache / planner view

def store_dir(store: Path, key: str) -> Path:
    return Path(store) / STORE_DIR / key


def cached(store: Path, key: str) -> dict | None:
    path = store_dir(store, key) / "profile.json"
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return body if body.get("schema") == SCHEMA and body.get("key") == key else None


def retain(store: Path, key: str, body: Mapping) -> Path:
    directory = store_dir(store, key)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "profile.json"
    tmp = directory / ".profile.json.tmp"
    tmp.write_text(json.dumps({**body, "schema": SCHEMA, "key": key}, indent=1,
                              sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def primary_window(np: int) -> str:
    return "concurrent_decode" if np > 1 else "short_decode"


def observation(body: Mapping, *, record: Path | str, np: int,
                skipped: Mapping[str, str]) -> dict:
    """The planner-facing view: per-window tables, skips recorded, never dropped."""
    windows = dict(body.get("windows") or {})
    for name, reason in skipped.items():
        windows.setdefault(name, {"status": "skipped", "reason": reason})
    for name in WINDOWS:
        windows.setdefault(name, {"status": "skipped", "reason": "not planned"})
    primary = primary_window(np)
    if (windows.get(primary) or {}).get("status") != "observed":
        primary = next((n for n in WINDOWS if (windows.get(n) or {}).get("status") == "observed"),
                       primary)
    return {"status": "observed", "key": body.get("key"), "record": str(record),
            "primary_window": primary, "clock": (body.get("clock") or {}).get("method"),
            "hip_proof": (body.get("hip_proof") or {}).get("status"),
            "occupancy": "gfx90a estimate from code-object VGPR/AGPR/SGPR/LDS",
            "windows": {n: windows[n] for n in WINDOWS},
            "launch_weighted": launch_weighted({n: windows[n] for n in WINDOWS})}


def launch_weighted(windows: Mapping[str, Mapping], *,
                    demand: Mapping[str, float] | None = None) -> dict:
    """AKX-ALL-29.2a: kernels ordered by sum over windows of `ns_per_token` (= launches
    per token x mean time at that window's shape), each window weighted by `demand`
    (default 1.0 per observed window with a token count).

    Only observed windows WITH a token count contribute; a window table is truncated at
    `TABLE_ROWS`, so a kernel outside a window's top rows contributes 0 there (a lower
    bound). `status` is "unavailable" when no window has tokens. Ranking input only: it
    never gates a keep (operator ruling 2026-10-20)."""
    used, scores, per = [], {}, {}
    for name, window in windows.items():
        if (window or {}).get("status") != "observed" or not window.get("tokens"):
            continue
        weight = 1.0 if demand is None else float(demand.get(name, 0.0))
        if weight <= 0:
            continue
        used.append(name)
        for row in window.get("kernels") or ():
            if row.get("ns_per_token") is None:
                continue
            kernel = row["kernel"]
            scores[kernel] = scores.get(kernel, 0.0) + weight * float(row["ns_per_token"])
            per.setdefault(kernel, {})[name] = {
                "ns_per_token": row["ns_per_token"],
                "launches_per_token": row.get("launches_per_token"),
                "mean_us": row.get("mean_us"), "calls": row.get("calls"),
                "total_ns": row.get("total_ns")}
    if not used:
        return {"status": "unavailable", "reason": "no observed window carries a token count",
                "windows": [], "kernels": []}
    ranked = sorted(scores, key=lambda k: (-scores[k], k))
    return {"status": "observed", "windows": used,
            "demand": dict(demand) if demand is not None else {name: 1.0 for name in used},
            "kernels": [{"kernel": k, "weighted_ns_per_token": round(scores[k], 1),
                         "by_window": per[k]} for k in ranked]}


def _kernel_matches(kernel: str, symbol: str) -> bool:
    """`symbol` names this kernel: equal to its base name (text before the template),
    or an identifier inside its template arguments (a device function passed as one)."""
    import re
    symbol = (symbol or "").strip().split("::")[-1]
    if not symbol:
        return False
    base = kernel.split("<", 1)[0].strip()
    # The recorded GPU MMQ hypotheses name the host launcher. Its source in mmq.cuh
    # launches both the main kernel and, for Stream K, the fixup kernel. Map this
    # known route explicitly; stripping "launch_" from arbitrary symbols would
    # claim a device-kernel relationship that the source has not established.
    if symbol == "launch_mul_mat_q" and base in {"mul_mat_q", "mul_mat_q_stream_k_fixup"}:
        return True
    if base == symbol:
        return True
    return symbol in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", kernel.partition("<")[2])


def kernel_weight(view: Mapping | None, target_symbol: str | None) -> float | None:
    """The launch-weighted ns/token of the kernels `target_symbol` names, summed; None
    when the view has no launch-weighted table or no kernel matches. The serialized tail
    times candidates in descending order of this value (AKX-ALL-29.2a)."""
    table = (view or {}).get("launch_weighted") or {}
    if table.get("status") != "observed" or not target_symbol:
        return None
    hits = [row["weighted_ns_per_token"] for row in table.get("kernels") or ()
            if _kernel_matches(row["kernel"], target_symbol)]
    return round(sum(hits), 1) if hits else None


def hotspot_rows(view: Mapping) -> list:
    """`hotspots.Hotspot` rows for the planner's kernel table.

    AKX-ALL-29.2a: ordered by the launch-weighted score when the view has one, with each
    kernel's calls, device ns and device-time share summed over the windows that score
    used. Without token counts: the primary window's rows in device-time order, as before."""
    from .hotspots import Hotspot
    table = view.get("launch_weighted") or {}
    windows = view.get("windows") or {}
    if table.get("status") == "observed":
        used = table.get("windows") or []
        total = sum(int((windows.get(name) or {}).get("kernel_ns") or 0) for name in used) or 1
        rows = []
        for row in (table.get("kernels") or ())[:TABLE_ROWS]:
            ns = sum(int(cell.get("total_ns") or 0) for cell in row["by_window"].values())
            calls = sum(int(cell.get("calls") or 0) for cell in row["by_window"].values())
            rows.append(Hotspot(signature=row["kernel"], total_duration_ns=ns, calls=calls,
                                share_of_device_time=round(ns / total, 6)))
        return rows
    window = windows.get(view.get("primary_window")) or {}
    return [Hotspot(signature=row["kernel"], total_duration_ns=int(row["total_ns"]),
                    calls=int(row["calls"]), share_of_device_time=float(row["share"]))
            for row in window.get("kernels") or ()]


__all__ = ["GPU_HOST_CPU_LIST", "GPU_HOST_PREFIX", "LONG_CONTEXT_WINDOWS", "LongContextHook",
           "ProfileFailed", "SCHEMA", "WINDOWS", "WindowPlan", "analyze", "cache_key",
           "cached", "choose_clock", "code_object_kernels", "cut", "hotspot_rows",
           "kernel_resources", "kernel_table", "kernel_weight", "launch_weighted",
           "normalize_name", "observation", "occupancy",
           "parse_trace", "plan_windows", "primary_window", "profile_command",
           "request_digest", "retain", "run"]
