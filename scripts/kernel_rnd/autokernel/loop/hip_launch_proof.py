#!/usr/bin/env python3
"""Per-launch proof that a GPU serving launch actually ran on HIP (G6, 2026-10-04).

"I invoked the HIP build" is not evidence of a HIP run (CLAUDE.md, Debugging). The
global VRAM floor the residency sampler already applies is necessary but not
sufficient on a shared device: another process's allocation clears it too. A launch
is PROVEN on HIP only when all three hold:

1. **ggml linkage** -- `scripts/utils/verify_ggml_linkage.sh` passes for the launched
   binary under the launch's own `LD_LIBRARY_PATH` (the three-ggml-generations
   hazard). Run once per (binary, loader path) per process; the result is cached.
2. **KFD registration with non-zero VRAM** -- the launched PID itself appears under
   `/sys/class/kfd/kfd/proc/<pid>` with `vram_*` > 0 in a sample taken DURING the run
   (`residency.Sampler.watch_pid`).
3. **libggml-hip mapped from the build** -- `/proc/<pid>/maps` shows `libggml-hip`
   and every mapped `libggml*` resolves inside the launched build's `bin/`
   (llama.cpp dlopens the HIP backend, so `ldd` alone cannot show it).

Three-valued, like the residency record: `proven`, `unproven` (an instrument could
not be read -- recorded, never a refusal), and `refuted` (positive evidence of a
non-HIP or wrong-tree run -- the launch is refused by the serving owner).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from typing import Mapping

SCHEMA = "epyc.autokernel.hip_launch_proof.v1"
PROVEN, UNPROVEN, REFUTED = "proven", "unproven", "refuted"
#: Research repo root: autokernel/loop/<this> -> parents[4] is the repo root.
LINKAGE_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "utils" / "verify_ggml_linkage.sh"
_LINKAGE_CACHE: dict[tuple, dict] = {}


def linkage(binary: Path | str, ld_library_path: str | None, *,
            script: Path = LINKAGE_SCRIPT, timeout_s: float = 120.0) -> dict:
    """Run the linkage verifier once per (binary identity, loader path); cached.

    Exit 0 = PASS. Any other exit of a verifier that RAN is REFUTED: 1 = a ggml library
    resolves outside the tree, 2 = the script could prove nothing about this binary --
    both are "DO NOT TRUST THE MEASUREMENT" in the script's own words, and a GPU launch
    whose linkage cannot be proven is not a GPU measurement. Only a verifier that could
    not run (missing script, unreadable binary, spawn failure) is UNPROVEN.
    """
    binary = Path(binary)
    try:
        stat = binary.stat()
        identity = (str(binary.resolve()), stat.st_mtime_ns, stat.st_size)
    except OSError as exc:
        return {"status": UNPROVEN, "reason": f"binary unreadable: {exc}"[:256]}
    key = (identity, ld_library_path or "")
    if key in _LINKAGE_CACHE:
        return dict(_LINKAGE_CACHE[key], cached=True)
    if not Path(script).is_file():
        return {"status": UNPROVEN, "reason": f"linkage verifier missing: {script}"}
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    if ld_library_path:
        env["LD_LIBRARY_PATH"] = ld_library_path
    try:
        done = subprocess.run(["bash", str(script), str(binary), str(binary.parent)],
                              capture_output=True, text=True, timeout=timeout_s, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": UNPROVEN, "reason": f"linkage verifier failed to run: {exc}"[:256]}
    text = (done.stdout or "") + (done.stderr or "")
    status = PROVEN if done.returncode == 0 and "PASS" in text else REFUTED
    result = {"status": status, "rc": done.returncode,
              "script_sha256": hashlib.sha256(Path(script).read_bytes()).hexdigest(),
              "ld_library_path": ld_library_path, "tail": text[-512:]}
    _LINKAGE_CACHE[key] = result
    return dict(result, cached=False)


def mapped_ggml(pid: int, build_bin: Path | str, *, proc_root: Path = Path("/proc")) -> dict:
    """Which ggml libraries the LIVE process mapped, and from where."""
    try:
        maps = (Path(proc_root) / str(pid) / "maps").read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"status": UNPROVEN, "readable": False, "reason": f"maps unreadable: {exc}"[:256]}
    libs = sorted({line.split()[-1] for line in maps.splitlines()
                   if "libggml" in line and line.split()[-1].startswith("/")})
    hip = [lib for lib in libs if "libggml-hip" in lib]
    root = os.path.realpath(str(build_bin)) + os.sep
    foreign = [lib for lib in libs if not os.path.realpath(lib).startswith(root)]
    if not libs:
        # Not a ggml process image at all (a wrapper that has not exec'd, a recycled
        # PID): this observes nothing about the server, so it cannot refute it.
        return {"status": UNPROVEN, "readable": True, "ggml_libs": [], "hip_mapped": [],
                "foreign": [], "reason": "process maps no ggml library"}
    status = PROVEN if hip and not foreign else REFUTED
    return {"status": status, "readable": True, "ggml_libs": libs, "hip_mapped": hip,
            "foreign": foreign}


def fold(residency_record: Mapping, *, link: Mapping | None, maps: Mapping | None) -> dict:
    """Combine the three legs into one record (pure)."""
    own_reads = int(residency_record.get("own_pid_kfd_reads") or 0)
    own_vram = int(residency_record.get("own_pid_peak_vram_bytes") or 0)
    covered = bool(residency_record.get("covers_request_phase"))
    maps = dict(maps or {"status": UNPROVEN, "readable": False, "reason": "not observed"})
    link = dict(link or {"status": UNPROVEN, "reason": "not run"})
    if own_vram > 0:
        kfd = PROVEN
    elif own_reads >= 2 and covered and maps.get("ggml_libs"):
        # A live process we watched for the whole request phase never held device
        # memory under its own PID: that is a CPU run, whatever the global VRAM says.
        kfd = REFUTED
    else:
        kfd = UNPROVEN
    legs = {"linkage": link["status"], "kfd_own_vram": kfd, "maps": maps["status"]}
    status = (REFUTED if REFUTED in legs.values()
              else PROVEN if set(legs.values()) == {PROVEN} else UNPROVEN)
    return {"schema": SCHEMA, "status": status, "legs": legs, "linkage": link,
            "maps": maps, "own_pid_peak_vram_bytes": own_vram,
            "own_pid_kfd_reads": own_reads,
            "peak_kfd_processes": residency_record.get("peak_kfd_processes")}


__all__ = ["LINKAGE_SCRIPT", "PROVEN", "REFUTED", "SCHEMA", "UNPROVEN", "fold", "linkage",
           "mapped_ggml"]
