#!/usr/bin/env python3
"""Kernel feature-preservation gate: no keep or fold may silently drop a kernel path.

WHY (operator 2026-10-04: "it's important to have a process in place to get this right
immediately to avoid accidentally regressing the kernels on our models"). Two incidents
lost a working kernel feature with nobody noticing:

* INC-20260925-parallel-repack-lost-in-bundled-revert: the OpenMP tensor-repack
  parallelisation (`52ddd3200`, 1.5-2.5x CPU model load) rode in v6 Stage 1a
  `814e81782` beside two failing changes. Reverting that bundle (`358f0c748`) removed
  it too, and v7-v10 and the AutoKernel champion carried 0 omp pragmas in repack.cpp
  until 2026-09-25. Every DS41 launch paid a single-threaded ~3.5 min load. No serving
  A/B could see it: the loss was in LOAD, and no gate inventoried features.
* INC-20260706-iqk-missing-subsystem: a branch forked before the iqk port carried 0 of 8
  `GGML_IQK` references on its way to becoming v7.

Both losses show up in artefacts the loop already has, so the gate costs ~1 s:

1. STATIC (`static_manifest`): `nm -C --defined-only` over the build's ggml/llama DSOs,
   classified into kernel families -- repack `tensor_traits<block, INTER, COLS>`,
   `ggml_gemm_*`/`ggml_gemv_*` bodies, OpenMP outlined regions (`[clone ._omp_fn.N]`,
   which is exactly how the lost parallel repack appears), tinyBLAS gemm
   instantiations, `iqk_*` entry points and `ggml_compute_forward_*` op bodies. Counted,
   so losing one of two omp regions in a function is a loss.
2. SOURCE (`source_inventory`): one `git grep` per ref -- `#pragma omp` and `GGML_IQK`
   counts per file, repack trait definitions, gemm/gemv definitions, `getenv` feature
   knobs, CMake options and the `tests/` file list. Works on refs and trees, so a fold
   or forward-port is checked before anything is built.
3. RUNTIME (`runtime_manifest`): per model and serving shape, what EXECUTED -- the
   `[iqk] ACTIVE:` first-engagement lines (per quant type, dense GEMM and MoE), the
   extra-buffer sizes (`CPU_REPACK model buffer size`: a shrink means tensors lost
   their repack) and, when logged, per-tensor `repack tensor ... with <type>_<N>x<M>`.
   Taken from llama-server stderr of launches the A/B already makes: `serving.py`
   hands each launch's stderr to `open_launch_sink` when capture is enabled
   (`run.py` enables it at `<store>/kernel-coverage`), the text is compacted to a
   marker sidecar at teardown and the raw log deleted. Off by default; never a
   measurement input; a sink failure falls back to /dev/null.

THE RULE (`verdict`). A kernel key present in the base and absent (or fewer) in the
candidate is a HARD FAIL, unless the change DECLARES the replacement --
`KERNEL-REPLACES: <old key glob> => <new key glob>` in the hypothesis text, the patch's
added lines, or (fold) a commit message -- the new key is present in the candidate, and
the replacement was MEASURED: a runtime loss on model T needs an A/B on T; a static or
source loss (model-agnostic) needs an A/B on every bound target. A runtime marker that
the base's own launches disagree on is reported as unstable, never failed on.

`fold-check` (CLI below) runs layers 1-2 between a champion ref and a fold/forward-port
candidate; `fold2_gates.py` runs it as G0.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

STATIC_SCHEMA = "epyc.autokernel.kernel_static_manifest.v1"
SOURCE_SCHEMA = "epyc.autokernel.kernel_source_inventory.v1"
RUNTIME_SCHEMA = "epyc.autokernel.kernel_runtime_manifest.v1"
SIDECAR_SCHEMA = "epyc.autokernel.kernel_launch_markers.v1"
VERDICT_SCHEMA = "epyc.autokernel.kernel_preservation.v1"
CAPTURE_DIR = "kernel-coverage"
VERDICT_DIR = "kernel-preservation"

# ------------------------------------------------------------------ static (symbols)

#: DSOs whose kernels serve our models. A library present in the base and missing in
#: the candidate is itself a loss.
LIBRARIES = ("libggml-cpu.so", "libggml-base.so", "libggml.so", "libllama.so",
             "libggml-hip.so")
_CLONE = re.compile(r" \[clone \.(?!_omp_fn)[^\]]*\]")
_OMP = re.compile(r"^(.*) \[clone \._omp_fn\.\d+\]$")
_OMP_C = re.compile(r"^([A-Za-z_]\w*)\._omp_fn\.\d+$")
_COLD = re.compile(r"(?:\.cold(?:\.\d+)?|\[clone \.cold(?:\.\d+)?\])$")
_REPACK_TRAITS = re.compile(r"ggml::cpu::repack::tensor_traits<(\w+), (\d+)l?, (\d+)l?")
_GEMM = re.compile(r"^(ggml_(?:gemm|gemv)_\w+)$")
_TINYBLAS = re.compile(r"(tinyBLAS\w*<[^>]*>)::(\w+)<([^>]*)>")
_IQK = re.compile(r"^(iqk_\w+)$")
_FORWARD = re.compile(r"^(ggml_compute_forward_\w+)$")


def _depth0_head(signature: str) -> str:
    """The demangled function name without its parameter list or return type."""
    depth, head_end, last_space = 0, len(signature), -1
    for index, char in enumerate(signature):
        if char in "<{":
            depth += 1
        elif char in ">}":
            depth = max(0, depth - 1)
        elif char == "(" and depth == 0:
            if signature.startswith("operator", max(0, index - 8)) or index == 0:
                continue
            head_end = index
            break
        elif char == " " and depth == 0:
            last_space = index
    return signature[last_space + 1:head_end].strip()


def classify_symbol(name: str) -> tuple[str, str] | None:
    """(family, key) for one demangled defined symbol, or None (not a kernel path)."""
    if _COLD.search(name):
        return None   # the cold split of a function already counted
    omp = _OMP.match(name)
    if omp:
        return "omp_region", _depth0_head(_CLONE.sub("", omp.group(1)))
    omp = _OMP_C.match(name)
    if omp:
        return "omp_region", omp.group(1)
    name = _CLONE.sub("", name)
    traits = _REPACK_TRAITS.search(name)
    if traits:
        if not name.endswith("::repack(ggml_tensor*, void const*, unsigned long)"):
            return None   # one key per instantiation: its repack() method
        block, inter, cols = traits.groups()
        return "repack_traits", f"{block.removeprefix('block_')}_{cols}x{inter}"
    head = _depth0_head(name)
    tiny = _TINYBLAS.search(name)
    if tiny:
        return "tinyblas", f"{tiny.group(1)}::{tiny.group(2)}<{tiny.group(3)}>"
    for family, pattern in (("gemm_kernel", _GEMM), ("iqk", _IQK), ("forward", _FORWARD)):
        match = pattern.match(head)
        if match:
            return family, match.group(1)
    return None


def library_paths(build_dir: Path | str) -> dict[str, Path]:
    build = Path(build_dir)
    found: dict[str, Path] = {}
    for directory in (build / "bin", build):
        if not directory.is_dir():
            continue
        names = list(LIBRARIES) + sorted(p.name for p in directory.glob("libggml-cpu-*.so"))
        for name in names:
            path = directory / name
            if name not in found and path.exists():
                found[name] = path
    return found


def _nm(path: Path, *, timeout: int = 120) -> list[str]:
    result = subprocess.run(["nm", "-C", "--defined-only", str(path)], text=True,
                            capture_output=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"nm {path.name} failed: {result.stderr.strip()[:200]}")
    out = []
    for line in result.stdout.splitlines():
        parts = line.split(" ", 2)
        if len(parts) == 3 and parts[1] in {"T", "t", "W", "w"}:
            out.append(parts[2])
    return out


def static_manifest(build_dir: Path | str, *, nm=None) -> dict[str, Any]:
    """Kernel families per library, counted. Raises when no library is found."""
    nm = nm or _nm
    libraries = library_paths(build_dir)
    if not libraries:
        raise FileNotFoundError(f"no ggml/llama library under {build_dir}")
    body: dict[str, Any] = {}
    for name, path in sorted(libraries.items()):
        counts: Counter[str] = Counter()
        for symbol in nm(path):
            hit = classify_symbol(symbol)
            if hit is not None:
                counts[f"{hit[0]}:{hit[1]}"] += 1
        body[name] = dict(sorted(counts.items()))
    return {"schema": STATIC_SCHEMA, "build_dir": str(build_dir), "libraries": body}


# ------------------------------------------------------------------ source (git)

SOURCE_PATHS = ("ggml/src", "ggml/include", "src", "common", "tools/server", "tests",
                "CMakeLists.txt", "ggml/CMakeLists.txt")
_SOURCE_NEEDLES = ("pragma omp", "GGML_IQK", "IQK_MULMAT", "tensor_traits<", "ggml_gemm_",
                   "ggml_gemv_", "getenv(", "option(")
_SOURCE_RULES: tuple[tuple[str, re.Pattern, str], ...] = (
    ("omp_pragma", re.compile(r"#\s*pragma\s+omp\b"), "file"),
    ("iqk_ref", re.compile(r"GGML_IQK|GGML_USE_IQK_MULMAT"), "file"),
    ("repack_traits_def", re.compile(
        r"static\s+const\s+ggml::cpu::repack::tensor_traits<[^;]*>\s+(\w+)\s*;"), "name"),
    ("gemm_def", re.compile(r"^\s*void\s+(ggml_(?:gemm|gemv)_\w+)\s*\("), "name"),
    ("env_knob", re.compile(r"getenv\(\s*\"(\w+)\""), "name"),
    ("cmake_option", re.compile(r"^\s*option\(\s*(\w+)"), "name"),
)
_SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".txt", ".cmake")


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], text=True,
                            capture_output=True, check=False)
    if check and result.returncode not in (0, 1):   # 1: git grep found nothing
        raise RuntimeError(f"git {' '.join(args[:3])} failed: {result.stderr.strip()[:300]}")
    return result.stdout


def source_inventory(repo: Path | str, ref: str, *, paths: Sequence[str] = SOURCE_PATHS,
                     git=_git) -> dict[str, Any]:
    """Feature inventory of one ref (commit or tree), without a checkout."""
    repo = Path(repo)
    resolved = git(repo, "rev-parse", "--verify", f"{ref}^{{tree}}").strip()
    if not resolved:
        raise RuntimeError(f"{ref} does not resolve to a tree in {repo}")
    argv = ["grep", "-n", "-I", "-F"]
    for needle in _SOURCE_NEEDLES:
        argv += ["-e", needle]
    output = git(repo, *argv, resolved, "--", *paths)
    counts: Counter[str] = Counter()
    prefix = resolved + ":"
    for line in output.splitlines():
        if not line.startswith(prefix):
            continue
        path, _, rest = line[len(prefix):].partition(":")
        _lineno, _, content = rest.partition(":")
        if not path.endswith(_SOURCE_SUFFIXES):
            continue
        for family, pattern, keyed in _SOURCE_RULES:
            if keyed == "file":
                if pattern.search(content):
                    counts[f"{family}:{path}"] += 1
            else:
                for match in pattern.finditer(content):
                    counts[f"{family}:{match.group(1)}"] += 1
    for path in git(repo, "ls-tree", "-r", "--name-only", resolved, "--", "tests").split():
        counts[f"test_file:{path}"] += 1
    return {"schema": SOURCE_SCHEMA, "ref": ref, "tree": resolved,
            "inventory": dict(sorted(counts.items()))}


# ------------------------------------------------------------------ runtime (stderr)

_CAPTURE: dict[str, Any] = {"root": None, "per_shape": 3, "keep_fingerprints": 64}
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_IQK_GEMM = re.compile(r"\[iqk\] ACTIVE: ik_llama GEMM kernels engaged \(first mul_mat "
                       r"type=(\d+) activation=(\d+)")
_IQK_MOE = re.compile(r"\[iqk\] ACTIVE: MoE mul_mat_id via ik kernels \(type=(\d+) "
                      r"activation=(\d+)")
_ACTIVE = re.compile(r"^\[([\w.-]+)\] ACTIVE: (.+)$")
_BUFFER = re.compile(r"\b([\w.]+) model buffer size =\s*([\d.]+) MiB")
_REPACK_TENSOR = re.compile(r"repack tensor (\S+) with (\w+)")
_NUMERIC_PARAM = re.compile(r"\b(?:ne\d*|n_\w+)=\d+")
#: ggml_type ids (ggml.h) for readable keys; an unknown id stays numeric.
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
              9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
              15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
              20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
              26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
              35: "TQ2_0", 39: "MXFP4"}
#: Buffers whose SIZE is a kernel-path quantity (bytes routed through a repacked
#: layout). Device/mapped buffers are placement, recorded but never failed on.
EXTRA_BUFFER_MARKERS = ("REPACK", "AMX", "KLEIDI")


def _type(value: str) -> str:
    return GGML_TYPES.get(int(value), f"type{value}")


def parse_markers(text: str) -> dict[str, Any]:
    """Kernel-path markers in one llama-server/llama-bench stderr text."""
    markers: set[str] = set()
    buffers: Counter[str] = Counter()
    for raw in text.splitlines():
        line = _ANSI.sub("", raw).strip()
        gemm, moe = _IQK_GEMM.search(line), _IQK_MOE.search(line)
        if gemm:
            markers.add(f"iqk.gemm:{_type(gemm.group(1))}:act={_type(gemm.group(2))}")
            continue
        if moe:
            markers.add(f"iqk.moe:{_type(moe.group(1))}:act={_type(moe.group(2))}")
            continue
        active = _ACTIVE.match(line)
        if active:
            text_key = re.sub(r"\s+", " ", _NUMERIC_PARAM.sub("", active.group(2))).strip()
            markers.add(f"active.{active.group(1)}:{text_key}")
            continue
        buffer = _BUFFER.search(line)
        if buffer:
            buffers[buffer.group(1)] += float(buffer.group(2))
            continue
        repack = _REPACK_TENSOR.search(line)
        if repack:
            role = re.sub(r"\.\d+\.", ".*.", repack.group(1))
            markers.add(f"repack.tensor:{role}:{repack.group(2)}")
    for name in buffers:
        markers.add(f"buffer:{name}")
    return {"markers": sorted(markers),
            "buffers_mib": {k: round(v, 2) for k, v in sorted(buffers.items())}}


def enable_capture(root: Path | str | None, *, per_shape: int = 3,
                   keep_fingerprints: int = 64) -> None:
    """Turn launch-stderr capture on (a directory) or off (None). Process-global."""
    _CAPTURE.update(root=None if root is None else Path(root), per_shape=per_shape,
                    keep_fingerprints=keep_fingerprints)
    if root is not None:
        Path(root).mkdir(parents=True, exist_ok=True)


def capture_root() -> Path | None:
    return _CAPTURE["root"]


def build_fingerprint(build_dir: Path | str) -> str:
    """Stat identity of the served executable and DSOs: a rebuild in a reused lane
    directory changes it, so two candidates built in one directory never mix."""
    build = Path(build_dir)
    rows = []
    for directory in (build / "bin", build):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("lib*.so*")) + [directory / "llama-server",
                                                         directory / "llama-bench"]:
            try:
                real = path.resolve(strict=True)
                stat = real.stat()
            except OSError:
                continue
            rows.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ino}")
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:20]


def _shape_key(recipe: Any) -> str:
    name = str(getattr(recipe, "name", None) or "unnamed")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:120]


class LaunchSink:
    def __init__(self, raw: Path, sidecar: Path, meta: dict[str, Any]):
        self.raw, self.sidecar, self.meta = raw, sidecar, meta
        self.handle = raw.open("wb")


def open_launch_sink(recipe: Any, build_dir: Path | str) -> LaunchSink | None:
    """A file for one launch's stderr, or None (capture off, shape already sampled
    `per_shape` times for this build, or any error -- the caller uses /dev/null)."""
    root = _CAPTURE["root"]
    if root is None:
        return None
    try:
        fingerprint = build_fingerprint(build_dir)
        shape = _shape_key(recipe)
        directory = root / fingerprint / shape
        directory.mkdir(parents=True, exist_ok=True)
        if len(list(directory.glob("*.json"))) >= int(_CAPTURE["per_shape"]):
            return None
        stem = f"{time.time_ns()}-{os.getpid()}"
        return LaunchSink(directory / f"{stem}.stderr", directory / f"{stem}.json", {
            "schema": SIDECAR_SCHEMA, "fingerprint": fingerprint, "shape": shape,
            "build_dir": str(build_dir), "started_at": time.time()})
    except Exception:  # noqa: BLE001 -- auxiliary: never touches the measurement
        return None


def close_launch_sink(sink: LaunchSink | None) -> None:
    """Compact the raw stderr into its marker sidecar and delete the raw log."""
    if sink is None:
        return
    try:
        sink.handle.close()
        parsed = parse_markers(sink.raw.read_bytes().decode("utf-8", "replace"))
        sink.sidecar.write_text(json.dumps({**sink.meta, **parsed}, sort_keys=True) + "\n",
                                encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            sink.raw.unlink()
        except OSError:
            pass
        _prune()


def _prune() -> None:
    root = _CAPTURE["root"]
    if root is None:
        return
    try:
        dirs = sorted((p for p in root.iterdir() if p.is_dir()),
                      key=lambda p: p.stat().st_mtime)
        for stale in dirs[:max(0, len(dirs) - int(_CAPTURE["keep_fingerprints"]))]:
            for path in sorted(stale.rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
            stale.rmdir()
    except OSError:
        pass


def runtime_manifest(build_dir: Path | str, *, root: Path | str | None = None
                     ) -> dict[str, Any]:
    """Per serving shape: launches observed, markers in ALL / ANY of them, buffers."""
    root = Path(root) if root is not None else _CAPTURE["root"]
    fingerprint = build_fingerprint(build_dir)
    shapes: dict[str, Any] = {}
    directory = None if root is None else root / fingerprint
    if directory is not None and directory.is_dir():
        for shape_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
            rows = []
            for path in sorted(shape_dir.glob("*.json")):
                try:
                    rows.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    continue
            if not rows:
                continue
            sets = [set(row.get("markers") or ()) for row in rows]
            buffers: dict[str, list[float]] = {}
            for row in rows:
                for name, mib in (row.get("buffers_mib") or {}).items():
                    buffers.setdefault(name, []).append(float(mib))
            shapes[shape_dir.name] = {"launches": len(rows),
                                      "stable": sorted(set.intersection(*sets)),
                                      "any": sorted(set.union(*sets)),
                                      "buffers_mib": {k: min(v) for k, v in buffers.items()}}
    return {"schema": RUNTIME_SCHEMA, "build_dir": str(build_dir),
            "fingerprint": fingerprint, "shapes": shapes}


# ------------------------------------------------------------------ declarations

_DECLARE = re.compile(r"KERNEL-REPLACES:\s*(\S+)\s*(?:=>|->)\s*(\S+)")


def declarations(*texts: str | None) -> list[dict[str, str]]:
    found = []
    for text in texts:
        for match in _DECLARE.finditer(text or ""):
            row = {"old": match.group(1).strip("`'\""), "new": match.group(2).strip("`'\".,;")}
            if row not in found:
                found.append(row)
    return found


def added_lines(diff: str) -> str:
    return "\n".join(line[1:] for line in diff.splitlines()
                     if line.startswith("+") and not line.startswith("+++"))


# ------------------------------------------------------------------ diff + verdict

def _counter_losses(base: Mapping[str, int], cand: Mapping[str, int]) -> list[dict]:
    return [{"key": key, "base": int(count), "candidate": int(cand.get(key, 0))}
            for key, count in sorted(base.items()) if int(cand.get(key, 0)) < int(count)]


def _counter_gains(base: Mapping[str, int], cand: Mapping[str, int]) -> list[str]:
    return sorted(k for k, v in cand.items() if int(v) > int(base.get(k, 0)))


def diff_static(base: Mapping[str, Any], cand: Mapping[str, Any]) -> dict[str, Any]:
    losses, gains = [], []
    for library, counts in sorted(base["libraries"].items()):
        other = cand["libraries"].get(library)
        if other is None:
            losses.append({"layer": "static", "scope": library, "key": f"library:{library}",
                           "base": 1, "candidate": 0})
            continue
        losses += [{"layer": "static", "scope": library, **row}
                   for row in _counter_losses(counts, other)]
        gains += [f"{library}:{key}" for key in _counter_gains(counts, other)]
    return {"losses": losses, "gained": gains}


def diff_source(base: Mapping[str, Any], cand: Mapping[str, Any]) -> dict[str, Any]:
    return {"losses": [{"layer": "source", "scope": "tree", **row} for row in
                       _counter_losses(base["inventory"], cand["inventory"])],
            "gained": _counter_gains(base["inventory"], cand["inventory"])}


def diff_runtime(base: Mapping[str, Any], cand: Mapping[str, Any], *, target: str,
                 shape: str, identical: bool = False) -> dict[str, Any]:
    """One target shape. `identical` also lists every marker or buffer that changed."""
    b_any, c_any = set(base["any"]), set(cand["any"])
    stable = set(base["stable"])
    scope = f"{target}/{shape}"
    losses = [{"layer": "runtime", "scope": scope, "key": key, "base": 1, "candidate": 0}
              for key in sorted(stable - c_any)]
    unstable = sorted((b_any - stable) - c_any)
    for name, mib in sorted(base["buffers_mib"].items()):
        if not any(marker in name.upper() for marker in EXTRA_BUFFER_MARKERS):
            continue
        now = cand["buffers_mib"].get(name)
        if now is not None and now < mib * 0.995:
            losses.append({"layer": "runtime", "scope": scope, "key": f"buffer_mib:{name}",
                           "base": mib, "candidate": now})
    gained = sorted(c_any - b_any)
    changed = []
    if identical:
        changed = sorted(b_any ^ c_any) + sorted(
            name for name in set(base["buffers_mib"]) | set(cand["buffers_mib"])
            if base["buffers_mib"].get(name) != cand["buffers_mib"].get(name))
    return {"losses": losses, "unstable": unstable, "gained": gained, "changed": changed}


def _present_keys(static: Mapping | None, source: Mapping | None,
                  runtime: Mapping[str, Mapping] | None) -> set[str]:
    keys: set[str] = set()
    if static is not None:
        for counts in static["libraries"].values():
            keys.update(counts)
    if source is not None:
        keys.update(source["inventory"])
    for manifest in (runtime or {}).values():
        for shape in manifest["shapes"].values():
            keys.update(shape["any"])
    return keys


def verdict(*, static: tuple[Mapping, Mapping] | None = None,
            source: tuple[Mapping, Mapping] | None = None,
            runtime: Mapping[str, tuple[Mapping, Mapping]] | None = None,
            declared: Sequence[Mapping[str, str]] = (),
            measured_targets: Iterable[str] = (), all_targets: Iterable[str] = (),
            context: Mapping | None = None) -> dict[str, Any]:
    """The preservation verdict over whichever layers were collected.

    `runtime` maps target id -> (base manifest, candidate manifest), diffed on the
    shapes both builds were observed on; an unobserved target is recorded, not failed
    (a bench-measured keep launches no server)."""
    measured, targets = set(measured_targets), set(all_targets)
    losses: list[dict] = []
    failures: list[str] = []
    notes: list[str] = []
    gained: dict[str, list[str]] = {}
    unstable: dict[str, list[str]] = {}
    if static is not None:
        part = diff_static(*static)
        losses += part["losses"]
        gained["static"] = part["gained"]
    if source is not None:
        part = diff_source(*source)
        losses += part["losses"]
        gained["source"] = part["gained"]
    for target, (base, cand) in sorted((runtime or {}).items()):
        shared = sorted(set(base["shapes"]) & set(cand["shapes"]))
        if not shared:
            notes.append(f"{target}: runtime coverage not observed on both builds")
            continue
        for shape in shared:
            part = diff_runtime(base["shapes"][shape], cand["shapes"][shape], target=target,
                                shape=shape)
            losses += part["losses"]
            if part["gained"]:
                gained[f"runtime:{target}/{shape}"] = part["gained"]
            if part["unstable"]:
                unstable[f"{target}/{shape}"] = part["unstable"]
    present = _present_keys(static[1] if static else None, source[1] if source else None,
                            {t: pair[1] for t, pair in (runtime or {}).items()})
    unmeasured_static = sorted(targets - measured)
    for loss in losses:
        key = loss["key"]
        decl = next((d for d in declared if fnmatch.fnmatchcase(key, d["old"])), None)
        if decl is None:
            loss["status"] = "undeclared"
        elif not any(fnmatch.fnmatchcase(k, decl["new"]) for k in present):
            loss["status"] = f"declared replacement {decl['new']} absent from the candidate"
        elif loss["layer"] == "runtime" and loss["scope"].split("/", 1)[0] not in measured:
            loss["status"] = "declared but not measured on that model"
        elif loss["layer"] != "runtime" and unmeasured_static:
            loss["status"] = ("declared but not measured on " + ", ".join(unmeasured_static))
        else:
            loss["status"] = "declared_replacement"
            loss["declaration"] = dict(decl)
            continue
        failures.append(f"{loss['layer']} {loss['scope']}: {key} "
                        f"{loss['base']}->{loss['candidate']} ({loss['status']})")
    reason = ("; ".join(failures[:8]) + (f"; +{len(failures) - 8} more"
                                          if len(failures) > 8 else "")) if failures else \
        "no kernel path lost" + (" (declared replacements measured)"
                                 if any(l.get("status") == "declared_replacement"
                                        for l in losses) else "")
    return {"schema": VERDICT_SCHEMA, "passed": not failures, "reason": reason,
            "layers": {"static": static is not None, "source": source is not None,
                       "runtime": sorted((runtime or {}))},
            "losses": losses, "failures": failures, "gained": gained,
            "unstable": unstable, "notes": notes, "declarations": list(declared),
            "measured_targets": sorted(measured), "all_targets": sorted(targets),
            **({"context": dict(context)} if context else {})}


def peer_coverage(base: Mapping[str, Any], cand: Mapping[str, Any]) -> dict[str, Any]:
    """A peer target's executed-path change between two builds (its own shapes only).

    `unchanged` is the per-target-lineage trunk condition: observed on both builds, no
    stable marker lost, no extra-buffer shrink, and no marker or buffer changed at all."""
    shared = sorted(set(base["shapes"]) & set(cand["shapes"]))
    out: dict[str, Any] = {"observed": bool(shared), "shapes": shared, "losses": [],
                           "changed": [], "gained": []}
    for shape in shared:
        part = diff_runtime(base["shapes"][shape], cand["shapes"][shape], target="peer",
                            shape=shape, identical=True)
        out["losses"] += part["losses"]
        out["changed"] += [f"{shape}:{key}" for key in part["changed"]]
        out["gained"] += [f"{shape}:{key}" for key in part["gained"]]
    out["unchanged"] = bool(shared) and not out["losses"] and not out["changed"]
    return out


def record(store: Path | str, body: Mapping[str, Any], *, label: str) -> Path:
    directory = Path(store) / VERDICT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{time.time_ns()}-{re.sub(r'[^A-Za-z0-9_.-]+', '_', label)[:80]}.json"
    path.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------ keep gate

def keep_gate(*, store: Path | str, repo: Path | str, base_ref: str, candidate_ref: str,
              anchor_build: Path | str, candidate_build: Path | str, own_target: str,
              peer_shapes: Mapping[str, Sequence[str]] | None = None,
              measured_peers: Iterable[str] = (), declaration_texts: Sequence[str] = (),
              mechanism_id: str = "") -> dict[str, Any]:
    """Every keep: static + source + OWN-target runtime layers, anchor vs candidate.

    The verdict (`passed`) vetoes the keep: static and source losses are model-agnostic
    (they hit the keep's own lineage too), and own-target runtime losses are its own
    model's. `peer_shapes` maps each peer target id to the serving-shape names its A/B
    launched; their diffs land in `peers` (`peer_coverage`) and decide shared-trunk vs
    target-only (`cross_target.decide`), never the veto. A static/source layer that
    cannot be collected fails closed; an unobserved runtime layer is a note."""
    repo = Path(repo)
    diff = _git(repo, "diff", base_ref, candidate_ref, check=False)
    declared = declarations(*declaration_texts, added_lines(diff))
    peer_shapes = {target: list(shapes) for target, shapes in (peer_shapes or {}).items()}
    context = {"mechanism_id": mechanism_id, "base_ref": base_ref,
               "candidate_ref": candidate_ref, "anchor_build": str(anchor_build),
               "candidate_build": str(candidate_build)}
    errors = []
    static = source = None
    try:
        static = (static_manifest(anchor_build), static_manifest(candidate_build))
    except Exception as exc:  # noqa: BLE001 -- fail closed, recorded
        errors.append(f"static manifest unavailable: {type(exc).__name__}: {exc}")
    try:
        source = (source_inventory(repo, base_ref), source_inventory(repo, candidate_ref))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"source inventory unavailable: {type(exc).__name__}: {exc}")
    runtime, peers = None, {}
    if capture_root() is not None:
        base_rt, cand_rt = runtime_manifest(anchor_build), runtime_manifest(candidate_build)
        owner = {shape: target for target, shapes in peer_shapes.items() for shape in shapes}

        def only(manifest, target):
            return {**manifest, "shapes": {s: v for s, v in manifest["shapes"].items()
                                           if owner.get(s, own_target) == target}}
        runtime = {own_target: (only(base_rt, own_target), only(cand_rt, own_target))}
        peers = {target: peer_coverage(only(base_rt, target), only(cand_rt, target))
                 for target in peer_shapes}
    body = verdict(static=static, source=source, runtime=runtime, declared=declared,
                   measured_targets={own_target, *measured_peers},
                   all_targets={own_target, *peer_shapes}, context=context)
    body["peers"] = peers
    if errors:
        body["failures"] = errors + body["failures"]
        body["passed"] = False
        body["reason"] = "; ".join(body["failures"][:8])
    body["record"] = str(record(store, body, label=mechanism_id or "keep"))
    return body


def shape_names(*templates: Any) -> list[str]:
    return [_shape_key(t) for t in templates if t is not None]


# ------------------------------------------------------------------ fold check (CLI)

def fold_check(*, repo: Path | str, base: str, candidate: str,
               base_build: Path | str | None = None,
               candidate_build: Path | str | None = None,
               declaration_file: Path | str | None = None) -> dict[str, Any]:
    """A champion fold / forward-port / merge: did any feature disappear?

    Declarations come from the commit messages in base..candidate and an optional
    file. A fold carries no per-model A/B of its own here, so a declared replacement is
    accepted as declared and its measurement is the fold battery's (G5) to show."""
    repo = Path(repo)
    texts = [_git(repo, "log", "--format=%B", f"{base}..{candidate}", check=False)]
    if declaration_file is not None:
        texts.append(Path(declaration_file).read_text(encoding="utf-8"))
    static = None
    if base_build is not None and candidate_build is not None:
        static = (static_manifest(base_build), static_manifest(candidate_build))
    body = verdict(static=static, source=(source_inventory(repo, base),
                                          source_inventory(repo, candidate)),
                   declared=declarations(*texts), measured_targets={"fold"},
                   all_targets={"fold"},
                   context={"mode": "fold", "base": base, "candidate": candidate})
    return body


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m scripts.kernel_rnd.autokernel.loop.kernel_coverage",
        description="Kernel feature-preservation manifests and the fold-time check.")
    sub = parser.add_subparsers(dest="command", required=True)
    fold = sub.add_parser("fold-check", help="champion ref vs fold/forward-port candidate")
    fold.add_argument("--repo", type=Path, required=True)
    fold.add_argument("--base", required=True, help="champion / production ref")
    fold.add_argument("--candidate", required=True, help="fold or forward-port ref")
    fold.add_argument("--base-build", type=Path)
    fold.add_argument("--candidate-build", type=Path)
    fold.add_argument("--declarations", type=Path,
                      help="file with KERNEL-REPLACES: <old> => <new> lines")
    fold.add_argument("--json", type=Path, help="write the verdict here")
    manifest = sub.add_parser("manifest", help="print a build's static manifest")
    manifest.add_argument("build_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "manifest":
        print(json.dumps(static_manifest(args.build_dir), indent=1, sort_keys=True))
        return 0
    if (args.base_build is None) != (args.candidate_build is None):
        print("--base-build and --candidate-build go together", file=sys.stderr)
        return 2
    body = fold_check(repo=args.repo, base=args.base, candidate=args.candidate,
                      base_build=args.base_build, candidate_build=args.candidate_build,
                      declaration_file=args.declarations)
    if args.json is not None:
        args.json.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(("PASS " if body["passed"] else "FAIL ") + body["reason"])
    for failure in body["failures"][:40]:
        print(f"  - {failure}")
    return 0 if body["passed"] else 1


__all__ = ["added_lines", "build_fingerprint", "capture_root", "classify_symbol",
           "close_launch_sink", "declarations", "diff_runtime", "diff_source", "diff_static",
           "enable_capture", "fold_check", "keep_gate", "open_launch_sink", "parse_markers",
           "peer_coverage",
           "runtime_manifest", "shape_names", "source_inventory", "static_manifest",
           "verdict"]

if __name__ == "__main__":
    sys.exit(main())
