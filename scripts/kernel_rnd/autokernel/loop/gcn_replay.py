"""C5-ISA-2 GCN-Gym GPU half, host side: captured-launch -> hipModuleLaunchKernel replay plan, driver argv, output
comparison, identity gate and ABBA timing for an edited code object. No GPU here; every GPU step goes through an injected
``runner(argv, timeout_s) -> (rc, stdout, stderr)`` so the logic is testable with a fake.

Pieces (see gcn_capture.c and gcn_replay_driver.cpp beside this file):

* ``kernel_meta(notes_text)``      -- per kernel: kernarg segment size and explicit args (offset, size, value_kind) from
  ``llvm-readelf --notes`` of the code object (hidden_* args are runtime-filled and skipped).
* ``read_records(jsonl)``          -- the capture shim's records split into dumped launches and in-situ timings per key.
* ``build_plan(rec, meta, co, blob_dir)`` -- the driver plan; every non-null ``global_buffer`` arg must have resolved to a
  captured allocation, else ``ReplayRefused`` (an unresolved pointer would read the capturing process's address space).
* ``compare_outputs(ref, cand)``   -- byte-exact plus an f32 view (max abs diff, NaN-pattern agreement).
* ``identity_gate(...)``           -- the unedited replay must reproduce the in-situ output byte-exactly and its median
  per-launch time must be within +-10 % of the in-situ median.
* ``abba(...)``                    -- A=incumbent co, B=candidate co, A B B A per pair -> ``gcn_gym.RunResult``.
"""
from __future__ import annotations

import collections
import json
import math
import re
import statistics
import struct
from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml

HERE = Path(__file__).resolve().parent
DRIVER_SOURCE = HERE / "gcn_replay_driver.cpp"
CAPTURE_SOURCE = HERE / "gcn_capture.c"
HIPCC = "/opt/rocm/bin/hipcc"
CLANG = "/opt/rocm/llvm/bin/clang"
TIMING_TOLERANCE = 0.10

# Mangled-name EREs for the two C5-ISA-2 targets (gcn_gym.TARGETS). Enum template args mangle as L<type><value>E:
# mul_mat_q<GGML_TYPE_Q8_0 (=8), J=64, fallback> and flash_attn_ext_f16<DKQ in {192,256,320,512,576}, ...>.
TARGET_RE = {
    "MMQ Q8_0 J=64 prefill tile": r"mul_mat_qIL9ggml_type8ELi64E",
    "MMA-FA DKQ>=192": r"flash_attn_ext_f16ILi(192|256|320|512|576)E",
}
CAPTURE_MATCH = "|".join(f"({v})" for v in TARGET_RE.values())

Runner = Callable[[Sequence[str], float], tuple]


class ReplayRefused(ValueError):
    pass


def target_of(kernel: str) -> str | None:
    for t, rx in TARGET_RE.items():
        if re.search(rx, kernel):
            return t
    return None


# ------------------------------------------------------------------ metadata / capture
def kernel_meta(notes_text: str) -> dict[str, dict]:
    i = notes_text.find("amdhsa.kernels:")
    if i < 0:
        return {}
    j = notes_text.find("\namdhsa.", i + 1)
    doc = yaml.load(notes_text[i:j if j > 0 else None], Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)) or {}
    out = {}
    for k in doc.get("amdhsa.kernels") or []:
        args = [{"offset": int(a[".offset"]), "size": int(a[".size"]), "kind": str(a.get(".value_kind", ""))}
                for a in k.get(".args") or [] if not str(a.get(".value_kind", "")).startswith("hidden_")]
        out[k[".name"]] = {"kernarg_segment_size": int(k.get(".kernarg_segment_size", 0)), "args": args}
    return out


def _key(r: Mapping) -> tuple:
    return (r["k"], tuple(r["grid"]), tuple(r["block"]), int(r["shmem"]))


def read_records(path: str | Path) -> dict[tuple, dict]:
    """{key: {"dumped": [rec...], "ms": [float...]}}; torn lines and failed timings (ms < 0) are dropped."""
    out: dict[tuple, dict] = collections.defaultdict(lambda: {"dumped": [], "ms": []})
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("timed"):
            if r.get("ms", -1) >= 0 and r.get("launch_rc", 0) == 0:
                out[_key(r)]["ms"].append(float(r["ms"]))
        elif "dumped" in r:
            out[_key(r)]["dumped"].append(r)
    return dict(out)


def read_skipped(path: str | Path) -> dict[str, list[str]]:
    """Kernel names the shim saw but did not select: {"no_match": [...], "no_argtable": [...]} (sorted, unique)."""
    out: dict[str, set] = {"no_match": set(), "no_argtable": set()}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "skipped" in r:
            out.setdefault(r.get("why", "?"), set()).add(r["skipped"])
    return {k: sorted(v) for k, v in out.items()}


def build_plan(rec: Mapping, meta: Mapping, co: str | Path, blob_dir: str | Path | None, *, seed: int = 1) -> dict:
    """Driver plan for one dumped launch. blob_dir=None -> deterministic fill instead of the captured inputs."""
    m = meta.get(rec["k"])
    if m is None:
        raise ReplayRefused(f"{rec['k']}: not in the code object's metadata")
    ka = bytes.fromhex(rec["kernarg"])
    if len(ka) != m["kernarg_segment_size"]:
        raise ReplayRefused(f"{rec['k']}: kernarg {len(ka)} B != segment {m['kernarg_segment_size']} B")
    if blob_dir is not None and not rec.get("dumped"):
        raise ReplayRefused(f"{rec['k']} id {rec.get('id')}: launch not dumped (skipped={rec.get('dump_skipped')})")
    resolved = {int(o): (int(j), int(d)) for o, j, d in rec.get("ptrs", [])}
    ptrs = []
    for a in m["args"]:
        if a["kind"] != "global_buffer":
            continue
        if a["size"] != 8:
            raise ReplayRefused(f"{rec['k']}: global_buffer arg of size {a['size']}")
        if a["offset"] in resolved:
            j, d = resolved[a["offset"]]
            ptrs.append((a["offset"], j, d))
        elif int.from_bytes(ka[a["offset"]:a["offset"] + 8], "little") != 0:
            raise ReplayRefused(f"{rec['k']}: pointer arg at kernarg+{a['offset']} did not resolve to an allocation")
    allocs = []
    for j, al in enumerate(rec.get("allocs", [])):
        pre = str(Path(blob_dir) / al["pre"]) if blob_dir is not None else "-"
        post = str(Path(blob_dir) / al["post"]) if blob_dir is not None else None
        allocs.append({"j": j, "bytes": int(al["size"]), "pre": pre, "post": post, "seed": seed + j})
    used = {j for _o, j, _d in ptrs}
    if any(j >= len(allocs) for j in used):
        raise ReplayRefused(f"{rec['k']}: pointer to an unrecorded allocation")
    for o, j, d in ptrs:
        if d >= allocs[j]["bytes"]:
            raise ReplayRefused(f"{rec['k']}: pointer delta beyond its allocation")
    return {"co": str(co), "kernel": rec["k"], "grid": list(rec["grid"]), "block": list(rec["block"]),
            "lds": int(rec["shmem"]), "kernarg": ka.hex(), "args": [(a["offset"], a["size"]) for a in m["args"]],
            "allocs": allocs, "ptrs": ptrs, "capture_id": rec.get("id")}


def render_plan(plan: Mapping, co: str | Path | None = None) -> str:
    for s in (str(co or plan["co"]), plan["kernel"]) + tuple(a["pre"] for a in plan["allocs"]):
        if not s or any(c.isspace() for c in s):
            raise ReplayRefused(f"whitespace or empty token in plan: {s!r}")
    L = [f"# capture id {plan.get('capture_id')}", f"co {co or plan['co']}", f"kernel {plan['kernel']}",
         "grid %d %d %d" % tuple(plan["grid"]), "block %d %d %d" % tuple(plan["block"]), f"lds {plan['lds']}",
         f"kernarg {plan['kernarg']}"]
    L += [f"arg {o} {s}" for o, s in plan["args"]]
    L += [f"alloc {a['j']} {a['bytes']} {a['pre']} {a['seed']}" for a in plan["allocs"]]
    L += [f"ptr {o} {j} {d}" for o, j, d in plan["ptrs"]]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ argv
def driver_build_argv(out: str | Path, src: str | Path = DRIVER_SOURCE) -> list[str]:
    return [HIPCC, "-O2", "-o", str(out), str(src)]


def capture_build_argv(out: str | Path, src: str | Path = CAPTURE_SOURCE) -> list[str]:
    return [CLANG, "-O2", "-shared", "-fPIC", "-D__HIP_PLATFORM_AMD__", "-I/opt/rocm/include", "-o", str(out), str(src),
            "-L/opt/rocm/lib", "-lamdhip64", "-ldl", "-lpthread"]


def driver_argv(driver: str | Path, plan_path: str | Path, outdir: str | Path, *, warmup: int = 5,
                iters: int = 50) -> list[str]:
    return [str(driver), str(plan_path), str(outdir), str(warmup), str(iters)]


def run_driver(runner: Runner, driver, plan: Mapping, workdir: str | Path, *, co=None, tag: str = "replay",
               warmup: int = 5, iters: int = 50, timeout_s: float = 300) -> dict:
    """Write the plan, run the driver once, return {ok, rc, times_ms, outdir, outs, err}."""
    wd = Path(workdir) / tag
    wd.mkdir(parents=True, exist_ok=True)
    pp = wd / "plan.txt"
    pp.write_text(render_plan(plan, co))
    rc, so, se = runner(driver_argv(driver, pp, wd, warmup=warmup, iters=iters), timeout_s)
    last = next((l for l in reversed((so or "").splitlines()) if l.startswith("{")), None)
    try:
        js = json.loads(last) if last else {}
    except ValueError:
        js = {}
    ok = rc == 0 and js.get("ok") is True and len(js.get("times_ms", [])) == iters
    return {"ok": ok, "rc": rc, "times_ms": js.get("times_ms", []), "outdir": str(wd),
            "outs": [str(wd / f"a{a['j']}.out") for a in plan["allocs"]],
            "err": None if ok else (js.get("err") or (se or "")[-400:] or f"rc {rc}")}


# ------------------------------------------------------------------ comparison / gates
def compare_bytes(a: bytes, b: bytes) -> dict:
    if len(a) != len(b):
        return {"equal": False, "size_mismatch": [len(a), len(b)]}
    if a == b:
        return {"equal": True, "n_diff_bytes": 0, "max_abs_f32": 0.0, "nan_mismatch": 0}
    nd = sum(1 for x, y in zip(a, b) if x != y)
    n = len(a) // 4
    fa, fb = struct.unpack(f"<{n}f", a[:4 * n]), struct.unpack(f"<{n}f", b[:4 * n])
    mx, nanm = 0.0, 0
    for x, y in zip(fa, fb):
        if math.isnan(x) or math.isnan(y):
            nanm += math.isnan(x) != math.isnan(y)
        elif math.isinf(x) or math.isinf(y):
            nanm += x != y
        else:
            mx = max(mx, abs(x - y))
    return {"equal": False, "n_diff_bytes": nd, "max_abs_f32": mx, "nan_mismatch": nanm}


def compare_outputs(ref_paths: Sequence[str | Path], cand_paths: Sequence[str | Path]) -> dict:
    if len(ref_paths) != len(cand_paths):
        return {"equal": False, "per_alloc": [], "reason": "allocation count differs"}
    per = []
    for r, c in zip(ref_paths, cand_paths):
        if not Path(r).exists() or not Path(c).exists():
            per.append({"equal": False, "missing": [str(p) for p in (r, c) if not Path(p).exists()]})
        else:
            per.append(compare_bytes(Path(r).read_bytes(), Path(c).read_bytes()))
    return {"equal": all(p["equal"] for p in per), "per_alloc": per}


def identity_gate(insitu_ms: Sequence[float], replay_ms: Sequence[float], cmp: Mapping, *,
                  tol: float = TIMING_TOLERANCE, min_samples: int = 5) -> dict:
    """PASS iff the unedited replay is byte-identical to the in-situ output and its median per-launch time is within
    +-tol of the in-situ median. Too few in-situ or replay samples is INSUFFICIENT, never a pass."""
    if len(insitu_ms) < min_samples or len(replay_ms) < min_samples:
        return {"verdict": "INSUFFICIENT", "n_insitu": len(insitu_ms), "n_replay": len(replay_ms),
                "output_equal": bool(cmp.get("equal"))}
    mi, mr = statistics.median(insitu_ms), statistics.median(replay_ms)
    ratio = mr / mi if mi > 0 else float("inf")
    reasons = []
    if not cmp.get("equal"):
        reasons.append("output_differs")
    if abs(ratio - 1.0) > tol:
        reasons.append("timing_outside_tolerance")
    return {"verdict": "PASS" if not reasons else "FAIL", "reasons": reasons, "insitu_median_ms": mi,
            "replay_median_ms": mr, "ratio": ratio, "tol": tol, "n_insitu": len(insitu_ms), "n_replay": len(replay_ms),
            "output_equal": bool(cmp.get("equal"))}


def abba(runner: Runner, driver, plan: Mapping, workdir, *, incumbent_co, candidate_co, pairs: int = 2,
         iters: int = 50, oracle: Callable[[Mapping], bool] | None = None, timeout_s: float = 300):
    """A B B A per pair on one replay plan -> gcn_gym.RunResult (+ raw legs). The oracle defaults to byte-exact
    agreement of every allocation between the candidate's and the incumbent's first legs; pass ``oracle`` to use a
    tolerance instead (it receives the compare_outputs result)."""
    from gcn_gym import RunResult  # same directory
    legs = {"A": [], "B": []}
    first = {}
    for p in range(pairs):
        for i, side in enumerate("ABBA"):
            co = incumbent_co if side == "A" else candidate_co
            r = run_driver(runner, driver, plan, workdir, co=co, tag=f"abba{p}{i}{side}", iters=iters, timeout_s=timeout_s)
            if not r["ok"]:
                return RunResult(oracle_pass=False, candidate_ms=(), incumbent_ms=()), {"error": r, "leg": f"{p}{i}{side}"}
            first.setdefault(side, r)
            legs[side].append(statistics.median(r["times_ms"]))
    cmp = compare_outputs(first["A"]["outs"], first["B"]["outs"])
    ok = oracle(cmp) if oracle else cmp["equal"]
    cand = [(legs["B"][2 * p] + legs["B"][2 * p + 1]) / 2 for p in range(pairs)]
    inc = [(legs["A"][2 * p] + legs["A"][2 * p + 1]) / 2 for p in range(pairs)]
    return RunResult(oracle_pass=bool(ok), candidate_ms=tuple(cand), incumbent_ms=tuple(inc)), {"compare": cmp, "legs": legs}


def identity_replay(runner: Runner, driver, rec: Mapping, meta: Mapping, co, blob_dir, insitu_ms, workdir, *,
                    iters: int = 50, timeout_s: float = 300) -> dict:
    """The identity gate for one dumped launch: replay the ORIGINAL code object on the captured inputs and compare
    with the captured in-situ output (.post blobs) and in-situ timings."""
    try:
        plan = build_plan(rec, meta, co, blob_dir)
    except ReplayRefused as e:
        return {"verdict": "REFUSED", "reason": str(e), "kernel": rec.get("k"), "capture_id": rec.get("id")}
    r = run_driver(runner, driver, plan, workdir, tag=f"id{rec.get('id')}", iters=iters, timeout_s=timeout_s)
    if not r["ok"]:
        return {"verdict": "DRIVER_ERROR", "err": r["err"], "rc": r["rc"], "kernel": rec["k"], "capture_id": rec.get("id")}
    cmp = compare_outputs([a["post"] for a in plan["allocs"]], r["outs"])
    g = identity_gate(insitu_ms, r["times_ms"], cmp)
    return dict(g, kernel=rec["k"], target=target_of(rec["k"]), capture_id=rec.get("id"), compare=cmp,
                grid=plan["grid"], block=plan["block"], lds=plan["lds"])
