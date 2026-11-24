#!/usr/bin/env python3
"""AK parallel loop P3: the per-stage job bodies of an akloop chain row (build -> oracle-kld -> ab-timing -> fold-check).

Design: epyc-root docs/design/autokernel-parallel-loop.md §2 ("Parallel loop") and §4 P3. The chain scaffolder
(epyc-root scripts/coordination/akloop/chain.py) writes one row per stage; each row's job.py calls

    python3 -m autokernel.loop.stage_bodies --stage S --candidate cand.json --out out/RESULT.json \
            --lane cpu|gpu --rowdir DIR [--test-mode]

and this module drives the EXISTING loop code paths in-process (gates.compiles, gates.op_correctness,
gates.graph_kld_run + the graph_kld_policy.json gate, bench.compare with n >= bench.MIN_PAIRS, the advance_champion
compare-and-swap, the P4 lineage check). It writes ONE RESULT.json with a polar token (gate_eval L2/`declared.polar`),
so gate_eval files every stage with zero LLM turns (verdicts start PASS/KEEP/FAIL/FUTILITY, the row template's
check_complete.py contract):

    build       verdict PASS_BUILT / FUTILITY_CANDIDATE_COMPILE / FUTILITY_HARNESS_*      polar PASS|FAIL
    oracle-kld  verdict PASS_ORACLE / FAIL_CORRECTNESS + graph_kld model blocks       polar PASS|FAIL (L2 decides)
    op-perf     SCREEN ONLY (RSL-R3): FAIL_OP_PERF_* polar FAIL | FUTILITY_SCREEN_ONLY_NO_KEEP, never a polar PASS
    ab-timing   verdict PASS_FASTER / FAIL_NO_GAIN + speed{surface,effect_pct,pairs,model}     polar PASS|FAIL (L2 decides)
    fold-check  verdict KEEP_LANDED / FUTILITY_HARNESS_STALE_BASE / FAIL_*            polar PASS|FAIL

HARD RULES enforced here, not only in briefs (a violation is a FUTILITY_HARNESS_* RESULT, never an action):
  * production FROZEN: never build in, commit to, or move a ref of production-consolidated-* or the canonical tree
    /mnt/raid0/llm/llama.cpp; the CAS only moves `ak/champion/*` or `llama.cpp-experimental*` refs;
  * HIP_VISIBLE_DEVICES=-1 for every stage that is not on the gpu lane (the keeper grants the GPU only to gpu rows);
  * a build compiles only on the cpus this row was granted (sched affinity, never q3 = 72-95/168-191) — another row's
    quarter is never touched because the affinity IS the grant;
  * no keep lands on a base it was not built from: fold-check refuses unless patch^ == base == champion head and the
    build stage's RESULT names the same patch sha.

--test-mode (or AK_STAGE_TEST_MODE=1): no build, no GPU, no git ref moves outside the given fixture repo. Outcomes come
from the candidate's `test_outcomes` map (stage -> "pass"|"fail"|...), so the offline end-to-end simulation exercises
every path that decides what lands.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from . import census_coverage as cc
from . import op_perf_screen as ops
from . import sampling_axis
from . import served_shape_cases as ssc

SCHEMA = "epyc.autokernel.stage_result.v1"
STAGES = ("build", "oracle-kld", "op-perf", "ab-timing", "fold-check")
CANONICAL_TREE = Path("/mnt/raid0/llm/llama.cpp")
PRODUCTION_BRANCH_RX = re.compile(r"(^|/)production-consolidated-v\d+")
ALLOWED_REF_RX = re.compile(r"^(refs/heads/)?(ak/champion/|llama\.cpp-experimental)")
GPU_HOST_CPUS = frozenset(range(72, 96)) | frozenset(range(168, 192))  # q3 cores + siblings (MI210 host threads)
GPU_STAGES = frozenset({"oracle-kld", "op-perf", "ab-timing", "fold-check"})
#: Stages whose device is the candidate PATH's (RSL-R6): a cpu path runs them on the cpu lane (the served host's own
#: topology), never in a GPU grant; the legacy default path is the 27B on the MI210.
PATH_STAGES = frozenset({"op-perf", "ab-timing"})
DEFAULT_PATH = "27b-q8_0|mi210"
LOOP_STORE = Path("/mnt/raid0/llm/autokernel/loop-memory")
QUALITY_UNGATED = "quality-ungated"
_SERVING_KEYS = ("port", "slots", "ngl", "no_mmap", "cache_ram", "numa_instances", "numa_ports", "serving_shape", "recipe")
#: Default work area for per-candidate source trees and builds (never the canonical tree).
WORK_ROOT = Path("/mnt/raid0/llm/tmp/akchain-work")


class StageRefused(RuntimeError):
    """A hard rule refused the stage before it acted. `token` is the FUTILITY_HARNESS_* verdict to write."""

    def __init__(self, token: str, why: str):
        super().__init__(why)
        self.token = token


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git(repo: Path | str, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


# ---- hard-rule guards -----------------------------------------------------------------------------------------

def guard_device(stage: str, lane: str, env: dict, need: str | None = None) -> None:
    """HIP_VISIBLE_DEVICES=-1 outside a GPU grant. Mutates `env` (normally os.environ). `need` is the path's device."""
    if lane != "gpu" or stage not in GPU_STAGES or need == "cpu":
        env["HIP_VISIBLE_DEVICES"] = "-1"
        env["ROCR_VISIBLE_DEVICES"] = "-1"


def guard_tree(tree: Path, git: Callable[..., str] | None = None) -> None:
    """Refuse the canonical production tree and anything checked out on a production branch."""
    t = Path(tree).resolve()
    if t == CANONICAL_TREE.resolve():
        raise StageRefused("FUTILITY_HARNESS_PRODUCTION_TREE", f"{t} is the frozen production tree")
    run = git or (lambda *a: _git(t, *a))
    try:
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
    except (subprocess.CalledProcessError, OSError):
        branch = ""
    if PRODUCTION_BRANCH_RX.search(branch or ""):
        raise StageRefused("FUTILITY_HARNESS_PRODUCTION_TREE", f"{t} is on production branch {branch}")


def guard_ref(ref: str) -> None:
    if PRODUCTION_BRANCH_RX.search(ref) or not ALLOWED_REF_RX.match(ref):
        raise StageRefused("FUTILITY_HARNESS_PRODUCTION_REF",
                           f"refusing to move {ref}: only ak/champion/* and llama.cpp-experimental* refs move")


def guard_build_cpus(cpus: set[int], *, test_mode: bool) -> str:
    """The cpu list a build may use: exactly this row's granted affinity, never q3, never the whole host."""
    if test_mode:
        return ",".join(str(c) for c in sorted(cpus)) or "0"
    if not cpus:
        raise StageRefused("FUTILITY_HARNESS_WRONG_GRANT", "empty cpu affinity")
    if cpus & GPU_HOST_CPUS:
        raise StageRefused("FUTILITY_HARNESS_WRONG_GRANT",
                           f"build affinity holds q3 / MI210 host cpus {sorted(cpus & GPU_HOST_CPUS)[:6]}...")
    if len(cpus) > 48:   # one quarter is 24 cores + 24 siblings: more means no grant narrowed us
        raise StageRefused("FUTILITY_HARNESS_WRONG_GRANT", f"affinity {len(cpus)} cpus: not running under a quarter grant")
    return _cpu_list(cpus)


def _cpu_list(cpus: set[int]) -> str:
    out, run = [], []
    for c in sorted(cpus):
        if run and c == run[-1] + 1:
            run.append(c)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [c]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


# ---- primitives (live = existing loop code paths; test = fixture outcomes) --------------------------------------

@dataclass
class Primitives:
    """The four side effects a stage may have. Live ones call the loop's own code; tests inject stubs."""
    source_tree: Callable[[Mapping[str, Any]], Path]
    compile: Callable[[Path, Path, str], dict]
    oracle: Callable[[Mapping[str, Any], Path, Path], dict]
    timing: Callable[[Mapping[str, Any], Path, Path], dict]
    champion_head: Callable[[str], str]
    cas: Callable[[str, str, str], None]
    parent_of: Callable[[str], str]
    lineage: Callable[[Mapping[str, Any]], dict]
    paths: dict = field(default_factory=dict)
    op_perf: Callable[..., dict] | None = None


def _live_source_tree(cand: Mapping[str, Any]) -> Path:
    """A detached worktree of the champion clone at the candidate's patch sha (content proof: HEAD == patch sha)."""
    patch = str(cand["patch_sha"])
    d = WORK_ROOT / str(cand["id"]) / "src"
    clone = Path(cand.get("llama_repo") or "/mnt/raid0/llm/llama.cpp")
    if not d.exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        _git(clone, "worktree", "add", "--detach", str(d), patch)
    guard_tree(d)
    head = _git(d, "rev-parse", "HEAD")
    if not head.startswith(patch) and not patch.startswith(head):
        raise StageRefused("FUTILITY_HARNESS_SOURCE_MISMATCH", f"{d} HEAD {head[:12]} != patch {patch[:12]}")
    return d


def _live_compile(src: Path, build_dir: Path, cpu_list: str) -> dict:
    """The champion build path run.py uses (GFX90A_ROCM62_V10_RECIPE, the direct-launch serving recipe; graph KLD
    needs llama-perplexity), confined to this row's granted cpus."""
    from . import gates
    from ..controller import build_recipe
    recipe = build_recipe.GFX90A_ROCM62_V10_RECIPE
    v = gates.compiles(src, build_dir, cmake_defines=recipe.cmake_defines(),
                       jobs=max(1, len(_expand(cpu_list))), cpu_list=cpu_list,
                       targets=tuple(dict.fromkeys(gates.DEFAULT_TARGETS + ("llama-perplexity",))))
    return {"passed": bool(v.passed), "gate": v.gate, "reason": v.reason, "verdicts": [v.to_dict()]}


def _expand(cpu_list: str) -> set[int]:
    out: set[int] = set()
    for p in str(cpu_list).split(","):
        p = p.strip()
        if "-" in p:
            a, b = p.split("-", 1)
            out |= set(range(int(a), int(b) + 1))
        elif p.isdigit():
            out.add(int(p))
    return out


def _live_oracle(cand: Mapping[str, Any], anchor_build: Path, cand_build: Path) -> dict:
    """Op oracle (test-backend-ops on the touched op) + the 27B-anchored graph KLD (graph_kld_policy.json)."""
    from . import gates
    op = str(cand.get("op") or "MUL_MAT")
    v = gates.op_correctness(cand_build, op=op, backend="ROCm0")
    out: dict[str, Any] = {"op_oracle": v.to_dict(), "op_passed": bool(v.passed), "graph_kld": {}}
    if not v.passed:
        return out
    policy = gates.load_graph_kld_policy("gpu")
    if policy is None:
        out["kld_unavailable"] = "graph_kld_policy.json unreadable (threshold_unset)"
        return out
    ncm = {c.model: c.n_cpu_moe for c in policy.clauses if c.n_cpu_moe is not None}
    base_dir = WORK_ROOT / "kld-base" / str(cand["base_sha"])
    base_dir.mkdir(parents=True, exist_ok=True)
    for key in dict.fromkeys(c.model for c in policy.clauses):
        extra = ("-ngl", "999") + (("--n-cpu-moe", str(ncm[key])) if key in ncm else ())
        text = gates.graph_kld_run(anchor_build, cand_build, model=policy.models[key], corpus=policy.corpus,
                                   base_logits=str(base_dir / f"{key}.kld-base"), chunks=policy.chunks, extra=extra)
        out["graph_kld"][key] = gates.parse_graph_kld(text)
    return out


def _runner(fn: Callable[..., dict], *a: Any, **k: Any) -> dict:
    """A live-runner refusal (op_perf_runner.RunnerRefused) is a stage refusal carrying its FUTILITY_HARNESS_* token."""
    from .op_perf_runner import RunnerRefused
    try:
        return fn(*a, **k)
    except RunnerRefused as e:
        raise StageRefused(e.token, str(e))


def _live_timing(cand: Mapping[str, Any], anchor_build: Path, cand_build: Path) -> dict:
    """Alternating paired A/B, n >= bench.MIN_PAIRS, on the candidate's served path. A CPU path runs the served host's
    topology (cpu_timing_runner); a GPU path runs llama-bench under the path's OWN A/A floor (effect in PERCENT)."""
    from . import bench, cpu_timing_runner as ctr
    pairs = max(int(cand.get("pairs") or bench.MIN_PAIRS), bench.MIN_PAIRS)
    spec = cand["path_spec"]
    if spec["lane"] != "gpu":
        return _runner(ctr.time_path, cand, anchor_build, cand_build, spec, store=Path(cand.get("store") or LOOP_STORE))
    model = Path(spec["gguf"])
    floor = (bench.floor_rows(spec["surface"], model, Path(cand.get("store") or LOOP_STORE)) or {}).get(pairs)
    c = bench.compare(bench.Arm("anchor", anchor_build / "bin" / "llama-bench"),
                      bench.Arm("candidate", cand_build / "bin" / "llama-bench"), model,
                      pp=0, tg=128, pairs=pairs, surface=spec["surface"], noise_floor_pct=floor)
    return {"effect_pct": float(c.effect) * 100.0, "pairs": int(c.pairs), "decisive": c.decisive,
            "noise_floor_pct": c.noise_floor_pct, "device_seconds": c.device_seconds,
            "anchor_samples": list(c.anchor_samples), "candidate_samples": list(c.candidate_samples),
            "model": str(model)}


def _live_op_perf(cand: Mapping[str, Any], anchor_build: Path, cand_build: Path, cells: list) -> dict:
    """Paired `test-backend-ops perf` per cell (op_perf_runner) under per-cell A/A floors; census-weighted."""
    from . import op_perf_runner as opr
    return _runner(opr.run, anchor_build, cand_build, cells, resolve_path(cand), pairs=cand.get("pairs"),
                   store=Path(cand.get("store") or LOOP_STORE))


def _live_cas(ref: str, new: str, old: str) -> None:
    from . import pool
    guard_ref(ref)
    full = ref if ref.startswith("refs/") else f"refs/heads/{ref}"
    _git(pool.CHAMPION_TREE, "update-ref", full, new, old)


def _live_lineage(event: Mapping[str, Any]) -> dict:
    """P4: regenerate the keep ledger / fold list after a KEEP (or a composite event)."""
    return trigger_lineage(event)


def live_primitives() -> Primitives:
    from . import pool
    champ = pool.CHAMPION_TREE
    return Primitives(source_tree=_live_source_tree, compile=_live_compile, oracle=_live_oracle, timing=_live_timing,
                      champion_head=lambda ref: _git(champ, "rev-parse", ref if ref.startswith("refs/") else
                                                     f"refs/heads/{ref}"),
                      cas=_live_cas, parent_of=lambda sha: _git(champ, "rev-parse", f"{sha}^"),
                      lineage=_live_lineage, op_perf=_live_op_perf)


def trigger_lineage(event: Mapping[str, Any], *, out_root: Path | None = None,
                    run: Callable[[list[str]], int] | None = None) -> dict:
    """Run `lineage generate` for a KEEP or composite event into <out_root>/<stamp>/. Never publishes anything."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = Path(out_root or Path("/mnt/raid0/llm/tmp/akchain-lineage")) / f"{stamp}-{event.get('kind', 'keep')}"
    out.mkdir(parents=True, exist_ok=True)
    argv = ["generate", "--out", str(out)]
    if event.get("llama_repo"):
        argv += ["--repo", str(event["llama_repo"])]
    if event.get("epyc_root"):
        argv += ["--root", str(event["epyc_root"])]
    if run is None:
        from ..lineage import __main__ as lineage_cli
        run = lineage_cli.main
    try:
        rc = run(argv)
    except SystemExit as e:  # argparse / CLI exits
        rc = int(e.code or 0)
    (out / "event.json").write_text(json.dumps(dict(event), indent=1, sort_keys=True) + "\n")
    return {"lineage_out": str(out), "lineage_rc": rc, "refused": rc == 3}


# ---- served paths (RSL-R6 / R8): model GGUF, device and serving contract come from the registry, not constants -------

def path_specs(registry: Mapping[str, Any]) -> dict[str, dict]:
    """{"model|device": spec} for every path `census_coverage.served_set` names (live and planned; DS41 has none).
    `gguf`/`serving` are read off the registry's non-alias host of the path's model_role + device. A PLANNED path
    has no served host yet: it borrows the role's GGUF and carries serving=None (no argv invented for it)."""
    hosts: dict = {}
    for name, e in sorted((registry.get("server_mode") or {}).items()):
        if isinstance(e, dict) and e.get("model_role") and not e.get("alias_of"):
            hosts.setdefault((e["model_role"], cc._device(e)), (name, e))
    out = {}
    for (model, dev), s in cc.served_set(registry).items():
        name, e = hosts.get((s["model_role"], dev)) or next(
            (h for (r, _), h in hosts.items() if r == s["model_role"]), (None, {}))
        acc = e.get("acceleration") if isinstance(e.get("acceleration"), dict) else {}
        serving = None
        if s["status"] == "live":
            serving = {k: e[k] for k in _SERVING_KEYS if e.get(k) is not None}
            serving.update({k: acc[k] for k in ("spec_type", "draft_max", "draft_model") if k in acc})
        gguf = next((g for g in (e.get("model_path"), e.get("model")) if isinstance(g, str) and os.path.isabs(g)), None)
        out[f"{model}|{dev}"] = {"key": f"{model}|{dev}", "model": model, "device": dev, "status": s["status"],
                                 "model_role": s["model_role"], "host": name, "gguf": gguf, "serving": serving,
                                 "n": sorted(s["n"]), "lane": "gpu" if dev == "mi210" else "cpu",
                                 "surface": "tg128"}  # greedy decode cell: the surface family bench.floor_rows keys
    return out


def resolve_path(cand: Mapping[str, Any]) -> dict:
    """The candidate's served path spec (`serving_path`, default the 27B on the MI210). Refuses, never guesses."""
    reg = cand.get("test_registry") or cc.load_registry()
    if not reg:
        raise StageRefused("FUTILITY_HARNESS_MISSING_INPUT", "model registry unreadable: no served set")
    specs, key = path_specs(reg), str(cand.get("serving_path") or DEFAULT_PATH)
    if key not in specs or not specs[key]["gguf"]:
        raise StageRefused("FUTILITY_HARNESS_MISSING_INPUT", f"{key} is not a served path with a GGUF; served: {sorted(specs)}")
    return specs[key]


def timing_floor(spec: Mapping[str, Any], cand: Mapping[str, Any]) -> bool:
    """True iff this path has its OWN A/A floor (bench.floor_rows keys floors by surface AND model, never borrowed)."""
    if cand.get("test_registry"):   # fixture mode: the candidate names the paths that have a floor cell
        return spec["key"] in (cand.get("test_floor_paths") or [])
    from . import bench, cpu_timing_runner as ctr
    if spec["lane"] == "cpu":   # keyed per model AND topology (cpu_aa_calibration), never the GPU floor file
        return ctr.floor_pct(spec, Path(cand.get("store") or LOOP_STORE), max(int(cand.get("pairs") or bench.MIN_PAIRS), bench.MIN_PAIRS)) is not None
    return bench.floor_rows(spec["surface"], spec["gguf"], Path(cand.get("store") or LOOP_STORE)) is not None


def quality_status(key: str, policy: Mapping[str, Any]) -> dict:
    """graph_kld_policy `paths[key]` -> gated iff a decisive clause for its (backend, model) exists; else QUALITY_UNGATED
    (no entry, or an entry with floor null / needs-floor-row). Never invents a floor."""
    e = (policy.get("paths") or {}).get(key)
    if not e:
        return {"path": key, "status": QUALITY_UNGATED, "reason": "no-policy-entry"}
    ok = any(c.get("model") == e.get("model") and c.get("decisive")
             for c in (policy.get("backends") or {}).get(e.get("backend"), []))
    return {"path": key, "status": "gated" if ok else QUALITY_UNGATED, "model": e.get("model"),
            "backend": e.get("backend"), "reason": None if ok else e.get("status") or "needs-floor-row"}


def coverage_report(registry: Mapping[str, Any], policy: Mapping[str, Any], cand: Mapping[str, Any] | None = None) -> dict:
    """Per served path: timing cell / `ungated`, quality `gated` / `quality-ungated`, op-perf grid / `ungated` (for /loop)."""
    cand = cand or {}
    return {k: {"status": s["status"], "device": s["device"],
                "timing": "cell" if timing_floor(s, cand) else "ungated",
                "quality": quality_status(k, policy)["status"],
                "op_perf": "grid" if ssc.lane_for_model(s["gguf"]) or ssc.dense_lane_for_model(s["gguf"]) else "ungated"}
            for k, s in path_specs(registry).items()}


# ---- stage bodies -----------------------------------------------------------------------------------------------

def _base(stage: str, cand: Mapping[str, Any]) -> dict:
    return {"schema": SCHEMA, "stage": stage, "row": cand.get("row"), "candidate": cand.get("id"),
            "base_sha": cand.get("base_sha"), "patch_sha": cand.get("patch_sha"),
            "hypothesis_id": cand.get("hypothesis_id"), "at": _now(), "complete": True,
            "sampling": sampling_axis.normalize(cand.get("sampling"))}


def _build_dir(cand: Mapping[str, Any], which: str = "candidate") -> Path:
    return Path(cand.get("build_root") or WORK_ROOT) / str(cand["id"]) / f"build-{which}"


def stage_build(cand, prim: Primitives, *, cpus: set[int], test_mode: bool) -> dict:
    r = _base("build", cand)
    cpu_list = guard_build_cpus(cpus, test_mode=test_mode)
    src = prim.source_tree(cand)
    if not test_mode:
        guard_tree(src)
    bd = _build_dir(cand)
    c = prim.compile(src, bd, cpu_list)
    r.update({"build_dir": str(bd), "source_tree": str(src), "cpu_list": cpu_list, "built_sha": cand["patch_sha"],
              "verdicts": c.get("verdicts", [])})
    if c.get("harness"):
        r.update(verdict=f"FUTILITY_HARNESS_BUILD_ENV ({c.get('reason')})")
    elif c.get("passed"):
        r.update(verdict="PASS_BUILT", polar="PASS")
    else:
        r.update(verdict=f"FUTILITY_CANDIDATE_COMPILE ({str(c.get('reason'))[:160]})", polar="FAIL")
    return r


def _prior_result(cand: Mapping[str, Any], stage: str) -> dict | None:
    p = (cand.get("stage_results") or {}).get(stage)
    if not p:
        return None
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _require_built(cand) -> dict:
    b = _prior_result(cand, "build")
    if not b or b.get("polar") != "PASS":
        raise StageRefused("FUTILITY_HARNESS_MISSING_INPUT", "build stage RESULT missing or not PASS")
    if str(b.get("built_sha")) != str(cand["patch_sha"]):
        raise StageRefused("FUTILITY_HARNESS_SOURCE_MISMATCH",
                           f"build RESULT built {b.get('built_sha')} but candidate patch is {cand['patch_sha']}")
    return b


def stage_oracle(cand, prim: Primitives, *, test_mode: bool) -> dict:
    """The 27B-anchored gate decides a FAIL as before; a PASS on a path with no policy floor is non-deciding (RSL-R8)."""
    r = _stage_oracle(cand, prim, test_mode=test_mode)
    from . import gate_eval
    q = quality_status(resolve_path(cand)["key"], gate_eval.load_policy())
    r["quality"] = q
    if q["status"] == QUALITY_UNGATED and r.get("polar") == "PASS":   # another model's clauses never keep this path
        r.pop("polar")
        r["verdict"] = f"FUTILITY_QUALITY_UNGATED ({q['path']} {q['reason']}; oracle {r['verdict']} on the 27B/Q38FN clauses is non-deciding)"
    return r


def _stage_oracle(cand, prim: Primitives, *, test_mode: bool) -> dict:
    r = _base("oracle-kld", cand)
    b = _require_built(cand)
    anchor = Path(cand.get("anchor_build") or _build_dir(cand, "anchor"))
    o = prim.oracle(cand, anchor, Path(b["build_dir"]))
    r.update(backend="gpu", op_oracle=o.get("op_oracle"))
    if not o.get("op_passed"):
        r.update(verdict="FAIL_CORRECTNESS (op oracle)", polar="FAIL")
        return r
    if o.get("kld_unavailable") or not o.get("graph_kld"):
        r.update(verdict=f"FUTILITY_HARNESS_INSTRUMENT ({o.get('kld_unavailable') or 'no graph KLD blocks'})")
        return r
    r["graph_kld"] = o["graph_kld"]
    # polar from the same policy gate_eval applies (gate_eval L2 re-derives it; a FAIL check always wins there)
    from . import gate_eval
    checks = gate_eval.policy_checks(r, gate_eval.load_policy(), "gpu", gate_eval._constants())
    bad = [c for c in checks if not c["passed"]]
    r.update(verdict="PASS_ORACLE" if checks and not bad else
             f"FAIL_CORRECTNESS ({bad[0]['gate']} {bad[0]['metric']}={bad[0]['observed']})" if bad else
             "FUTILITY_HARNESS_INSTRUMENT (no decisive KLD check)")
    if checks:
        r["polar"] = "FAIL" if bad else "PASS"
    return r


def stage_op_perf(cand, prim: Primitives, *, test_mode: bool) -> dict:
    """RSL-R3 SCREEN, never a gate: served-shape grid x registry n-set, futility-only per cell, weighted aggregate.
    A passing screen is filed FUTILITY (no polar), so gate_eval can never read it as a keep or a gate PASS."""
    r = _base("op-perf", cand)
    b = _require_built(cand)
    spec, sc = resolve_path(cand), cand.get("shape_class")
    if not isinstance(sc, Mapping):
        raise StageRefused("FUTILITY_HARNESS_MISSING_INPUT", "candidate carries no shape_class")
    lane = ssc.lane_for_model(spec["gguf"]) or ssc.dense_lane_for_model(spec["gguf"])
    r["op_perf_path"] = spec["key"]
    if lane is None:
        return {**r, "ungated": spec["key"],
                "verdict": f"FUTILITY_HARNESS_INSTRUMENT (op-perf ungated: no served-shape lane for {spec['key']})"}
    try:
        cells = ops.grid(sc, ssc.lane_served_shapes(lane), spec["n"])
    except (ValueError, KeyError, TypeError) as e:
        raise StageRefused("FUTILITY_HARNESS_INPUT", f"bad shape_class: {e}")
    anchor = Path(cand.get("anchor_build") or _build_dir(cand, "anchor"))
    m = prim.op_perf(cand, anchor, Path(b["build_dir"]), cells)
    res = ops.polar_result(ops.decide(m["cells"], m["weights"]), r)
    res["stage"] = "op-perf"
    if res.get("polar") == "PASS":
        res.pop("polar")
        res.update(screen_pass=True, verdict="FUTILITY_SCREEN_ONLY_NO_KEEP " + res["verdict"].replace("PASS_OP_PERF_SCREEN", "(op-perf screen passed)", 1))
    return res


def stage_timing(cand, prim: Primitives, *, test_mode: bool) -> dict:
    r = _base("ab-timing", cand)
    b = _require_built(cand)
    spec = resolve_path(cand)
    r["timing_path"] = {k: spec[k] for k in ("key", "status", "device", "host", "gguf")}
    if not timing_floor(spec, cand):   # no cell with its own A/A floor: never timed against a borrowed floor
        return {**r, "ungated": spec["key"],
                "verdict": f"FUTILITY_HARNESS_INSTRUMENT (ungated: no A/A floor cell for {spec['key']}; needs-aa-floor-row)"}
    cand = {**cand, "timing_model": spec["gguf"], "path_spec": spec}
    anchor = Path(cand.get("anchor_build") or _build_dir(cand, "anchor"))
    t = prim.timing(cand, anchor, Path(b["build_dir"]))
    from . import bench
    if int(t.get("pairs") or 0) < bench.MIN_PAIRS:
        r.update(verdict=f"FUTILITY_HARNESS_INSTRUMENT (pairs {t.get('pairs')} < {bench.MIN_PAIRS})")
        return r
    eff = float(t["effect_pct"])
    gmin = float(cand.get("geomean_min") or 1.0)
    r.update(backend="gpu", speed={"surface": spec["surface"], "effect_pct": eff, "pairs": int(t["pairs"]),
                                   "model": str(t.get("model") or spec["gguf"])},
             geomean=round(1.0 + eff / 100.0, 6), rule={"geomean_min": gmin}, timing=t)
    win = eff > 0 and bool(t.get("decisive", True))
    r.update(verdict="PASS_FASTER" if win else f"FAIL_NO_GAIN (effect {eff:+.3f}%)", polar="PASS" if win else "FAIL")
    return r


def stage_fold_check(cand, prim: Primitives, *, test_mode: bool) -> dict:
    """advance_champion CAS + P4 lineage check. Lands the keep ONLY on the base it was built from."""
    r = _base("fold-check", cand)
    b = _require_built(cand)
    for st in ("oracle-kld", "ab-timing"):
        if st in (cand.get("skip_stages") or ()):
            continue
        p = _prior_result(cand, st)
        if not p or p.get("polar") != "PASS":
            raise StageRefused("FUTILITY_HARNESS_MISSING_INPUT", f"{st} RESULT missing or not PASS")
    ref = str(cand.get("champion_ref") or "ak/champion/llama-cpp-ffc1bac82eec")
    guard_ref(ref)
    base, patch = str(cand["base_sha"]), str(cand["patch_sha"])
    parent = prim.parent_of(patch)
    head = prim.champion_head(ref)
    r.update(champion_ref=ref, champion_head_before=head, patch_parent=parent, built_sha=b.get("built_sha"))
    if parent != base:
        r.update(verdict=f"FUTILITY_HARNESS_SOURCE_MISMATCH (patch parent {parent[:12]} != base {base[:12]})")
        return r
    if head != base:
        # stale: never measured on the new head -> a rebase-and-recheck chain, not a keep and not a redraw
        r.update(verdict=f"FUTILITY_HARNESS_STALE_BASE (champion {head[:12]} != base {base[:12]})",
                 stale_base={"base_sha": base, "champion_head": head, "action": "rebase-and-recheck"})
        return r
    try:
        prim.cas(ref, patch, base)
    except subprocess.CalledProcessError as e:
        r.update(verdict=f"FUTILITY_HARNESS_STALE_BASE (CAS refused: {str(e.stderr or e)[:120]})",
                 stale_base={"base_sha": base, "champion_head": prim.champion_head(ref),
                             "action": "rebase-and-recheck"})
        return r
    r["champion_head_after"] = prim.champion_head(ref)
    lin = prim.lineage({"kind": "keep", "candidate": cand.get("id"), "keep_sha": patch, "base_sha": base,
                        "hypothesis_id": cand.get("hypothesis_id"), "champion_ref": ref,
                        **{k: cand[k] for k in ("llama_repo", "epyc_root") if cand.get(k)}})
    r["lineage"] = lin
    r.update(verdict="KEEP_LANDED", polar="PASS")
    return r


BODIES = {"build": stage_build, "oracle-kld": stage_oracle, "op-perf": stage_op_perf, "ab-timing": stage_timing, "fold-check": stage_fold_check}


# ---- test-mode primitives ---------------------------------------------------------------------------------------

def test_primitives(cand: Mapping[str, Any], *, fixture_repo: Path | None = None,
                    lineage_log: list | None = None) -> Primitives:
    """Fixture outcomes from cand["test_outcomes"] (stage -> pass|fail|harness). Ref moves happen ONLY in
    `fixture_repo` (a throwaway git repo); with none, an in-memory ref table is used."""
    out = dict(cand.get("test_outcomes") or {})
    refs: dict[str, str] = dict(cand.get("test_refs") or {})
    parents: dict[str, str] = dict(cand.get("test_parents") or {})

    def oc(st):
        return str(out.get(st, "pass"))

    def compile_(src, bd, cpu_list):
        o = oc("build")
        return {"passed": o == "pass", "harness": o == "harness", "reason": f"test-mode {o}", "verdicts": []}

    def oracle(c, a, bd):
        o = oc("oracle-kld")
        if o == "harness":
            return {"op_passed": True, "graph_kld": {}, "kld_unavailable": "test-mode harness"}
        kld = 0.0011 if o == "pass" else 0.0090
        return {"op_passed": True, "op_oracle": {"gate": "test", "passed": True},
                "graph_kld": {"q38_27b": {"kld_mean": kld, "dppl_pct": 0.1}, "q38fn": {"same_top_p_pct": 97.0}}}

    def timing(c, a, bd):
        o = oc("ab-timing")
        return {"effect_pct": 2.0 if o == "pass" else -0.4, "pairs": 5 if o != "harness" else 2, "decisive": True,
                "model": c.get("timing_model")}

    if fixture_repo is not None:
        head = lambda ref: _git(fixture_repo, "rev-parse", ref if ref.startswith("refs/") else f"refs/heads/{ref}")

        def cas(ref, new, old):
            guard_ref(ref)
            _git(fixture_repo, "update-ref", ref if ref.startswith("refs/") else f"refs/heads/{ref}", new, old)
        parent_of = lambda sha: _git(fixture_repo, "rev-parse", f"{sha}^")
    else:
        head = lambda ref: refs[ref]

        def cas(ref, new, old):
            guard_ref(ref)
            if refs.get(ref) != old:
                raise subprocess.CalledProcessError(1, "update-ref", stderr=f"{ref} is not at {old}")
            refs[ref] = new
        parent_of = lambda sha: parents[sha]

    def lineage(ev):
        if lineage_log is not None:
            lineage_log.append(dict(ev))
        if cand.get("test_lineage_log"):   # cross-process log for the offline simulation
            with open(cand["test_lineage_log"], "a") as f:
                f.write(json.dumps(dict(ev), sort_keys=True) + "\n")
        return {"lineage_out": "test-mode", "lineage_rc": 0, "refused": False}

    def op_perf(c, a, bd, cells):
        o = oc("op-perf")   # fail: the first cell regresses beyond its floor while every other cell wins
        mk = lambda i, x: {"key": x["key"], "anchor": [100.0] * 5, "floor_pct": 1.0,
                           "candidate": [] if o == "harness" else [110.0 if o == "fail" and i == 0 else 95.0] * 5}
        return {"cells": [mk(i, x) for i, x in enumerate(cells)], "weights": {x["key"]: 1.0 for x in cells}}

    return Primitives(source_tree=lambda c: Path(c.get("test_src") or "/nonexistent-test-src"), compile=compile_,
                      oracle=oracle, timing=timing, champion_head=head, cas=cas, parent_of=parent_of,
                      lineage=lineage, paths={"refs": refs}, op_perf=op_perf)


# ---- driver -----------------------------------------------------------------------------------------------------

def run_stage(stage: str, cand: Mapping[str, Any], *, lane: str, prim: Primitives | None = None,
              test_mode: bool = False, env: dict | None = None, cpus: set[int] | None = None) -> dict:
    if stage not in BODIES:
        raise ValueError(f"unknown stage {stage!r}")
    env = os.environ if env is None else env
    need = None   # the device a PATH stage's served path lives on; a CPU path is refused on the gpu lane and vice versa
    if stage in PATH_STAGES:
        try:
            need = resolve_path(cand)["lane"]
        except StageRefused as e:
            guard_device(stage, lane, env, "cpu")
            return {**_base(stage, cand), "verdict": f"{e.token} ({e})"}
    guard_device(stage, lane, env, need)
    if (stage in GPU_STAGES and need != "cpu" and lane != "gpu") or (need == "cpu" and lane != "cpu"):
        return {**_base(stage, cand), "verdict": f"FUTILITY_HARNESS_WRONG_GRANT ({need or 'gpu'} stage on the {lane} lane"
                + ("; a cpu path needs the served host's topology" if need == "cpu" else "") + ")"}
    if prim is None and test_mode:
        fx = cand.get("test_fixture_repo")
        prim = test_primitives(cand, fixture_repo=Path(fx) if fx else None)
    prim = prim or live_primitives()
    try:
        if stage == "build":
            aff = cpus if cpus is not None else (set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                                                 else set())
            return BODIES[stage](cand, prim, cpus=aff, test_mode=test_mode)
        return BODIES[stage](cand, prim, test_mode=test_mode)
    except StageRefused as e:
        return {**_base(stage, cand), "verdict": f"{e.token} ({e})"}
    except (subprocess.CalledProcessError, OSError, FileNotFoundError) as e:
        return {**_base(stage, cand), "verdict": f"FUTILITY_HARNESS_INSTRUMENT ({type(e).__name__}: {str(e)[:160]})"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lane", required=True, choices=["cpu", "gpu"])
    ap.add_argument("--row")
    ap.add_argument("--test-mode", action="store_true", default=os.environ.get("AK_STAGE_TEST_MODE") == "1")
    a = ap.parse_args(argv)
    cand = json.loads(a.candidate.read_text())
    if a.row:
        cand["row"] = a.row
    res = run_stage(a.stage, cand, lane=a.lane, test_mode=a.test_mode)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, a.out)
    print(json.dumps({"stage": a.stage, "verdict": res.get("verdict"), "polar": res.get("polar")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
