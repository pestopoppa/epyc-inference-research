#!/usr/bin/env python3
"""INF-45 E5 Stage-B v2 — the CORRECTED placement grid (S-04/S-05), plus depth probes.

Box: epyc-root handoffs/active/numa-placement-and-contention.md ->
"RE-MEASURE E5 Stage-B on the corrected placement". Five binding requirements:

  (a) full machine 0-95 + numactl --interleave=all is a SWEPT shape for both models
      (W1 qwen36_q8_0 and W4 re-seated on the current CPU role Qwen3.8-Flash-Next
      UD-IQ4_XS). W2 gemma is DROPPED: gemma-4-26B-A4B-it-ORIG-Q4_K_M is retired
      (research registry, 2026-09-22) and its GGUF is gone; its banked A7 group was
      adjudicated 0 valid / 4 invalid / 4 unresolved and is not re-run.
  (b) every straddling cpuset carries --interleave, or is DECLARED
      ``defect_replication_control: true``. Never implicit: the generator refuses to
      write a cell that is neither.
  (c) a per-cell cache reset + ``pages_by_node`` witness. The harness already does
      this (scripts/lib/cell_cache_reset.py): drop_caches under the quiet-host hold
      (q0..q3 + gpu-quiet exclusive), else the INF-70/C7 numa_evict path
      (scripts/utils/numa_evict.py) as the in-grant substitute; the row records
      ``cache_reset_method``. Cells declare ``cache_reset: per_cell`` and
      ``placement_protocol: P-BENCH-PLACEMENT-1`` so a numa_evict-only cell can never
      grade decision-grade (F2-CHK needs drop_caches per arm).
  (d) numactl policy attested per instance, fail closed: at generation time a policy
      that does not cover its cpuset's nodes is refused; after a run, ``attest``
      checks each instance's observed mmap pages_by_node against its declared policy
      and fails the cell closed on disagreement (or on a missing witness).
  (e) the harness (server_numa_np_sweep.py) is UNCHANGED. Depth probes ride the
      schema owner's opt-in ``depth_probe`` block (e5_cell_manifests.py).

Halves and private copies (pipeline-mgmt 2026-11-11). With a shared mmap the two C1b
instances share ONE page cache of the GGUF, so each half's weights interleave over all
four nodes. C1b cells therefore set ``no_mmap: true`` (harness emits --no-mmap) and each
half runs under ``numactl --membind`` of ITS OWN nodes (half0 membind=0,1, half1
membind=2,3): each process allocates a private weight copy local to its half. The C1b
defect controls (policy none, shared mmap) still replicate the old defect.

Depth probes (INF-45 capacity curves; >16k points the old grid lacked): C1 only
(production full-machine shape), np 1/2/4/8 at target depths 16k and 32k tokens,
one pinned long-context pool per depth (``build-pools``), content-pinned by sha256.

Subcommands:
  generate  [--out-dir D] [--dry-run]   emit + self-validate every group
  build-pools [--out-dir D]             (re)build the depth pools deterministically
  dryrun    --output-root R             harness dry invocation per group (no --execute)
  attest    --run-dir RUN               per-cell numactl-policy attestation (fail closed)
  groups                                print the group plan (json)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import e5_cell_manifests as e5  # noqa: E402

GRID_VERSION = "e5-stageb-v2"
DEFAULT_OUT = REPO / "data" / "batched_decode" / "e5_v2_manifests"
PLACEMENT_PROTOCOL = "P-BENCH-PLACEMENT-1"
HARNESS = HERE / "server_numa_np_sweep.py"

# NPS4 topology (EPYC 9655, 96 physical cores; SMT siblings +96).
NODE_CPUS = {0: (0, 23), 1: (24, 47), 2: (48, 71), 3: (72, 95)}

W4_KEY = "qwen38_flash_next_ud_iq4xs"
W4_MODEL = {
    "model_path": "/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/"
                  "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
    "quant": "UD-IQ4_XS",
    "architecture": "qwen4exp",
    "reasoning": "off",  # registry qwen38_flash_next_ud_iq4xs_local: reasoning "off"
    "port_base": 19480,
    "spec_dec": {
        # scripts/lib/qwen38_flash_next_recipe.py MTP_FLAGS: separate shared-Q8_0 head.
        "enabled": True,
        "spec_type": "draft-mtp",
        "draft_model_path": "/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/MTP/"
                            "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
        "draft_max": 4,
        "draft_min": None,
        "draft_p_min": 0.5,
        "draft_p_split": None,
        "threads_draft": None,
        "ngram_mod": None,
        "device_draft": "none",
        "record_accept_rate": True,
        "disabled_reason": None,
    },
    "notes": (
        "W4 re-seated 2026-11-11 on the current CPU role (registry "
        "qwen38_flash_next_ud_iq4xs_local). Recipe deviations held for E5 comparability, "
        "stated: KV q8_0 (recipe f16; protocol decision 1), mmap+mlock (recipe --no-mmap; "
        "harness has no knob), -t = cpuset size (recipe 48)."
    ),
}

MODEL_GROUPS = {
    "qwen36_q8_0": {"window": "W1"},
    W4_KEY: {"window": "W4"},
}

C1_NP = (1, 2, 4, 8, 16, 32)
C1B_NP = (1, 2, 4, 8, 16)
C1B_DEFECT_NP = (4, 8)
DEPTH_TARGETS = (16384, 32768)
DEPTH_NP = (1, 2, 4, 8)
DEPTH_N_PROMPTS = 8
CHARS_PER_TOKEN_EST = 3.6  # nominal; the harness records real prompt_tokens per request
DEPTH_SEED = 4545
MAX_GROUP_S = 4800  # one grant per group; min_window_s ~= the largest group

# 2026-07-29 Stage-B per-cell wall (cells.jsonl wall_seconds): W1 64.6 min / 11 cells,
# W4 (80B Q4_K_M) 88.3 min / 12 cells. Flash-Next is ~1.9x the 80B's bytes.
EST_CELL_S = {"qwen36_q8_0": 360, W4_KEY: 540}
EST_DEPTH_GROUP_S = {("qwen36_q8_0", 16384): 1800, ("qwen36_q8_0", 32768): 3600,
                     (W4_KEY, 16384): 2700, (W4_KEY, 32768): 5400}


# ---------------------------------------------------------------------------
# topology helpers
# ---------------------------------------------------------------------------

def nodes_of(cpu_list: str) -> list[int]:
    cpus = e5.parse_cpulist(cpu_list) or set()
    out = set()
    for c in cpus:
        phys = c % 96
        for n, (lo, hi) in NODE_CPUS.items():
            if lo <= phys <= hi:
                out.add(n)
    return sorted(out)


def policy_nodes(policy: str) -> list[int] | None:
    if policy == "interleave=all":
        return sorted(NODE_CPUS)
    if policy.startswith(("interleave=", "membind=")):
        return sorted(int(x) for x in policy.split("=", 1)[1].split(","))
    return None


def placement_violations(cell: dict) -> list[str]:
    """Requirements (b)+(d) at generation time. Fail closed."""
    errs = []
    defect = cell.get("defect_replication_control")
    if not isinstance(defect, bool):
        errs.append("defect_replication_control must be stated (true|false), never implicit")
    for i, inst in enumerate(cell.get("instances") or []):
        cpu_nodes = nodes_of(inst["cpu_list"])
        pol = inst.get("numactl_policy", "none")
        pn = policy_nodes(pol)
        straddles = len(cpu_nodes) > 1
        if pn is None:
            if straddles and defect is not True:
                errs.append(f"instances[{i}] straddles nodes {cpu_nodes} with policy none and is "
                            "not a declared defect_replication_control")
        else:
            if not set(cpu_nodes) <= set(pn):
                errs.append(f"instances[{i}] policy {pol} does not cover its cpuset nodes {cpu_nodes} "
                            "(policy/cpuset mismatch)")
            if pol.startswith("membind=") and (set(pn) != set(cpu_nodes) or not cell.get("no_mmap")):
                errs.append(f"instances[{i}] {pol} must equal its cpuset nodes {cpu_nodes} and needs "
                            "no_mmap (a shared mmap cannot be bound per instance)")
            if defect is True:
                errs.append(f"instances[{i}] carries {pol} but the cell is tagged a defect control")
    return errs


# ---------------------------------------------------------------------------
# cell construction
# ---------------------------------------------------------------------------

def _instances(model_key: str, shape: str) -> list[dict]:
    base = e5.MODELS[model_key]["port_base"] if model_key in e5.MODELS else W4_MODEL["port_base"]
    if shape == "C1":
        return [{"cpu_list": e5.CPUSET_FULL, "threads": 96, "numactl_policy": "interleave=all", "port": base}]
    if shape == "C1b-defect":
        pol0 = pol1 = "none"
    else:  # C1b: private --no-mmap copy per half, bound to the half's own nodes
        pol0 = "membind=" + ",".join(map(str, nodes_of(e5.CPUSET_HALF0)))
        pol1 = "membind=" + ",".join(map(str, nodes_of(e5.CPUSET_HALF1)))
    return [{"cpu_list": e5.CPUSET_HALF0, "threads": 96, "numactl_policy": pol0, "port": base},
            {"cpu_list": e5.CPUSET_HALF1, "threads": 96, "numactl_policy": pol1, "port": base + 1}]


def _families(shape: str, np_: int) -> list[str]:
    n_inst = 1 if shape == "C1" else 2
    T = n_inst * np_
    fam = []
    if shape in ("C1", "C1b") and T in (2, 4, 8, 16, 32):
        fam.append(f"isoT_T{T}")  # pre-registered iso-T pair: C1@T vs C1b@T/2
    if shape == "C1" and np_ == 1:
        fam.append("anchor_C1")
    if shape == "C1b-defect":
        fam.append(f"defect_control_T{T}")
    return fam


def make_v2_cell(model_key: str, shape: str, np_: int, depth: dict | None = None) -> dict:
    window = MODEL_GROUPS[model_key]["window"]
    config_id = "C1" if shape == "C1" else "C1b"
    cell = e5.make_cell("qwen36_q8_0", config_id, np_, window,
                        decision_grade_intent=True, n_predict=e5.N_PREDICT_STAGE_B)
    if model_key == W4_KEY:
        for k in ("model_path", "quant", "architecture", "reasoning"):
            cell[k] = W4_MODEL[k]
        cell["spec_dec"] = copy.deepcopy(W4_MODEL["spec_dec"])
        cell["notes"] = W4_MODEL["notes"]
        cell["chat_template_kwargs"] = {"enable_thinking": False}
    cell["model_key"] = model_key
    cell["instances"] = _instances(model_key, shape)
    if shape == "C1b":  # key absent elsewhere: C1 / defect-control manifests stay byte-identical
        cell["no_mmap"] = True
        cell["notes"] = ((cell.get("notes") or "") + " C1b (2026-11-11): --no-mmap private weight "
                         "copy per half + numactl --membind of the half's own nodes.").strip()
    cell["ubatch_size"] = e5.UBATCH_96T
    suffix = {"C1": "", "C1b": "", "C1b-defect": "-defectctl"}[shape]
    cell["defect_replication_control"] = shape == "C1b-defect"
    cell["stage_b_families"] = _families(shape, np_)
    cell["grid_version"] = GRID_VERSION
    cell["placement_protocol"] = PLACEMENT_PROTOCOL
    cell["placement_declared"] = True
    cell["cache_reset"] = "per_cell"
    cell["cache_reset_mechanism"] = (
        "harness scripts/lib/cell_cache_reset.py: drop_caches under the quiet-host hold "
        "(q0..q3 + gpu-quiet exclusive held by this process tree), else numa_evict "
        "(scripts/utils/numa_evict.py, INF-70/C7) scoped to the cell's nodes; the row's "
        "cache_reset_method records which ran, cache_reset.pages_by_node before/after")
    cell["policy_attestation"] = {
        "tool": "scripts/benchmark/e5_v2_stageb_grid.py attest",
        "expected_nodes_per_instance": [policy_nodes(i["numactl_policy"]) or nodes_of(i["cpu_list"])
                                        for i in cell["instances"]],
        "shared_mmap": len(cell["instances"]) > 1 and not cell.get("no_mmap"),
    }
    if depth is not None:
        tgt = depth["target_depth_tokens"]
        cell["depth_probe"] = depth
        cell["per_stream_ctx"] = depth["per_stream_ctx"]
        cell["ctx"] = depth["per_stream_ctx"] * np_
        cell["prompt_caps"]["max_prompt_chars"] = depth["max_prompt_chars"]
        cell["prompt_batch"] = {
            "source": depth["pool_path"], "selection": "pinned_qids", "qids": list(depth["qids"]),
            "pinned_from": f"sha256:{depth['pool_sha256']}", "tier": 1, "seed": 42,
            "limit": len(depth["qids"]),
        }
        cell["stage_b_families"] = [f"depth_{tgt}"]
        suffix = f"-d{tgt // 1024}k"
    cell["cell_id"] = f"{model_key}-{config_id}-np{np_}{suffix}"
    return cell


def group_name(model_key: str, depth: int | None) -> str:
    return model_key if depth is None else f"{model_key}-d{depth // 1024}k"


# ---------------------------------------------------------------------------
# depth pools
# ---------------------------------------------------------------------------

def _pool_rows(pool_path: Path) -> list[dict]:
    rows = []
    with pool_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("__pool_metadata__"):
                continue
            text = r.get("prompt")
            if isinstance(r.get("context"), str) and r["context"]:
                text = f"{r['context']}\n\n{text}"
            if isinstance(text, str) and 200 <= len(text) <= 20000:
                rows.append({"id": str(r.get("id")), "text": text})
    rows.sort(key=lambda r: hashlib.sha256(f"{DEPTH_SEED}:{r['id']}".encode()).hexdigest())
    return rows


def build_depth_pool(target_tokens: int, source: Path, n_prompts: int = DEPTH_N_PROMPTS) -> list[dict]:
    """Deterministic long-context prompts: unrelated pool documents concatenated to
    ~target tokens, then one closing question about the LAST document (so the model
    must read to the end; the answer itself is not scored here)."""
    rows = _pool_rows(source)
    target_chars = int(target_tokens * CHARS_PER_TOKEN_EST)
    out, cursor = [], 0
    for k in range(n_prompts):
        parts, total = [], 0
        while total < target_chars:
            r = rows[cursor % len(rows)]
            cursor += 1
            block = f"### Document {len(parts) + 1}\n{r['text']}\n"
            if total + len(block) > target_chars and parts:
                block = block[: target_chars - total]
            parts.append(block)
            total += len(block)
        prompt = ("".join(parts) + "\n\nIn two sentences, state what the last document above asks for.")
        out.append({"id": f"e5v2_depth{target_tokens}_{k:02d}", "suite": "e5v2_depth", "prompt": prompt})
    return out


def pool_path_for(out_dir: Path, target: int) -> Path:
    return out_dir / "depth_pools" / f"depth_pool_{target}.jsonl"


def write_pool(rows: list[dict], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    path.write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode()).hexdigest()


def depth_block(target: int, rows: list[dict], path: Path, sha: str) -> dict:
    max_chars = max(len(r["prompt"]) for r in rows) + 256
    per_stream = -(-int(target * 1.3 + e5.N_PREDICT_STAGE_B + 512) // 1024) * 1024
    return {"target_depth_tokens": target, "per_stream_ctx": per_stream, "max_prompt_chars": max_chars,
            "qids": [r["id"] for r in rows], "pool_path": str(path), "pool_sha256": sha,
            "builder": "scripts/benchmark/e5_v2_stageb_grid.py build-pools",
            "chars_per_token_est": CHARS_PER_TOKEN_EST,
            "note": "nominal depth; the realised depth is each request's prompt_tokens"}


def load_depth_blocks(out_dir: Path, published_root: Path | None = None) -> dict[int, dict]:
    """Depth blocks from the committed pools. ``published_root`` rewrites pool_path to the
    canonical research checkout path the harness will read at run time."""
    blocks = {}
    for t in DEPTH_TARGETS:
        p = pool_path_for(out_dir, t)
        data = p.read_bytes()
        rows = [json.loads(l) for l in data.decode().splitlines() if l.strip()]
        rp = (published_root / p.relative_to(REPO)) if published_root else p
        blocks[t] = depth_block(t, rows, rp, hashlib.sha256(data).hexdigest())
    return blocks


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

def build_groups(depth_blocks: dict[int, dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for mk in MODEL_GROUPS:
        cells = [make_v2_cell(mk, "C1", n) for n in C1_NP]
        cells += [make_v2_cell(mk, "C1b", n) for n in C1B_NP]
        cells += [make_v2_cell(mk, "C1b-defect", n) for n in C1B_DEFECT_NP]
        groups[group_name(mk, None)] = cells
        for t in DEPTH_TARGETS:
            groups[group_name(mk, t)] = [make_v2_cell(mk, "C1", n, depth_blocks[t]) for n in DEPTH_NP]
    return groups


def cell_estimate_s(cell: dict) -> int:
    depth = (cell.get("depth_probe") or {}).get("target_depth_tokens")
    mk = cell["model_key"]
    return EST_DEPTH_GROUP_S[(mk, depth)] // len(DEPTH_NP) if depth else EST_CELL_S[mk]


def split_groups(groups: dict[str, list[dict]], max_s: int = MAX_GROUP_S) -> dict[str, list[dict]]:
    """One group = one grant. A group whose estimate exceeds max_s is split into
    consecutive parts (-p1, -p2, ...) in grid order, so every grant fits min_window."""
    out: dict[str, list[dict]] = {}
    for g, cells in groups.items():
        total = sum(cell_estimate_s(c) for c in cells)
        if total <= max_s:
            out[g] = cells
            continue
        parts = -(-total // max_s)
        size = -(-len(cells) // parts)
        for k in range(parts):
            chunk = cells[k * size:(k + 1) * size]
            if chunk:
                out[f"{g}-p{k + 1}"] = chunk
    return out


def group_plan(groups: dict[str, list[dict]]) -> list[dict]:
    return [{"group": g, "model_key": cells[0]["model_key"], "n_cells": len(cells),
             "estimate_s": sum(cell_estimate_s(c) for c in cells),
             "cells": [c["cell_id"] for c in cells]} for g, cells in groups.items()]


def validate_all(groups: dict[str, list[dict]]) -> list[str]:
    errs = []
    for g, cells in groups.items():
        for c in cells:
            for v in e5.validate_cell_manifest(c) + placement_violations(c):
                errs.append(f"{g}/{c['cell_id']}: {v}")
        if len({c["model_key"] for c in cells}) != 1:
            errs.append(f"{g}: mixed model keys")
    return errs


def cmd_generate(a) -> int:
    out = Path(a.out_dir)
    blocks = load_depth_blocks(out, Path(a.published_root) if a.published_root else None)
    groups = split_groups(build_groups(blocks))
    errs = validate_all(groups)
    if errs:
        print("REFUSING: generated cells failed validation:\n  " + "\n  ".join(errs))
        return 1
    n = 0
    for g, cells in groups.items():
        for c in cells:
            n += 1
            if not a.dry_run:
                p = out / g / f"{c['cell_id']}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(c, indent=2) + "\n")
    if not a.dry_run:
        (out / "groups.json").write_text(json.dumps(group_plan(groups), indent=2) + "\n")
    for row in group_plan(groups):
        print(f"{row['group']:36s} {row['n_cells']:3d} cells  est {row['estimate_s'] // 60} min")
    print(f"{'would write' if a.dry_run else 'wrote'} {n} manifests under {out}; all valid")
    return 0


def cmd_build_pools(a) -> int:
    out = Path(a.out_dir)
    for t in DEPTH_TARGETS:
        rows = build_depth_pool(t, Path(a.source))
        sha = write_pool(rows, pool_path_for(out, t))
        print(f"depth {t}: {len(rows)} prompts, max {max(len(r['prompt']) for r in rows)} chars, sha256 {sha}")
    return 0


def verified_pool(out_dir: Path, depth: dict) -> Path:
    """The local depth pool for a depth block, sha256-verified against its pin
    (fail closed). Runs pass it to the harness via --question-pool."""
    p = pool_path_for(out_dir, depth["target_depth_tokens"])
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != depth["pool_sha256"]:
        raise SystemExit(f"depth pool {p} sha256 {got} != pinned {depth['pool_sha256']}")
    return p


def cmd_pool_for(a) -> int:
    m = json.loads(Path(a.manifest).read_text())
    if not m.get("depth_probe"):
        return 0
    print(verified_pool(Path(a.out_dir), m["depth_probe"]))
    return 0


def cmd_dryrun(a) -> int:
    out = Path(a.out_dir)
    plan = json.loads((out / "groups.json").read_text())
    rc_all = 0
    for row in plan:
        files = sorted((out / row["group"]).glob("*.json"))
        argv = [sys.executable, str(HARNESS), "--llama-server", a.llama_server,
                "--output-root", a.output_root, "--run-id", f"e5v2-dry-{row['group']}"]
        for f in files:
            argv += ["--cell-manifest", str(f)]
        depth = json.loads(files[0].read_text()).get("depth_probe")
        if depth:
            argv += ["--question-pool", str(verified_pool(out, depth))]
        proc = subprocess.run(argv, capture_output=True, text=True)
        print(f"{row['group']:36s} {len(files):3d} manifests  harness dry-run rc={proc.returncode}")
        if proc.returncode != 0:
            print(proc.stdout[-2000:], proc.stderr[-2000:])
            rc_all = 1
    return rc_all


# ---------------------------------------------------------------------------
# attestation (requirement d, post-run, fail closed)
# ---------------------------------------------------------------------------

LOCAL_SHARE_MIN = 0.85
BALANCE_MIN = 0.5  # each interleave node holds >= BALANCE_MIN x its even share


def attest_instance(declared_policy: str, cpu_list: str, pages_by_node: dict | None,
                    defect_control: bool) -> tuple[str, str]:
    if not pages_by_node:
        return "FAIL", "no pages_by_node witness"
    pages = {int(k.lstrip("N")): int(v) for k, v in pages_by_node.items()}
    total = sum(pages.values())
    if total <= 0:
        return "FAIL", "zero model pages observed"
    pn = policy_nodes(declared_policy)
    if pn is None:
        if defect_control:
            return "DEFECT-CONTROL", "policy none on a declared defect control; layout recorded"
        nodes = nodes_of(cpu_list)
        share = sum(pages.get(n, 0) for n in nodes) / total
        return ("PASS", f"first-touch local share {share:.3f}") if share >= LOCAL_SHARE_MIN else (
            "FAIL", f"policy none but only {share:.3f} of pages on cpuset nodes {nodes}")
    share = sum(pages.get(n, 0) for n in pn) / total
    if share < LOCAL_SHARE_MIN:
        return "FAIL", f"{declared_policy}: only {share:.3f} of pages inside nodes {pn}"
    if declared_policy.startswith("membind="):  # bind: locality only, no balance requirement
        return "PASS", f"{declared_policy}: private copy, share {share:.3f} inside nodes {pn}"
    even = total / len(pn)
    thin = [n for n in pn if pages.get(n, 0) < BALANCE_MIN * even]
    if thin:
        return "FAIL", f"{declared_policy}: nodes {thin} under-populated (not interleaved; warm cache?)"
    return "PASS", f"{declared_policy}: share {share:.3f}, balanced"


def attest_run(run_dir: Path) -> list[dict]:
    out = []
    for line in (run_dir / "cells.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row.get("cell_id")
        verdicts = []
        art_path = run_dir / "affinity" / f"{cid}.json"
        art = json.loads(art_path.read_text()) if art_path.exists() else {}
        insts = art.get("instances") or []
        defect = bool(row.get("defect_replication_control"))
        for i, inst in enumerate(row.get("instances") or []):
            obs = insts[i] if i < len(insts) else {}
            v, why = attest_instance(inst.get("numactl_policy", "none"), inst.get("cpu_list", ""),
                                     (obs.get("memory_placement") or {}).get("pages_by_node"), defect)
            verdicts.append({"instance": i, "verdict": v, "why": why})
        method = row.get("cache_reset_method")
        cell_v = "PASS"
        if not verdicts or any(v["verdict"] == "FAIL" for v in verdicts):
            cell_v = "FAIL"
        elif any(v["verdict"] == "DEFECT-CONTROL" for v in verdicts):
            cell_v = "DEFECT-CONTROL"
        if method is None:
            cell_v = "FAIL"
        out.append({"cell_id": cid, "policy_attestation": cell_v, "cache_reset_method": method,
                    "instances": verdicts})
    return out


def cmd_attest(a) -> int:
    run = Path(a.run_dir)
    res = attest_run(run)
    dest = Path(a.output) if a.output else run / "policy_attestation.jsonl"
    dest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in res))
    bad = [r["cell_id"] for r in res if r["policy_attestation"] == "FAIL"]
    print(f"attested {len(res)} cells -> {dest}; FAIL {len(bad)}: {bad}")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out-dir", default=str(DEFAULT_OUT))
    g.add_argument("--published-root", default="/mnt/raid0/llm/epyc-inference-research",
                   help="checkout path recorded as the depth pool source (\"\" = this tree)")
    g.add_argument("--dry-run", action="store_true", help="validate only, write nothing")
    g.set_defaults(func=cmd_generate)
    b = sub.add_parser("build-pools")
    b.add_argument("--out-dir", default=str(DEFAULT_OUT))
    b.add_argument("--source", default=e5.QUESTION_POOL_PATH)
    b.set_defaults(func=cmd_build_pools)
    d = sub.add_parser("dryrun")
    d.add_argument("--out-dir", default=str(DEFAULT_OUT))
    d.add_argument("--output-root", required=True)
    d.add_argument("--llama-server", default="/mnt/raid0/llm/kernels/production/cpu/llama-server")
    d.set_defaults(func=cmd_dryrun)
    t = sub.add_parser("attest")
    t.add_argument("--run-dir", required=True)
    t.add_argument("--output", default=None)
    t.set_defaults(func=cmd_attest)
    q = sub.add_parser("pool-for", help="print the sha-verified local depth pool for a manifest (empty if none)")
    q.add_argument("--out-dir", default=str(DEFAULT_OUT))
    q.add_argument("manifest")
    q.set_defaults(func=cmd_pool_for)
    p = sub.add_parser("groups")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.set_defaults(func=lambda a: print((Path(a.out_dir) / "groups.json").read_text()) or 0)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
