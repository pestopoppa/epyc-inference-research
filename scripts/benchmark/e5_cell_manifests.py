#!/usr/bin/env python3
"""E5 cell-manifest schema owner + pre-registered NUMAxnp grid generator.

Schema version ``e5-cell-manifest/1``. One JSON file per cell; the multi-server
harness (``server_numa_np_sweep.py``) and the affinity-preflight cell mode
(epyc-orchestrator ``affinity_preflight.py --cell-manifest``) consume these
files as their cross-repo contract. This module is the ONLY owner of the
schema: it exposes ``SCHEMA_VERSION``, ``validate_cell_manifest()`` (returns a
list of human-readable violation strings; empty list = valid) and a CLI:

    generate  — emit the full pre-registered W0 scout + Stage-B cell grid as
                one JSON file per cell under data/batched_decode/e5_manifests/,
                grouped by model_key.
    validate  — validate manifest file(s); exit 1 on any violation.

Spec source of truth: epyc-root handoffs/active/batched-decode-measurement.md
("2026-07-23 — E5 harness preparation" cell grid + protocol decisions 1-6 +
decision rules R1-R4, and the "Pre-execution audit (2026-07-23)" corrections
C1-C4). Pre-registered grid encoded here:

- Models (indexed by model+quant, NEVER role — feedback_model_not_role_indexing):
  qwen36_q8_0 (35B-A3B MoE MTP Q8), qwen36_27b_q8 (dense control, MTP Q8),
  qwen3_next_80b (SSM+MoE hybrid ingest arm, W4), gemma4_26b_a4b_q4km_mtp.
- Configs are ALTERNATIVES, never co-deployed: C1 (1xhalf0; gemma: 1xfull +
  interleave — NO half shape, half-pinning crashes the MTP draft path), C1b
  (2xhalf, one per NUMA node; half1 is SYNTHESIZED on bench ports only), C2
  (2xq on q2+q3, mechanism probe), C3 (4xq — the status-quo production shape
  per audit C4; C1 is a provisioning CANDIDATE + E1-continuity anchor).
- K ladders (K = per-instance -np): C1 x{1,2,4,8,16,32}; C1b/C2 x{1,2,4,8,16};
  C3 x{1,2,4,8}; total in-flight len(instances)*np <= 43 (fixed P-BENCH-3 batch).
- ctx sizing (protocol decision 1): -c = max(8192, 2048*np), per_stream_ctx
  2048 verified against the pinned batch (max server-measured prompt 557 tok +
  256 gen cap + cross-tokenizer/template/draft-ahead margin).
- Prompt batch: pinned 43-qid replay of the E1 selection. Re-sampling
  tier/seed from the current pool is FORBIDDEN — the pool was rebuilt
  2026-07-21 (E7 boundary) and re-sampling selects a ~101k-char prompt.
- Spec-dec (protocol decision 2): production TOP recipe per model, never an
  unoptimized baseline. qwen36 arms: NEXTN self-draft (draft-mtp dm=4 ps=0,
  no -md). gemma: draft-mtp dm=2 full launch recipe (assistant-v6 drafter,
  draft_p_min 0.0, threads_draft 16, ub 512). qwen3_next_80b: spec-dec OFF is
  the top recipe for the SSM/hybrid arch (registry quirk: SSM incompatible
  with all speculation) — explicitly labeled, not a wedge fallback.
- kv_unified (protocol decision 1 correction): per-cell manifest+attestation
  field, default false (split KV, E1-comparable realized-stack shape); true
  only for the scout paired probe (qwen36 C1@16 -kvu); MUST be false for
  SSM/hybrid arms (tree-spec Phase-8 scar).
- Ports: bench range [19000, 19999] ONLY (prod is 8070-8485 / 18070 / 8000 /
  8090-8095); deterministic per model+instance, no collisions within a cell.
- ubatch per shape: 96t shapes -ub 8192 (E1 parity), 48t quarters -ub 512
  (feedback_psplit_default); gemma always 512 (MTP recipe override).

Additive (optional) fields beyond the frozen schema, documented for consumers:
``stage_b_families`` (list[str] iso-T/scaling/anchor pairing tags for the
summarizer), ``model_path_present`` / ``draft_model_path_present`` (bool,
generation-time file-presence flags — absent files are FLAGGED, not fatal).
Consumers must ignore unknown additive fields; unknown ``schema_version``
values are refused (fail closed on drift).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "data" / "batched_decode" / "e5_manifests"

SCHEMA_VERSION = "e5-cell-manifest/1"
PROTOCOL_ID = "P-BENCH-3"

# Instrument-era stamps (verified against epyc-orchestrator
# orchestration/instrument_eras.yaml — E6-cpu-kernel: v7 cutover
# 2026-07-20T13:30:13Z; E7-eval-instrument: pool/scorer boundary 2026-07-21).
ERA_CPU_KERNEL = "E6-cpu-kernel"
ERA_EVAL_INSTRUMENT = "E7-eval-instrument"
ERA_SOURCE = "epyc-orchestrator/orchestration/instrument_eras.yaml"
INSTRUMENT_ERAS_PATH = Path(
    "/mnt/raid0/llm/epyc-orchestrator/orchestration/instrument_eras.yaml"
)

# NUMA shapes — mirror epyc-orchestrator scripts/server/stack_numa.py constants.
# NODE1 (half1) has NO production instance; the harness synthesizes it on bench
# ports only (no prod cpuset changes).
CPUSET_Q0A = "0-23,96-119"
CPUSET_Q0B = "24-47,120-143"
CPUSET_Q1A = "48-71,144-167"
CPUSET_Q1B = "72-95,168-191"
CPUSET_HALF0 = "0-47,96-143"
CPUSET_HALF1 = "48-95,144-191"
CPUSET_FULL = "0-95"

PORT_MIN = 19000
PORT_MAX = 19999

PER_STREAM_CTX = 2048
CTX_FLOOR = 8192
N_PREDICT_STAGE_B = 256
N_PREDICT_SCOUT = 64
MAX_PROMPT_CHARS = 4096
MAX_TOTAL_IN_FLIGHT = 43
WARMUP_PROMPTS = 1
WARMUP_N_PREDICT = 32
UBATCH_96T = 8192
UBATCH_48T = 512

VALID_WINDOWS = ("W0", "W1", "W2", "W3", "W4")
VALID_CONFIG_IDS = ("C1", "C1b", "C2", "C3")
VALID_NUMACTL_POLICIES = ("none", "interleave=all")
CONFIG_INSTANCE_COUNT = {"C1": 1, "C1b": 2, "C2": 2, "C3": 4}
K_LADDER = {
    "C1": (1, 2, 4, 8, 16, 32),
    "C1b": (1, 2, 4, 8, 16),
    "C2": (1, 2, 4, 8, 16),
    "C3": (1, 2, 4, 8),
}
# Architectures for which kv_unified MUST stay false (tree-spec Phase-8 scar:
# hybrid + kv_unified allocator/acceptance failures).
SSM_HYBRID_ARCHITECTURES = ("ssm_hybrid", "ssm_moe_hybrid")

QUESTION_POOL_PATH = (
    "/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl"
)
E1_PINNED_FROM = (
    "data/batched_decode/e1-pbench3-clean-20260703T1912Z/selected_prompts.jsonl"
)

# The fixed P-BENCH-3 prompt batch: the 43 qids of the E1 clean-run selection
# (tier=1, seed=42, limit=43 at the PRE-2026-07-21 pool — recorded for
# provenance only; selection is by pinned qid replay, never re-sampling).
# Copied verbatim (order preserved) from
# data/batched_decode/e1-pbench3-clean-20260703T1912Z/selected_prompts.jsonl.
E1_PINNED_QIDS = [
    "debugbench_fizz-buzz_python",
    "bcb_BigCodeBench/819",
    "simpleqa_general_02785",
    "simpleqa_general_01165",
    "gsm8k_01291",
    "mmlu_miscellaneous_07724",
    "debugbench_final-prices-with-a-special-discount-in-a-shop_java",
    "hellaswag_34416",
    "debugbench_plus-one_cpp",
    "hellaswag_45409",
    "hellaswag_04688",
    "bcb_BigCodeBench/1041",
    "bcb_BigCodeBench/976",
    "debugbench_transpose-matrix_java",
    "gsm8k_01141",
    "simpleqa_general_00501",
    "hellaswag_24725",
    "hellaswag_48346",
    "bcb_BigCodeBench/869",
    "hellaswag_38109",
    "gsm8k_00492",
    "hellaswag_34292",
    "hellaswag_04032",
    "gsm8k_01200",
    "hellaswag_11520",
    "hellaswag_45031",
    "simpleqa_general_02940",
    "bcb_BigCodeBench/212",
    "ifeval_1592",
    "hellaswag_04893",
    "hellaswag_16769",
    "simpleqa_general_02925",
    "mmlu_us_foreign_policy_13684",
    "gsm8k_01032",
    "hellaswag_15114",
    "debugbench_increasing-decreasing-string_java",
    "debugbench_maximum-number-of-balloons_java",
    "hellaswag_38184",
    "debugbench_remove-palindromic-subsequences_java",
    "hellaswag_27524",
    "hellaswag_18832",
    "hellaswag_48793",
    "simpleqa_general_02228",
]

# Per-model spec-dec TOP recipes (protocol decision 2 / operator directive:
# "anchor" NEVER means unoptimized). Field names mirror the orchestrator flag
# surface (orchestrator_stack.py _append_spec_decode_args).
_SPEC_QWEN36 = {
    # NEXTN self-draft: draft = same GGUF, launcher omits -md (draft_model_path
    # null). Registry frontdoor recipe: spec_type draft-mtp, draft_max 4,
    # p_split 0 (linear).
    "enabled": True,
    "spec_type": "draft-mtp",
    "draft_model_path": None,
    "draft_max": 4,
    "draft_min": None,
    "draft_p_min": None,
    "draft_p_split": 0,
    "threads_draft": None,
    "ngram_mod": None,
    "device_draft": "none",
    "record_accept_rate": True,
    "disabled_reason": None,
}
_SPEC_GEMMA = {
    # gemma4 MTP launch recipe (stack_numa worker spec_overrides dm=2 ps=0 +
    # registry acceleration block + live production launch cmd): draft-mtp with
    # the assistant-v6 Q8 drafter, greedy p_min 0.0, 16 dedicated draft
    # threads, -ub 512 (the ubatch_size field carries the 512 override).
    "enabled": True,
    "spec_type": "draft-mtp",
    "draft_model_path": "/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf",
    "draft_max": 2,
    "draft_min": None,
    "draft_p_min": 0.0,
    "draft_p_split": 0,
    "threads_draft": 16,
    "ngram_mod": None,
    "device_draft": "none",
    "record_accept_rate": True,
    "disabled_reason": None,
}
_SPEC_QWEN3_NEXT = {
    # SSM/hybrid: spec-dec OFF is the model's TOP production recipe (registry
    # quirk qwen3_next_80b: "SSM models incompatible with all speculation
    # methods"). Explicitly labeled per protocol decision 2 — never silent,
    # and NOT a wedge fallback.
    "enabled": False,
    "spec_type": None,
    "draft_model_path": None,
    "draft_max": None,
    "draft_min": None,
    "draft_p_min": None,
    "draft_p_split": None,
    "threads_draft": None,
    "ngram_mod": None,
    "device_draft": "none",
    "record_accept_rate": False,
    "disabled_reason": (
        "ssm_hybrid_no_speculation: SSM/MoE-hybrid arch has no working "
        "speculation path (registry quirk qwen3_next_80b) — spec-off IS the "
        "top production recipe for this arm, not a wedge fallback"
    ),
}

# Pre-registered model table. Paths verified against the audited file set
# (pre-execution audit "Models all present"); the generator re-checks at
# generation time and FLAGS absent files instead of failing.
MODELS = {
    "qwen36_q8_0": {
        # MTP variant GGUF (NEXTN self-draft requires it; the non-MTP
        # Qwen_Qwen3.6-35B-A3B-Q8_0.gguf E1 file cannot self-draft).
        "model_path": "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf",
        "quant": "Q8_0",
        "architecture": "qwen35moe",
        "port_base": 19080,
        "configs": ("C1", "C1b", "C2", "C3"),
        "stage_b_window": "W1",
        "spec_dec": _SPEC_QWEN36,
        "full_only": False,
        "notes": "",
    },
    "qwen36_27b_q8": {
        "model_path": "/mnt/raid0/llm/models/Qwen3.6-27B-MTP-Q8_0.gguf",
        "quant": "Q8_0",
        "architecture": "qwen35",
        "port_base": 19180,
        "configs": ("C1", "C1b", "C2", "C3"),
        "stage_b_window": "W3",
        "spec_dec": _SPEC_QWEN36,
        "full_only": False,
        "notes": (
            "dense control; NEXTN self-draft recipe mirrored from the 35B arm "
            "(dm=4, ps=0) on the MTP-variant GGUF"
        ),
    },
    "qwen3_next_80b": {
        # Registry key caveat: qwen3_next_80b is not resolvable to a model.path
        # under that key in the research registry — path resolved explicitly
        # (audit-confirmed Q4_K_M, lmstudio-community; NOT the i1-IQ2_M file).
        "model_path": (
            "/mnt/raid0/llm/lmstudio/models/lmstudio-community/"
            "Qwen3-Next-80B-A3B-Instruct-GGUF/"
            "Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf"
        ),
        "quant": "Q4_K_M",
        "architecture": "ssm_moe_hybrid",
        "port_base": 19280,
        "configs": ("C1", "C1b", "C3"),
        "stage_b_window": "W4",
        "spec_dec": _SPEC_QWEN3_NEXT,
        "full_only": False,
        "notes": (
            "ingest arm (operator-added 2026-07-23); kv_unified forced false "
            "(SSM/hybrid, tree-spec Phase-8 scar); C1b half-pair ratio is "
            "same-shape evidence for WP-9"
        ),
    },
    "gemma4_26b_a4b_q4km_mtp": {
        # Production worker model pair per the lean/research registry worker
        # entries + live launch cmd (ORIG-Q4_K_M + assistant-v6-Q8_0 drafter).
        # The audit prose names gemma-4-26B-A4B-it-Q4_K_M-current.gguf; the
        # registry/live-production path is authoritative here (M29 top recipe).
        "model_path": "/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "architecture": "gemma4",
        "port_base": 19380,
        "configs": ("C1", "C3"),
        "stage_b_window": "W2",
        "spec_dec": _SPEC_GEMMA,
        "full_only": True,  # NO half shape: half-pinning crashes the MTP draft path
        "notes": (
            "gemma MTP: 1xfull(0-95)+interleave or 4xq ONLY — no half shape "
            "(half-pinning crashes the MTP draft path, 'tensor buffer not set'); "
            "-ub 512 MTP recipe override on every shape"
        ),
    },
}

# Stage-B decision-grade K sets per model family (iso-T pairs {C1b@T/2 vs
# C3@T/4} T in {8,16,32}; mechanism {C1@T vs C2@T/2} T in {16,32}; scaling
# {C1@K vs C1b@K} K in {4,8}; anchors C1@1 / C3@1). Gemma: whole-machine
# pairs are {C1full@T vs C3@T/4} plus the C1full@1 anchor (~8 cells).
STAGE_B_K = {
    "qwen36_q8_0": {"C1": (1, 4, 8, 16, 32), "C1b": (4, 8, 16), "C2": (8, 16), "C3": (1, 2, 4, 8)},
    "qwen36_27b_q8": {"C1": (1, 4, 8, 16, 32), "C1b": (4, 8, 16), "C2": (8, 16), "C3": (1, 2, 4, 8)},
    "qwen3_next_80b": {"C1": (1, 4, 8, 16, 32), "C1b": (4, 8, 16), "C3": (1, 2, 4, 8)},
    "gemma4_26b_a4b_q4km_mtp": {"C1": (1, 8, 16, 32), "C3": (1, 2, 4, 8)},
}


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def parse_cpulist(spec: str) -> set[int] | None:
    """Parse kernel cpulist syntax ('0-23,96-119') into a set of CPU ids.

    Returns None on any syntax error (validator reports it as a violation).
    """
    if not isinstance(spec, str) or not spec.strip():
        return None
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        try:
            if "-" in part:
                lo_s, hi_s = part.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
                if hi < lo or lo < 0:
                    return None
                cpus.update(range(lo, hi + 1))
            else:
                val = int(part)
                if val < 0:
                    return None
                cpus.add(val)
        except ValueError:
            return None
    return cpus or None


def compute_ctx(np: int) -> int:
    """Protocol decision 1: -c = max(8192, per_stream_ctx * np)."""
    return max(CTX_FLOOR, PER_STREAM_CTX * np)


def _numactl_policy_for(cpu_list: str) -> str:
    # interleave=all is ONLY legal (and mandatory) on the full-machine shape
    # (gemma-MTP / dense-scout full); halves+quarters run taskset-only
    # first-touch locality matching stack_numa production wiring.
    return "interleave=all" if cpu_list == CPUSET_FULL else "none"


def _config_cpusets(model_key: str, config_id: str, full_variant: bool = False) -> list[str]:
    full_only = MODELS[model_key]["full_only"]
    if config_id == "C1":
        if full_only or full_variant:
            return [CPUSET_FULL]
        return [CPUSET_HALF0]
    if config_id == "C1b":
        return [CPUSET_HALF0, CPUSET_HALF1]
    if config_id == "C2":
        return [CPUSET_Q1A, CPUSET_Q1B]
    if config_id == "C3":
        return [CPUSET_Q0A, CPUSET_Q0B, CPUSET_Q1A, CPUSET_Q1B]
    raise ValueError(f"unknown config_id: {config_id}")


def _instances_for(model_key: str, config_id: str, full_variant: bool = False) -> list[dict]:
    port_base = MODELS[model_key]["port_base"]
    instances = []
    for idx, cpu_list in enumerate(_config_cpusets(model_key, config_id, full_variant)):
        cpus = parse_cpulist(cpu_list)
        assert cpus is not None
        instances.append(
            {
                "cpu_list": cpu_list,
                "port": port_base + idx,
                "threads": len(cpus),
                "numactl_policy": _numactl_policy_for(cpu_list),
            }
        )
    return instances


def _ubatch_for(model_key: str, instances: list[dict]) -> int:
    if model_key == "gemma4_26b_a4b_q4km_mtp":
        return UBATCH_48T  # MTP recipe override: -ub 512 on every shape
    if all(inst["threads"] == 96 for inst in instances):
        return UBATCH_96T
    return UBATCH_48T


def _stage_b_families(model_key: str, config_id: str, np: int) -> list[str]:
    """Iso-T / scaling / anchor pairing tags for the summarizer (additive field)."""
    fams: list[str] = []
    n_inst = CONFIG_INSTANCE_COUNT[config_id]
    total = n_inst * np
    gemma = MODELS[model_key]["full_only"]
    if config_id == "C1b" and total in (8, 16, 32):
        fams.append(f"whole_machine_T{total}")
    if config_id == "C3" and total in (8, 16, 32):
        fams.append(f"whole_machine_T{total}")
    if gemma and config_id == "C1" and np in (8, 16, 32):
        fams.append(f"whole_machine_T{np}")  # gemma pairs C1full@T vs C3@T/4
    if not gemma:
        if config_id == "C1" and np in (16, 32):
            fams.append(f"mechanism_T{np}")
        if config_id == "C2" and total in (16, 32):
            fams.append(f"mechanism_T{total}")
        if config_id in ("C1", "C1b") and np in (4, 8):
            fams.append(f"scaling_K{np}")
    if np == 1 and config_id in ("C1", "C3"):
        fams.append(f"anchor_{config_id}")
    return fams


# ---------------------------------------------------------------------------
# Cell construction
# ---------------------------------------------------------------------------


def make_cell(
    model_key: str,
    config_id: str,
    np: int,
    window: str,
    *,
    decision_grade_intent: bool,
    n_predict: int,
    kv_unified: bool = False,
    full_variant: bool = False,
    cell_id_suffix: str = "",
    extra_notes: str = "",
    stage_b_families: list[str] | None = None,
) -> dict:
    """Build one cell manifest in the frozen e5-cell-manifest/1 field order."""
    model = MODELS[model_key]
    instances = _instances_for(model_key, config_id, full_variant)
    notes = "; ".join(part for part in (model["notes"], extra_notes) if part)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "era": {
            "cpu_kernel": ERA_CPU_KERNEL,
            "eval_instrument": ERA_EVAL_INSTRUMENT,
            "source": ERA_SOURCE,
        },
        "window": window,
        "cell_id": f"{model_key}-{config_id}-np{np}{cell_id_suffix}",
        "model_key": model_key,
        "model_path": model["model_path"],
        "quant": model["quant"],
        "architecture": model["architecture"],
        "config_id": config_id,
        "instances": instances,
        "np": np,
        "per_stream_ctx": PER_STREAM_CTX,
        "ctx": compute_ctx(np),
        "ubatch_size": _ubatch_for(model_key, instances),
        "prompt_caps": {
            "n_predict": n_predict,
            "max_prompt_chars": MAX_PROMPT_CHARS,
            "max_total_in_flight": MAX_TOTAL_IN_FLIGHT,
        },
        "prompt_batch": {
            "source": QUESTION_POOL_PATH,
            "selection": "pinned_qids",
            "qids": list(E1_PINNED_QIDS),
            "pinned_from": E1_PINNED_FROM,
            "tier": 1,
            "seed": 42,
            "limit": 43,
        },
        "spec_dec": copy.deepcopy(model["spec_dec"]),
        "kv": {
            "type_k": "q8_0",
            "type_v": "q8_0",
            "flash_attn": True,
            "kv_unified": kv_unified,
        },
        "env_expectation": {
            "ggml_iqk": "1",
            "omp_source": "scripts/lib/canonical_recipe.CANONICAL_OMP_ENV",
            "kmp_blocktime": "10",
        },
        "mlock": True,
        "jinja": True,
        "warmup": {"prompts": WARMUP_PROMPTS, "n_predict": WARMUP_N_PREDICT},
        "decision_grade_intent": decision_grade_intent,
        "notes": notes,
        "stage_b_families": stage_b_families if stage_b_families is not None else [],
    }


def build_grid() -> list[dict]:
    """Emit the full pre-registered grid: W0 scout cells + Stage-B cells.

    W0 scout — all models, full K ladders, 64-token cap, decision_grade_intent
    false, plus the dense-control C1-shape scout pair (half0 vs full at K in
    {1,8}) and the qwen36 C1@16 split-vs--kvu paired probe.
    Stage-B — per-model decision-grade sets (W1 qwen36_q8_0, W2 gemma, W3
    dense control, W4 qwen3_next_80b), 256-token cap.
    """
    cells: list[dict] = []

    # --- W0 scout: full grid, non-decision-grade, 64-token cap -------------
    for model_key, model in MODELS.items():
        for config_id in model["configs"]:
            for np in K_LADDER[config_id]:
                cells.append(
                    make_cell(
                        model_key,
                        config_id,
                        np,
                        "W0",
                        decision_grade_intent=False,
                        n_predict=N_PREDICT_SCOUT,
                        cell_id_suffix="-scout",
                    )
                )

    # Scout special 1: dense-control C1 shape UNRESOLVED — paired full-machine
    # variant at K in {1,8} (vs the half0 cells already in the grid). The
    # full-machine rows double as E1 continuity anchors (direction-only: the
    # -c convention differs from E1).
    for np in (1, 8):
        cells.append(
            make_cell(
                "qwen36_27b_q8",
                "C1",
                np,
                "W0",
                decision_grade_intent=False,
                n_predict=N_PREDICT_SCOUT,
                full_variant=True,
                cell_id_suffix="-scout-full",
                extra_notes=(
                    "dense-control C1-shape scout pair, FULL-MACHINE variant "
                    "(pair with qwen36_27b_q8-C1-np%d-scout half0); Stage-B C1 "
                    "adopts the winner; doubles as E1 continuity anchor "
                    "(direction-only — different -c convention)" % np
                ),
                stage_b_families=["scout_dense_c1_shape_pair"],
            )
        )

    # Scout special 2: split-vs-unified KV paired probe at qwen36 C1@16.
    cells.append(
        make_cell(
            "qwen36_q8_0",
            "C1",
            16,
            "W0",
            decision_grade_intent=False,
            n_predict=N_PREDICT_SCOUT,
            kv_unified=True,
            cell_id_suffix="-scout-kvu",
            extra_notes=(
                "scout paired probe: unified KV (-kvu) vs the split-KV "
                "qwen36_q8_0-C1-np16-scout cell (protocol decision 1); >=5% "
                "delta escalates a Stage-B split-vs-unified arm to the operator"
            ),
            stage_b_families=["scout_kvu_probe"],
        )
    )

    # --- Stage-B decision-grade sets ---------------------------------------
    for model_key, k_sets in STAGE_B_K.items():
        window = MODELS[model_key]["stage_b_window"]
        for config_id, np_levels in k_sets.items():
            for np in np_levels:
                extra = ""
                if model_key == "qwen36_27b_q8" and config_id == "C1":
                    extra = (
                        "C1 shape PROVISIONAL (half0): dense-control C1 shape "
                        "unresolved — the W0 scout pair (half0 vs full-machine "
                        "at K in {1,8}) adopts the winner; regenerate the W3 C1 "
                        "cells on the full-machine shape if it wins"
                    )
                cells.append(
                    make_cell(
                        model_key,
                        config_id,
                        np,
                        window,
                        decision_grade_intent=True,
                        n_predict=N_PREDICT_STAGE_B,
                        extra_notes=extra,
                        stage_b_families=_stage_b_families(model_key, config_id, np),
                    )
                )

    return cells


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_bool(value) -> bool:
    return isinstance(value, bool)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_cell_manifest(manifest: dict) -> list[str]:
    """Validate one cell manifest; return human-readable violations ([] = valid).

    Fails closed: unknown schema_version, out-of-range ports, overlapping
    cpusets, thread/cpuset mismatches, K-cap or ctx-rule violations, unpinned
    prompt batches, SSM+kv_unified, silent spec-dec-off, scout/decision-grade
    contradictions are all refused.
    """
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest is not a JSON object"]

    def err(msg: str) -> None:
        errors.append(msg)

    # -- schema / protocol / era (fail closed on drift) ---------------------
    sv = manifest.get("schema_version")
    if sv != SCHEMA_VERSION:
        return [
            f"schema_version {sv!r} is not {SCHEMA_VERSION!r} — refusing to "
            f"interpret an unknown schema (fail closed on drift)"
        ]
    if manifest.get("protocol_id") != PROTOCOL_ID:
        err(
            f"protocol_id {manifest.get('protocol_id')!r} is not {PROTOCOL_ID!r} "
            f"(the waypoint blesses P-BENCH-3 reuse only)"
        )
    era = manifest.get("era")
    if not isinstance(era, dict):
        err("era block missing or not an object")
    else:
        if era.get("cpu_kernel") != ERA_CPU_KERNEL:
            err(f"era.cpu_kernel {era.get('cpu_kernel')!r} is not {ERA_CPU_KERNEL!r}")
        if era.get("eval_instrument") != ERA_EVAL_INSTRUMENT:
            err(
                f"era.eval_instrument {era.get('eval_instrument')!r} is not "
                f"{ERA_EVAL_INSTRUMENT!r}"
            )
        if era.get("source") != ERA_SOURCE:
            err(f"era.source {era.get('source')!r} is not {ERA_SOURCE!r}")

    # -- window / identity ---------------------------------------------------
    window = manifest.get("window")
    if window not in VALID_WINDOWS:
        err(f"window {window!r} is not one of {VALID_WINDOWS}")
    model_key = manifest.get("model_key")
    if not isinstance(model_key, str) or not model_key:
        err("model_key missing or empty (results index by model+quant, never role)")
    model_path = manifest.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        err("model_path missing or empty")
    if not isinstance(manifest.get("quant"), str) or not manifest.get("quant"):
        err("quant missing or empty")
    architecture = manifest.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        err("architecture missing or empty")
    config_id = manifest.get("config_id")
    if config_id not in VALID_CONFIG_IDS:
        err(f"config_id {config_id!r} is not one of {VALID_CONFIG_IDS}")

    np = manifest.get("np")
    if not _is_int(np) or np < 1:
        err(f"np {np!r} is not a positive integer")

    cell_id = manifest.get("cell_id")
    if not isinstance(cell_id, str) or not cell_id:
        err("cell_id missing or empty")
    elif "/" in cell_id:
        err(f"cell_id {cell_id!r} contains '/' (used as a filename)")
    elif isinstance(model_key, str) and config_id in VALID_CONFIG_IDS and _is_int(np):
        prefix = f"{model_key}-{config_id}-np{np}"
        rest = cell_id[len(prefix):] if cell_id.startswith(prefix) else None
        if rest is None or (rest and not rest.startswith("-")):
            err(
                f"cell_id {cell_id!r} does not match "
                f"'{{model_key}}-{{config_id}}-np{{np}}[-suffix]' (expected "
                f"prefix {prefix!r})"
            )

    # -- instances -----------------------------------------------------------
    instances = manifest.get("instances")
    if not isinstance(instances, list) or not instances:
        err("instances missing or empty")
        instances = []
    if config_id in CONFIG_INSTANCE_COUNT and instances:
        expected_n = CONFIG_INSTANCE_COUNT[config_id]
        if len(instances) != expected_n:
            err(
                f"config {config_id} requires exactly {expected_n} instance(s), "
                f"got {len(instances)}"
            )
    seen_cpusets: list[tuple[int, str, set[int]]] = []
    seen_ports: dict[int, int] = {}
    for idx, inst in enumerate(instances):
        if not isinstance(inst, dict):
            err(f"instances[{idx}] is not an object")
            continue
        cpu_list = inst.get("cpu_list")
        cpus = parse_cpulist(cpu_list) if isinstance(cpu_list, str) else None
        if cpus is None:
            err(f"instances[{idx}].cpu_list {cpu_list!r} is not valid kernel cpulist syntax")
        else:
            for prev_idx, prev_list, prev_cpus in seen_cpusets:
                overlap = cpus & prev_cpus
                if overlap:
                    err(
                        f"instances[{idx}].cpu_list {cpu_list!r} overlaps "
                        f"instances[{prev_idx}].cpu_list {prev_list!r} "
                        f"(shared CPUs e.g. {sorted(overlap)[:4]}) — cell cpusets "
                        f"must be pairwise disjoint"
                    )
            seen_cpusets.append((idx, cpu_list, cpus))
        threads = inst.get("threads")
        if not _is_int(threads) or threads < 1:
            err(f"instances[{idx}].threads {threads!r} is not a positive integer")
        elif cpus is not None and threads != len(cpus):
            err(
                f"instances[{idx}].threads {threads} != cpuset cardinality "
                f"{len(cpus)} for cpu_list {cpu_list!r}"
            )
        port = inst.get("port")
        if not _is_int(port):
            err(f"instances[{idx}].port {port!r} is not an integer")
        else:
            if not (PORT_MIN <= port <= PORT_MAX):
                err(
                    f"instances[{idx}].port {port} outside the E5 bench range "
                    f"[{PORT_MIN},{PORT_MAX}] — REFUSED (prod serves on "
                    f"8070-8485, 18070 eval lane, 8000 API, 8090-8095 embedders)"
                )
            if port in seen_ports:
                err(
                    f"instances[{idx}].port {port} collides with "
                    f"instances[{seen_ports[port]}].port within the cell"
                )
            else:
                seen_ports[port] = idx
        policy = inst.get("numactl_policy")
        if policy not in VALID_NUMACTL_POLICIES:
            err(
                f"instances[{idx}].numactl_policy {policy!r} is not one of "
                f"{VALID_NUMACTL_POLICIES}"
            )
        elif policy == "interleave=all" and cpu_list != CPUSET_FULL:
            err(
                f"instances[{idx}]: numactl_policy 'interleave=all' is ONLY "
                f"legal on the full-machine shape cpu_list='{CPUSET_FULL}' "
                f"(got {cpu_list!r}) — halves/quarters run taskset-only "
                f"first-touch locality"
            )
        elif policy == "none" and cpu_list == CPUSET_FULL:
            err(
                f"instances[{idx}]: full-machine shape cpu_list='{CPUSET_FULL}' "
                f"requires numactl_policy 'interleave=all' (canonical "
                f"full-machine recipe / gemma-MTP shape)"
            )

    # -- K caps / in-flight --------------------------------------------------
    prompt_caps = manifest.get("prompt_caps")
    if not isinstance(prompt_caps, dict):
        err("prompt_caps missing or not an object")
        prompt_caps = {}
    max_in_flight = prompt_caps.get("max_total_in_flight")
    if max_in_flight != MAX_TOTAL_IN_FLIGHT:
        err(
            f"prompt_caps.max_total_in_flight {max_in_flight!r} != "
            f"{MAX_TOTAL_IN_FLIGHT} (the fixed P-BENCH-3 batch size)"
        )
    if config_id in K_LADDER and _is_int(np) and np not in K_LADDER[config_id]:
        err(
            f"np {np} is not in the pre-registered {config_id} ladder "
            f"{K_LADDER[config_id]}"
        )
    if _is_int(np) and instances:
        total = len(instances) * np
        if total > MAX_TOTAL_IN_FLIGHT:
            err(
                f"total in-flight len(instances)*np = {len(instances)}*{np} = "
                f"{total} exceeds the cap {MAX_TOTAL_IN_FLIGHT}"
            )

    # -- ctx sizing rule -----------------------------------------------------
    per_stream_ctx = manifest.get("per_stream_ctx")
    if per_stream_ctx != PER_STREAM_CTX:
        err(
            f"per_stream_ctx {per_stream_ctx!r} != {PER_STREAM_CTX} (the "
            f"verified sizing constant, protocol decision 1)"
        )
    ctx = manifest.get("ctx")
    if _is_int(np):
        expected_ctx = compute_ctx(np)
        if ctx != expected_ctx:
            err(
                f"ctx {ctx!r} != max({CTX_FLOOR}, {PER_STREAM_CTX}*{np}) = "
                f"{expected_ctx} (KV budget scales with K, floor {CTX_FLOOR})"
            )

    ubatch = manifest.get("ubatch_size")
    if not _is_int(ubatch) or ubatch < 1:
        err(f"ubatch_size {ubatch!r} is not a positive integer")

    # -- prompt caps / batch pinning -----------------------------------------
    n_predict = prompt_caps.get("n_predict")
    if n_predict not in (N_PREDICT_SCOUT, N_PREDICT_STAGE_B):
        err(
            f"prompt_caps.n_predict {n_predict!r} is not {N_PREDICT_STAGE_B} "
            f"(Stage-B work parity) or {N_PREDICT_SCOUT} (W0 scout)"
        )
    max_prompt_chars = prompt_caps.get("max_prompt_chars")
    if not _is_int(max_prompt_chars) or not (0 < max_prompt_chars <= MAX_PROMPT_CHARS):
        err(
            f"prompt_caps.max_prompt_chars {max_prompt_chars!r} must be in "
            f"(0,{MAX_PROMPT_CHARS}] — fail-closed guard against qid-pinning "
            f"bypass (rebuilt pool holds a 101,655-char prompt)"
        )

    batch = manifest.get("prompt_batch")
    if not isinstance(batch, dict):
        err("prompt_batch missing or not an object")
        batch = {}
    if batch.get("selection") != "pinned_qids":
        err(
            f"prompt_batch.selection {batch.get('selection')!r} != 'pinned_qids' "
            f"— re-sampling tier/seed from the current pool is FORBIDDEN (pool "
            f"rebuilt 2026-07-21; would select a ~25k-token prompt)"
        )
    qids = batch.get("qids")
    if qids != E1_PINNED_QIDS:
        err(
            "prompt_batch.qids is not exactly the 43-qid E1 pinned batch "
            f"(e1-pbench3-clean-20260703T1912Z selection; got "
            f"{len(qids) if isinstance(qids, list) else type(qids).__name__} entries)"
        )
    if not isinstance(batch.get("source"), str) or not batch.get("source"):
        err("prompt_batch.source missing or empty")
    if not isinstance(batch.get("pinned_from"), str) or not batch.get("pinned_from"):
        err("prompt_batch.pinned_from missing or empty")
    for field, expected in (("tier", 1), ("seed", 42), ("limit", 43)):
        if batch.get(field) != expected:
            err(f"prompt_batch.{field} {batch.get(field)!r} != {expected} (provenance record)")

    # -- spec-dec ------------------------------------------------------------
    spec = manifest.get("spec_dec")
    if not isinstance(spec, dict):
        err("spec_dec missing or not an object")
        spec = {}
    enabled = spec.get("enabled")
    if not _is_bool(enabled):
        err(f"spec_dec.enabled {enabled!r} is not a boolean")
    if spec.get("device_draft") != "none":
        err(
            f"spec_dec.device_draft {spec.get('device_draft')!r} != 'none' — "
            f"the v7 HIP-capable binary would auto-offload draft work to the "
            f"MI210 and contaminate the CPU bench"
        )
    if enabled is True:
        if not isinstance(spec.get("spec_type"), str) or not spec.get("spec_type"):
            err("spec_dec.spec_type missing/empty while spec_dec.enabled is true")
        if spec.get("record_accept_rate") is not True:
            err(
                "spec_dec.record_accept_rate must be true when spec-dec is "
                "enabled (protocol decision 2: per-cell accept rates)"
            )
        if spec.get("disabled_reason") is not None:
            err("spec_dec.disabled_reason must be null when spec_dec.enabled is true")
    elif enabled is False:
        reason = spec.get("disabled_reason")
        if not isinstance(reason, str) or not reason.strip():
            err(
                "spec_dec.enabled false without a non-empty disabled_reason — "
                "spec-off arms must be explicitly labeled, never silent "
                "(operator directive 2026-07-23)"
            )

    # -- kv ------------------------------------------------------------------
    kv = manifest.get("kv")
    if not isinstance(kv, dict):
        err("kv missing or not an object")
        kv = {}
    if kv.get("type_k") != "q8_0" or kv.get("type_v") != "q8_0":
        err(
            f"kv.type_k/type_v ({kv.get('type_k')!r}/{kv.get('type_v')!r}) must "
            f"both be 'q8_0' (protocol decision 1)"
        )
    if kv.get("flash_attn") is not True:
        err("kv.flash_attn must be true (-fa on in every cell)")
    kv_unified = kv.get("kv_unified")
    if not _is_bool(kv_unified):
        err(f"kv.kv_unified {kv_unified!r} is not a boolean (never assumed — per-cell field)")
    elif kv_unified and isinstance(architecture, str) and architecture in SSM_HYBRID_ARCHITECTURES:
        err(
            f"kv.kv_unified true is REFUSED for SSM/hybrid architecture "
            f"{architecture!r} (tree-spec Phase-8 scar: hybrid+kv_unified "
            f"allocator/acceptance failures)"
        )

    # -- env expectation (audit C1) ------------------------------------------
    env_exp = manifest.get("env_expectation")
    if not isinstance(env_exp, dict):
        err("env_expectation missing or not an object")
        env_exp = {}
    if env_exp.get("ggml_iqk") != "1":
        err(
            f"env_expectation.ggml_iqk {env_exp.get('ggml_iqk')!r} != '1' "
            f"(audit C1, execution-blocking: the v7 iqk runtime gate)"
        )
    if env_exp.get("omp_source") != "scripts/lib/canonical_recipe.CANONICAL_OMP_ENV":
        err(
            f"env_expectation.omp_source {env_exp.get('omp_source')!r} must be "
            f"'scripts/lib/canonical_recipe.CANONICAL_OMP_ENV' (no private env copies)"
        )
    if env_exp.get("kmp_blocktime") != "10":
        err(
            f"env_expectation.kmp_blocktime {env_exp.get('kmp_blocktime')!r} != "
            f"'10' (E1 idle-spin fix, feedback_ik_llamacpp_omp_idle_spin)"
        )

    # -- E1 parity flags / warmup -------------------------------------------
    if manifest.get("mlock") is not True:
        err("mlock must be true (E1 parity flag --mlock)")
    if manifest.get("jinja") is not True:
        err("jinja must be true (E1 parity flag --jinja)")
    warmup = manifest.get("warmup")
    if not isinstance(warmup, dict):
        err("warmup missing or not an object")
    else:
        if not _is_int(warmup.get("prompts")) or warmup.get("prompts") < 1:
            err(f"warmup.prompts {warmup.get('prompts')!r} must be an integer >= 1")
        if not _is_int(warmup.get("n_predict")) or warmup.get("n_predict") < 1:
            err(f"warmup.n_predict {warmup.get('n_predict')!r} must be an integer >= 1")

    # -- decision-grade coherence -------------------------------------------
    dgi = manifest.get("decision_grade_intent")
    if not _is_bool(dgi):
        err(f"decision_grade_intent {dgi!r} is not a boolean")
    else:
        if window == "W0" and dgi:
            err("W0 scout cells MUST set decision_grade_intent=false")
        if dgi and n_predict != N_PREDICT_STAGE_B:
            err(
                f"decision_grade_intent=true requires prompt_caps.n_predict="
                f"{N_PREDICT_STAGE_B} (work parity); scout caps are never "
                f"decision-grade"
            )

    if "notes" in manifest and not isinstance(manifest.get("notes"), str):
        err("notes must be a string when present")

    # -- additive optional fields (typed when present) ----------------------
    fams = manifest.get("stage_b_families")
    if fams is not None and (
        not isinstance(fams, list) or any(not isinstance(f, str) for f in fams)
    ):
        err("stage_b_families must be a list of strings when present")
    for flag in ("model_path_present", "draft_model_path_present"):
        if flag in manifest and not _is_bool(manifest[flag]):
            err(f"{flag} must be a boolean when present")

    return errors


# ---------------------------------------------------------------------------
# File-presence flagging (generation-time; absent files are flagged, not fatal)
# ---------------------------------------------------------------------------


def annotate_file_presence(cells: list[dict]) -> None:
    """Stamp model_path_present / draft_model_path_present on every cell.

    Read-only stat; a missing file appends a note and sets the flag false —
    it never fails generation (the operator resolves paths at run time).
    """
    presence_cache: dict[str, bool] = {}

    def present(path: str) -> bool:
        if path not in presence_cache:
            presence_cache[path] = Path(path).is_file()
        return presence_cache[path]

    for cell in cells:
        ok = present(cell["model_path"])
        cell["model_path_present"] = ok
        if not ok:
            cell["notes"] = "; ".join(
                part
                for part in (
                    cell.get("notes", ""),
                    f"WARNING: model file absent at generation time: {cell['model_path']}",
                )
                if part
            )
        draft = cell["spec_dec"].get("draft_model_path")
        if draft:
            dok = present(draft)
            cell["draft_model_path_present"] = dok
            if not dok:
                cell["notes"] = "; ".join(
                    part
                    for part in (
                        cell.get("notes", ""),
                        f"WARNING: draft model file absent at generation time: {draft}",
                    )
                    if part
                )


def _warn_if_era_stamps_stale() -> None:
    """Best-effort drift check of the pinned era ids against instrument_eras.yaml."""
    try:
        text = INSTRUMENT_ERAS_PATH.read_text()
    except OSError:
        return
    for era_id in (ERA_CPU_KERNEL, ERA_EVAL_INSTRUMENT):
        if era_id not in text:
            print(
                f"WARN: pinned era id {era_id!r} not found in "
                f"{INSTRUMENT_ERAS_PATH} — era registry may have moved past "
                f"this generator; re-verify before decision-grade cells.",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_generate(args: argparse.Namespace) -> int:
    cells = build_grid()
    if not args.skip_file_checks:
        annotate_file_presence(cells)
        _warn_if_era_stamps_stale()

    bad = 0
    for cell in cells:
        violations = validate_cell_manifest(cell)
        if violations:
            bad += 1
            print(f"GENERATOR BUG — {cell.get('cell_id')} failed self-validation:")
            for v in violations:
                print(f"  - {v}")
    if bad:
        print(f"REFUSING to write: {bad} generated cell(s) failed validation")
        return 1

    out_root = Path(args.output_dir)
    counts: dict[str, dict[str, int]] = {}
    for cell in cells:
        model_dir = out_root / cell["model_key"]
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{cell['cell_id']}.json"
        with path.open("w") as fh:
            json.dump(cell, fh, indent=2)
            fh.write("\n")
        per_model = counts.setdefault(cell["model_key"], {})
        per_model[cell["window"]] = per_model.get(cell["window"], 0) + 1

    total = 0
    for model_key in MODELS:
        per_model = counts.get(model_key, {})
        n = sum(per_model.values())
        total += n
        windows = ", ".join(f"{w}={per_model[w]}" for w in sorted(per_model))
        print(f"{model_key}: {n} cells ({windows})")
    print(f"wrote {total} cell manifests under {out_root}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    any_bad = False
    for raw_path in args.files:
        path = Path(raw_path)
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"FAIL {path}: unreadable or invalid JSON: {exc}")
            any_bad = True
            continue
        violations = validate_cell_manifest(manifest)
        if violations:
            any_bad = True
            print(f"FAIL {path}:")
            for v in violations:
                print(f"  - {v}")
        else:
            print(f"OK {path}")
    return 1 if any_bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="e5_cell_manifests",
        description="E5 cell-manifest schema owner + pre-registered grid generator.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser(
        "generate",
        help="Emit the full pre-registered W0 scout + Stage-B cell grid "
        "(one JSON per cell, grouped by model_key).",
    )
    pg.add_argument(
        "--output-dir",
        default=str(DEFAULT_MANIFEST_DIR),
        help=f"Destination root (default: {DEFAULT_MANIFEST_DIR})",
    )
    pg.add_argument(
        "--skip-file-checks",
        action="store_true",
        help="Skip model-file presence stamping and the era-stamp drift check "
        "(hermetic/test use).",
    )
    pg.set_defaults(func=_cmd_generate)

    pv = sub.add_parser("validate", help="Validate manifest file(s); exit 1 on any error.")
    pv.add_argument("files", nargs="+", help="Cell-manifest JSON file(s).")
    pv.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
