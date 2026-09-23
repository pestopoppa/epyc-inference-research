"""PRELIMINARY canonical recipe for DeepSeek-V4.1-Flash on the EPYC 9655 CPU path.

REPO PATH: epyc-inference-research/scripts/lib/deepseek_v41_flash_recipe.py

★★ THIS RECIPE IS **PRELIMINARY** AND **SPEC-DEC-INCOMPLETE**. Read §0 before
   quoting anything out of it. ``STATUS`` and ``SPEC_DEC`` are the two fields that
   say so in machine-readable form, and ``preflight()`` prints both loudly.

WHY THIS FILE EXISTS
--------------------
Same reason as ``qwen38_flash_next_recipe.py``, which is this module's structural
template: a recipe carried in handoff prose gets transcribed wrong. The recipe is
DATA — import the constants, do not retype them, and do not read them out of
``handoffs/active/deepseek-v41-flash-evaluation.md``.

The companion objects, and the boundary between them:

  ``scripts/lib/canonical_recipe.py``      the GLOBAL llama-bench baseline recipe
                                           (taskset/numactl prefix, OMP stack,
                                           GGML_IQK, ``-fa 1``, ``-mmp 0``,
                                           pre-evict, placement proof). It is
                                           ``-t 96`` and MODEL-AGNOSTIC.
  THIS MODULE                              the PER-MODEL overlay for
                                           DeepSeek-V4.1-Flash: thread counts,
                                           artifact identity, binary identity,
                                           and the spec-dec fields.

Everything this module does not override is inherited from ``canonical_recipe``
UNCHANGED, and ``assert_inherits_canonical()`` proves that at runtime rather than
by assertion in a comment.

OPERATOR DIRECTIVE (2026-09-23), verbatim, and it is the reason THREADS != 96:

  "use 48 threads as the preliminary canonical recipe for this model — decode is
   more important than prefill"

  "MAKE SURE we're using spec decode as part of the recipe. I DO NOT CARE ABOUT
   BASELINE, ONLY MAX PERFORMANCE."

The first directive IS satisfied by this module (§5). **The second is NOT, and
cannot be today** — see §0 and §6. That gap is recorded as explicit ``pending``
values, never as absent keys, because a missing key reads as "not applicable"
and this one is "not yet built".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Optional

_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

import canonical_recipe as CR  # noqa: E402  (the global recipe this one overlays)


RECIPE_ID = "deepseek-v41-flash-cpu"
RECIPE_REVISION = "2026-09-23b"   # DSpark drafter landed; see §5
SURFACE = "cpu"          # decode AND prefill; see THREADS / THREADS_BATCH


class RecipeViolation(AssertionError):
    """Raised when a command/env does not match the codified recipe."""


# ---------------------------------------------------------------------------
# 0. ★★ STATUS — PRELIMINARY, AND SPEC-DEC INCOMPLETE
# ---------------------------------------------------------------------------
# Two separate limitations. Neither is a caveat you may drop when quoting a
# number out of this module (MEASUREMENT_POLICY: caveat placement must not be
# inversely correlated with caveat severity).
STATUS = {
    "grade": "PRELIMINARY",
    "why_preliminary": (
        "The thread choice rests on ONE thread sweep taken on 2026-09-23 "
        "(DS41-T6b/T6c) on a port that is itself experimental and unpromoted. No "
        "protocol-cited repeat, no cross-window replication, no serving-class "
        "number of any kind. Every entry in MEASURED below is instrument_class "
        "'bench' and therefore INADMISSIBLE as a headline or a rate "
        "(MEASUREMENT.md INSTRUMENT-CLASS-1)."
    ),
    "why_spec_dec_incomplete": (
        "SUPERSEDED 2026-09-23b. The DSpark drafter now EXISTS and is measured "
        "(DeepSeek-V4.1-Flash-DSpark.gguf, --spec-type draft-dspark, patches 01-06 "
        "+ 10 on the port worktree). What is incomplete is narrower and is now its "
        "own field: GREEDY_EXACTNESS. At temp > 0 the drafter is a 1.24-1.27x win "
        "(block 2: 10.48 vs 8.28 t/s control). At temp <= 0 the server forces the "
        "SERIAL verification path, which decodes one token per target decode and "
        "therefore CANNOT exceed 1x -- it measures 6.01-7.61 t/s against an 8.21 "
        "t/s no-drafter control, i.e. speculation makes greedy SLOWER. The 1.27x "
        "exists only at temp > 0 today. Whether greedy may take the batched path "
        "is DS41-T5/C1 and is UNMEASURED."
    ),
    # ★ The category this recipe's numbers belong to, per MEASUREMENT_POLICY.
    # It is deliberately NOT 'OPTIMUM': the policy's own rule is that the
    # unaccelerated run is the OPTIMUM only when NO draft path exists for the
    # model. A draft path DOES exist for DeepSeek-V4.1-Flash (DSpark is in the
    # official checkpoint); we have merely not built it. Calling this OPTIMUM
    # would claim a ceiling the model does not have.
    "measurement_category": "CANDIDATE",
    "measurement_category_note": (
        "NOT 'OPTIMUM'. A draft path exists for this model (DSpark ships in the "
        "official checkpoint); ours does not implement it. The unaccelerated-run-"
        "is-the-optimum rule applies only where no draft path exists at all."
    ),
    "instrument_class": "bench",   # for MEASURED; MEASURED_SERVING is 'serving'
    "supersedes": None,
    "superseded_by": None,
    "record": "handoffs/active/deepseek-v41-flash-evaluation.md (DS41-T6, DS41-B13, DS41-T5/C1)",
}


# ---------------------------------------------------------------------------
# 1. ARTIFACT IDENTITY
# ---------------------------------------------------------------------------
# The joined Q4 serving artifact. `.part1` + `.part2` were published separately
# and MUST be joined: blk.14.engram_embd straddles the boundary.
TRUNK_GGUF = (
    "/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/"
    "DeepSeek-V4.1-Flash-Q4.gguf"
)
TRUNK_BYTES = 518_596_067_328          # 482.98 GiB, verified DS41-A2 2026-09-23
TRUNK_ARCH = "deepseek41"              # general.architecture
TRUNK_TENSORS = 1046                   # blk.0-39 only
TRUNK_GGUF_VERSION = 3
TRUNK_ALIGNMENT = 16384

# ★ THE TENSOR THE ARTIFACT DOES NOT HAVE. Zero `mtp.*` tensors: antirez's
# converter drops every mtp.*/vision.*/aligner.* tensor
# (gguf-tools/deepseek41_quantize.py:149), and vcruz's GGUF strips them too. The
# earlier "MTP retained" reading came from the embedded HF-config KV string,
# which antirez writes and never reads. This is why the DRAFTER IS A SEPARATE
# FILE (below) rather than a block inside the trunk GGUF.
TRUNK_MTP_TENSORS = 0

# ★ THE DRAFTER, 2026-09-23b. Built from the official shards 44-46 (the 2,401
#   mtp.* tensors the trunk artifact does not carry), NOT from a re-download.
#   It is a SEPARATE GGUF passed with -md; --spec-type draft-dspark selects the
#   DSpark accept/verify shape (anchor-first block layout + Markov head), which
#   is NOT the same code path as draft-mtp (sequential nextn depth).
DRAFT_GGUF = "/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash-DSpark.gguf"
DRAFT_BYTES = 10_004_008_256           # 9.32 GiB, on disk 2026-09-23
DRAFT_SPEC_TYPE = "draft-dspark"

TRUNK_PART_SHA256 = {
    # As published; both verified against their published digests, DS41-A2.
    "part1": "6442b1f9",   # short form as the handoff records it
    "part2": "7c3e1064",
}
TRUNK_PART2_RETAINED = True   # 39 GB, deletable once the join is trusted


# ---------------------------------------------------------------------------
# 2. BINARY / KERNEL IDENTITY  —  NOT PRODUCTION, AND NOT PROMOTABLE YET
# ---------------------------------------------------------------------------
# ★★ DO NOT resolve this model through canonical_recipe.discover_canonical_bench_binary().
# That function resolves the PRODUCTION kernel store
# (/mnt/raid0/llm/kernels/production/cpu). Production does not know the
# `deepseek41` architecture at all: the loader, the KV-key adapter, the
# compress-ratio/RoPE deltas and the DSpark graph live ONLY on the experimental
# branch below. A production binary will refuse the GGUF, not measure it slowly.
#
# So this model is an EXPLICIT-IDENTITY arm in canonical_recipe's sense: it must
# pass --binary / --source-root / --library-path together, which routes it
# through assert_explicit_bench_identity() and pins the candidate's own library
# directory first in LD_LIBRARY_PATH. That is the correct mechanism, not a
# workaround: three ggml generations live on this host.
BINARY_STATUS = "EXPERIMENTAL — unpromoted; revisit at every promotion"
KERNEL_BRANCH = "experimental/deepseek41-port-20260923"
KERNEL_COMMIT = "7c18bb8c1"
# ★ NOT A CLEAN TREE. The measured DSpark numbers below were taken on this branch
#   PLUS uncommitted DSpark patches 01-06 + 10 in the worktree. A commit hash is
#   therefore NOT a sufficient identity for those numbers; the working-tree diff
#   is part of the identity until the patches land.
KERNEL_WORKTREE_PATCHES = "DSpark 01-06 + 10, uncommitted in the port worktree (2026-09-23)"
KERNEL_BUILD_NUMBER = 10303
KERNEL_SOURCE_ROOT = "/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923"
# CORRECTED 2026-09-23b: the tree has no `build/`; the CPU build is `build-cpu`.
# assert_binary_exists() would have caught this on the first real call.
KERNEL_BINDIR = os.path.join(KERNEL_SOURCE_ROOT, "build-cpu", "bin")
KERNEL_BENCH = os.path.join(KERNEL_BINDIR, "llama-bench")
KERNEL_SERVER = os.path.join(KERNEL_BINDIR, "llama-server")

# ⚠ Build 10303 is ALSO the build number of production-consolidated-v10 and of
# the INF-70 headline instrument. The build number is NOT an identity here;
# the (branch, commit, bindir) triple is. Never select this binary by build
# number alone.
KERNEL_BUILD_NUMBER_IS_NOT_AN_IDENTITY = (
    "build 10303 collides with production v10 (ffc1bac82) and with the INF-70 "
    "headline instrument (2516c9807). Select by source_root+commit, never by "
    "build number."
)

# ★ WHAT FLIPS THIS BLOCK. When the deepseek41 port is promoted into a
# production kernel (v11 or later), delete KERNEL_* and set
# USE_PRODUCTION_DISCOVERY = True; canonical_recipe's store-resolved,
# fail-closed discovery then applies unchanged.
USE_PRODUCTION_DISCOVERY = False


# ---------------------------------------------------------------------------
# 3. INHERITED FROM canonical_recipe — NOT RESTATED, NOT FORKED
# ---------------------------------------------------------------------------
# Everything here is a REFERENCE to the global recipe, so a change there reaches
# this model automatically and assert_inherits_canonical() catches a fork.
PREFIX = CR.CANONICAL_PREFIX                     # taskset -c 0-95 numactl --interleave=all
OMP_ENV = CR.CANONICAL_OMP_ENV                   # incl. GGML_IQK=1
PRE_EVICT_GIB = CR.CANONICAL_PRE_EVICT_GIB       # 40
MAX_NODE_SHARE_PCT = CR.CANONICAL_MAX_NODE_SHARE_PCT   # 40
LLVM20_LIBDIR = CR.LLVM20_LIBDIR

# -fa 1 and -mmp 0 come from CR.CANONICAL_BENCH_FLAGS_LLAMA_BENCH. The ONLY
# element of that list this recipe overrides is the thread count (§5).
INHERITED_BENCH_FLAGS = list(CR.CANONICAL_BENCH_FLAGS_LLAMA_BENCH)   # -t 96 -fa 1 -mmp 0

# ★ Placement proof is REQUIRED, not optional, and on this artifact it is the
# gate that has actually been failing: 483 GiB cannot be shown to be evenly
# placed by inspecting the command line. bench_canonical.sh enforces it.
# 2026-09-23 (DS41-T6c): the in-window sampler never fired on five consecutive
# runs of this model — see the DS41-T6c note in bench_canonical.sh. A run whose
# placement proof is missing is an OBSERVATION and must not be quoted here.
PLACEMENT_PROOF_REQUIRED = True
PLACEMENT_PROOF_TOOL = "scripts/benchmark/bench_canonical.sh (writes placement.log + .rc)"


# ---------------------------------------------------------------------------
# 4. ★ THE THREAD SPLIT, AND WHICH TOOLS CAN EXPRESS IT
# ---------------------------------------------------------------------------
# Decode and prefill do NOT want the same thread count on this model, and the
# operator ruled which one wins. Two numbers — and exactly ONE of our four tools
# can express the difference between them. Verified in the source, not assumed:
#
#   llama-server / llama-cli      CAN. `-t N` sets decode threads, `-tb N` /
#   (direct invocation only)      `--threads-batch N` sets batch and prompt-
#                                 processing threads (common/arg.cpp:1337;
#                                 defaults to --threads when omitted).
#                                 build_serve_command() below emits BOTH.
#
#   llama-bench                   CANNOT. tools/llama-bench/llama-bench.cpp
#                                 parses only `-t/--threads` (:792) and calls
#                                 llama_set_n_threads(ctx, n, n) (:2383, :2419)
#                                 — batch threads are FORCED equal by the
#                                 instrument. A `-t 48` bench therefore measures
#                                 pp512 at 48 threads too: that is the 137.0 t/s
#                                 row below, NOT the 144.0-145.8 t/s the model
#                                 does at 96. Never read a bench pp number as
#                                 this recipe's served prefill rate.
#
#   canonical_recipe.py /         CANNOT, because they build llama-bench
#   bench_canonical.sh            commands. build_bench_command() overrides only
#                                 the `-t` value and says so; nothing else moves.
#
#   autokernel serving recipes    CANNOT, AND REFUSES. serving.Recipe.server_argv
#   (artifacts/serving-recipes/   emits `-t <threads> -tb <threads>` — always
#    *.json, schema                EQUAL (serving.py:507) — and
#    epyc.autokernel.canonical_    resolved_recipe.py:600 raises
#    recipe.v1)                    ResolutionError("canonical -tb differs from
#                                  -t") on any command that splits them. So the
#                                  JSON schema has ONE `threads` field by
#                                  design. A serving-recipe JSON for this model
#                                  must carry threads=48 and thereby accept the
#                                  ~5% pp512 cost; it CANNOT encode -tb 96.
#
# This is the same unit-of-work discipline MEASUREMENT_POLICY demands: "137.0
# pp512" and "the recipe prefills at 96 threads" are both true and describe
# different instruments. Name the instrument every time.
THREADS = 48                  # DECODE. The operator-directed preliminary canonical value.
THREADS_BATCH = 96            # PREFILL / batch. Server-side only (-tb); llama-bench ignores it.

THREADS_RATIONALE = (
    "Operator directive 2026-09-23: 'use 48 threads as the preliminary canonical "
    "recipe for this model — decode is more important than prefill'. The sweep "
    "supports it: decode is FLAT from 24 to 96 threads (12.89 / 13.18 / 12.74-12.81 "
    "t/s at 24 / 48 / 96) with 48 the best of them, and COLLAPSES at 192 (4.99 t/s, "
    "-62%). tg512 separates the same way and harder: 11.70 @48 vs 10.59 @96 (+10.5%). "
    "Prefill runs the other way — 137.0 @48, 137.7 @64, 144.0-145.8 @96, 104.1 @192 — "
    "so 48 costs about 5% of pp512 to buy about 10% of tg512. Decode wins by ruling."
)

THREADS_REJECTED = {
    96:  "prefill-optimal (144.0-145.8 pp512) but -3% tg128 and -9.5% tg512; the "
         "operator ruled decode over prefill. Kept as THREADS_BATCH.",
    64:  "pp512 137.7 — indistinguishable from 48 on prefill; no decode row measured. "
         "Not a candidate.",
    24:  "tg128 12.89 (-2.2% vs 48) and prefill unmeasured. Decode is flat here, not "
         "better.",
    192: "REFUSED. tg128 4.99 (-62%) and pp512 104.1 (-24%): 192 threads oversubscribes "
         "the 96 physical cores onto their SMT siblings. Not a tuning option.",
}


# ---------------------------------------------------------------------------
# 5. ★★ SPECULATIVE DECODING — EVERY FIELD EXISTS, EVERY FIELD IS `pending`
# ---------------------------------------------------------------------------
# ★ THE FIELDS BELOW ARE DELIBERATELY PRESENT AND DELIBERATELY UNSET.
#   An ABSENT spec-dec block reads as "this model has no draft path". This model
#   HAS one — DSpark ships in the official checkpoint — and we have not built it.
#   Those are different statements and this block exists to keep them different.
#
# The operator's requirement ("MAKE SURE we're using spec decode as part of the
# recipe... ONLY MAX PERFORMANCE") is therefore RECORDED AS UNSATISFIED rather
# than quietly dropped. build_serve_command() refuses by default unless the
# caller passes spec_dec=False and thereby states in the call site that it is
# knowingly launching the no-draft configuration.
SPEC_DEC_STATUS = "measured"   # 2026-09-23b: the drafter exists and is measured

SPEC_DEC = {
    "status": SPEC_DEC_STATUS,
    "required_by": "operator directive 2026-09-23 (MAX PERFORMANCE, spec decode in the recipe)",
    "satisfied": False,

    # --- what the drafter IS (from the reference implementation, not guessed) ---
    "drafter_name": "DSpark",
    "drafter_kind": "3-block draft transformer over a 5-token block, bidirectional in-block",
    "drafter_topology": (
        "own 128-expert/top-3 MoE, rank-256 Markov bias head, confidence head; "
        "embed/head TIED to the backbone; fed by the hc-mean of the attention "
        "input at layers 37-39"
    ),
    "drafter_reference": "inference/model.py:1032-1156 (official DeepSeek-V4.1-Flash repo)",
    "drafter_config_fields": "dspark_* (block 5, target layers 37-39, Markov rank 256, noise token 128799)",

    # --- LANDED 2026-09-23b ---
    "draft_gguf": DRAFT_GGUF,
    "draft_gguf_source": "official shards 44-46 (~8 GB, 2,401 mtp.* tensors)",
    "draft_gguf_download_state": "PRESENT on disk (converted 2026-09-23)",
    "draft_gguf_conversion": "done: shards 44-46 -> DeepSeek-V4.1-Flash-DSpark.gguf",

    "spec_type": "draft-dspark",           # --spec-type draft-dspark
    "spec_draft_n_max": 2,                 # the DSpark BLOCK size; see SPEC_DEC_VARIANTS
    "spec_draft_p_min": None,              # not swept; leave unset rather than guess
    "alpha": 0.516,                        # block 2 acceptance, temp 0.7. Block 3: 0.444.
    "drafted_per_token": None,             # not extracted yet; derive from draft_n/verif steps

    # --- why the flags above cannot simply be filled in ---
    "accept_verify_loop": "IMPLEMENTED (port patches 01-06 + 10). Two shapes: BATCHED "
                          "verification (temp > 0) and SERIAL verification (temp <= 0, "
                          "forced by use_serial_speculative_verify in "
                          "tools/server/server-context.cpp). See GREEDY_EXACTNESS.",
    "graph_mtp_blocker": (
        "our graph_mtp asserts n_layer_nextn == 1 and is the WRONG AXIS: it models "
        "sequential depth, DSpark is block-parallel. LLM_ARCH_DFLASH is the closer template."
    ),
    "rollback_blocker": (
        "DS41-B14: rollback here is VALUE-level, not a position rewind. The 128-slot "
        "window ring, the compressor's kv_state/score_state accumulators on the 4 "
        "kv-source layers, the compressed-KV/indexer row writes and the n-gram hash "
        "state are all destructively mutated. A naive llama_kv_cache_seq_rm port "
        "corrupts them SILENTLY. This is a precondition, not a follow-up."
    ),

    # --- ★ THE ORIGINAL FLIP CONDITION, retained as the record of what was required.
    #     Items 1-3 are DONE; 4 is partially done (alpha measured, p_min not swept);
    #     5 is OPEN and is the reason STATUS stays PRELIMINARY.
    "flips_on": [
        "1. DS41-B13a — shards 44-46 downloaded and converted to a draft GGUF; "
        "set draft_gguf + its bytes/sha256.",
        "2. DS41-B14 — value-level rollback implemented for the window ring, the "
        "compressor accumulators, the compressed-KV/indexer rows and the n-gram "
        "hash state; DS41-T4 (forced rejection at every draft position, rollback vs "
        "replay, full vs chunked prefill) PASSES.",
        "3. DS41-B13b — a block-parallel DSpark graph (NOT the n_layer_nextn==1 "
        "sequential-depth graph) plus the accept/verify loop; DS41-T5 exact-parity "
        "gates pass at each depth separately.",
        "4. A measured alpha and drafted-per-token at the production prompt mix, "
        "with a coherence gate at production prompt length; set alpha, "
        "drafted_per_token, spec_draft_n_max, spec_draft_p_min from THAT measurement "
        "and not from the GLM/Qwen precedents.",
        "5. Re-derive THREADS with the drafter ON. Verification is a BATCH step: a "
        "drafter changes the decode/prefill thread balance this recipe was tuned "
        "against, so -t 48 is NOT assumed to survive. Then set "
        "SPEC_DEC_STATUS = 'measured' and STATUS['grade'] = whatever the evidence "
        "supports.",
    ],
    "flips_on_state_20260923b": {
        "1_draft_gguf": "DONE",
        "2_rollback_and_T4": "DONE for the serial path by construction (it never decodes "
                             "a rejected token); the BATCHED path's value-level rollback "
                             "is exercised at temp > 0 but DS41-T4's forced-rejection "
                             "matrix has not been run. OPEN.",
        "3_block_parallel_graph_and_T5": "graph DONE; DS41-T5 exact-parity gates OPEN "
                                         "(this is DS41-T5/C1).",
        "4_alpha_and_knobs": "alpha DONE (0.516 @ block 2, 0.444 @ block 3, temp 0.7); "
                             "p_min and drafted_per_token OPEN.",
        "5_rederive_threads_with_drafter_on": "OPEN. -t 48 was tuned with NO drafter. "
                                              "Verification is a batch step, so the "
                                              "decode/prefill balance has moved and 48 is "
                                              "NOT assumed to survive.",
    },
    "owner": "handoffs/active/deepseek-v41-flash-evaluation.md DS41-B13 / DS41-B14 / DS41-T4 / DS41-T5",
}

# Things that are NOT this model's drafter. Recorded so the substitution is
# refused by name instead of being rediscovered.
SPEC_DEC_REJECTED = {
    "GLM-5.3-Flash-DFlash2 drafter": "deleted, and GLM-specific.",
    "Qwen3.8-Flash-Next MTP head": "different model and different mechanism "
                                   "(sequential nextn depth, not a block-parallel drafter).",
    "the antirez GGUF's embedded HF-config KV string": (
        "it advertises num_nextn_predict_layers=3 and antirez never reads it back. "
        "The tensor census is the authority: zero mtp.* tensors."
    ),
}


# ---------------------------------------------------------------------------
# 5b. ★★ GREEDY EXACTNESS — the field that decides which variant you may serve
# ---------------------------------------------------------------------------
# WHY THIS EXISTS AS A FIELD AT ALL. Upstream llama.cpp documents speculative
# decoding as output-identical at greedy. That claim is CONDITIONAL on the target's
# multi-token forward computing each row exactly as its single-token forward would
# -- batch invariance -- which is a property of the KERNELS, not of the algorithm.
# This fork measured the condition and found it false on every compute plane we
# have (wiki/speculative-decoding.md:1327, confidence verified): the N==1 vs N>1
# dispatch split exists in llamafile_sgemm's mnpack blocking, in iqk's funcs[ny-1]
# dispatch, and deliberately on gfx90a (commit a6b4b5263, whose own message says
# "numerically-valid (not bit-exact)", bought for +17.4% MTP on MI210).
#
# So: DO NOT re-derive "speculative decoding is exact at greedy" from upstream's
# documentation. On this stack it is not, and the two driver-side fixes that assume
# only the bonus row is unchecked (LLAMA_SPEC_EXACT=drop / =redecode) both FAILED
# their greedy-identity gate (2/3 and 3/3 FAIL, INF-70 E2a) because the divergence
# is in the VERIFIED rows too. The only exact configuration is not to batch.
GREEDY_EXACTNESS = {
    "upstream_claim": "speculative decoding is output-identical at greedy",
    "holds_here": False,
    "why": (
        "batch invariance is not a property any of our three compute planes holds; "
        "row i of a (k+1)-wide verification forward takes a different kernel, and "
        "therefore a different reduction order, from a 1-wide decode at the same "
        "position. Argmax flips wherever the top-1/top-2 margin is below that "
        "perturbation (measured onset margins 0.005-0.079, INF-70 / DF2-6)."
    ),
    "mechanism_record": "INF-70 E2a (handoffs/active/cpu-decode-roofline-program.md, "
                        "root cause src/models/delta-net-base.cpp:435); "
                        "wiki/speculative-decoding.md:1293-1327",
    "dspark_specific": (
        "tools/server/server-context.cpp:3843 -- quantized RECURRENT targets are not "
        "batch-invariant (llama.cpp issue #25618), and the compressor/window/indexer "
        "state is destructively accumulated, so a batched verify both perturbs the "
        "logits and writes state for tokens that may be rolled back (DS41-B14)."
    ),
    # ★ What is NOT yet known, and the only thing that can move the recommendation.
    "measured_here": False,
    "measurement": "DS41-T5/C1 -- /mnt/raid0/llm/tmp/ds41-exactness/ "
                   "(PROTOCOL.md, MECHANISM.md, collect_arm.py, parity_diff.py)",
    "prior": "GPU precedent DF2-6: serial arms bit-exact 12/12; batched arms 5/12 and "
             "6/12 diverged (12 prompts, temp 0, top_k 1, seed 42, 256 tokens). "
             "Expect a similar order of magnitude here, not zero.",
    "switch": "LLAMA_SPEC_EXACT=batched-greedy-inexact (server env; patch "
              "/mnt/raid0/llm/tmp/ds41-exactness/patches/01-spec-exact-batched-greedy.patch). "
              "Default OFF; any unrecognised value falls back to the EXACT serial path.",
    "cost_of_exactness": (
        "serial verification decodes one token per target decode and therefore cannot "
        "exceed 1x. Measured: 6.01-7.61 t/s greedy against an 8.21 t/s NO-DRAFTER "
        "control -- i.e. today, speculation makes greedy SLOWER, and the 1.24-1.27x "
        "exists only at temp > 0."
    ),
    "what_serial_actually_guarantees": (
        "agreement with the NO-DRAFTER path ON THE SAME BINARY. It does not survive a "
        "kernel change, a thread-count change, an ubatch change or toggling GGML_IQK, "
        "all of which also change reduction order. Do not sell it as reproducibility "
        "in general."
    ),
}


# ---------------------------------------------------------------------------
# 5c. ★★ SERVING VARIANTS — because exact and fastest are not the same config
# ---------------------------------------------------------------------------
# The measurement says they differ, so the recipe carries both rather than
# averaging them into one dishonest default. Each variant states its own status;
# build_serve_command() refuses any variant that is not 'measured'.
#
# All numbers below: instrument_class SERVING, protocol id NONE => OBSERVATIONS.
# Same server config, same prompt, -t 48, trunk Q4 + DSpark drafter. They are not
# decision-gating claims and must not be quoted as rates (MEASUREMENT_POLICY).
SPEC_DEC_VARIANTS = {
    "greedy-exact": {
        "status": "measured",
        "intent": "bit-exact greedy continuation, equal to the no-drafter path on this binary",
        "spec_draft_n_max": 3,
        "spec_exact_env": None,      # serial is what the server does by default here
        "verification_path": "SERIAL (forced by use_serial_speculative_verify: "
                             "seq_rm RS + temp<=0 + draft-dspark)",
        "greedy_tps": 7.61,          # vs 8.21 no-drafter control
        "temp07_tps": 10.26,
        "alpha": 0.444,
        "honest_note": (
            "★ AT GREEDY THIS IS A LOSS, NOT A WIN: 7.61 t/s against an 8.21 t/s "
            "no-drafter control (-7.3%). It buys exactness with throughput. If you "
            "want exact greedy AND max speed, serve NO DRAFTER at greedy (8.21) -- "
            "that is strictly better than this variant on both axes."
        ),
    },
    "max-throughput": {
        "status": "measured",
        "intent": "the operator's max-performance requirement, temp > 0 only",
        "spec_draft_n_max": 2,
        "spec_exact_env": None,
        "verification_path": "BATCHED (temp > 0 never takes the serial path)",
        "greedy_tps": None,          # ★ NOT MEASURED at block 2 greedy; do not infer
        "temp07_tps": 10.48,         # 1.266x over the 8.28 t/s control at temp 0.7
        "alpha": 0.516,
        "honest_note": (
            "The 1.27x is a temp > 0 number. At temp <= 0 this same launch config "
            "silently falls back to SERIAL and loses the win -- the variant name "
            "does not describe what a greedy request gets."
        ),
    },
    "greedy-batched": {
        "status": "measured",        # ★ DS41-T5 parity ran 2026-09-23; operator adopted it
        "intent": "let greedy take the batched path; the 1.27x at temp <= 0",
        "spec_draft_n_max": 2,
        "spec_exact_env": "batched-greedy-inexact",
        "verification_path": "BATCHED at every temperature",
        "greedy_tps": 17.85,         # median over the 4 parity prompts generating >=32 tok
        "temp07_tps": 10.48,
        "alpha": 0.516,
        "flipped_on": (
            "DS41-T5/C1 parity measurement: (a) serial == no-drafter 12/12 (the "
            "positive control), (b) batched-vs-no-drafter divergence confined to "
            "high-entropy prompts and never inside the first ~10 generated tokens, "
            "(c) the arithmetic-chain prompt still reaches a correct answer, (d) the "
            "1.2x+ gain reproduces at greedy. Fewer is not enough. If divergence "
            "lands on LOW-entropy or structured prompts, that is the signature of a "
            "state-rollback defect (DS41-B14), not of rounding -- keep serial and "
            "open a defect.\n\n"
            "MEASURED 2026-09-23, 12 prompts + a 3-prompt low-entropy re-run:\n"
            "  (a) PASS. serial == no-drafter 0/12 divergences -- the positive "
            "control holds, so the harness can detect sameness.\n"
            "  (b) PARTIAL. batched diverges on 5/12, onset 0.5721 per 100 tokens, "
            "every onset at token >=16. But the original p09/p10 low-entropy pair "
            "was VACUOUS -- both returned 1 token then EOS (empty content) in every "
            "arm, because they were phrased as instructions and this model ships no "
            "chat template. They proved nothing and must not be cited. The re-run "
            "with continuation-phrased prompts gives 1 of 3 diverging: the "
            "arithmetic chain forked at char 211 into a SYNONYM ('Total loss before "
            "minute 20' vs 'Total lost by minute 20') with every number preserved, "
            "while the markdown table and the counting sequence were identical over "
            "160 tokens. So divergence is NOT confined to high-entropy text; it "
            "tracks near-ties, and maximally-constrained tokens never move.\n"
            "  (c) PASS in substance. The arithmetic stayed correct across the fork.\n"
            "  (d) PASS, exceeded. 17.85 t/s median greedy vs 11.43 plain (1.56x); "
            "on low-entropy prompts 18.79-22.08 vs 12.37-12.59."
        ),
        "honest_note": (
            "Output under this variant is NOT bit-exact with the no-drafter path. It "
            "is a valid greedy sample of a faithful execution of the model, just not "
            "of the same execution. Never describe it as exact."
        ),
    },
}

# Operator decision 2026-09-23 ("of course take batched"): greedy takes the batched
# path. Output is a valid greedy sample of a faithful execution, but NOT bit-identical
# to the no-drafter path -- see the variant's honest_note. `greedy-exact` remains for
# anything that needs reproducibility.
SPEC_DEC_VARIANT_DEFAULT = "greedy-batched"

# Block 5 is recorded as MEASURED AND REJECTED so it is not rediscovered.
SPEC_DEC_BLOCK_REJECTED = {
    5: "REJECTED. 6.01/7.22 t/s greedy (serial path) and 8.24 at temp 0.7 (batched) "
       "-- at temp 0.7 it is BELOW the 8.28 no-drafter control. A wider block costs "
       "more per rejection than its extra acceptances are worth here.",
}


# ---------------------------------------------------------------------------
# 6. MEASURED — 2026-09-23, bench class, OBSERVATION-GRADE
# ---------------------------------------------------------------------------
# ⚠ instrument_class = 'bench'. These are llama-bench numbers on an experimental
# port. Per INSTRUMENT-CLASS-1 they are a valid build-vs-build A/B surface and
# are INADMISSIBLE as an absolute headline or a rate. There is no serving-class
# number for this model at all yet.
MEASURED = {
    "date": "2026-09-23",
    "record": "DS41-T6b / DS41-T6c",
    "instrument_class": "bench",
    "category": "CANDIDATE",
    "binary": f"{KERNEL_BRANCH} @ {KERNEL_COMMIT} (build {KERNEL_BUILD_NUMBER})",
    "artifact": TRUNK_GGUF,
    "spec_dec": "NONE — no drafter exists (see SPEC_DEC)",
    "protocol": None,   # ★ no protocol citation => OBSERVATION, not a decision-gating claim
    "tg128_tps_by_threads":  {24: 12.89, 48: 13.18, 96: (12.74, 12.81), 192: 4.99},
    "tg512_tps_by_threads":  {48: 11.70, 96: 10.59},
    "pp512_tps_by_threads":  {48: 137.0, 64: 137.7, 96: (144.0, 145.8), 192: 104.1},
    "placement": "EVEN — 25.0% per node, independent check PASS",
    "placement_note": (
        "captured only after the bench_canonical.sh sampler defect (DS41-T6c) was "
        "worked around; the five earlier canonical runs on this artifact carry NO "
        "placement proof and are OBSERVATIONS by their own report."
    ),
    "notes": (
        "Decode is FLAT 24->96 threads and collapses at 192. Prefill rises to 96 "
        "and collapses at 192. 48 is the decode peak and costs ~5% of pp512."
    ),
}


# ---------------------------------------------------------------------------
# 6b. MEASURED_SERVING — 2026-09-23b, SERVING class, still OBSERVATION-GRADE
# ---------------------------------------------------------------------------
# instrument_class 'serving' (a real llama-server with the production drafter),
# but protocol id NONE and reps unrecorded => OBSERVATION, never a decision-gating
# claim and never a rate. A serving number and a bench number are NOT comparable
# (INSTRUMENT-CLASS-1): do not put these in a table next to MEASURED above.
MEASURED_SERVING = {
    "date": "2026-09-23",
    "record": "DS41-T5/C1 measured state (operator)",
    "instrument_class": "serving",
    "category": "CANDIDATE",
    "protocol": None,            # ★ OBSERVATION
    "reps": None,                # ★ unrecorded; a single-window read
    "binary": f"{KERNEL_BRANCH} @ {KERNEL_COMMIT} + {KERNEL_WORKTREE_PATCHES}",
    "binary_bindir": os.path.join(KERNEL_SOURCE_ROOT, "build-cpu"),
    "artifact": TRUNK_GGUF,
    "drafter": DRAFT_GGUF,
    "threads": THREADS,
    "arms_tps": {
        # (greedy, temp 0.7)
        "control_no_drafter": (8.21, 8.28),
        "dspark_block_5":     ((6.01, 7.22), 8.24),   # greedy SERIAL; 8.24 < 8.28 control
        "dspark_block_3":     (7.61, 10.26),
        "dspark_block_2":     (None, 10.48),
    },
    "alpha_by_block": {3: 0.444, 2: 0.516},
    "headline": (
        "1.266x at temp 0.7 (10.48 vs 8.28, block 2). At greedy the drafter is a "
        "LOSS on every block measured, because the serial verification path decodes "
        "one token per target decode and cannot exceed 1x."
    ),
    "gap": "block 2 greedy was not measured; the no-drafter control IS the greedy "
           "optimum until DS41-T5/C1 says otherwise.",
}


# ---------------------------------------------------------------------------
# 7. BUILDERS
# ---------------------------------------------------------------------------
def build_bench_env(extra: Optional[dict] = None) -> dict:
    """Canonical env for a bench of this model, with the candidate bindir pinned.

    Delegates to canonical_recipe.build_canonical_env so the OMP stack, GGML_IQK
    and the clang-20 libomp override cannot drift from the global recipe.
    """
    return CR.build_canonical_env(extra_vars=extra, library_path=KERNEL_BINDIR)


def build_bench_command(
    n_prompt: int = 0,
    n_gen: int = 512,
    reps: int = 5,
    model: str = TRUNK_GGUF,
    extra_flags: Optional[Iterable[str]] = None,
) -> tuple[str, list[str], dict]:
    """The blessed llama-bench command for this model.

    Calls canonical_recipe.build_canonical_bench_command with the EXPLICIT
    identity tuple (binary/source_root/library_path), which is what routes it
    through assert_explicit_bench_identity(). Production discovery is wrong for
    this model until USE_PRODUCTION_DISCOVERY flips: production does not know
    the `deepseek41` architecture.

    ★ The ONLY canonical flag this overrides is `-t`: 96 -> THREADS (48). -fa 1,
      -mmp 0 and the taskset/numactl prefix are inherited untouched. llama-bench
      has no -tb, so THREADS_BATCH is UNREACHABLE here by construction — see §4.
    """
    if USE_PRODUCTION_DISCOVERY:
        raise RecipeViolation(
            "USE_PRODUCTION_DISCOVERY is True but this builder still pins the "
            "experimental tree; update §2 before flipping it."
        )
    flags = ["-t", str(THREADS)]
    if extra_flags:
        flags += list(extra_flags)
    binary, cmd, env = CR.build_canonical_bench_command(
        model=model,
        n_prompt=n_prompt,
        n_gen=n_gen,
        reps=reps,
        extra_flags=flags,
        binary=KERNEL_BENCH,
        source_root=KERNEL_SOURCE_ROOT,
        library_path=KERNEL_BINDIR,
        ggml_iqk="1",
    )
    # The canonical flag list carries `-t 96`; ours appends `-t 48`. llama-bench
    # takes the LAST value, but a command carrying both is unreadable in a log,
    # so strip the inherited pair rather than relying on parse order.
    cmd = _strip_first_flag_pair(cmd, "-t", "96")
    return binary, cmd, env


def _strip_first_flag_pair(cmd: list[str], flag: str, value: str) -> list[str]:
    """Remove the FIRST `flag value` pair from cmd, leaving any later one."""
    for i in range(len(cmd) - 1):
        if cmd[i] == flag and cmd[i + 1] == value:
            return cmd[:i] + cmd[i + 2:]
    return list(cmd)


BASE_SERVER_FLAGS = [
    "--no-webui",
    "--no-mmap",                 # == -mmp 0; mmap defeats numactl --interleave=all
    "-t", str(THREADS),          # decode
    "-tb", str(THREADS_BATCH),   # prefill/batch — llama-server only
]
FLASH_ATTN_FLAGS = ["-fa", "on"]   # `--fa` DOES NOT EXIST; the long form is --flash-attn

KNOWN_BAD_FLAG_FORMS = {
    "--fa": "does not exist; the long form is --flash-attn (SYNC-10 lost 7 MTP arms to this)",
    "--spec-type draft-mtp": "WRONG SHAPE. DSpark is block-parallel with an "
                             "anchor-first layout and a Markov head; draft-mtp models "
                             "sequential nextn depth. Use draft-dspark.",
}


def build_serve_command(
    model: str = TRUNK_GGUF,
    bindir: str = KERNEL_BINDIR,
    host: str = "127.0.0.1",
    port: int = 18499,
    context: int = 8192,
    parallel_slots: int = 1,
    spec_dec: bool = True,
    variant: str = SPEC_DEC_VARIANT_DEFAULT,
    extra_flags: Optional[Iterable[str]] = None,
) -> tuple[list[str], dict]:
    """Serve command for this model. Returns (argv, extra_env).

    ★ STILL REFUSES BY DEFAULT — the refusal has only moved down one level.
      spec_dec defaults to True because the operator's requirement is max
      performance. It raises whenever the requested configuration cannot honour
      that requirement: when SPEC_DEC_STATUS is not 'measured' (as before), and
      now also when the requested VARIANT is not 'measured'. A caller that wants
      the no-draft configuration must still pass spec_dec=False, which makes the
      choice visible AT THE CALL SITE.

    ★ RETURN TYPE CHANGED from list to (argv, env). A variant may require a server
      environment variable (LLAMA_SPEC_EXACT), which is read ONCE at server
      construction and cannot be expressed as a flag. Returning the argv alone
      would have silently dropped it -- and dropping it fails SAFE (back to the
      exact serial path), which is exactly the kind of silent downgrade that would
      have been reported as a throughput mystery.
    """
    env: dict = {}

    if spec_dec:
        if SPEC_DEC_STATUS != "measured":
            raise RecipeViolation(
                "speculative decoding is REQUIRED by the operator directive of "
                f"2026-09-23 and is {SPEC_DEC_STATUS!r} for {RECIPE_ID}.\n"
                f"  drafter        : {SPEC_DEC['drafter_name']} "
                f"({SPEC_DEC['draft_gguf_download_state']})\n"
                f"  blocking work  : {SPEC_DEC['owner']}\n"
                f"  flips on       : {SPEC_DEC['flips_on'][0]}\n"
                "Pass spec_dec=False to launch the NO-DRAFT configuration "
                "knowingly. That configuration is a CANDIDATE, never an OPTIMUM: "
                "a draft path exists for this model, ours does not implement it."
            )
        if variant not in SPEC_DEC_VARIANTS:
            raise RecipeViolation(
                f"unknown spec-dec variant {variant!r}; known: "
                f"{sorted(SPEC_DEC_VARIANTS)}"
            )
        v = SPEC_DEC_VARIANTS[variant]
        if v["status"] != "measured":
            raise RecipeViolation(
                f"spec-dec variant {variant!r} is {v['status']!r} for {RECIPE_ID}.\n"
                f"  intent   : {v['intent']}\n"
                f"  flips on : {v.get('flips_on')}\n"
                f"  note     : {v['honest_note']}\n"
                f"  measure  : {GREEDY_EXACTNESS['measurement']}"
            )
        if not os.path.isfile(DRAFT_GGUF):
            raise RecipeViolation(f"drafter GGUF not found: {DRAFT_GGUF}")
        if v["spec_exact_env"]:
            env["LLAMA_SPEC_EXACT"] = v["spec_exact_env"]

    cmd = list(PREFIX)
    cmd.append(str(Path(bindir) / "llama-server"))
    cmd += BASE_SERVER_FLAGS
    cmd += ["-np", str(parallel_slots), "-c", str(context)]
    cmd += ["--host", host, "--port", str(port)]
    cmd += ["-m", model]
    cmd += FLASH_ATTN_FLAGS
    if spec_dec:
        v = SPEC_DEC_VARIANTS[variant]
        # draft-dspark requires --parallel 1 (tools/server/server.cpp:157,
        # pending llama.cpp issue #26741). Refuse rather than let the server
        # error out after a 483 GiB load.
        if parallel_slots != 1:
            raise RecipeViolation(
                f"--spec-type {DRAFT_SPEC_TYPE} requires -np 1 "
                f"(server.cpp:157, llama.cpp issue #26741); got {parallel_slots}"
            )
        cmd += ["-md", DRAFT_GGUF,
                "--spec-type", DRAFT_SPEC_TYPE,
                "--spec-draft-n-max", str(v["spec_draft_n_max"])]
    if extra_flags:
        cmd += list(extra_flags)
    return cmd, env


# ---------------------------------------------------------------------------
# 8. VALIDATORS
# ---------------------------------------------------------------------------
def assert_inherits_canonical() -> None:
    """Prove this overlay has not forked from canonical_recipe.

    Structural, not a comment: if the global recipe changes its prefix, OMP
    stack or pre-evict target, this fires instead of the two drifting apart.
    """
    if PREFIX is not CR.CANONICAL_PREFIX:
        raise RecipeViolation("PREFIX has been forked from canonical_recipe.CANONICAL_PREFIX")
    if OMP_ENV is not CR.CANONICAL_OMP_ENV:
        raise RecipeViolation("OMP_ENV has been forked from canonical_recipe.CANONICAL_OMP_ENV")
    if OMP_ENV.get("GGML_IQK") != "1":
        raise RecipeViolation("GGML_IQK must be 1; the iqk kernels are runtime-gated")
    if "-fa" not in INHERITED_BENCH_FLAGS or "-mmp" not in INHERITED_BENCH_FLAGS:
        raise RecipeViolation("-fa / -mmp 0 are no longer in the canonical bench flags")


def assert_no_bad_flag_forms(cmd: list[str]) -> None:
    for tok in cmd:
        if tok in KNOWN_BAD_FLAG_FORMS:
            raise RecipeViolation(f"{tok}: {KNOWN_BAD_FLAG_FORMS[tok]}")


def assert_decode_threads(cmd: list[str]) -> None:
    """`-t` must be THREADS, and it must appear exactly once."""
    idxs = [i for i, tok in enumerate(cmd) if tok == "-t"]
    if len(idxs) != 1:
        raise RecipeViolation(
            f"-t appears {len(idxs)} times; exactly one decode thread count is allowed "
            f"(recipe value {THREADS})"
        )
    if cmd[idxs[0] + 1] != str(THREADS):
        raise RecipeViolation(
            f"-t {cmd[idxs[0] + 1]}; the preliminary canonical value is {THREADS} "
            f"(operator directive 2026-09-23). {THREADS_RATIONALE}"
        )


def assert_no_silent_spec_dec_claim(text: str) -> None:
    """Refuse a write-up that claims spec decode for this model while it is pending.

    Cheap guard for report generators: a sentence saying this recipe uses
    speculative decoding is FALSE today, and the failure mode is a report that
    silently satisfies the operator's requirement on paper.
    """
    lowered = text.lower()
    if SPEC_DEC_STATUS != "measured":
        for phrase in ("spec decode enabled", "speculative decoding enabled",
                       "with speculative decoding", "draft model:"):
            if phrase in lowered:
                raise RecipeViolation(
                    f"text claims speculative decoding for {RECIPE_ID}, which is "
                    f"{SPEC_DEC_STATUS!r}: {phrase!r}"
                )
        return

    # ★ 2026-09-23b: the drafter landed, so the OLD claim became true and this
    # guard would have gone silent exactly when a NEW false claim became possible.
    # A guard that retires itself on success is not a guard. The false claim now
    # in reach is the EXACTNESS one, and the throughput one.
    if not GREEDY_EXACTNESS["holds_here"]:
        for phrase in ("bit-exact", "bit exact", "output-identical",
                       "identical to greedy", "exact at greedy",
                       "lossless speculative"):
            if phrase in lowered:
                raise RecipeViolation(
                    f"text claims greedy exactness for {RECIPE_ID}. On this stack "
                    f"speculative decoding is NOT output-identical at greedy: "
                    f"{GREEDY_EXACTNESS['why']} (offending phrase: {phrase!r}). "
                    f"The serial path IS exact, but only against the no-drafter path "
                    f"on the same binary -- say which."
                )
    if not GREEDY_EXACTNESS["measured_here"]:
        for phrase in ("greedy speedup", "faster at greedy", "1.27x at greedy",
                       "speculative decoding speeds up greedy"):
            if phrase in lowered:
                raise RecipeViolation(
                    f"text claims a GREEDY speedup for {RECIPE_ID}. Measured: the "
                    f"drafter is a LOSS at greedy on every block "
                    f"({MEASURED_SERVING['headline']}). The batched-greedy variant "
                    f"is {SPEC_DEC_VARIANTS['greedy-batched']['status']!r} "
                    f"(offending phrase: {phrase!r})."
                )


def assert_artifacts_exist(model: str = TRUNK_GGUF) -> None:
    if not os.path.isfile(model):
        raise RecipeViolation(f"artifact not found: {model}")
    size = os.path.getsize(model)
    if size != TRUNK_BYTES:
        raise RecipeViolation(
            f"{model} is {size} bytes, recipe expects {TRUNK_BYTES} "
            f"(482.98 GiB, both parts joined — an unjoined part1 is 480.0 GB)"
        )


def assert_binary_exists() -> None:
    if USE_PRODUCTION_DISCOVERY:
        return
    for path in (KERNEL_BENCH, KERNEL_SERVER):
        if not os.path.isfile(path):
            raise RecipeViolation(
                f"experimental binary not found: {path}\n"
                f"  branch {KERNEL_BRANCH} @ {KERNEL_COMMIT}, build {KERNEL_BUILD_NUMBER}\n"
                f"  {KERNEL_BUILD_NUMBER_IS_NOT_AN_IDENTITY}\n"
                "Production discovery is NOT a fallback: production does not know "
                "the `deepseek41` architecture and will refuse this GGUF."
            )


# ---------------------------------------------------------------------------
# 9. PREFLIGHT — prints the two limitations every time
# ---------------------------------------------------------------------------
STATUS_BANNER = (
    "=" * 78 + "\n"
    f"{RECIPE_ID} rev {RECIPE_REVISION} — ★ PRELIMINARY, ★ SPEC-DEC {SPEC_DEC_STATUS.upper()}\n"
    + "=" * 78 + "\n"
    f"  grade      : {STATUS['grade']} ({STATUS['measurement_category']}, "
    f"instrument_class={STATUS['instrument_class']})\n"
    f"  threads    : -t {THREADS} decode / -tb {THREADS_BATCH} prefill "
    f"(llama-bench CANNOT express the split; llama-server can)\n"
    f"  binary     : {KERNEL_BRANCH} @ {KERNEL_COMMIT} — EXPERIMENTAL, unpromoted\n"
    f"  spec decode: {SPEC_DEC_STATUS.upper()} — {SPEC_DEC['drafter_name']} drafter, "
    f"--spec-type {DRAFT_SPEC_TYPE}, variants {sorted(SPEC_DEC_VARIANTS)}\n"
    f"               temp>0: {MEASURED_SERVING['arms_tps']['dspark_block_2'][1]} t/s "
    f"(block 2) vs {MEASURED_SERVING['arms_tps']['control_no_drafter'][1]} control "
    f"= 1.27x\n"
    "  ★ GREEDY   : speculation is a LOSS at greedy today. The server forces the\n"
    "               SERIAL verification path at temp<=0, which decodes one token per\n"
    "               target decode and cannot exceed 1x: 7.61 t/s (block 3) against an\n"
    "               8.21 t/s NO-DRAFTER control. For greedy, NO DRAFTER is the optimum\n"
    "               until DS41-T5/C1 measures the batched-greedy variant.\n"
    "  ★ EXACTNESS: speculative decoding is NOT output-identical at greedy on this\n"
    "               stack. Batch invariance holds on none of our three compute planes.\n"
    f"               See GREEDY_EXACTNESS; measurement: {GREEDY_EXACTNESS['measurement']}\n"
    + "=" * 78
)


def preflight(model: str = TRUNK_GGUF, check_host: bool = True) -> None:
    """Validate everything checkable without running inference. Prints the banner."""
    print(STATUS_BANNER, file=sys.stderr)
    assert_inherits_canonical()
    assert_artifacts_exist(model)
    assert_binary_exists()
    if check_host:
        CR.validate_host_environment(skip_perf_paranoid=True)


def _main(argv: list[str]) -> int:
    preflight(check_host="--no-host" not in argv)
    print(f"OK: {RECIPE_ID} rev {RECIPE_REVISION} preflight passed "
          f"(spec-dec {SPEC_DEC_STATUS}).")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
