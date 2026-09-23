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
RECIPE_REVISION = "2026-09-23"
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
        "The operator's standing requirement is MAX PERFORMANCE, and for this "
        "model max performance means the DSpark drafter. It does not exist here "
        "yet: the antirez GGUF ships ZERO mtp.* tensors (1046 tensors, blk.0-39 "
        "only), the official shards 44-46 (~8 GB, 2,401 mtp.* tensors) are "
        "DOWNLOADING as of 2026-09-23, and the accept/verify loop is unwritten "
        "IN THE REFERENCE TOO (generate.py never calls forward_spec). So today's "
        "recipe cannot express the requirement, and every number produced under "
        "it is a NO-DRAFT number."
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
    "instrument_class": "bench",
    "supersedes": None,
    "superseded_by": None,
    "record": "handoffs/active/deepseek-v41-flash-evaluation.md (DS41-T6, DS41-B13)",
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
# which antirez writes and never reads. This is why SPEC_DEC is pending.
TRUNK_MTP_TENSORS = 0

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
KERNEL_BUILD_NUMBER = 10303
KERNEL_SOURCE_ROOT = "/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923"
KERNEL_BINDIR = os.path.join(KERNEL_SOURCE_ROOT, "build", "bin")
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
SPEC_DEC_STATUS = "pending"

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

    # --- the four things that are not true yet, each with its own field ---
    "draft_gguf": None,                    # pending: no draft artifact exists
    "draft_gguf_source": "official shards 44-46 (~8 GB, 2,401 mtp.* tensors)",
    "draft_gguf_download_state": "DOWNLOADING as of 2026-09-23",
    "draft_gguf_conversion": "pending: shards 44-46 -> a separate draft GGUF; no 510 GB re-download needed",

    "spec_type": None,                     # pending: e.g. "draft-mtp" — NOT yet valid here
    "spec_draft_n_max": None,              # pending
    "spec_draft_p_min": None,              # pending
    "alpha": None,                         # pending: acceptance rate, never measured
    "drafted_per_token": None,             # pending

    # --- why the flags above cannot simply be filled in ---
    "accept_verify_loop": "NOT IMPLEMENTED — and not implemented in the reference either "
                          "(generate.py never calls forward_spec). We would write it.",
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

    # --- ★ THE FLIP CONDITION. All five, in order. Fewer is not enough. ---
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
    "--threads-batch-draft": "there is no drafter for this model yet; see SPEC_DEC",
}


def build_serve_command(
    model: str = TRUNK_GGUF,
    bindir: str = KERNEL_BINDIR,
    host: str = "127.0.0.1",
    port: int = 18499,
    context: int = 8192,
    parallel_slots: int = 1,
    spec_dec: bool = True,
    extra_flags: Optional[Iterable[str]] = None,
) -> list[str]:
    """Serve command for this model.

    ★ REFUSES BY DEFAULT. spec_dec defaults to True because the operator's
      requirement is max performance, and with SPEC_DEC_STATUS == 'pending'
      there is no way to honour it — so the default path raises. A caller that
      genuinely wants today's no-draft configuration must pass spec_dec=False,
      which makes the limitation visible AT THE CALL SITE rather than in a
      module nobody reads.
    """
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
    cmd = list(PREFIX)
    cmd.append(str(Path(bindir) / "llama-server"))
    cmd += BASE_SERVER_FLAGS
    cmd += ["-np", str(parallel_slots), "-c", str(context)]
    cmd += ["--host", host, "--port", str(port)]
    cmd += ["-m", model]
    cmd += FLASH_ATTN_FLAGS
    if extra_flags:
        cmd += list(extra_flags)
    return cmd


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
    if SPEC_DEC_STATUS == "measured":
        return
    lowered = text.lower()
    for phrase in ("spec decode enabled", "speculative decoding enabled",
                   "with speculative decoding", "draft model:"):
        if phrase in lowered:
            raise RecipeViolation(
                f"text claims speculative decoding for {RECIPE_ID}, which is "
                f"{SPEC_DEC_STATUS!r}: {phrase!r}"
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
    f"  spec decode: {SPEC_DEC_STATUS.upper()} — {SPEC_DEC['drafter_name']} drafter "
    f"({SPEC_DEC['draft_gguf_download_state']}), accept/verify loop unimplemented.\n"
    "               The operator's MAX-PERFORMANCE requirement is NOT satisfied by\n"
    "               this recipe. Numbers taken under it are NO-DRAFT numbers and are\n"
    "               CANDIDATE, never OPTIMUM.\n"
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
