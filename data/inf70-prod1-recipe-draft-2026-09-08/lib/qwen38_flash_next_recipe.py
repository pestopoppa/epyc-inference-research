"""Canonical serving recipe for Qwen3.8-Flash-Next on the EPYC 9655 CPU path.

INTENDED REPO PATH: epyc-inference-research/scripts/lib/qwen38_flash_next_recipe.py

WHY THIS FILE EXISTS
--------------------
The INF-70 CPU-decode campaign carried its serving recipe in PROSE, spread over ~10
occurrences in a 3900-line handoff. Prose gets transcribed wrong, and it did:

  * an agent wrote ``--fa 1``. That long-form flag DOES NOT EXIST (``-fa 1``, ``-fa on``
    and ``--flash-attn on`` are all valid; the long name is ``--flash-attn``, never
    ``--fa``). ``llama-server`` dies at argument-parse time, BEFORE model load.
    Two agents lost arms to it in one session; SYNC-10 lost ALL SEVEN MTP arms of a
    hard-won region-lock turn.
  * the PROD-1 prose recipe also states ``GGML_FA_SPLIT_KV=0`` for the 23.16 t/s arms.
    The live ``/proc/PID/environ`` readback of those arms shows it was NOT exported and
    no ``-fa`` flag was passed at all. The prose was a HYBRID of two configurations.

So: the recipe is DATA. Import the constants. Do not retype them, and do not read them
out of the handoff.

The operator's standing direction (OP-35, 2026-09-05, resolved): the MTP head is part of
the MODEL, not an experiment. **Every serving path for this model carries its MTP head by
default.** ``build_serve_command()`` therefore emits the MTP flags unless a caller passes
``mtp=False``, and ``assert_mtp_present()`` exists so a launcher can fail closed.

VALIDATE, DO NOT TRUST
----------------------
``assert_flag_forms_exist(binary)`` dry-runs every flag in this module against a real
``llama-server`` with a nonexistent model path. The parse either reaches model load
(flag exists) or reports ``invalid argument`` (flag does not). It costs ~1 second per
flag and needs no model, no GPU and no region lock. Run it in preflight. It is the
mechanism that would have caught ``--fa 1`` in one second.

PROVENANCE
----------
Every constant below is sourced from a file on disk, not from the handoff prose:

  arm script   /mnt/raid0/llm/tmp/inf70/agents/champion3/arm.sh
  driver       /mnt/raid0/llm/tmp/inf70/agents/champion3/driver3.sh
  build        /mnt/raid0/llm/tmp/inf70/agents/champion3/build3.sh
  env readback /mnt/raid0/llm/tmp/inf70/agents/champion3/runs/Q_C3_r1.env
  digests      /mnt/raid0/llm/tmp/inf70/agents/champion3/binaries.sha256
  workload     /mnt/raid0/llm/tmp/inf70/agents/champion3/client.py
  handoff      handoffs/active/cpu-decode-roofline-program.md (CHAMPION-3 record, ~L1297)

Related codified recipe (the CPU llama-bench baseline, a DIFFERENT object — that one is
``-t 96`` bare llama-bench, this one is ``-t 48`` served llama-server with MTP):
  epyc-inference-research/scripts/lib/canonical_recipe.py
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional

RECIPE_ID = "qwen38-flash-next-cpu-mtp"
RECIPE_REVISION = "2026-09-08"   # CHAMP-2 THP shim ADOPTED into the recipe


class RecipeViolation(AssertionError):
    """Raised when a command/env does not match the codified recipe."""


# ---------------------------------------------------------------------------
# 1. ARTIFACT IDENTITY
# ---------------------------------------------------------------------------
# The trunk. NOTE THE NAME LIES: "IQ4_XS-uniform" is NOT uniform. Its per-class
# composition is mixed and its EFFECTIVE width is 4.995 bpw, not 4.0 — every
# roofline/bytes-per-token calculation must use 4.995, and anyone budgeting from
# the filename will be ~25% low.
TRUNK_GGUF = (
    "/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/"
    "IQ4_XS-uniform/Qwen3.8-Flash-Next-IQ4_XS-uniform.gguf"
)
TRUNK_BYTES = 98_392_912_256
TRUNK_SHA256 = "4bfb98496364f8721c1e3ea084a238d690c52b1042a317e05fb43b756c9f8957"
TRUNK_ARCH = "qwen4exp"          # general.architecture
TRUNK_TENSORS = 1224

TRUNK_EFFECTIVE_BPW = 4.995      # NOT 4.0 — see TRUNK_QUANT_COMPOSITION
TRUNK_ACTIVE_PARAMS_B = 6.671    # resident tensors + 10/512 of each expert
TRUNK_BYTES_PER_TOKEN_GB = 4.166 # 6.671e9 * 4.995 / 8 — the roofline denominator

# The "uniform" in the directory name is FALSE and the error compounds: assuming
# 4.0 bpw and 6.0 B active params under-predicts the streamed bytes by 38.9%
# (1.112 x 1.249). Measured tensor census (b11/gguf_inv.py over the real GGUF):
TRUNK_QUANT_CENSUS = {          # tensor count by dtype
    "IQ4_XS": 586,
    "IQ4_NL": 182,
    "Q5_K": 54,
    "Q5_1": 12,                  # blk.0-5 ffn_down_exps / ffn_down_shexp (use_more_bits)
    "Q6_K": 1,                   # output.weight (lm_head)
    # remainder: F32 norms, ssm_conv1d, and the ffn_gate_inp routers
}
TRUNK_QUANT_COMPOSITION = {      # tensor class -> stored dtype, with GB/token
    "ffn_gate_inp (router), all norms, ssm_conv1d": ("F32",   32.00, 0.259),
    "output.weight (lm_head)":                      ("Q6_K",   6.56, 0.522),
    "attn_qkv (36 GDN layers), attn_v":             ("Q5_K",   5.50, 0.660),
    "ffn_down_exps (42 layers), ffn_down_shexp, hc_*_up": ("IQ4_NL", 4.50, 0.510),
    "ffn_down_exps (layers 0-5)":                   ("Q5_1",   5.50, None),
    "everything else":                              ("IQ4_XS", 4.25, 2.215),
}

# The MTP head. A SEPARATE GGUF loaded as the draft model (-md), not embedded.
#
# ★ IT IS THE *SHARED* HEAD, AND THAT IS A FEATURE, NOT AN INHERITANCE.
# The shared head carries NO output.weight and NO token_embd.weight of its own:
# `build_qwen4exp_mtp` falls through to `model.output` (src/models/qwen4exp.cpp:582,
# `layer.nextn.shared_head_head ? ... : model.output`), and the loader logs
# `borrow_shared_tensor: tensor output.weight taken from the target model`.
# So ONE 521.5 MB q6_K slab has TWO consumers, and each verification step re-warms
# exactly the bytes the next draft step reads — very likely the reason B10 measured
# 1.6x the DRAM ceiling. B12 tested giving the draft its OWN head and REFUTED it:
# a private head ADDS to the working set rather than replacing it (521.5 + 337.7 =
# 859.2 MB, +64.8%, against ~402 MB of L3), and both requantised candidates measured
# SLOWER than the borrowed q6_K baseline (1.077x Q4_K, 1.097x IQ4_XS wall).
# Generalisable rule from that miss: when a change gives a consumer its OWN copy of
# a shared tensor, cost the WORKING SET, not the tensor.
#
# B12's dependency on PROD-1 is therefore DISCHARGED IN FAVOUR OF THE STATUS QUO:
# the head artifact is frozen as shared-Q8_0.
MTP_HEAD_GGUF = (
    "/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/"
    "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
)
MTP_HEAD_KIND = "shared-Q8_0"
MTP_HEAD_BYTES = 2_786_568_256
MTP_HEAD_SHA256 = "5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6"
MTP_HEAD_PROVENANCE = "hf:unsloth/Qwen3.8-Flash-Next-GGUF MTP/ @ revision 5d16c055"

# Do NOT substitute the self-contained head. It exists
# (.../MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf, 4,137,429,120 B, LFS oid cd87e5d1...)
# and it is the WRONG choice for exactly the working-set reason above.
MTP_HEAD_REJECTED = {
    "mtp-Qwen3.8-Flash-Next-Q8_0.gguf": "self-contained head: +64.8% head working set",
    "mtp-shared-Q8_0-headIQ4_XS.gguf": "B12 candidate, 1.097x wall vs borrowed q6_K — NO-GO",
    "mtp-shared-Q8_0-headQ4_K.gguf": "B12 candidate, 1.077x wall vs borrowed q6_K — NO-GO",
}

# Artifact-axis note (do NOT silently substitute): `IQ4_XS-uniform-gateup-r16`
# (98,267,079,776 B, sha256 6468558b42579664af5a4551292828264905109f9a8d3fea4d8e10e3c7ce47d7)
# is measured better (+1.69% MTP, +0.63% plain, PPL null at MDE 1.041%) and is the
# Axis B/D COMPARISON BASELINE. It is deliberately NOT the recipe's artifact: it is the
# only non-bit-identical lever in the campaign, and adopting it would retire the
# bit-identity property that makes the champion kernel promotable on a digest match.
# `IQ4_XS-uniform` stays the era anchor and this recipe's artifact until PROD-3 decides.
# The production-SERVED file today is a different object again: the 3-shard
# `.../UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf`. This recipe
# describes the CAMPAIGN artifact; reconciling served-vs-anchor is PROD-3's job.

# ---------------------------------------------------------------------------
# 2. KERNEL IDENTITY (the champion)
# ---------------------------------------------------------------------------
# ★★ THE CHAMPION IS NOT A COMMIT. IT IS A COMMIT PLUS A LAUNCH RECIPE.
#    ef81196d5 + GGML_NOHUGEPAGE_PROCESS=1 at launch (see CHAMPION_GGML_ENV /
#    THP_SHIM). Naming only the commit names a different, slower, ~8x noisier
#    thing. This constant block is HALF the artifact; the env block is the other.
#
# ★ THE CHAMPION MOVED 2026-09-08 (the fold). `inf70/champion3` @ 9c4f73e29 is now
#   an ANCESTOR, not the champion -- it is contained in ef81196d5 by ancestry.
#   Standing rule: THE CHAMPION IS ALWAYS CURRENT.
# ★ THE PIN IS NOT YET RESOLVED TO THE CURRENT CHAMPION, AND THE MODULE SAYS SO.
# CHAMPION_* below still name the PIN-VERIFIABLE artifact (champion3, build 10241),
# because that is the only tree whose object digests were measured. The CURRENT
# champion is the folded ef81196d5, for which no build number and no digests exist
# yet. Rather than pin a half-known artifact or silently keep naming a superseded
# one, the module carries BOTH and refuses to certify identity until WRAP-10 resolves
# it (CHAMPION_PIN_RESOLVED). A recipe that quietly names last week's champion is the
# failure this module exists to prevent.
CHAMPION_BRANCH = "inf70/champion3"
CHAMPION_COMMIT = "9c4f73e2965ff347593ef335d52aea585af92106"
CHAMPION_BUILD_NUMBER = 10241
CHAMPION_TREE = "/mnt/raid0/llm/worktrees/inf70/champion2"  # branch inf70/champion3
CHAMPION_BINDIR = "/mnt/raid0/llm/worktrees/inf70/champion2/build-cpu/bin"

# ★★ THE CURRENT CHAMPION. commit + recipe; the commit alone under-specifies it.
CURRENT_CHAMPION = {
    "commit": "ef81196d5",          # ⚠ SHORT FORM -- full sha not yet recorded here
    "branch": "inf70/fold-candidate-20260908",
    "recipe": "GGML_NOHUGEPAGE_PROCESS=1 at launch (see CHAMPION_GGML_ENV / THP_SHIM)",
    "build_number": None,           # ⚠ NOT MEASURED -- do NOT reuse 10241
    "sha256": None,                 # ⚠ NOT MEASURED
    "folded": "2026-09-08",
    "lineages_by_ancestry": {
        "gpu_tip":            "bff30cebe",
        "champion_of_record": "445e93a8",
        "cpu_lineage":        "9c4f73e29",   # == CHAMPION_COMMIT above (now an ANCESTOR)
        "frozen_production":  "0db32c06e",
    },
    "fold_properties": (
        "zero files deleted; exactly one --no-ff merge; NO cherry-picks; "
        "57 akm- keeps reachable; pre-fold rollback tag on the fork; "
        "FOLD-2 passed in full (G1 SSM_SCAN 7/7 incl. K=4/K=3 rollback, "
        "G2 MUL_MAT 1140/1140, G3 GDN 39/39, G4 dispatch observed -- 27,516 nodes, "
        "SSM_SCAN=0, recurrent ops on ROCm0; G5 tg128 +0.052% NOT decisive)"
    ),
    "do_not_fold": {
        "inf70/sync17-fix2 @ 2516c9807": (
            "GGML_SCALE_SPLIT and GGML_SOLO_YIELD_ROWCOL BOTH DEFAULT ON, and that "
            "default IS the -2.136% regression. Flip both OFF before any fold."
        ),
        "inf70/retest1-fix1 @ 2516c9807": (
            "measurement instrument only -- all five knobs runtime-switchable; "
            "it is the binary the final numbers came from. Do NOT merge."
        ),
        "feature/tree-draft-v6": (
            "the champion carries a LATER CONTRADICTING DECISION. Folding a superseded "
            "decision is a failure class ancestry cannot see, and cherry-equivalence "
            "waves it through."
        ),
    },
}

# ★ FAIL-CLOSED. assert_binary_identity() and the headline both depend on this.
# While False, the module can verify the PRIOR artifact and must NOT be read as
# certifying the current champion's binary.
CHAMPION_PIN_RESOLVED = False
CHAMPION_PIN_GAP = (
    "CHAMPION_* name champion3 (build 10241, digests measured). CURRENT_CHAMPION is "
    "ef81196d5 with no build number and no digests. WRAP-10 must either rebuild and "
    "re-digest at ef81196d5, or land the module with this flag False and a loud "
    "refusal in preflight."
)
# Byte-identical preserved copy of the same binaries (survives a worktree rebuild):
CHAMPION_BINDIR_ARCHIVE = "/mnt/raid0/llm/tmp/inf70/agents/champion3/bin-c3"

# sha256 of the shipped objects, from champion3/binaries.sha256. Check these before
# quoting a number against this recipe.
CHAMPION_SHA256 = {
    "llama-server": "9f69bac917d70b5288caa87427a38c64446be2175974bbb24eb45182053171bc",
    "libggml-cpu.so.0.16.0": "2fb5b67ec39ab747e604ba2e6dd1ea898c199e3b664197851c99b92617e9ad87",
    "libggml-base.so.0.16.0": "fa004e2a782690315d66de29bd89656dd0c3f7a6ca79a7d03605c694848b2df0",
    "libllama-common.so.0.0.10241": "27b654efe66fa3ca654cfeab98d1d1f1283d24c2092006899757601d9e33352a",
}

# Exactly the cmake line that produced the champion (build3.sh). NOT a reconstruction.
CHAMPION_CMAKE_ARGS = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_NATIVE=ON",
    "-DGGML_OPENMP=ON",
    "-DGGML_LLAMAFILE=ON",
    "-DGGML_BACKEND_DL=OFF",
    "-DGGML_CPU_ALL_VARIANTS=OFF",
    "-DGGML_CUDA=OFF",
    "-DGGML_HIP=OFF",
    "-DGGML_BLAS=OFF",
    "-DGGML_RPC=OFF",
    "-DLLAMA_CURL=OFF",
    "-DBUILD_SHARED_LIBS=ON",
]
CHAMPION_BUILD_TARGETS = ["llama-server", "llama-cli", "test-backend-ops"]
CHAMPION_COMPILER = "GNU 15.2.0"


# ---------------------------------------------------------------------------
# 3. PROCESS WRAPPING + OMP ENV STACK  (MANDATORY — omitting silently changes the number)
# ---------------------------------------------------------------------------
# taskset BEFORE numactl; that order is what the canonical baseline protocol asserts.
# 0-95 = the 96 physical cores of the bench region. 184-191 are the SMT SIBLINGS of
# 88-95 and are INSIDE this region's contention domain even though a naive
# "disjoint from 0-95" test calls them disjoint (a mislabel that shipped in every
# champion-1-derived co-residency report).
SERVE_PREFIX = ["taskset", "-c", "0-95", "numactl", "--interleave=all"]

CANONICAL_OMP_ENV = {
    "OMP_PROC_BIND": "spread",
    "OMP_PLACES": "cores",
    "OMP_WAIT_POLICY": "active",   # 'passive' is a measured -81.6% trap
    "OMP_DYNAMIC": "false",
}

# GGML runtime env. The first three are EXACTLY as read back from the live
# champion-3 process (champion3/runs/Q_C3_r1.env). ★ THE FOURTH IS NOT: the
# champion-3 arms did NOT carry GGML_NOHUGEPAGE_PROCESS -- it was adopted
# 2026-09-08 on separate, later evidence (CHAMP-2), and adding it here is a
# DELIBERATE change of the measured condition, not a transcription of that
# readback. See THP_SHIM and the HEADLINES caveat.
# GGML_IQK defaults to OFF in this tree
# (iqk_dispatch.cpp: `s && atoi(s) != 0`) — the kernels are compiled in but
# runtime-gated, so omitting GGML_IQK=1 measures a different kernel entirely.
CHAMPION_GGML_ENV = {
    "GGML_IQK": "1",
    "GGML_FUSED_DECODE_OFF": "1",  # champion runs the GRAPH path, not the megakernel
    "GGML_FA_SPLIT_KV": "0",       # BE-2: makes 1-row decode and n-row verify row-exact
    # ★ CHAMP-2, ADOPTED 2026-09-08 (operator-ruled). UNIT = SESSION (one process
    # launch), set AT LAUNCH. See THP_SHIM below for the full record. This entry is
    # what makes the champion artifact "commit + recipe" rather than "commit".
    "GGML_NOHUGEPAGE_PROCESS": "1",
}

# ---------------------------------------------------------------------------
# 3b. ★ THE THP SHIM (CHAMP-2) — ADOPTED, AND ITS UNIT IS THE SESSION
# ---------------------------------------------------------------------------
# ★ THE CHAMPION COMMIT ALONE UNDER-SPECIFIES THE CHAMPION.
# The artifact is `ef81196d5` + GGML_NOHUGEPAGE_PROCESS=1 AT LAUNCH. Quoting the
# commit without this env entry names a different, slower, ~8x noisier thing.
#
# ⚠⚠ TWO KNOBS, NOT ONE. NEVER WRITE ONE AND MEAN THE OTHER. Spell both out:
#
#   GGML_NOHUGEPAGE          madvise(MADV_NOHUGEPAGE) on the model buffer inside
#                            ggml_aligned_malloc (libggml-base). ALREADY ON by
#                            compiled default; 86.4% of the champion. LEAVE UNSET.
#   GGML_NOHUGEPAGE_PROCESS  prctl(PR_SET_THP_DISABLE) for the WHOLE PROCESS, taken
#                            BEFORE the 92 GB model allocation (common/common.cpp).
#                            Compiled default OFF (opt-in). ★ EXPORT IT =1.
#
# Conflating them discards 86.4% of the champion. They have different mechanisms
# (madvise vs prctl), different scopes (one buffer vs the whole process), and
# different default states.
#
# GATING: the process shim requires the madvise knob to be on --
#   common_thp_env_on("GGML_NOHUGEPAGE") && common_thp_env_opt_in("GGML_NOHUGEPAGE_PROCESS")
# so GGML_NOHUGEPAGE=0 is a MASTER OFF that silently disables this too.
THP_SHIM = {
    "knob": "GGML_NOHUGEPAGE_PROCESS",
    "value": "1",
    "mechanism": "prctl(PR_SET_THP_DISABLE) before the 92 GB model allocation",
    "set_at": "LAUNCH (process environment)",
    # ★ THE FIELD PEOPLE GET WRONG. A floor carries its harness, its n, its
    # contention model, its host state AND ITS UNIT.
    "unit": "SESSION (one process launch) -- NEVER the arm",
    "compiled_default": "OFF (opt-in)",
    "recommended_default": "ON",
    "not_this_knob": "GGML_NOHUGEPAGE (madvise, already ON, 86.4% of the champion)",
    "requires": "GGML_NOHUGEPAGE on (its own default)",
    # Verification, also session-unit: 1 = THP allowed (shim OFF), 0 = prctl in force
    # (shim ON). Read ONCE PER LAUNCH, fail-closed -- abort a session whose requested
    # state does not match the observed one. AnonHugePages is NOT a valid
    # discriminator (0.06% of Rss at load, ~6% minutes later on the same process).
    "verify": "grep ^THP_enabled /proc/<pid>/status   # 0 = shim ON, 1 = shim OFF",
    "verify_expect": "0",
    "fail_closed": True,
    # Evidence. Pre-registered paired sign test, PREREG-THP-DECISION.md frozen
    # 14:08:05Z (sha256 337200315c6b...) BEFORE the lock; boundary in tools/decide.py,
    # unit-tested before the run.
    "decision": "ADOPTED -- likely improvement, KEEP (operator-ruled 2026-09-08)",
    "design": "paired sign test, 6 pairs (12 sessions), order-balanced, early stop at first look",
    "result": "6/6 pairs ON-faster",
    # ★ EXACT alpha, ENUMERATED over all 2**10 sequences with nested stopping.
    # A union bound would have said 0.078 and been OVER BUDGET SILENTLY.
    "alpha_exact_two_sided": 0.0430,
    "alpha_union_bound_would_have_said": 0.078,
    # ★ DIRECTION IS CLAIMED. MAGNITUDE IS NOT -- the design sized for direction only.
    "magnitude_claimed": False,
    "median_effect_pct": 5.23,
    "pair_effects_pct": (4.843, 5.617, 5.934, 5.664, 0.321, 0.558),
    # ★ It bought PRECISION as well as throughput -- this may outlast the speed result.
    "between_launch_sd_pct_off": 5.081,   # 9 launches, range 12.55% -- RETIRED config
    "between_launch_sd_pct_on": 0.609,    # 6 launches, range 1.79%  -- ADOPTED config
    "sd_ratio": 8.3,
    "variance_ratio": 70,
    "paired_test_variance_ratio": 25.3,
    # Launches needed for a given 95% CI, ON vs OFF.
    "launches_for_ci": {"1.0%": (2, 100), "0.5%": (6, 397), "0.25%": (23, 1587)},
    # ★ NO FOLD, NO BRANCH, NO REBUILD. The champion binary stays ef81196d5,
    # BIT-IDENTICAL. This is a RECIPE change, not a kernel change, and the evidence
    # was collected via exactly this per-session env route.
    "delivery": "launcher env only -- no code change, no rebuild, trivially reversible",
    "binary_unchanged": "ef81196d5",
    # Correctness: page backing, not arithmetic. Every champion-state comparison in
    # the campaign was 24/24 byte-identical, including cross-binary. No shim-ON-vs-OFF
    # specific correctness gate was run, and none is claimed.
    "correctness": "24/24 byte-identical across the campaign; no shim-specific gate run",
    # ★ THE 1200-FOLD ERROR THIS FIELD EXISTS TO PREVENT.
    # Applying the ARM floor (0.501% within-session) to this SESSION-unit question
    # said "4 sessions/side" for +0.16%. The correct session-unit answer is 4780.
    # The shim cannot be switched between arms in a live process; a per-arm number
    # for it is meaningless BY CONSTRUCTION.
    "arm_floor_does_not_transfer": (
        "within-session arm sd 0.501% vs between-session launch sd 2.793% (~13x coarser). "
        "Using the arm floor here gave 4 sessions/side where the answer is 4780 -- a "
        "1200-fold error."
    ),
    "record": "handoffs/active/cpu-decode-roofline-program.md (CHAMP-2); "
              "/mnt/raid0/llm/tmp/inf70/agents/retest1/FOLD-RECORD-THP.md",
}

# No GOMP_* and no KMP_* anywhere in this recipe: the build is libgomp (GNU 15.2.0),
# so OpenMP is controlled entirely through OMP_*. KMP_BLOCKTIME belongs to the
# ik_llama/Intel-OpenMP path and does not apply here.


# ---------------------------------------------------------------------------
# 4. KNOBS COMPILED INTO THE CHAMPION, WITH THEIR INTENDED DEFAULT STATE
# ---------------------------------------------------------------------------
# Verified against the shipped binary with `strings`, not read off the handoff. The
# binary self-describes via these markers; assert_knob_markers() re-checks them.
CHAMPION_MARKERS = [
    "INF70_CHAMPION_CPU_DEFAULT_ON=GGML_ROWCOL_SPLIT,GGML_TINY_SOLO,GGML_EMPTY_SKIP"
    ";DEFAULT_OFF=GGML_VEC_SIGMOID",
    "INF70_CHAMPION3_CPU_DEFAULT_ON=GGML_QSPLIT(multi-row-only,GGML_QSPLIT_MIN=INT64_MAX)"
    ";DEFAULT_INERT=GGML_TINY_SOLO_ROWS=1,GGML_TINY_SOLO_ROWS_MAX",
    "INF70_CHAMPION3_QUANTS_DEFAULT_ON=GGML_VEC_Q8K",
    "INF70_CHAMPION3_PROCESS_THP_DISABLE=DEFAULT_OFF"
    ";OPT_IN=GGML_NOHUGEPAGE_PROCESS=1;MASTER_OFF=GGML_NOHUGEPAGE=0",
]

# name -> (intended state, how the default is reached, source of truth in the tree)
# "leave unset" means the compiled default IS the champion state. Setting a knob to
# its default value is harmless; setting it to the other value leaves the champion.
CHAMPION_KNOBS = {
    # --- must be EXPORTED (compiled default is NOT the champion state) ---
    "GGML_IQK":               ("1",  "export", "compiled default OFF; iqk kernels runtime-gated"),
    "GGML_FUSED_DECODE_OFF":  ("1",  "export", "compiled default: fused decode ON; champion runs graph path"),
    "GGML_FA_SPLIT_KV":       ("0",  "export", "compiled default ON (ops.cpp: e ? atoi(e)!=0 : true)"),
    # ★ CHAMP-2 ADOPTED 2026-09-08. Was ("off","leave unset") through 2026-09-07.
    # UNIT = SESSION. NOT GGML_NOHUGEPAGE (see THP_SHIM). Requires GGML_NOHUGEPAGE on.
    "GGML_NOHUGEPAGE_PROCESS":("1",  "export", "CHAMP-2 prctl(PR_SET_THP_DISABLE) shim, compiled default OFF (opt-in); ADOPTED -- 6/6 pairs ON-faster, exact alpha 0.0430, unit=SESSION; verify THP_enabled==0 per launch"),
    # --- default ON, leave unset ---
    "GGML_ROWCOL_SPLIT":      ("on", "leave unset", "common.h: unset/empty => true; =0 restores upstream row-only split"),
    "GGML_TINY_SOLO":         ("on", "leave unset", "ggml-cpu.c: unset/empty => true"),
    "GGML_EMPTY_SKIP":        ("on", "leave unset", "ggml-cpu.c: unset/empty => true"),
    "GGML_VEC_Q8K":           ("on", "leave unset", "quants.c: unset/empty => 1 (CHAMPION-3)"),
    "GGML_QSPLIT":            ("on", "leave unset", "iqk_dispatch.cpp: unset/empty => true (CHAMPION-3)"),
    "GGML_MMID_SLAB":         ("on", "leave unset", "B3-k expert slab partition; unset => enabled"),
    "GGML_CPU_CONCAT_DIM0_ROWS": ("on", "leave unset", "ops.cpp: explicit opt-OUT only"),
    "GGML_NOHUGEPAGE":        ("on", "leave unset", "the ggml_aligned_malloc madvise in libggml-base; 86.4% of the champion"),
    # --- default OFF / inert, leave unset ---
    "GGML_QSPLIT_MIN":        ("INT64_MAX", "leave unset", "multi-row branch only; verified in machine code (movabs 0x7fffffffffffffff)"),
    "GGML_ROWEXACT_N":        ("0",  "leave unset", "ggml-cpu.c GGML_ROWEXACT_DEFAULT_N 0 — inert"),
    "GGML_VEC_SIGMOID":       ("off","leave unset", "common.h: set-and-nonzero required; not bit-identical to libm expf"),
    # (GGML_NOHUGEPAGE_PROCESS moved to the EXPORTED block above -- CHAMP-2 adopted
    #  2026-09-08. Its former note quoted "pooled +0.16%, 73/120 = zero", which was an
    #  ARM-unit result for a SESSION-unit knob: SUPERSEDED, see THP_SHIM.)
    "GGML_TINY_SOLO_ROWS":    ("1",  "leave unset", "SYNC-15 widened predicate; 1 = champion"),
    "GGML_IQK_Q8_0":          ("off","leave unset", "iqk_dispatch.cpp: set-and-nonzero required"),
    "GGML_ROWCOL_MIN_ELEMS":  ("512","leave unset", "common.h default"),
    "GGML_GET_ROWS_MIN_BYTES":("65536","leave unset", "ggml-cpu.c default 64*1024"),
}

# ⚠ GGML_NOHUGEPAGE vs GGML_NOHUGEPAGE_PROCESS are DIFFERENT KNOBS and conflating them
# discards 86.4% of the champion. The first is the madvise inside ggml_aligned_malloc
# (libggml-base) and STAYS ON — proven live at AnonHugePages 0.057% of RSS against
# pristine's 99.888%. Only the process-wide PR_SET_THP_DISABLE in libllama-common
# (GGML_NOHUGEPAGE_PROCESS) is default-OFF *in the binary* — and as of 2026-09-08 the
# RECIPE turns it ON. Compiled default OFF, recipe default ON: both statements are
# true and they are not in conflict. See THP_SHIM (§3b) for the unit and the evidence.


# ---------------------------------------------------------------------------
# 5. SERVER FLAGS
# ---------------------------------------------------------------------------
THREADS = 48                     # NOT 96: the served decode optimum on this model
CONTEXT = 8192
PARALLEL_SLOTS = 1

# Base flags for every serve of this model.
BASE_SERVER_FLAGS = [
    "--no-webui",
    "-np", str(PARALLEL_SLOTS),
    "-c", str(CONTEXT),
    "-t", str(THREADS),
    "--no-mmap",                 # mmap SHARES NUMA placement; only --no-mmap respects
                                 # numactl --interleave=all for the weights
    "-lv", "4",
]

# Flash attention. THE FLAG NAME IS `-fa` (short) OR `--flash-attn` (long).
# `--fa` DOES NOT EXIST. Never construct this string by hand.
FLASH_ATTN_FLAGS = ["-fa", "on"]

# KV cache: f16, per B9. DO NOT QUANTISE THE KV CACHE for this model. Quantising it
# costs acceptance directly (alpha 0.8274 -> 0.8166), so the KV saving is paid back
# out of the speculation multiplier — the wrong end of the trade.
KV_CACHE_FLAGS = ["-ctk", "f16", "-ctv", "f16"]

# MTP. OP-35 resolved 2026-09-05: this is part of the model, not an experiment.
# n-max 4 is the CODING-ROLE value per the 2026-09-04 operator ruling (fleet default 3).
# p-min 0.5 is the measured optimum: alpha rises monotonically with p_min
# (0.752 / 0.827 / 0.920 / 0.963 at 0 / 0.5 / 0.75 / 0.9) while drafted-per-token falls
# (0.994 / 0.890 / 0.763 / 0.693), and throughput peaks IN BETWEEN, at 0.5.
MTP_N_MAX = 4
MTP_P_MIN = 0.5
MTP_SPEC_TYPE = "draft-mtp"

# Measured acceptance. TWO values exist and they are DIFFERENT MEASUREMENTS — never
# cross-quote them as one:
#   0.8274 (dpt 0.8900) — speed-claim claim-grade ABA, build 10221, f16 KV, arm B1-3.
#                         Reproduces to 4 dp across all seven n-max-4 arms.
#   0.8209 (dpt 0.8961) — champion-3 / SYNC-15 arms, build 10241. Every MTP arm of
#                         every binary in that window.
# alpha is KV-precision-sensitive: B9 measured 0.8274 -> 0.8166 under quantised KV.
# That is one of the two reasons the KV cache stays f16.
MTP_ALPHA = {
    "speed_claim_b1_3_build10221_kvf16": 0.8274,
    "champion3_build10241_kvf16": 0.8209,
    "b9_quantised_kv": 0.8166,   # the cost of quantising KV — do not do it
}
MTP_FLAGS = [
    "-md", MTP_HEAD_GGUF,
    "--spec-type", MTP_SPEC_TYPE,
    "--spec-draft-n-max", str(MTP_N_MAX),
    "--spec-draft-p-min", str(MTP_P_MIN),
]

# Flag STRINGS this recipe may emit, for the dry-run existence check.
_FLAG_FORMS_TO_VERIFY = [
    ["-fa", "on"],
    ["--flash-attn", "on"],
    ["-ctk", "f16"],
    ["-ctv", "f16"],
    ["--spec-type", MTP_SPEC_TYPE],
    ["--spec-draft-n-max", str(MTP_N_MAX)],
    ["--spec-draft-p-min", str(MTP_P_MIN)],
    ["-t", str(THREADS)],
    ["-c", str(CONTEXT)],
    ["-np", str(PARALLEL_SLOTS)],
    ["--no-mmap"],
    ["--no-webui"],
]

# Forms known NOT to exist. Kept so a launcher can assert it is not about to emit one,
# and so the failure has a name instead of being rediscovered a third time.
KNOWN_BAD_FLAG_FORMS = {
    "--fa": "does not exist; the long form is --flash-attn (SYNC-10 lost 7 MTP arms to this)",
}


# ---------------------------------------------------------------------------
# 6. MEASURED HEADLINE — quote the RATIOS, not the absolutes
# ---------------------------------------------------------------------------
# The host drifts ~3%/hours but repeats to 1.1% within a window, so absolutes are only
# comparable inside the window that produced them. Both entries below are real; they are
# different binaries in different windows and MUST NOT be mixed.
HEADLINES = {
    "champion3": {
        "binary": "build 10241 (9c4f73e29)",
        "window": "2026-09-07, region-lock tag inf70-champion3-confirm",
        "mtp_served_tps": 33.370,          # window 3; 34.433 in window 1
        "plain_served_tps": 20.589,
        "ratio_vs_champion1": 1.0427,      # 117/120 paired per-prompt wins
        "ratio_vs_pristine_mtp": 1.4993,   # 120/120
        "ratio_vs_pristine_plain": 1.7151,
        "alpha": 0.8209,
        "drafted_per_token": 0.8961,
        "coherence": "20 COHERENT + 4 SHORT in all 48 arms; no round or arm excluded",
        "note": "the ~36.9 t/s anchor-referred figure is a PROJECTION, not a measurement",
    },
    "speed_claim_aba": {
        "binary": "build 10221 (c51e4dabf) — pristine + BE-1/BE-2, NOT the champion",
        "window": "2026-09-04, single lock window, ABA x3",
        "mtp_served_tps": 23.16,           # sd 0.14, spread 1.1%, range 23.01-23.26
        "plain_served_tps": 12.35,         # sd 0.24
        "ratio": 1.876,                    # per-round 1.845 / 1.886 / 1.897; B won 20/20 every round
        "alpha": 0.8274,
        "drafted_per_token": 0.8900,
        "note": (
            "CLAIM-GRADE. Supersedes the unreplicated 23.623 / 1.892 single arm. "
            "⚠ These arms passed NO -fa flag (ran on 'auto') and did NOT export "
            "GGML_FA_SPLIT_KV; the handoff's PROD-1 prose adds both retroactively, so "
            "the prose recipe is NOT the configuration that produced 23.16."
        ),
    },
}
# ★ THE FINAL CHARACTERISATION, 2026-09-08. Unit = LAUNCH; precision = between-launch.
# 18 launches, none dropped, all 24/24 rows complete, shim ON verified per launch.
HEADLINES["champion_final_20260908"] = {
    "binary": "ef81196d5 (folded champion) + GGML_NOHUGEPAGE_PROCESS=1 at launch",
    "window": "2026-09-08 15:05:39Z-16:12:47Z, GPU loop down, host exclusive",
    "prereg": "PREREG-FINAL.md frozen 15:05:15Z sha256 1d8f4ddc... BEFORE the lock",
    "unit": "LAUNCH",
    "plain_served_tps": 27.893,        # n=6, between-launch sd 0.609%, CI +/-0.487%
    "mtp_served_tps": 43.281,          # n=6, between-launch sd 0.356%, CI +/-0.285%
    "pristine_plain_tps": 12.762,      # n=3, sd 0.360%
    "pristine_mtp_tps": 23.709,        # n=3, sd 0.926%
    "ratio_vs_pristine_plain": 2.1857, # CI [2.1730, 2.1974]
    "ratio_vs_pristine_mtp": 1.8255,   # CI [1.8081, 1.8399]
    "ratio_mtp_over_plain": 1.5516,    # CI [1.5439, 1.5598]
    "alpha": 0.821,                    # identical on champion and pristine
    # ★ HEADLINE FORM IS A SIGN CLAIM WITH A BOUNDED MAGNITUDE (operator-ruled).
    # Bounds are the LOWER ENDS of 95% bootstrap CIs over launches, not point
    # estimates. Do NOT restate as "2.19x".
    "headline": ("champion beats pristine by >=117% plain and >=81% served-MTP; "
                 "MTP beats plain by >=54%"),
    # ⚠ CAVEAT 1 -- must travel with EVERY ratio.
    "ratio_is_recipe_to_recipe": (
        "NOT knob-controlled. Pristine contains NEITHER THP knob (0 occurrences of "
        "both, no marker), so equal shim state is impossible BY CONSTRUCTION. What IS "
        "controlled: same harness, window, prompt set, server flags, host state, and "
        "adjacent interleaved launches. ★ RETROACTIVE COROLLARY: no champion-vs-"
        "pristine ratio this campaign ever quoted was knob-controlled either -- the "
        "THP difference sat inside all of them, unlabelled."
    ),
    # ⚠ CAVEAT 2 -- must travel with EVERY ratio.
    "open_flag": (
        "CHAMPION-DIVERGENCE.md STAYS OPEN: 2.1857x here vs the standing 1.7151x. Two "
        "conditions differ at once (shim state AND harness/window). Pristine reproduces "
        "across both (12.762 vs 12.366, +3.2%); the champion does not. Adoption explains "
        "PART of the movement and the instability -- NOT asserted to explain all of it."
    ),
    "supersedes": ("1.5149x, +4.50%, 1.7151x, 1.4993x, +4.27%, 33.370 t/s and the "
                   "~36.9 t/s projection -- all champion3 / shim OFF / old harness. "
                   "MARK THEM SUPERSEDED; do not silently replace."),
}

# ★ The prior headlines were measured WITHOUT the adopted knob and on a retired
# harness. Adopting a launch-time knob CHANGES THE MEASURED CONDITION and invalidates
# a recipe's floor until it is re-calibrated -- the same rule cpu_list carries.
HEADLINES["champion3"]["status"] = (
    "SUPERSEDED 2026-09-08 by champion_final_20260908; measured shim OFF on the "
    "retired harness. Retained as the record of a configuration, not as the champion."
)
HEADLINES["speed_claim_aba"]["status"] = (
    "build-10221 number, NEVER a champion number (already flagged); also pre-shim."
)

CANONICAL_HEADLINE = "champion_final_20260908"

# The workload the headline was measured on. Reproducing the number requires this too.
WORKLOAD = {
    "endpoint": "/v1/chat/completions",
    "prompts": "/mnt/raid0/llm/tmp/inf70/agents/e3-alpha/prompts.json",  # 24-prompt production mix
    "max_tokens": 200,
    "temperature": 0.0,                 # greedy
    "cache_prompt": False,
    "chat_template_kwargs": {"enable_thinking": False},
    "client": "/mnt/raid0/llm/tmp/inf70/agents/champion3/client.py",
    "rate": "token-weighted over prompts with pred_n >= 16",
}

# Preconditions the arm script enforces before EVERY arm. A number measured without
# these is not comparable to the headline.
PRECONDITIONS = {
    "region_lock": "region-lock run --cpu-list 0-95 --role bench",
    "page_cache_eviction": "/mnt/raid0/llm/tmp/inf70/evict_nodes_force.sh 58   # GiB per node",
    "load_gate": "wait until 1-min loadavg < 10 (max 300 s)",
    "linkage_proof": "the libggml-cpu.so in /proc/PID/maps must be under the arm's own bindir",
    "placement_proof": "numastat -p PID; max per-node deviation <= 15%, else re-evict and retry",
    "env_readback": "tr '\\0' '\\n' < /proc/PID/environ | grep -E '^(GGML_|OMP_|LLAMA_)'",
    # ⚠ AnonHugePages/Rss is NOT a valid discriminator for the PROCESS shim: it reads
    # 0.06% of Rss at load and ~6% minutes later ON THE SAME PROCESS, so a pass or a
    # fail depends on WHEN you looked. It remains a useful sanity read for the
    # madvise knob (GGML_NOHUGEPAGE), and only for that.
    "thp_readback_madvise_only": "AnonHugePages / Rss from /proc/PID/smaps_rollup (champion ~0.057%)",
    # ★ THE AUTHORITATIVE, TIME-INVARIANT CHECK for CHAMP-2. Read ONCE PER LAUNCH.
    # 0 = prctl in force (shim ON, what the champion recipe wants); 1 = THP allowed.
    # FAIL CLOSED: abort the session on a mismatch. A launch that cannot prove its
    # shim state did not measure the champion.
    "thp_process_shim": (
        "grep ^THP_enabled /proc/PID/status  ==  'THP_enabled:\t0'   # fail-closed, once per LAUNCH"
    ),
}

# ★ Unit register. A floor, a knob and a claim each carry a UNIT, and mixing them is
# the 1200-fold error (arm floor 0.501% vs launch floor 2.793%, ~13x coarser).
MEASUREMENT_UNITS = {
    "GGML_NOHUGEPAGE_PROCESS": "SESSION (process launch)",
    "champion_vs_pristine_ratio": "LAUNCH",
    "arm-scoped GGML_* knobs": "ARM (within one server process)",
}
FLOORS = {
    # (unit, sd %, what it governs)
    "arm_best_case":      ("arm, within session",       0.071, "4 consecutive undisturbed arms"),
    "arm_routine":        ("arm, within session",       0.433, "A/A gate, 5 kept arms"),
    "arm_campaign":       ("arm, within session",       0.501, "arm-scoped knobs"),
    "session":            ("session, between launches", 2.793, "PROCESS-scoped knobs -- e.g. CHAMP-2"),
    "launch_shim_on":     ("launch, shim ON",           0.609, "the adopted recipe (champion plain, n=6)"),
    "launch_shim_off":    ("launch, shim OFF",          5.081, "RETIRED config (n=9, range 12.55%)"),
}


# ---------------------------------------------------------------------------
# 7. BUILDERS
# ---------------------------------------------------------------------------
def build_serve_env(
    bindir: str = CHAMPION_BINDIR,
    extra: Optional[dict] = None,
    base_env: Optional[dict] = None,
) -> dict:
    """Full environment for a champion serve. LD_LIBRARY_PATH is PREPENDED with the
    binary's own directory: three ggml generations live on this host and a binary that
    inherits another tree's ggml runs silently wrong."""
    env = dict(base_env if base_env is not None else os.environ)
    env.update(CANONICAL_OMP_ENV)
    env.update(CHAMPION_GGML_ENV)
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{bindir}:{prev}" if prev else bindir
    if extra:
        env.update(extra)
    return env


def build_serve_command(
    bindir: str = CHAMPION_BINDIR,
    model: str = TRUNK_GGUF,
    mtp: bool = True,
    host: str = "127.0.0.1",
    port: int = 18497,
    extra_flags: Optional[Iterable[str]] = None,
) -> list[str]:
    """The ONLY blessed way to construct a serve command for this model.

    ``mtp=False`` produces the PLAIN CONTROL ARM, not a production configuration:
    OP-35 says production always carries the head.
    """
    cmd = list(SERVE_PREFIX)
    cmd.append(str(Path(bindir) / "llama-server"))
    cmd += BASE_SERVER_FLAGS
    cmd += ["--host", host, "--port", str(port)]
    cmd += ["-m", model]
    cmd += FLASH_ATTN_FLAGS
    cmd += KV_CACHE_FLAGS
    if mtp:
        cmd += MTP_FLAGS
    if extra_flags:
        cmd += list(extra_flags)
    return cmd


# ---------------------------------------------------------------------------
# 8. VALIDATORS
# ---------------------------------------------------------------------------
def assert_canonical_prefix(cmd: list[str]) -> None:
    if cmd[: len(SERVE_PREFIX)] != SERVE_PREFIX:
        raise RecipeViolation(
            f"command must begin with {SERVE_PREFIX!r} (taskset BEFORE numactl); got {cmd[:5]!r}"
        )


def assert_no_bad_flag_forms(cmd: list[str]) -> None:
    for tok in cmd:
        if tok in KNOWN_BAD_FLAG_FORMS:
            raise RecipeViolation(f"{tok}: {KNOWN_BAD_FLAG_FORMS[tok]}")


def assert_mtp_present(cmd: list[str]) -> None:
    """OP-35: the MTP head is part of the model. Fail closed if a serve drops it."""
    missing = [f for f in ("-md", "--spec-type") if f not in cmd]
    if missing:
        raise RecipeViolation(
            f"MTP head missing from serve command ({missing}); OP-35 requires every "
            f"serving path for {RECIPE_ID} to carry it"
        )
    if cmd[cmd.index("--spec-type") + 1] != MTP_SPEC_TYPE:
        raise RecipeViolation(f"--spec-type must be {MTP_SPEC_TYPE}")


def assert_kv_f16(cmd: list[str]) -> None:
    for flag in ("-ctk", "-ctv"):
        if flag in cmd and cmd[cmd.index(flag) + 1] != "f16":
            raise RecipeViolation(f"{flag} must be f16 for this model (B9): do not quantise the KV cache")


def assert_canonical_env(env: dict) -> None:
    for k, v in {**CANONICAL_OMP_ENV, **CHAMPION_GGML_ENV}.items():
        if env.get(k) != v:
            raise RecipeViolation(f"env {k}={env.get(k)!r}, recipe requires {v!r}")
    for knob, (state, how, _why) in CHAMPION_KNOBS.items():
        if how == "leave unset" and knob in env:
            raise RecipeViolation(
                f"env {knob} is set to {env[knob]!r}; the champion state is the COMPILED "
                f"default ({state}) and this knob must be left unset"
            )


def assert_artifacts_exist(model: str = TRUNK_GGUF, head: str = MTP_HEAD_GGUF) -> None:
    """Presence + SIZE. Size is a cheap identity check that catches a truncated or
    swapped artifact without hashing 92 GB; use assert_artifact_digests() when the
    stakes justify the read."""
    for path, want_bytes in ((model, TRUNK_BYTES), (head, MTP_HEAD_BYTES)):
        f = Path(path)
        if not f.is_file():
            raise RecipeViolation(f"artifact missing: {path}")
        got = f.stat().st_size
        if got != want_bytes:
            raise RecipeViolation(f"{path}: {got} bytes, recipe pins {want_bytes}")


def assert_artifact_digests(model: str = TRUNK_GGUF, head: str = MTP_HEAD_GGUF) -> None:
    """Full sha256 of both artifacts. Reads ~95 GB; run it when promoting or when a
    number disagrees with the recipe, not on every serve."""
    for path, want in ((model, TRUNK_SHA256), (head, MTP_HEAD_SHA256)):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            raise RecipeViolation(f"{path}: sha256 {h.hexdigest()} != recipe {want}")


def assert_knob_markers(bindir: str = CHAMPION_BINDIR) -> None:
    """Prove the knob defaults this recipe claims are the ones actually COMPILED IN.

    The champion embeds self-describing marker strings; if a rebuild changes a default
    the marker changes with it, so the recipe cannot silently drift from the binary.
    """
    libs = [str(p) for p in Path(bindir).glob("lib*.so.*") if p.is_file()]
    if not libs:
        raise RecipeViolation(f"no shared libraries found under {bindir}")
    out = subprocess.run(["strings", "-a", *libs], capture_output=True, text=True).stdout
    for marker in CHAMPION_MARKERS:
        if marker not in out:
            raise RecipeViolation(f"marker absent from {bindir}: {marker}")


def assert_binary_identity(bindir: str = CHAMPION_BINDIR) -> None:
    """sha256 every object this recipe pins. A number quoted against a binary that does
    not match these digests is a number about a different kernel."""
    for name, want in CHAMPION_SHA256.items():
        p = Path(bindir) / name
        if not p.is_file():
            raise RecipeViolation(f"missing {p}")
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != want:
            raise RecipeViolation(f"{p}: sha256 {h} != recipe {want}")


def assert_flag_forms_exist(binary: str, forms: Optional[list[list[str]]] = None) -> None:
    """Dry-run every flag form against a REAL llama-server with a nonexistent model.

    This is the check that would have saved SYNC-10's seven arms. It needs no model,
    no lock and no GPU, and costs about a second per flag: argument parsing happens
    before model load, so a bad flag reports `invalid argument` while a good one
    reaches `failed to open GGUF file`.
    """
    bad = []
    for form in forms if forms is not None else _FLAG_FORMS_TO_VERIFY:
        r = subprocess.run(
            [binary, "-m", "/nonexistent-recipe-flagcheck.gguf", *form, "--no-warmup"],
            capture_output=True, text=True, timeout=60,
        )
        blob = r.stdout + r.stderr
        if "invalid argument" in blob or "unrecognized" in blob.lower():
            bad.append(" ".join(form))
    if bad:
        raise RecipeViolation(f"{binary} rejects these flag forms: {bad}")


def preflight(bindir: str = CHAMPION_BINDIR, check_digests: bool = True) -> None:
    """Composite gate. Run this before a serve; it is cheap and needs no lock."""
    assert_artifacts_exist()
    assert_knob_markers(bindir)
    if check_digests:
        assert_binary_identity(bindir)
    assert_flag_forms_exist(str(Path(bindir) / "llama-server"))
    cmd = build_serve_command(bindir=bindir)
    assert_canonical_prefix(cmd)
    assert_no_bad_flag_forms(cmd)
    assert_mtp_present(cmd)
    assert_kv_f16(cmd)
    assert_canonical_env(build_serve_env(bindir=bindir, base_env={}))


def recipe_sha256() -> str:
    """Provenance: a recipe whose content cannot be hashed is not a codified recipe."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    import argparse
    import json
    import shlex

    ap = argparse.ArgumentParser(prog="qwen38_flash_next_recipe")
    ap.add_argument("subcommand",
                    choices=["emit-serve-command", "preflight", "check-flags", "show-knobs"])
    ap.add_argument("--bindir", default=CHAMPION_BINDIR)
    ap.add_argument("--model", default=TRUNK_GGUF)
    ap.add_argument("--port", type=int, default=18497)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--plain", action="store_true",
                    help="drop the MTP head — CONTROL ARM ONLY, never a serving config (OP-35)")
    ap.add_argument("--skip-digests", action="store_true",
                    help="preflight without hashing the binaries")
    a = ap.parse_args(argv[1:])

    if a.subcommand == "emit-serve-command":
        cmd = build_serve_command(bindir=a.bindir, model=a.model, mtp=not a.plain,
                                  host=a.host, port=a.port)
        print(json.dumps({
            "recipe_id": RECIPE_ID,
            "recipe_revision": RECIPE_REVISION,
            "recipe_sha256": recipe_sha256(),
            "mtp": not a.plain,
            "cmd": cmd,
            "env": {**CANONICAL_OMP_ENV, **CHAMPION_GGML_ENV},
            "ld_library_path_prefix": a.bindir,
            "shell": " ".join(shlex.quote(c) for c in cmd),
        }, indent=2))
        return 0
    if a.subcommand == "preflight":
        preflight(a.bindir, check_digests=not a.skip_digests)
        print(f"PREFLIGHT OK  recipe_sha256={recipe_sha256()}")
        return 0
    if a.subcommand == "check-flags":
        assert_flag_forms_exist(str(Path(a.bindir) / "llama-server"))
        print("FLAG FORMS OK")
        return 0
    if a.subcommand == "show-knobs":
        for k, (state, how, why) in sorted(CHAMPION_KNOBS.items()):
            print(f"{k:28s} {state:10s} {how:12s} {why}")
        return 0
    return 2


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv))
