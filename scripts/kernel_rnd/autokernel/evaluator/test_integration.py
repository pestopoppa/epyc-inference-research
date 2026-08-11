#!/usr/bin/env python3
"""test_integration.py — one candidate, end to end, through the assembled AK3 evaluator.

WHY THIS FILE EXISTS
--------------------
`api.py`, `controls.py`, `correctness.py`, `integrity.py`, `recipes.py`,
`statistics.py` and `surface.py` were written in parallel against one interface
and each one passes its own suite. Every defect this file was written to catch
lived BETWEEN two of them, where each module was individually correct and the two
descriptions of the same object did not match:

  * `statistics.PairedBlockReducer` produced `api.EffectEstimate.raw_samples` as
    nested TUPLES (the estimate is a frozen, hashable dataclass); `schemas.
    canonical_json` REFUSES tuples. Every real reduction raised `TypeError` out
    of `content_hash(event)` — the record could not be hashed, journaled or
    emitted — and no unit suite noticed, because each side's own fixture used the
    shape its own module wanted.
  * `controls.ControlPanelResult.definitions_check` — the control-definitions and
    predicate tamper digest — had no field in `api.WindowAttestations`, so *"any
    post-hoc change to … the control definitions"* could not reach
    `api.check_void_conditions` at all.
  * `api.CalibrationOutputs.e_process_construction_id` accepted any string while
    `statistics.CONSTRUCTIONS` is the bundle's registry; three suites' fixtures
    named a construction no reducer implements.
  * `integrity.SourceIntegrityInputs.declared_surface_scope` is a caller-supplied
    string; `surface.AffectedSurface.full_tree` is the DERIVED answer to the same
    question, and nothing compared them.
  * a gate runner returning zero gates derived to `status: pass`.

WHAT THIS SUITE DOES
--------------------
It walks one campaign: acquire a real device claim (on a made-up device id, in a
temp lock root), run the sanctioned preflight substitute against it, resolve the
control bundle, solve the calibration block, derive the affected surface, take
one fixture candidate through the §8.5.1 source-integrity gates and then the T0
correctness gates, reduce a T1 rate comparison against a fixture anchor, compute
the verdict, render the grammar line, emit the evaluation event, journal it, and
assert it validates. Then the negative paths: an anchor-less run is INVALID; a
correctness failure yields no speed rank; a voided window is INVALID with its
reason; a degraded candidate never ranks.

WHAT IT DOES NOT DO
-------------------
NO inference, NO benchmark, NO build, NO model, NO GPU, NO server. No process is
started, stopped or signalled — the only claim acquired is acquired and released
by this process. Every path it writes is under a `tempfile` tree it removes. The
"measurements" are synthetic numbers from `random.Random(seed)`; nothing here is
a measurement of anything, and nothing here may be reported as one.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_integration.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/evaluator/test_integration.py
"""
from __future__ import annotations

import hashlib
import os
import random
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

# Import through the PACKAGE so `api.schemas` is the same module object the
# journal validates with (README, "Import convention").
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import journal as J                    # noqa: E402
from autokernel import schemas as S                    # noqa: E402
from autokernel import storage as STG                  # noqa: E402
from autokernel.evaluator import api                   # noqa: E402
from autokernel.evaluator import controls as CT        # noqa: E402
from autokernel.evaluator import correctness as CO     # noqa: E402
from autokernel.evaluator import devices as DV         # noqa: E402
from autokernel.evaluator import integrity as IG       # noqa: E402
from autokernel.evaluator import statistics as ST      # noqa: E402
from autokernel.evaluator import surface as SU         # noqa: E402
from autokernel.resource import claim_witness as CW    # noqa: E402
from autokernel.resource import device_claim as DC     # noqa: E402
from autokernel.resource import preflight as PF        # noqa: E402

PASS = S.Check(S.PASS)
NOW = "2026-08-03T12:00:00+00:00"
V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
CAMPAIGN = "ak-llama_gpu-decode-20260803"
CAMPAIGN_SEED = "ak-campaign-seed-4711"
DEVICE = "akevaldev0"
CONSTRUCTION_ID = "sign_martingale_predictable_lambda/v1"
WORKTREE = "/mnt/raid0/llm/tmp/ak-campaigns/ak-llama_gpu-decode-20260803/wt-0001"
LIBROOT = f"{WORKTREE}/build/bin"


def sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def fail(*reasons: str) -> S.Check:
    return S.Check(S.FAIL, tuple(reasons) or ("failed",))


# =============================================================================
# A minimal ELF64 writer, so integrity.py's reader has controlled input.
# The evaluator never builds anything; these bytes stand in for the two
# artifacts a real campaign would have compiled.
# =============================================================================

_BIND = {"LOCAL": 0, "GLOBAL": 1, "WEAK": 2}
_TYPE = {"NOTYPE": 0, "OBJECT": 1, "FUNC": 2}
_VIS = {"DEFAULT": 0, "HIDDEN": 2}


def build_elf64(symbols) -> bytes:
    strtab = bytearray(b"\x00")
    offsets = {}
    for name, *_rest in symbols:
        offsets[name] = len(strtab)
        strtab += name.encode("utf-8") + b"\x00"

    syms = bytearray(struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0))
    for name, bind, styp, vis, defined in symbols:
        info = (_BIND[bind] << 4) | _TYPE[styp]
        syms += struct.pack("<IBBHQQ", offsets[name], info, _VIS[vis],
                            1 if defined else 0, 0x1000, 8)

    shstr = bytearray(b"\x00")
    sh_off = {}
    for name in (".dynstr", ".dynsym", ".shstrtab"):
        sh_off[name] = len(shstr)
        shstr += name.encode("ascii") + b"\x00"

    o_strtab = 64
    o_syms = o_strtab + len(strtab)
    o_shstr = o_syms + len(syms)
    o_shdrs = o_shstr + len(shstr)
    ident = b"\x7fELF" + bytes([2, 1, 1, 0, 0]) + b"\x00" * 7
    ehdr = ident + struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, o_shdrs, 0, 64,
                               0, 0, 64, 4, 3)

    def shdr(name_off, sh_type, offset, size, link=0, info=0, entsize=0):
        return struct.pack("<IIQQQQIIQQ", name_off, sh_type, 0, 0, offset, size,
                           link, info, 1, entsize)

    shdrs = b"".join([
        shdr(0, 0, 0, 0),
        shdr(sh_off[".dynstr"], 3, o_strtab, len(strtab)),
        shdr(sh_off[".dynsym"], 11, o_syms, len(syms), link=1, info=1, entsize=24),
        shdr(sh_off[".shstrtab"], 3, o_shstr, len(shstr)),
    ])
    return bytes(ehdr) + bytes(strtab) + bytes(syms) + bytes(shstr) + shdrs


def fn(name: str, *, defined=True, bind="GLOBAL", vis="DEFAULT"):
    return (name, bind, "FUNC", vis, defined)


ABI = [
    fn("ggml_mul_mat"),
    fn("ggml_mul_mat_id"),
    fn("ggml_backend_hip_supports_op"),
    fn("_ZN4ggml6detail15kernel_dispatchILi4EEEvPKfPfi"),
]

OPS_SOURCE = {
    "ggml/src/ggml-cuda/ops.cpp": (
        "GGML_OP_REGISTER(GGML_OP_MUL_MAT, 2);\n"
        "GGML_OP_REGISTER(GGML_OP_MUL_MAT_ID, 3);\n"
    ),
}
DISPATCH_SOURCE = {
    "ggml/src/ggml-cuda/supports.cpp": (
        "bool ggml_backend_hip_supports_op(...) { switch (op) {\n"
        "  case GGML_OP_MUL_MAT: return true;\n"
        "  case GGML_OP_MUL_MAT_ID: return true;\n"
        "} }\n"
    ),
}
OP_EXTRACTOR = IG.PatternRegistrationExtractor(
    kind=IG.KIND_OP_REGISTRATION,
    patterns={"ggml_hip_op_table":
              r"GGML_OP_REGISTER\((?P<key>GGML_OP_[A-Z_0-9]+),\s*(?P<arity>\d+)\)"},
    declared_by="adapter:llama_gpu/v1")
DISPATCH_EXTRACTOR = IG.PatternRegistrationExtractor(
    kind=IG.KIND_DISPATCH_PREDICATE,
    patterns={"ggml_backend_hip_supports_op": r"case\s+(?P<key>GGML_OP_[A-Z_0-9]+):"},
    declared_by="adapter:llama_gpu/v1")

CANDIDATE_DIFF = """\
diff --git a/ggml/src/ggml-cuda/mmq.cuh b/ggml/src/ggml-cuda/mmq.cuh
--- a/ggml/src/ggml-cuda/mmq.cuh
+++ b/ggml/src/ggml-cuda/mmq.cuh
@@ -120,4 +120,6 @@ static __device__ void mmq_id_tile(
     const int lane = threadIdx.x;
-    const int tile = 32;
+    const int tile = (K >= 4096) ? 64 : 32;
+    // wide-tile dispatch for the id path
+    __syncthreads();
     load_tile(tile);
     mma_accumulate();
"""

# The build system's OWN generated dependency information (§6.4 stage 1). Nothing
# here is guessed from a directory prefix: that provenance is refused by name.
DEPFILE = ("CMakeFiles/ggml-hip.dir/ggml-cuda.cu.o: ../ggml/src/ggml-cuda/ggml-cuda.cu "
           "../ggml/include/ggml.h ../ggml/src/ggml-cuda/mmq.cuh\n")
LINKLINE = ("/usr/bin/c++ -O3 CMakeFiles/ggml-hip.dir/ggml-cuda.cu.o "
            "-o bin/llama-server -lamdhip64")


# =============================================================================
# The campaign — assembled once, because the calibration solve is the slow part
# and it is deterministic.
# =============================================================================

def make_blocks(n, *, effect, noise, seed, stratum, split=None, unit_prefix="u",
                start=0, base=100.0, reps=3, schedule=None):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        idx = start + i
        anchor_arm = tuple(base + rng.gauss(0, noise * base) for _ in range(reps))
        a_med = sorted(anchor_arm)[reps // 2]
        cand_arm = tuple(a_med * (1.0 + effect) + rng.gauss(0, noise * base)
                         for _ in range(reps))
        unit = f"{unit_prefix}-{idx}"
        if split is not None:
            while split.assign(unit) != stratum:
                unit += "x"
        order = (schedule.order_for(idx) if schedule is not None
                 else (ST.ORDER_ANCHOR_FIRST if idx % 2 == 0
                       else ST.ORDER_CANDIDATE_FIRST))
        out.append(ST.PairedBlock(
            block_index=idx, unit_id=unit, stratum=stratum, order=order,
            anchor_samples=anchor_arm, candidate_samples=cand_arm, measured_at=NOW))
    return tuple(out)


def _run_candidate(stats, split, *, candidate_id, effect, noise, seed,
                   stratum=api.STRATUM_SELECTION):
    """Drive one candidate through the pre-committed stopping rule.

    Nothing is measured: each "block" is two synthetic arms from
    `random.Random(seed)`. What IS real is the control flow — the rule decides
    how many blocks there are, which order each runs in, and when to stop.
    """
    seq = stats.sequential_evaluation(candidate_id=candidate_id, stratum=stratum,
                                      metric_direction="higher_better")
    rng = random.Random(seed)
    blocks = []
    while not seq.terminal:
        req = seq.next_block_request()
        anchor_arm = tuple(100.0 + rng.gauss(0, noise * 100.0) for _ in range(3))
        a_med = sorted(anchor_arm)[1]
        cand_arm = tuple(a_med * (1.0 + effect) + rng.gauss(0, noise * 100.0)
                         for _ in range(3))
        unit = f"sel-{req.block_index}"
        while split.assign(unit) != stratum:
            unit += "x"
        block = ST.PairedBlock(
            block_index=req.block_index, unit_id=unit, stratum=stratum,
            order=req.order, anchor_samples=anchor_arm, candidate_samples=cand_arm,
            segment=req.segment, extension_round=req.extension_round,
            measured_at=NOW)
        seq.submit_block(block)
        blocks.append(block)
    return seq, tuple(blocks)


def _anchor_calibration_values(n=200, seed=3, base=100.0, sd=1.0):
    """The anchor cell's own calibration values — output 4's material."""
    rng = random.Random(seed)
    return tuple(base + rng.gauss(0, sd) for _ in range(n))


class Campaign:
    """Everything fixed before the first candidate is measured."""

    _cache = None

    @classmethod
    def get(cls):
        if cls._cache is not None:
            return cls._cache

        controls_decl = api.CampaignControls(
            calibration_block_count=200, contribution_floor=0.10, max_candidates=10,
            confirmation_admission_count=2, max_blocks_per_candidate=20,
            storage_floor_bytes_free=200 * 1024 ** 3)
        rule = ST.StoppingRule(
            rule_id="ak-stop-1/v1", final_table="t1_paired_block_table",
            decisions=(("evidence_threshold_crossed", "compose_into_champion_lineage"),
                       ("extension_exhausted", "abandon"),
                       ("block_ceiling_reached", "abandon")),
            extension=ST.BoundedExtension(max_rounds=1, blocks_per_round=5),
            max_blocks_per_candidate=20)
        construction = ST.select_construction(CONSTRUCTION_ID)
        inputs = ST.CalibrationInputs(
            backend="llama_gpu", phase="decode", cell_class="instrument_tokens_per_s",
            campaign_seed=CAMPAIGN_SEED, controls=controls_decl, stopping_rule=rule,
            construction=construction, effect_scale=ST.EFFECT_SCALE_RELATIVE,
            metric_direction="higher_better", hypothesis=ST.HYPOTHESIS_IMPROVEMENT,
            margin=0.0,
            aa_blocks=make_blocks(200, effect=0.0, noise=0.01, seed=1,
                                  stratum=api.STRATUM_SELECTION, unit_prefix="aa"),
            neutral_blocks=make_blocks(60, effect=0.0, noise=0.01, seed=2,
                                       stratum=api.STRATUM_SELECTION, unit_prefix="nt"),
            anchor_calibration_values=_anchor_calibration_values(),
            samples_ref="ak-raw://ak-llama_gpu-decode-20260803/calibration/0001")
        solve = ST.solve_calibration(inputs)
        outputs = solve.require_accepted()
        commitment = ST.StoppingRuleCommitment.commit(rule, campaign_id=CAMPAIGN,
                                                      committed_at=NOW)
        split = ST.StratumSplitRule(
            rule_id="ak-split-1/v1", campaign_seed=CAMPAIGN_SEED,
            confirmation_fraction=0.3,
            rotation=ST.RotationSchedule(schedule_id="ak-rot-1/v1", period_campaigns=4))
        stats = ST.CampaignStatistics(
            campaign_id=CAMPAIGN, campaign_seed=CAMPAIGN_SEED,
            effect_scale=ST.EFFECT_SCALE_RELATIVE,
            hypothesis=ST.HYPOTHESIS_IMPROVEMENT, margin=0.0, stopping_rule=rule,
            stopping_rule_commitment=commitment, split_rule=split,
            construction=construction, calibration=outputs,
            aa_effect_pool=solve.aa_effect_pool,
            anchor_calibration_values=solve.anchor_calibration_values)
        cls._cache = (controls_decl, rule, construction, inputs, solve, outputs,
                      commitment, split, stats)
        return cls._cache


def anchor(**overrides) -> api.AnchorIdentity:
    kwargs = dict(source_commit=V8_COMMIT, binary_sha256=sha("anchor-binary"),
                  linkage_sha256=sha("anchor-linkage"),
                  measurement_event_ids=("ake-anchor-0001",))
    kwargs.update(overrides)
    return api.AnchorIdentity(**kwargs)


# =============================================================================
# The scenario
# =============================================================================

class EndToEndScenario(unittest.TestCase):
    """One candidate, from device claim to journaled record.

    Assembled once in `setUpClass` so each assertion inspects the SAME run rather
    than a fresh one that might differ. Every negative path below rebuilds only
    the one thing it breaks.
    """

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ak3-eval-integration-")
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        tmp = Path(cls.tmp)

        (controls_decl, rule, construction, cal_inputs, solve, calibration,
         commitment, split, stats) = Campaign.get()
        cls.campaign_controls = controls_decl
        cls.stopping_rule = rule
        cls.calibration = calibration
        cls.solve = solve
        cls.split = split
        cls.stats = stats

        # --- 1. artifacts the evaluator READS but never builds -----------------
        cls.anchor_so = tmp / "anchor.so"
        cls.cand_so = tmp / "candidate.so"
        cls.anchor_so.write_bytes(build_elf64(ABI))
        cls.cand_so.write_bytes(build_elf64(ABI + [fn("mmq_id_tile_v2")]))
        cls.anchor_syms = IG.extract_elf_symbols(cls.anchor_so, label="anchor")
        cls.cand_syms = IG.extract_elf_symbols(cls.cand_so, label="candidate")

        # --- 2. the resource claim: ACQUIRED, never inferred (precondition 1) ---
        cls.lock_root = os.path.join(cls.tmp, "locks")
        os.makedirs(cls.lock_root, exist_ok=True)
        cls.claim_journal = DC.ClaimJournal(os.path.join(cls.tmp, "claims.jsonl"))
        cls.claim = DC.acquire_device_claim(
            DEVICE, purpose="AK3 evaluator integration scenario", campaign_id=CAMPAIGN,
            journal=cls.claim_journal, holder_label="ak3-integration",
            lock_root=cls.lock_root, timeout_s=5.0)
        cls.claim_receipt_id = cls.claim.claim_id
        cls.held_at_open = DC.check_device_claim_held(cls.claim.receipt(),
                                                      lock_root=cls.lock_root)

        # --- 3. preflight, WITH the claim held (precondition 2) -----------------
        cls.journal = J.Journal(os.path.join(cls.tmp, "journal"), campaign_id=CAMPAIGN)
        cls.journal.initialize()
        scope = PF.PreflightScope.gpu("ak3-integration-decode", [DEVICE])
        sources = CW.gpu_claim_sources([DEVICE], lock_root=cls.lock_root)
        cls.preflight = PF.require_no_concurrent_inference(scope, sources)
        cls.preflight_entry = cls.journal.append_preflight_attestation(cls.preflight)
        cls.preflight_attestation_ref = (cls.preflight_entry.record_id
                                         or cls.preflight_entry.event_id)
        # ... and the same preflight with NO claim reader must not manufacture a PASS.
        cls.blind_preflight = PF.claim_witness_preflight(
            scope, PF.ClaimSources(region_lock_dir=Path(cls.lock_root)))

        # --- 4. the control bundle, hash-pinned (Controls) ----------------------
        cls.control_bundle = CT.resolve_control_bundle(
            pinned_definitions_digest=CT.CONTROL_DEFINITIONS_DIGEST,
            aa_cadence=CT.AACadence(every_n_windows=5, every_n_seconds=3600.0,
                                    declared_at=NOW),
            seed_rotation=CT.SeedRotationSchedule(rotate_every_windows=10,
                                                  declared_at=NOW),
            historical_win_replays=(),
            source_label="evaluator-bundle@ak3-integration")

        # --- 5. stage 1: the DERIVED affected surface (§6.4) --------------------
        index = SU.build_dependency_index(
            label="candidate", build_dir="build-hip", source_root="/repo/llama.cpp",
            dep_edges=SU.parse_make_depfile(DEPFILE, origin_ref="ggml-hip.d"),
            link_edges=[SU.parse_cmake_link_txt(LINKLINE,
                                                origin_ref="server/link.txt")],
            backend_link_targets={"llama_gpu": ["bin/llama-server"]})
        surface_diff = SU.SourceDiff(
            base_commit=V8_COMMIT, candidate_commit="b" * 40,
            entries=(SU.DiffEntry(path="ggml/src/ggml-cuda/mmq.cuh",
                                  change_kind="modified"),),
            origin_ref="git diff --name-status")
        cls.derived_surface = SU.derive_affected_surface(
            candidate_id="akc-0001", diff=surface_diff, indexes=(index,),
            change_class="dispatcher")

        # --- 6. §8.5.1 source-integrity inputs ---------------------------------
        cls.integrity_inputs = IG.SourceIntegrityInputs(
            candidate_id="akc-0001", backend="llama_gpu", change_class="dispatcher",
            artifact_binary_sha256=cls.cand_syms.file_sha256,
            anchor_symbols=cls.anchor_syms, candidate_symbols=cls.cand_syms,
            signature_index={
                "ggml_mul_mat": {"anchor": 3, "candidate": 3},
                "ggml_mul_mat_id": {"anchor": 4, "candidate": 4},
                "ggml_backend_hip_supports_op": {"anchor": 2, "candidate": 2},
            },
            anchor_registrations=(OP_EXTRACTOR.extract_text("anchor", OPS_SOURCE),
                                  DISPATCH_EXTRACTOR.extract_text("anchor",
                                                                  DISPATCH_SOURCE)),
            candidate_registrations=(OP_EXTRACTOR.extract_text("candidate", OPS_SOURCE),
                                     DISPATCH_EXTRACTOR.extract_text("candidate",
                                                                     DISPATCH_SOURCE)),
            declared_symbol_deltas=IG.DeclaredSymbolDeltas(
                added=frozenset({"mmq_id_tile_v2"}), removed=frozenset(),
                arity_changed=frozenset()),
            declared_surface=IG.DeclaredSurface(
                files=frozenset({"ggml/src/ggml-cuda/mmq.cuh"}),
                symbols=frozenset({"mmq_id_tile_v2"})),
            # DERIVED by surface.py, not typed here: `surface_scope_for` is the
            # projection, and the runner below verifies it against the manifest.
            declared_surface_scope=IG.surface_scope_for(cls.derived_surface),
            diff=IG.parse_unified_diff(CANDIDATE_DIFF),
            envelope=IG.ChangeClassEnvelope(
                change_class="dispatcher", max_files_touched=3, max_changed_lines=200,
                max_hunks=10, max_file_shrinkage_ratio=0.60,
                allows_file_creation=False, allows_file_deletion=False,
                allows_pure_deletion_hunks=False,
                declared_by="adapter:llama_gpu/v1"),
            complexity_ceiling=IG.ComplexityCeiling(
                backend="llama_gpu", max_diff_lines=150, max_files_touched=4,
                shared_core_modification_requires_review=True,
                declared_by="adapter:llama_gpu/v1"),
            core_header_policy=IG.CoreHeaderPolicy(
                core_path_prefixes=("ggml/include", "ggml/src/ggml.c"),
                core_path_globs=("ggml/src/*.h",),
                backends_served=("llama_cpu", "llama_gpu"),
                declared_by="adapter:llama/v1"),
            original_line_counts={"ggml/src/ggml-cuda/mmq.cuh": 900},
            build=IG.BuildProvenance(
                candidate_id="akc-0001", snapshot_sha256=sha("cand-source"),
                source_root=f"{WORKTREE}/snapshot",
                build_dir=f"{WORKTREE}/build",
                build_dir_created_for_this_build=True,
                build_dir_pre_build_digest=IG.EMPTY_TREE_SHA256,
                actor_worktree=f"{WORKTREE}/actor",
                production_tree_paths=("/mnt/raid0/llm/llama.cpp",
                                       "/mnt/raid0/llm/whisper.cpp"),
                toolchain="rocm-6.2", compiler="hipcc 6.2.0",
                command="cmake --build . -j 32",
                build_log_path=f"{WORKTREE}/build/build.log",
                build_log_sha256=sha("build-log"),
                output_binary_sha256=cls.cand_syms.file_sha256,
                incremental_output_binary_sha256=None),
            snapshot_recompute_root=None,
            snapshot_attested_by="storage.verify_durability:ak-llama_gpu-decode-20260803",
            repair=None)

        # --- 7. the runners, composed §8.5.1-first (§8.6) ----------------------
        cls.integrity_runner = IG.SourceIntegrityGateRunner(
            tier="T0", inputs_by_candidate={"akc-0001": cls.integrity_inputs},
            derived_surfaces={"akc-0001": cls.derived_surface})
        cls.t0_policy = CO.T0Policy(
            required_backend_ops=("MUL_MAT", "MUL_MAT_ID"),
            symbol_shrinkage_reject_ratio=0.02,
            diff_ceiling=CO.DiffComplexityCeiling(
                backend="llama_gpu", max_changed_lines=400, max_files_touched=8,
                shared_core_forces_review=True),
            determinism_min_runs=3, coherence_tolerance_floor=0.995,
            policy_ref="evaluator-bundle://t0/policy/llama_gpu/v1")
        cls.correctness_runner = CO.T0CorrectnessRunner(
            provider=CO.StaticEvidenceProvider({"akc-0001": cls.t0_evidence()}),
            policy=cls.t0_policy)
        cls.t0_runner = IG.SourceIntegrityFirstRunner(
            integrity=cls.integrity_runner, behavioural=cls.correctness_runner)

        # --- 8. T0: source integrity, then correctness -------------------------
        cls.t0_request = cls.request(tier="T0", calibration=None)
        cls.t0_window = cls.window()
        cls.t0_dispatcher = api.TierDispatcher(gate_runners={"T0": cls.t0_runner})
        cls.t0_outcome = cls.t0_dispatcher.dispatch(cls.t0_request, cls.t0_window,
                                                    effect=None)
        cls.t0_entry = cls.journal.append(J.KIND_EVALUATION_EVENT, cls.t0_outcome.event)

        # --- 9. T1: the rate comparison against the fixture anchor -------------
        cls.reducer = ST.PairedBlockReducer(cls.stats)
        cls.t1_request = cls.request(tier="T1", calibration=cls.calibration)
        # The blocks are produced by asking the PRE-COMMITTED stopping rule for
        # each one — `next_block_request()` is the only way to obtain another
        # block, so the order, the segment and the extension round on every block
        # come from the rule rather than from this fixture.
        cls.sequential, cls.blocks = _run_candidate(
            cls.stats, cls.split, candidate_id="akc-0001", effect=0.062,
            noise=0.008, seed=99)
        cls.reduction = cls.reducer.reduce(cls.t1_request, cls.blocks)
        cls.effect = cls.reducer.reduce_blocks(cls.t1_request, cls.blocks)

        # --- 10. the control panel for this window -----------------------------
        cls.control_result = cls.run_controls()
        cls.t1_window = cls.window(
            **CT.window_control_attestations(cls.control_result),
            **cls.reduction.window_checks)
        cls.t1_runner = _T1Runner("T1")
        cls.t1_dispatcher = api.TierDispatcher(gate_runners={"T1": cls.t1_runner})
        cls.t1_outcome = cls.t1_dispatcher.dispatch(cls.t1_request, cls.t1_window,
                                                    effect=cls.effect)
        cls.t1_entry = cls.journal.append(J.KIND_EVALUATION_EVENT, cls.t1_outcome.event)

        # --- 11. window close: the claim is STILL held, by the same holder ------
        cls.held_at_close = DC.check_device_claim_held(cls.claim.receipt(),
                                                       lock_root=cls.lock_root)
        cls.claim.release()
        cls.held_after_release = DC.check_device_claim_held(cls.claim.receipt(),
                                                            lock_root=cls.lock_root)

    # -- fixture builders ---------------------------------------------------

    @classmethod
    def t0_evidence(cls, **overrides) -> CO.T0Evidence:
        surface = CO.ChangeSurface(
            derived_touches_memory=True, derived_touches_threading=False,
            derived_touches_dispatch=True, derived_touches_persistent_state=False,
            derived_ops=("MUL_MAT_ID",),
            derived_files=(f"{WORKTREE}/ggml/src/ggml-cuda/mmq.cuh",),
            declared_touches_memory=True, declared_touches_threading=False,
            declared_ops=("MUL_MAT_ID",), touches_shared_core_header=False,
            derivation_ref="ake-derivation-0001")
        kwargs = dict(
            control_role=None,
            change_surface=surface,
            symbols=CO.SymbolTableDiff(
                removed_symbols=(), arity_changed_symbols=(),
                added_symbols=("mmq_id_tile_v2",), removed_op_registrations=(),
                removed_dispatch_predicates=(), declared_removals=(),
                anchor_symbol_count=4, candidate_symbol_count=5,
                tool_id="nm -D --defined-only",
                receipt_ref="data/ak/akc-0001/symbols.json", produced_by="evaluator"),
            build=CO.BuildProvenance(
                built_from_snapshot_sha256=sha("cand-source"),
                build_dir=f"{WORKTREE}/build", build_dir_was_fresh=True,
                incremental_objects_present=False, compiler_id="hipcc",
                compiler_version="6.2.0", build_log_ref="data/ak/akc-0001/build.log",
                production_tree_paths_touched=(),
                output_binary_sha256=cls.cand_syms.file_sha256,
                produced_by="evaluator"),
            diff=CO.DiffPolicyEvidence(
                files_touched=(f"{WORKTREE}/ggml/src/ggml-cuda/mmq.cuh",),
                declared_surface_files=(f"{WORKTREE}/ggml/src/ggml-cuda/mmq.cuh",),
                unrelated_deletions=(), changed_lines=118, change_class="dispatcher",
                envelope=CO.ChangeClassEnvelope(change_class="dispatcher",
                                                max_changed_lines=300,
                                                max_files_touched=4),
                branch_name="llama.cpp-experimental/ak-mmq-id-tile",
                commit_was_pathspec_limited=True, production_tree_paths=(),
                record_schema_violations=(), diff_ref="data/ak/akc-0001/diff.patch",
                produced_by="evaluator"),
            static_analysis=CO.StaticAnalysisEvidence(
                compiler_id="hipcc", compiler_version="6.2.0",
                anchor_compiler_id="hipcc", anchor_compiler_version="6.2.0",
                error_count=0, warning_count=0, anchor_warning_count=0,
                anchor_source_commit=V8_COMMIT,
                anchor_binary_sha256=cls.anchor_syms.file_sha256,
                anchor_linkage_sha256=sha("anchor-linkage"),
                warnings_as_errors=True, analyzer_id="clang-tidy-18",
                analyzer_error_findings=(),
                receipt_ref="data/ak/akc-0001/static.json", produced_by="evaluator"),
            sanitizers=CO.SanitizerEvidence(
                invocation=CO.build_sanitizer_invocation(
                    source_dir=WORKTREE, build_dir=f"{WORKTREE}/build-asan",
                    target="test-backend-ops",
                    run_argv=(f"{WORKTREE}/build-asan/bin/test-backend-ops",
                              "-o", "MUL_MAT_ID"),
                    jobs=8, backend="llama_gpu"),
                executed=True, exit_code=0, asan_findings=(), ubsan_findings=(),
                sanitizer_build_binary_sha256=sha("cand-binary-asan"),
                log_ref="data/ak/akc-0001/sanitizer.log", produced_by="evaluator"),
            op_suite=CO.OpSuiteEvidence(
                suite_id="test-backend-ops", suite_source_sha256=sha("cand-source"),
                suite_seed=4711,
                ops_exercised=("MUL_MAT", "MUL_MAT_ID"), ops_failed=(),
                cases_by_op=(("MUL_MAT", 4231, 4231), ("MUL_MAT_ID", 1188, 1188)),
                shapes_ref="data/ak/akc-0001/shapes.json",
                receipt_ref="data/ak/akc-0001/tbo.json", produced_by="evaluator"),
            reference=CO.ReferenceEvidence(
                comparisons=(
                    CO.ReferenceComparison(
                        shape_id="m4096n1k4096-q4_K", op="MUL_MAT",
                        mode="exact_bitwise", mismatch_count=0, max_ulp_observed=None,
                        tolerance_ulp=None, oracle_id="ik_llama.cpp@iqk-ref",
                        oracle_is_candidate_derived=False),
                    CO.ReferenceComparison(
                        shape_id="e128t1k4096-q4_K", op="MUL_MAT_ID",
                        mode="ulp_bounded", mismatch_count=0, max_ulp_observed=1.0,
                        tolerance_ulp=2.0, oracle_id="ik_llama.cpp@iqk-ref",
                        oracle_is_candidate_derived=False)),
                undefined_for=(), oracle_registry_ref="evaluator-bundle://oracles/v1",
                produced_by="evaluator"),
            boundary_shapes=CO.BoundaryShapeEvidence(
                unseen_shapes=("m1n1k4096", "m8191n7k4096"),
                boundary_shapes=("m0n0k0", "m1n2048k1"), failures=(),
                selection_rule_id="ak.holdout.shape_partition/v1",
                selection_seed=CAMPAIGN_SEED, held_out_from_planner=True,
                receipt_ref="data/ak/akc-0001/boundary.json", produced_by="evaluator"),
            dispatch_trace=CO.DispatchTraceEvidence(
                derived_surface=("MUL_MAT_ID", "mmq_id_tile", "mmq_id_tile_v2"),
                traced_kernels=("MUL_MAT_ID", "mmq_id_tile_v2"), fallback_events=(),
                fallback_instrumentation_active=True,
                trace_ref="data/ak/akc-0001/dispatch.jsonl", produced_by="evaluator"),
            state_safety=CO.StateSafetyEvidence(
                rollback_tested=True, teardown_tested=True, race_detector_id="tsan",
                race_findings=(), leaked_resources=(), orphan_processes=(),
                receipt_ref="data/ak/akc-0001/state.json", produced_by="evaluator"),
            coherence=CO.CoherenceEvidence(
                candidate_output_sha256=sha("gen-out"), candidate_output_len=160,
                anchor_output_sha256=sha("gen-out"), anchor_output_len=160,
                sampler_id="greedy-topk1-temp0", sampler_is_greedy=True, seed=42,
                tokens_requested=160, token_agreement_ratio=1.0,
                divergence_first_index=None, anchor_determinism_class="bitwise_stable",
                anchor_source_commit=V8_COMMIT,
                anchor_binary_sha256=cls.anchor_syms.file_sha256,
                anchor_linkage_sha256=sha("anchor-linkage"),
                prompt_ref="evaluator-bundle://prompts/coherence/v1",
                receipt_ref="data/ak/akc-0001/coherence.json", produced_by="evaluator"),
            determinism=CO.DeterminismEvidence(
                seed=42, runs=3, candidate_output_digests=(sha("gen-out"),) * 3,
                anchor_output_digests=(sha("gen-out"),) * 3,
                anchor_determinism_class="bitwise_stable",
                anchor_source_commit=V8_COMMIT,
                anchor_binary_sha256=cls.anchor_syms.file_sha256,
                anchor_linkage_sha256=sha("anchor-linkage"),
                declared_class_change=False, declared_class_change_ref=None,
                receipt_ref="data/ak/akc-0001/determinism.json",
                produced_by="evaluator"),
            linkage=CO.LinkageEvidence(
                binary_sha256=cls.cand_syms.file_sha256,
                linkage_sha256=sha("cand-linkage"),
                anchor_source_commit=V8_COMMIT,
                anchor_binary_sha256=cls.anchor_syms.file_sha256,
                anchor_linkage_sha256=sha("anchor-linkage"),
                resolved_libraries=(("libggml-base.so", f"{LIBROOT}/libggml-base.so",
                                     sha("ggml-base")),
                                    ("libggml-hip.so", f"{LIBROOT}/libggml-hip.so",
                                     sha("ggml-hip"))),
                expected_library_root=LIBROOT, verifier_id="verify_ggml_linkage.sh",
                receipt_ref="data/ak/akc-0001/linkage.json", produced_by="evaluator"),
            anti_reward_hacking=CO.AntiRewardHackingEvidence(
                cache_state="cold", correctness_verdict_source="evaluator",
                candidate_output_used_as_oracle=False,
                oracle_ids=("ik_llama.cpp@iqk-ref",),
                delivered_unit_name="generated_tokens", delivered_units_candidate=160,
                delivered_units_anchor=160,
                anchor_source_commit=V8_COMMIT,
                anchor_binary_sha256=cls.anchor_syms.file_sha256,
                anchor_linkage_sha256=sha("anchor-linkage"),
                environment_probe_findings=(),
                timing_dependent_branch_findings=(),
                receipt_ref="data/ak/akc-0001/integrity.json",
                environment_probe_detector_id="environment-probe/v1",
                timing_dependent_branch_detector_id="timing-branch/v1",
                stream_creation_detector_id="stream-creation/v1",
                async_escape_detector_id="async-escape/v1",
                instrument_frame_detector_id="instrument-frame/v1",
                pointer_memoization_detector_id="pointer-memoization/v1",
                structured_short_circuit_detector_id="structured-short-circuit/v1"),
        )
        kwargs.update(overrides)
        return CO.T0Evidence(**kwargs)

    @classmethod
    def request(cls, **overrides) -> api.EvaluationRequest:
        kwargs = dict(
            event_id="ake-t1-0001", campaign_id=CAMPAIGN, candidate_id="akc-0001",
            tier="T1", backend="llama_gpu", phase="decode",
            cell_class="instrument_tokens_per_s", protocol_id=api.PROTOCOL_VERSIONED_ID,
            artifact=api.ArtifactIdentity(source_sha256=sha("cand-source"),
                                          binary_sha256=cls.cand_syms.file_sha256,
                                          linkage_sha256=sha("cand-linkage")),
            anchor=anchor(binary_sha256=cls.anchor_syms.file_sha256),
            evaluator=api.EvaluatorIdentity(
                id="P-AK-SEARCH-1/v1", bundle_sha256=sha("evaluator-bundle"),
                runtime_source_label_ref="ake-srclabel-0003"),
            scope_denominator=api.ScopeDenominator(
                machine_subset="partial", numa_nodes=(), devices=("mi210_0",), cores=8),
            scope_manifest_sha256=sha("scope-manifest"), co_residency="single",
            determinism=api.DeterminismReport(determinism_class="bitwise_stable",
                                              same_seed_repeat_runs=3),
            metric="decode_tokens_per_s", metric_direction="higher_better", reps=10,
            change_class="parameter", anchor_tier="T1",
            transfer_ratio_to=(),
            created_at=NOW, campaign_controls=cls.campaign_controls,
            calibration=cls.calibration,
            device_state=DV.DeviceState(
                device_id="mi210_0", source="fixture/rocm-smi",
                nominal_sclk_mhz=1700, min_sclk_ratio=0.9,
                samples=(DV.DeviceStateSample(1700, 1600, 180, 55, True),),
                receipt_ref="fixture://device-state/integration"))
        kwargs.update(overrides)
        if "anchor_tier" not in overrides:
            kwargs["anchor_tier"] = kwargs["tier"]
        if kwargs["tier"] == "T0":
            kwargs.setdefault("event_id", "ake-t0-0001")
        return api.EvaluationRequest(**kwargs)

    @classmethod
    def window(cls, **overrides) -> api.WindowAttestations:
        a = anchor(binary_sha256=cls.anchor_syms.file_sha256)
        kwargs = dict(
            resource_claim_receipt=cls.claim_receipt_id,
            resource_claim_open=PASS, resource_claim_close=PASS,
            resource_claim_same_holder=PASS, no_concurrent_inference=PASS,
            preflight_attestation_ref=cls.preflight_attestation_ref,
            host_receipt="host-health-20260803T1159Z", host_health=PASS,
            anchor_at_open=a, anchor_at_close=a, anchor_gate=PASS,
            evaluator_bundle=PASS, runtime_source_label=PASS,
            recipe=api.RecipeReceipt(
                constructor_id="ak.microbench.llama_gpu.decode/v1",
                constructor_sha256=sha("recipe-constructor"), argv_sha256=sha("argv")),
            storage_open=PASS, storage_close=PASS, strata=PASS,
            stopping_rule_id=cls.stopping_rule.rule_id, rule_immutability=PASS,
            order_randomized=PASS, order_seed=f"{CAMPAIGN_SEED}:akc-0001",
            aa_cadence=PASS,
            controls=api.ControlPanel(positive=PASS, neutral=PASS,
                                      degraded_negative=PASS, aa=PASS,
                                      historical_replay=None,
                                      historical_replay_unavailable_reason=(
                                          "llama_gpu declares no historical_win_replay "
                                          "entry in this campaign manifest"),
                                      operator_escalation_ref="ake-op-escalation-0001"),
            calibration=PASS, control_definitions_immutable=PASS,
            raw_evidence_ref="data/ak-llama_gpu-decode-20260803/raw/akc-0001/")
        kwargs.update(overrides)
        return api.WindowAttestations(**kwargs)

    @classmethod
    def run_controls(cls) -> CT.ControlPanelResult:
        """Evaluate the five controls from fixture verdicts. Nothing is measured."""
        harness = CT.ControlHarness(bundle=cls.control_bundle,
                                    runner=_NullControlRunner())
        # llama_gpu declares no historical-win entry in this manifest: the
        # UNAVAILABLE branch, with the operator's call on the record.
        historical = CT.HistoricalWinResolution(
            backend="llama_gpu", available=False, declaration=None,
            check=S.Check(S.COULD_NOT_CHECK,
                          ("the campaign manifest declares no historical_win_replay "
                           "entry for llama_gpu",)),
            marker=CT.HISTORICAL_REPLAY_UNAVAILABLE)
        escalation = CT.OperatorEscalation(
            escalation_ref="ake-op-escalation-0001", raised_at="2026-08-03T09:00:00Z",
            decision=CT.OPERATOR_DECISION_PROCEED_ON_FOUR,
            decided_at="2026-08-03T10:00:00Z", decided_by="operator")
        context = CT.ControlContext(
            campaign_id=CAMPAIGN, backend="llama_gpu", phase="decode",
            cell_class="instrument_tokens_per_s", window_id="akw-0001",
            historical=historical,
            neutral_dispersion=CT.neutral_dispersion_check(cls.solve),
            calibration=cls.calibration)
        observations = (
            CT.ControlObservation(control_id=CT.CONTROL_POSITIVE, ran=True,
                                  verdict=_control_verdict(value=0.31)),
            # A neutral control's true effect is centred on zero: it MEASURES
            # something, and what it measures does not advance.
            CT.ControlObservation(control_id=CT.CONTROL_NEUTRAL, ran=True,
                                  verdict=_control_verdict(value=0.0012),
                                  abs_effects=(0.001, 0.002)),
            CT.ControlObservation(control_id=CT.CONTROL_DEGRADED_NEGATIVE, ran=True,
                                  verdict=_degraded_verdict()),
            # "An A/A that measured nothing is not an A/A that found nothing."
            CT.ControlObservation(control_id=CT.CONTROL_AA, ran=True,
                                  verdict=_control_verdict(value=0.0008)),
        )
        return harness.evaluate(observations=observations, context=context,
                                aa_cadence=PASS, escalation=escalation,
                                pinned_definitions_digest=CT.CONTROL_DEFINITIONS_DIGEST)

    # =====================================================================
    # THE HAPPY PATH
    # =====================================================================

    def test_the_claim_was_acquired_and_held_across_the_whole_window(self):
        """Precondition 1: ACQUIRED, never inferred; re-verified at close."""
        self.assertEqual(self.held_at_open.outcome, S.PASS, self.held_at_open.reasons)
        self.assertEqual(self.held_at_close.outcome, S.PASS, self.held_at_close.reasons)
        # And once released it is NOT held — the check is real, not a constant.
        self.assertNotEqual(self.held_after_release.outcome, S.PASS)
        # The receipt identifier travels onto every record.
        self.assertEqual(self.t1_outcome.event["resource_claim_receipt"],
                         self.claim_receipt_id)
        resolved = CW.check_event_claim_receipt(self.t1_outcome.event,
                                                self.claim_journal)
        self.assertEqual(resolved.outcome, S.PASS, resolved.reasons)

    def test_preflight_passed_with_the_claim_and_could_not_check_without_it(self):
        """Precondition 2: the sanctioned substitute, and its fail-closed twin."""
        self.assertEqual(self.preflight.verdict, "PASS", self.preflight.findings)
        self.assertNotEqual(self.blind_preflight.verdict, "PASS")
        self.assertTrue(self.preflight_attestation_ref)
        self.assertEqual(self.preflight_entry.kind,
                         J.KIND_PREFLIGHT_ATTESTATION)

    def test_the_calibration_block_was_solved_in_the_normative_order(self):
        """Campaign calibration block: every threshold derived, none supplied."""
        self.assertTrue(self.solve.accepted)
        self.assertEqual(self.calibration.solve_order_recorded,
                         api.CALIBRATION_SOLVE_ORDER)
        self.assertGreaterEqual(self.calibration.b_min_blocks, 5)
        self.assertLessEqual(self.calibration.alpha_sel,
                             self.campaign_controls.alpha_sel_ceiling())
        self.assertLessEqual(self.calibration.alpha_conf,
                             self.campaign_controls.alpha_conf_ceiling(
                                 self.calibration.alpha_sel))
        self.assertEqual(self.calibration.e_process_construction_id, CONSTRUCTION_ID)
        self.assertIn(self.calibration.e_process_construction_id,
                      api.E_PROCESS_CONSTRUCTION_IDS)
        # Both the failed and the accepted attempts are retained.
        self.assertTrue(self.solve.attempts)

    def test_the_declared_surface_scope_was_derived_not_typed(self):
        """§6.4: the affected surface is a DERIVED manifest, never a declaration."""
        self.assertEqual(self.integrity_inputs.declared_surface_scope,
                         IG.surface_scope_for(self.derived_surface))
        binding = next(g for g in self.integrity_runner.run_gates(self.t0_request)
                       if g.gate_id == IG.GATE_SURFACE_SCOPE_BINDING)
        self.assertEqual(binding.check.outcome, S.PASS, binding.check.reasons)
        self.assertTrue(self.integrity_runner.surface_binding)

    def test_t0_ran_source_integrity_before_any_behavioural_gate(self):
        """§8.6: the source-integrity gates run before any behavioural check."""
        ids = [g.gate_id for g in self.t0_outcome.verdict.gates]
        self.assertEqual(ids[:len(IG.RUNNER_GATE_IDS)], list(IG.RUNNER_GATE_IDS))
        self.assertIn(IG.GATE_SURFACE_SCOPE_BINDING, ids)
        for gate_id in CO.T0_GATE_IDS:
            self.assertIn(gate_id, ids)
        self.assertNotIn(IG.GATE_BEHAVIOURAL_NOT_RUN, ids)

    def test_t0_passed_every_gate_and_produced_a_valid_event(self):
        outcome = self.t0_outcome
        failing = [(g.gate_id, g.check.outcome, g.check.reasons)
                   for g in outcome.verdict.gates if g.check.outcome != S.PASS]
        self.assertEqual(failing, [])
        self.assertEqual(outcome.verdict.status, api.STATUS_PASS)
        self.assertEqual(outcome.verdict.integrity_flags, ())
        self.assertTrue(outcome.emitted)
        self.assertEqual(outcome.event_violations, ())
        self.assertEqual(S.validate_evaluation_event(outcome.event), [])
        # A T0 record is not a rate comparison, so it carries no speed rank.
        self.assertFalse(outcome.verdict.speed_rank_admissible)

    def test_the_t1_reduction_is_admissible_and_publishes_its_mde(self):
        """Statistical requirements: e-value, threshold, MDE and floor, together."""
        self.assertEqual(self.reduction.admissible.outcome, S.PASS,
                         self.reduction.admissible.reasons)
        self.assertIsNotNone(self.effect)
        # "never fewer than the calibrated B_min", and never more than the
        # declared ceiling: the count came from the rule, not from this fixture.
        self.assertGreaterEqual(self.effect.paired_blocks,
                                self.calibration.b_min_blocks)
        self.assertLessEqual(self.effect.paired_blocks,
                             self.stopping_rule.max_blocks_per_candidate)
        self.assertEqual(self.effect.paired_blocks, len(self.blocks))
        self.assertEqual(self.effect.threshold,
                         self.calibration.threshold_for(api.STRATUM_SELECTION))
        self.assertEqual(self.effect.noise_floor, self.calibration.noise_floor_phi)
        self.assertTrue(self.reduction.mde.found)
        self.assertGreater(self.effect.mde, 0.0)
        # The MDE is a function of the CALIBRATION material and the campaign's
        # DECLARED window only: the candidate's own blocks — including how many
        # of them there turned out to be — are not an input to it.
        #
        # `b_min`, not `effect.paired_blocks`. `solve_mde`'s `block_count` is the
        # BASE SEGMENT length and it adds `max_rounds * blocks_per_round` on top
        # to build its resampling windows, so passing the realized count asks for
        # the MDE of a window the stopping rule cannot license — optimistic by
        # 18.8% (selection) and 43.5% (confirmation) on the execution layer's own
        # campaign. Latent until runs started pooling their declared extension
        # round, because until then `len(blocks) == b_min`. See
        # `PairedBlockReducer`'s own docstring.
        again = self.reducer.mde_for(self.reducer.campaign.b_min,
                                     stratum=api.STRATUM_SELECTION,
                                     metric_direction="higher_better")
        self.assertEqual(again.value, self.reduction.mde.value)
        if self.effect.paired_blocks != self.reducer.campaign.b_min:
            realized = self.reducer.mde_for(self.effect.paired_blocks,
                                            stratum=api.STRATUM_SELECTION,
                                            metric_direction="higher_better")
            self.assertNotEqual(
                realized.value, self.reduction.mde.value,
                "this run realized more blocks than B_min, so the two MDEs are "
                "different numbers and the test above is load-bearing")

    def test_the_t1_verdict_is_a_ranked_improvement_and_the_event_validates(self):
        outcome = self.t1_outcome
        self.assertEqual(outcome.verdict.status, api.STATUS_PASS,
                         outcome.verdict.derivation)
        self.assertTrue(outcome.verdict.search_grade.satisfied,
                        outcome.verdict.search_grade.failed)
        self.assertEqual(outcome.verdict.effect_resolution, api.EFFECT_IMPROVEMENT)
        self.assertTrue(outcome.verdict.speed_rank_admissible)
        key = outcome.verdict.rank_key()
        self.assertGreater(key[0], 0.0)
        self.assertTrue(outcome.emitted)
        self.assertEqual(outcome.event_violations, ())
        self.assertEqual(S.validate_evaluation_event(outcome.event), [])
        self.assertEqual(outcome.states[-1], "EMITTED")

    def test_the_record_is_canonicalizable_hashable_and_journaled(self):
        """The seam the reducer's nested TUPLES used to break."""
        # `content_hash` on the emitted event is what `dispatch()` already did;
        # asserting it here names the failure if the shape ever regresses.
        self.assertTrue(S.content_hash(self.t1_outcome.event))
        self.assertEqual(self.t1_outcome.record_content_hash,
                         S.content_hash(self.t1_outcome.event))
        raw = self.t1_outcome.event["performance"]["raw_samples"]
        self.assertIsInstance(raw, list)
        self.assertTrue(all(isinstance(row, list) for row in raw))
        self.assertTrue(S.content_hash(self.t1_outcome.durable_payload))
        # It is in the journal, and it comes back out.
        # Keyed by event id, not by position: other tests in this class append
        # their own INVALID records, and a positional assertion would make this
        # test depend on the order unittest happens to run them in.
        by_id = {e.payload["event_id"]: e for e in self.journal.read_all()
                 if e.kind == J.KIND_EVALUATION_EVENT}
        self.assertIn(self.t0_request.event_id, by_id)
        self.assertIn(self.t1_request.event_id, by_id)
        self.assertEqual(by_id[self.t1_request.event_id].payload["status"],
                         api.STATUS_PASS)
        self.assertEqual(by_id[self.t0_request.event_id].payload["status"],
                         api.STATUS_PASS)

    def test_the_grammar_line_carries_every_field_the_protocol_makes_mandatory(self):
        line = api.render_search_record_grammar(
            request=self.t1_request, window=self.t1_window,
            verdict=self.t1_outcome.verdict, effect=self.effect)
        for token in ("SEARCH RECORD, NOT A CLAIM", "P-AK-SEARCH-1",
                      "category=CANDIDATE", "tier T1", "vs anchor",
                      f"blocks={self.effect.paired_blocks}", "e=", "thr=", "MDE=",
                      "floor=", "stratum=selection", "det=bitwise_stable", "scope=",
                      "controls=4/5 (HISTORICAL_REPLAY_UNAVAILABLE)",
                      f"campaign={CAMPAIGN}", "eval=", "srclabel=", "recipe=",
                      f"res={self.claim_receipt_id}", "host=", "raw=", "2026-08-03"):
            with self.subTest(field=token):
                self.assertIn(token, line)
        self.assertNotIn("attest=", line)

    def test_the_control_panel_travelled_into_the_window_by_projection(self):
        """`controls.window_control_attestations` is the seam, not a hand copy."""
        projected = CT.window_control_attestations(self.control_result)
        self.assertEqual(set(projected),
                         {"controls", "aa_cadence", "control_definitions_immutable"})
        self.assertIs(self.t1_window.controls, projected["controls"])
        self.assertEqual(self.t1_window.control_definitions_immutable.outcome, S.PASS)
        self.assertEqual(self.control_result.marker,
                         "4/5 (HISTORICAL_REPLAY_UNAVAILABLE)")
        self.assertTrue(self.control_result.may_rank)
        self.assertEqual(self.control_result.gate_defects, ())

    def test_the_reduction_projected_its_own_window_attestations(self):
        """`statistics.BlockReduction.window_checks` is the mirror-image seam."""
        checks = self.reduction.window_checks
        self.assertEqual(set(checks),
                         {"strata", "rule_immutability", "order_randomized",
                          "calibration"})
        for name, chk in checks.items():
            with self.subTest(field=name):
                self.assertEqual(chk.outcome, S.PASS, chk.reasons)
                self.assertEqual(getattr(self.t1_window, name).outcome, S.PASS)

    def test_the_reduction_is_reproducible_from_the_samples_on_the_record(self):
        """Record grammar: a reduction that cannot be recomputed is INVALID."""
        chk = ST.verify_reduction_reproducible(self.effect, self.reducer,
                                               self.t1_request)
        self.assertEqual(chk.outcome, S.PASS, chk.reasons)
        # An estimate whose recorded samples no longer hash to its own
        # content-addressed ref does not reproduce, and says so.
        tampered = ST.api.EffectEstimate(
            metric=self.effect.metric, metric_direction=self.effect.metric_direction,
            value=self.effect.value, e_value=self.effect.e_value,
            threshold=self.effect.threshold, mde=self.effect.mde,
            noise_floor=self.effect.noise_floor,
            paired_blocks=self.effect.paired_blocks, stratum=self.effect.stratum,
            raw_samples=self.effect.raw_samples[:-1] + (
                self.effect.raw_samples[-1][:-1] + ((9999.0, 9999.0, 9999.0),),),
            raw_samples_ref=self.effect.raw_samples_ref)
        self.assertNotEqual(
            ST.verify_reduction_reproducible(tampered, self.reducer,
                                             self.t1_request).outcome, S.PASS)

    def test_the_storage_floor_has_exactly_one_definition(self):
        """Precondition 7 / "There is no fifth output"."""
        self.assertIn("storage_floor_bytes_free",
                      api.CampaignControls.__dataclass_fields__)
        self.assertNotIn("storage_floor_bytes_free",
                         api.CalibrationOutputs.__dataclass_fields__)
        # The floor's OWNER is the evidence-retention clause of MEASUREMENT.md §5,
        # implemented by `storage.py`. The evaluator declares the quantity and
        # attests the check; it does not define a second floor of its own.
        self.assertTrue(hasattr(STG, "DISK_PRESSURE"),
                        "storage.py is the module that owns the campaign floor")
        self.assertEqual(self.t1_window.storage_open.outcome, S.PASS)
        self.assertEqual(self.t1_window.storage_close.outcome, S.PASS)

    # =====================================================================
    # THE NEGATIVE PATHS
    # =====================================================================

    def test_an_anchorless_run_is_invalid_and_emits_an_anchorless_record(self):
        """Precondition 4: a run without an explicit anchor is INVALID."""
        request = self.request(tier="T1", anchor=None)
        outcome = self.t1_dispatcher.dispatch(request, self.t1_window,
                                              effect=self.effect)
        self.assertEqual(outcome.verdict.status, api.STATUS_INVALID)
        self.assertIn(api.VOID_ANCHOR_MISSING_OR_MUTATED, outcome.void_scan.reasons())
        # ... and it is still DURABLE. A voided run is never silently discarded.
        self.assertTrue(S.content_hash(outcome.durable_payload))
        self.assertIn("NO-ANCHOR", outcome.grammar_line)
        with self.assertRaises(api.SpeedRankUnavailable):
            outcome.verdict.rank_key()
        # v3: it is a valid RECORD as well, with the anchor block absent rather
        # than invented, so the journal can hold it as primary evidence.
        self.assertTrue(outcome.emitted)
        self.assertIsNone(outcome.event_blocked_reason)
        self.assertEqual(outcome.event_violations, ())
        self.assertNotIn("anchor", outcome.event)
        self.assertEqual(outcome.event["status"], api.STATUS_INVALID)

    def test_the_anchorless_void_reaches_the_journal_as_a_primary_record(self):
        """The seam the two halves of one protocol sentence used to fall between.

        The evaluator refused to fabricate an anchor digest (correct) and
        `evaluation_event.v2` required one (also correct on its own terms), so
        *"A voided run is journaled as INVALID with its reason"* had no
        implementation for the ANCHOR-MISSING void: `Journal.append` rejected the
        only record that case can produce. This walks the whole path — dispatch,
        emit, append, read back — because a validator call alone would not have
        caught it.
        """
        request = self.request(tier="T1", anchor=None)
        outcome = self.t1_dispatcher.dispatch(request, self.t1_window,
                                              effect=self.effect)
        entry = self.journal.append(J.KIND_EVALUATION_EVENT, {
            **outcome.event, "event_id": "ake-void-anchor-missing"})
        stored = {e.payload["event_id"]: e for e in self.journal.read_all()
                  if e.kind == J.KIND_EVALUATION_EVENT}["ake-void-anchor-missing"]
        self.assertEqual(stored.payload["status"], api.STATUS_INVALID)
        self.assertNotIn("anchor", stored.payload)
        self.assertIn(f"VOID:{api.VOID_ANCHOR_MISSING_OR_MUTATED}:{S.FAIL}",
                      stored.payload["integrity_flags"])
        self.assertEqual(S.validate_record(stored.payload), [])
        self.assertEqual(entry.payload, stored.payload)
        # The reason is retrievable without the prose, and the anchor checker
        # still reports the record for what it is: a ratio with no denominator.
        self.assertEqual(S.check_anchor_binding(stored.payload).outcome, S.FAIL)

    def test_a_correctness_failure_yields_no_speed_rank(self):
        """Correctness precedence: no speed rank at all — not a penalised one."""
        broken = CO.T0CorrectnessRunner(
            provider=CO.StaticEvidenceProvider({"akc-0001": self.t0_evidence(
                op_suite=CO.OpSuiteEvidence(
                    suite_id="test-backend-ops",
                    suite_source_sha256=sha("cand-source"),
                    suite_seed=4711,
                    ops_exercised=("MUL_MAT",), ops_failed=(),
                    cases_by_op=(("MUL_MAT", 4231, 4231),),
                    shapes_ref="s", receipt_ref="r", produced_by="evaluator"))}),
            policy=self.t0_policy)
        dispatcher = api.TierDispatcher(gate_runners={"T1": _CorrectnessAsT1(broken)})
        outcome = dispatcher.dispatch(self.t1_request, self.t1_window,
                                      effect=self.effect)
        self.assertEqual(outcome.verdict.status, api.STATUS_FAIL)
        self.assertFalse(outcome.verdict.speed_rank_admissible)
        with self.assertRaises(api.SpeedRankUnavailable):
            outcome.verdict.rank_key()
        self.assertIn("no speed rank at all",
                      outcome.verdict.speed_rank_withheld_reason())
        ranked, unrankable = api.rank_candidates([outcome.verdict])
        self.assertEqual(ranked, ())
        self.assertEqual(len(unrankable), 1)
        # The failure is still journalable, with the failing gate named.
        self.assertTrue(any(f.startswith("CORRECTNESS:") for f in
                            outcome.verdict.integrity_flags))
        self.assertEqual(S.validate_evaluation_event(outcome.event), [])

    def test_a_voided_window_is_invalid_with_its_reason_and_still_journaled(self):
        """"A voided run is journaled as INVALID with its reason, and is never
        silently discarded."" """
        for label, overrides, reason in (
                ("failed A/A", dict(controls=api.ControlPanel(
                    positive=PASS, neutral=PASS, degraded_negative=PASS,
                    aa=fail("the A/A control resolved a significant effect"),
                    historical_replay=PASS)), api.VOID_AA_CONTROL_FAILED),
                ("anchor gate", dict(anchor_gate=fail("anchor median outside the band")),
                 api.VOID_ANCHOR_GATE_FAILED),
                ("hand-typed argv", dict(recipe=None), api.VOID_HAND_TYPED_ARGV),
                ("control definitions", dict(control_definitions_immutable=fail(
                    "CONTROL_PREDICATES_DIGEST no longer matches the pinned bundle")),
                 api.VOID_POST_HOC_RULE_CHANGE),
        ):
            with self.subTest(void=label):
                win = self.window(**{**CT.window_control_attestations(
                    self.control_result), **self.reduction.window_checks, **overrides})
                outcome = self.t1_dispatcher.dispatch(self.t1_request, win,
                                                      effect=self.effect)
                self.assertEqual(outcome.verdict.status, api.STATUS_INVALID)
                self.assertIn(reason, outcome.void_scan.reasons())
                finding = next(f for f in outcome.void_scan.findings
                               if f.reason == reason)
                self.assertEqual(finding.protocol_phrase,
                                 api.VOID_REASON_PHRASES[reason])
                # INVALID is not FAIL: a voided window says nothing about the
                # candidate, so it is never recorded as a candidate failure.
                self.assertNotEqual(outcome.verdict.status, api.STATUS_FAIL)
                self.assertFalse(outcome.verdict.speed_rank_admissible)
                if reason != api.VOID_HAND_TYPED_ARGV:
                    # `recipe=None` also empties a mandatory grammar field, which
                    # blocks emission; the rest still emit an INVALID record.
                    self.assertTrue(outcome.emitted)
                    self.assertEqual(outcome.event["status"], api.STATUS_INVALID)
                    entry = self.journal.append(J.KIND_EVALUATION_EVENT, {
                        **outcome.event,
                        "event_id": f"ake-void-{reason.lower()}"})
                    self.assertEqual(entry.payload["status"], api.STATUS_INVALID)

    def test_a_degraded_candidate_never_ranks_however_fast_it_looks(self):
        """Control 3: deliberately fast-looking but wrong => no speed rank at all."""
        # A candidate that silently falls back to the generic path, with a huge
        # apparent gain. `mmq_id_tile_v2` executed nothing; the generic kernel did.
        degraded = CO.T0CorrectnessRunner(
            provider=CO.StaticEvidenceProvider({"akc-0001": self.t0_evidence(
                dispatch_trace=CO.DispatchTraceEvidence(
                    derived_surface=("MUL_MAT_ID", "mmq_id_tile_v2"),
                    traced_kernels=("MUL_MAT_ID", "mmq_id_tile_v2"),
                    fallback_events=("mmq_id_tile_v2 -> ggml_cuda_mul_mat_id_generic",),
                    fallback_instrumentation_active=True,
                    trace_ref="data/ak/akc-0001/dispatch.jsonl",
                    produced_by="evaluator"))}),
            policy=self.t0_policy)
        huge = api.EffectEstimate(
            metric="decode_tokens_per_s", metric_direction="higher_better",
            value=0.94, e_value=1e6,
            threshold=self.calibration.threshold_for(api.STRATUM_SELECTION),
            mde=self.effect.mde, noise_floor=self.calibration.noise_floor_phi,
            paired_blocks=self.effect.paired_blocks, stratum=api.STRATUM_SELECTION,
            raw_samples=self.effect.raw_samples,
            raw_samples_ref=self.effect.raw_samples_ref)
        dispatcher = api.TierDispatcher(gate_runners={"T1": _CorrectnessAsT1(degraded)})
        outcome = dispatcher.dispatch(self.t1_request, self.t1_window, effect=huge)
        self.assertEqual(outcome.verdict.status, api.STATUS_FAIL)
        self.assertFalse(outcome.verdict.speed_rank_admissible)
        with self.assertRaises(api.SpeedRankUnavailable):
            outcome.verdict.rank_key()
        # And it is ABSENT from the ranking with a stated reason, not sorted last.
        ranked, unrankable = api.rank_candidates(
            [self.t1_outcome.verdict, outcome.verdict])
        self.assertEqual([v.tier for v in ranked], ["T1"])
        self.assertEqual(len(ranked), 1)
        self.assertIs(ranked[0], self.t1_outcome.verdict)
        self.assertEqual(len(unrankable), 1)
        self.assertIn("no speed rank at all", unrankable[0][1])

    def test_a_below_floor_estimate_does_not_rank_however_strong_the_evidence(self):
        """Calibration output 1: MUST NOT be ranked, banked, or composed."""
        below = api.EffectEstimate(
            metric="decode_tokens_per_s", metric_direction="higher_better",
            value=self.calibration.noise_floor_phi * 0.5, e_value=1e9,
            threshold=self.calibration.threshold_for(api.STRATUM_SELECTION),
            mde=1e-9, noise_floor=self.calibration.noise_floor_phi,
            paired_blocks=self.effect.paired_blocks, stratum=api.STRATUM_SELECTION,
            raw_samples=self.effect.raw_samples,
            raw_samples_ref=self.effect.raw_samples_ref)
        outcome = self.t1_dispatcher.dispatch(self.t1_request, self.t1_window,
                                              effect=below)
        self.assertEqual(outcome.verdict.effect_resolution,
                         api.EFFECT_BELOW_NOISE_FLOOR)
        self.assertFalse(outcome.verdict.speed_rank_admissible)

    def test_a_reduction_below_b_min_is_refused_and_rides_on_the_exception(self):
        """The reducer never answers None for a non-conforming run."""
        short = self.blocks[:2]
        with self.assertRaises(ST.ReductionInadmissible) as ctx:
            self.reducer.reduce_blocks(self.t1_request, short)
        self.assertEqual(ctx.exception.reduction.check("block_count").outcome, S.FAIL)
        self.assertIsNone(ctx.exception.reduction.estimate)
        # The full reduction is journalable as INVALID with its reason.
        self.assertTrue(S.content_hash(ctx.exception.reduction.to_dict()))

    def test_a_blocked_design_is_caught_by_order_control(self):
        """Statistical requirements: blocked designs are forbidden."""
        blocked = tuple(
            ST.PairedBlock(block_index=b.block_index, unit_id=b.unit_id,
                           stratum=b.stratum, order=ST.ORDER_ANCHOR_FIRST,
                           anchor_samples=b.anchor_samples,
                           candidate_samples=b.candidate_samples,
                           measured_at=b.measured_at)
            for b in self.blocks)
        reduction = self.reducer.reduce(self.t1_request, blocked)
        self.assertEqual(reduction.check("order_control").outcome, S.FAIL)
        self.assertIsNone(reduction.estimate)

    def test_a_release_tier_is_refused_at_wiring_time_and_at_dispatch(self):
        """Scope: it does NOT apply to T3 or any release gate."""
        with self.assertRaises(api.TierNotOwned):
            api.TierDispatcher(gate_runners={"T3": _T1Runner("T3")})
        t3_request = self.request(tier="T3")
        with self.assertRaises(api.TierNotOwned):
            self.t1_dispatcher.dispatch(t3_request, self.t1_window, effect=self.effect)

    def test_an_unwired_tier_and_an_empty_gate_list_both_refuse(self):
        """An unrun tier with no gate results would derive to PASS."""
        with self.assertRaises(api.EvaluatorNotWired):
            self.t1_dispatcher.dispatch(self.request(tier="T2"), self.t1_window)
        empty = api.TierDispatcher(gate_runners={"T1": _T1Runner("T1", gates=())})
        with self.assertRaises(api.EvaluatorNotWired):
            empty.dispatch(self.t1_request, self.t1_window, effect=self.effect)

    def test_source_integrity_failure_blocks_the_behavioural_gates(self):
        """§8.6: a behavioural PASS on an unverified binary is worse than no result."""
        mismatched = IG.SourceIntegrityGateRunner(
            tier="T0",
            inputs_by_candidate={"akc-0001": self.integrity_inputs},
            derived_surfaces={"akc-0001": self.derived_surface})
        composed = IG.SourceIntegrityFirstRunner(integrity=mismatched,
                                                 behavioural=self.correctness_runner)
        # An anchor the evidence is not evidence FOR: the binding gate blocks.
        bad_request = self.request(tier="T0", anchor=anchor(binary_sha256=sha("other")))
        gates = composed.run_gates(bad_request)
        ids = [g.gate_id for g in gates]
        self.assertIn(IG.GATE_BEHAVIOURAL_NOT_RUN, ids)
        self.assertNotIn(CO.GID_OP_UNITS, ids)
        blocked = next(g for g in gates if g.gate_id == IG.GATE_BEHAVIOURAL_NOT_RUN)
        self.assertEqual(blocked.check.outcome, S.COULD_NOT_CHECK)

    def test_the_journal_is_internally_consistent_after_the_whole_scenario(self):
        entries = self.journal.read_all()
        self.assertTrue(entries)
        views = J.rebuild_views(entries)
        consistency = J.check_view_consistency(entries, views)
        self.assertEqual(consistency.outcome, S.PASS, consistency.reasons)


# =============================================================================
# Fixture runners — they run NOTHING. They hand back what they were given.
# =============================================================================

class _T1Runner:
    """A T1 gate runner. The rate comparison itself is the reducer's job; the
    gates here are the mechanism/performance surfaces a real T1 runner reports."""

    def __init__(self, tier: str, gates=None) -> None:
        self.tier = tier
        self._gates = (
            (api.GateResult(gate_id="t1.mechanism_prediction",
                            gate_class=api.GATE_MECHANISM, check=PASS,
                            notes=("wide-tile dispatch reduces id-path launches",)),
             api.GateResult(gate_id="t1.cell_scope_matches_manifest",
                            gate_class=api.GATE_STABILITY, check=PASS))
            if gates is None else tuple(gates))

    def run_gates(self, request):
        return self._gates


class _CorrectnessAsT1:
    """Runs the T0 correctness set under a T1 request, so a correctness failure
    can be observed on a record that also carries a rate comparison."""

    def __init__(self, runner) -> None:
        self.tier = "T1"
        self._runner = runner

    def run_gates(self, request):
        t0_request = EndToEndScenario.request(
            tier="T0", event_id="ake-t0-probe", calibration=None,
            anchor=request.anchor)
        return tuple(self._runner.run_gates(t0_request)) + (
            api.GateResult(gate_id="t1.mechanism_prediction",
                           gate_class=api.GATE_MECHANISM, check=PASS),)


class _NullControlRunner:
    """The `ControlRunner` seam, unimplemented on purpose.

    Running a control means driving the candidate pipeline for its fixture, which
    is inference. This suite supplies control OBSERVATIONS directly and never
    calls `run_all`, so the runner exists only to satisfy the harness's type gate
    — and it raises rather than returning a fabricated observation.
    """

    runner_id = "ak3-integration-null-runner"

    def run_control(self, definition, context):
        raise NotImplementedError(
            "this suite runs no control: a control run drives the candidate "
            "pipeline, which is inference. Observations are supplied directly.")


def _control_verdict(*, value: float) -> api.Verdict:
    """A minted `api.Verdict` for a control observation. Nothing is measured here:
    the numbers are fixtures, and a real control run is the `ControlRunner` seam."""
    scenario_anchor = anchor()
    effect = api.EffectEstimate(
        metric="decode_tokens_per_s", metric_direction="higher_better",
        value=value, e_value=5000.0, threshold=100.0,
        mde=0.02, noise_floor=0.009, paired_blocks=10,
        stratum=api.STRATUM_SELECTION, raw_samples=((100.0, 100.0 + 100.0 * value),),
        raw_samples_ref="ak-raw://control")
    return api.compute_verdict(
        tier="T1", gates=(api.GateResult(gate_id="t0.control", gate_class=api.GATE_CORRECTNESS,
                                         check=PASS),),
        void_scan=api.VoidScan(findings=(), evaluated=api.VOID_REASONS,
                               not_applicable=()),
        search_grade=api.SearchGradeResult(
            satisfied=True, evaluated=tuple(c.id for c in api.SEARCH_GRADE_CONJUNCTS),
            failed=(), not_applicable=(), reasons=()),
        anchor=scenario_anchor, effect=effect)


def _degraded_verdict() -> api.Verdict:
    """The degraded-negative control's verdict: a correctness FAIL, no rank."""
    return api.compute_verdict(
        tier="T1",
        gates=(api.GateResult(gate_id=CO.GID_NO_FALLBACK,
                              gate_class=api.GATE_INTEGRITY,
                              check=fail("1 fallback dispatch observed")),),
        void_scan=api.VoidScan(findings=(), evaluated=api.VOID_REASONS,
                               not_applicable=()),
        search_grade=api.SearchGradeResult(
            satisfied=True, evaluated=tuple(c.id for c in api.SEARCH_GRADE_CONJUNCTS),
            failed=(), not_applicable=(), reasons=()),
        anchor=anchor(),
        effect=api.EffectEstimate(
            metric="decode_tokens_per_s", metric_direction="higher_better",
            value=0.90, e_value=1e6, threshold=100.0, mde=0.02, noise_floor=0.009,
            paired_blocks=10, stratum=api.STRATUM_SELECTION,
            raw_samples=((100.0, 190.0),), raw_samples_ref="ak-raw://control-3"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
