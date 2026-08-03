#!/usr/bin/env python3
"""test_composition.py — the regression barrier for AK4 champion maintenance (§8.9).

WHY THIS FILE EXISTS
--------------------
Every property below is one of §8.9's sentences turned into a failing test, and
each of them is a way the champion model can be quietly wrong:

  * **composition inferred from member results.** `compose_champion()` accepts no
    member evidence, demands the COMBINED candidate's own passing T0 and T1, and
    the module contains no arithmetic that could combine two measurements. All
    three are asserted, including from the module's own AST.
  * **an unreconciled surface combined anyway.** The reconciliation verdict comes
    from `evaluator/surface`, is bound to the candidate record, and a
    COULD_NOT_CHECK reconciliation is not a reconciled one.
  * **the search collapsing to one family.** Retention fills a per-class quota
    before it fills by preference, and a capacity below the floor is a refusal.
  * **a champion crossing a source tree.** `llama_cpu` and `llama_gpu` share one
    champion (§1.5); `whisper_stt` cannot join it; `serving_runtime` has no
    champion at all.
  * **an anchor that moved.** Composition refuses, comparisons are superseded
    carrying both identities, T0 and candidate records survive, the re-anchor
    cannot shortcut re-measurement, and the operator notice is the four-part
    package `state_machine.check_stop_evidence` accepts.

NO inference, NO benchmark, NO build, NO model call, NO process. Every file this
suite writes lives under a per-test temporary directory.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/controller/test_composition.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/controller/test_composition.py
"""
from __future__ import annotations

import dataclasses
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

# Import through the PACKAGE so `composition.schemas` is the same module object
# the journal validates with (README, "Import convention").
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import journal as J  # noqa: E402
from autokernel import schemas as S  # noqa: E402
from autokernel.controller import composition as C  # noqa: E402
from autokernel.controller import state_machine as SM  # noqa: E402
from autokernel.evaluator import surface as SF  # noqa: E402

V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
V9_COMMIT = "1122334455667788990011223344556677889900"
V7_COMMIT = "6ad45fa3ff6718c07c000061dbc6e29c1771f6e3"
CAMPAIGN = "ak-llama_gpu-decode-20260803"
ANCHOR_BINARY = "anchor-binary"
ANCHOR_LINKAGE = "anchor-linkage"


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


# =============================================================================
# Fixtures — records and surfaces, never measurements
# =============================================================================

def _anchor(*, commit: str = V8_COMMIT, backends=("llama_cpu", "llama_gpu"),
            binary: str = ANCHOR_BINARY, linkage: str = ANCHOR_LINKAGE,
            tree: str = "llama.cpp",
            branch: str = "production-consolidated-v8") -> SM.AnchorIdentity:
    return SM.AnchorIdentity(
        source_tree=tree,
        branch=branch,
        commit=commit,
        binary_sha256={b: _sha(f"{binary}-{b}") for b in backends},
        linkage_sha256={b: _sha(f"{linkage}-{b}") for b in backends},
    )


def _derived(candidate_id: str, backends=("llama_gpu",), *,
             coverage=None, extra_symbol: str = "") -> SF.AffectedSurface:
    backends = tuple(sorted(backends))
    return SF.AffectedSurface(
        candidate_id=candidate_id,
        backends=backends,
        link_targets=("libggml-hip.so",),
        objects=("ggml-hip/mmq.o",),
        touched_files=("ggml/src/ggml-hip/mmq.hip",),
        symbols=("ggml_hip_mul_mat_q",) + ((extra_symbol,) if extra_symbol else ()),
        op_registrations=tuple(
            SF.OpRegistration(op_name="MUL_MAT", backend=b, dispatch_predicate="K>=4096")
            for b in backends
        ),
        dispatch_predicates=("K>=4096",),
        over_approximations=(),
        axes_derived=SF.SURFACE_AXES,
        coverage=coverage if coverage is not None else S.Check(S.PASS),
        full_tree=False,
        inputs={"diff_ref": f"diff-{candidate_id}"},
    )


def _traced(candidate_id: str, backends=("llama_gpu",)) -> SF.TracedSurface:
    events = tuple(
        SF.DispatchEvent(op_name="MUL_MAT", backend=b,
                         kernel_symbol="ggml_hip_mul_mat_q",
                         link_target="libggml-hip.so",
                         dispatch_predicate="K>=4096")
        for b in sorted(backends)
    )
    return SF.TracedSurface(
        candidate_id=candidate_id,
        trace_ref=f"trace-{candidate_id}",
        events=events,
        truncated=False,
        completeness=S.Check(S.PASS),
        no_fallback=S.Check(S.PASS),
    )


def _reconciliation(candidate_id: str, backends=("llama_gpu",), *,
                    traced: bool = True, escape: bool = False):
    derived = _derived(candidate_id, backends)
    if not traced:
        return SF.reconcile_surface(derived, None)
    trace_backends = tuple(backends) + (("whisper_stt",) if escape else ())
    return SF.reconcile_surface(derived, _traced(candidate_id, trace_backends))


def _candidate(candidate_id: str, reconciliation, *, status: str = "banked",
               base: str = V8_COMMIT, champion_status: str = "frontier") -> dict:
    block = SF.candidate_affected_surface_block(reconciliation)
    tag = candidate_id.rsplit("-", 1)[-1]
    return {
        "schema": S.SCHEMA_CANDIDATE,
        "candidate_id": candidate_id,
        "campaign_id": CAMPAIGN,
        "proposal_id": f"akp-20260803-{tag}",
        "parent_candidate_id": None,
        "worktree": {
            "path": "/mnt/raid0/llm/llama.cpp-ak-llama_gpu-decode-20260803",
            "branch": f"ak/{CAMPAIGN}/akp-{tag}",
            "source_commit": V7_COMMIT,
            "clean": True,
        },
        "source_snapshot": {
            "snapshot_sha256": _sha(f"snapshot-{candidate_id}"),
            "patch_bundle_sha256": _sha(f"patch-{candidate_id}"),
        },
        "ancestry": {
            "production_base_commit": base,
            "is_descendant_of_production_base": True,
            "proof": "git merge-base --is-ancestor -> 0",
        },
        "build": {
            "toolchain": "rocm-6.2",
            "compiler": "hipcc 6.2.0",
            "command": "cmake --build build-hip -j 96",
            "build_dir": f"/mnt/raid0/llm/tmp/ak-build/{candidate_id}",
            "log_path": f"data/{CAMPAIGN}/build/{candidate_id}.log",
            "log_sha256": _sha(f"build-log-{candidate_id}"),
        },
        "artifacts": {
            "binary_sha256": _sha(f"binary-{candidate_id}"),
            "linkage_sha256": _sha(f"linkage-{candidate_id}"),
            "library_sha256s": {"libggml.so": _sha(f"libggml-{candidate_id}")},
        },
        "dispatch": {
            "feature_flags": ["GGML_AK_WIDE_TILE"],
            "dispatch_predicate": "K >= 4096",
        },
        "affected_surface": block,
        "determinism": {"class": "bitwise_stable", "same_seed_repeat_runs": 3},
        "evaluator": {
            "id": "P-AK-SEARCH-1/v1",
            "bundle_sha256": _sha("evaluator-bundle"),
        },
        "receipts": {
            "host_receipt": "rcpt-host-20260803T101500Z",
            "resource_claim_receipt": "rcpt-gpu-claim-0042",
        },
        "storage": {"footprint_gb": 3.0, "durability_class": "durable_untracked"},
        "evaluation_event_ids": [],
        "derived_verdicts": {},
        "controller": {
            "provider": "fake",
            "model_id": "fake-planner",
            "effort": "high",
            "prompt_bundle_sha256": _sha("prompt-bundle"),
        },
        "champion_status": champion_status,
        "status": status,
        "supersession_reason": None,
        "created_at": "2026-08-03T10:00:00+00:00",
    }


def _anchor_measurement(event_id: str, *, anchor: SM.AnchorIdentity,
                        backend: str = "llama_gpu") -> dict:
    """A BASELINE cell measuring the anchor binary — what a ratio divides by.

    It names ITSELF as its own anchor measurement: the artifact it timed is the
    anchor binary, so the reference resolves to a cell that measured exactly the
    binary it claims. `schemas` requires a non-empty list for every rate tier and
    has no separate shape for the baseline itself.
    """
    event = _event(event_id, candidate_id="akc-20260803-base", tier="T1",
                   anchor=anchor, backend=backend, measurement_ids=[event_id])
    event["claim_grammar"]["category"] = "BASELINE"
    event["artifact"] = {
        "source_sha256": _sha("anchor-source"),
        "binary_sha256": anchor.binary_sha256[backend],
        "linkage_sha256": anchor.linkage_sha256[backend],
    }
    return event


def _event(event_id: str, *, candidate_id: str, tier: str,
           anchor: SM.AnchorIdentity, backend: str = "llama_gpu",
           status: str = "pass", metric: str = "decode_tokens_per_s",
           measurement_ids=None, created_at: str = "2026-08-03T11:00:00+00:00") -> dict:
    record = {
        "schema": S.SCHEMA_EVALUATION_EVENT,
        "event_id": event_id,
        "campaign_id": CAMPAIGN,
        "candidate_id": candidate_id,
        "tier": tier,
        "claim_grammar": {
            "category": "CANDIDATE",
            "protocol_id": "P-AK-SEARCH-1/v1",
            "metric": metric,
            "metric_direction": "higher_better",
            "reps": 5,
            "attestation_ref": "rcpt-host-20260803T101500Z",
        },
        "evaluator": {
            "id": "P-AK-SEARCH-1/v1",
            "bundle_sha256": _sha("evaluator-bundle"),
        },
        "artifact": {
            "source_sha256": _sha(f"snapshot-{candidate_id}"),
            "binary_sha256": _sha(f"binary-{candidate_id}"),
            "linkage_sha256": _sha(f"linkage-{candidate_id}"),
        },
        "anchor": {
            "source_commit": anchor.commit,
            "binary_sha256": anchor.binary_sha256[backend],
            "linkage_sha256": anchor.linkage_sha256[backend],
            "measurement_event_ids": (["ake-20260801-0009"] if measurement_ids is None
                                      else list(measurement_ids)),
        },
        "scope_manifest_sha256": _sha(f"scope-{candidate_id}"),
        "host_receipt": "rcpt-host-20260803T101500Z",
        "resource_claim_receipt": "rcpt-gpu-claim-0042",
        "co_residency": "single",
        "correctness": {"test_backend_ops": "pass"},
        "quality": {},
        "stability": {},
        "scope_denominator": {
            "machine_subset": "partial",
            "numa_nodes": [0],
            "devices": ["gfx90a:0"],
            "cores": 8,
        },
        "determinism": {"class": "bitwise_stable", "same_seed_repeat_runs": 3},
        "mechanism": {},
        "integrity_flags": [],
        "status": status,
        "supersedes": [],
        "created_at": created_at,
    }
    record["performance"] = {
        "raw_samples": [51.2, 51.4, 51.1],
        "paired_blocks": 3,
        "estimate": 51.23,
        "uncertainty": {"e_process_value": 12.4},
    }
    if tier == "T0":
        record["anchor"]["measurement_event_ids"] = []
    return record


class _Fixture:
    """One journal on disk plus the objects a composition needs to reference it.

    Deliberately a plain object rather than a mixin: several tests want two
    journals, or a journal they mutate after building the views.
    """

    def __init__(self, root: str, *, anchor: SM.AnchorIdentity):
        self.anchor = anchor
        self.journal = J.Journal(root, campaign_id=CAMPAIGN)
        self.journal.initialize()
        self.reconciliations: dict = {}

    def add_candidate(self, candidate_id: str, backends=("llama_gpu",), **kwargs):
        reconciliation = kwargs.pop("reconciliation", None)
        if reconciliation is None:
            reconciliation = _reconciliation(candidate_id, backends)
        self.reconciliations[candidate_id] = reconciliation
        record = _candidate(candidate_id, reconciliation, **kwargs)
        self.journal.append(J.KIND_CANDIDATE_RECORDED, record)
        return record, reconciliation

    def add_event(self, event_id: str, **kwargs):
        record = _event(event_id, anchor=kwargs.pop("anchor", self.anchor), **kwargs)
        self.journal.append(J.KIND_EVALUATION_EVENT, record)
        return record

    def add_anchor_measurement(self, event_id: str, backend: str = "llama_gpu"):
        record = _anchor_measurement(event_id, anchor=self.anchor, backend=backend)
        self.journal.append(J.KIND_EVALUATION_EVENT, record)
        return record

    def views(self) -> J.Views:
        return J.rebuild_views(self.journal.read_all())

    def entry_id(self, record_id: str) -> str:
        """The JOURNAL ENTRY id (`akj-…`) carrying a record — supersession's key.

        `journal.append_superseded` resolves `target_event_id` against
        `JournalEntry.event_id`, which is not the payload's own `event_id`. Tests
        that confuse the two would pass a dangling reference and learn nothing.
        """
        for entry in self.journal.read_all():
            if entry.record_id == record_id:
                return entry.event_id
        raise KeyError(record_id)

    def frontier(self, candidate_id: str, mechanism_class: str = "layout"):
        record = self.views().candidates[candidate_id]
        return C.admit_to_frontier(record, self.reconciliations[candidate_id],
                                   mechanism_class=mechanism_class)


class CompositionTestCase(unittest.TestCase):
    """Base: a temp dir per test, removed on teardown."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name

    def fixture(self, name: str = "journal", *, anchor=None) -> _Fixture:
        return _Fixture(str(Path(self.root) / name),
                        anchor=anchor if anchor is not None else _anchor())

    def composed_fixture(self, *, member_backends=("llama_gpu",),
                         combined_backends=("llama_gpu",)):
        """Two members plus a combined candidate with passing T0 and T1."""
        fx = self.fixture()
        fx.add_anchor_measurement("ake-20260801-0009")
        fx.add_candidate("akc-20260803-0001", member_backends)
        fx.add_candidate("akc-20260803-0002", member_backends)
        fx.add_candidate("akc-20260803-0100", combined_backends)
        fx.add_event("ake-20260803-1000", candidate_id="akc-20260803-0100", tier="T0")
        fx.add_event("ake-20260803-1001", candidate_id="akc-20260803-0100", tier="T1")
        return fx

    def lineage_of(self, fx, classes=("layout", "fusion")):
        members = [
            fx.frontier("akc-20260803-0001", classes[0]),
            fx.frontier("akc-20260803-0002", classes[1]),
        ]
        return C.propose_lineage(members, anchor_commit=V8_COMMIT)


# =============================================================================
# The three concepts stay separate
# =============================================================================

class ThreeConceptsTests(CompositionTestCase):

    def test_banked_candidate_becomes_a_frontier_candidate(self):
        fx = self.fixture()
        record, reconciliation = fx.add_candidate("akc-20260803-0001")
        candidate = C.admit_to_frontier(record, reconciliation,
                                        mechanism_class="layout")
        self.assertEqual(candidate.candidate_id, "akc-20260803-0001")
        self.assertEqual(candidate.source_tree, "llama.cpp")
        self.assertTrue(candidate.surface_reconciled)

    def test_unbanked_candidate_is_not_a_frontier_candidate(self):
        """§9.6: banking is the evaluator's disposition, and this module reads it."""
        fx = self.fixture()
        record, reconciliation = fx.add_candidate("akc-20260803-0001",
                                                  status="evaluating",
                                                  champion_status="none")
        with self.assertRaises(C.NotBanked):
            C.admit_to_frontier(record, reconciliation, mechanism_class="layout")

    def test_spike_is_a_different_type_with_no_composable_surface(self):
        fx = self.fixture()
        record, _ = fx.add_candidate("akc-20260803-0007", status="evaluating",
                                     champion_status="none")
        spike = C.record_experiment(record, kind="spike", mechanism_class="fusion",
                                    receipt="ake-20260803-0500")
        self.assertIsInstance(spike, C.Experiment)
        field_names = {f.name for f in dataclasses.fields(C.Experiment)}
        self.assertNotIn("source_tree", field_names)
        self.assertNotIn("derived_surface_sha256", field_names)

    def test_spike_may_never_accumulate(self):
        fx = self.fixture()
        record, _ = fx.add_candidate("akc-20260803-0007", status="evaluating",
                                     champion_status="none")
        spike = C.record_experiment(record, kind="spike", mechanism_class="fusion",
                                    receipt="ake-20260803-0500")
        with self.assertRaises(C.ExperimentMayNotAccumulate) as caught:
            C.propose_lineage([spike], anchor_commit=V8_COMMIT,
                              source_tree="llama.cpp")
        self.assertIn("never accumulate", str(caught.exception))

    def test_banked_candidate_cannot_be_relabelled_an_experiment(self):
        fx = self.fixture()
        record, _ = fx.add_candidate("akc-20260803-0001")
        with self.assertRaises(C.ExperimentMayNotAccumulate):
            C.record_experiment(record, kind="spike", mechanism_class="fusion",
                                receipt="ake-20260803-0500")

    def test_experiment_needs_a_receipt(self):
        """§8.4.1: a refuted spike closes a direction WITH a receipt."""
        with self.assertRaises(ValueError):
            C.Experiment(candidate_id="akc-1", kind="spike",
                         mechanism_class="fusion", receipt="  ")

    def test_a_lineage_is_not_a_champion(self):
        """A ComposedLineage carries no evidence field at all."""
        names = {f.name for f in dataclasses.fields(C.ComposedLineage)}
        self.assertEqual(names, {"source_tree", "anchor_commit", "branch", "members"})


# =============================================================================
# One champion per SOURCE TREE (§1.5, AK-D11)
# =============================================================================

class SourceTreeTests(CompositionTestCase):

    def test_cpu_and_gpu_share_one_champion(self):
        fx = self.fixture()
        fx.add_candidate("akc-20260803-0001", ("llama_cpu",))
        fx.add_candidate("akc-20260803-0002", ("llama_gpu",))
        lineage = C.propose_lineage(
            [fx.frontier("akc-20260803-0001", "arithmetic"),
             fx.frontier("akc-20260803-0002", "layout")],
            anchor_commit=V8_COMMIT,
        )
        self.assertEqual(lineage.source_tree, "llama.cpp")
        self.assertEqual(lineage.backends, ("llama_cpu", "llama_gpu"))

    def test_whisper_cannot_join_the_llama_champion(self):
        fx = self.fixture()
        fx.add_candidate("akc-20260803-0001", ("llama_gpu",))
        fx.add_candidate("akc-20260803-0002", ("whisper_stt",))
        with self.assertRaises(C.IncompatibleMember) as caught:
            C.propose_lineage(
                [fx.frontier("akc-20260803-0001", "layout"),
                 fx.frontier("akc-20260803-0002", "arithmetic")],
                anchor_commit=V8_COMMIT,
            )
        self.assertIn("source tree", str(caught.exception))

    def test_serving_runtime_has_no_kernel_champion(self):
        """AK-D9/AK-D23: the scheduler lane releases through the stack-change gate."""
        with self.assertRaises(C.IncompatibleMember) as caught:
            C.source_tree_for_backends(("serving_runtime",))
        self.assertIn("stack-change", str(caught.exception))

    def test_champion_branch_is_namespaced_and_never_production(self):
        branch = C.champion_branch_for("llama.cpp", V8_COMMIT)
        self.assertTrue(branch.startswith("ak/champion/"))
        self.assertNotIn("production-", branch)

    def test_empty_lineage_still_names_its_tree(self):
        lineage = C.propose_lineage([], anchor_commit=V8_COMMIT,
                                    source_tree="llama.cpp")
        self.assertEqual(lineage.members, ())
        with self.assertRaises(ValueError):
            C.propose_lineage([], anchor_commit=V8_COMMIT)


# =============================================================================
# Only RECONCILED surfaces may be combined (§8.9, §6.4, invariant 18)
# =============================================================================

class ReconciliationTests(CompositionTestCase):

    def test_untraced_surface_is_not_reconciled_and_cannot_combine(self):
        fx = self.fixture()
        reconciliation = _reconciliation("akc-20260803-0003", traced=False)
        self.assertEqual(reconciliation.check.outcome, S.COULD_NOT_CHECK)
        fx.add_candidate("akc-20260803-0003", reconciliation=reconciliation)
        candidate = fx.frontier("akc-20260803-0003", "layout")
        self.assertFalse(candidate.surface_reconciled)
        with self.assertRaises(C.UnreconciledSurface):
            C.propose_lineage([candidate], anchor_commit=V8_COMMIT)

    def test_escaped_dispatch_is_a_hard_failure_and_cannot_combine(self):
        fx = self.fixture()
        reconciliation = _reconciliation("akc-20260803-0004", escape=True)
        self.assertTrue(reconciliation.hard_failure)
        fx.add_candidate("akc-20260803-0004", reconciliation=reconciliation)
        with self.assertRaises(C.UnreconciledSurface):
            C.propose_lineage([fx.frontier("akc-20260803-0004", "layout")],
                              anchor_commit=V8_COMMIT)

    def test_reconciliation_verdict_comes_from_surface_not_from_here(self):
        """The module must not reimplement §6.4 stage 3."""
        reconciliation = _reconciliation("akc-20260803-0001")
        block = SF.candidate_affected_surface_block(reconciliation)
        fx = self.fixture()
        record, _ = fx.add_candidate("akc-20260803-0001",
                                     reconciliation=reconciliation)
        self.assertEqual(record["affected_surface"], block)

    def test_record_and_reconciliation_must_agree(self):
        """A healthy reconciliation may not be presented for a different record."""
        good = _reconciliation("akc-20260803-0001")
        record = _candidate("akc-20260803-0001", good)
        record["affected_surface"] = dict(record["affected_surface"])
        record["affected_surface"]["derived_sha256"] = _sha("some-other-surface")
        with self.assertRaises(C.CompositionError) as caught:
            C.admit_to_frontier(record, good, mechanism_class="layout")
        self.assertIn("disagree", str(caught.exception))

    def test_reconciliation_for_another_candidate_is_refused(self):
        fx = self.fixture()
        record, _ = fx.add_candidate("akc-20260803-0001")
        other = _reconciliation("akc-20260803-0002")
        with self.assertRaises(C.IncompatibleMember):
            C.admit_to_frontier(record, other, mechanism_class="layout")

    def test_a_bare_boolean_is_not_a_reconciliation(self):
        fx = self.fixture()
        record, _ = fx.add_candidate("akc-20260803-0001")
        with self.assertRaises(TypeError):
            C.admit_to_frontier(record, True, mechanism_class="layout")


# =============================================================================
# The composed champion is RE-MEASURED, never inferred (§8.9, §12, denial 9)
# =============================================================================

class ComposeChampionTests(CompositionTestCase):

    def test_composed_champion_is_a_valid_record_citing_combined_evidence(self):
        fx = self.composed_fixture()
        lineage = self.lineage_of(fx)
        record = C.compose_champion(
            lineage,
            combined_candidate_id="akc-20260803-0100",
            combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
            views=fx.views(),
            recorded_anchor=fx.anchor,
            observed_anchor=_anchor(),
            storage_gb=12.0,
        )
        self.assertEqual(S.validate_champion(record), [])
        self.assertEqual(record["combined_candidate_id"], "akc-20260803-0100")
        self.assertEqual(record["last_t0"]["event_id"], "ake-20260803-1000")
        self.assertEqual(record["last_t1"]["event_id"], "ake-20260803-1001")
        self.assertIsNone(record["last_t2"])
        self.assertEqual(record["member_candidates"],
                         ["akc-20260803-0001", "akc-20260803-0002"])

    def test_champion_is_appendable_to_the_journal(self):
        fx = self.composed_fixture()
        record = C.compose_champion(
            self.lineage_of(fx),
            combined_candidate_id="akc-20260803-0100",
            combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
            views=fx.views(),
            recorded_anchor=fx.anchor,
            observed_anchor=_anchor(),
            storage_gb=12.0,
        )
        entry = C.record_champion(fx.journal, record)
        self.assertEqual(entry.kind, J.KIND_CHAMPION_UPDATED)
        self.assertEqual(fx.views().champions["llama.cpp"]["combined_candidate_id"],
                         "akc-20260803-0100")

    def test_missing_combined_t1_refuses_even_when_every_member_passed(self):
        """The exact shape of 'inferred by multiplying local speedups'."""
        fx = self.fixture()
        fx.add_anchor_measurement("ake-20260801-0009")
        fx.add_candidate("akc-20260803-0001")
        fx.add_candidate("akc-20260803-0002")
        fx.add_candidate("akc-20260803-0100")
        # Both MEMBERS have glowing T0+T1. The composition has neither.
        for i, member in enumerate(("akc-20260803-0001", "akc-20260803-0002")):
            fx.add_event(f"ake-20260803-200{i}", candidate_id=member, tier="T0")
            fx.add_event(f"ake-20260803-201{i}", candidate_id=member, tier="T1")
        with self.assertRaises(C.CompositionEvidenceMissing) as caught:
            C.compose_champion(
                self.lineage_of(fx),
                combined_candidate_id="akc-20260803-0100",
                combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
                views=fx.views(),
                recorded_anchor=fx.anchor,
                observed_anchor=_anchor(),
                storage_gb=1.0,
            )
        self.assertIn("T0", str(caught.exception))
        self.assertIn("combined full candidate", str(caught.exception))

    def test_failing_combined_t1_is_not_evidence(self):
        fx = self.fixture()
        fx.add_anchor_measurement("ake-20260801-0009")
        fx.add_candidate("akc-20260803-0001")
        fx.add_candidate("akc-20260803-0002")
        fx.add_candidate("akc-20260803-0100")
        fx.add_event("ake-20260803-1000", candidate_id="akc-20260803-0100", tier="T0")
        fx.add_event("ake-20260803-1001", candidate_id="akc-20260803-0100", tier="T1",
                     status="fail")
        with self.assertRaises(C.CompositionEvidenceMissing) as caught:
            C.compose_champion(
                self.lineage_of(fx),
                combined_candidate_id="akc-20260803-0100",
                combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
                views=fx.views(),
                recorded_anchor=fx.anchor,
                observed_anchor=_anchor(),
                storage_gb=1.0,
            )
        self.assertIn("'fail'", str(caught.exception))

    def test_combined_t1_measured_against_another_anchor_is_refused(self):
        fx = self.composed_fixture()
        stale = _anchor(commit=V7_COMMIT, binary="old-binary", linkage="old-linkage")
        fx.journal.append(J.KIND_SUPERSEDED, {
            "target_event_id": fx.entry_id("ake-20260803-1001"),
            "reason": "replaced by a re-run against another anchor (test fixture)",
            "superseded_by": None,
        })
        fx.add_event("ake-20260803-1002", candidate_id="akc-20260803-0100", tier="T1",
                     anchor=stale)
        with self.assertRaises(C.CompositionEvidenceMissing) as caught:
            C.compose_champion(
                self.lineage_of(fx),
                combined_candidate_id="akc-20260803-0100",
                combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
                views=fx.views(),
                recorded_anchor=fx.anchor,
                observed_anchor=_anchor(),
                storage_gb=1.0,
            )
        self.assertIn("champion anchor", str(caught.exception))

    def test_rate_tier_needs_its_anchor_measurement_to_resolve(self):
        """P-AK-SEARCH-1 precondition 4: a ratio needs its denominator bound."""
        fx = self.fixture()
        fx.add_candidate("akc-20260803-0001")
        fx.add_candidate("akc-20260803-0002")
        fx.add_candidate("akc-20260803-0100")
        fx.add_event("ake-20260803-1000", candidate_id="akc-20260803-0100", tier="T0")
        # T1 names an anchor measurement that is not in this journal.
        fx.add_event("ake-20260803-1001", candidate_id="akc-20260803-0100", tier="T1")
        with self.assertRaises(C.CompositionEvidenceMissing) as caught:
            C.compose_champion(
                self.lineage_of(fx),
                combined_candidate_id="akc-20260803-0100",
                combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
                views=fx.views(),
                recorded_anchor=fx.anchor,
                observed_anchor=_anchor(),
                storage_gb=1.0,
            )
        self.assertIn("anchor binding", str(caught.exception))

    def test_combined_candidate_may_not_be_one_of_its_members(self):
        fx = self.composed_fixture()
        lineage = self.lineage_of(fx)
        with self.assertRaises(C.IncompatibleMember) as caught:
            C.compose_champion(
                lineage,
                combined_candidate_id="akc-20260803-0001",
                combined_reconciliation=fx.reconciliations["akc-20260803-0001"],
                views=fx.views(),
                recorded_anchor=fx.anchor,
                observed_anchor=_anchor(),
                storage_gb=1.0,
            )
        self.assertIn("both a member and the composition", str(caught.exception))

    def test_composition_must_reach_every_member_backend(self):
        """A CPU+GPU lineage composed into a GPU-only artifact is not that lineage."""
        fx = self.fixture()
        fx.add_anchor_measurement("ake-20260801-0009")
        fx.add_candidate("akc-20260803-0001", ("llama_cpu",))
        fx.add_candidate("akc-20260803-0002", ("llama_gpu",))
        fx.add_candidate("akc-20260803-0100", ("llama_gpu",))
        fx.add_event("ake-20260803-1000", candidate_id="akc-20260803-0100", tier="T0")
        fx.add_event("ake-20260803-1001", candidate_id="akc-20260803-0100", tier="T1")
        lineage = C.propose_lineage(
            [fx.frontier("akc-20260803-0001", "arithmetic"),
             fx.frontier("akc-20260803-0002", "layout")],
            anchor_commit=V8_COMMIT)
        with self.assertRaises(C.IncompatibleMember) as caught:
            C.compose_champion(
                lineage,
                combined_candidate_id="akc-20260803-0100",
                combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
                views=fx.views(),
                recorded_anchor=fx.anchor,
                observed_anchor=_anchor(),
                storage_gb=1.0,
            )
        self.assertIn("llama_cpu", str(caught.exception))

    def test_combined_candidate_on_a_stale_base_is_refused(self):
        fx = self.fixture()
        fx.add_anchor_measurement("ake-20260801-0009")
        fx.add_candidate("akc-20260803-0001")
        fx.add_candidate("akc-20260803-0002")
        fx.add_candidate("akc-20260803-0100", base=V7_COMMIT)
        fx.add_event("ake-20260803-1000", candidate_id="akc-20260803-0100", tier="T0")
        fx.add_event("ake-20260803-1001", candidate_id="akc-20260803-0100", tier="T1")
        with self.assertRaises(C.IncompatibleMember) as caught:
            C.compose_champion(
                self.lineage_of(fx),
                combined_candidate_id="akc-20260803-0100",
                combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
                views=fx.views(),
                recorded_anchor=fx.anchor,
                observed_anchor=_anchor(),
                storage_gb=1.0,
            )
        self.assertIn("invariant 1", str(caught.exception))

    def test_member_without_a_record_is_refused(self):
        fx = self.composed_fixture()
        lineage = self.lineage_of(fx)
        ghost = dataclasses.replace(lineage.members[0],
                                    candidate_id="akc-20260803-9999")
        haunted = dataclasses.replace(lineage, members=(ghost,) + lineage.members[1:])
        with self.assertRaises(C.CompositionEvidenceMissing) as caught:
            C.compose_champion(
                haunted,
                combined_candidate_id="akc-20260803-0100",
                combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
                views=fx.views(),
                recorded_anchor=fx.anchor,
                observed_anchor=_anchor(),
                storage_gb=1.0,
            )
        self.assertIn("akc-20260803-9999", str(caught.exception))

    def test_a_hand_built_lineage_cannot_smuggle_an_unreconciled_member(self):
        """`ComposedLineage` is public, so the gate is re-run from the journal."""
        fx = self.fixture()
        fx.add_anchor_measurement("ake-20260801-0009")
        untraced = _reconciliation("akc-20260803-0001", traced=False)
        fx.add_candidate("akc-20260803-0001", reconciliation=untraced)
        fx.add_candidate("akc-20260803-0100")
        fx.add_event("ake-20260803-1000", candidate_id="akc-20260803-0100", tier="T0")
        fx.add_event("ake-20260803-1001", candidate_id="akc-20260803-0100", tier="T1")
        smuggled = C.ComposedLineage(
            source_tree="llama.cpp",
            anchor_commit=V8_COMMIT,
            branch=C.champion_branch_for("llama.cpp", V8_COMMIT),
            members=(C.LineageMember(
                candidate_id="akc-20260803-0001",
                mechanism_class="layout",
                backends=("llama_gpu",),
                anchor_commit=V8_COMMIT,
                derived_surface_sha256=untraced.derived.sha256(),
                traced_surface_sha256=None,
            ),),
        )
        with self.assertRaises(C.UnreconciledSurface) as caught:
            C.compose_champion(
                smuggled,
                combined_candidate_id="akc-20260803-0100",
                combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
                views=fx.views(),
                recorded_anchor=fx.anchor,
                observed_anchor=_anchor(),
                storage_gb=1.0,
            )
        self.assertIn("own record", str(caught.exception))

    def test_a_hand_built_lineage_cannot_smuggle_an_unbanked_member(self):
        fx = self.fixture()
        fx.add_anchor_measurement("ake-20260801-0009")
        fx.add_candidate("akc-20260803-0001", status="evaluating",
                         champion_status="none")
        fx.add_candidate("akc-20260803-0100")
        fx.add_event("ake-20260803-1000", candidate_id="akc-20260803-0100", tier="T0")
        fx.add_event("ake-20260803-1001", candidate_id="akc-20260803-0100", tier="T1")
        smuggled = C.ComposedLineage(
            source_tree="llama.cpp",
            anchor_commit=V8_COMMIT,
            branch=C.champion_branch_for("llama.cpp", V8_COMMIT),
            members=(C.LineageMember(
                candidate_id="akc-20260803-0001",
                mechanism_class="layout",
                backends=("llama_gpu",),
                anchor_commit=V8_COMMIT,
                derived_surface_sha256=fx.reconciliations[
                    "akc-20260803-0001"].derived.sha256(),
                traced_surface_sha256=None,
            ),),
        )
        with self.assertRaises(C.NotBanked):
            C.compose_champion(
                smuggled,
                combined_candidate_id="akc-20260803-0100",
                combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
                views=fx.views(),
                recorded_anchor=fx.anchor,
                observed_anchor=_anchor(),
                storage_gb=1.0,
            )

    def test_t2_is_recorded_when_present_and_never_required(self):
        fx = self.composed_fixture()
        fx.add_event("ake-20260803-1002", candidate_id="akc-20260803-0100", tier="T2")
        record = C.compose_champion(
            self.lineage_of(fx),
            combined_candidate_id="akc-20260803-0100",
            combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
            views=fx.views(),
            recorded_anchor=fx.anchor,
            observed_anchor=_anchor(),
            storage_gb=1.0,
        )
        self.assertEqual(record["last_t2"]["event_id"], "ake-20260803-1002")

    def test_latest_qualifying_event_wins_deterministically(self):
        fx = self.composed_fixture()
        fx.add_event("ake-20260803-1009", candidate_id="akc-20260803-0100", tier="T1",
                     created_at="2026-08-03T13:00:00+00:00")
        record = C.compose_champion(
            self.lineage_of(fx),
            combined_candidate_id="akc-20260803-0100",
            combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
            views=fx.views(),
            recorded_anchor=fx.anchor,
            observed_anchor=_anchor(),
            storage_gb=1.0,
        )
        self.assertEqual(record["last_t1"]["event_id"], "ake-20260803-1009")


# =============================================================================
# Readiness is cited, never narrated (invariant 14)
# =============================================================================

class ReadinessCitationTests(CompositionTestCase):

    def _compose(self, fx, **kwargs):
        return C.compose_champion(
            self.lineage_of(fx),
            combined_candidate_id="akc-20260803-0100",
            combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
            views=fx.views(),
            recorded_anchor=fx.anchor,
            observed_anchor=_anchor(),
            storage_gb=1.0,
            **kwargs,
        )

    def test_default_readiness_is_an_uncited_skeleton(self):
        fx = self.composed_fixture()
        record = self._compose(fx)
        cells = record["readiness"]["by_backend"]["llama_gpu"]["phases"]
        self.assertEqual(sorted(cells), ["decode", "prefill"])
        self.assertFalse(cells["decode"]["measured"])

    def test_a_readiness_cell_must_cite_the_combined_candidates_events(self):
        fx = self.composed_fixture()
        with self.assertRaises(C.CompositionEvidenceMissing) as caught:
            self._compose(fx, readiness_by_backend={
                "llama_gpu": {"decode": {"event_ids": ["ake-20260803-1001"]},
                              "prefill": {"event_ids": []}}})
        self.assertIn("must cite at least one", str(caught.exception))

    def test_a_readiness_cell_citing_a_member_event_is_refused(self):
        """The narrated-composition hole, closed structurally."""
        fx = self.composed_fixture()
        fx.add_event("ake-20260803-3000", candidate_id="akc-20260803-0001", tier="T1")
        with self.assertRaises(C.CompositionEvidenceMissing) as caught:
            self._compose(fx, readiness_by_backend={
                "llama_gpu": {"decode": {"event_ids": ["ake-20260803-3000"]}}})
        self.assertIn("not of the combined candidate", str(caught.exception))

    def test_a_readiness_cell_citing_a_phantom_event_is_refused(self):
        fx = self.composed_fixture()
        with self.assertRaises(C.CompositionEvidenceMissing):
            self._compose(fx, readiness_by_backend={
                "llama_gpu": {"decode": {"event_ids": ["ake-does-not-exist"]}}})

    def test_readiness_may_not_name_an_unreached_backend(self):
        fx = self.composed_fixture()
        with self.assertRaises(C.IncompatibleMember):
            self._compose(fx, readiness_by_backend={
                "whisper_stt": {"decode": {"event_ids": ["ake-20260803-1001"]}}})

    def test_reference_signal_is_rendered_here_and_cannot_be_supplied(self):
        fx = self.composed_fixture()
        record = self._compose(fx, readiness_by_backend={
            "llama_gpu": {"decode": {"event_ids": ["ake-20260803-1001"],
                                     "note": "supplied by the reducer"}}})
        signal = record["readiness"]["reference_signal"]
        self.assertIn("akc-20260803-0100", signal)
        self.assertIn("No member result contributes", signal)
        parameters = C.compose_champion.__code__.co_varnames
        self.assertNotIn("reference_signal", parameters)


# =============================================================================
# Composition never multiplies (§8.9, §12, P-AK-SEARCH-1 denial 9)
# =============================================================================

class NoComposedEstimateTests(CompositionTestCase):

    def test_module_holds_no_arithmetic_and_reads_no_measured_quantity(self):
        self.assertEqual(C.audit_no_composed_estimate_arithmetic().outcome, S.PASS)

    def test_audit_catches_a_multiplication(self):
        check = C.audit_no_composed_estimate_arithmetic(
            "def combined(a, b):\n    return a * b\n")
        self.assertEqual(check.outcome, S.FAIL)
        self.assertTrue(any("Mult" in r for r in check.reasons))

    def test_audit_catches_a_read_of_a_measured_quantity(self):
        check = C.audit_no_composed_estimate_arithmetic(
            "def f(event):\n    return event['estimate']\n")
        self.assertEqual(check.outcome, S.FAIL)

    def test_audit_catches_an_attribute_read(self):
        check = C.audit_no_composed_estimate_arithmetic(
            "def f(verdict):\n    return verdict.effect\n")
        self.assertEqual(check.outcome, S.FAIL)

    def test_audit_cannot_check_unparseable_source(self):
        check = C.audit_no_composed_estimate_arithmetic("def (:\n")
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)

    def test_audit_cannot_check_a_non_string(self):
        self.assertEqual(
            C.audit_no_composed_estimate_arithmetic(object()).outcome,
            S.COULD_NOT_CHECK)

    def test_lineage_member_carries_no_measured_quantity(self):
        names = {f.name for f in dataclasses.fields(C.LineageMember)}
        self.assertEqual(names & set(C._FORBIDDEN_EVIDENCE_KEYS), set())

    def test_a_member_type_carrying_a_result_is_refused_at_definition(self):
        @dataclasses.dataclass(frozen=True)
        class Tainted:
            candidate_id: str
            speedup: float

        with self.assertRaises(C.CompositionError) as caught:
            C._assert_no_forbidden_fields(Tainted)
        self.assertIn("measured quantity", str(caught.exception))

    def test_compose_champion_takes_no_member_evidence_parameter(self):
        parameters = set(C.compose_champion.__code__.co_varnames)
        self.assertEqual(parameters & set(C._FORBIDDEN_EVIDENCE_KEYS), set())
        self.assertNotIn("member_verdicts", parameters)
        self.assertNotIn("member_events", parameters)


# =============================================================================
# Mechanism-class diversity (§8.9)
# =============================================================================

class DiversityTests(CompositionTestCase):

    def _pool(self, spec):
        """`spec` is [(candidate_id_suffix, mechanism_class), ...] in preference order."""
        fx = self.fixture()
        out = []
        for suffix, klass in spec:
            fx.add_candidate(f"akc-20260803-{suffix}")
            out.append(fx.frontier(f"akc-20260803-{suffix}", klass))
        return out

    def test_a_sweeping_family_cannot_evict_the_last_of_another(self):
        pool = self._pool([("0001", "layout"), ("0002", "layout"),
                           ("0003", "layout"), ("0004", "fusion")])
        kept = C.retain_frontier(pool, capacity=2)
        classes = {c.mechanism_class for c in kept}
        self.assertEqual(classes, {"layout", "fusion"})
        self.assertEqual([c.candidate_id for c in kept],
                         ["akc-20260803-0001", "akc-20260803-0004"])

    def test_preference_order_is_preserved(self):
        pool = self._pool([("0001", "layout"), ("0002", "fusion"),
                           ("0003", "arithmetic")])
        kept = C.retain_frontier(pool, capacity=3)
        self.assertEqual([c.candidate_id for c in kept],
                         ["akc-20260803-0001", "akc-20260803-0002",
                          "akc-20260803-0003"])

    def test_spare_capacity_goes_to_preference_after_the_quota(self):
        pool = self._pool([("0001", "layout"), ("0002", "layout"),
                           ("0003", "layout"), ("0004", "fusion")])
        kept = C.retain_frontier(pool, capacity=3)
        self.assertEqual([c.candidate_id for c in kept],
                         ["akc-20260803-0001", "akc-20260803-0002",
                          "akc-20260803-0004"])

    def test_capacity_below_the_floor_is_a_refusal_not_a_truncation(self):
        pool = self._pool([("0001", "layout"), ("0002", "fusion"),
                           ("0003", "arithmetic")])
        with self.assertRaises(C.DiversityFloorUnmet) as caught:
            C.retain_frontier(pool, capacity=2)
        self.assertIn("single family", str(caught.exception))

    def test_min_per_class_above_one(self):
        pool = self._pool([("0001", "layout"), ("0002", "layout"),
                           ("0003", "fusion"), ("0004", "fusion"),
                           ("0005", "layout")])
        kept = C.retain_frontier(pool, capacity=4, min_per_class=2)
        self.assertEqual([c.candidate_id for c in kept],
                         ["akc-20260803-0001", "akc-20260803-0002",
                          "akc-20260803-0003", "akc-20260803-0004"])

    def test_retention_refuses_experiments(self):
        fx = self.fixture()
        record, _ = fx.add_candidate("akc-20260803-0007", status="evaluating",
                                     champion_status="none")
        spike = C.record_experiment(record, kind="spike", mechanism_class="fusion",
                                    receipt="ake-1")
        with self.assertRaises(TypeError):
            C.retain_frontier([spike], capacity=1)

    def test_diversity_check_is_could_not_check_without_the_available_set(self):
        pool = self._pool([("0001", "layout"), ("0002", "layout")])
        check = C.check_mechanism_diversity(pool, min_classes=2)
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)

    def test_diversity_check_fails_when_another_class_was_available(self):
        pool = self._pool([("0001", "layout"), ("0002", "layout")])
        check = C.check_mechanism_diversity(
            pool, min_classes=2, available_classes=("layout", "fusion"))
        self.assertEqual(check.outcome, S.FAIL)

    def test_diversity_check_passes_when_one_class_is_all_there_is(self):
        pool = self._pool([("0001", "layout"), ("0002", "layout")])
        check = C.check_mechanism_diversity(
            pool, min_classes=2, available_classes=("layout",))
        self.assertEqual(check.outcome, S.PASS)

    def test_diversity_check_on_an_empty_frontier_is_could_not_check(self):
        self.assertEqual(
            C.check_mechanism_diversity([], min_classes=1).outcome,
            S.COULD_NOT_CHECK)

    def test_mechanism_class_vocabulary_is_the_ratified_closed_set(self):
        self.assertIs(C.MECHANISM_CLASSES, S.CHANGE_CLASSES)
        fx = self.fixture()
        record, reconciliation = fx.add_candidate("akc-20260803-0001")
        with self.assertRaises(ValueError):
            C.admit_to_frontier(record, reconciliation,
                                mechanism_class="a_brand_new_family")


# =============================================================================
# Re-anchoring at a freeze (§8.9)
# =============================================================================

class ReanchorTests(CompositionTestCase):

    def _champion(self, fx):
        return C.compose_champion(
            self.lineage_of(fx),
            combined_candidate_id="akc-20260803-0100",
            combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
            views=fx.views(),
            recorded_anchor=fx.anchor,
            observed_anchor=_anchor(),
            storage_gb=12.0,
        )

    def test_members_in_production_are_dropped_and_the_rest_rebase(self):
        fx = self.composed_fixture()
        champion = self._champion(fx)
        plan = C.plan_reanchor(champion, new_anchor=_anchor(commit=V9_COMMIT),
                               members_in_production=["akc-20260803-0001"])
        self.assertEqual(plan.dropped_members, ("akc-20260803-0001",))
        self.assertEqual(plan.rebase_sources, ("akc-20260803-0002",))
        self.assertEqual(plan.new_anchor_commit, V9_COMMIT)
        self.assertIn(V9_COMMIT[:12], plan.new_branch)

    def test_t1_evidence_is_invalidated_and_source_is_preserved(self):
        fx = self.composed_fixture()
        champion = self._champion(fx)
        plan = C.plan_reanchor(champion, new_anchor=_anchor(commit=V9_COMMIT))
        self.assertIn("ake-20260803-1001", plan.invalidated_comparison_event_ids)
        self.assertIn("ake-20260803-1000", plan.invalidated_artifact_event_ids)
        self.assertEqual(set(plan.preserved_candidate_ids),
                         {"akc-20260803-0001", "akc-20260803-0002"})
        self.assertEqual(plan.requires_remeasure_tiers, ("T0", "T1"))

    def test_a_non_empty_reanchor_cannot_produce_a_champion_without_remeasuring(self):
        fx = self.composed_fixture()
        champion = self._champion(fx)
        plan = C.plan_reanchor(champion, new_anchor=_anchor(commit=V9_COMMIT))
        self.assertFalse(plan.is_empty)
        with self.assertRaises(C.CompositionEvidenceMissing) as caught:
            plan.to_champion_record(storage_gb=0.0)
        self.assertIn("RE-MEASURED", str(caught.exception))

    def test_an_emptied_lineage_is_a_recordable_champion_at_the_new_tip(self):
        fx = self.composed_fixture()
        champion = self._champion(fx)
        plan = C.plan_reanchor(
            champion, new_anchor=_anchor(commit=V9_COMMIT),
            members_in_production=["akc-20260803-0001", "akc-20260803-0002"])
        self.assertTrue(plan.is_empty)
        record = plan.to_champion_record(storage_gb=0.0)
        self.assertEqual(S.validate_champion(record), [])
        self.assertEqual(record["member_candidates"], [])
        self.assertIsNone(record["combined_candidate_id"])
        self.assertIsNone(record["last_t1"])
        self.assertEqual(record["anchor_commit"], V9_COMMIT)

    def test_a_member_at_the_old_anchor_cannot_rejoin_the_rebased_lineage(self):
        """The enforcement that 'the remainder rebases on the new tip' is real."""
        fx = self.composed_fixture()
        old_member = fx.frontier("akc-20260803-0002", "fusion")
        with self.assertRaises(C.IncompatibleMember) as caught:
            C.propose_lineage([old_member], anchor_commit=V9_COMMIT)
        self.assertIn("rebased", str(caught.exception))

    def test_reanchor_refuses_a_member_that_is_not_in_the_lineage(self):
        fx = self.composed_fixture()
        champion = self._champion(fx)
        with self.assertRaises(C.ReanchorRefused):
            C.plan_reanchor(champion, new_anchor=_anchor(commit=V9_COMMIT),
                            members_in_production=["akc-20260803-7777"])

    def test_reanchor_refuses_a_new_anchor_for_another_tree(self):
        fx = self.composed_fixture()
        champion = self._champion(fx)
        with self.assertRaises(C.ReanchorRefused):
            C.plan_reanchor(champion,
                            new_anchor=_anchor(commit=V9_COMMIT, tree="whisper.cpp",
                                               backends=("whisper_stt",),
                                               branch="production-speech-v1"))

    def test_reanchor_refuses_a_base_that_did_not_move(self):
        fx = self.composed_fixture()
        champion = self._champion(fx)
        with self.assertRaises(C.ReanchorRefused) as caught:
            C.plan_reanchor(champion, new_anchor=_anchor())
        self.assertIn("does not move the base", str(caught.exception))

    def test_reanchor_refuses_an_invalid_champion(self):
        with self.assertRaises(C.ReanchorRefused):
            C.plan_reanchor({"schema": S.SCHEMA_CHAMPION},
                            new_anchor=_anchor(commit=V9_COMMIT))


# =============================================================================
# ANCHOR_MOVED (§8.9 items 1-5, AK-D22)
# =============================================================================

class AnchorMovedTests(CompositionTestCase):

    def _moved(self):
        return _anchor(commit=V9_COMMIT, binary="hotfix-binary",
                       linkage="hotfix-linkage")

    def test_composition_halts_while_the_anchor_disagrees(self):
        fx = self.composed_fixture()
        with self.assertRaises(C.AnchorMovedRefusal) as caught:
            C.compose_champion(
                self.lineage_of(fx),
                combined_candidate_id="akc-20260803-0100",
                combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
                views=fx.views(),
                recorded_anchor=fx.anchor,
                observed_anchor=self._moved(),
                storage_gb=1.0,
            )
        self.assertIn("no new candidate work", str(caught.exception))

    def test_an_unobserved_anchor_is_refused_not_assumed_good(self):
        fx = self.composed_fixture()
        with self.assertRaises(SM.AnchorUncheckable):
            C.compose_champion(
                self.lineage_of(fx),
                combined_candidate_id="akc-20260803-0100",
                combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
                views=fx.views(),
                recorded_anchor=fx.anchor,
                observed_anchor=None,
                storage_gb=1.0,
            )

    def test_a_commit_move_reaches_every_backend_the_tree_serves(self):
        """§1.5: CPU and GPU share the tree, so a commit move reaches both."""
        affected = C.affected_backends_for_move(_anchor(), self._moved())
        self.assertEqual(affected, ("llama_cpu", "llama_gpu"))

    def test_a_single_backend_binary_change_is_narrower(self):
        recorded = _anchor()
        observed = SM.AnchorIdentity(
            source_tree=recorded.source_tree,
            branch=recorded.branch,
            commit=recorded.commit,
            binary_sha256={**recorded.binary_sha256,
                           "llama_gpu": _sha("rebuilt-gpu-binary")},
            linkage_sha256=dict(recorded.linkage_sha256),
        )
        self.assertEqual(C.affected_backends_for_move(recorded, observed),
                         ("llama_gpu",))

    def test_comparisons_are_superseded_and_t0_survives(self):
        fx = self.composed_fixture()
        response = C.respond_to_anchor_move(
            recorded_anchor=fx.anchor, observed_anchor=self._moved(),
            entries=fx.journal.read_all())
        self.assertIn("ake-20260803-1001", response.sweep.superseded_record_ids)
        self.assertIn("ake-20260803-1000", response.sweep.preserved_record_ids)
        self.assertEqual(set(response.sweep.preserved_candidate_ids),
                         {"akc-20260803-0001", "akc-20260803-0002",
                          "akc-20260803-0100"})

    def test_supersession_payload_carries_both_identities(self):
        fx = self.composed_fixture()
        response = C.respond_to_anchor_move(
            recorded_anchor=fx.anchor, observed_anchor=self._moved(),
            entries=fx.journal.read_all())
        payload = response.sweep.payload_for(fx.entry_id("ake-20260803-1001"))
        self.assertEqual(payload["record_id"], "ake-20260803-1001")
        self.assertEqual(payload["tier"], "T1")
        self.assertTrue(payload[C.SUPERSEDED_BY_ANCHOR_MOVE])
        self.assertEqual(payload["old_anchor"]["commit"], V8_COMMIT)
        self.assertEqual(payload["new_anchor"]["commit"], V9_COMMIT)
        self.assertIn(C.SUPERSEDED_BY_ANCHOR_MOVE, payload["reason"])

    def test_applying_the_sweep_removes_comparisons_from_the_views_only(self):
        fx = self.composed_fixture()
        response = C.respond_to_anchor_move(
            recorded_anchor=fx.anchor, observed_anchor=self._moved(),
            entries=fx.journal.read_all())
        C.apply_anchor_move_supersession(fx.journal, response.sweep)
        views = fx.views()
        self.assertNotIn("ake-20260803-1001", views.evaluations)
        self.assertIn("ake-20260803-1000", views.evaluations)
        self.assertIn("akc-20260803-0100", views.candidates)
        # Invariant 8: the record survives the derived view.
        surviving = {e.record_id for e in fx.journal.read_all()}
        self.assertIn("ake-20260803-1001", surviving)

    def test_composition_is_impossible_after_the_sweep(self):
        """The comparisons died, so the champion cannot be rebuilt on them."""
        fx = self.composed_fixture()
        response = C.respond_to_anchor_move(
            recorded_anchor=fx.anchor, observed_anchor=self._moved(),
            entries=fx.journal.read_all())
        C.apply_anchor_move_supersession(fx.journal, response.sweep)
        with self.assertRaises(C.CompositionEvidenceMissing):
            C.compose_champion(
                self.lineage_of(fx),
                combined_candidate_id="akc-20260803-0100",
                combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
                views=fx.views(),
                recorded_anchor=fx.anchor,
                observed_anchor=_anchor(),
                storage_gb=1.0,
            )

    def test_sweep_refuses_to_supersede_a_t0_record(self):
        fx = self.composed_fixture()
        response = C.respond_to_anchor_move(
            recorded_anchor=fx.anchor, observed_anchor=self._moved(),
            entries=fx.journal.read_all())
        tainted = dataclasses.replace(
            response.sweep,
            superseded_entry_ids=response.sweep.superseded_entry_ids
            + (fx.entry_id("ake-20260803-1000"),))
        with self.assertRaises(C.SupersessionScopeViolation) as caught:
            C.apply_anchor_move_supersession(fx.journal, tainted)
        self.assertIn("correctness result", str(caught.exception))

    def test_sweep_refuses_to_supersede_a_candidate_record(self):
        fx = self.composed_fixture()
        response = C.respond_to_anchor_move(
            recorded_anchor=fx.anchor, observed_anchor=self._moved(),
            entries=fx.journal.read_all())
        tainted = dataclasses.replace(
            response.sweep,
            superseded_entry_ids=response.sweep.superseded_entry_ids
            + (fx.entry_id("akc-20260803-0001"),))
        with self.assertRaises(C.SupersessionScopeViolation) as caught:
            C.apply_anchor_move_supersession(fx.journal, tainted)
        self.assertIn("CANDIDATE_RECORDED", str(caught.exception))

    def test_a_candidate_with_no_declared_backends_is_swept_fail_closed(self):
        fx = self.composed_fixture()
        recorded = _anchor()
        observed = SM.AnchorIdentity(
            source_tree=recorded.source_tree, branch=recorded.branch,
            commit=recorded.commit,
            binary_sha256={**recorded.binary_sha256,
                           "llama_cpu": _sha("rebuilt-cpu-binary")},
            linkage_sha256=dict(recorded.linkage_sha256))
        response = C.respond_to_anchor_move(
            recorded_anchor=recorded, observed_anchor=observed, entries=fx.journal.read_all(),
            backends_by_candidate={"akc-20260803-0001": ["llama_gpu"]})
        self.assertIn("akc-20260803-0100", response.sweep.fail_closed_candidates)
        self.assertIn("ake-20260803-1001", response.sweep.superseded_record_ids)

    def test_a_declared_unaffected_backend_keeps_its_comparisons(self):
        fx = self.composed_fixture()
        recorded = _anchor()
        observed = SM.AnchorIdentity(
            source_tree=recorded.source_tree, branch=recorded.branch,
            commit=recorded.commit,
            binary_sha256={**recorded.binary_sha256,
                           "llama_cpu": _sha("rebuilt-cpu-binary")},
            linkage_sha256=dict(recorded.linkage_sha256))
        response = C.respond_to_anchor_move(
            recorded_anchor=recorded, observed_anchor=observed, entries=fx.journal.read_all(),
            backends_by_candidate={
                "akc-20260803-0001": ["llama_gpu"],
                "akc-20260803-0002": ["llama_gpu"],
                "akc-20260803-0100": ["llama_gpu"],
                "akc-20260803-base": ["llama_gpu"],
            })
        self.assertEqual(response.sweep.superseded_entry_ids, ())
        self.assertEqual(response.sweep.fail_closed_candidates, ())

    def test_response_carries_a_reanchor_plan_when_a_champion_exists(self):
        fx = self.composed_fixture()
        champion = C.compose_champion(
            self.lineage_of(fx),
            combined_candidate_id="akc-20260803-0100",
            combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
            views=fx.views(), recorded_anchor=fx.anchor,
            observed_anchor=_anchor(), storage_gb=1.0)
        response = C.respond_to_anchor_move(
            recorded_anchor=fx.anchor, observed_anchor=self._moved(),
            entries=fx.journal.read_all(), champion_record=champion)
        self.assertEqual(response.reanchor_plan.trigger,
                         C.REANCHOR_TRIGGER_ANCHOR_MOVED)
        self.assertEqual(response.reanchor_plan.rebase_sources,
                         ("akc-20260803-0001", "akc-20260803-0002"))

    def test_stop_request_satisfies_the_machines_anchor_moved_evidence(self):
        fx = self.composed_fixture()
        response = C.respond_to_anchor_move(
            recorded_anchor=fx.anchor, observed_anchor=self._moved(),
            entries=fx.journal.read_all())
        request = response.to_stop_request()
        self.assertEqual(request.state, SM.ANCHOR_MOVED)
        check = SM.check_stop_evidence(request.state, request.reason, request.detail)
        self.assertEqual(check.outcome, S.PASS, check.reasons)

    def test_operator_notice_is_the_four_part_decision_package(self):
        fx = self.composed_fixture()
        response = C.respond_to_anchor_move(
            recorded_anchor=fx.anchor, observed_anchor=self._moved(),
            entries=fx.journal.read_all())
        request = response.to_operator_input_request()
        self.assertEqual(request.state, SM.OPERATOR_INPUT_REQUIRED)
        check = SM.check_stop_evidence(request.state, request.reason, request.detail)
        self.assertEqual(check.outcome, S.PASS, check.reasons)
        self.assertGreaterEqual(len(request.detail["options"]), 2)

    def test_no_response_when_the_anchor_did_not_move(self):
        fx = self.composed_fixture()
        with self.assertRaises(C.CompositionError) as caught:
            C.respond_to_anchor_move(recorded_anchor=fx.anchor,
                                     observed_anchor=_anchor(), entries=fx.journal.read_all())
        self.assertIn("no move to respond to", str(caught.exception))

    def test_no_response_when_the_anchor_could_not_be_observed(self):
        fx = self.composed_fixture()
        with self.assertRaises(SM.AnchorUncheckable):
            C.respond_to_anchor_move(recorded_anchor=fx.anchor,
                                     observed_anchor=None, entries=fx.journal.read_all())

    def test_a_sweep_naming_no_affected_backend_is_refused(self):
        fx = self.composed_fixture()
        with self.assertRaises(ValueError):
            C.plan_anchor_move_supersession(
                fx.journal.read_all(), old_anchor=fx.anchor,
                new_anchor=self._moved(), affected_backends=())

    def test_stop_detail_names_the_halted_tree_and_both_identities(self):
        fx = self.composed_fixture()
        response = C.respond_to_anchor_move(
            recorded_anchor=fx.anchor, observed_anchor=self._moved(),
            entries=fx.journal.read_all())
        self.assertEqual(response.stop_detail["halted_source_tree"], "llama.cpp")
        self.assertEqual(response.stop_detail["recorded_anchor"]["commit"], V8_COMMIT)
        self.assertEqual(response.stop_detail["observed_anchor"]["commit"], V9_COMMIT)


# =============================================================================
# Type hygiene at the seams
# =============================================================================

class SeamTests(CompositionTestCase):

    def test_compose_champion_refuses_a_non_lineage(self):
        fx = self.composed_fixture()
        with self.assertRaises(TypeError):
            C.compose_champion(
                {"members": []}, combined_candidate_id="akc-20260803-0100",
                combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
                views=fx.views(), recorded_anchor=fx.anchor,
                observed_anchor=_anchor(), storage_gb=1.0)

    def test_compose_champion_refuses_a_non_views(self):
        fx = self.composed_fixture()
        with self.assertRaises(TypeError):
            C.compose_champion(
                self.lineage_of(fx), combined_candidate_id="akc-20260803-0100",
                combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
                views={"candidates": {}}, recorded_anchor=fx.anchor,
                observed_anchor=_anchor(), storage_gb=1.0)

    def test_record_champion_refuses_a_non_journal(self):
        with self.assertRaises(TypeError):
            C.record_champion(object(), {})

    def test_apply_sweep_refuses_a_non_sweep(self):
        fx = self.fixture()
        with self.assertRaises(TypeError):
            C.apply_anchor_move_supersession(fx.journal, object())

    def test_duplicate_member_is_refused(self):
        fx = self.fixture()
        fx.add_candidate("akc-20260803-0001")
        member = fx.frontier("akc-20260803-0001", "layout")
        with self.assertRaises(C.IncompatibleMember) as caught:
            C.propose_lineage([member, member], anchor_commit=V8_COMMIT)
        self.assertIn("twice", str(caught.exception))

    def test_a_malformed_candidate_record_raises(self):
        with self.assertRaises(C.CompositionError):
            C.admit_to_frontier({"schema": S.SCHEMA_CANDIDATE},
                                _reconciliation("akc-20260803-0001"),
                                mechanism_class="layout")

    def test_an_absent_candidate_record_raises(self):
        with self.assertRaises(TypeError):
            C.admit_to_frontier(None, _reconciliation("akc-20260803-0001"),
                                mechanism_class="layout")

    def test_errors_are_controller_errors(self):
        self.assertTrue(issubclass(C.CompositionError, SM.ControllerError))


# =============================================================================
# Red-team regressions (2026-08-03)
#
# Each test below is an exploit that WORKED against the first version of this
# module. They are grouped by the property that failed rather than by function,
# because that is the property a future edit has to keep.
# =============================================================================

class ChecksThatPassedOnNothingTests(CompositionTestCase):
    """A check you can satisfy by DELETING what it inspects is not a check."""

    def test_diversity_cannot_pass_by_emptying_the_producing_classes(self):
        """`available_classes=[]` made `reachable` zero, so every frontier cleared.

        The honest answer for the same frontier is FAIL; supplying an empty
        producing set turned that into PASS.
        """
        fx = self.fixture()
        fx.add_candidate("akc-20260803-0001")
        fx.add_candidate("akc-20260803-0002")
        frontier = [fx.frontier("akc-20260803-0001", "layout"),
                    fx.frontier("akc-20260803-0002", "layout")]
        self.assertEqual(
            C.check_mechanism_diversity(
                frontier, min_classes=3,
                available_classes=["layout", "fusion", "arithmetic"]).outcome,
            S.FAIL)
        starved = C.check_mechanism_diversity(frontier, min_classes=3,
                                              available_classes=[])
        self.assertEqual(starved.outcome, S.COULD_NOT_CHECK)
        self.assertIn("contradicts", " ".join(starved.reasons))

    def test_diversity_refuses_producing_classes_that_omit_a_held_class(self):
        """A frontier candidate is banked, so a class it holds MUST have produced one."""
        fx = self.fixture()
        fx.add_candidate("akc-20260803-0001")
        frontier = [fx.frontier("akc-20260803-0001", "layout")]
        check = C.check_mechanism_diversity(frontier, min_classes=2,
                                            available_classes=["fusion"])
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)

    def test_the_self_audit_cannot_pass_on_an_empty_source(self):
        """Handing the audit nothing matched no forbidden node, so it reported PASS."""
        for text in ("", "   \n\n  ", "# nothing to see\n"):
            with self.subTest(source=text):
                self.assertEqual(
                    C.audit_no_composed_estimate_arithmetic(text).outcome,
                    S.COULD_NOT_CHECK)

    def test_the_self_audit_catches_a_summation(self):
        """§12's row is *summed* local gains, and `sum()` contains no `*`."""
        check = C.audit_no_composed_estimate_arithmetic(
            "def combined(members):\n    return sum(m.value for m in members)\n")
        self.assertEqual(check.outcome, S.FAIL)
        self.assertTrue(any("sum" in r for r in check.reasons))

    def test_the_self_audit_catches_a_keyword_argument_and_a_definition(self):
        """`ast.keyword.arg` is a bare str, not an `ast.arg` — invisible before."""
        self.assertEqual(
            C.audit_no_composed_estimate_arithmetic(
                "def f(**kw):\n    return kw\n\nf(estimate=1)\n").outcome, S.FAIL)
        self.assertEqual(
            C.audit_no_composed_estimate_arithmetic(
                "def estimate(x):\n    return x\n").outcome, S.FAIL)

    def test_the_module_still_passes_its_own_audit(self):
        self.assertEqual(C.audit_no_composed_estimate_arithmetic().outcome, S.PASS)


class SweepIntegrityTests(CompositionTestCase):
    """The sweep's scope is re-derived from the JOURNAL, never from the sweep."""

    def _moved(self):
        return _anchor(commit=V9_COMMIT, binary="hotfix-binary",
                       linkage="hotfix-linkage")

    def _forged(self, fx, entry_id, record_id):
        """A publicly-constructible sweep whose own maps LIE about `entry_id`."""
        return C.AnchorMoveSweep(
            source_tree="llama.cpp",
            old_anchor=fx.anchor.to_dict(), new_anchor=self._moved().to_dict(),
            affected_backends=("llama_gpu",),
            reason="superseded_by_anchor_move: forged",
            superseded_entry_ids=(entry_id,), superseded_record_ids=(record_id,),
            preserved_entry_ids=(), preserved_record_ids=(),
            preserved_candidate_ids=(),
            entry_kind={entry_id: J.KIND_EVALUATION_EVENT},
            entry_tier={entry_id: "T1"},
            entry_record_id={entry_id: record_id},
            fail_closed_candidates=(),
        )

    def test_a_forged_sweep_cannot_supersede_a_candidate_record(self):
        """§8.9 item 3 preserves the work. The old check read the sweep's own map,
        so asserting `{entry: EVALUATION_EVENT/'T1'}` erased a candidate record
        permanently — `Journal.append` only checks that the target EXISTS."""
        fx = self.composed_fixture()
        entry_id = fx.entry_id("akc-20260803-0001")
        forged = self._forged(fx, entry_id, "akc-20260803-0001")
        with self.assertRaises(C.SupersessionScopeViolation):
            C.apply_anchor_move_supersession(fx.journal, forged)
        self.assertIn("akc-20260803-0001", fx.views().candidates)

    def test_a_forged_sweep_cannot_supersede_a_t0_correctness_event(self):
        fx = self.composed_fixture()
        entry_id = fx.entry_id("ake-20260803-1000")
        forged = self._forged(fx, entry_id, "ake-20260803-1000")
        with self.assertRaises(C.SupersessionScopeViolation) as caught:
            C.apply_anchor_move_supersession(fx.journal, forged)
        self.assertIn("T0", str(caught.exception))
        self.assertIn("ake-20260803-1000", fx.views().evaluations)

    def test_a_sweep_that_misdescribes_a_real_comparison_is_refused(self):
        """Even a legitimately-superseded entry is refused when the sweep's own
        description of it disagrees with the log."""
        fx = self.composed_fixture()
        entry_id = fx.entry_id("ake-20260803-1001")
        forged = C.AnchorMoveSweep(
            source_tree="llama.cpp",
            old_anchor=fx.anchor.to_dict(), new_anchor=self._moved().to_dict(),
            affected_backends=("llama_gpu",), reason="superseded_by_anchor_move",
            superseded_entry_ids=(entry_id,),
            superseded_record_ids=("ake-20260803-1001",),
            preserved_entry_ids=(), preserved_record_ids=(),
            preserved_candidate_ids=(),
            entry_kind={entry_id: J.KIND_EVALUATION_EVENT},
            entry_tier={entry_id: "T1"},
            entry_record_id={entry_id: "ake-99999999-0000"},   # the lie
            fail_closed_candidates=(),
        )
        with self.assertRaises(C.SupersessionScopeViolation):
            C.apply_anchor_move_supersession(fx.journal, forged)

    def test_an_empty_backend_declaration_does_not_narrow_the_sweep(self):
        """`{'akc-…': []}` intersected nothing, so the comparison was PRESERVED —
        a live ratio left pointing at a dead denominator, and `fail_closed` empty."""
        fx = self.composed_fixture()
        sweep = C.plan_anchor_move_supersession(
            fx.journal.read_all(), old_anchor=fx.anchor, new_anchor=self._moved(),
            affected_backends=("llama_cpu", "llama_gpu"),
            backends_by_candidate={"akc-20260803-0100": [],
                                   "akc-20260803-base": []})
        self.assertIn("ake-20260803-1001", sweep.superseded_record_ids)
        self.assertIn("akc-20260803-0100", sweep.fail_closed_candidates)

    def test_a_bare_string_backend_declaration_does_not_narrow_the_sweep(self):
        """`set('llama_gpu')` is a set of CHARACTERS and intersects no backend."""
        fx = self.composed_fixture()
        sweep = C.plan_anchor_move_supersession(
            fx.journal.read_all(), old_anchor=fx.anchor, new_anchor=self._moved(),
            affected_backends=("llama_cpu", "llama_gpu"),
            backends_by_candidate={"akc-20260803-0100": "llama_gpu",
                                   "akc-20260803-base": "llama_gpu"})
        self.assertIn("ake-20260803-1001", sweep.superseded_record_ids)
        self.assertIn("akc-20260803-0100", sweep.fail_closed_candidates)

    def test_an_undeclared_backend_name_does_not_narrow_the_sweep(self):
        fx = self.composed_fixture()
        sweep = C.plan_anchor_move_supersession(
            fx.journal.read_all(), old_anchor=fx.anchor, new_anchor=self._moved(),
            affected_backends=("llama_cpu", "llama_gpu"),
            backends_by_candidate={"akc-20260803-0100": ["gpu"],
                                   "akc-20260803-base": ["gpu"]})
        self.assertIn("akc-20260803-0100", sweep.fail_closed_candidates)

    def test_an_honest_declaration_still_narrows_the_sweep(self):
        """The widening must not swallow the feature: a candidate that genuinely
        does not touch an affected backend keeps its comparison."""
        fx = self.composed_fixture()
        sweep = C.plan_anchor_move_supersession(
            fx.journal.read_all(), old_anchor=fx.anchor, new_anchor=self._moved(),
            affected_backends=("llama_gpu",),
            backends_by_candidate={"akc-20260803-0100": ["llama_cpu"],
                                   "akc-20260803-base": ["llama_cpu"]})
        self.assertEqual(sweep.fail_closed_candidates, ())
        self.assertIn("ake-20260803-1001", sweep.preserved_record_ids)

    def test_the_supersession_payload_does_not_copy_the_candidate_list(self):
        """Copying every candidate id into every payload makes an append-only log
        grow with (comparisons superseded x candidates ever recorded)."""
        fx = self.composed_fixture()
        response = C.respond_to_anchor_move(
            recorded_anchor=fx.anchor, observed_anchor=self._moved(),
            entries=fx.journal.read_all())
        payload = response.sweep.payload_for(fx.entry_id("ake-20260803-1001"))
        self.assertNotIn("preserved_candidate_ids", payload)
        self.assertEqual(payload["preserved_candidate_count"],
                         len(response.sweep.preserved_candidate_ids))


class CombinedEvidenceSelectionTests(CompositionTestCase):
    """Which of the combined candidate's events becomes the champion's evidence."""

    def _compose(self, fx, **kwargs):
        return C.compose_champion(
            self.lineage_of(fx), combined_candidate_id="akc-20260803-0100",
            combined_reconciliation=fx.reconciliations["akc-20260803-0100"],
            views=fx.views(), recorded_anchor=fx.anchor,
            observed_anchor=fx.anchor, storage_gb=3.0, **kwargs)

    def test_a_later_failing_t1_is_not_answered_with_the_earlier_pass(self):
        """A re-run that FAILED was silently answered with the pass it re-ran."""
        fx = self.composed_fixture()
        fx.add_event("ake-20260803-1099", candidate_id="akc-20260803-0100",
                     tier="T1", status="fail",
                     created_at="2026-08-03T23:00:00+00:00")
        with self.assertRaises(C.CompositionEvidenceMissing) as caught:
            self._compose(fx)
        self.assertIn("most recent", str(caught.exception))

    def test_a_later_failing_t0_is_not_answered_with_the_earlier_pass(self):
        fx = self.composed_fixture()
        fx.add_event("ake-20260803-1098", candidate_id="akc-20260803-0100",
                     tier="T0", status="fail",
                     created_at="2026-08-03T23:00:00+00:00")
        with self.assertRaises(C.CompositionEvidenceMissing):
            self._compose(fx)

    def test_an_earlier_failing_t1_does_not_block_a_later_pass(self):
        """The rule is about the MOST RECENT measurement, not about any failure."""
        fx = self.composed_fixture()
        fx.add_event("ake-20260803-0900", candidate_id="akc-20260803-0100",
                     tier="T1", status="fail",
                     created_at="2026-08-03T09:00:00+00:00")
        record = self._compose(fx)
        self.assertEqual(record["last_t1"]["event_id"], "ake-20260803-1001")

    def test_a_failing_t2_is_carried_with_its_status_not_as_null(self):
        """`last_t2: null` means "never run". A composition whose interaction
        check FAILED is the one thing T2 exists to say (§9.7)."""
        fx = self.composed_fixture()
        fx.add_event("ake-20260803-1002", candidate_id="akc-20260803-0100",
                     tier="T2", status="fail",
                     created_at="2026-08-03T12:00:00+00:00")
        with self.assertRaises(C.CompositionError) as caught:
            self._compose(fx)
        self.assertIn("blocking_conditions", str(caught.exception))
        record = self._compose(fx, blocking_conditions=["T2_INTERACTION_FAILED"])
        self.assertEqual(record["last_t2"],
                         {"event_id": "ake-20260803-1002", "status": "fail"})

    def test_evidence_is_ordered_by_instant_not_by_timestamp_string(self):
        """`'+' < '.' < 'Z'`, so a lexicographic sort called the OLDER event the
        most recent whenever two legal ISO encodings met."""
        fx = self.composed_fixture()
        fx.add_event("ake-20260803-1060", candidate_id="akc-20260803-0100",
                     tier="T1", created_at="2026-08-03T20:00:00Z")
        fx.add_event("ake-20260803-1061", candidate_id="akc-20260803-0100",
                     tier="T1", created_at="2026-08-03T20:00:00.500000+00:00")
        self.assertEqual(self._compose(fx)["last_t1"]["event_id"],
                         "ake-20260803-1061")

    def test_evidence_ordering_respects_a_non_utc_offset(self):
        fx = self.composed_fixture()
        fx.add_event("ake-20260803-1070", candidate_id="akc-20260803-0100",
                     tier="T1", created_at="2026-08-03T23:00:00+09:00")  # 14:00Z
        fx.add_event("ake-20260803-1071", candidate_id="akc-20260803-0100",
                     tier="T1", created_at="2026-08-03T20:00:00+00:00")  # 20:00Z
        self.assertEqual(self._compose(fx)["last_t1"]["event_id"],
                         "ake-20260803-1071")

    def test_a_baseline_cell_is_not_the_compositions_own_evidence(self):
        """A BASELINE cell measures the ANCHOR. Selected as `last_t1` it made the
        composition its own denominator (invariant 15)."""
        fx = self.composed_fixture()
        event = _event("ake-20260803-1050", candidate_id="akc-20260803-0100",
                       tier="T1", anchor=fx.anchor,
                       created_at="2026-08-03T20:00:00+00:00")
        event["claim_grammar"]["category"] = "BASELINE"
        fx.journal.append(J.KIND_EVALUATION_EVENT, event)
        self.assertEqual(self._compose(fx)["last_t1"]["event_id"],
                         "ake-20260803-1001")

    def test_an_anchor_block_may_not_mix_two_backends_digests(self):
        """One backend's binary with another's linkage is a denominator that
        never existed on any host, assembled from two that did."""
        fx = self.fixture()
        fx.add_anchor_measurement("ake-20260801-0009")
        fx.add_candidate("akc-20260803-0001")
        fx.add_candidate("akc-20260803-0002")
        fx.add_candidate("akc-20260803-0100")
        for event_id, tier in (("ake-20260803-1000", "T0"),
                               ("ake-20260803-1001", "T1")):
            event = _event(event_id, candidate_id="akc-20260803-0100", tier=tier,
                           anchor=fx.anchor,
                           measurement_ids=([] if tier == "T0"
                                            else ["ake-20260801-0009"]))
            event["anchor"]["linkage_sha256"] = fx.anchor.linkage_sha256["llama_cpu"]
            fx.journal.append(J.KIND_EVALUATION_EVENT, event)
        with self.assertRaises(C.CompositionEvidenceMissing) as caught:
            self._compose(fx)
        self.assertIn("DIFFERENT backends", str(caught.exception))


class RetentionCountsArtifactsTests(CompositionTestCase):

    def test_retain_frontier_refuses_the_same_candidate_under_two_classes(self):
        """`mechanism_class` is supplied to `admit_to_frontier()`, so one banked
        candidate can be admitted twice under two labels. It then filled two
        class quotas by itself AND made the result exceed `capacity`."""
        fx = self.fixture()
        fx.add_candidate("akc-20260803-0001")
        fx.add_candidate("akc-20260803-0002")
        one = fx.frontier("akc-20260803-0001", "layout")
        relabelled = fx.frontier("akc-20260803-0001", "fusion")
        other = fx.frontier("akc-20260803-0002", "layout")
        with self.assertRaises(C.IncompatibleMember) as caught:
            C.retain_frontier([one, relabelled, other], capacity=2)
        self.assertIn("twice", str(caught.exception))

    def test_retention_never_returns_more_than_capacity(self):
        fx = self.fixture()
        for index in range(1, 6):
            fx.add_candidate(f"akc-20260803-000{index}")
        frontier = [fx.frontier(f"akc-20260803-000{index}",
                                "layout" if index < 4 else "fusion")
                    for index in range(1, 6)]
        for capacity in (2, 3, 4):
            with self.subTest(capacity=capacity):
                self.assertLessEqual(
                    len(C.retain_frontier(frontier, capacity=capacity)), capacity)

    def test_the_diversity_floor_still_holds_a_representative_of_each_class(self):
        fx = self.fixture()
        for index in range(1, 5):
            fx.add_candidate(f"akc-20260803-000{index}")
        frontier = [fx.frontier("akc-20260803-0001", "layout"),
                    fx.frontier("akc-20260803-0002", "layout"),
                    fx.frontier("akc-20260803-0003", "layout"),
                    fx.frontier("akc-20260803-0004", "fusion")]
        kept = C.retain_frontier(frontier, capacity=2)
        self.assertEqual({c.mechanism_class for c in kept}, {"layout", "fusion"})


class LineageBranchTests(CompositionTestCase):

    def test_a_supplied_champion_branch_must_be_namespaced(self):
        """A `ComposedLineage` could carry the FROZEN production branch name and
        only be caught later, inside `compose_champion()` (invariant 3)."""
        fx = self.fixture()
        fx.add_candidate("akc-20260803-0001")
        member = fx.frontier("akc-20260803-0001", "layout")
        with self.assertRaises(ValueError):
            C.propose_lineage([member], anchor_commit=V8_COMMIT,
                              branch="production-consolidated-v8")

    def test_a_namespaced_branch_is_still_accepted(self):
        fx = self.fixture()
        fx.add_candidate("akc-20260803-0001")
        member = fx.frontier("akc-20260803-0001", "layout")
        lineage = C.propose_lineage([member], anchor_commit=V8_COMMIT,
                                    branch="ak/champion/custom")
        self.assertEqual(lineage.branch, "ak/champion/custom")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
