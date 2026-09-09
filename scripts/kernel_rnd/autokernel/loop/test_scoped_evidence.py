from dataclasses import replace
import copy
import math

import pytest

from autokernel import schemas
from autokernel.loop import scoped_evidence as E


H = "a" * 64
H2 = "b" * 64
PHASES = ["setup", "placement", "load", "warmup", "steady", "bursts", "teardown"]


def scope(*, target="cpu", backend="llama_cpu", model="m", quant="q8",
          workload="decode", allocation="cores0-47"):
    return {"target": target, "backend": backend, "model": model, "quant": quant,
            "workload": workload, "allocation": allocation}


def claim_dict(*, target_scope=None, bound=2.0, dependencies=None,
               mechanism_digest=H):
    return {"schema": E.CLAIM_KEY_SCHEMA, "target_scope": target_scope or scope(),
            "control_identity": {"recipe": "anchor", "digest": H},
            "intervention_identity": {"recipe": "candidate", "digest": H2},
            "mechanism_identity": {"name": "thp", "implementation_digest": mechanism_digest},
            "estimand": "level", "metric": "tokens_per_second",
            "metric_direction": "higher",
            "effect_question": {"kind": "absolute_effect_bound", "bound": bound,
                                "unit": "percent"},
            "dependency_identities": dependencies or {"recipe:cpu": H}}


def finding_dict(claim, number, *, conclusion="positive", value=1.0,
                 epoch="epoch-1", frontier=None, raw_grade=None):
    return {"schema": E.FINDING_SCHEMA, "finding_id": f"finding-{number}",
            "source": {"schema": E.SOURCE_REF_SCHEMA, "event_id": f"event-{number}",
                       "artifact_digest": H, "locator": f"journal:{number}"},
            "claim_key": claim.to_dict(), "conclusion": conclusion, "value": value,
            "tested_scope": dict(claim.target_scope),
            "tested_question": dict(claim.effect_question),
            "raw_grade": raw_grade if raw_grade is not None else {"Q": "Witnessed", "T": "Attested"},
            "epoch": epoch, "record_class": "strict_search",
            "dependency_generations": {dep: 0 for dep in claim.dependency_identities},
            "intended_use_disposition": {"intended_use": "screen_out",
                                         "disposition": "certificate_candidate"},
            "authority_reference": "opaque-grade-ref", "frontier": number if frontier is None else frontier}


def finding(claim, number, **kwargs):
    return E.Finding.from_dict(finding_dict(claim, number, **kwargs))


def invalidation(event_id="inv-1", dependency="recipe:cpu", generation=1, frontier=100):
    return {"schema": E.INVALIDATION_SCHEMA, "event_id": event_id,
            "dependency_id": dependency, "generation": generation,
            "kind": "retraction", "frontier": frontier}


def trusted_index(findings, invalidations=(), **kwargs):
    return E.EvidenceIndex(
        findings, invalidations, current_epoch=kwargs.pop("current_epoch", "epoch-1"),
        scope_verifier=kwargs.pop("scope_verifier", lambda *_: True),
        use_verifier=kwargs.pop("use_verifier", lambda *_: True),
        result_verifier=kwargs.pop("result_verifier", lambda *_: True),
        support_rule_identity=kwargs.pop(
            "support_rule_identity", "test:registered-support:v1"), **kwargs)


def test_strict_immutable_claim_and_direct_constructor_revalidation():
    original = claim_dict()
    claim = E.ClaimKey.from_dict(original)
    digest = claim.digest
    original["target_scope"]["quant"] = "q4"
    assert claim.digest == digest and claim.target_scope["quant"] == "q8"
    with pytest.raises(TypeError):
        claim.target_scope["quant"] = "q4"
    bad = replace(claim, effect_question={"kind": "absolute_effect_bound",
                                          "bound": True, "unit": "percent"})
    with pytest.raises(E.EvidenceValidationError, match="finite number"):
        E.EvidenceIndex((), current_epoch="epoch-1").retrieve(
            bad.target_scope, bad, "explore")


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(schema="unknown.v2"),
    lambda c: c.update(extra=True),
    lambda c: c["effect_question"].update(bound=math.nan),
    lambda c: c.update(dependency_identities={"recipe:cpu": "bad"}),
])
def test_claim_rejects_unknown_nonfinite_and_malformed(mutation):
    obj = claim_dict()
    mutation(obj)
    with pytest.raises(E.EvidenceValidationError):
        E.ClaimKey.from_dict(obj)


def test_refutation_at_position_41_is_mandatory_before_top_k():
    claim = E.ClaimKey.from_dict(claim_dict())
    rows = [finding(claim, i, value=float(i)) for i in range(41)]
    rows.append(finding(claim, 41, conclusion="refutation", value=-5.0))
    result = trusted_index(rows).retrieve(scope(), claim, "screen_out", limit=40)
    assert len(result.findings) == 40
    assert [row.finding.finding_id for row in result.mandatory_conflicts] == ["finding-41"]


def test_cross_quant_and_mechanism_alias_do_not_match():
    q8 = E.ClaimKey.from_dict(claim_dict())
    q4 = E.ClaimKey.from_dict(claim_dict(target_scope=scope(quant="q4")))
    other_mechanism = E.ClaimKey.from_dict(claim_dict(mechanism_digest=H2))
    index = trusted_index([finding(q8, 1)])
    assert not index.retrieve(q4.target_scope, q4, "explore").findings
    assert not index.retrieve(other_mechanism.target_scope, other_mechanism,
                              "explore").findings


def test_broader_scope_refutation_is_mandatory_only_after_trusted_applicability():
    exact = E.ClaimKey.from_dict(claim_dict())
    broad = E.ClaimKey.from_dict(claim_dict(
        target_scope=scope(target="cpu-family", allocation="all-cpu")))
    calls = []

    def applies(row, query, requested_scope):
        calls.append((row.finding_id, query.digest, dict(requested_scope)))
        return True

    index = trusted_index(
        [finding(broad, 50, conclusion="refutation", value=-2.0)],
        scope_verifier=applies)
    result = index.retrieve(scope(), exact, "screen_out")
    assert [row.finding.finding_id for row in result.mandatory_conflicts] == [
        "finding-50"]
    assert result.mandatory_conflicts[0].applicability == "trusted_broader_scope"
    assert calls and calls[0][1] == exact.digest


def test_broader_mandatory_dependencies_are_bound_into_proposal():
    exact = E.ClaimKey.from_dict(claim_dict())
    broad = E.ClaimKey.from_dict(claim_dict(
        target_scope=scope(target="cpu-family", allocation="all-cpu"),
        dependencies={"broader-topology": H2}))
    index = trusted_index([finding(exact, 1),
                           finding(broad, 2, conclusion="refutation")])
    proposal = index.proposal_snapshot(exact, intended_use="screen_out")
    assert set(proposal.dependency_generations) == {
        "recipe:cpu", "broader-topology"}
    assert proposal.dependency_frontiers["broader-topology"] == 2


def test_broader_candidate_with_different_quant_or_bound_is_not_considered():
    exact = E.ClaimKey.from_dict(claim_dict())
    wrong_quant = E.ClaimKey.from_dict(claim_dict(
        target_scope=scope(target="cpu-family", quant="q4", allocation="all-cpu")))
    wrong_bound = E.ClaimKey.from_dict(claim_dict(
        target_scope=scope(target="cpu-family", allocation="all-cpu"), bound=3.0))
    calls = []
    index = trusted_index(
        [finding(wrong_quant, 51, conclusion="refutation"),
         finding(wrong_bound, 52, conclusion="refutation")],
        scope_verifier=lambda *args: calls.append(args) or True)
    result = index.retrieve(scope(), exact, "screen_out")
    assert not result.mandatory_conflicts
    assert not calls


def test_cross_epoch_number_is_preserved_but_never_ranked():
    claim = E.ClaimKey.from_dict(claim_dict())
    old = finding(claim, 1, value=9.5, epoch="epoch-0")
    row = trusted_index([old]).retrieve(scope(), claim, "explore").findings[0]
    assert row.finding.value == 9.5
    assert row.ranking_value is None and row.magnitude_status == "stale_cross_epoch"


def test_same_epoch_search_numbers_rank_only_through_shared_use_verifier():
    claim = E.ClaimKey.from_dict(claim_dict())
    rows = [finding(claim, 1, value=1.0), finding(claim, 2, value=4.0)]
    trusted = trusted_index(rows).retrieve(scope(), claim, "rank")
    assert [row.ranking_value for row in trusted.findings] == [4.0, 1.0]
    untrusted = E.EvidenceIndex(rows, current_epoch="epoch-1").retrieve(
        scope(), claim, "rank")
    assert not untrusted.complete_for_intended_use
    assert all(row.ranking_value is None for row in untrusted.findings)


def test_lower_is_better_ranking_uses_declared_metric_direction():
    obj = claim_dict()
    obj["metric_direction"] = "lower"
    claim = E.ClaimKey.from_dict(obj)
    rows = [finding(claim, 1, value=4.0), finding(claim, 2, value=1.0)]
    result = trusted_index(rows).retrieve(scope(), claim, "rank")
    assert [row.ranking_value for row in result.findings] == [1.0, 4.0]


def test_null_applies_only_to_same_key_and_effect_bound():
    tested = E.ClaimKey.from_dict(claim_dict(bound=2.0))
    changed = E.ClaimKey.from_dict(claim_dict(bound=1.0))
    null = finding(tested, 1, conclusion="null", value=.2)
    index = trusted_index([null])
    row = index.retrieve(scope(), tested, "explore").findings[0]
    assert row.magnitude_status == "tested_null" and row.ranking_value is None
    assert not index.retrieve(scope(), changed, "explore").findings


def test_raw_grade_pass_is_opaque_not_certificate_authority():
    claim = E.ClaimKey.from_dict(claim_dict())
    row = finding(claim, 1, raw_grade={"grade": "PASS", "authority": True})
    index = E.EvidenceIndex([row], current_epoch="epoch-1")
    result = index.retrieve(scope(), claim, "screen_out")
    assert not result.complete_for_intended_use
    assert "trusted registered" in " ".join(result.reasons)


def test_complete_retrieval_is_separate_from_supported_certificate_use():
    claim = E.ClaimKey.from_dict(claim_dict())
    row = finding(claim, 1, raw_grade={"grade": "PASS", "certificate": True})
    index = E.EvidenceIndex(
        [row], current_epoch="epoch-1", scope_verifier=lambda *_: True,
        use_verifier=lambda *_: True)
    result = index.retrieve(scope(), claim, "screen_out")
    assert result.retrieval_complete
    assert not result.supported_for_intended_use
    assert not result.complete_for_intended_use


def test_trusted_full_result_verifier_receives_mandatory_context():
    claim = E.ClaimKey.from_dict(claim_dict())
    negative = finding(claim, 2, conclusion="refutation", value=-1.0)
    seen = []

    def registered(ordinary, mandatory, checked_claim, intended_use):
        seen.append((ordinary, mandatory, checked_claim.digest, intended_use))
        return len(mandatory) == 1 and mandatory[0].finding.conclusion == "refutation"

    result = trusted_index([negative], result_verifier=registered).retrieve(
        scope(), claim, "screen_out")
    assert result.complete_for_intended_use
    assert seen[0][2:] == (claim.digest, "screen_out")


def test_empty_or_scope_refused_certificate_is_not_vacuously_complete():
    claim = E.ClaimKey.from_dict(claim_dict())
    empty = trusted_index([]).retrieve(scope(), claim, "screen_out")
    assert not empty.complete_for_intended_use
    assert "no applicable findings" in " ".join(empty.reasons)
    refused = trusted_index(
        [finding(claim, 1)], scope_verifier=lambda *_: "scope refused").retrieve(
            scope(), claim, "screen_out")
    assert not refused.complete_for_intended_use
    assert refused.findings[0].applicability == "scope_unverified"


def test_duplicate_ids_and_boolean_verifier_injection_are_rejected():
    claim = E.ClaimKey.from_dict(claim_dict())
    row = finding(claim, 1)
    with pytest.raises(E.EvidenceValidationError, match="duplicate finding_id"):
        E.EvidenceIndex([row, row], current_epoch="epoch-1")
    with pytest.raises(E.EvidenceValidationError, match="trusted callable"):
        E.EvidenceIndex([row], current_epoch="epoch-1", scope_verifier=True)


def test_local_retraction_stales_finding_and_cached_proposal():
    claim = E.ClaimKey.from_dict(claim_dict())
    row = finding(claim, 1)
    before = trusted_index([row])
    proposal = before.proposal_snapshot(claim, intended_use="screen_out")
    after = trusted_index([row], [invalidation()])
    result = after.retrieve(scope(), claim, "screen_out")
    assert not result.complete_for_intended_use
    assert result.findings[0].applicability == "stale_dependency"
    admission = E.EvidenceIndex.admit_cached(
        proposal, after.fence_snapshot(), intended_use="screen_out",
        authority_verifier=lambda _: True)
    assert admission.status == "stale" and admission.affected_dependencies == ("recipe:cpu",)


def test_malformed_scoped_and_global_invalidations_fail_closed_for_certificates():
    claim = E.ClaimKey.from_dict(claim_dict())
    row = finding(claim, 1)
    scoped = dict(invalidation())
    scoped["generation"] = True
    index = trusted_index([row], [scoped])
    assert index.quarantines[0].affected_dependencies == ("recipe:cpu",)
    assert not index.retrieve(scope(), claim, "screen_out").complete_for_intended_use
    exploratory = index.retrieve(scope(), claim, "explore")
    assert exploratory.supported_for_intended_use
    assert not exploratory.retrieval_complete

    global_bad = {"schema": "bad.v9", "event_id": "bad-global", "frontier": 101}
    global_index = trusted_index([row], [global_bad])
    assert global_index.quarantines[0].global_scope
    assert not global_index.retrieve(scope(), claim, "screen_out").complete_for_intended_use


def test_duplicate_id_idempotence_conflict_and_out_of_order_quarantine():
    claim = E.ClaimKey.from_dict(claim_dict())
    event = invalidation()
    exact = trusted_index([finding(claim, 1)], [event, dict(event)])
    assert not exact.quarantines and exact.dependency_generations["recipe:cpu"] == 1
    conflicting = dict(event)
    conflicting["generation"] = 2
    conflict = trusted_index([finding(claim, 1)], [event, conflicting])
    assert conflict.quarantines and "reused" in conflict.quarantines[0].reason
    gap = trusted_index([finding(claim, 1)], [invalidation(generation=2)])
    assert "out-of-order" in gap.quarantines[0].reason
    malformed = dict(invalidation(event_id="malformed-repeat"))
    malformed["generation"] = True
    repeated = trusted_index([finding(claim, 1)], [malformed, dict(malformed)])
    assert len(repeated.quarantines) == 1
    assert repeated.fence_snapshot().dependency_fence_generations["recipe:cpu"] == 1


def test_restart_replay_preserves_quarantine_and_projection():
    claim = E.ClaimKey.from_dict(claim_dict())
    bad = dict(invalidation())
    bad["kind"] = "unknown"
    first = trusted_index([finding(claim, 1)], [bad])
    second = trusted_index([finding(claim, 1)], [bad])
    assert first.to_dict() == second.to_dict()
    assert first.quarantines == second.quarantines


def test_serialized_projection_restores_quarantines_outage_and_fences_not_authority():
    claim = E.ClaimKey.from_dict(claim_dict())
    scoped = dict(invalidation(event_id="bad-scoped"))
    scoped["generation"] = True
    global_bad = {"schema": "bad.v9", "event_id": "bad-global", "frontier": 105}
    original = trusted_index(
        [finding(claim, 1)], [scoped, global_bad], projection_available=False)
    restored = E.EvidenceIndex.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    assert not restored.fence_snapshot().available
    assert restored.fence_snapshot().global_fence_generation == 1
    assert restored.fence_snapshot().support_rule_identity is None
    assert restored.fence_snapshot().dependency_fence_generations["recipe:cpu"] == 1
    certificate = restored.retrieve(scope(), claim, "screen_out")
    assert certificate.retrieval_complete is False
    assert certificate.supported_for_intended_use is False


def test_projection_restore_rejects_tampered_fence_even_with_rehashed_body():
    claim = E.ClaimKey.from_dict(claim_dict())
    projected = trusted_index([finding(claim, 1)]).to_dict()
    projected["dependency_fence_generations"]["recipe:cpu"] = 9
    unsigned = {key: value for key, value in projected.items() if key != "index_digest"}
    projected["index_digest"] = schemas.content_hash(unsigned)
    with pytest.raises(E.EvidenceValidationError, match="fences do not replay"):
        E.EvidenceIndex.from_dict(projected)


def test_projection_restore_rederives_content_and_semantic_fences_with_explicit_context():
    claim = E.ClaimKey.from_dict(claim_dict())
    original = trusted_index([finding(claim, 1)])
    restored = E.EvidenceIndex.from_dict(
        original.to_dict(), scope_verifier=lambda *_: True,
        use_verifier=lambda *_: True, result_verifier=lambda *_: True,
        support_rule_identity="test:registered-support:v1")
    assert restored.fence_snapshot().to_dict() == original.fence_snapshot().to_dict()
    proposal = original.proposal_snapshot(claim, intended_use="screen_out")
    assert E.EvidenceIndex.admit_cached(
        proposal, restored.fence_snapshot(), intended_use="screen_out",
        authority_verifier=lambda _: True).status == "eligible"


def test_finding_future_generation_is_unknown_not_current():
    claim = E.ClaimKey.from_dict(claim_dict())
    obj = finding_dict(claim, 1)
    obj["dependency_generations"]["recipe:cpu"] = 1
    result = trusted_index([E.Finding.from_dict(obj)]).retrieve(
        scope(), claim, "screen_out")
    assert result.findings[0].applicability == "stale_dependency"
    assert not result.supported_for_intended_use


def test_unrelated_generation_does_not_stale_cached_proposal():
    claim = E.ClaimKey.from_dict(claim_dict())
    index = trusted_index([finding(claim, 1)])
    proposal = index.proposal_snapshot(claim, intended_use="screen_out")
    unrelated_bad = dict(invalidation(event_id="other-bad", dependency="unrelated"))
    unrelated_bad["generation"] = True
    local = trusted_index([finding(claim, 1)], [unrelated_bad]).fence_snapshot()
    result = E.EvidenceIndex.admit_cached(
        proposal, local, intended_use="screen_out",
        authority_verifier=lambda _: True)
    assert result.status == "eligible"


def test_new_relevant_finding_stales_cache_but_unrelated_finding_does_not():
    claim = E.ClaimKey.from_dict(claim_dict())
    first = finding(claim, 1)
    proposal = trusted_index([first]).proposal_snapshot(
        claim, intended_use="screen_out")
    relevant = trusted_index([first, finding(claim, 2)])
    stale = E.EvidenceIndex.admit_cached(
        proposal, relevant.fence_snapshot(), intended_use="screen_out",
        authority_verifier=lambda _: True)
    assert stale.status == "stale"

    unrelated_claim = E.ClaimKey.from_dict(claim_dict(
        target_scope=scope(target="gpu", backend="other", allocation="gpu0"),
        dependencies={"other-device": H2}))
    unrelated = trusted_index([first, finding(unrelated_claim, 3)])
    eligible = E.EvidenceIndex.admit_cached(
        proposal, unrelated.fence_snapshot(), intended_use="screen_out",
        authority_verifier=lambda _: True)
    assert eligible.status == "eligible"


def test_cached_numeric_rank_refuses_changed_epoch():
    claim = E.ClaimKey.from_dict(claim_dict())
    base = trusted_index([finding(claim, 1)])
    proposal = base.proposal_snapshot(claim, intended_use="rank")
    changed = trusted_index(
        [finding(claim, 1)], current_epoch="epoch-2",
        result_verifier=lambda *_: False)
    result = E.EvidenceIndex.admit_cached(
        proposal, changed.fence_snapshot(), intended_use="rank",
        authority_verifier=lambda _: True)
    assert result.status == "stale"
    assert result.reasons == ("support epoch changed",)


def test_new_broader_conflict_changes_semantic_fence_without_prior_dependency():
    claim = E.ClaimKey.from_dict(claim_dict())
    base = trusted_index([finding(claim, 1)])
    proposal = base.proposal_snapshot(claim, intended_use="screen_out")
    broad = E.ClaimKey.from_dict(claim_dict(
        target_scope=scope(target="all", allocation="all"),
        dependencies={"other:dep": H}))
    changed = trusted_index(
        [finding(claim, 1), finding(broad, 2, conclusion="refutation")],
        result_verifier=lambda *_: False)
    result = E.EvidenceIndex.admit_cached(
        proposal, changed.fence_snapshot(), intended_use="screen_out",
        authority_verifier=lambda _: True)
    assert result.status == "stale"
    assert result.reasons == ("relevant semantic retrieval-set fence changed",)


def test_semantic_empty_bucket_is_explicit_and_missing_bucket_is_incomplete():
    claim = E.ClaimKey.from_dict(claim_dict())
    base = trusted_index([finding(claim, 1)])
    proposal = base.proposal_snapshot(claim, intended_use="screen_out")
    fences = base.fence_snapshot()
    semantic_key = next(iter(proposal.semantic_fences))
    assert fences.semantic_fences[semantic_key] == schemas.content_hash([])
    assert E.EvidenceIndex.admit_cached(
        proposal, fences, intended_use="screen_out",
        authority_verifier=lambda _: True).status == "eligible"

    missing = replace(fences, semantic_fences={})
    result = E.EvidenceIndex.admit_cached(
        proposal, missing, intended_use="screen_out",
        authority_verifier=lambda _: True)
    assert result.status == "incomplete"
    assert result.reasons == ("local semantic retrieval-set fence is missing",)


def test_removed_relevant_finding_changes_content_digest_with_same_max_frontier():
    claim = E.ClaimKey.from_dict(claim_dict())
    base = trusted_index([finding(claim, 1), finding(claim, 2)])
    proposal = base.proposal_snapshot(claim, intended_use="screen_out")
    changed = trusted_index([finding(claim, 2)], result_verifier=lambda *_: False)
    assert changed.fence_snapshot().dependency_frontiers["recipe:cpu"] == 2
    result = E.EvidenceIndex.admit_cached(
        proposal, changed.fence_snapshot(), intended_use="screen_out",
        authority_verifier=lambda _: True)
    assert result.status == "stale"
    assert result.affected_dependencies == ("recipe:cpu",)


def test_changed_or_missing_registered_support_rule_refuses_cached_use():
    claim = E.ClaimKey.from_dict(claim_dict())
    base = trusted_index([finding(claim, 1)], support_rule_identity="rule:v1")
    proposal = base.proposal_snapshot(claim, intended_use="screen_out")
    changed = trusted_index([finding(claim, 1)], support_rule_identity="rule:v2")
    changed_result = E.EvidenceIndex.admit_cached(
        proposal, changed.fence_snapshot(), intended_use="screen_out",
        authority_verifier=lambda _: True)
    assert changed_result.status == "incomplete"
    missing = E.EvidenceIndex(
        [finding(claim, 1)], current_epoch="epoch-1",
        scope_verifier=lambda *_: True, use_verifier=lambda *_: True,
        result_verifier=lambda *_: True)
    missing_result = E.EvidenceIndex.admit_cached(
        proposal, missing.fence_snapshot(), intended_use="screen_out",
        authority_verifier=lambda _: True)
    assert missing_result.status == "incomplete"


def test_cached_admission_missing_or_unavailable_local_state_fails_closed():
    claim = E.ClaimKey.from_dict(claim_dict())
    proposal = trusted_index([finding(claim, 1)]).proposal_snapshot(
        claim, intended_use="screen_out")
    empty = E.LocalFenceSnapshot(
        E.LOCAL_FENCE_SCHEMA, True, 0, {}, {}, {}, {}, {}, "epoch-1",
        "test:registered-support:v1", 1)
    missing = E.EvidenceIndex.admit_cached(
        proposal, empty, intended_use="screen_out", authority_verifier=lambda _: True)
    assert missing.status == "incomplete"
    assert missing.affected_dependencies == ("recipe:cpu",)
    unavailable = replace(empty, available=False)
    result = E.EvidenceIndex.admit_cached(
        proposal, unavailable, intended_use="screen_out",
        authority_verifier=lambda _: True)
    assert result.status == "incomplete" and "unavailable" in result.reasons[0]
    malformed = replace(empty, global_fence_generation=True)
    with pytest.raises(E.EvidenceValidationError, match="integer"):
        E.EvidenceIndex.admit_cached(
            proposal, malformed, intended_use="screen_out",
            authority_verifier=lambda _: True)


def test_malformed_scoped_or_global_event_stales_prior_cached_proposal():
    claim = E.ClaimKey.from_dict(claim_dict())
    row = finding(claim, 1)
    proposal = trusted_index([row]).proposal_snapshot(claim, intended_use="screen_out")
    scoped = dict(invalidation(event_id="malformed-scoped"))
    scoped["generation"] = True
    scoped_result = E.EvidenceIndex.admit_cached(
        proposal, trusted_index([row], [scoped]).fence_snapshot(),
        intended_use="screen_out", authority_verifier=lambda _: True)
    assert scoped_result.status == "stale"
    assert scoped_result.affected_dependencies == ("recipe:cpu",)
    global_bad = {"schema": "bad.v9", "event_id": "malformed-global", "frontier": 8}
    global_result = E.EvidenceIndex.admit_cached(
        proposal, trusted_index([row], [global_bad]).fence_snapshot(),
        intended_use="screen_out", authority_verifier=lambda _: True)
    assert global_result.status == "stale"
    assert "global uncertainty" in global_result.reasons[0]


def test_cached_admission_binds_intended_use_support_and_trusted_callback():
    claim = E.ClaimKey.from_dict(claim_dict())
    index = trusted_index([finding(claim, 1)])
    explore = index.proposal_snapshot(claim, intended_use="explore")
    mismatch = E.EvidenceIndex.admit_cached(
        explore, index.fence_snapshot(), intended_use="screen_out",
        authority_verifier=lambda _: True)
    assert mismatch.status == "incomplete"
    no_callback = E.EvidenceIndex.admit_cached(
        explore, index.fence_snapshot(), intended_use="explore",
        authority_verifier=None)
    assert no_callback.status == "incomplete"
    forged = replace(explore, supported_for_intended_use=True,
                     retrieval_complete=True)
    assert E.EvidenceIndex.admit_cached(
        forged, index.fence_snapshot(), intended_use="explore",
        authority_verifier=None).status == "incomplete"


def test_unknown_intended_use_never_bypasses_verification():
    claim = E.ClaimKey.from_dict(claim_dict())
    with pytest.raises(E.EvidenceValidationError, match="unsupported"):
        trusted_index([finding(claim, 1)]).retrieve(scope(), claim, "screen-otu")


def test_direct_nested_tuple_inputs_are_deep_frozen_at_public_boundary():
    base = E.ClaimKey.from_dict(claim_dict())
    nested = [{"modes": ["a", "b"]}]
    direct = replace(base, control_identity={"nested": tuple(nested)})
    expected = E.ClaimKey.from_dict(copy.deepcopy(direct.to_dict())).digest
    proposal = trusted_index([]).proposal_snapshot(direct, intended_use="explore")
    nested[0]["modes"].append("mutated")
    assert proposal.claim_digest == expected


def test_projection_outage_retains_local_fence_and_allows_exploration():
    claim = E.ClaimKey.from_dict(claim_dict())
    index = trusted_index([finding(claim, 1)], [invalidation()],
                          projection_available=False)
    cert = index.retrieve(scope(), claim, "screen_out")
    assert not cert.complete_for_intended_use
    assert cert.findings[0].applicability == "stale_dependency"
    exploratory = index.retrieve(scope(), claim, "explore")
    assert exploratory.supported_for_intended_use
    assert not exploratory.retrieval_complete


class GetOnlyGenerations(dict):
    def items(self):  # pragma: no cover - failure path is the assertion
        raise AssertionError("fast path scanned local generations")

    def __iter__(self):  # pragma: no cover
        raise AssertionError("fast path scanned local generations")


def test_admission_fastpath_only_reads_declared_dependencies():
    claim = E.ClaimKey.from_dict(claim_dict())
    proposal = trusted_index([finding(claim, 1)]).proposal_snapshot(
        claim, intended_use="screen_out")
    generations = GetOnlyGenerations(
        {"recipe:cpu": 0, **{f"other:{i}": i for i in range(1000)}})
    fences = GetOnlyGenerations(
        {"recipe:cpu": 0, **{f"other:{i}": i for i in range(1000)}})
    frontiers = GetOnlyGenerations(
        {"recipe:cpu": 1, **{f"other:{i}": i for i in range(1000)}})
    evidence_digests = GetOnlyGenerations(
        {"recipe:cpu": proposal.dependency_evidence_digests["recipe:cpu"],
         **{f"other:{i}": H for i in range(1000)}})
    semantic_fences = GetOnlyGenerations(dict(proposal.semantic_fences))
    local = E.LocalFenceSnapshot(
        E.LOCAL_FENCE_SCHEMA, True, 0, generations, fences, frontiers,
        evidence_digests, semantic_fences, "epoch-1",
        "test:registered-support:v1", 1)
    result = E.EvidenceIndex.admit_cached(
        proposal, local, intended_use="screen_out",
        authority_verifier=lambda _: True)
    assert result.status == "eligible"


def transfer_dict(source, destination, *, transfer_type="serving_effect",
                  receipt_id="transfer-1"):
    return {"schema": E.TRANSFER_SCHEMA, "receipt_id": receipt_id,
            "source_scope": dict(source.target_scope),
            "destination_scope": dict(destination.target_scope),
            "transfer_type": transfer_type,
            "mechanism_identity": dict(source.mechanism_identity),
            "intervention_identity": dict(source.intervention_identity),
            "effect_question": dict(source.effect_question),
            "dependency_identities": dict(destination.dependency_identities),
            "authority_reference": "registered:transfer:v1",
            "source_ref": {"schema": E.SOURCE_REF_SCHEMA, "event_id": receipt_id,
                           "artifact_digest": H, "locator": "journal:transfer"}}


def three_claims():
    a = E.ClaimKey.from_dict(claim_dict(target_scope=scope(target="A")))
    b = E.ClaimKey.from_dict(claim_dict(target_scope=scope(target="B")))
    c = E.ClaimKey.from_dict(claim_dict(target_scope=scope(target="C")))
    return a, b, c


def test_transfer_is_direct_nontransitive_and_asymmetric():
    a, b, c = three_claims()
    ab = E.TransferReceipt.from_dict(transfer_dict(a, b, receipt_id="ab"))
    bc = E.TransferReceipt.from_dict(transfer_dict(b, c, receipt_id="bc"))
    assert E.transfer_disposition([ab], a, b, "timing",
                                  authority_verifier=lambda *_: True).status == "permitted"
    assert E.transfer_disposition([ab, bc], a, c, "timing",
                                  authority_verifier=lambda *_: True).status == "unsupported"
    assert E.transfer_disposition([ab], b, a, "timing",
                                  authority_verifier=lambda *_: True).status == "unsupported"


def test_correctness_transfer_never_authorizes_timing_and_labels_are_not_authority():
    a, b, _ = three_claims()
    correctness = E.TransferReceipt.from_dict(
        transfer_dict(a, b, transfer_type="correctness"))
    assert E.transfer_disposition([correctness], a, b, "timing",
                                  authority_verifier=lambda *_: True).status == "refused"
    timing = E.TransferReceipt.from_dict(transfer_dict(a, b))
    result = E.transfer_disposition([timing], a, b, "timing", authority_verifier=None)
    assert result.status == "policy_undefined"
    with pytest.raises(E.EvidenceValidationError, match="trusted callable"):
        E.transfer_disposition([timing], a, b, "timing", authority_verifier=True)
    with pytest.raises(E.EvidenceValidationError, match="unsupported"):
        E.transfer_disposition([timing], a, b, "timign",
                               authority_verifier=lambda *_: True)


def test_authority_callback_exception_fails_closed():
    a, b, _ = three_claims()
    timing = E.TransferReceipt.from_dict(transfer_dict(a, b))

    def broken(*_):
        raise RuntimeError("adapter outage")

    result = E.transfer_disposition([timing], a, b, "timing",
                                    authority_verifier=broken)
    assert result.status == "refused"
    assert "failed closed" in result.reasons[0]


def route_dict(source, destination, *, route_id="route-1", revision="r1"):
    return {"schema": E.ROUTE_SCHEMA, "route_id": route_id, "revision": revision,
            "source_scope": dict(source.target_scope),
            "destination_scope": dict(destination.target_scope),
            "mechanism_identity": dict(source.mechanism_identity),
            "intervention_identity": dict(source.intervention_identity),
            "dependency_identities": dict(destination.dependency_identities),
            "preserved_dimensions": ["model", "workload"],
            "required_path_witnesses": ["executed_path"],
            "covered_targets": [dict(destination.target_scope)],
            "disposition": {"kind": "may_screen_out",
                            "scope": dict(destination.target_scope),
                            "effect_question": dict(destination.effect_question)}}


def witness(event="path-1"):
    return E.SourceRef.from_dict({"schema": E.SOURCE_REF_SCHEMA, "event_id": event,
                                  "artifact_digest": H, "locator": "journal:path"})


def no_audit(claim, route):
    return E.select_reject_audit(
        claim, route, probability=0.0, mechanism_stratum="thp",
        allocation_stratum="cpu-48", target_confirmation_budget_id="audit-1",
        budget_remaining=0)


def test_route_requires_executed_path_verifier_not_actor_annotation():
    a, b, _ = three_claims()
    route = E.Route.from_dict(route_dict(a, b))
    missing = E.route_disposition(route, a, b, path_witnesses={},
                                  dependency_generation=0,
                                  authority_verifier=lambda *_: True)
    assert missing.status == "exploration_only"
    labelled = E.route_disposition(route, a, b,
                                   path_witnesses={"executed_path": witness()},
                                   dependency_generation=0, authority_verifier=None)
    assert labelled.status == "exploration_only"
    trusted = E.route_disposition(route, a, b,
                                  path_witnesses={"executed_path": witness()},
                                  dependency_generation=0,
                                  authority_verifier=lambda *_: True,
                                  audit_decision=no_audit(b, route))
    assert trusted.status == "may_screen_out"


def test_reject_audit_selection_is_stable_and_budget_absence_visible():
    a, b, _ = three_claims()
    route = E.Route.from_dict(route_dict(a, b))
    kwargs = {"probability": 1.0, "mechanism_stratum": "thp",
              "allocation_stratum": "cpu-48", "target_confirmation_budget_id": "audit-1"}
    first = E.select_reject_audit(b, route, budget_remaining=1, **kwargs)
    restarted = E.select_reject_audit(b, route, budget_remaining=1, **kwargs)
    exhausted = E.select_reject_audit(b, route, budget_remaining=0, **kwargs)
    assert first.to_dict() == restarted.to_dict() and first.status == "selected"
    assert exhausted.selected and exhausted.status == "selected_budget_unavailable"
    common = {"path_witnesses": {"executed_path": witness()},
              "dependency_generation": 0, "authority_verifier": lambda *_: True}
    assert E.route_disposition(route, a, b, audit_decision=first,
                               **common).status == "target_audit_required"
    assert E.route_disposition(route, a, b, audit_decision=exhausted,
                               **common).status == "audit_budget_unavailable"


def test_successful_audit_revokes_only_route_scope_and_generation():
    a, b, _ = three_claims()
    route = E.Route.from_dict(route_dict(a, b))
    revocation = E.RouteRevocation.from_dict({
        "schema": E.REVOCATION_SCHEMA, "event_id": "revoke-1",
        "route_id": route.route_id, "route_revision": route.revision,
        "destination_scope": dict(b.target_scope), "dependency_generation": 2,
        "audit_finding_id": "audit-finding", "target_scale_success": True})
    kwargs = {"path_witnesses": {"executed_path": witness()},
              "authority_verifier": lambda *_: True, "revocations": [revocation],
              "audit_decision": no_audit(b, route)}
    assert E.route_disposition(route, a, b, dependency_generation=2,
                               **kwargs).status == "revoked"
    assert E.route_disposition(route, a, b, dependency_generation=3,
                               **kwargs).status == "may_screen_out"
    other = E.Route.from_dict(route_dict(a, b, route_id="route-2"))
    other_kwargs = dict(kwargs)
    other_kwargs["audit_decision"] = no_audit(b, other)
    assert E.route_disposition(other, a, b, dependency_generation=2,
                               **other_kwargs).status == "may_screen_out"


def coexistence_dict(victim, neighbors, *, margin=True):
    return {"schema": E.COEXISTENCE_SCHEMA, "receipt_id": "coexist-1",
            "victim_scope": victim, "neighbor_mode": "multiset",
            "neighbors": neighbors, "pressure_envelope": None,
            "physical_claims": ["cpu:0-47"], "lifecycle_phases": PHASES,
            "dependency_identities": {"topology": H},
            "estimands": ["latency", "throughput"],
            "equivalence_margin": ({"registered_margin_ref": "margin:v1"}
                                   if margin else None),
            "uncertainty": ({"interval_ref": "interval:v1"} if margin else None),
            "authority_reference": "registered:coexist:v1" if margin else None}


def test_neighbor_multiset_multiplicity_triples_and_victim_direction():
    victim = scope(target="A")
    b = scope(target="B")
    c = scope(target="C")
    receipt = E.CoexistenceReceipt.from_dict(coexistence_dict(victim, [b, b]))
    good = E.coexistence_disposition(
        receipt, victim, neighbors=[b, b], lifecycle_phases=PHASES,
        equivalence_verifier=lambda _: True)
    assert good.status == "coexistence_supported"
    assert E.coexistence_disposition(
        receipt, victim, neighbors=[b], lifecycle_phases=PHASES,
        equivalence_verifier=lambda _: True).status == "serialized_owned"
    assert E.coexistence_disposition(
        receipt, victim, neighbors=[b, b, c], lifecycle_phases=PHASES,
        equivalence_verifier=lambda _: True).status == "serialized_owned"
    assert E.coexistence_disposition(
        receipt, b, neighbors=[b, b], lifecycle_phases=PHASES,
        equivalence_verifier=lambda _: True).status == "serialized_owned"


def test_bursts_and_equivalence_absence_leave_serialized_admission():
    victim, neighbor = scope(target="A"), scope(target="B")
    without_margin = E.CoexistenceReceipt.from_dict(
        coexistence_dict(victim, [neighbor], margin=False))
    assert E.coexistence_disposition(
        without_margin, victim, neighbors=[neighbor], lifecycle_phases=PHASES,
        equivalence_verifier=lambda _: True).status == "serialized_owned"
    complete = E.CoexistenceReceipt.from_dict(coexistence_dict(victim, [neighbor]))
    assert E.coexistence_disposition(
        complete, victim, neighbors=[neighbor],
        lifecycle_phases=[phase for phase in PHASES if phase != "bursts"],
        equivalence_verifier=lambda _: True).status == "serialized_owned"
    assert E.coexistence_disposition(
        complete, victim, neighbors=[neighbor], lifecycle_phases=PHASES,
        equivalence_verifier=None).status == "serialized_owned"


def test_pressure_envelope_requires_separate_registered_verifier():
    victim, neighbor = scope(target="A"), scope(target="B")
    obj = coexistence_dict(victim, [])
    obj["neighbor_mode"] = "pressure_envelope"
    obj["pressure_envelope"] = {"registered_envelope_ref": "pressure:v1"}
    receipt = E.CoexistenceReceipt.from_dict(obj)
    assert E.coexistence_disposition(
        receipt, victim, neighbors=[neighbor], lifecycle_phases=PHASES,
        equivalence_verifier=lambda _: True).status == "serialized_owned"
    assert E.coexistence_disposition(
        receipt, victim, neighbors=[neighbor], lifecycle_phases=PHASES,
        pressure_verifier=lambda *_: True,
        equivalence_verifier=lambda _: True).status == "coexistence_supported"
