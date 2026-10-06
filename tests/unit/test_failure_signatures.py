"""Synthetic native journal diagnostics; no model, endpoint or runtime fixture."""

from scripts.autopilot.experiment_journal import (
    ExperimentJournal, JournalEntry, build_error_signature, build_error_scope,
    negative_evidence_for_prompt, prior_repeated_failures,
)
from scripts.autopilot.stm_generated_view import render_generated_stm


def signature(**changes):
    args = dict(passed=False, retry_not_revert=False, categories=["regression"],
                species="prompt_forge", action_type="prompt_mutation", tier=1)
    args.update(changes)
    return build_error_signature(**args)


def scope(**changes):
    args = dict(core_id="core-v1", regime_digest="a" * 64,
                baseline_pin={"eval_quality_era": "q-v1", "autopilot_speed_era": "s-v1"},
                comparability={"status": "COMPARABLE"})
    args.update(changes)
    return build_error_scope(**args)


def row(trial_id=1, **changes):
    args = dict(trial_id=trial_id, timestamp="2026-10-06T00:00:00Z", species="prompt_forge",
                action_type="prompt_mutation", tier=1, quality=1.0, speed=1.0,
                cost=0.1, reliability=1.0, pareto_status="dominated", failure_analysis="bad",
                error_signature=signature(), error_scope=scope())
    args.update(changes)
    return JournalEntry(**args)


def test_machine_identity_only_actual_failure_codes():
    assert signature(categories=["regression", "quality_floor", "regression"]) == signature(
        categories=["quality_floor", "regression"])
    for changes in ({"passed": True}, {"retry_not_revert": True},
                    {"categories": ["seq_refuted", "quality_not_measured"]},
                    {"categories": []}):
        assert signature(**changes) is None
    for changes in ({"species": "seeder"}, {"action_type": "code_mutation"}, {"tier": 2}):
        assert signature(**changes) != signature()


def test_scope_is_explicit_and_missing_means_unknown():
    for changes in ({"core_id": ""}, {"regime_digest": ""}, {"baseline_pin": {}},
                    {"comparability": {"status": "UNVERIFIED"}}, {"core_id": None}):
        assert scope(**changes) is None
    assert prior_repeated_failures(row(error_scope=None), []) is None
    assert prior_repeated_failures(row(error_signature=None), []) is None


def test_total_prior_not_streak_and_churn_does_not_identify_failure():
    first = row(1, timestamp="old", failure_analysis="changed text /random/path")
    success = row(2, pareto_status="frontier", failure_analysis="", error_signature=None)
    current = row(3, timestamp="new", failure_analysis="different sentence")
    assert prior_repeated_failures(current, [first, success, current]) == 1
    assert prior_repeated_failures(row(4), [first, current]) == 2
    assert prior_repeated_failures(row(4, error_scope=scope(core_id="core-v2")), [first]) == 0


def test_trust_axes_never_count():
    rows = [row(1, bug_corrupted_by="fix"), row(2, outcome_status="skipped"),
            row(3, eval_details={"learning_exclusion": {"reason": "infra"}}),
            row(4, pareto_status="frontier", error_signature=None),
            row(5, keep_revert_decision="excluded"),
            row(6, error_scope=None), row(7, error_signature=None)]
    assert prior_repeated_failures(row(8), rows) == 0
    for excluded in rows[:3]:
        assert prior_repeated_failures(excluded, []) is None


def test_native_append_reload_and_no_legacy_backfill(tmp_path):
    journal = ExperimentJournal(journal_dir=tmp_path)
    first, second = row(1), row(2)
    journal.record(first)
    journal.record(second)
    assert first.repeated_failures_prior == 0
    assert second.repeated_failures_prior == 1
    loaded = ExperimentJournal(journal_dir=tmp_path)
    assert loaded._entries[-1].error_signature == signature()
    legacy = row(3, error_signature=None, error_scope=None)
    loaded.record(legacy)
    assert ExperimentJournal(journal_dir=tmp_path)._entries[-1].error_signature is None
    assert legacy.repeated_failures_prior is None


def test_folded_history_recount_ignores_superseded_corruption(tmp_path):
    journal = ExperimentJournal(journal_dir=tmp_path)
    first, second = row(1), row(2)
    journal.record(first)
    journal.record(second)
    journal.append_supersession_event(
        target_trial_ids=[1], fields={"bug_corrupted_by": "fix"},
        reason="synthetic corruption repair", policy_version="supersession-v1", actor="unit-test")
    reloaded = ExperimentJournal(journal_dir=tmp_path)
    folded = reloaded.entries_with_supersessions()
    assert folded[0].bug_corrupted_by == "fix"
    assert reloaded._entries[0].bug_corrupted_by == ""
    assert reloaded._entries[1].repeated_failures_prior == 1
    assert second.repeated_failures_prior == 1
    assert "prior_total=0" in negative_evidence_for_prompt(second, entries=folded)


def test_compact_render_preserves_typed_and_unknown_negative_evidence(tmp_path):
    first, second = row(1), row(2, failure_analysis="RAW ATTEMPT " * 200)
    typed = negative_evidence_for_prompt(second, entries=[first, second], limit=60)
    assert "prior_total=1" in typed and "RAW ATTEMPT" not in typed and "\n" not in typed
    unknown = row(3, error_signature=None, error_scope=None, failure_analysis="legacy\n" * 200)
    legacy = negative_evidence_for_prompt(unknown, limit=160)
    assert legacy.startswith("[unclassified]") and len(legacy) <= 160 and "\n" not in legacy
    journal = ExperimentJournal(journal_dir=tmp_path)
    for item in (first, second, unknown):
        journal.record(item)
    assert "Negative: failure/v1" in journal.insights_structured_text()
    assert journal.insights_structured_text().count("Negative: failure/v1") == 1
    assert "Negative: [unclassified]" in journal.insights_structured_text()
    rendered = render_generated_stm(journal.entries_with_supersessions())
    assert "prior_total=1" in rendered and "RAW ATTEMPT" not in rendered
    assert "[unclassified]" in rendered


def test_invalid_schema_primitives_and_explicit_exclusion_stay_unknown():
    for signature_value in (
        {**signature(), "schema_version": True}, {**signature(), "extra": "not-v1"},
        {**signature(), "tier": True}, {**signature(), "species": "bad\nlabel"},
    ):
        assert prior_repeated_failures(row(error_signature=signature_value), []) is None
    for scope_value in ({**scope(), "schema_version": True}, {**scope(), "extra": "not-v1"},
                        {**scope(), "core_id": 3}):
        assert prior_repeated_failures(row(error_scope=scope_value), []) is None
    assert prior_repeated_failures(row(keep_revert_decision="excluded"), []) is None


def test_record_unknown_fast_path_does_not_read_fold(tmp_path, monkeypatch):
    journal = ExperimentJournal(journal_dir=tmp_path)
    def forbidden_fold():
        raise AssertionError("unknown/ineligible diagnostic must not fold ledger")
    monkeypatch.setattr(journal, "entries_with_supersessions", forbidden_fold)
    for item in (row(1, error_signature=None), row(2, error_scope=None),
                 row(3, keep_revert_decision="excluded")):
        journal.record(item)
        assert item.repeated_failures_prior is None


def test_empty_prose_machine_failure_remains_visible_and_scope_disambiguates(tmp_path):
    journal = ExperimentJournal(journal_dir=tmp_path)
    first, second = row(1, failure_analysis=""), row(2, failure_analysis="")
    third = row(3, failure_analysis="", error_scope=scope(core_id="core-v2"))
    for item in (first, second, third):
        journal.record(item)
    assert second.repeated_failures_prior == 1
    assert third.repeated_failures_prior == 0
    left = negative_evidence_for_prompt(second, entries=journal.entries_with_supersessions())
    right = negative_evidence_for_prompt(third, entries=journal.entries_with_supersessions())
    assert left != right and "prompt_forge/prompt_mutation tier=1" in left
    assert "prior_total=1" in journal.insights_text()
    assert journal.insights_structured_text().count("Negative: failure/v1") == 2
    assert "prior_total=1" in render_generated_stm(journal.entries_with_supersessions())


def test_insight_render_folds_history_once(tmp_path, monkeypatch):
    journal = ExperimentJournal(journal_dir=tmp_path)
    for item in (row(1), row(2), row(3)):
        journal.record(item)
    original = journal.entries_with_supersessions
    calls = []
    def tracked_fold():
        calls.append(True)
        return original()
    monkeypatch.setattr(journal, "entries_with_supersessions", tracked_fold)
    for renderer in (journal.summary_text, journal.insights_text, journal.insights_structured_text):
        calls.clear()
        renderer()
        assert len(calls) == 1
