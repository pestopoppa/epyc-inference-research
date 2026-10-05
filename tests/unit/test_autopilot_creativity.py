"""Tests for the constrained-creativity planner upgrade (2026-05-23):

- `_build_exploration_block()` switches between lean and rich fragments
  based on stagnation signals.
- `tail_action_candidates()` semantics (now consumed as seeds, not candidates).
- `unfalsified_hypotheses()` filter logic.
- New `JournalEntry.falsifier` / `rubric_scores` fields round-trip through
  JSONL.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

experiment_journal = importlib.import_module("experiment_journal")
ExperimentJournal = experiment_journal.ExperimentJournal
JournalEntry = experiment_journal.JournalEntry

# Match the dotted-path form used by test_autopilot_numeric_apply.py so that
# both tests share the same sys.modules entry — otherwise monkeypatches on
# `scripts.autopilot.autopilot` and reads through bare `autopilot` would
# diverge when the suites run in the same session.
autopilot = importlib.import_module("scripts.autopilot.autopilot")
pareto_archive = importlib.import_module("pareto_archive")
ParetoArchive = pareto_archive.ParetoArchive
ParetoEntry = pareto_archive.ParetoEntry


KNOWN_ACTIONS = [
    "seed_batch", "numeric_trial", "prompt_mutation",
    "gepa_optimize", "code_mutation", "structural_experiment",
    "structural_prune", "slot_compact", "train_routing_models",
    "distill_skillbank", "reset_memories", "deep_eval",
    "rollback", "distill_knowledge",
]


class _DeterministicEmbedder:
    """Fixed local vector fixture; it makes no semantic-quality claim."""

    def embed_text(self, _text: str) -> np.ndarray:
        return np.full(1024, 1.0 / np.sqrt(1024), dtype=np.float32)


# ── helpers ─────────────────────────────────────────────────────


def _entry(trial_id: int, action_type: str, **kw) -> JournalEntry:
    return JournalEntry(
        trial_id=trial_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        species=kw.pop("species", "auto"),
        action_type=action_type,
        tier=1,
        quality=kw.pop("quality", 0.5),
        speed=10.0,
        cost=0.1,
        reliability=1.0,
        pareto_status=kw.pop("pareto_status", "dominated"),
        **kw,
    )


def _populate(journal: ExperimentJournal, entries: list[JournalEntry]) -> None:
    """Append entries to the journal's in-memory list and persist via record()."""
    for e in entries:
        journal.record(e)


def _fresh_journal(tmp_path: Path) -> ExperimentJournal:
    return ExperimentJournal(journal_dir=tmp_path)


# ── JournalEntry round-trip ─────────────────────────────────────


def test_journal_entry_persists_new_fields(tmp_path: Path) -> None:
    j = _fresh_journal(tmp_path)
    j.record(_entry(
        0, "seed_batch",
        hypothesis="seed math broadly",
        falsifier="no quality gain after 20 questions",
        rubric_scores={"info_gain": 4, "coherence": 5, "usefulness": 3},
        stagnation_signal="hv_slope_10=+0.00000 < eps=0.00100",
    ))
    # Reload from disk and confirm fields survived.
    reloaded = ExperimentJournal(journal_dir=tmp_path)
    e = reloaded.all_entries()[-1]
    assert e.falsifier == "no quality gain after 20 questions"
    assert e.rubric_scores == {"info_gain": 4, "coherence": 5, "usefulness": 3}
    assert e.stagnation_signal == "hv_slope_10=+0.00000 < eps=0.00100"


def test_action_diversity_by_gate_buckets_correctly(tmp_path: Path) -> None:
    j = _fresh_journal(tmp_path)
    # 4 rich (stagnation fired) trials with 3 distinct action_types.
    for i, t in enumerate(["seed_batch", "numeric_trial", "prompt_mutation", "numeric_trial"]):
        j.record(_entry(i, t, stagnation_signal="hv_slope_10 < eps"))
    # 6 lean trials, all seed_batch (the exploit phase).
    for i, t in enumerate(["seed_batch"] * 6):
        j.record(_entry(i + 10, t, stagnation_signal=""))

    stats = j.action_diversity_by_gate(window=20)
    assert stats["window"] == 20
    assert stats["rich"]["count"] == 4
    assert stats["rich"]["distinct_action_types"] == 3
    assert stats["rich"]["entropy_bits"] > 1.0   # 3 types across 4 trials
    assert stats["lean"]["count"] == 6
    assert stats["lean"]["distinct_action_types"] == 1
    assert stats["lean"]["entropy_bits"] == 0.0


# ── hv_slope noise-floor auto-calibration ───────────────────────


def _empty_archive(tmp_path: Path) -> "ParetoArchive":
    """Build a ParetoArchive that doesn't load production state."""
    return ParetoArchive(state_path=tmp_path / "nonexistent_archive.json")


def test_hv_slope_noise_floor_falls_back_to_default_with_short_history(tmp_path: Path) -> None:
    a = _empty_archive(tmp_path)
    # No history at all → must return the supplied default unchanged.
    floor = a.hv_slope_noise_floor(floor_default=1e-3)
    assert floor == 1e-3


def test_hv_slope_noise_floor_clips_below_default_with_calm_history(tmp_path: Path) -> None:
    a = _empty_archive(tmp_path)
    # Push a long, very-low-noise hypervolume trajectory (linear-ish).
    # Slopes should be near-zero variance → calibrated floor below default.
    for tid in range(50):
        a._hv_hist().append((tid, 10.0 + 0.001 * tid))
    floor = a.hv_slope_noise_floor(floor_default=1e-3, floor_min=1e-6)
    assert floor <= 1e-3
    assert floor >= 1e-6


def test_hv_slope_noise_floor_never_exceeds_default_even_with_noisy_history(tmp_path: Path) -> None:
    a = _empty_archive(tmp_path)
    import random
    rng = random.Random(0)
    # Inject a high-noise random walk: slope variance will be large, but the
    # function must clip up to (i.e. never exceed) floor_default.
    h = 10.0
    for tid in range(80):
        h += rng.uniform(-2.0, 2.0)
        a._hv_hist().append((tid, h))
    floor = a.hv_slope_noise_floor(floor_default=1e-3, floor_min=1e-6)
    assert floor <= 1e-3
    assert floor >= 1e-6


def test_journal_entry_defaults_when_legacy_jsonl_missing_fields(tmp_path: Path) -> None:
    # Simulate a legacy file that predates the new fields.
    path = tmp_path / "autopilot_journal.jsonl"
    legacy = {
        "trial_id": 7, "timestamp": "2025-01-01T00:00:00+00:00",
        "species": "auto", "action_type": "seed_batch",
        "tier": 1, "quality": 0.5, "speed": 10.0, "cost": 0.1,
        "reliability": 1.0, "pareto_status": "dominated",
    }
    path.write_text(json.dumps(legacy) + "\n")
    j = ExperimentJournal(journal_dir=tmp_path)
    e = j.all_entries()[-1]
    assert e.falsifier == ""
    assert e.rubric_scores == {}


# ── unfalsified_hypotheses filter ───────────────────────────────


def test_unfalsified_hypotheses_returns_only_entries_with_both_fields(tmp_path: Path) -> None:
    j = _fresh_journal(tmp_path)
    _populate(j, [
        _entry(0, "seed_batch", hypothesis="h0", falsifier=""),                    # no falsifier
        _entry(1, "numeric_trial", hypothesis="", falsifier="some falsifier"),     # no hypothesis
        _entry(2, "prompt_mutation", hypothesis="h2", falsifier="f2"),             # keep
        _entry(3, "code_mutation", hypothesis="h3", falsifier="f3"),               # keep
    ])
    out = j.unfalsified_hypotheses(n=5)
    assert [t[0] for t in out] == [2, 3]
    assert out[0] == (2, "h2", "f2")


def test_unfalsified_hypotheses_excludes_bug_corrupted(tmp_path: Path) -> None:
    j = _fresh_journal(tmp_path)
    _populate(j, [
        _entry(0, "seed_batch", hypothesis="h0", falsifier="f0",
               bug_corrupted_by="deadbeef"),
        _entry(1, "numeric_trial", hypothesis="h1", falsifier="f1"),
    ])
    out = j.unfalsified_hypotheses(n=5)
    assert [t[0] for t in out] == [1]


# ── _build_exploration_block ────────────────────────────────────


class _FakeArchive:
    """Minimal stand-in exposing .geometry() with the keys autopilot reads."""

    def __init__(self, hv_slope_10: float | None) -> None:
        self._slope = hv_slope_10

    def geometry(self) -> dict[str, float]:
        return {"hv_slope_10": self._slope} if self._slope is not None else {}


def _well_populated_journal(tmp_path: Path, varied_types: list[str]) -> ExperimentJournal:
    """Build a journal with ≥5 trustworthy trials and diverse action_types so the
    low-signal flag and the streak signal both stay quiet."""
    j = _fresh_journal(tmp_path)
    for i, t in enumerate(varied_types):
        j.record(_entry(i, t, hypothesis=f"h{i}"))
    return j


def test_lean_fragment_when_no_stagnation_signal(tmp_path: Path) -> None:
    j = _well_populated_journal(
        tmp_path,
        ["seed_batch", "numeric_trial", "prompt_mutation",
         "code_mutation", "structural_experiment", "deep_eval"],
    )
    archive = _FakeArchive(hv_slope_10=0.1)  # healthy growth
    block, signal = autopilot._build_exploration_block(
        journal=j, archive=archive, known_actions=KNOWN_ACTIONS,
    )
    assert "STAGNATING" not in block
    assert "up to 3 alternatives" in block
    assert signal == "none (lean prompt)"


def test_rich_fragment_when_hv_slope_below_eps(tmp_path: Path) -> None:
    j = _well_populated_journal(
        tmp_path,
        ["seed_batch", "numeric_trial", "prompt_mutation",
         "code_mutation", "structural_experiment", "deep_eval"],
    )
    archive = _FakeArchive(hv_slope_10=0.0)  # frontier flat
    block, signal = autopilot._build_exploration_block(
        journal=j, archive=archive, known_actions=KNOWN_ACTIONS,
    )
    assert "STAGNATING" in block
    assert "Generate 3 candidate actions" in block
    assert "candidate actions" in block
    assert "BT tiebreak" not in block
    assert "hv_slope_10" in signal


def test_rich_fragment_when_action_type_streak(tmp_path: Path) -> None:
    j = _well_populated_journal(
        tmp_path,
        # 5 varied seed entries to clear low_signal, then a 3-trial streak
        ["seed_batch", "numeric_trial", "prompt_mutation",
         "code_mutation", "structural_experiment",
         "deep_eval", "deep_eval", "deep_eval"],
    )
    archive = _FakeArchive(hv_slope_10=0.5)  # healthy
    block, signal = autopilot._build_exploration_block(
        journal=j, archive=archive, known_actions=KNOWN_ACTIONS,
    )
    assert "STAGNATING" in block
    assert "deep_eval" in signal


def test_rich_fragment_when_trustworthy_low_signal(tmp_path: Path) -> None:
    j = _fresh_journal(tmp_path)
    # Only 2 trustworthy entries → triggers low_signal flag.
    j.record(_entry(0, "seed_batch"))
    j.record(_entry(1, "numeric_trial"))
    archive = _FakeArchive(hv_slope_10=0.5)
    block, signal = autopilot._build_exploration_block(
        journal=j, archive=archive, known_actions=KNOWN_ACTIONS,
    )
    assert "STAGNATING" in block
    assert "trustworthy=" in signal


def test_rich_fragment_lists_tail_seeds_and_unfalsified(tmp_path: Path) -> None:
    j = _fresh_journal(tmp_path)
    # Build a journal where seed_batch dominates (so most others are tail) AND
    # at least one entry has both hypothesis+falsifier.
    for i in range(8):
        j.record(_entry(i, "seed_batch"))
    j.record(_entry(
        8, "prompt_mutation",
        hypothesis="trim verbose section", falsifier="quality drops >2pp",
    ))
    # Force stagnation via flat slope.
    archive = _FakeArchive(hv_slope_10=0.0)
    block, signal = autopilot._build_exploration_block(
        journal=j, archive=archive, known_actions=KNOWN_ACTIONS,
    )
    # Tail seeds should include some under-used action_type names.
    assert any(a in block for a in KNOWN_ACTIONS if a != "seed_batch")
    # Unfalsified hypothesis surfaces with its falsifier.
    assert "trim verbose section" in block
    assert "quality drops >2pp" in block


def test_action_availability_filters_non_viable_tail_actions(tmp_path: Path) -> None:
    j = _fresh_journal(tmp_path)
    j.record(_entry(
        185,
        "prompt_mutation",
        failure_analysis="prompt edit regressed quality badly",
        hypothesis="edit frontdoor",
    ))
    availability, viable, selectable = autopilot._build_action_availability(
        journal=j,
        known_actions=KNOWN_ACTIONS,
        memory_count=100,
        converged=False,
        slot_memory_text="  healthy queried ports with empty KV cache: frontdoor:8070",
        blacklist=[{"pattern": {"type": "distill_skillbank"}, "reason": "blacklisted"}],
    )
    assert "`slot_compact`" in availability
    assert "not evidence that the eval instrument or host is contaminated" in availability
    assert "`train_routing_models`" in availability
    assert "`reset_memories`" in availability
    assert "`distill_skillbank`" in availability
    assert "`prompt_mutation`" in availability
    assert "Capability registry levers (generated):" in availability
    assert "`edit_transaction_auto_routing`: operator-only" in availability
    assert "slot_compact" not in viable
    assert "train_routing_models" not in viable
    assert "reset_memories" not in viable
    assert "rollback" not in viable
    assert "distill_knowledge" not in viable
    assert "distill_skillbank" not in viable
    assert "seed_batch" in selectable


def test_action_availability_blocks_seed_batch_when_fallbacks_exhausted(
    tmp_path: Path,
) -> None:
    j = _fresh_journal(tmp_path)
    availability, viable, selectable = autopilot._build_action_availability(
        journal=j,
        known_actions=KNOWN_ACTIONS,
        memory_count=10_000,
        converged=True,
        slot_memory_text="  healthy queried ports with empty KV cache: frontdoor:8070",
        blacklist=[
            {
                "pattern": {"type": "seed_batch", "n_questions": n_questions},
                "reason": f"blocked {n_questions}",
            }
            for n_questions in autopilot.FALLBACK_SEED_CANDIDATES
        ],
    )

    assert "`seed_batch`" in availability
    assert "all configured measured seed fallback candidates are blacklisted" in availability
    assert "seed_batch" not in viable
    assert "seed_batch" not in selectable


def test_action_availability_surfaces_w8_candidate_generation_priority(
    tmp_path: Path,
) -> None:
    j = _fresh_journal(tmp_path)

    availability, viable, selectable = autopilot._build_action_availability(
        journal=j,
        known_actions=KNOWN_ACTIONS,
        memory_count=10_000,
        converged=True,
        slot_memory_text="  healthy queried ports with empty KV cache: frontdoor:8070",
        blacklist=[],
        w8_replay_pressure_text=(
            "W8 replay pressure: 0/2 accumulating candidate(s) are replayable "
            "(blocked=numeric_trial_missing_params:1,unreplayable_action=seed_batch:1)."
        ),
    )

    assert "Priority pressure:" in availability
    assert "W8 candidate generation is the active strict blocker" in availability
    assert "actions are ONLY" in availability
    assert "Do not emit seed_batch, deep_eval, or structural_prune" in availability
    assert "Optuna-suggested numeric_trial that journals applied params" in availability
    assert "deep_eval" not in viable
    assert "seed_batch" not in viable
    assert "structural_prune" not in viable
    assert "deep_eval" not in selectable
    assert "seed_batch" not in selectable
    assert "structural_prune" not in selectable

    rendered = autopilot._format_available_action_schemas(selectable)
    assert '- Numeric: {"type": "numeric_trial"' in rendered
    assert "Valid params by surface:" in rendered
    assert "memrl_retrieval.semantic_k" in rendered
    assert "repl.turn_token_cap" in rendered
    assert "tool_activation_threshold" in rendered
    assert '- Structural: {"type": "structural_experiment"' in rendered
    assert "- Seed:" not in rendered
    assert "- Deep eval:" not in rendered
    assert "- Prune:" not in rendered


def test_controller_prompt_binds_availability_before_global_action_schema() -> None:
    template = autopilot.CONTROLLER_PROMPT_TEMPLATE

    binding_idx = template.index("Action Availability")
    filtered_idx = template.index("already filtered")
    schema_idx = template.index("{available_action_schemas}")

    assert binding_idx < filtered_idx < schema_idx


def test_action_schema_surfaces_new_file_memory_schema_root() -> None:
    rendered = autopilot._format_available_action_schemas(["code_mutation"])

    assert '"mutation": "targeted_fix"' in rendered
    assert 'use mutation "new_file" only under' in rendered
    assert "orchestration/repl_memory/schema_evolution" in rendered
    assert "memory schema-evolution files must be default-inert" in rendered


def test_bsv3_conflict_policy_default_off_allows_state_promotion(monkeypatch) -> None:
    monkeypatch.delenv("AUTOPILOT_BSV3_CONFLICT_POLICY", raising=False)

    decision = autopilot._bsv3_conflict_policy_decision(
        {"severity": "blocking", "conflict_count": 2}
    )

    assert decision["enabled"] is False
    assert decision["policy"] == "off"
    assert decision["ledger_update_allowed"] is True
    assert decision["scope"] == "bsv_ledger_incumbent_state_only"


def test_bsv3_conflict_policy_blocks_only_configured_severities() -> None:
    block_watch = autopilot._bsv3_conflict_policy_decision(
        {"severity": "watch", "conflict_count": 1},
        policy="block",
    )
    block_blocking = autopilot._bsv3_conflict_policy_decision(
        {"severity": "blocking", "conflict_count": 1},
        policy="block",
    )
    review_watch = autopilot._bsv3_conflict_policy_decision(
        {"severity": "watch", "conflict_count": 1},
        policy="review",
    )

    assert block_watch["ledger_update_allowed"] is True
    assert block_blocking["ledger_update_allowed"] is False
    assert block_blocking["incumbent_update_allowed"] is False
    assert review_watch["ledger_update_allowed"] is False


def test_w8_candidate_generation_pressure_ignores_replayable_candidates() -> None:
    assert autopilot._w8_candidate_generation_pressure(
        "W8 replay pressure: 1/1 accumulating candidate(s) are replayable"
    ) is False


def test_w8_replay_pressure_remains_active_for_replayable_candidates() -> None:
    assert autopilot._w8_replay_pressure_active(
        "W8 replay pressure: 3/10 accumulating candidate(s) are replayable; "
        "prefer collecting replay/confirmation evidence before opening unrelated "
        "W8 candidate generation."
    ) is True


def test_action_availability_blocks_deferrals_during_w8_replay_pressure(
    tmp_path: Path,
) -> None:
    j = _fresh_journal(tmp_path)

    availability, viable, selectable = autopilot._build_action_availability(
        journal=j,
        known_actions=KNOWN_ACTIONS,
        memory_count=10_000,
        converged=True,
        slot_memory_text="  healthy queried ports with empty KV cache: frontdoor:8070",
        blacklist=[],
        w8_replay_pressure_text=(
            "W8 replay pressure: 3/10 accumulating candidate(s) are replayable; "
            "prefer collecting replay/confirmation evidence before opening unrelated "
            "W8 candidate generation."
        ),
    )

    assert "W8 replay/confirmation evidence is active" in availability
    assert "avoid seed_batch, deep_eval, and structural_prune" in availability
    assert "deep_eval" not in viable
    assert "seed_batch" not in viable
    assert "structural_prune" not in viable
    assert "deep_eval" not in selectable
    assert "seed_batch" not in selectable
    assert "structural_prune" not in selectable


def test_dispatch_allowlist_applies_only_to_planner_selected_actions() -> None:
    assert autopilot._dispatch_allowed_action_types(
        ["seed_batch", "numeric_trial"],
        seq_due_bypassed_planner=False,
        seq_fresh_eval_context=None,
        seq_baseline_draw_reference=None,
        seq_candidate_replay_context=None,
    ) == ["seed_batch", "numeric_trial"]

    assert (
        autopilot._dispatch_allowed_action_types(
            ["numeric_trial"],
            seq_due_bypassed_planner=False,
            seq_fresh_eval_context=None,
            seq_baseline_draw_reference=None,
            seq_candidate_replay_context=None,
            seq_gate_preflight={
                "status": "deferred",
                "replacement_action": {"type": "seed_batch", "n_questions": 50},
            },
        )
        is None
    )

    assert autopilot._dispatch_allowed_action_types(
        ["numeric_trial"],
        seq_due_bypassed_planner=False,
        seq_fresh_eval_context=None,
        seq_baseline_draw_reference=None,
        seq_candidate_replay_context=None,
        seq_gate_preflight={
            "status": "blocked_no_fallback",
            "replacement_action": None,
        },
    ) == ["numeric_trial"]

    assert (
        autopilot._dispatch_allowed_action_types(
            ["seed_batch", "numeric_trial"],
            seq_due_bypassed_planner=True,
            seq_fresh_eval_context=None,
            seq_baseline_draw_reference=None,
            seq_candidate_replay_context=None,
        )
        is None
    )

    assert (
        autopilot._dispatch_allowed_action_types(
            ["seed_batch", "numeric_trial"],
            seq_due_bypassed_planner=False,
            seq_fresh_eval_context={"candidate": "candidate-a"},
            seq_baseline_draw_reference=None,
            seq_candidate_replay_context=None,
        )
        is None
    )


def test_w8_candidate_generation_replaces_deferral_with_numeric(monkeypatch) -> None:
    monkeypatch.setattr(autopilot, "_configured_numeric_surfaces", lambda: ("monitor",))

    action, rationale = autopilot._replace_w8_candidate_generation_deferral(
        {"type": "seed_batch", "n_questions": 40},
        [],
        {"falsifier": "original"},
        trial_counter=123,
        w8_replay_pressure_text=(
            "W8 replay pressure: 0/1 accumulating candidate(s) are replayable "
            "(blocked=unreplayable_action=seed_batch:1)."
        ),
    )

    assert action == {"type": "numeric_trial", "surface": "monitor", "params": {}}
    assert rationale["w8_candidate_generation_replaced"] is True
    assert rationale["w8_candidate_generation_reason"] == "unreplayable_action=seed_batch"
    assert rationale["falsifier"] == "original"


def test_w8_candidate_generation_keeps_numeric_optuna_request() -> None:
    action, rationale = autopilot._replace_w8_candidate_generation_deferral(
        {"type": "numeric_trial", "surface": "monitor", "params": {}},
        [],
        {"falsifier": "keep"},
        trial_counter=123,
        w8_replay_pressure_text=(
            "W8 replay pressure: 0/1 accumulating candidate(s) are replayable "
            "(blocked=numeric_trial_missing_params:1)."
        ),
    )

    assert action == {"type": "numeric_trial", "surface": "monitor", "params": {}}
    assert rationale == {"falsifier": "keep"}


def test_action_availability_surfaces_suppressed_numeric_surfaces(
    tmp_path: Path,
) -> None:
    j = _fresh_journal(tmp_path)
    try:
        autopilot._PLANNER_SUPPRESSED_NUMERIC_SURFACES.clear()
        autopilot._PLANNER_SUPPRESSED_NUMERIC_SURFACES.add("kv_compaction")
        availability, viable, selectable = autopilot._build_action_availability(
            journal=j,
            known_actions=KNOWN_ACTIONS,
            memory_count=10_000,
            converged=True,
            slot_memory_text="  healthy queried ports with empty KV cache: frontdoor:8070",
            blacklist=[],
            suppressed_numeric_surfaces={"kv_compaction"},
        )
    finally:
        autopilot._PLANNER_SUPPRESSED_NUMERIC_SURFACES.clear()

    assert "convention-suppressed numeric surfaces are unavailable" in availability
    assert "kv_compaction" in availability
    assert "numeric_trial" in viable
    assert "numeric_trial" in selectable


def test_configured_numeric_surfaces_hide_planner_suppressed_surface() -> None:
    try:
        autopilot._PLANNER_SUPPRESSED_NUMERIC_SURFACES.clear()
        autopilot._PLANNER_SUPPRESSED_NUMERIC_SURFACES.add("kv_compaction")
        surfaces = autopilot._configured_numeric_surfaces()
    finally:
        autopilot._PLANNER_SUPPRESSED_NUMERIC_SURFACES.clear()

    assert "kv_compaction" not in surfaces
    assert "think_harder" in surfaces


def test_feature_flags_block_surfaces_convention_denylist() -> None:
    class FakeLab:
        def current_flags(self):
            return {"graph_router": False, "specialist_routing": True}

        def flag_schema(self):
            return [
                {"name": "graph_router", "dependencies": ["specialist_routing"]},
                {"name": "specialist_routing", "dependencies": []},
            ]

    block = autopilot._build_feature_flags_block(
        FakeLab(),
        denylisted_flags={"graph_router"},
    )

    assert "Convention-denylisted flags: graph_router" in block
    assert "never propose structural_experiment for convention-denylisted flags" in block


def test_planner_convention_install_does_not_touch_w6_audit_env(monkeypatch) -> None:
    monkeypatch.setattr(autopilot, "_PLANNER_HINTS_ENABLED", True)
    w6_env = {
        "AUTOPILOT_W6_AUDIT_BLOCK": "1",
        "AUTOPILOT_W6_AUDIT_N": "30",
        "AUTOPILOT_W6_AUDIT_EVERY_N_TRIALS": "2",
        "AUTOPILOT_W6_AUDIT_SHADOW_ONLY": "1",
    }
    for key, value in w6_env.items():
        monkeypatch.setenv(key, value)

    class FakeStore:
        def retrieve_conventions(self, *, species, journal):
            assert journal == "journal"
            if species == "structural_lab":
                return [
                    type(
                        "Entry",
                        (),
                        {
                            "metadata": {
                                "bind_status": "live",
                                "bind_identifiers": ["graph_router"],
                            }
                        },
                    )()
                ]
            if species == "numeric_swarm":
                return [
                    type(
                        "Entry",
                        (),
                        {
                            "metadata": {
                                "bind_status": "live",
                                "bind_identifiers": ["kv_compaction"],
                            }
                        },
                    )()
                ]
            raise AssertionError(f"unexpected species: {species}")

    suppressed_calls: list[set[str]] = []
    monkeypatch.setattr(
        autopilot.controller_io,
        "set_suppressed_numeric_surfaces",
        lambda surfaces: suppressed_calls.append(set(surfaces)),
    )

    try:
        autopilot._PLANNER_DENYLISTED_FEATURE_FLAGS.clear()
        autopilot._PLANNER_SUPPRESSED_NUMERIC_SURFACES.clear()
        autopilot._install_planner_convention_bindings(FakeStore(), "journal")

        assert autopilot._PLANNER_DENYLISTED_FEATURE_FLAGS == {"graph_router"}
        assert autopilot._PLANNER_SUPPRESSED_NUMERIC_SURFACES == {"kv_compaction"}
        assert suppressed_calls == [{"kv_compaction"}]
        assert {key: os.environ[key] for key in w6_env} == w6_env
    finally:
        autopilot._PLANNER_DENYLISTED_FEATURE_FLAGS.clear()
        autopilot._PLANNER_SUPPRESSED_NUMERIC_SURFACES.clear()


def test_planner_convention_bindings_refresh_without_restart(monkeypatch) -> None:
    monkeypatch.setattr(autopilot, "_PLANNER_HINTS_ENABLED", True)

    class FakeStore:
        def __init__(self) -> None:
            self.structural_flags = {"graph_router"}
            self.numeric_surfaces = {"kv_compaction"}

        def retrieve_conventions(self, *, species, journal):
            assert journal == "journal"
            if species == "structural_lab":
                identifiers = self.structural_flags
            elif species == "numeric_swarm":
                identifiers = self.numeric_surfaces
            else:
                raise AssertionError(f"unexpected species: {species}")
            return [
                SimpleNamespace(
                    metadata={
                        "bind_status": "live",
                        "bind_identifiers": sorted(identifiers),
                    },
                )
            ]

    suppressed_calls: list[set[str]] = []
    monkeypatch.setattr(
        autopilot.controller_io,
        "set_suppressed_numeric_surfaces",
        lambda surfaces: suppressed_calls.append(set(surfaces)),
    )

    store = FakeStore()
    try:
        autopilot._PLANNER_DENYLISTED_FEATURE_FLAGS.clear()
        autopilot._PLANNER_SUPPRESSED_NUMERIC_SURFACES.clear()

        autopilot._refresh_planner_convention_bindings(
            store,
            "journal",
            reason="planner_turn:1",
        )
        assert autopilot._PLANNER_DENYLISTED_FEATURE_FLAGS == {"graph_router"}
        assert autopilot._PLANNER_SUPPRESSED_NUMERIC_SURFACES == {"kv_compaction"}

        store.structural_flags = {"xmas_routing_enforce"}
        store.numeric_surfaces = {"monitor"}
        autopilot._refresh_planner_convention_bindings(
            store,
            "journal",
            reason="planner_turn:2",
        )

        assert autopilot._PLANNER_DENYLISTED_FEATURE_FLAGS == {
            "xmas_routing_enforce"
        }
        assert autopilot._PLANNER_SUPPRESSED_NUMERIC_SURFACES == {"monitor"}
        assert suppressed_calls[-2:] == [{"kv_compaction"}, {"monitor"}]
    finally:
        autopilot._PLANNER_DENYLISTED_FEATURE_FLAGS.clear()
        autopilot._PLANNER_SUPPRESSED_NUMERIC_SURFACES.clear()


def test_operator_numeric_surface_suppression_applies_without_hints(monkeypatch) -> None:
    monkeypatch.setattr(autopilot, "_PLANNER_HINTS_ENABLED", False)
    monkeypatch.setenv("AUTOPILOT_SUPPRESSED_NUMERIC_SURFACES", "kv_compaction")
    suppressed_calls: list[set[str]] = []
    monkeypatch.setattr(
        autopilot.controller_io,
        "set_suppressed_numeric_surfaces",
        lambda surfaces: suppressed_calls.append(set(surfaces)),
    )

    try:
        autopilot._PLANNER_SUPPRESSED_NUMERIC_SURFACES.clear()
        autopilot._refresh_planner_convention_bindings(None, None, reason="startup")
        assert autopilot._PLANNER_SUPPRESSED_NUMERIC_SURFACES == {"kv_compaction"}
        assert suppressed_calls == [{"kv_compaction"}]
    finally:
        autopilot._PLANNER_SUPPRESSED_NUMERIC_SURFACES.clear()


def test_planner_strategy_hints_default_off_does_not_read_store(monkeypatch) -> None:
    monkeypatch.setattr(autopilot, "_PLANNER_HINTS_ENABLED", False)

    class FakeStore:
        def retrieve_conventions(self, **_kwargs):
            raise AssertionError("disabled planner hints must not read StrategyStore")

        def retrieve_for_journal(self, *_args, **_kwargs):
            raise AssertionError("disabled planner hints must not read StrategyStore")

    text = autopilot._build_planner_strategy_hints(FakeStore(), "journal")

    assert "disabled" in text


def test_planner_strategy_hints_reflect_new_rows_between_turns(monkeypatch) -> None:
    monkeypatch.setattr(autopilot, "_PLANNER_HINTS_ENABLED", True)
    rows: list[SimpleNamespace] = []

    class FakeStore:
        def retrieve_conventions(self, *, species, journal, limit):
            assert species
            assert journal == "journal"
            assert limit == 3
            return []

        def retrieve_for_journal(self, query, *, journal, k, species):
            assert query
            assert journal == "journal"
            assert k == 3
            if species == "seeder":
                return list(rows)
            return []

    first = autopilot._build_planner_strategy_hints(
        FakeStore(),
        "journal",
        max_rows=3,
    )
    rows.append(
        SimpleNamespace(
            id="late-tool-use-hint",
            species="seeder",
            entry_type="pattern",
            title="Explore native tool use",
            generalized_content=(
                "Prefer a bounded REPL/tool-use measurement before another "
                "plain seed batch."
            ),
            metadata={
                "bind_status": "future",
                "bind_identifiers": ["tools", "repl"],
            },
        )
    )
    second = autopilot._build_planner_strategy_hints(
        FakeStore(),
        "journal",
        max_rows=3,
    )

    assert "no StrategyStore rows matched" in first
    assert "Explore native tool use" in second
    assert "tools,repl" in second
    assert "scope=orchestrator_eval_tools_not_planner_tools" in second


def test_planner_strategy_hints_see_external_store_writes_without_restart(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(autopilot, "_PLANNER_HINTS_ENABLED", True)
    monkeypatch.delenv("EPISODIC_ALLOW_DEGRADED_EMBEDDINGS", raising=False)
    from orchestration.repl_memory import embedder as embedder_module
    from orchestration.repl_memory.strategy_store import StrategyStore

    task_embedder = Mock(
        side_effect=AssertionError("fixture must not construct TaskEmbedder")
    )
    monkeypatch.setattr(embedder_module, "TaskEmbedder", task_embedder)
    strategy_path = tmp_path / "strategies"
    live_store = StrategyStore(path=strategy_path, embedder=_DeterministicEmbedder())
    writer_store = StrategyStore(path=strategy_path, embedder=_DeterministicEmbedder())
    journal = object()
    try:
        first = autopilot._build_planner_strategy_hints(
            live_store,
            journal,
            max_rows=3,
        )
        writer_store.store(
            description="tool use sentinel lane native tools repl activation latency",
            insight=(
                "Planner should steer a bounded REPL/tool-use measurement "
                "without restarting AutoPilot."
            ),
            source_trial_id=1107,
            species="structural_lab",
            entry_type="pattern",
            metadata={
                "bind_status": "future",
                "bind_identifiers": ["tools", "repl", "react_mode"],
                "source_handoff": "tool-use-eval-contract",
            },
            title="Fresh external tool-use hint",
            generalized_content=(
                "Prefer a tool-use sentinel measurement before another plain "
                "seed batch when native tools are enabled but unused."
            ),
        )

        second = autopilot._build_planner_strategy_hints(
            live_store,
            journal,
            max_rows=3,
        )

        assert "no StrategyStore rows matched" in first
        assert "Fresh external tool-use hint" in second
        assert "tools,repl,react_mode" in second
        assert "scope=orchestrator_eval_tools_not_planner_tools" in second
        task_embedder.assert_not_called()
    finally:
        writer_store.close()
        live_store.close()


def test_planner_strategy_hints_surface_search_index_degradation(monkeypatch) -> None:
    monkeypatch.setattr(autopilot, "_PLANNER_HINTS_ENABLED", True)

    class FakeStore:
        def search_index_health(self):
            return {
                "healthy": False,
                "summary": (
                    "degraded: sqlite=10, faiss=5, id_map=5, "
                    "faiss_coverage=50.0%, missing_faiss=5"
                ),
                "repair_hint": "StrategyStore.rebuild_search_indexes()",
            }

        def retrieve_for_journal(self, *_args, **_kwargs):
            return []

        def retrieve_conventions(self, *_args, **_kwargs):
            return []

    text = autopilot._build_planner_strategy_hints(
        FakeStore(),
        object(),
        max_rows=3,
    )

    assert "StrategyStore search index degraded" in text
    assert "faiss_coverage=50.0%" in text
    assert "StrategyStore.rebuild_search_indexes()" in text


def test_planner_strategy_hints_refresh_store_rows_for_prompt(monkeypatch) -> None:
    monkeypatch.setattr(autopilot, "_PLANNER_HINTS_ENABLED", True)
    calls: list[tuple[str, str, object, int]] = []

    class FakeStore:
        def retrieve_conventions(self, *, species, journal, limit):
            calls.append(("conventions", species, journal, limit))
            if species == "structural_lab":
                return [
                    SimpleNamespace(
                        id="operator-convention",
                        species="structural_lab",
                        entry_type="convention",
                        title="Gate tool-use sentinel lane",
                        generalized_content=(
                            "Use tool_helpfulness, not raw call count, when "
                            "judging tool-use exploration."
                        ),
                        metadata={
                            "bind_status": "future",
                            "bind_identifiers": ["tool_use_sentinel_lane"],
                            "source_handoff": "tool-use-eval-contract",
                        },
                    )
                ]
            return []

        def retrieve_for_journal(self, query, *, journal, k, species):
            calls.append(("journal", species, journal, k))
            assert query
            if species == "structural_lab":
                return [
                    SimpleNamespace(
                        id="tool-activation",
                        species="structural_lab",
                        entry_type="pattern",
                        title="Test v6 tool activation",
                        generalized_content=(
                            "Measure whether CALL() reduces latency without "
                            "quality loss on retrieval, math, and code-check tasks."
                        ),
                        metadata={
                            "bind_status": "future",
                            "bind_identifiers": ["tools", "repl", "react_mode"],
                            "source_handoff": (
                                "operator-observation-2026-07-03-tool-use-zero-call-v6"
                            ),
                        },
                    )
                ]
            return []

    journal = object()
    text = autopilot._build_planner_strategy_hints(
        FakeStore(),
        journal,
        max_rows=6,
    )

    assert "Gate tool-use sentinel lane" in text
    assert "tool_helpfulness" in text
    assert "Test v6 tool activation" in text
    assert "tools,repl,react_mode" in text
    assert "scope=orchestrator_eval_tools_not_planner_tools" in text
    assert ("conventions", "structural_lab", journal, 3) in calls
    assert ("journal", "structural_lab", journal, 3) in calls


def test_planner_strategy_hints_include_explicit_operator_seeds(monkeypatch) -> None:
    monkeypatch.setattr(autopilot, "_PLANNER_HINTS_ENABLED", True)

    class FakeResult:
        def fetchall(self):
            return [
                {
                    "id": "opseed-green-routing-handoff",
                    "species": "routing",
                    "entry_type": "pattern",
                    "description": "routing handoff that fixed queries miss",
                    "insight": "Planner should still see explicit operator seeds.",
                    "metadata_json": json.dumps({
                        "seeded_by": "operator",
                        "seed_campaign": "operator-handoff-distillation",
                        "bind_status": "context",
                        "bind_identifiers": [],
                        "source_handoff": "planner-hint-distillation",
                        "tranche": "green",
                        "insight_format": {
                            "title": "Route handoff visible without vector hit",
                            "generalized_content": (
                                "Surface explicit operator handoff hints even "
                                "when the fixed planner queries do not match."
                            ),
                        },
                    }),
                    "created_at": "2026-07-04T00:00:00+00:00",
                }
            ]

    class FakeConn:
        def execute(self, sql):
            assert "entry_type IN ('pattern', 'convention')" in sql
            return FakeResult()

    class FakeStore:
        _conn = FakeConn()

        def retrieve_conventions(self, *, species, journal, limit):
            assert species
            assert journal == "journal"
            assert limit == 3
            return []

        def retrieve_for_journal(self, query, *, journal, k, species):
            assert query
            assert journal == "journal"
            assert k == 3
            assert species
            return []

    text = autopilot._build_planner_strategy_hints(
        FakeStore(),
        "journal",
        max_rows=4,
    )

    assert "Route handoff visible without vector hit" in text
    assert "planner-hint-distillation" in text
    assert "fixed planner queries do not match" in text


def test_controller_prompt_scopes_strategy_tool_hints_to_eval_tools() -> None:
    template = autopilot.CONTROLLER_PROMPT_TEMPLATE

    assert "StrategyStore Planner Hints" in template
    assert "orchestrator/model execution inside AutoPilot actions" in template
    assert "planner process is read-only" in template
    assert "Never use Bash, Edit" in template
    assert "MultiEdit" in template
    assert "Write" in template
    assert "apply_patch" in template
    assert "let the orchestrator dispatch it" in template


def test_controller_prompt_includes_higher_tier_pressure_section() -> None:
    template = autopilot.CONTROLLER_PROMPT_TEMPLATE

    assert "Higher-Tier Objective Pressure" in template
    assert "{higher_tier_pressure}" in template
    rendered = autopilot._format_available_action_schemas(["deep_eval"])
    assert '{"type": "deep_eval", "tier": 3}' in rendered
    assert "expert/hard workflow coverage or frontier evidence is thin" in rendered


def test_controller_prompt_deep_eval_tiers_match_validator_contract() -> None:
    template = autopilot.CONTROLLER_PROMPT_TEMPLATE
    rendered = autopilot._format_available_action_schemas(["deep_eval"])
    constitution = autopilot.CONSTITUTION_PATH.read_text(encoding="utf-8")

    assert "Supported tiers: 0, 1, 2, or 3" in rendered
    assert autopilot._format_deep_eval_tier_options() == "0, 1, 2, or 3"
    assert "Only tier is supported" not in template
    assert "Supported tiers: 0, 1, or 2" not in rendered
    assert "T0, T1, T2, and T3" in constitution
    assert "T3 is the expert/hard workflow validation lane" in constitution
    assert "A lower T2 or T3 number is not a T1 regression" in constitution


def test_controller_prompt_includes_outcome_progress_pressure_section() -> None:
    template = autopilot.CONTROLLER_PROMPT_TEMPLATE

    assert "Outcome Progress Pressure" in template
    assert "{outcome_progress_pressure}" in template
    assert "planner-learning, non-authority" in template


def test_higher_tier_pressure_preserves_same_tier_comparison() -> None:
    class FakeArchive:
        def summary(self, *, tier):
            if tier == 1:
                return {
                    "frontier_size": 4,
                    "best_quality": 2.0,
                    "best_speed": 42.0,
                    "hv_slope_50": 0.0002,
                }
            if tier == 2:
                return {
                    "frontier_size": 2,
                    "best_quality": 1.4,
                    "best_speed": 18.5,
                    "hv_slope_50": 0.0001,
                }
            if tier == 3:
                return {
                    "frontier_size": 0,
                    "best_quality": 0.0,
                    "best_speed": 0.0,
                    "hv_slope_50": 0.0,
                }
            raise AssertionError(f"unexpected tier {tier}")

    class FakeBaseline:
        def quality_for_tier(self, tier):
            return {2: 1.1, 3: 0.2}[tier]

    text = autopilot._build_higher_tier_planner_pressure(
        FakeArchive(),
        SimpleNamespace(baseline=FakeBaseline()),
    )

    assert "expert/hard workflow tasks" in text
    assert "prefer deep_eval tier 3 if T3 coverage/frontier is thin" in text
    assert "Never compare raw quality across tiers" in text
    assert "T1 gains that never lift T2/T3 are overfit risk" in text
    assert "T3 hard-workflow probes should favor technical tool-use, REPL" in text
    assert "Plateau signal: T1 hv_slope_50=+0.000200, T2 hv_slope_50=+0.000100" in text
    assert "until the next instrument/kernel era resets frontier-speed evidence" in text
    assert "T2: frontier=2, best_q=1.400, delta_vs_baseline=+0.300" in text
    assert "T3: empty frontier; baseline_q=0.200" in text
    assert "deployment safety lane" in text


def test_higher_tier_pressure_defers_deep_eval_when_w8_candidate_generation_active() -> None:
    class FakeArchive:
        def summary(self, *, tier):
            if tier == 1:
                return {
                    "frontier_size": 4,
                    "best_quality": 2.0,
                    "best_speed": 42.0,
                    "hv_slope_50": 0.0002,
                }
            if tier == 2:
                return {
                    "frontier_size": 0,
                    "best_quality": 0.0,
                    "best_speed": 0.0,
                    "hv_slope_50": 0.0,
                }
            if tier == 3:
                return {
                    "frontier_size": 0,
                    "best_quality": 0.0,
                    "best_speed": 0.0,
                    "hv_slope_50": 0.0,
                }
            raise AssertionError(f"unexpected tier {tier}")

    class FakeBaseline:
        def quality_for_tier(self, tier):
            return {2: 1.1, 3: 0.2}[tier]

    text = autopilot._build_higher_tier_planner_pressure(
        FakeArchive(),
        SimpleNamespace(baseline=FakeBaseline()),
        w8_candidate_generation_active=True,
    )

    assert "W8 candidate-generation override" in text
    assert "do not emit seed_batch, deep_eval, or structural_prune" in text
    assert "numeric_trial with journaled applied params" in text
    assert "one-flag structural_experiment" in text
    assert "prefer deep_eval tier 3 if T3 coverage/frontier is thin" not in text
    assert "schedule deep_eval tier 3" not in text
    assert "defer the deep_eval tier 3 coverage probe until W8 candidate generation clears" in text


def test_eval_coverage_pressure_reports_repeat_factor_and_pool_denominator() -> None:
    tier3_entry = _entry(
        2,
        "deep_eval",
        eval_details={
            "question_results": [
                {"suite": "coder", "qid": "a", "correct": True},
                {"suite": "agentic", "qid": "c", "correct": False},
            ]
        },
    )
    tier3_entry.tier = 3
    journal = SimpleNamespace(
        entries_with_supersessions=lambda: [
            _entry(
                1,
                "numeric_trial",
                eval_details={
                    "question_results": [
                        {"suite": "coder", "qid": "a", "correct": True},
                        {"suite": "coder", "qid": "b", "correct": False},
                    ]
                },
            ),
            tier3_entry,
        ]
    )

    text = autopilot._build_eval_coverage_pressure(
        journal,
        pool_total_questions=100,
        pool_tier_questions={1: 20, 2: 70, 3: 10},
    )

    assert "3 distinct qids / 4 scored rows" in text
    assert "repeat_factor=1.33x" in text
    assert "pool_coverage<=3.00% of 100" in text
    assert "eval trials by tier: T1=1, T3=1" in text
    assert "Tier detail:" in text
    assert "T1:trials=1,rows=2,distinct=2,pool=20,coverage<=10.00%" in text
    assert "T2:trials=0,rows=0,distinct=0,pool=70,coverage<=0.00%" in text
    assert "T3:trials=1,rows=2,distinct=2" in text
    assert "pool=10,coverage<=20.00%" in text
    assert "Higher-tier coverage is thin (T2=0 trial(s), T3=1 trial(s))" in text
    assert "hard workflow, tool-use, REPL, and multi-turn task coverage" in text
    assert "Least-covered non-sentinel suites: agentic=1, coder=2" in text
    assert "under-covered suites" in text
    assert "T1-only gains are overfit risk" in text
    assert "fixed authority-core evidence separate" in text


def test_eval_coverage_pressure_defers_seed_and_deep_eval_when_w8_candidate_generation_active() -> None:
    tier3_entry = _entry(
        2,
        "deep_eval",
        eval_details={
            "question_results": [
                {"suite": "coder", "qid": "a", "correct": True},
                {"suite": "agentic", "qid": "c", "correct": False},
            ]
        },
    )
    tier3_entry.tier = 3
    journal = SimpleNamespace(
        entries_with_supersessions=lambda: [
            _entry(
                1,
                "numeric_trial",
                eval_details={
                    "question_results": [
                        {"suite": "coder", "qid": "a", "correct": True},
                        {"suite": "coder", "qid": "b", "correct": False},
                    ]
                },
            ),
            tier3_entry,
        ]
    )

    text = autopilot._build_eval_coverage_pressure(
        journal,
        pool_total_questions=100,
        pool_tier_questions={1: 20, 2: 70, 3: 10},
        w8_candidate_generation_active=True,
    )

    assert "Higher-tier coverage is thin" in text
    assert "W8 candidate generation is the active strict blocker" in text
    assert "do not propose seed_batch or deep_eval for coverage this turn" in text
    assert "available replayable candidate action" in text
    assert "Prefer an available replayable numeric_trial" in text
    assert "deep_eval tier 2/3" not in text
    assert "seed_batch on under-covered suites" not in text


def test_eval_coverage_pressure_no_rows_respects_w8_candidate_generation_active() -> None:
    journal = SimpleNamespace(entries_with_supersessions=lambda: [])

    text = autopilot._build_eval_coverage_pressure(
        journal,
        w8_candidate_generation_active=True,
    )

    assert "No scored question-result rows" in text
    assert "W8 candidate generation is the active strict blocker" in text
    assert "do not use seed_batch or deep_eval" in text


def test_outcome_progress_pressure_reports_stall_and_rates(monkeypatch) -> None:
    def fake_outcome_progress_report(**kwargs):
        assert kwargs["max_trials_since_frontier"] == 5
        assert kwargs["max_trials_since_promotion"] == 10
        assert kwargs["recent_window_trials"] == 20
        return {
            "status": "attention",
            "latest_trial_id": 1180,
            "frontier_admissions": 84,
            "latest_frontier_trial_id": 1005,
            "trials_since_frontier": 175,
            "max_trials_since_frontier": 5,
            "baseline_promotions": 1,
            "latest_promotion_trial_id": 969,
            "trials_since_promotion": 211,
            "max_trials_since_promotion": 10,
            "rates": {
                "keepable_rate": {"count": 1, "total": 20, "rate": 0.05},
                "wasted_eval_rate": {"count": 15, "total": 20, "rate": 0.75},
                "learning_excluded_rate": {"count": 4, "total": 20, "rate": 0.2},
                "regression_per_active_trial": {"count": 3, "total": 20, "rate": 0.15},
                "promotions_per_100_active_trials": {
                    "count": 1,
                    "total": 20,
                    "per_100": 5.0,
                },
            },
            "blockers": ["frontier admission stale"],
        }

    monkeypatch.setattr(
        autopilot,
        "_phase_outcome_progress_report",
        fake_outcome_progress_report,
    )

    text = autopilot._build_outcome_progress_pressure(
        max_trials_since_frontier=5,
        max_trials_since_promotion=10,
        recent_window_trials=20,
    )

    assert "status=attention" in text
    assert "trials_since_frontier=175/5" in text
    assert "regression_per_active_trial=3/20 (15.0%)" in text
    assert "promotions_per_100_active_trials=1/20 (5.0/100)" in text
    assert "trials_since_promotion=211/10" in text
    assert "keepable=1/20 (5.0%)" in text
    assert "wasted_eval=15/20 (75.0%)" in text
    assert "learning_excluded=4/20 (20.0%)" in text
    assert "Outcome blockers: frontier admission stale" in text
    assert "credible path to keepable frontier or promotion evidence" in text
    assert "seed-only churn" in text


def test_controller_prompt_uses_fresh_strategy_hints_section(monkeypatch) -> None:
    monkeypatch.setattr(autopilot, "_PLANNER_HINTS_ENABLED", True)
    rows: list[SimpleNamespace] = []

    class FakeStore:
        def retrieve_conventions(self, *, species, journal, limit):
            assert species
            assert journal == "journal"
            assert limit == 3
            return []

        def retrieve_for_journal(self, query, *, journal, k, species):
            assert query
            assert journal == "journal"
            assert k == 3
            if species == "structural_lab":
                return list(rows)
            return []

    def format_controller_prompt() -> str:
        planner_strategy_hints = autopilot._build_planner_strategy_hints(
            FakeStore(),
            "journal",
            max_rows=3,
        )
        return autopilot.CONTROLLER_PROMPT_TEMPLATE.format(
            constitution="constitution",
            system_card="system-card",
            pareto_summary="pareto",
            pareto_geometry="geometry",
            planner_evidence="evidence",
            journal_trustworthiness="trust",
            hypotheses_under_test="hypotheses",
            journal_summary="journal-summary",
            seeder_status="seeder",
            batch_telemetry="batch",
            species_effectiveness="species",
            health_status="OK",
            memory_count=1,
            converged=False,
            slot_memory="slots",
            action_availability="actions",
            available_action_schemas="schemas",
            model_gate_advisory="fable-gate",
            higher_tier_pressure="higher-tier",
            eval_coverage_pressure="coverage",
            outcome_progress_pressure="outcome-progress",
            planner_strategy_hints=planner_strategy_hints,
            repo_readiness_advisory="repo",
            budget="budget",
            suite_quality_trends="suite-trends",
            insights_structured="insights",
            stagnation_signal="none",
            exploration_block="explore",
            short_term_memory="stm",
            prior_planner_decisions="prior",
            last_criticism="criticism",
            model_signatures="models",
            blacklist_text="blacklist",
            operator_outbox_feedback="outbox",
            operator_hypotheses_block="operator-hypotheses",
            vidya_settled_ground_block="settled-ground",
            vidya_kvq_evidence_block="kvq-evidence",
            feature_flags_block="flags",
            last_invalid_feedback="invalid",
            plot_paths="plots",
        )

    first_prompt = format_controller_prompt()
    rows.append(
        SimpleNamespace(
            id="fresh-planner-tool-hint",
            species="structural_lab",
            entry_type="pattern",
            title="Fresh planner-loop tool hint",
            generalized_content=(
                "Steer the next planner turn toward a bounded REPL/tool-use "
                "measurement when the sentinel lane is enabled."
            ),
            metadata={
                "bind_status": "future",
                "bind_identifiers": ["tools", "repl", "react_mode"],
                "source_handoff": "tool-use-eval-contract",
            },
        )
    )
    second_prompt = format_controller_prompt()

    assert "### StrategyStore Planner Hints (refreshed each planner turn)" in first_prompt
    assert "### Fable 5 Gate Advisory (latest generated report, non-authority)" in first_prompt
    assert "no StrategyStore rows matched" in first_prompt
    assert "Fresh planner-loop tool hint" not in first_prompt
    assert "Fresh planner-loop tool hint" in second_prompt
    assert "tools,repl,react_mode" in second_prompt


def test_slot_query_ports_from_stack_priors_uses_live_primary_llama_entries(tmp_path: Path) -> None:
    priors = tmp_path / "stack_priors.yaml"
    priors.write_text(
        """
roles:
  frontdoor:
    deployment_status: live_stack
    serving:
      binary: llama.cpp
      launch:
        entries:
          - {port: 8070, alias: false}
          - {port: 8080, alias: false}
  coder_escalation:
    deployment_status: live_stack
    serving:
      binary: llama.cpp
      launch:
        entries:
          - {port: 8070, alias: true}
  worker_general:
    deployment_status: live_stack
    serving:
      binary: ik-pr1744
      launch:
        runtime:
          binary_path: /mnt/raid0/llm/ik_llama.cpp/build/bin/llama-server
        entries:
          - {port: 8072, alias: false}
  reap_candidate:
    deployment_status: benchmark_only
    serving:
      binary: llama.cpp
      launch:
        entries:
          - {port: 8099, alias: false}
  embedder:
    deployment_status: live_stack
    serving:
      binary: embedding-server
      launch:
        entries:
          - {port: 8090, alias: false}
""".lstrip(),
        encoding="utf-8",
    )

    assert autopilot._slot_query_ports_from_stack_priors(priors) == {
        "frontdoor": [8070, 8080],
        "worker_general": [8072],
    }


def test_slot_query_ports_falls_back_when_stack_priors_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(autopilot, "_slot_query_ports_from_stack_priors", lambda: {})

    assert autopilot._slot_query_ports() == autopilot._FALLBACK_SLOT_QUERY_PORTS


def test_query_slot_memory_separates_empty_and_unavailable_ports(monkeypatch) -> None:
    monkeypatch.setattr(
        autopilot,
        "_slot_query_ports",
        lambda: {"frontdoor": [8070], "worker_general": [8072]},
    )

    class _Response:
        def __init__(self, status_code: int, payload: list[dict] | None = None) -> None:
            self.status_code = status_code
            self._payload = payload or []

        def json(self) -> list[dict]:
            return self._payload

    def _fake_get(url: str, timeout: float) -> _Response:
        assert timeout == 3.0
        if url.endswith(":8070/slots"):
            return _Response(200, [{"id": 0, "state": "idle", "n_past": 0}])
        raise OSError("connection refused")

    monkeypatch.setattr("httpx.get", _fake_get)

    text = autopilot._query_slot_memory()

    assert "healthy queried ports with empty KV cache: frontdoor:8070" in text
    assert "unavailable configured replica ports: worker_general:8072" in text
    assert "do not infer eval-instrument contamination" in text
    assert "all slots empty or servers offline" not in text


def test_build_exploration_block_resilient_to_archive_errors(tmp_path: Path) -> None:
    j = _well_populated_journal(
        tmp_path,
        ["seed_batch", "numeric_trial", "prompt_mutation",
         "code_mutation", "structural_experiment", "deep_eval"],
    )

    class _BrokenArchive:
        def geometry(self) -> dict:
            raise RuntimeError("boom")

    block, signal = autopilot._build_exploration_block(
        journal=j, archive=_BrokenArchive(), known_actions=KNOWN_ACTIONS,
    )
    # Should still return a usable lean block (no other stagnation signals).
    assert "alternatives" in block
    assert signal == "none (lean prompt)"
