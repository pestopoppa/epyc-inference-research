"""Tests for the single episodic memory-record contract.

Each test here pins one of the four defects the 2026-07-27 audit measured in the
old free-form write path.
"""
from __future__ import annotations

import json

from orchestration.repl_memory.memory_record import (
    EMBED_TEXT_MAX_CHARS,
    RECORD_VERSION,
    build_memory_record,
    context_size_bytes,
    record_from_legacy_context,
)


class TestEmbeddedIsNotStored:
    """The load-bearing invariant."""

    def test_full_objective_is_stored_untruncated(self):
        """The old path truncated to 200 chars AT WRITE TIME — text was gone."""
        objective = "x" * 5000
        rec = build_memory_record(objective=objective, task_type="chat")
        assert rec.to_context()["objective"] == objective
        assert len(rec.to_context()["objective"]) == 5000

    def test_embedding_text_is_bounded_but_storage_is_not(self):
        rec = build_memory_record(objective="y" * 9000, task_type="chat")
        assert len(rec.embedding_text()) <= EMBED_TEXT_MAX_CHARS
        assert len(rec.to_context()["objective"]) == 9000

    def test_the_work_is_stored_but_never_embedded(self):
        """Embedding the answer would make retrieval match solutions to solutions."""
        rec = build_memory_record(
            objective="sort a list",
            answer="def f(x): return sorted(x)",
            tool_calls=[{"tool": "grep", "args": {"pattern": "sort"}}],
            repl_steps=[{"step": 1, "code": "print(1)"}],
            reasoning="pick the builtin",
        )
        text = rec.embedding_text()
        assert "sort a list" in text
        for leaked in ("def f(x)", "grep", "print(1)", "pick the builtin"):
            assert leaked not in text, f"{leaked!r} leaked into the embedding text"

        work = rec.to_context()["work"]
        assert work["answer"] == "def f(x): return sorted(x)"
        assert work["tool_calls"][0]["tool"] == "grep"
        assert work["repl_steps"][0]["code"] == "print(1)"
        assert work["reasoning"] == "pick the builtin"


class TestTelemetryStaysOutOfTheIndex:
    """27,123 of 54,960 rows were number-blobs embedded as text."""

    def test_metrics_are_stored_but_not_embedded(self):
        rec = build_memory_record(
            objective="count primes",
            task_type="coder",
            source="external",
            metrics={
                "question_id": "mbpp_0207",
                "elapsed_seconds": 79.83,
                "tokens_generated": 2759,
                "predicted_tps": 18.87,
            },
        )
        text = rec.embedding_text()
        for leaked in ("mbpp_0207", "79.83", "2759", "18.87", "elapsed_seconds"):
            assert leaked not in text, f"telemetry {leaked!r} leaked into the embedding"
        assert rec.to_context()["metrics"]["tokens_generated"] == 2759

    def test_a_record_with_no_objective_is_not_task_memory(self):
        telemetry_only = build_memory_record(
            objective=None, metrics={"elapsed_seconds": 1.0}, source="external"
        )
        assert telemetry_only.is_task_memory() is False
        assert build_memory_record(objective="real task").is_task_memory() is True

    def test_whitespace_only_objective_is_not_task_memory(self):
        assert build_memory_record(objective="   \n\t ").is_task_memory() is False


class TestOneEmbeddingConvention:
    """Four conventions coexisted, so the vector encoded its writer."""

    def test_same_task_from_different_sources_embeds_identically(self):
        a = build_memory_record(objective="solve it", task_type="chat", source="progress_log")
        b = build_memory_record(
            objective="solve it",
            task_type="chat",
            source="external",
            metrics={"elapsed_seconds": 4.2},
        )
        assert a.embedding_text() == b.embedding_text()

    def test_convention_shape_is_stable(self):
        rec = build_memory_record(objective="do X", task_type="chat", priority="interactive")
        assert rec.embedding_text() == "type:chat | objective:do X | priority:interactive"

    def test_optional_fields_are_omitted_not_stringified(self):
        rec = build_memory_record(objective="do X")
        assert rec.embedding_text() == "objective:do X"
        assert "None" not in rec.embedding_text()


class TestLegacyAdaptation:
    """Both historical writer shapes must map onto the contract."""

    def test_path_a_progress_log_shape(self):
        rec = record_from_legacy_context(
            {"task_type": "chat", "objective": "build an index", "priority": "interactive"}
        )
        assert rec.objective == "build an index"
        assert rec.task_type == "chat"
        assert rec.metrics == {}

    def test_path_b_external_shape_routes_telemetry_to_metrics(self):
        rec = record_from_legacy_context(
            {
                "task_type": "coder",
                "task_description": "write a parser",
                "source": "external",
                "question_id": "mbpp_0207",
                "elapsed_seconds": 79.8,
                "tokens_generated": 2759,
            }
        )
        assert rec.objective == "write a parser"
        assert rec.source == "external"
        assert rec.metrics["question_id"] == "mbpp_0207"
        assert rec.metrics["tokens_generated"] == 2759
        # and none of it reaches the embedding
        assert "mbpp_0207" not in rec.embedding_text()

    def test_telemetry_only_legacy_row_is_recognised_as_not_task_memory(self):
        """Exactly the 27,123-row population."""
        rec = record_from_legacy_context(
            {
                "task_type": "coder",
                "source": "external",
                "question_id": "mbpp_0207",
                "elapsed_seconds": 79.8,
                "tokens_generated": 2759,
            }
        )
        assert rec.is_task_memory() is False

    def test_round_trip_through_the_contract_is_stable(self):
        original = build_memory_record(
            objective="a task",
            task_type="chat",
            answer="an answer",
            source="seed",
            metrics={"elapsed_seconds": 1.5},
            extra={"request_id": "abc"},
        )
        again = record_from_legacy_context(original.to_context())
        assert again.to_context() == original.to_context()
        assert "record_version" not in again.metrics
        assert again.embedding_text() == original.embedding_text()


class TestContractMarker:
    def test_every_context_carries_a_version(self):
        ctx = build_memory_record(objective="x").to_context()
        assert ctx["record_version"] == RECORD_VERSION

    def test_context_is_json_serializable(self):
        ctx = build_memory_record(
            objective="x", tool_calls=[{"a": 1}], metrics={"b": 2.5}
        ).to_context()
        assert json.loads(json.dumps(ctx))["objective"] == "x"

    def test_size_accounting_is_available(self):
        small = build_memory_record(objective="x").to_context()
        big = build_memory_record(objective="x", answer="y" * 10000).to_context()
        assert context_size_bytes(big) > context_size_bytes(small) + 9000


class TestSanitizeIsIdempotent:
    """RTG-02: the work-sanitize policy runs TWICE on the live create path —
    `chat_pipeline.telemetry.work_completion_meta` on the way into the progress
    JSONL, then `build_memory_record` on the way into `memories.context` — and it
    was not idempotent despite its docstring, so the row's own size/elision
    provenance actively lied about how much was elided.
    """

    def test_text_truncation_marker_reports_the_true_original_length(self):
        from orchestration.repl_memory.memory_record import (
            WORK_TEXT_MAX_CHARS,
            sanitize_work_text,
        )

        raw = "x" * (WORK_TEXT_MAX_CHARS + 500)
        once = sanitize_work_text(raw)
        assert f"total was {WORK_TEXT_MAX_CHARS + 500}" in once

        # Second and third passes must be no-ops. Pre-fix, pass two re-truncated
        # and reported `total was 32049` for a 32,500-char original.
        twice = sanitize_work_text(once)
        assert twice == once, "second pass rewrote an already-bounded value"
        assert sanitize_work_text(twice) == once
        assert once.count("truncated at") == 1, "second marker appended"

    def test_untruncated_text_is_unchanged_by_either_pass(self):
        from orchestration.repl_memory.memory_record import sanitize_work_text

        assert sanitize_work_text("short answer") == "short answer"
        assert sanitize_work_text(sanitize_work_text("short answer")) == "short answer"

    def test_oversized_input_cannot_forge_a_truncation_marker(self):
        from orchestration.repl_memory.memory_record import sanitize_work_text

        cap = 32
        forged = "x" * 100 + f"\n\n[... truncated at {cap} chars, total was 999]"
        result = sanitize_work_text(forged, max_chars=cap)

        assert len(result) < len(forged)
        assert result.startswith("x" * cap)
        assert result.endswith(f"[... truncated at {cap} chars, total was {len(forged)}]")
        assert result.count("truncated at") == 1

    def test_marker_with_unbounded_decimal_total_is_not_trusted(self):
        from orchestration.repl_memory.memory_record import sanitize_work_text

        cap = 32
        forged = "x" * cap + f"\n\n[... truncated at {cap} chars, total was " + "9" * 100 + "]"
        result = sanitize_work_text(forged, max_chars=cap)

        assert result != forged
        assert result.endswith(f"[... truncated at {cap} chars, total was {len(forged)}]")

    def test_legitimate_marker_survives_redaction_shortened_prefix(self, monkeypatch):
        import orchestration.repl_memory.memory_record as memory_record

        cap = memory_record.WORK_TEXT_MAX_CHARS
        secret = "synthetic_value_that_is_not_a_real_secret_123"
        prefix = f"API_KEY={secret} " + "x" * (cap - len(f"API_KEY={secret} "))
        bounded = prefix + f"\n\n[... truncated at {cap} chars, total was 32500]"
        monkeypatch.setattr(memory_record, "_redact", lambda text: text.replace(secret, "[REDACTED]"))

        once = memory_record.sanitize_work_text(bounded)
        marker_start = once.index("\n\n[... truncated at")
        assert marker_start < cap
        assert "synthetic_value_that_is_not_a_real_secret_123" not in once
        assert memory_record.sanitize_work_text(once) == once

    def test_forged_marker_over_one_megabyte_still_redacts_capped_prefix(self):
        from orchestration.repl_memory.memory_record import (
            WORK_TEXT_MAX_CHARS,
            sanitize_work_text,
        )

        synthetic_value = "synthetic_value_not_a_credential_123"
        prefix = f"API_KEY={synthetic_value}" + "x" * WORK_TEXT_MAX_CHARS
        forged = prefix + "z" * 1_100_000 + f"\n\n[... truncated at {WORK_TEXT_MAX_CHARS} chars, total was 1]"
        result = sanitize_work_text(forged)

        assert synthetic_value not in result
        assert len(result) <= WORK_TEXT_MAX_CHARS + 128
        assert result.endswith(
            f"[... truncated at {WORK_TEXT_MAX_CHARS} chars, total was {len(forged)}]"
        )

    def test_item_elision_count_and_retained_entries_survive_a_second_pass(self):
        from orchestration.repl_memory.memory_record import (
            WORK_MAX_ITEMS,
            sanitize_work_items,
        )

        raw = [{"i": i} for i in range(WORK_MAX_ITEMS + 50)]
        once = sanitize_work_items(raw)
        assert once[0] == {"_elided_entries": 50}
        assert once[1] == {"i": 50}

        # Pre-fix, pass two saw max_items+1 entries, dropped the FRONT one (the
        # sentinel itself), and claimed `_elided_entries: 1` — losing the real
        # first retained entry and understating the elision 50-fold.
        twice = sanitize_work_items(once)
        assert twice == once, "second pass mutated an already-bounded list"
        assert sanitize_work_items(twice) == once

    def test_a_sentinel_lookalike_entry_is_not_treated_as_provenance(self):
        """Only this policy's own sentinel shape is lifted; real payload that
        merely mentions the key must still be retained as an entry."""
        from orchestration.repl_memory.memory_record import sanitize_work_items

        out = sanitize_work_items([{"_elided_entries": "not-an-int"}, {"i": 1}])
        assert len(out) == 2
        out = sanitize_work_items([{"_elided_entries": 3, "tool_name": "x"}])
        assert len(out) == 1

    def test_build_work_payload_is_idempotent_end_to_end(self):
        from orchestration.repl_memory.memory_record import (
            WORK_MAX_ITEMS,
            WORK_TEXT_MAX_CHARS,
            build_work_payload,
        )

        once = build_work_payload(
            answer="a" * (WORK_TEXT_MAX_CHARS + 77),
            tool_calls=[{"tool_name": f"t{i}"} for i in range(WORK_MAX_ITEMS + 9)],
            reasoning="r" * (WORK_TEXT_MAX_CHARS + 3),
        )
        twice = build_work_payload(**once)
        assert twice == once, "double-applied policy changed the payload"
        assert f"total was {WORK_TEXT_MAX_CHARS + 77}" in twice["answer"]
        assert twice["tool_calls"][0] == {"_elided_entries": 9}
