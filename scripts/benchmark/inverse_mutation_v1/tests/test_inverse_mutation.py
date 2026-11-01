from __future__ import annotations

import pytest

from scripts.benchmark.gate_verdict_vocab import VERDICT_FAIL, VERDICT_OUT_OF_COVERAGE, VERDICT_PASS
from scripts.benchmark.inverse_mutation_v1 import fixtures, scorer


def test_generates_all_four_reversible_mutation_kinds_from_pinned_source() -> None:
    generated = fixtures.generate_fixtures()
    assert {fixture.mutation for fixture in generated} == {
        "operator-swap",
        "boolean-flip",
        "off-by-one",
        "removed-guard",
    }
    assert len({fixture.fixture_id for fixture in generated}) == 4
    for fixture in generated:
        assert fixture.mutated_source != fixture.expected_source
        assert fixture.expected_source == fixture.source
        assert fixture.provenance["commit"] == fixtures.SOURCE_COMMIT
        assert fixture.provenance["source_sha256"] == fixtures.SOURCE_SHA256
        assert fixture.provenance["license"].startswith("MIT")


@pytest.mark.parametrize("fixture", fixtures.generate_fixtures(), ids=lambda f: f.mutation)
def test_correct_inverse_and_format_only_variants_pass(fixture: fixtures.Fixture) -> None:
    assert scorer.score(fixture.fixture_id, fixture.expected_source)["verdict"] == VERDICT_PASS
    # Ruff removes trailing whitespace and normalizes the final newline.
    formatted_variant = fixture.expected_source.rstrip() + "   \n"
    assert scorer.score(fixture.fixture_id, formatted_variant)["verdict"] == VERDICT_PASS


@pytest.mark.parametrize("fixture", fixtures.generate_fixtures(), ids=lambda f: f.mutation)
def test_mutant_and_wrong_inverse_fail(fixture: fixtures.Fixture) -> None:
    assert scorer.score(fixture.fixture_id, fixture.mutated_source)["verdict"] == VERDICT_FAIL
    wrong = fixture.expected_source.replace("b + c", "b - c", 1)
    assert scorer.score(fixture.fixture_id, wrong)["verdict"] == VERDICT_FAIL


def test_actor_payload_does_not_expose_reference_or_provenance() -> None:
    payload = fixtures.generate_fixtures()[0].task_payload()
    assert set(payload) == {"id", "instruction", "source"}
    assert "expected_source" not in payload
    assert "provenance" not in payload


def test_score_record_captures_decision_pins_and_hashes() -> None:
    fixture = fixtures.generate_fixtures()[0]
    result = scorer.score(fixture.fixture_id, fixture.expected_source)
    assert result["recorded_at_utc"].endswith("+00:00")
    assert result["instruction"] == fixture.instruction
    assert result["instruction_sha256"]
    assert result["source_sha256"] == fixture.provenance["source_sha256"]
    assert result["mutation"] == fixture.mutation
    assert result["candidate_raw_sha256"]
    assert result["candidate_normalized_sha256"]
    assert result["expected_normalized_sha256"]
    assert result["formatter"] == {
        "version": scorer.RUFF_VERSION,
        "configuration": scorer.FORMAT_CONFIG,
        "executable_sha256": scorer.RUFF_SHA256,
    }


def test_absent_candidate_and_unknown_fixture_are_out_of_coverage() -> None:
    assert scorer.score("cj7a-operator-swap-001", None)["verdict"] == VERDICT_OUT_OF_COVERAGE
    assert (
        scorer.score("missing-fixture", "print('candidate')\n")["verdict"]
        == VERDICT_OUT_OF_COVERAGE
    )


def test_formatter_unavailable_is_not_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable() -> tuple:
        raise scorer.FormatUnknown("unsupported", "test formatter unavailable")

    monkeypatch.setattr(scorer, "_formatter", unavailable)
    result = scorer.score("cj7a-operator-swap-001", "anything")
    assert result["verdict"] == VERDICT_OUT_OF_COVERAGE
    assert result["cause"] == "unsupported"
    assert result["candidate_raw_sha256"]
    assert result["candidate_normalized_sha256"] is None
    assert result["expected_normalized_sha256"] is None
    assert result["formatter"]["executable_sha256"] is None


def test_formatter_timeout_is_not_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout() -> tuple:
        raise scorer.FormatUnknown("timeout", "test timeout")

    monkeypatch.setattr(scorer, "_formatter", timeout)
    result = scorer.score("cj7a-operator-swap-001", "anything")
    assert result["verdict"] == VERDICT_OUT_OF_COVERAGE
    assert result["cause"] == "timeout"


def test_formatter_refusal_keeps_observed_identity_but_no_normalized_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuses(_: str, __, formatter: dict) -> str:
        raise scorer.FormatUnknown("checker_error", "test refusal", formatter)

    monkeypatch.setattr(scorer, "_format_with", refuses)
    result = scorer.score("cj7a-operator-swap-001", "candidate")
    assert result["verdict"] == VERDICT_OUT_OF_COVERAGE
    assert result["cause"] == "checker_error"
    assert result["candidate_raw_sha256"]
    assert result["candidate_normalized_sha256"] is None
    assert result["expected_normalized_sha256"] is None
    assert result["formatter"] == {
        "version": scorer.RUFF_VERSION,
        "configuration": scorer.FORMAT_CONFIG,
        "executable_sha256": scorer.RUFF_SHA256,
    }


def test_hostile_ambient_ruff_config_cannot_change_score_or_normalization(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fixture = fixtures.generate_fixtures()[0]
    before = scorer.score(fixture.fixture_id, fixture.expected_source)
    (tmp_path / ".ruff.toml").write_text(
        'line-length = 20\nindent-width = 2\nquote-style = "single"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    after = scorer.score(fixture.fixture_id, fixture.expected_source)
    assert after["verdict"] == before["verdict"] == VERDICT_PASS
    assert after["candidate_normalized_sha256"] == before["candidate_normalized_sha256"]
    assert after["expected_normalized_sha256"] == before["expected_normalized_sha256"]
    assert after["formatter"] == before["formatter"]
