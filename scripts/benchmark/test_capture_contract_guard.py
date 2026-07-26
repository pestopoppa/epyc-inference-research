"""Static CI guard for lossless benchmark capture and scoring boundaries.

This is intentionally narrow.  It covers the three paths that turn a model
prompt/response into a score or a SWE patch.  Display-only tails and previews
elsewhere are outside its scope; they must not be reused as scoring input.
"""
from __future__ import annotations

import ast
from pathlib import Path


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
RUNNER = RESEARCH_ROOT / "scripts/benchmark/v7_quality_gate_runner.py"
JUDGE_SCORER = RESEARCH_ROOT / "scripts/benchmark/score_with_claude.py"
SWE_CONVERTER = RESEARCH_ROOT / "artifacts/architect-code-eval-20260724/convert_sr_to_patch.py"


def sliced_capture_names(path: Path) -> list[str]:
    """Return direct prompt/response slices in a capture or scorer source file.

    These files have no display-only use of the two payload variables.  A slice
    here is therefore an accidental loss channel, including ``response[-4000:]``.
    The check deliberately does not scan repository-wide log/preview helpers.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
            continue
        if node.value.id not in {"prompt", "response"}:
            continue
        if isinstance(node.slice, ast.Slice):
            names.append(node.value.id)
    return names


def test_capture_and_judge_paths_never_slice_model_payloads():
    for path in (RUNNER, JUDGE_SCORER, SWE_CONVERTER):
        assert sliced_capture_names(path) == [], (
            f"{path.relative_to(RESEARCH_ROOT)} slices a prompt/response in a "
            "scoring or conversion path; preserve it losslessly or mark the row "
            "provisional before any scoring."
        )


def test_guard_detects_the_historical_response_tail_loss(tmp_path: Path):
    regressed = tmp_path / "regressed_runner.py"
    regressed.write_text("stored = response[-4000:]\n", encoding="utf-8")

    assert sliced_capture_names(regressed) == ["response"]


def test_runner_contract_persists_full_response_and_provenance():
    source = RUNNER.read_text(encoding="utf-8")
    required_fragments = (
        'CAPTURE_SCHEMA_VERSION = "v7_quality_gate_capture.v4"',
        '"prompt": prompt',
        '"prompt_fingerprint": prompt_fingerprint',
        '"runner_source_sha256": RUNNER_SOURCE_SHA256',
        '"response_fingerprint": response_fingerprint',
        '"response": response',
        '"reasoning": reasoning',
        "write_live_capture_status(",
    )
    for fragment in required_fragments:
        assert fragment in source, f"runner capture contract missing {fragment!r}"


def test_judge_contract_sends_fingerprinted_full_payload_or_marks_ineligible():
    source = JUDGE_SCORER.read_text(encoding="utf-8")
    required_fragments = (
        '"prompt_identity": text_identity(prompt)',
        '"response_identity": text_identity(response)',
        '"serialized_payload": scorer_input',
        'data = scorer_input["serialized_payload"]',
        "provisional_input_over_budget",
    )
    for fragment in required_fragments:
        assert fragment in source, f"judge contract missing {fragment!r}"


def test_converter_rejects_missing_or_mismatched_current_capture_fingerprints():
    source = SWE_CONVERTER.read_text(encoding="utf-8")
    required_fragments = (
        'CURRENT_CAPTURE_SCHEMA = "v7_quality_gate_capture.v4"',
        'runner_fingerprint == computed',
        '"response_fingerprint_status": fingerprint_status',
        'and prompt_status == "verified"',
        '"scoring_eligible": scoring_eligible',
    )
    for fragment in required_fragments:
        assert fragment in source, f"converter capture contract missing {fragment!r}"
