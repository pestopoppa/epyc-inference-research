"""Tests for the review_f1 harness (mock-transport + dry-run; NO inference).

Runnable both under pytest and stand-alone. Uses MockTransport and a temp out
dir so nothing contacts a server.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import assemble_golden_set  # noqa: E402
import conftest  # noqa: E402
import harness  # noqa: E402


def _config(out_dir, runs=3, resume=True):
    return harness.HarnessConfig(
        golden_path=str(conftest.SYNTHETIC_GOLDEN),
        out_dir=out_dir,
        model="gemma4-26B-A4B",
        quant="Q4_K_M",
        judge_model="qwen3-coder-30B-A3B",
        judge_quant="Q4_K_M",
        runs=runs,
        resume=resume,
    )


def _perfect_responder(golden):
    by_case = {}
    for case in golden["cases"]:
        findings = [
            {"criterion": g["criterion"], "file": g["location"]["file"],
             "line_start": g["location"]["line_start"], "line_end": g["location"]["line_end"],
             "comment": g["comment"]}
            for g in case["golden_findings"] if g["severity"] != "low"
        ]
        by_case[case["case_id"]] = json.dumps({"findings": findings})
    return by_case


def test_parse_findings_tolerates_prose_wrapping():
    content = 'Sure, here is my review:\n{"findings": [{"criterion": "logic_bug", "file": "a.py", "line_start": 3}]}\nDone.'
    findings = harness.parse_findings(content)
    assert len(findings) == 1
    assert findings[0]["criterion"] == "logic_bug"
    assert findings[0]["location"]["file"] == "a.py"


def test_parse_findings_empty_on_garbage():
    assert harness.parse_findings("not json at all") == []
    assert harness.parse_findings("") == []


def test_dry_run_contacts_no_server():
    with tempfile.TemporaryDirectory() as d:
        cfg = _config(d)
        plan = harness.print_dry_run(cfg)
        assert plan["endpoint"] == "/v1/chat/completions"
        assert plan["n_prs"] == 3 and plan["runs_per_pr"] == 3
        assert plan["total_requests"] == 9
        assert plan["enable_thinking"] is False
        assert plan["model_quant_key"] == "gemma4-26B-A4B__Q4_K_M"
        # Nothing written under the result root.
        assert not (Path(d) / "gemma4-26B-A4B__Q4_K_M").exists()


def test_mock_transport_end_to_end_perfect_score():
    golden = conftest.load_synthetic_golden()
    with tempfile.TemporaryDirectory() as d:
        cfg = _config(d)
        transport = harness.MockTransport(_perfect_responder(golden))
        summary = harness.run(cfg, transport)
        assert summary["aggregate"]["mean_f1"] == 1.0
        assert summary["aggregate"]["protocol_ok"] is True
        assert len(transport.calls) == 3 * 3  # 3 PRs x 3 runs


def test_results_indexed_by_model_quant_not_role():
    golden = conftest.load_synthetic_golden()
    with tempfile.TemporaryDirectory() as d:
        cfg = _config(d)
        harness.run(cfg, harness.MockTransport(_perfect_responder(golden)))
        root = Path(d) / "gemma4-26B-A4B__Q4_K_M"
        assert root.is_dir()
        pr_files = sorted(p.name for p in root.glob("*.json") if not p.name.startswith("_"))
        assert len(pr_files) == 3  # one file per PR
        assert (root / "_summary.json").exists()


def test_per_pr_incremental_persistence_and_resume():
    golden = conftest.load_synthetic_golden()
    with tempfile.TemporaryDirectory() as d:
        # First pass: only 1 run.
        cfg1 = _config(d, runs=1)
        harness.run(cfg1, harness.MockTransport(_perfect_responder(golden)))
        pr_path = Path(d) / "gemma4-26B-A4B__Q4_K_M" / (golden["cases"][0]["case_id"] + ".json")
        assert len(json.loads(pr_path.read_text())["runs"]) == 1

        # Second pass: 3 runs, resume=True -> only 2 more requests per PR.
        cfg2 = _config(d, runs=3, resume=True)
        transport = harness.MockTransport(_perfect_responder(golden))
        harness.run(cfg2, transport)
        assert len(json.loads(pr_path.read_text())["runs"]) == 3
        assert len(transport.calls) == 3 * 2  # 3 PRs x (3-1) remaining runs


def test_resume_skips_fully_complete_prs():
    golden = conftest.load_synthetic_golden()
    with tempfile.TemporaryDirectory() as d:
        cfg = _config(d, runs=3)
        harness.run(cfg, harness.MockTransport(_perfect_responder(golden)))
        # Re-run with resume: everything already complete -> zero requests.
        transport = harness.MockTransport(_perfect_responder(golden))
        summary = harness.run(cfg, transport)
        assert len(transport.calls) == 0
        assert summary["cases_skipped_resume"] == 3


def test_judge_swap_config_recorded():
    golden = conftest.load_synthetic_golden()
    with tempfile.TemporaryDirectory() as d:
        cfg = _config(d)
        summary = harness.run(cfg, harness.MockTransport(_perfect_responder(golden)))
        jc = summary["judge_config"]
        assert jc["judge_model"] == "qwen3-coder-30B-A3B"
        assert jc["cross_family_required"] is True
        assert jc["swap_tolerance_pp"] == 2.0
        # Also persisted in each per-PR file.
        pr_path = Path(d) / "gemma4-26B-A4B__Q4_K_M" / (golden["cases"][0]["case_id"] + ".json")
        assert json.loads(pr_path.read_text())["judge_config"]["judge_model"] == "qwen3-coder-30B-A3B"


def test_payload_shape_has_no_thinking_and_seeded_runs():
    golden = conftest.load_synthetic_golden()
    cfg = _config("/tmp/unused")
    p0 = harness.build_payload(golden["cases"][0], 0, cfg)
    p1 = harness.build_payload(golden["cases"][0], 1, cfg)
    assert p0["enable_thinking"] is False
    assert p0["seed"] == 42 and p1["seed"] == 43  # per-run seed for StdDev
    assert p0["messages"][0]["role"] == "system"
    assert p0["model"] == "gemma4-26B-A4B"


def test_assemble_checksum_is_deterministic():
    # Re-assembling the same raw fixtures yields the same checksum every time.
    a = assemble_golden_set.assemble(str(conftest.RAW_SAMPLE_DIR), "augment-v1-synthetic")
    b = assemble_golden_set.assemble(str(conftest.RAW_SAMPLE_DIR), "augment-v1-synthetic")
    assert a["checksum"] == b["checksum"]
    # And it matches the checked-in assembled golden set.
    assert a["checksum"] == conftest.load_synthetic_golden()["checksum"]


def test_assemble_counts_exclude_low_from_scored():
    a = assemble_golden_set.assemble(str(conftest.RAW_SAMPLE_DIR), "augment-v1-synthetic")
    assert a["n_cases"] == 3
    assert a["n_golden_total"] == 8  # 3 + 2 + 3
    assert a["n_golden_scored"] == 6  # two low-severity findings excluded


def test_assemble_checksum_changes_with_content():
    import tempfile as _tf

    a = assemble_golden_set.assemble(str(conftest.RAW_SAMPLE_DIR), "augment-v1-synthetic")
    with _tf.TemporaryDirectory() as d:
        (Path(d) / "extra.json").write_text(json.dumps({
            "pr_title": "extra", "repo": "org/x", "number": 9,
            "comments": [{"comment": "c", "criterion": "logic_bug", "severity": "high"}],
        }))
        # copy one existing raw file so the dir differs by exactly the extra PR
        b = assemble_golden_set.assemble(d, "augment-v1-synthetic")
    assert a["checksum"] != b["checksum"]


# --------------------------------------------------------------------------- #
def _run_standalone() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
