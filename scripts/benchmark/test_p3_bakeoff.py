"""Tests for the Phase-3 shadow bake-off harness (zero-inference).

Covers: typed-verdict parsing, exact McNemar / kappa, critic corpus
construction, manifest pinning + tamper detection, runner plan/execute
gating, critic replay scoring, and the paired report (pairing discipline).

Run (targeted, repo convention):
    uv run pytest scripts/benchmark/test_p3_bakeoff.py -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import p3_bakeoff_common as common
import p3_bakeoff_critic_build as critic_build
import p3_bakeoff_critic_score as critic_score
import p3_bakeoff_manifest as manifest_mod
import p3_bakeoff_report as report_mod
import p3_bakeoff_runner as runner_mod

RESEARCH_ROOT = Path("/mnt/raid0/llm/epyc-inference-research")
REAL_MANIFEST = (
    RESEARCH_ROOT
    / "artifacts/p3-shadow-bakeoff-20260728/manifest/p3_bakeoff_manifest.json"
)


# ---------------------------------------------------------------------------
# common: typed-verdict parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decision", common.REVIEW_DECISIONS)
def test_parse_verdict_all_enum_values(decision):
    text = json.dumps(
        {"decision": decision, "confidence": 0.7, "blocking": {"tripwire": False}}
    )
    v = common.parse_typed_verdict(text)
    assert v["parse_status"] == "ok"
    assert v["decision"] == decision
    assert v["confidence"] == 0.7
    assert v["tripwire"] is False
    assert v["decision_class"] in ("accept", "reject", "noncommittal")


def test_parse_verdict_fenced_and_prose_wrapped():
    fenced = 'Here is my verdict:\n```json\n{"decision": "approve", "confidence": 1, "blocking": {"tripwire": false}}\n```\nDone.'
    v = common.parse_typed_verdict(fenced)
    assert v["parse_status"] == "ok" and v["decision"] == "approve"
    prose = 'I think {"decision": "reject", "confidence": 0.4, "blocking": {"tripwire": true}} overall.'
    v = common.parse_typed_verdict(prose)
    assert v["parse_status"] == "ok"
    assert v["decision_class"] == "reject"
    assert v["tripwire"] is True


def test_parse_verdict_failure_modes():
    assert common.parse_typed_verdict("")["parse_status"] == "empty_response"
    assert common.parse_typed_verdict("no json here")["parse_status"] == "no_json_object"
    assert common.parse_typed_verdict('{"decision": "approve",}')["parse_status"] == "malformed_json"
    assert common.parse_typed_verdict('{"confidence": 0.5}')["parse_status"] == "missing_decision"
    v = common.parse_typed_verdict('{"decision": "LGTM"}')
    assert v["parse_status"] == "invalid_decision"
    assert v["decision_class"] is None


def test_parse_verdict_out_of_range_confidence_dropped():
    v = common.parse_typed_verdict('{"decision": "approve", "confidence": 3.0}')
    assert v["parse_status"] == "ok" and v["confidence"] is None


# ---------------------------------------------------------------------------
# common: statistics
# ---------------------------------------------------------------------------


def test_mcnemar_exact_known_values():
    # b=0,c=6 (FG-1 Laguna-vs-A3): exact two-sided p = 2*(1/2)^6 = 0.03125
    assert common.mcnemar_exact(0, 6) == pytest.approx(0.03125)
    # b=2,c=6 (FG-1 FF-vs-A3): p = 2*P(X<=2 | n=8) = 2*(37/256) ~= 0.2891
    assert common.mcnemar_exact(2, 6) == pytest.approx(0.2890625)
    assert common.mcnemar_exact(0, 0) == 1.0
    assert common.mcnemar_exact(3, 3) == pytest.approx(1.0)


def test_cohens_kappa_known_values():
    # Perfect agreement
    assert common.cohens_kappa(10, 0, 0, 10) == pytest.approx(1.0)
    # Chance-level agreement on balanced marginals -> ~0
    assert common.cohens_kappa(5, 5, 5, 5) == pytest.approx(0.0)
    assert common.cohens_kappa(0, 0, 0, 0) is None


def test_paired_mde_honest_scale():
    # n=40, psi=0.20 -> ~0.198: the spec's "cannot resolve small gaps" claim
    assert common.paired_mde(40, 0.20) == pytest.approx(0.198, abs=0.005)
    assert common.paired_mde(53, 0.25) < common.paired_mde(40, 0.25)


# ---------------------------------------------------------------------------
# critic corpus construction
# ---------------------------------------------------------------------------


def _pq_row(qid, arm, response, correct, finish="stop", **kw):
    row = {
        "id": qid, "suite": "livecodebench_hard", "arm": arm,
        "response": response, "correct": correct, "finish_reason": finish,
        "request_error": "", "truncated": finish == "length",
        "empty_response": not response, "completion_tokens": 10,
    }
    row.update(kw)
    return row


@pytest.fixture()
def critic_fixture(tmp_path):
    questions = [{"id": f"lcb_q{i}", "prompt": f"Task {i}"} for i in range(6)]
    qfile = tmp_path / "questions.json"
    qfile.write_text(json.dumps(questions))
    rows = []
    for i in range(6):
        rows.append(_pq_row(f"lcb_q{i}", "A3", f"solution A3 {i}", i % 2 == 0))
        rows.append(_pq_row(f"lcb_q{i}", "A4", f"solution A4 {i}", i % 3 == 0))
    # ineligible rows: truncated, empty, request_error, duplicate response
    rows.append(_pq_row("lcb_q0", "A3", "truncated resp", False, finish="length"))
    rows.append(_pq_row("lcb_q1", "A3", "", False))
    rows.append(_pq_row("lcb_q2", "A3", "err resp", False, request_error="boom"))
    rows.append(_pq_row("lcb_q3", "A4", "solution A3 3", True))  # dup of A3 q3
    src = tmp_path / "pq.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return qfile, src


def test_critic_build_eligibility_and_dedupe(critic_fixture):
    _, src = critic_fixture
    candidates, provenance = critic_build.mine_candidates([src])
    assert len(candidates) == 12  # 6 A3 + 6 A4, ineligible + dup excluded
    assert provenance[0]["sha256"] == common.sha256_file(src)
    assert all(c["gold_label"] in ("known_correct", "known_wrong")
               for c in candidates)


def test_critic_build_balance_and_determinism(critic_fixture):
    _, src = critic_fixture
    candidates, _ = critic_build.mine_candidates([src])
    s1 = critic_build.balance_select(candidates, per_class=4, seed=42)
    s2 = critic_build.balance_select(list(candidates), per_class=4, seed=42)
    assert [c["response_sha256"] for c in s1] == [c["response_sha256"] for c in s2]
    labels = [c["gold_label"] for c in s1]
    assert labels.count("known_correct") == 4
    assert labels.count("known_wrong") == 4


def test_critic_build_cli_runner_compatible_shape(critic_fixture, tmp_path):
    qfile, src = critic_fixture
    out = tmp_path / "critic_tasks.json"
    rc = _run_critic_build_cli(qfile, src, out, per_class=3)
    assert rc == 0
    payload = json.loads(out.read_text())
    # v7_quality_gate_runner.load_questions replays pinned["suites"][suite]
    tasks = payload["suites"][common.CRITIC_SUITE]
    assert payload["prevalence"] == {"known_correct": 3, "known_wrong": 3}
    for t in tasks:
        assert t["suite"] == common.CRITIC_SUITE
        assert t["expected"] == "__typed_verdict__"
        assert t["scoring_config"]["gold_label"] in ("known_correct", "known_wrong")
        assert "Candidate solution under review" in t["prompt"]
        assert '"decision"' in t["prompt"]
        # gold label must never leak into the model-visible prompt
        assert "known_correct" not in t["prompt"]
        assert "known_wrong" not in t["prompt"]
    assert (out.parent / (out.name + ".sha256")).exists()


def _run_critic_build_cli(qfile, src, out, per_class):
    argv = sys.argv
    sys.argv = ["p3_bakeoff_critic_build.py", "--sources", str(src),
                "--questions", str(qfile), "--per-class", str(per_class),
                "--output", str(out)]
    try:
        return critic_build.main()
    finally:
        sys.argv = argv


# ---------------------------------------------------------------------------
# manifest build/verify
# ---------------------------------------------------------------------------


@pytest.fixture()
def mini_manifest(tmp_path, critic_fixture):
    qfile, src = critic_fixture
    critic_out = tmp_path / "critic_tasks.json"
    assert _run_critic_build_cli(qfile, src, critic_out, per_class=2) == 0
    swe = [{"id": f"swe_{i}", "prompt": f"fix {i}", "expected": "__patch__",
            "suite": "swebench_oracle"} for i in range(4)]
    swe_file = tmp_path / "swe.json"
    swe_file.write_text(json.dumps(swe))
    lcb_file = tmp_path / "lcb.json"
    lcb_file.write_text(json.dumps(
        [{"id": f"lcb_{i}", "prompt": f"p{i}", "expected": "__exec__",
          "suite": "livecodebench_hard"} for i in range(3)]))
    fg1_file = tmp_path / "fg1.json"
    fg1_file.write_text(json.dumps(
        {"swe40": {"unsolved_by_all_six": ["swe_1", "swe_3"]}}))
    manifest = manifest_mod.build_manifest(
        critic_out, {}, created_utc="2026-07-28T00:00:00+00:00",
        swe_questions=swe_file, lcb_questions=lcb_file, fg1_results=fg1_file,
    )
    mpath = tmp_path / "manifest.json"
    common.write_json(mpath, manifest, sort_keys=False)
    return manifest, mpath


def test_manifest_pins_and_hard_core(mini_manifest):
    manifest, _ = mini_manifest
    swe = manifest["duties"]["coder"]["suites"]["swebench_oracle"]
    assert swe["hard_core_tag"]["ids"] == ["swe_1", "swe_3"]
    assert swe["n"] == 4
    assert len(swe["questions_file"]["sha256"]) == 64
    cocritic = manifest["duties"]["cocritic"]
    assert cocritic["n"] == 4
    assert cocritic["sampling"]["temperature"] == 0.6
    assert cocritic["sampling"]["seed"] == 42
    assert cocritic["sampling"]["enable_thinking"] is False
    assert "no lineup change" in manifest["invariants"]["not_authorized"].lower() \
        or "NO lineup change" in manifest["invariants"]["not_authorized"]


def test_manifest_hard_core_unknown_id_fails(tmp_path, mini_manifest):
    fg1_bad = tmp_path / "fg1_bad.json"
    fg1_bad.write_text(json.dumps(
        {"swe40": {"unsolved_by_all_six": ["not_in_manifest"]}}))
    with pytest.raises(ValueError, match="not in SWE manifest"):
        manifest_mod.load_hard_core(fg1_bad, {"swe_1"})


def test_manifest_verify_ok_and_tamper_detection(mini_manifest):
    manifest, _ = mini_manifest
    assert manifest_mod.verify_manifest(manifest) == []
    tampered = json.loads(json.dumps(manifest))
    tampered["duties"]["cocritic"]["tasks_file"]["sha256"] = "0" * 64
    failures = manifest_mod.verify_manifest(tampered)
    assert any("cocritic/tasks" in f and "mismatch" in f for f in failures)


# ---------------------------------------------------------------------------
# runner: plan-only default + execute gating
# ---------------------------------------------------------------------------


def test_runner_plan_mode_emits_commands_only(mini_manifest, capsys, tmp_path):
    _, mpath = mini_manifest
    rc = runner_mod.main([
        "--manifest", str(mpath), "--arm", "stock27b",
        "--out-root", str(tmp_path / "runs"), "--run-id", "t0",
    ])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "plan_only"
    suites = sorted(c["suite"] for c in plan["commands"])
    assert suites == sorted(
        ["swebench_oracle", "livecodebench_hard", common.CRITIC_SUITE])
    for cmd in plan["commands"]:
        argv = cmd["argv"]
        assert "v7_quality_gate_runner.py" in argv[1]
        assert "--no-enable-thinking" in argv
        assert argv[argv.index("--temperature") + 1] == "0.6"
        assert argv[argv.index("--seed") + 1] == "42"
        assert argv[argv.index("--port") + 1] == "18100"
    # plan mode must not create run directories (no side effects)
    assert not (tmp_path / "runs" / "t0").exists()


def test_runner_execute_refused_without_grant(mini_manifest, tmp_path):
    _, mpath = mini_manifest
    rc = runner_mod.main([
        "--manifest", str(mpath), "--arm", "stock27b", "--execute",
        "--out-root", str(tmp_path / "runs"),
    ])
    assert rc == 1


def test_runner_execute_refused_without_model_hash(mini_manifest, tmp_path):
    _, mpath = mini_manifest
    # ff27b has no sha256 in the mini manifest (no --model-hashes given)
    rc = runner_mod.main([
        "--manifest", str(mpath), "--arm", "ff27b", "--execute",
        "--i-have-operator-grant", "--out-root", str(tmp_path / "runs"),
    ])
    assert rc == 1


def test_runner_unknown_arm(mini_manifest):
    _, mpath = mini_manifest
    assert runner_mod.main(["--manifest", str(mpath), "--arm", "nope"]) == 2


def test_runner_refuses_on_pin_drift(mini_manifest, tmp_path):
    manifest, mpath = mini_manifest
    tampered = json.loads(mpath.read_text())
    tampered["duties"]["cocritic"]["tasks_file"]["sha256"] = "0" * 64
    bad = tmp_path / "bad_manifest.json"
    bad.write_text(json.dumps(tampered))
    assert runner_mod.main(["--manifest", str(bad), "--arm", "stock27b"]) == 1


# ---------------------------------------------------------------------------
# critic replay scorer
# ---------------------------------------------------------------------------


def _capture_row(task_id, response, truncated=False, schema_ok=True,
                 fp_ok=True):
    fp = common.sha256_text(response) if fp_ok else "0" * 64
    return {
        "id": task_id, "suite": common.CRITIC_SUITE,
        "capture_schema_version":
            critic_score.CAPTURE_SCHEMA_VERSION if schema_ok else "v3",
        "response": response,
        "response_fingerprint": {"chars": len(response),
                                 "utf8_bytes": len(response.encode()),
                                 "sha256": fp},
        "request_error": "", "finish_reason": "length" if truncated else "stop",
        "truncated": truncated, "completion_tokens": 42,
    }


def _task(task_id, gold):
    return {"id": task_id, "suite": common.CRITIC_SUITE,
            "expected": "__typed_verdict__",
            "scoring_config": {"gold_label": gold}}


def _verdict(decision, conf=0.9):
    return json.dumps({"decision": decision, "confidence": conf,
                       "blocking": {"tripwire": False}})


def test_critic_score_confusion_and_rates():
    tasks = [
        _task("t1", "known_correct"),  # approve  -> TP
        _task("t2", "known_correct"),  # reject   -> FR (fn)
        _task("t3", "known_wrong"),    # approve  -> FA (fp)
        _task("t4", "known_wrong"),    # request_changes -> TN
        _task("t5", "known_wrong"),    # abstain  -> noncommittal
        _task("t6", "known_correct"),  # garbage  -> parse fail
    ]
    rows = [
        _capture_row("t1", _verdict("approve")),
        _capture_row("t2", _verdict("reject")),
        _capture_row("t3", _verdict("approve")),
        _capture_row("t4", _verdict("request_changes")),
        _capture_row("t5", _verdict("abstain")),
        _capture_row("t6", "I refuse to answer in JSON"),
    ]
    result = critic_score.score_rows(tasks, rows)
    s = result["summary"]
    assert s["confusion_committed"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert s["fa_rate"] == pytest.approx(0.5)
    assert s["fr_rate"] == pytest.approx(0.5)
    assert s["fa_fr_ratio"] == pytest.approx(1.0)
    assert s["noncommittal_rate"] == pytest.approx(1 / 6)
    assert s["parse_failure_rate"] == pytest.approx(1 / 6)
    assert s["verdict_accuracy_all"] == pytest.approx(2 / 6)
    assert s["verdict_accuracy_committed"] == pytest.approx(0.5)
    assert s["prevalence_gold_correct"] == pytest.approx(0.5)
    by_id = {r["id"]: r for r in result["per_row"]}
    assert by_id["t1"]["verdict_correct"] is True
    assert by_id["t2"]["verdict_correct"] is False
    assert by_id["t5"]["decision_class"] == "noncommittal"


def test_critic_score_fail_closed_on_bad_rows():
    tasks = [_task("t1", "known_correct"), _task("t2", "known_wrong"),
             _task("t3", "known_wrong")]
    rows = [
        _capture_row("t1", _verdict("approve")),
        _capture_row("t2", _verdict("reject"), fp_ok=False),      # tampered
        # t3 missing entirely
    ]
    result = critic_score.score_rows(tasks, rows)
    assert result["missing_ids"] == ["t3"]
    assert result["quarantined"] == [
        {"id": "t2", "reason": "response_fingerprint_mismatch"}]
    assert result["summary"]["n_scored"] == 1


def test_critic_score_cli_fail_closed_exit(tmp_path):
    tasks_payload = {
        "schema_version": common.CRITIC_TASKS_SCHEMA_VERSION,
        "suites": {common.CRITIC_SUITE: [_task("t1", "known_correct"),
                                         _task("t2", "known_wrong")]},
    }
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps(tasks_payload))
    cap = tmp_path / "pq.jsonl"
    cap.write_text(json.dumps(_capture_row("t1", _verdict("approve"))) + "\n")
    out = tmp_path / "score.json"
    rc = critic_score.main(["--tasks", str(tasks_file), "--capture", str(cap),
                            "--arm", "x", "--output", str(out)])
    assert rc == 1  # t2 missing -> fail closed
    rc = critic_score.main(["--tasks", str(tasks_file), "--capture", str(cap),
                            "--arm", "x", "--output", str(out),
                            "--allow-partial"])
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["schema_version"] == common.CRITIC_SCORE_SCHEMA_VERSION
    # wrong expected hash fails closed
    rc = critic_score.main(["--tasks", str(tasks_file), "--capture", str(cap),
                            "--arm", "x", "--output", str(out),
                            "--expect-tasks-sha256", "0" * 64])
    assert rc == 1


# ---------------------------------------------------------------------------
# paired report
# ---------------------------------------------------------------------------


def test_report_pairing_discipline_fail_closed():
    a = {"q1": True, "q2": False}
    b = {"q1": True, "q3": False}
    with pytest.raises(ValueError, match="pairing violation"):
        report_mod.paired_compare(a, b, label_a="A", label_b="B")


def test_report_paired_compare_counts():
    a = {"q1": True, "q2": True, "q3": False, "q4": False}
    b = {"q1": True, "q2": False, "q3": True, "q4": False}
    c = report_mod.paired_compare(a, b, label_a="A", label_b="B")
    assert c["discordant"] == {"A_only": 1, "B_only": 1}
    assert c["mcnemar_exact_p_two_sided"] == 1.0
    assert c["ids_solved_only_by"]["A"] == ["q2"]
    sub = report_mod.paired_compare(a, b, label_a="A", label_b="B",
                                    subset=["q1", "q2"])
    assert sub["n_pairs"] == 2


def test_report_swe_report_extraction(tmp_path):
    rep = tmp_path / "swe_report.json"
    rep.write_text(json.dumps({"resolved_ids": ["i1"],
                               "unresolved_ids": ["i2"]}))
    m = report_mod.correctness_from_swe_report(rep, {"i1", "i2"})
    assert m == {"i1": True, "i2": False}
    with pytest.raises(ValueError, match="outside the pinned"):
        report_mod.correctness_from_swe_report(rep, {"i2"})
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"something": 1}))
    with pytest.raises(ValueError, match="resolved_ids"):
        report_mod.correctness_from_swe_report(bad, {"i1"})


def test_report_end_to_end_lcb(mini_manifest, tmp_path):
    _, mpath = mini_manifest
    ids = [f"lcb_{i}" for i in range(3)]
    cap_a = tmp_path / "a.jsonl"
    cap_b = tmp_path / "b.jsonl"
    rows_a = [{"id": i, "suite": "livecodebench_hard",
               "correct": i != "lcb_2", "completion_tokens": 100}
              for i in ids]
    rows_b = [{"id": i, "suite": "livecodebench_hard",
               "correct": i == "lcb_0", "completion_tokens": 60}
              for i in ids]
    cap_a.write_text("\n".join(json.dumps(r) for r in rows_a) + "\n")
    cap_b.write_text("\n".join(json.dumps(r) for r in rows_b) + "\n")
    out = tmp_path / "report.json"
    rc = report_mod.main([
        "--manifest", str(mpath), "--suite", "livecodebench_hard",
        "--label-a", "stock27b", "--label-b", "ff27b",
        "--capture-a", str(cap_a), "--capture-b", str(cap_b),
        "--output", str(out),
    ])
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["comparison"]["n_pairs"] == 3
    assert doc["comparison"]["discordant"] == {"stock27b_only": 1,
                                               "ff27b_only": 0}
    assert doc["token_economics"]["ff27b"]["median_completion_tokens"] == 60
    assert doc["grade"] == "observation"
    assert "not_authorized" in doc
    assert out.with_suffix(".md").exists()


# ---------------------------------------------------------------------------
# integration against the real pinned artifacts (skipped if absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_MANIFEST.exists(),
                    reason="real bake-off manifest not built")
def test_real_manifest_hard_core_is_fg1_14():
    manifest = json.loads(REAL_MANIFEST.read_text())
    hc = manifest["duties"]["coder"]["suites"]["swebench_oracle"]["hard_core_tag"]
    assert len(hc["ids"]) == 14
    assert "django__django-10999" in hc["ids"]
    assert "sympy__sympy-11618" in hc["ids"]
    assert manifest["duties"]["coder"]["suites"]["swebench_oracle"]["n"] == 40
    assert manifest["duties"]["coder"]["suites"]["livecodebench_hard"]["n"] == 53


@pytest.mark.skipif(not REAL_MANIFEST.exists(),
                    reason="real bake-off manifest not built")
def test_real_manifest_verifies():
    manifest = json.loads(REAL_MANIFEST.read_text())
    assert manifest_mod.verify_manifest(manifest) == []
    for arm in manifest["arms"].values():
        assert arm["sha256"], "every arm must carry a pinned model sha256"
