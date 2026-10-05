from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

# NIB2-75 (2026-09-23): tests that needed the retired E8 sealed bundles were removed per OP-19.

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "scripts/benchmark/run_e8_quality_baseline_reseed.py"
VALIDATOR = Path("/mnt/raid0/llm/epyc-root/artifacts/operator/prepare_e8_quality_baseline_reseed_20260726.sh")
LEGACY_T1_R1 = Path(
    "/mnt/raid0/llm/epyc-root/artifacts/operator/"
    ".e8_quality_baseline_candidate_v4_20260727.staging-e5812cb262dc4f4bb424f2a649defa1f"
)

spec = importlib.util.spec_from_file_location("e8_reseed", MODULE_PATH)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules["e8_reseed"] = runner
spec.loader.exec_module(runner)

class FakeQuestionResult:
    def __init__(
        self,
        qid: str,
        *,
        answer: str,
        error: str | None = None,
        concurrency: int = 3,
    ) -> None:
        self.qid = qid
        self.question_id = qid
        self.suite = "suite_a"
        self.answer = answer
        self.correct = error is None
        self.error = error
        self.partial = False
        self.degraded = False
        self.route_used = "frontdoor"
        self.eval_concurrency = concurrency


class FakeAggregate:
    def __init__(self, results: list[FakeQuestionResult], tier: int) -> None:
        self.quality = 3.0 if not any(row.error for row in results) else 0.0
        self.per_suite_quality = {"suite_a": self.quality}
        self.per_suite_counts = {"suite_a": len(results)}
        self.tier = tier


class FakeTower:
    calls = 0
    last_questions: list[dict] = []
    error_on_call: int | None = None
    wrong_vector_on_call: int | None = None
    concurrency = 3

    def __init__(self, **_kwargs: object) -> None:
        self.timeout = 1

    def _eval_batch(self, questions: list[dict], *_args: object, **_kwargs: object) -> list[FakeQuestionResult]:
        type(self).calls += 1
        type(self).last_questions = [dict(question) for question in questions]
        error = "backend failed" if type(self).calls == type(self).error_on_call else None
        rows = [
            FakeQuestionResult(
                str(question["id"]),
                answer=str(question["expected"]),
                error=error,
                concurrency=type(self).concurrency,
            )
            for question in questions
        ]
        if type(self).calls == type(self).wrong_vector_on_call:
            rows[0].qid = "unexpected-qid"
        return rows

    def _aggregate(self, results: list[FakeQuestionResult], tier: int) -> FakeAggregate:
        return FakeAggregate(results, tier)


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    state = tmp_path / "state.json"
    registry = tmp_path / "registry.yaml"
    lineup = tmp_path / "lineup.yaml"
    state.write_text(json.dumps({
        "active_instrument_eras": {"eval_quality": "E8"},
        "e8_quality_rebaseline": {"status": "hold_open"},
        "baseline_state": {"eval_quality_era": "E7-eval-instrument"},
    }))
    registry.write_text("models: []\n")
    lineup.write_text("instances: []\n")
    journal = tmp_path / "journal.jsonl"
    journal.write_text("")
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n")
    return state, registry, lineup, journal, receipt


def _args(tmp_path: Path, *extra: str):
    state, registry, lineup, journal, receipt = _paths(tmp_path)
    return runner.parse_args([
        "--execute", "--output-dir", str(tmp_path / "evidence"),
        "--state-path", str(state), "--registry-path", str(registry), "--lean-registry-path", str(registry),
        "--runtime-facts-path", str(lineup), "--stack-priors-path", str(registry), "--orchestrator-state-path", str(registry),
        "--journal-path", str(journal), "--protocol-receipt", str(receipt),
        *extra,
    ])


def test_cli_accepts_only_the_canonical_t2_size() -> None:
    args = runner.parse_args(["--protocol-proposal", "--t2-n", "500"])
    assert args.t2_n == 500

    with pytest.raises(SystemExit):
        runner.parse_args(["--protocol-proposal", "--t2-n", "50"])


def test_cli_accepts_only_the_ratified_e8_request_timeout() -> None:
    args = runner.parse_args(["--protocol-proposal", "--evaltower-timeout-s", "300"])
    assert args.evaltower_timeout_s == runner.E8_EVAL_REQUEST_TIMEOUT_S

    with pytest.raises(SystemExit):
        runner.parse_args(["--protocol-proposal", "--evaltower-timeout-s", "120"])


def _patch_clean_environment(monkeypatch, *, mutate: Path | None = None) -> None:
    FakeTower.calls = 0
    FakeTower.last_questions = []
    FakeTower.error_on_call = None
    FakeTower.wrong_vector_on_call = None
    FakeTower.concurrency = 3
    monkeypatch.setattr(runner, "EvalTower", FakeTower)
    monkeypatch.setattr(runner, "apply_context_replacement_map", lambda _args, questions, **_kwargs: questions)
    monkeypatch.setattr(
        runner,
        "measurement_source_paths",
        lambda _args: [runner.RUNNER_PATH],
    )
    monkeypatch.setattr(runner, "autopilot_processes", lambda: [])
    monkeypatch.setattr(runner, "numeric_rerun_status", lambda *_args: {"completed": 16, "required": 16})
    monkeypatch.setattr(runner, "runtime_topology", lambda *_args: [{"port": 1, "roles": ["frontdoor"]}])
    monkeypatch.setattr(
        runner,
        "receipt_payload",
        lambda *_args: {
            "schema": "epyc.operator_e8_quality_baseline_protocol.v3",
            "decision": runner.PROTOCOL_DECISION,
            "era": "E8",
            "operator_attestation": "test",
            "protocol": {"protocol_id": runner.PROTOCOL_ID},
        },
    )
    monkeypatch.setattr(runner, "protocol_contract", lambda *_args: {})
    monkeypatch.setattr(
        runner,
        "frontdoor_context_coverage",
        lambda *_args, **_kwargs: {"schema": runner.E8_CONTEXT_COVERAGE_SCHEMA, "rows": []},
    )
    monkeypatch.setattr(
        runner,
        "runtime_binding",
        lambda _args, **_kwargs: {
            "runtime_facts_sha256": "runtime", "stack_priors_sha256": "priors",
            "orchestrator_state_sha256": "state", "stack_numa_mode": "both",
            "selected_ports": list(range(24)), "server_pids": {}, "server_binaries": {},
            "runtime_artifacts": {},
            "llama_server": "/fake/llama-server",
            **({"llama_server_sha256": "binary", "llama_server_version": "10107"} if _kwargs.get("include_binary_hash") else {}),
        },
    )
    original_repetition = runner.run_repetition
    def with_sidecar(*args, **kwargs):
        observation, detail = original_repetition(*args, **kwargs)
        sidecar = kwargs["sidecar_dir"] / f"question_results.e8-t{kwargs['tier']}-r{kwargs['repetition']}.jsonl"
        runner.write_text(sidecar, '{"row_type":"batch_complete"}\n')
        detail["sidecar_sha256"] = runner.sha256_path(sidecar)
        return observation, detail
    monkeypatch.setattr(runner, "run_repetition", with_sidecar)
    health = {
        "ok": True,
        "payload_sha256": "same",
        "payload": {"status": "ok"},
        "probe_urls": {
            group: f"http://127.0.0.1/{index}"
            for index, group in enumerate(sorted(runner.EXPECTED_PROBE_GROUPS), 1)
        },
    }
    monkeypatch.setattr(runner, "api_health", lambda *_args, **_kwargs: dict(health))
    monkeypatch.setattr(
        runner,
        "question_vector",
        lambda _tower, *, tier, **_kwargs: (
            [{"id": f"t{tier}-q1", "qid": f"t{tier}-q1", "suite": "suite_a", "prompt": "p", "expected": "e", "scoring_method": "exact_match"},
             {"id": f"t{tier}-q2", "qid": f"t{tier}-q2", "suite": "suite_a", "prompt": "p2", "expected": "e2", "scoring_method": "exact_match"}],
            f"core-t{tier}",
        ),
    )
    if mutate is not None:
        original = runner.run_repetition
        def mutated(*args, **kwargs):
            result = original(*args, **kwargs)
            mutate.write_text("models: changed\n")
            return result
        monkeypatch.setattr(runner, "run_repetition", mutated)


def test_execute_seals_and_atomically_publishes_six_observation_evidence(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    report, rc = runner.execute(_args(tmp_path))

    assert rc == 0
    assert report["decision_grade"] is True


def test_candidate_collection_never_requires_or_mints_human_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_clean_environment(monkeypatch)
    args = _args(tmp_path)
    args.protocol_receipt = tmp_path / "missing-human-receipt.json"
    assert not args.protocol_receipt.exists()
    proposal = {
        "schema": "epyc.e8_quality_baseline_protocol_proposal.v3",
        "protocol": {"protocol_id": runner.PROTOCOL_ID},
        "t1_core_file_sha256": "candidate-core",
    }
    monkeypatch.setattr(runner, "protocol_proposal", lambda _args: proposal)
    monkeypatch.setattr(
        runner,
        "receipt_payload",
        lambda *_args: pytest.fail("candidate collection must not read a human receipt"),
    )

    report, rc = runner.execute(args, candidate_mode=True)

    assert rc == 2
    assert report["blockers"] == ["E8 v4 repair candidate requires --legacy-t1-r1-dir"]


def test_candidate_execute_uses_one_focused_t1r1_repair_and_never_full_reruns(
    tmp_path: Path, monkeypatch
) -> None:
    """The legacy path is a 46+3 replay plus one request, never a 50-item rerun."""
    _patch_clean_environment(monkeypatch)
    args = _args(tmp_path)
    args.legacy_t1_r1_dir = tmp_path / "sealed-legacy-t1-r1"
    proposal = {
        "schema": "epyc.e8_quality_baseline_protocol_proposal.v3",
        "protocol": {"protocol_id": runner.PROTOCOL_ID},
        "t1_core_file_sha256": "candidate-core",
    }
    monkeypatch.setattr(runner, "protocol_proposal", lambda _args: proposal)
    migration = SimpleNamespace(
        legacy_dir=args.legacy_t1_r1_dir.resolve(),
        questions=[
            {"qid": "t1-q1", "suite": "suite_a"},
            {"qid": "t1-q2", "suite": "suite_a"},
        ],
        provenance={
            "schema": runner.LEGACY_T1_R1_MIGRATION_SCHEMA,
            "runtime_window": {
                "legacy_generation_window": {"started_at_s": 1.0, "ended_at_s": 2.0},
                "watcher_exception": {},
                "sidecar_timestamp_contradiction": {},
            },
        },
    )
    proposal["legacy_t1_r1_migration_candidate"] = {
        "schema": runner.LEGACY_T1_R1_MIGRATION_SCHEMA,
        "legacy_dir": str(migration.legacy_dir),
        "provenance_sha256": runner.canonical_hash(migration.provenance),
        "watcher_exception": {},
        "sidecar_timestamp_contradiction": {},
    }
    monkeypatch.setattr(runner, "prepare_legacy_t1_r1_migration", lambda *_args, **_kwargs: migration)
    monkeypatch.setattr(
        runner,
        "replay_legacy_t1_r1_scorer_tails",
        lambda *_args, **_kwargs: ([{"qid": "t1-q1"}, {"qid": "t1-q2"}], [], {"focused_generation_pending": 1}),
    )
    focused_calls = 0

    def focused(_tower, _migration, *, sidecar_dir: Path, **_kwargs):
        nonlocal focused_calls
        focused_calls += 1
        sidecar = sidecar_dir / "question_results.e8-t1-r1-focused-legacy-timeout-repair.jsonl"
        runner.write_text(sidecar, '{"row_type":"question_result"}\n')
        return {}, {
            "actual_eval_concurrency": 1,
            "sidecar_path": str(sidecar),
            "sidecar_sha256": runner.sha256_path(sidecar),
            "runtime_window_classification": "focused_replacement_window; not_a_fresh_full_t1_repetition",
        }

    monkeypatch.setattr(runner, "run_focused_legacy_t1_r1_generation", focused)
    monkeypatch.setattr(
        runner,
        "finalize_legacy_t1_r1_migration",
        lambda *_args, **_kwargs: ([
            {"qid": "t1-q1", "suite": "suite_a", "correct": True, "route_used": "frontdoor"},
            {"qid": "t1-q2", "suite": "suite_a", "correct": True, "route_used": "frontdoor"},
        ], {"scoring_audit": {"matches": True}}),
    )

    def write_migration(_migration, _responses, _traces, _detail, *, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name in ("responses", "judge_trace_history", "legacy_sidecar_snapshot", "migration_provenance"):
            path = output_dir / f"{name}.json"
            runner.write_text_create(path, "{}\n")
            paths[name] = str(path)
        return paths

    monkeypatch.setattr(runner, "write_finalized_legacy_t1_r1_migration", write_migration)

    def migrated(_migration, _responses, finalized, focused_detail, paths, *, output_dir: Path, published_dir: Path, core_id: str, **_kwargs):
        raw_path = output_dir / "raw.T1.r1.json"
        runner.write_json_create(raw_path, {
            "q": 3.0, "ts": runner.utc_now(), "core_id": core_id, "protocol_id": runner.PROTOCOL_ID,
            "n": 2, "era": runner.E8_ERA, "per_suite_quality": {"suite_a": 3.0},
            "per_suite_counts": {"suite_a": 2},
        })
        def published(value: str | Path) -> str:
            return str(
                runner.published_path(
                    Path(value), staging_dir=output_dir, output_dir=published_dir
                )
            )
        return ({"path": published(raw_path), "sha256": runner.sha256_path(raw_path), "q": 3.0,
                 "ts": runner.load_json(raw_path)["ts"], "core_id": core_id, "protocol_id": runner.PROTOCOL_ID,
                 "n": 2, "era": runner.E8_ERA}, {
            "tier": 1, "repetition": 1, "response_path": published(paths["responses"]),
            "response_sha256": runner.sha256_path(Path(paths["responses"])),
            "actual_eval_concurrency": [1], "error_classification": {}, "n_results": 2,
            "response_vector_matches_input": True, "all_routes_frontdoor": True,
            "runtime_binding_matches_pre": True, "per_suite_counts_match_input": True,
            "sidecar_path": published(focused_detail["sidecar_path"]),
            "sidecar_sha256": focused_detail["sidecar_sha256"],
            "judge_trace_path": published(paths["judge_trace_history"]),
            "judge_trace_sha256": runner.sha256_path(Path(paths["judge_trace_history"])),
            "scoring_audit": finalized["scoring_audit"], "mixed_window_contract": True,
            "migration_paths": {key: published(value) for key, value in paths.items()},
        })

    monkeypatch.setattr(runner, "migrated_t1_r1_observation", migrated)
    ordinary_calls: list[tuple[int, int]] = []
    original = runner.run_repetition

    def ordinary(*call_args, **kwargs):
        ordinary_calls.append((kwargs["tier"], kwargs["repetition"]))
        return original(*call_args, **kwargs)

    monkeypatch.setattr(runner, "run_repetition", ordinary)

    report, rc = runner.execute(args, candidate_mode=True)

    assert rc == 0
    assert report["decision_grade"] is True
    assert focused_calls == 1
    assert ordinary_calls == [(1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    assert (1, 1) not in ordinary_calls


def test_runtime_watcher_receipt_scope_is_explicit_and_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_clean_environment(monkeypatch)
    args = _args(tmp_path)
    args.protocol_receipt = tmp_path / "missing-human-receipt.json"
    binding = runner.runtime_binding(args)

    with pytest.raises(ValueError, match="immutable prerequisite is missing"):
        runner.RuntimeWatcher(args, binding)

    watcher = runner.RuntimeWatcher(args, binding, include_receipt=False)
    watcher.sample()

    assert watcher.samples[-1]["ok"] is True
    assert str(args.protocol_receipt) not in watcher.expected_fingerprints


def test_frontdoor_context_coverage_rejects_fixed_item_over_live_context(monkeypatch, tmp_path: Path) -> None:
    args = _args(tmp_path)
    binding = {
        "runtime_topology": [{"port": 8070, "roles": ["frontdoor"]}],
        "server_cmdlines": {"8070": ["llama-server", "--port", "8070", "-c", "32768"]},
    }

    def exact_metrics(_port: int, prompt: str, _timeout: float) -> tuple[int, str, int]:
        return (4, "template", 4) if prompt == runner.E8_TEMPLATE_SENTINEL else (30_721, "template", 30_721)

    monkeypatch.setattr(runner, "_frontdoor_template_metrics", exact_metrics)
    questions = [{
        "id": "longbench_671b3fa1bb02136c067d5353",
        "qid": "longbench_671b3fa1bb02136c067d5353",
        "prompt": "summarize this long document",
    }]

    with pytest.raises(ValueError, match="longbench_671b3fa1bb02136c067d5353"):
        runner.frontdoor_context_coverage(args, questions, binding)


def test_context_coverage_uses_sealed_server_admission_when_tokenize_undercounts(
    monkeypatch, tmp_path: Path
) -> None:
    args = _args(tmp_path)
    binding = {
        "runtime_topology": [{"port": 8070, "roles": ["frontdoor"]}],
        "server_cmdlines": {"8070": ["llama-server", "--port", "8070", "-c", "32768"]},
    }
    monkeypatch.setattr(
        runner, "_frontdoor_template_metrics", lambda *_args: (100, "template", 100)
    )
    question = {
        "id": "longbench_671b3fa1bb02136c067d5353",
        "qid": "longbench_671b3fa1bb02136c067d5353",
        "prompt": "small according to tokenize",
    }

    coverage = runner.frontdoor_context_coverage(args, [question], binding, fail_closed=False)

    row = coverage["rows"][0]
    assert row["tokenizer_required_tokens"] == 2148
    assert row["sealed_server_admission_tokens"] == 62_515
    assert row["server_required_tokens"] == 64_563
    assert row["fits"] is False


def test_frontdoor_context_coverage_records_exact_template_and_output_budget(monkeypatch, tmp_path: Path) -> None:
    args = _args(tmp_path)
    binding = {
        "runtime_topology": [{"port": 8070, "roles": ["frontdoor"]}],
        "server_cmdlines": {"8070": ["llama-server", "--port", "8070", "-c", "32768"]},
    }
    monkeypatch.setattr(
        runner, "_frontdoor_template_metrics", lambda *_args: (100, "template", 100)
    )

    coverage = runner.frontdoor_context_coverage(
        args, [{"id": "q", "qid": "q", "prompt": "explain"}], binding
    )

    assert coverage["frontdoor"]["context_length"] == 32768
    assert coverage["rows"] == [{
        "qid": "q",
        "prompt_tokens": 100,
        "rendered_utf8_bytes": 100,
        "max_tokens": 2048,
        "tokenizer_required_tokens": 2148,
        "sealed_server_admission_tokens": None,
        "server_required_tokens": 2148,
        "required_tokens": 2148,
        "context_length": 32768,
        "fits": True,
        "per_frontdoor": [{
            "port": 8070,
            "prompt_tokens": 100,
            "rendered_prompt_sha256": "template",
            "rendered_utf8_bytes": 100,
            "tokenizer_required_tokens": 2148,
            "server_required_tokens": 2148,
            "required_tokens": 2148,
            "context_length": 32768,
            "fits": True,
        }],
    }]


def test_frontdoor_context_coverage_checks_real_prompt_on_every_frontdoor(
    monkeypatch, tmp_path: Path
) -> None:
    """A common sentinel template does not prove a real prompt fits every route."""
    args = _args(tmp_path)
    binding = {
        "runtime_topology": [
            {"port": 8070, "roles": ["frontdoor"]},
            {"port": 8080, "roles": ["frontdoor"]},
        ],
        "server_cmdlines": {
            "8070": ["llama-server", "--port", "8070", "-c", "32768"],
            "8080": ["llama-server", "--port", "8080", "-c", "32768"],
        },
    }

    def metrics(port: int, prompt: str, _timeout: float) -> tuple[int, str, int]:
        if prompt == runner.E8_TEMPLATE_SENTINEL:
            return 4, "same-sentinel-template", 4
        return (100 if port == 8070 else 31_000), "route-template", 100

    monkeypatch.setattr(runner, "_frontdoor_template_metrics", metrics)
    coverage = runner.frontdoor_context_coverage(
        args, [{"id": "q", "qid": "q", "prompt": "real prompt"}], binding, fail_closed=False
    )

    row = coverage["rows"][0]
    assert row["fits"] is False
    assert row["required_tokens"] == 33_048
    assert [item["fits"] for item in row["per_frontdoor"]] == [True, False]


def test_prepare_blocks_before_inference_when_context_coverage_fails(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    monkeypatch.setattr(
        runner,
        "frontdoor_context_coverage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("E8 fixed vector exceeds live frontdoor context after exact template and output cap: longbench")
        ),
    )

    report = runner.prepare_report(_args(tmp_path))

    assert report["decision_grade"] is False
    assert any("exceeds live frontdoor context" in blocker for blocker in report["blockers"])
    assert FakeTower.calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("force_mode", "repl"), ("allow_delegation", True)],
)
def test_run_repetition_rejects_conflicting_direct_core_source_fields(
    tmp_path: Path, monkeypatch, field: str, value: object
) -> None:
    _patch_clean_environment(monkeypatch)
    args = _args(tmp_path)
    question = {
        "id": "q1",
        "qid": "q1",
        "suite": "suite_a",
        "prompt": "p",
        "expected": "e",
        "scoring_method": "exact_match",
        field: value,
    }

    with pytest.raises(ValueError, match=f"source {field}"):
        runner.run_repetition(
            FakeTower(),
            tier=1,
            repetition=1,
            questions=[question],
            core_id="core",
            output_dir=tmp_path,
            expected_binding={},
            args=args,
            sidecar_dir=tmp_path / "sidecars",
            published_dir=tmp_path,
        )


def test_execute_refuses_active_autopilot_before_any_evaluation(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    monkeypatch.setattr(runner, "autopilot_processes", lambda: ["123 autopilot.py start"])
    report, rc = runner.execute(_args(tmp_path))
    assert rc == 75
    assert report["decision_grade"] is False
    assert "AutoPilot is active" in report["blockers"][0]
    assert FakeTower.calls == 0


def test_partial_or_error_observation_never_decision_grades(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    FakeTower.error_on_call = 4
    report, rc = runner.execute(_args(tmp_path))
    assert rc == 2
    assert report["decision_grade"] is False
    assert report["observations"][2][0]["error_classification"] == {"request_or_scoring_error": 2}


def test_state_or_lineup_mutation_is_detected_after_execution(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    _patch_clean_environment(monkeypatch, mutate=args.registry_path)
    report, rc = runner.execute(args)
    assert rc == 2
    assert report["postconditions"]["checks"]["no_state_registry_lineup_mutation"] is False


def test_wrong_response_vector_never_decision_grades(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    FakeTower.wrong_vector_on_call = 5
    report, rc = runner.execute(_args(tmp_path))
    assert rc == 2
    assert report["decision_grade"] is False
    assert report["observations"][2][1]["response_vector_matches_input"] is False


def test_prepare_rejects_wrong_era_or_closed_hold(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    args = _args(tmp_path)
    args.state_path.write_text(json.dumps({
        "active_instrument_eras": {"eval_quality": "E7-eval-instrument"},
        "e8_quality_rebaseline": {"status": "closed"},
        "baseline_state": {"eval_quality_era": "E7-eval-instrument"},
    }))
    report = runner.prepare_report(args)
    assert report["decision_grade"] is False
    assert any("not E8" in blocker for blocker in report["blockers"])
    assert any("not open" in blocker for blocker in report["blockers"])


def test_monitor_persistence_failure_is_sticky_and_fails_closed(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    args = _args(tmp_path)
    bad_path = tmp_path / "monitor-directory"
    bad_path.mkdir()
    watcher = runner.RuntimeWatcher(args, runner.runtime_binding(args), bad_path)

    watcher.sample()

    assert watcher.fatal_error is not None
    assert watcher.samples[-1]["ok"] is False


def test_runtime_watcher_schedules_from_sample_start_not_completion(monkeypatch) -> None:
    clock = [0.0]
    waits: list[float] = []
    starts: list[float] = []

    class Stop:
        stopped = False

        def is_set(self) -> bool:
            return self.stopped

        def wait(self, delay: float) -> bool:
            waits.append(delay)
            clock[0] += delay
            return self.stopped

    stop = Stop()
    watcher = object.__new__(runner.RuntimeWatcher)
    watcher._stop = stop
    watcher._sample_lock = runner.threading.Lock()
    watcher._last_sample_started_monotonic = 0.0

    def slow_sample() -> None:
        watcher._last_sample_started_monotonic = clock[0]
        starts.append(clock[0])
        clock[0] += 2.75
        if len(starts) == 3:
            stop.stopped = True

    watcher.sample = slow_sample
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])

    watcher._watch()

    assert starts == [5.0, 10.0, 15.0]
    assert waits == [5.0, 2.25, 2.25]


def test_runtime_watcher_serializes_explicit_and_background_samples(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_clean_environment(monkeypatch)
    args = _args(tmp_path)
    clean_health = runner.api_health
    counter_lock = runner.threading.Lock()
    active = 0
    max_active = 0

    def delayed_health(*args, **kwargs):
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            runner.time.sleep(0.02)
            return clean_health(*args, **kwargs)
        finally:
            with counter_lock:
                active -= 1

    monkeypatch.setattr(runner, "api_health", delayed_health)
    watcher = runner.RuntimeWatcher(args, runner.runtime_binding(args))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _index: watcher.sample(), range(4)))

    assert len(watcher.samples) == 4
    assert max_active == 1


def _health_payload(*, failed_reason: str | None = None) -> dict:
    probes = {
        group: {"ok": True, "url": f"http://127.0.0.1/{index}", "status_code": 200}
        for index, group in enumerate(sorted(runner.EXPECTED_PROBE_GROUPS), 1)
    }
    if failed_reason is not None:
        probes["architect_general"] = {
            "ok": False,
            "url": "http://127.0.0.1/architect",
            "status_code": None,
            "failure_reason": failed_reason,
        }
    return {
        "status": "ok" if failed_reason is None else "degraded",
        "models_loaded": 6,
        "backend_probes": probes,
    }


def test_api_health_classifies_only_exact_backend_read_timeout_saturation(monkeypatch) -> None:
    class Response:
        status_code = 200

        def json(self):
            return _health_payload(failed_reason="read_timeout")

    monkeypatch.setattr(runner.httpx, "get", lambda *_args, **_kwargs: Response())

    health = runner.api_health("http://127.0.0.1:8000", 1.0)

    assert health["ok"] is False
    assert health["failure_class"] == "backend_probe_read_timeout"
    assert health["probe_failures"] == [{
        "group": "architect_general",
        "failure_reason": "read_timeout",
        "status_code": None,
        "url": "http://127.0.0.1/architect",
    }]


def test_runtime_watcher_accepts_readiness_saturation_only_inside_active_load(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_clean_environment(monkeypatch)
    args = _args(tmp_path)
    health = {
        "ok": False,
        "failure_class": "backend_probe_read_timeout",
        "probe_urls": {
            group: f"http://127.0.0.1/{index}"
            for index, group in enumerate(sorted(runner.EXPECTED_PROBE_GROUPS), 1)
        },
        "probe_failures": [{"group": "architect_general", "failure_reason": "read_timeout"}],
    }
    monkeypatch.setattr(runner, "api_health", lambda *_args, **_kwargs: dict(health))
    watcher = runner.RuntimeWatcher(
        args,
        runner.runtime_binding(args),
        tmp_path / "watch.jsonl",
        expected_probe_urls=health["probe_urls"],
    )

    watcher.sample()
    assert watcher.samples[-1]["ok"] is False
    assert watcher.samples[-1]["api_saturation_during_active_load"] is False

    with watcher.active_load(tier=1, repetition=2):
        watcher.sample()
    accepted = watcher.samples[-1]
    assert accepted["ok"] is True
    assert accepted["api_saturation_during_active_load"] is True
    assert accepted["active_load"] == {"tier": 1, "repetition": 2}
    assert accepted["api_probe_urls_match_preflight"] is True


def test_runtime_watcher_rejects_non_timeout_backend_failure_during_active_load(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_clean_environment(monkeypatch)
    args = _args(tmp_path)
    monkeypatch.setattr(
        runner,
        "api_health",
        lambda *_args, **_kwargs: {
            "ok": False,
            "failure_class": "readiness_contract_failed",
            "probe_urls": {
                group: f"http://127.0.0.1/{index}"
                for index, group in enumerate(sorted(runner.EXPECTED_PROBE_GROUPS), 1)
            },
            "probe_failures": [{"group": "architect_general", "failure_reason": "connect_error"}],
        },
    )
    watcher = runner.RuntimeWatcher(args, runner.runtime_binding(args), tmp_path / "watch.jsonl")

    with watcher.active_load(tier=1, repetition=1):
        watcher.sample()

    assert watcher.samples[-1]["ok"] is False
    assert watcher.samples[-1]["api_saturation_during_active_load"] is False


def test_runtime_watcher_rejects_read_timeout_with_changed_probe_url_during_active_load(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_clean_environment(monkeypatch)
    args = _args(tmp_path)
    expected_urls = {
        group: f"http://127.0.0.1/{index}"
        for index, group in enumerate(sorted(runner.EXPECTED_PROBE_GROUPS), 1)
    }
    observed_urls = {**expected_urls, "architect_general": "http://127.0.0.1/stale-architect"}
    monkeypatch.setattr(
        runner,
        "api_health",
        lambda *_args, **_kwargs: {
            "ok": False,
            "failure_class": "backend_probe_read_timeout",
            "probe_urls": observed_urls,
            "probe_failures": [{"group": "architect_general", "failure_reason": "read_timeout"}],
        },
    )
    watcher = runner.RuntimeWatcher(
        args,
        runner.runtime_binding(args),
        tmp_path / "watch.jsonl",
        expected_probe_urls=expected_urls,
    )

    with watcher.active_load(tier=1, repetition=1):
        watcher.sample()

    assert watcher.samples[-1]["ok"] is False
    assert watcher.samples[-1]["api_saturation_during_active_load"] is False
    assert watcher.samples[-1]["api_probe_urls_match_preflight"] is False


def test_runtime_watcher_rejects_read_timeout_without_a_preflight_probe_map(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_clean_environment(monkeypatch)
    args = _args(tmp_path)
    monkeypatch.setattr(
        runner,
        "api_health",
        lambda *_args, **_kwargs: {
            "ok": False,
            "failure_class": "backend_probe_read_timeout",
            "probe_failures": [{"group": "architect_general", "failure_reason": "read_timeout"}],
        },
    )
    watcher = runner.RuntimeWatcher(args, runner.runtime_binding(args), tmp_path / "watch.jsonl")

    with watcher.active_load(tier=1, repetition=1):
        watcher.sample()

    assert watcher.samples[-1]["ok"] is False
    assert watcher.samples[-1]["api_saturation_during_active_load"] is False


def test_delayed_monitor_samples_prevent_atomic_publish(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)

    class DelayedWatcher:
        def __init__(self, _args, _binding, artifact_path, **_kwargs):
            self._thread = type("Thread", (), {"is_alive": lambda self: False})()
            self.samples = []
            self.fatal_error = None
            self.artifact_path = artifact_path

        def start(self):
            runner.write_text(self.artifact_path, "{}\n")
            self.samples = [
                {"started_at": "2026-07-26T00:00:00Z", "finished_at": "2026-07-26T00:00:00Z", "ok": True},
                {"started_at": "2026-07-26T00:00:10Z", "finished_at": "2026-07-26T00:00:10Z", "ok": True},
            ]

        def active_load(self, **_kwargs):
            return nullcontext()

        def stop(self):
            return self.samples

    monkeypatch.setattr(runner, "RuntimeWatcher", DelayedWatcher)
    report, rc = runner.execute(_args(tmp_path))

    assert rc == 2
    assert report["postconditions"]["checks"]["continuous_clean_monitor"] is False
    assert not Path(report["evidence_manifest"]).exists()


def test_repetition_artifacts_are_independent_and_vectors_are_pinned(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    report, rc = runner.execute(_args(tmp_path))
    assert rc == 0
    manifest = json.loads(Path(report["evidence_manifest"]).read_text())
    raw_paths = []
    for source in manifest["source_records"]:
        summary = json.loads(Path(source["path"]).read_text())
        raw_paths.extend(Path(item["path"]) for item in summary["observations"])
    assert len({path.name for path in raw_paths}) == 6
    assert len({runner.sha256_path(path) for path in raw_paths}) == 6
    for path in report["question_vectors"].values():
        vector = json.loads(Path(path).read_text())
        assert vector["n"] == 2


def test_tampered_sealed_bundle_is_no_longer_hash_consistent(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    report, rc = runner.execute(_args(tmp_path))
    assert rc == 0
    manifest_path = Path(report["evidence_manifest"])
    manifest = json.loads(manifest_path.read_text())
    source = manifest["source_records"][0]
    summary_path = Path(source["path"])
    summary = json.loads(summary_path.read_text())
    raw_path = Path(summary["observations"][0]["path"])
    raw = json.loads(raw_path.read_text())
    raw["n"] = 99
    raw_path.write_text(json.dumps(raw, sort_keys=True))
    seal = json.loads((manifest_path.parent / "run_seal.json").read_text())
    assert runner.sha256_path(raw_path) != seal["bundle_sha256"][str(raw_path)]


def test_terminal_stack_or_health_drift_never_publishes_acceptable_manifest(tmp_path: Path, monkeypatch) -> None:
    _patch_clean_environment(monkeypatch)
    calls = 0

    def drifting_health(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "ok": calls < 3,
            "payload_sha256": "stable" if calls < 3 else "changed",
            "payload": {},
            "probe_urls": {
                group: f"http://127.0.0.1/{index}"
                for index, group in enumerate(sorted(runner.EXPECTED_PROBE_GROUPS), 1)
            },
        }
    monkeypatch.setattr(runner, "api_health", drifting_health)
    report, rc = runner.execute(_args(tmp_path))
    assert rc == 2
    assert report["decision_grade"] is False
    assert not Path(report["evidence_manifest"]).exists()


def test_ss_listener_identity_rejects_swapped_port_pid_ownership(monkeypatch) -> None:
    output = (
        'LISTEN 0 4096 127.0.0.1:8070 0.0.0.0:* users:(("llama-server",pid=222,fd=3))\n'
        'LISTEN 0 4096 127.0.0.1:8072 0.0.0.0:* users:(("llama-server",pid=111,fd=3))\n'
    )
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/ss")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )

    assert runner._missing_listener_identities({"8070": 111, "8072": 222}) == [
        "8070/pid=111",
        "8072/pid=222",
    ]


def test_ss_listener_identity_rejects_port_prefix_collision(monkeypatch) -> None:
    output = 'LISTEN 0 4096 127.0.0.1:18070 0.0.0.0:* users:(("llama-server",pid=111,fd=3))\n'
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/ss")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )

    assert runner._missing_listener_identities({"8070": 111}) == ["8070/pid=111"]


def test_ss_listener_identity_rejects_expected_pid_with_reuseport_co_listener(
    monkeypatch,
) -> None:
    output = (
        'LISTEN 0 4096 127.0.0.1:8070 0.0.0.0:* users:(("llama-server",pid=111,fd=3))\n'
        'LISTEN 0 4096 127.0.0.1:8070 0.0.0.0:* users:(("llama-server",pid=222,fd=4))\n'
    )
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/ss")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )

    assert runner._missing_listener_identities({"8070": 111}) == ["8070/pid=111"]


@pytest.mark.parametrize(
    "listener_inodes,pid_inodes",
    [
        ({"100", "200"}, {"100"}),
        ({"100"}, set()),
    ],
)
def test_proc_listener_identity_rejects_extra_or_unowned_inode(
    monkeypatch, listener_inodes, pid_inodes
) -> None:
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        runner,
        "_proc_tcp_listener_inodes",
        lambda: {8070: listener_inodes},
    )
    monkeypatch.setattr(
        runner,
        "_proc_pid_socket_inodes",
        lambda _pid: pid_inodes,
    )

    assert runner._missing_listener_identities({"8070": 111}) == ["8070/pid=111"]


def test_proc_listener_identity_rejects_recorded_pid_co_owner(monkeypatch) -> None:
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        runner,
        "_proc_tcp_listener_inodes",
        lambda: {8070: {"100"}, 8072: {"200"}},
    )
    monkeypatch.setattr(
        runner,
        "_proc_pid_socket_inodes",
        lambda pid: {111: {"100"}, 222: {"100", "200"}}[pid],
    )

    assert runner._missing_listener_identities({"8070": 111, "8072": 222}) == [
        "8070/pid=111"
    ]


def test_runtime_watcher_completes_newline_only_short_write_before_fsync(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)
    _patch_clean_environment(monkeypatch)
    artifact = tmp_path / "watch.jsonl"
    watcher = runner.RuntimeWatcher(args, runner.runtime_binding(args), artifact)
    original_write = runner.os.write
    write_payloads: list[bytes] = []
    fsync_calls: list[int] = []

    def short_write(fd: int, data) -> int:
        payload = bytes(data)
        write_payloads.append(payload)
        chunk = payload[:-1] if len(write_payloads) == 1 else payload
        return original_write(fd, chunk)

    monkeypatch.setattr(runner.os, "write", short_write)
    monkeypatch.setattr(runner.os, "fsync", lambda fd: fsync_calls.append(fd))

    watcher.sample()

    assert watcher.fatal_error is None
    assert write_payloads[-1] == b"\n"
    assert len(fsync_calls) == 1
    assert json.loads(artifact.read_text())["ok"] is True


def test_runtime_watcher_zero_write_is_fatal_and_never_fsyncs(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)
    _patch_clean_environment(monkeypatch)
    artifact = tmp_path / "watch.jsonl"
    watcher = runner.RuntimeWatcher(args, runner.runtime_binding(args), artifact)
    fsync_calls: list[int] = []
    monkeypatch.setattr(runner.os, "write", lambda _fd, _data: 0)
    monkeypatch.setattr(runner.os, "fsync", lambda fd: fsync_calls.append(fd))

    watcher.sample()

    assert "invalid progress: 0" in str(watcher.fatal_error)
    assert watcher.samples[-1]["ok"] is False
    assert fsync_calls == []


def test_runtime_binding_pins_full_cmdline_model_flags_and_state_path(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)
    ports = list(range(8070, 8094))
    expected_binary = "/fake/llama-server"
    args.runtime_facts_path = tmp_path / "runtime-facts.json"
    args.orchestrator_state_path = tmp_path / "orchestrator-state.json"
    args.stack_priors_path = tmp_path / "stack-priors.yaml"
    args.registry_path = tmp_path / "registry-full.yaml"
    args.lean_registry_path = tmp_path / "registry-lean.yaml"
    args.runtime_facts_path.write_text(
        json.dumps(
            {
                "schema": "epyc.orchestrator.runtime_facts",
                "runtime_stack": {
                    "stack_numa_mode": "both",
                    "selected_ports": ports,
                    "selected_servers": [
                        {"port": port, "roles": ["test_role"]} for port in ports
                    ],
                    "paths": {"llama_server": expected_binary},
                },
            }
        )
    )
    args.orchestrator_state_path.write_text(
        json.dumps(
            {
                f"server_{port}": {
                    "pid": 100000 + port,
                    "port": port,
                    "model_path": f"/models/model-{port}.gguf",
                }
                for port in ports
            }
        )
    )
    for path in (args.stack_priors_path, args.registry_path, args.lean_registry_path):
        path.write_text("pinned: true\n")

    cmdlines = {
        100000 + port: [
            expected_binary,
            "-m",
            f"/models/model-{port}.gguf",
            "--mmproj",
            f"/models/mmproj-{port}.gguf",
            "--port",
            str(port),
            "--ctx-size",
            "4096",
        ]
        for port in ports
    }
    monkeypatch.setattr(runner.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(runner.os, "readlink", lambda _path: expected_binary)
    monkeypatch.setattr(runner, "process_cmdline", lambda pid: list(cmdlines[pid]))
    monkeypatch.setattr(runner, "_missing_listener_identities", lambda _pids: [])
    # The frozen-tree identity check reads the live /mnt/raid0/llm/llama.cpp
    # checkout, which has since versioned past v8 (production is v9). This
    # test is about cmdline/model-flag pinning, so stub the host provenance
    # with the exact v8 identity the runner demands instead of editing the
    # frozen-kernel check itself (NIB2-69 triage 2026-09-15).
    monkeypatch.setattr(
        runner,
        "frozen_llama_source_provenance",
        lambda: {
            "path": str(runner.FROZEN_V8_LLAMA_TREE),
            "branch": runner.FROZEN_V8_LLAMA_BRANCH,
            "head": runner.FROZEN_V8_LLAMA_HEAD,
        },
    )
    monkeypatch.setattr(
        runner,
        "runtime_artifact_identities",
        lambda paths, *, include_sha256: {
            path: {
                "path": path,
                "st_dev": 1,
                "st_ino": index,
                "st_size": 10,
                "st_mtime_ns": 20,
                **({"sha256": str(index).zfill(64)} if include_sha256 else {}),
            }
            for index, path in enumerate(dict.fromkeys(paths), 1)
        },
    )

    before = runner.runtime_binding(args)

    assert before["server_state_model_paths"]["8070"] == "/models/model-8070.gguf"
    assert before["server_model_flags"]["8070"]["mmproj"] == ["/models/mmproj-8070.gguf"]
    assert before["server_cmdlines"]["8070"][-2:] == ["--ctx-size", "4096"]

    cmdlines[108070][-1] = "8192"
    after = runner.runtime_binding(args)
    assert after != before
    assert after["server_cmdline_sha256"]["8070"] != before["server_cmdline_sha256"]["8070"]


def test_watcher_failure_aborts_before_next_repetition(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    _patch_clean_environment(monkeypatch)

    class FailingWatcher:
        latest = None

        def __init__(self, _args, _binding, artifact_path, **_kwargs):
            type(self).latest = self
            self._thread = SimpleNamespace(is_alive=lambda: False)
            self.samples = [{"ok": True}]
            self.fatal_error = None
            self.artifact_path = artifact_path

        def start(self):
            runner.write_text(self.artifact_path, '{"ok":true}\n')

        def active_load(self, **_kwargs):
            return nullcontext()

        def stop(self):
            return self.samples

    monkeypatch.setattr(runner, "RuntimeWatcher", FailingWatcher)
    original = runner.run_repetition

    def fail_after_first(*call_args, **call_kwargs):
        result = original(*call_args, **call_kwargs)
        assert FailingWatcher.latest is not None
        FailingWatcher.latest.fatal_error = "runtime monitor persistence failed"
        return result

    monkeypatch.setattr(runner, "run_repetition", fail_after_first)

    with pytest.raises(RuntimeError, match="runtime monitor persistence failed"):
        runner.execute(args)

    assert FakeTower.calls == 1
    assert not args.output_dir.exists()
    staging = next(tmp_path.glob(f".{args.output_dir.name}.staging-*"))
    seal = json.loads((staging / "run_seal.json").read_text())
    assert seal["status"] == "terminal_aborted_no_admission"
    assert seal["writer"] == "run_e8_quality_baseline_reseed"


def test_receipt_requires_canonical_path_and_current_runner_hash(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)
    canonical_receipt = tmp_path / "canonical-receipt.json"
    monkeypatch.setattr(runner, "PROTOCOL_RECEIPT", canonical_receipt)
    # The predecessor-evidence hash gate runs before the runner-hash gate and
    # reads historical E8 operator artifacts that are no longer on the host.
    # Bind it to hermetic stand-ins with matching hashes so this test reaches
    # the runner-hash gate it exists to exercise (NIB2-69 triage 2026-09-15).
    supersedes = {}
    for name in runner.REPAIR_SUPERSEDES:
        predecessor = tmp_path / "predecessors" / f"{name}.json"
        predecessor.parent.mkdir(exist_ok=True)
        predecessor.write_text(json.dumps({"predecessor": name}) + "\n")
        supersedes[name] = {
            "path": str(predecessor.resolve()),
            "sha256": runner.sha256_path(predecessor),
        }
    monkeypatch.setattr(runner, "REPAIR_SUPERSEDES", supersedes)

    with pytest.raises(ValueError, match="canonical path"):
        runner.receipt_payload(args)

    args.protocol_receipt = canonical_receipt
    canonical_receipt.write_text(
        json.dumps(
            {
                "schema": "epyc.operator_e8_quality_baseline_protocol.v3",
                "decision": runner.PROTOCOL_DECISION,
                "era": "E8",
                "ratified_at": "2026-07-26T00:00:00+00:00",
                "operator_attestation": "test",
                "t2_decision": {},
                "protocol": {"protocol_id": runner.PROTOCOL_ID},
                "t1_core_file_sha256": "0" * 64,
                "expected_probe_groups": sorted(runner.EXPECTED_PROBE_GROUPS),
                "acceptance": {
                    "all_three_repetitions_clean": True,
                    "no_monitor_gap_seconds": 7,
                    "api_groups_exact": True,
                    "all_routes_frontdoor": True,
                    "sealed_atomic_publish": True,
                },
                "sha256": {"runner": "0" * 64},
                "repository_heads": {
                    "epyc_root": "a" * 40,
                    "epyc_orchestrator": "b" * 40,
                    "epyc_inference_research": "c" * 40,
                },
                "supersedes": runner.REPAIR_SUPERSEDES,
            }
        )
    )

    with pytest.raises(ValueError, match="runner hash"):
        runner.receipt_payload(args)
    assert runner.RUNNER_PATH in runner.immutable_paths(args)


def test_receipt_rejects_wrong_predecessor_evidence_before_runner_hash(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)
    canonical_receipt = tmp_path / "canonical-receipt.json"
    monkeypatch.setattr(runner, "PROTOCOL_RECEIPT", canonical_receipt)
    args.protocol_receipt = canonical_receipt
    receipt = {
        "schema": "epyc.operator_e8_quality_baseline_protocol.v3",
        "decision": runner.PROTOCOL_DECISION,
        "era": "E8",
        "ratified_at": "2026-07-26T00:00:00+00:00",
        "operator_attestation": "test",
        "t2_decision": {},
        "protocol": {"protocol_id": runner.PROTOCOL_ID},
        "t1_core_file_sha256": "0" * 64,
        "expected_probe_groups": sorted(runner.EXPECTED_PROBE_GROUPS),
        "acceptance": {},
        "sha256": {"runner": "0" * 64},
        "repository_heads": {},
        "supersedes": {**runner.REPAIR_SUPERSEDES, "historical_receipt": {"path": "/wrong", "sha256": "0" * 64}},
    }
    canonical_receipt.write_text(json.dumps(receipt))

    with pytest.raises(ValueError, match="predecessor evidence differs"):
        runner.receipt_payload(args)


def test_protocol_contract_rejects_request_timeout_mismatch(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    core_path = tmp_path / "core.jsonl"
    core_path.write_text("{}\n")
    vectors = {
        tier: {
            "core_id": f"core-{tier}",
            "n": args.t1_n if tier == 1 else args.t2_n,
            "dataset_sha256": f"dataset-{tier}",
        }
        for tier in (1, 2)
    }
    scoring_vectors = {tier: {"tier": tier} for tier in (1, 2)}
    binding = {"binding": "live"}
    topology = [{"port": 8070, "roles": ["frontdoor"]}]

    class CoreTower:
        def __init__(self, **_kwargs):
            pass

        def _load_designed_core(self, _core_id):
            return [], {}, core_path

    monkeypatch.setattr(runner, "EvalTower", CoreTower)
    monkeypatch.setattr(runner, "sha256_path", lambda _path: "hash")
    monkeypatch.setattr(runner, "runtime_binding", lambda *_args, **_kwargs: binding)
    monkeypatch.setattr(runner, "runtime_topology", lambda *_args, **_kwargs: topology)
    monkeypatch.setattr(runner, "frozen_llama_source_provenance", lambda: {"llama": "frozen"})
    monkeypatch.setattr(runner, "measurement_source_fingerprints", lambda _args: {"runner": "hash"})
    monkeypatch.setattr(
        runner,
        "context_replacement_map_identity",
        lambda _args: {"path": "map", "sha256": "map-hash", "schema": "map-v2", "replacements": []},
    )
    protocol = {
        "protocol_id": runner.PROTOCOL_ID,
        "seed": args.seed,
        "repetitions": runner.REPETITIONS,
        "generation_concurrency": runner.CONCURRENCY,
        "scoring_concurrency": runner.SCORING_CONCURRENCY,
        "request_timeout_s": runner.E8_EVAL_REQUEST_TIMEOUT_S - 1,
        "frontdoor_request_contract": runner.FRONTDOOR_REQUEST_CONTRACT,
        "watcher_contract": runner.WATCHER_CONTRACT,
        "context_coverage_contract": runner.CONTEXT_COVERAGE_CONTRACT,
        "baseline_mode": "direct_core_only_v1",
        "route_policy": "frontdoor_only",
        "selected_ports": [],
        "runtime_topology": topology,
        "runtime_facts_sha256": "hash",
        "runtime_binding": binding,
        "llama_source_provenance": {"llama": "frozen"},
        "measurement_source_sha256": {"runner": "hash"},
        "context_replacement_map": {"path": "map", "sha256": "map-hash", "schema": "map-v2"},
        "judge_defaults": {
            "orchestrator_api_url": args.api_url.rstrip("/"),
            "role": runner.JUDGE_DEFAULT_ROLE,
        },
        "expected_probe_groups": sorted(runner.EXPECTED_PROBE_GROUPS),
        "tiers": {
            str(tier): {
                "core_id": vectors[tier]["core_id"],
                "n": vectors[tier]["n"],
                "dataset_sha256": vectors[tier]["dataset_sha256"],
                "scoring_vector_sha256": runner.canonical_hash(scoring_vectors[tier]),
                "vector_sha256": runner.vector_sha256(vectors[tier]),
            }
            for tier in (1, 2)
        },
    }
    receipt = {
        "protocol": protocol,
        "t2_decision": {"n": args.t2_n, "recommended_default": 500, "alternatives": [500]},
        "t1_core_file_sha256": "hash",
    }

    with pytest.raises(ValueError, match="request_timeout_s"):
        runner.protocol_contract(args, receipt, vectors, scoring_vectors)


def test_fixed_t2_source_vector_fails_closed_on_zero_group_extract_patterns() -> None:
    t1_n, t2_n = 50, 500
    tower = runner.EvalTower(url="http://127.0.0.1:8000", timeout=1)
    t1_questions, _t1_core_id = runner.question_vector(
        tower,
        tier=1,
        t1_core_id="core_v2",
        n=t1_n,
        seed=runner.EVAL_SPEC_SEED,
    )
    questions, _core_id = runner.question_vector(
        tower,
        tier=2,
        t1_core_id="core_v2",
        n=t2_n,
        seed=runner.EVAL_SPEC_SEED,
    )
    # The tier-2 vector is a SEEDED SAMPLE of the scoreable question pool, so any
    # change to pool scoreability reshuffles which questions are drawn (2f366e6b
    # recovered 1,439 previously-unscoreable rows and moved this draw).  Assert the
    # properties the scorer contract actually depends on — derived from the runner's
    # own constant — instead of a census of one historical draw.
    assert len(t1_questions) == t1_n
    assert len(questions) == t2_n
    t1_methods = {question["scoring_method"] for question in t1_questions}
    t2_methods = {question["scoring_method"] for question in questions}
    assert t1_methods <= runner.INDEPENDENTLY_REPRODUCIBLE_SCORERS
    assert t2_methods <= runner.INDEPENDENTLY_REPRODUCIBLE_SCORERS
    # Both vectors really are the mixed real-data vectors, judge rows included —
    # what the removed llm_judge censuses were there to witness.
    assert "llm_judge" in t1_methods
    assert "llm_judge" in t2_methods
    assert len(t2_methods) > 1

    # Corrupt the FIRST TWO exact_match rows in vector order, so the qid the
    # validator names on first failure is deterministic without hardcoding a pool
    # question id.  validate_source_vector_scorer_config only inspects exact_match
    # rows carrying an extract_pattern, so these are the rows under test.
    targets = [
        runner._question_qid(question)
        for question in questions
        if question["scoring_method"] == "exact_match"
    ][:2]
    assert len(targets) == 2
    invalid_questions = []
    for question in questions:
        scoring_config = dict(question.get("scoring_config") or {})
        if runner._question_qid(question) in targets:
            scoring_config["extract_pattern"] = r"\d+"
        invalid_questions.append({**question, "scoring_config": scoring_config})
    invalid = {
        runner._question_qid(question): question["scoring_config"]["extract_pattern"]
        for question in invalid_questions
        if runner._question_qid(question) in targets
    }
    assert invalid == {target: r"\d+" for target in targets}
    with pytest.raises(ValueError, match=f"one capture group.*{re.escape(targets[0])}"):
        runner.validate_source_vector_scorer_config(invalid_questions, tier=2)


def test_protocol_proposal_rejects_invalid_source_vector_before_runtime_or_receipt_binding(
    monkeypatch,
) -> None:
    args = runner.parse_args(["--protocol-proposal", "--t2-n", "500"])

    def invalid_vector(_tower, *, tier, **_kwargs):
        return ([{
            "id": "real_suite_v1_0043" if tier == 2 else "t1-ok",
            "qid": "real_suite_v1_0043" if tier == 2 else "t1-ok",
            "suite": "suite_a",
            "prompt": "prompt",
            "expected": "256",
            "scoring_method": "exact_match",
            "scoring_config": {"extract_pattern": r"\d+"} if tier == 2 else {},
        }], "core_v2")

    monkeypatch.setattr(runner, "question_vector", invalid_vector)
    monkeypatch.setattr(
        runner,
        "runtime_binding",
        lambda *_args, **_kwargs: pytest.fail("runtime binding must not follow an invalid source vector"),
    )
    with pytest.raises(ValueError, match="real_suite_v1_0043"):
        runner.protocol_proposal(args)


def test_llm_judge_trace_is_total_for_blank_rows_and_row_identity_is_unique(
    tmp_path: Path,
) -> None:
    scorer = runner._load_orchestrator_debug_scorer()
    trace_path = tmp_path / "judge.jsonl"
    runner.write_text(trace_path, "")
    questions = [
        {"id": "judge-blank", "expected": "gold", "scoring_method": "llm_judge", "scoring_config": {}},
        {"id": "judge-fast", "expected": "gold", "scoring_method": "llm_judge", "scoring_config": {}},
    ]
    responses = [
        {"qid": "judge-blank", "answer": "", "correct": False, "error": None},
        {"qid": "judge-fast", "answer": "contains gold", "correct": True, "error": None},
    ]
    with runner.capture_llm_judge_traces(
        trace_path, default_api_url="http://127.0.0.1:8000"
    ):
        assert scorer._score_llm_judge("contains gold", "gold", {}) is True
    runner.seal_judge_trace_outcomes(
        trace_path,
        responses,
        questions,
        tier=2,
        repetition=1,
        default_api_url="http://127.0.0.1:8000",
    )
    audit = runner.validate_response_scoring(
        responses,
        questions,
        trace_path,
        default_api_url="http://127.0.0.1:8000",
        tier=2,
        repetition=1,
    )
    assert audit["judge_trace_rows"] == audit["expected_judge_trace_rows"] == 2
    traces = runner.load_jsonl(trace_path)
    assert [trace["mode"] for trace in traces] == ["blank_fast_failure", "substring_fast_path"]

    runner.write_text(trace_path, json.dumps(traces[0]) + "\n")
    with pytest.raises(ValueError, match="count does not match"):
        runner.validate_response_scoring(
            responses, questions, trace_path, default_api_url="http://127.0.0.1:8000", tier=2, repetition=1
        )


def test_llm_judge_trace_qid_binding_disambiguates_identical_correlations(
    tmp_path: Path,
) -> None:
    scorer = runner._load_orchestrator_debug_scorer()
    trace_path = tmp_path / "judge.jsonl"
    runner.write_text(trace_path, "")
    questions = [
        {"id": "q1", "qid": "q1", "expected": "gold", "scoring_method": "llm_judge", "scoring_config": {}},
        {"id": "q2", "qid": "q2", "expected": "gold", "scoring_method": "llm_judge", "scoring_config": {}},
    ]
    responses = [
        {"qid": "q1", "answer": "contains gold", "correct": True, "error": None},
        {"qid": "q2", "answer": "contains gold", "correct": True, "error": None},
    ]
    with runner.capture_llm_judge_traces(
        trace_path, default_api_url="http://127.0.0.1:8000"
    ):
        for question in questions:
            with runner.judge_trace_fixed_vector_identity(question["qid"]):
                assert scorer._score_llm_judge("contains gold", "gold", {}) is True
    runner.seal_judge_trace_outcomes(
        trace_path,
        responses,
        questions,
        tier=1,
        repetition=1,
        default_api_url="http://127.0.0.1:8000",
    )

    traces = runner.load_jsonl(trace_path)
    assert [trace["fixed_vector_row"]["qid"] for trace in traces] == ["q1", "q2"]
    assert [trace["fixed_vector_qid"] for trace in traces] == ["q1", "q2"]


def test_llm_judge_trace_preserves_fast_and_network_scorer_behavior(
    tmp_path: Path, monkeypatch
) -> None:
    scorer = runner._load_orchestrator_debug_scorer()
    original_scorer = scorer._score_llm_judge

    def judge_post(url, *, json, timeout):
        assert url == "http://127.0.0.1:8000/chat"
        assert json["force_role"] == runner.JUDGE_DEFAULT_ROLE
        return runner.httpx.Response(
            200,
            json={"answer": "true"},
            request=runner.httpx.Request("POST", url),
        )

    monkeypatch.setattr(runner.httpx, "post", judge_post)
    trace_path = tmp_path / "judge.jsonl"
    runner.write_text(trace_path, "")

    with runner.fixed_baseline_environment(tmp_path, "http://127.0.0.1:8000"):
        network_expected = original_scorer("final: mg/2", r"\frac{mg}{2}", {})
        with runner.capture_llm_judge_traces(
            trace_path, default_api_url="http://127.0.0.1:8000"
        ):
            assert scorer._score_llm_judge("contains gold", "gold", {}) is True
            assert (
                scorer._score_llm_judge("final: mg/2", r"\frac{mg}{2}", {})
                is network_expected
            )

    assert scorer._score_llm_judge is original_scorer
    traces = runner.load_jsonl(trace_path)
    assert [row["mode"] for row in traces] == [
        "substring_fast_path",
        "network_judge",
    ]
    assert runner.validate_llm_judge_trace(
        "contains gold",
        "gold",
        {},
        traces[0],
        default_api_url="http://127.0.0.1:8000",
    )
    assert runner.validate_llm_judge_trace(
        "final: mg/2",
        r"\frac{mg}{2}",
        {},
        traces[1],
        default_api_url="http://127.0.0.1:8000",
    )


def test_llm_judge_trace_is_thread_local_and_complete(
    tmp_path: Path, monkeypatch
) -> None:
    scorer = runner._load_orchestrator_debug_scorer()

    def judge_post(url, *, json, timeout):
        return runner.httpx.Response(
            200,
            json={"answer": "false"},
            request=runner.httpx.Request("POST", url),
        )

    monkeypatch.setattr(runner.httpx, "post", judge_post)
    trace_path = tmp_path / "judge.jsonl"
    runner.write_text(trace_path, "")
    calls = [
        ("contains gold", "gold", {}),
        ("answer A", "answer B", {}),
        ("contains silver", "silver", {}),
        ("answer C", "answer D", {}),
    ]
    with runner.fixed_baseline_environment(tmp_path, "http://127.0.0.1:8000"):
        with runner.capture_llm_judge_traces(
            trace_path, default_api_url="http://127.0.0.1:8000"
        ):
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda row: scorer._score_llm_judge(*row), calls))

    assert results == [True, False, True, False]
    traces = runner.load_jsonl(trace_path)
    assert len(traces) == len(calls)
    assert Counter(row["mode"] for row in traces) == {
        "network_judge": 2,
        "substring_fast_path": 2,
    }


def test_runtime_artifact_identity_detects_same_path_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"aaaa")
    cheap_before = runner.runtime_artifact_identities(
        [str(artifact), str(artifact)], include_sha256=False
    )
    full_before = runner.runtime_artifact_identities(
        [str(artifact)], include_sha256=True
    )

    artifact.write_bytes(b"bbbb")
    cheap_after = runner.runtime_artifact_identities(
        [str(artifact)], include_sha256=False
    )
    assert cheap_after != cheap_before

    previous_mtime = full_before[str(artifact.resolve())]["st_mtime_ns"]
    os.utime(artifact, ns=(previous_mtime, previous_mtime))
    full_after = runner.runtime_artifact_identities(
        [str(artifact)], include_sha256=True
    )
    assert full_after[str(artifact.resolve())]["sha256"] != full_before[str(artifact.resolve())]["sha256"]


def test_runtime_artifact_identities_hashes_duplicate_canonical_path_once(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"model")
    calls: list[Path] = []
    original = runner.sha256_path

    def counted_hash(path: Path) -> str:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(runner, "sha256_path", counted_hash)
    identities = runner.runtime_artifact_identities(
        [str(artifact), str(artifact)], include_sha256=True
    )

    canonical = str(artifact.resolve())
    assert list(identities) == [canonical]
    assert identities[canonical]["sha256"] == original(artifact)
    assert calls == [artifact.resolve()]


def test_runtime_artifact_identities_rejects_duplicate_path_stat_race(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"model")
    canonical = str(artifact.resolve())
    original_stat = Path.stat
    calls = 0

    def changing_stat(path: Path, *args, **kwargs):
        nonlocal calls
        result = original_stat(path, *args, **kwargs)
        if str(path) == canonical:
            calls += 1
            # The duplicate's identity read must fail before content hashing.
            if calls == 2:
                return os.stat_result((
                    result.st_mode,
                    result.st_ino,
                    result.st_dev,
                    result.st_nlink,
                    result.st_uid,
                    result.st_gid,
                    result.st_size + 1,
                    result.st_atime,
                    result.st_mtime,
                    result.st_ctime,
                ))
        return result

    monkeypatch.setattr(Path, "stat", changing_stat)
    monkeypatch.setattr(
        runner, "sha256_path", lambda _path: pytest.fail("hash must not run after identity race")
    )

    with pytest.raises(ValueError, match="identity changed while binding"):
        runner.runtime_artifact_identities([str(artifact), str(artifact)], include_sha256=True)


def test_atomic_publish_noreplace_preserves_racing_destination(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    destination = tmp_path / "evidence"
    source.mkdir()
    destination.mkdir()
    (source / "source-marker").write_text("source")
    (destination / "destination-marker").write_text("destination")

    with pytest.raises(FileExistsError):
        runner.atomic_publish_noreplace(source, destination)

    assert (destination / "destination-marker").read_text() == "destination"
    assert (source / "source-marker").read_text() == "source"


def test_scorer_tail_replay_is_once_only_and_preserves_failed_closed_result(monkeypatch) -> None:
    row = SimpleNamespace(answer="generated", error="scoring_unavailable: judge down", correct=False)
    question = {"qid": "q", "scoring_method": "llm_judge", "expected": "gold", "scoring_config": {}}
    calls = 0

    def unavailable(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return None, "scoring_unavailable: still down"

    monkeypatch.setattr(runner, "score_answer_or_error", unavailable)
    first = runner.replay_llm_judge_scorer_tail_once([row], [question])
    second = runner.replay_llm_judge_scorer_tail_once([row], [question])

    assert calls == 1
    assert first == [{"ordinal": 0, "qid": "q", "outcome": "failed_closed"}]
    assert second == []
    assert row.answer == "generated"
    assert row.error == "scoring_unavailable: still down"


def _legacy_t1_questions() -> list[dict]:
    tower = runner.EvalTower(url="http://localhost:8000", timeout=1)
    questions, _core_id = runner.question_vector(
        tower, tier=1, t1_core_id="core_v2", n=50, seed=runner.EVAL_SPEC_SEED
    )
    replacement_map = runner.load_json(runner.CONTEXT_REPLACEMENT_MAP)
    replacements = {
        row["old_id"]: row["new_row"]
        for row in replacement_map["replacements"]
        if row["tier"] == 1
    }
    return [dict(replacements.get(runner._question_qid(question), question)) for question in questions]




# NI74: tiny synthetic algorithm controls; no real-pool distribution proposition.
class _TinySourceVectorTower:
    def __init__(self, pool):
        self.pool = pool

    def _load_pool(self):
        return self.pool

    def _load_designed_core(self, core_id):
        assert core_id == "synthetic-core"
        return [{"id": "t1-only", "suite": "synthetic-a", "prompt": "fixture",
                 "scoring_method": "exact_match", "expected": "1"}], {}, None


def _tiny_source_vector_pool():
    def row(qid, method="exact_match", expected="1"):
        return {"id": qid, "suite": "synthetic-a", "prompt": "fixture " + qid,
                "scoring_method": method, "expected": expected}
    return {"synthetic-a": [row("t1-only"), row("a"), row("b"),
                             row("unscoreable", expected="")],
            "synthetic-b": [dict(row("c"), suite="synthetic-b"),
                             dict(row("d", "llm_judge"), suite="synthetic-b")]}


def test_tiny_t2_vector_is_seeded_and_excludes_t1_and_unscoreable_rows():
    tower = _TinySourceVectorTower(_tiny_source_vector_pool())
    first, core = runner.question_vector(tower, tier=2, t1_core_id="synthetic-core", n=3, seed=17)
    second, again = runner.question_vector(tower, tier=2, t1_core_id="synthetic-core", n=3, seed=17)
    assert first == second and core == again == "legacy_pool_t2_seed_17_n3"
    assert len(first) == 3
    assert {runner._question_qid(q) for q in first} <= {"a", "b", "c", "d"}
    assert len({runner._question_qid(q) for q in first}) == 3
    assert all(q["eval_partition"] == "core" for q in first)


def test_tiny_t2_vector_refuses_an_unavailable_pool():
    with pytest.raises(ValueError, match="current T2 question pool is unavailable"):
        runner.question_vector(_TinySourceVectorTower({}), tier=2,
                               t1_core_id="synthetic-core", n=1, seed=17)


def test_tiny_t2_vector_refuses_an_insufficient_scoreable_draw():
    with pytest.raises(ValueError, match="T2 full-pool draw returned 4, expected 9"):
        runner.question_vector(_TinySourceVectorTower(_tiny_source_vector_pool()), tier=2,
                               t1_core_id="synthetic-core", n=9, seed=17)


def test_tiny_t1_vector_retains_exact_core_cardinality_refusal():
    with pytest.raises(ValueError, match="T1 current full-pool core has 1 questions, expected 2"):
        runner.question_vector(_TinySourceVectorTower({}), tier=1,
                               t1_core_id="synthetic-core", n=2, seed=17)


def test_tiny_t2_vector_names_the_first_zero_group_pattern_refusal():
    questions = [{"id": name, "scoring_method": "exact_match",
                  "scoring_config": {"extract_pattern": r"\d+"}}
                 for name in ("first-invalid", "second-invalid")]
    with pytest.raises(ValueError, match="one capture group for first-invalid; got 0"):
        runner.validate_source_vector_scorer_config(questions, tier=2)


def test_tiny_t2_vector_accepts_one_capture_group_without_running_a_scorer():
    questions = [{"id": "synthetic-valid", "scoring_method": "exact_match",
                  "scoring_config": {"extract_pattern": r"(\d+)"}}]
    assert runner.validate_source_vector_scorer_config(questions, tier=2) is None
