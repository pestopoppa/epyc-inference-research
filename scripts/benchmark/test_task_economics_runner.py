import ast
import hashlib
import json
import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import task_economics as econ
import v7_quality_gate_economics_runner as runner
import v7_quality_gate_runner as frozen


def draw(prompt=10, completion=5, cost=0.5, correct=True):
    return {
        "completed": True,
        "correct": correct,
        "usage": econ.native_usage(
            {
                "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
                "total_cost_usd": cost,
            }
        ),
    }


def test_exact_arithmetic_and_unknown_denominators():
    report = econ.summarize([draw(), draw(correct=False)])
    assert report["quality"] == 0.5
    assert report["tokens_total"] == 30
    assert report["completions_per_million_tokens"] == pytest.approx(2e6 / 30)
    assert report["quality_per_dollar"] == 0.5
    assert econ.summarize([draw(0, 0, 0)])["tokens_total"] == 0
    assert econ.summarize([draw(0, 0, 0)])["completions_per_million_tokens"] is None
    assert econ.summarize([draw(0, 0, 0)])["quality_per_dollar"] is None
    for bad in (None, True, -1, float("nan"), "10"):
        assert econ.summarize([draw(prompt=bad)])["tokens_total"] is None
        assert econ.summarize([draw(cost=bad)])["quality_per_dollar"] is None
    assert (
        econ.summarize([draw(), {"correct": False, "usage": None}])["tokens_total"]
        is None
    )


QUESTIONS = [
    {"id": "q1", "prompt": "first", "expected": "A", "tier": 1},
    {"id": "q2", "prompt": "second", "expected": "B", "tier": 1},
]


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps(self.payload).encode()


def install_transport(monkeypatch, missing=False, failed=False):
    def urlopen(req, timeout):
        payload = json.loads(req.data)
        prompt = payload["messages"][0]["content"]
        if failed and prompt == "second":
            raise OSError("fixture transport failure")
        result = {
            "choices": [
                {
                    "message": {"content": "A" if prompt == "first" else "C"},
                    "finish_reason": "stop",
                }
            ]
        }
        if not missing:
            result.update(
                usage={
                    "prompt_tokens": 10 if prompt == "first" else 20,
                    "completion_tokens": 5,
                },
                total_cost_usd=0.5,
            )
        return Response(result)

    monkeypatch.setattr(runner.urllib.request, "urlopen", urlopen)
    return urlopen


def test_concurrent_identity_resume_and_behavior_parity(monkeypatch, tmp_path):
    for module in (runner, frozen):
        monkeypatch.setattr(module, "load_questions", lambda *args: QUESTIONS)
    install_transport(monkeypatch)
    capture = tmp_path / "capture.jsonl"
    with capture.open("a") as fh:
        got = runner.run_suite(
            "fixture",
            "http://fixture",
            2,
            42,
            concurrency=2,
            repeats=2,
            per_question_out=fh,
        )
    before = capture.read_bytes()
    rows = [json.loads(line) for line in before.splitlines()]
    assert {(r["id"], r["seed"]) for r in rows} == {
        (q, s) for q in ("q1", "q2") for s in (42, 43)
    }
    assert all(
        r["native_usage"]["prompt_tokens"] == (10 if r["id"] == "q1" else 20)
        for r in rows
    )
    assert got["economics"]["tokens_total"] == 80
    assert got["economics"]["quality_per_dollar"] == 0.25
    assert all(
        r["runner_source_sha256"]
        == hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest()
        for r in rows
    )
    parity = frozen.run_suite(
        "fixture", "http://fixture", 2, 42, concurrency=2, repeats=2
    )
    for key in (
        "accuracy",
        "n",
        "correct",
        "errors",
        "truncated",
        "per_tier",
        "per_item",
    ):
        assert got[key] == parity[key]
    monkeypatch.setattr(
        runner, "query_server_meta", lambda *a, **k: pytest.fail("resumed draw queried")
    )
    with capture.open("a") as fh:
        resumed = runner.run_suite(
            "fixture", "http://fixture", 2, 42, repeats=2, per_question_out=fh
        )
    assert resumed["economics"] == got["economics"]
    assert capture.read_bytes() == before


@pytest.mark.parametrize("missing,failed", [(True, False), (False, True)])
def test_cli_writes_honest_unknowns(monkeypatch, tmp_path, missing, failed):
    monkeypatch.setattr(runner, "load_questions", lambda *args: QUESTIONS)
    monkeypatch.setattr(runner, "wait_for_server", lambda *args, **kwargs: None)
    install_transport(monkeypatch, missing=missing, failed=failed)
    out = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            runner.__file__,
            "--output",
            str(out),
            "--suites",
            "mmlu_pro",
            "--n",
            "2",
            "--concurrency",
            "2",
        ],
    )
    runner.main()
    result = json.loads(out.read_text())
    assert result["suites"][0]["economics"]["n_draws"] == 2
    assert result["suites"][0]["economics"]["tokens_total"] is None
    assert result["suites"][0]["economics"]["quality_per_dollar"] is None


def test_reported_usage_on_failure_is_spend_but_not_a_completion(monkeypatch):
    monkeypatch.setattr(runner, "load_questions", lambda *args: QUESTIONS)

    def query(url, prompt, **kwargs):
        return {
            "text": "A" if prompt == "first" else "B",
            "finish_reason": "stop",
            "error": "failed_after_usage" if prompt == "second" else "",
            "native_usage": draw()["usage"],
        }

    monkeypatch.setattr(runner, "query_server_meta", query)
    result = runner.run_suite("fixture", "http://fixture", 2, 42, concurrency=2)
    economics = result["economics"]
    assert economics["n_draws"] == 2
    assert economics["n_completions"] == 1
    assert economics["tokens_total"] == 30
    assert economics["completions_per_million_tokens"] == pytest.approx(1e6 / 30)
    assert economics["quality"] == result["correct"] / result["n"] == result["accuracy"]
    assert economics["quality_per_dollar"] == 1


def test_capture_repair_refuses_without_modifying_original(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "load_questions", lambda *args: QUESTIONS)
    monkeypatch.setattr(
        runner,
        "query_server_meta",
        lambda *a, **k: pytest.fail("query after invalid resume"),
    )
    capture = tmp_path / "legacy.jsonl"
    original = b'{"suite":"fixture","id":"q1","seed":42}\n'
    capture.write_bytes(original)
    monkeypatch.setattr(runner.os, "replace", lambda *args: pytest.fail("rename"))
    monkeypatch.setattr(Path, "unlink", lambda *args: pytest.fail("delete"))
    with (
        capture.open("a") as fh,
        pytest.raises(RuntimeError, match="new --per-question-out"),
    ):
        runner.run_suite("fixture", "http://fixture", 2, 42, per_question_out=fh)
    assert capture.read_bytes() == original
    assert (
        json.loads(Path(str(capture) + ".rejected.jsonl").read_text())["reason"]
        == "resume_validation_failed"
    )


def test_actual_live_status_path_is_append_only(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "load_questions", lambda *args: QUESTIONS)
    install_transport(monkeypatch)
    monkeypatch.setattr(runner.os, "replace", lambda *args: pytest.fail("rename"))
    monkeypatch.setattr(Path, "unlink", lambda *args: pytest.fail("delete"))
    status = tmp_path / "live.jsonl"
    with (tmp_path / "capture.jsonl").open("a") as fh:
        runner.run_suite(
            "fixture",
            "http://fixture",
            2,
            42,
            per_question_out=fh,
            live_status_out=status,
            concurrency=2,
        )
    events = [json.loads(line) for line in status.read_text().splitlines()]
    assert [event["completed_draws"] for event in events] == [1, 2]
    assert events[-1]["complete"] is True


def test_epoch_fingerprint_fixture_resumes_with_unknown_usage(monkeypatch, tmp_path):
    """Adapt the frozen same-suite fixture to this epoch's two source identities."""
    question = {"id": "q1", "prompt": "prompt", "expected": "C", "tier": 1}
    monkeypatch.setattr(runner, "load_questions", lambda *a, **k: [question])
    monkeypatch.setattr(
        runner, "query_server_meta", lambda *a, **k: pytest.fail("resumed")
    )
    row = {
        "suite": "gpqa",
        "id": "q1",
        "seed": 42,
        "tier": "1",
        "correct": True,
        "empty_response": False,
        "truncated": False,
        "capture_schema_version": runner.CAPTURE_SCHEMA_VERSION,
        "runner_source_sha256": runner.RUNNER_SOURCE_SHA256,
        "economics_source_sha256": runner.ECONOMICS_SOURCE_SHA256,
        "prompt": "prompt",
        "expected": "C",
        "response": "C",
        "reasoning": "",
    }
    for key in ("prompt", "response", "reasoning"):
        row[key + "_fingerprint"] = runner.text_fingerprint(row[key])
    capture = tmp_path / "epoch.jsonl"
    capture.write_text(json.dumps(row) + "\n")
    original = capture.read_bytes()
    with capture.open("a") as fh:
        result = runner.run_suite("gpqa", "http://unused", 1, 42, per_question_out=fh)
    assert result["accuracy"] == 1
    assert result["economics"]["n_draws"] == 1
    assert result["economics"]["tokens_total"] is None
    assert capture.read_bytes() == original


@pytest.mark.parametrize("which", ["output", "questions-out"])
def test_cli_refuses_existing_unrelated_output_before_transport(
    monkeypatch, tmp_path, which
):
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"unrelated evidence\n")
    out = existing if which == "output" else tmp_path / "result.json"
    argv = [runner.__file__, "--output", str(out)]
    if which == "questions-out":
        argv += ["--questions-out", str(existing)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        runner, "wait_for_server", lambda *a, **k: pytest.fail("transport")
    )
    with pytest.raises(FileExistsError, match="refusing to clobber"):
        runner.main()
    assert existing.read_bytes() == b"unrelated evidence\n"


@pytest.mark.parametrize(
    "damage", ["legacy_v3", "source", "fingerprint", "malformed", "transport_failure"]
)
def test_epoch_replaces_destructive_resume_repair_with_refusal(
    monkeypatch, tmp_path, damage
):
    question = {"id": "q1", "prompt": "prompt", "expected": "C", "tier": 1}
    monkeypatch.setattr(runner, "load_questions", lambda *a, **k: [question])
    monkeypatch.setattr(
        runner,
        "query_server_meta",
        lambda *a, **k: pytest.fail("invalid evidence consumed"),
    )
    row = {
        "suite": "gpqa",
        "id": "q1",
        "seed": 42,
        "correct": True,
        "prompt": "prompt",
        "expected": "C",
        "response": "C",
        "reasoning": "",
        "capture_schema_version": runner.CAPTURE_SCHEMA_VERSION,
        "runner_source_sha256": runner.RUNNER_SOURCE_SHA256,
        "economics_source_sha256": runner.ECONOMICS_SOURCE_SHA256,
    }
    for key in ("prompt", "response", "reasoning"):
        row[key + "_fingerprint"] = runner.text_fingerprint(row[key])
    if damage == "legacy_v3":
        row["capture_schema_version"] = "v7_quality_gate_capture.v3"
        row.pop("prompt")
    elif damage == "source":
        row["runner_source_sha256"] = "0" * 64
    elif damage == "fingerprint":
        row["response"] = "tampered response"
    elif damage == "transport_failure":
        row["request_error"] = "connection failed"
    capture = tmp_path / "capture.jsonl"
    original = "not-json\n" if damage == "malformed" else json.dumps(row) + "\n"
    capture.write_text(original)
    with (
        capture.open("a") as fh,
        pytest.raises(RuntimeError, match="new --per-question-out"),
    ):
        runner.run_suite("gpqa", "http://unused", 1, 42, per_question_out=fh)
    assert capture.read_text() == original
    rejected = [
        json.loads(line)
        for line in Path(str(capture) + ".rejected.jsonl").read_text().splitlines()
    ]
    assert len(rejected) == 1


def test_empty_response_never_earns_quality_even_if_scorer_would_accept(monkeypatch):
    for module in (runner, frozen):
        monkeypatch.setattr(module, "load_questions", lambda *a, **k: [QUESTIONS[0]])
        monkeypatch.setattr(
            module,
            "query_server_meta",
            lambda *a, **k: {
                "text": "",
                "finish_reason": "stop",
                "native_usage": draw()["usage"],
            },
        )
        scorer_calls = []

        def accepting_scorer(*args, scorer_calls=scorer_calls):
            scorer_calls.append(args)
            return True

        monkeypatch.setattr(module, "score_response", accepting_scorer)
        result = module.run_suite("fixture", "http://unused", 1, 42)
        assert result["correct"] == 0
        assert result["accuracy"] == 0
        assert scorer_calls == []  # Both _work guards bypass scoring empty responses.
        if module is runner:
            assert result["economics"]["quality"] == result["accuracy"]
            assert result["economics"]["n_completions"] == 0


def test_resumed_quality_uses_official_stored_correctness_even_when_empty(
    monkeypatch, tmp_path
):
    question = QUESTIONS[0]
    monkeypatch.setattr(runner, "load_questions", lambda *a, **k: [question])
    monkeypatch.setattr(
        runner, "query_server_meta", lambda *a, **k: pytest.fail("resumed")
    )
    row = {
        "suite": "fixture",
        "id": "q1",
        "seed": 42,
        "correct": True,
        "prompt": "first",
        "expected": "A",
        "response": "",
        "reasoning": "",
        "empty_response": True,
        "capture_schema_version": runner.CAPTURE_SCHEMA_VERSION,
        "runner_source_sha256": runner.RUNNER_SOURCE_SHA256,
        "economics_source_sha256": runner.ECONOMICS_SOURCE_SHA256,
        "native_usage": draw()["usage"],
    }
    for key in ("prompt", "response", "reasoning"):
        row[key + "_fingerprint"] = runner.text_fingerprint(row[key])
    path = tmp_path / "resumed.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with path.open("a") as fh:
        result = runner.run_suite(
            "fixture", "http://unused", 1, 42, per_question_out=fh
        )
    assert (
        result["economics"]["quality"]
        == result["correct"] / result["n"]
        == result["accuracy"]
        == 1
    )
    assert result["economics"]["n_completions"] == 0


def test_belief_finalize_failure_publishes_plain_valid_output_once(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(runner, "load_questions", lambda *args: QUESTIONS)
    monkeypatch.setattr(runner, "wait_for_server", lambda *args, **kwargs: None)
    install_transport(monkeypatch)
    out = tmp_path / "result.json"

    def incomplete_attachment(output, **kwargs):
        assert not out.exists()  # No on-disk output can be truncated by finalization.
        return []  # Failure after attachment returns, before fields commit.

    monkeypatch.setitem(
        sys.modules,
        "v7_quality_gate_beliefs",
        types.SimpleNamespace(
            PROTOCOL_ID="fixture", attach_accuracy_beliefs=incomplete_attachment
        ),
    )
    modes = []
    original_open = Path.open

    def tracked_open(path, mode="r", *args, **kwargs):
        if path == out:
            modes.append(mode)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            runner.__file__,
            "--output",
            str(out),
            "--suites",
            "mmlu_pro",
            "--n",
            "2",
            "--belief-category",
            "BASELINE",
        ],
    )
    assert runner.main() == 3
    assert modes == ["x"]
    result = json.loads(out.read_text())
    assert result["suites"][0]["n"] == 2
    assert "belief_measurements" not in result
    assert "belief_attestation" not in result


@pytest.mark.parametrize("refuse", [False, True])
def test_actual_retained_belief_hook_success_and_refusal(monkeypatch, tmp_path, refuse):
    import v7_quality_gate_beliefs as beliefs

    monkeypatch.setattr(
        runner, "load_questions", lambda *args: [] if refuse else QUESTIONS
    )
    monkeypatch.setattr(runner, "wait_for_server", lambda *args, **kwargs: None)
    install_transport(monkeypatch)
    out = tmp_path / "result.json"
    actual_attach = beliefs.attach_accuracy_beliefs
    called = []

    def observed_attach(output, **kwargs):
        called.append(True)
        assert not out.exists()
        return actual_attach(output, **kwargs)

    monkeypatch.setattr(beliefs, "attach_accuracy_beliefs", observed_attach)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            runner.__file__,
            "--output",
            str(out),
            "--suites",
            "mmlu_pro",
            "--n",
            "2",
            "--belief-category",
            "BASELINE",
        ],
    )
    assert runner.main() == (3 if refuse else 0)
    assert called == [True]
    result = json.loads(out.read_text())
    if refuse:
        assert "belief_measurements" not in result
        assert "belief_attestation" not in result
    else:
        rows = result["belief_measurements"]
        assert rows[0]["value"] == result["suites"][0]["accuracy"]
        assert rows[0]["reps"] == result["suites"][0]["n"]
        assert rows[0]["extra"]["producer_sha256"] == runner.RUNNER_SOURCE_SHA256
        digest = beliefs.content_hash(
            {"meta": result["meta"], "suites": result["suites"]}
        )
        assert result["belief_attestation"]["attestation_sha256"] == digest
        assert rows[0]["extra"]["attestation_sha256"] == digest
        assert rows[0]["extra"]["attestation_path"] == str(out)


def test_ast_request_and_scoring_parity_with_frozen_epoch():
    def functions(module):
        tree = ast.parse(Path(module.__file__).read_text())
        return {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }

    before, after = functions(frozen), functions(runner)

    class RemoveNativeUsage(ast.NodeTransformer):
        def visit_Dict(self, node):
            self.generic_visit(node)
            pairs = [
                (key, value)
                for key, value in zip(node.keys, node.values)
                if not (isinstance(key, ast.Constant) and key.value == "native_usage")
            ]
            node.keys, node.values = (
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
            )
            return node

    normalized_query = RemoveNativeUsage().visit(after["query_server_meta"])
    assert ast.dump(normalized_query) == ast.dump(before["query_server_meta"])
    for name in ("query_server", "load_questions", "text_fingerprint"):
        assert ast.dump(after[name]) == ast.dump(before[name])
    after["_work"].args.defaults = []  # Explicitly bind this repeat's identical seed.
    after["_work"].args.args = [
        arg for arg in after["_work"].args.args if arg.arg != "rep_seed"
    ]
    assert ast.dump(after["_work"]) == ast.dump(before["_work"])

    def official_counter(function):
        return next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "not response"
            and isinstance(node.body[0], ast.AugAssign)
        )

    assert ast.dump(official_counter(after["_apply"])) == ast.dump(
        official_counter(before["_apply"])
    )


def test_explicit_repeat_binding_matches_frozen_concurrent_lifecycle(monkeypatch):
    def transport_fixture():
        barriers = {seed: threading.Barrier(2) for seed in (42, 43)}
        calls, completed = [], []
        lock = threading.Lock()

        def transport(request, timeout):
            payload = json.loads(request.data)
            prompt, seed = payload["messages"][0]["content"], payload["seed"]
            with lock:
                if seed == 43:
                    assert sorted(completed) == [("first", 42), ("second", 42)]
                calls.append((prompt, seed))
            barriers[seed].wait(timeout=3)
            with lock:
                completed.append((prompt, seed))
            return Response(
                {
                    "choices": [
                        {
                            "message": {"content": "A" if prompt == "first" else "C"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            )

        return transport, calls

    results = []
    for module in (frozen, runner):
        transport, calls = transport_fixture()
        monkeypatch.setattr(module, "load_questions", lambda *a, **k: QUESTIONS)
        monkeypatch.setattr(module.urllib.request, "urlopen", transport)
        results.append(
            module.run_suite(
                "fixture", "http://unused", 2, 42, concurrency=2, repeats=2
            )
        )
        assert sorted(calls) == [
            ("first", 42),
            ("first", 43),
            ("second", 42),
            ("second", 43),
        ]
    for key in (
        "accuracy",
        "n",
        "correct",
        "errors",
        "truncated",
        "per_tier",
        "per_item",
    ):
        assert results[0][key] == results[1][key]
