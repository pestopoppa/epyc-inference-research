"""Coherence judge support: measurement windows, light calibration, route, cloud argv, client.

Inference-free and process-free: window files are tmp files, the judge behind the route
and the cloud CLI runner are fakes, and the client posts through an injected function.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.runtime import measurement_windows as mw
from src.typed_decisions import cloud_judges
from src.typed_decisions import coherence_judge as cj
from src.typed_decisions import coherence_judge_calibration as cal

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


def _iso(delta_s: float) -> str:
    return (NOW + timedelta(seconds=delta_s)).isoformat()


def _cpu(path: Path, **fields) -> Path:
    data = {
        "schema": mw.CPU_WINDOW_SCHEMA,
        "state": "open",
        "loop_holds_claim": False,
        "expires_at": _iso(120),
        "est_close_at": _iso(3600),
    }
    data.update(fields)
    path.write_text(json.dumps(data))
    return path


# ── measurement windows ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fields,held",
    [
        ({}, False),
        ({"state": "closed", "loop_holds_claim": True}, True),
        ({"state": "closing"}, True),
        ({"loop_holds_claim": True}, True),
        ({"est_close_at": _iso(60)}, True),  # open but closing within the margin
        ({"state": "closed", "expires_at": _iso(-5)}, False),  # stale: loop is dead
        ({"state": "weird"}, True),  # unknown state fails closed
        ({"schema": "other"}, True),
    ],
)
def test_cpu_window(tmp_path, fields, held):
    path = _cpu(tmp_path / "cpu.json", **fields)
    assert (mw.cpu_hold(path, now=NOW.timestamp()) is not None) is held


def test_cpu_window_missing_and_garbled(tmp_path):
    assert mw.cpu_hold(tmp_path / "absent.json", now=NOW.timestamp()) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    hold = mw.cpu_hold(bad, now=NOW.timestamp())
    assert hold is not None and "fail closed" in hold.reason


@pytest.mark.parametrize(
    "holder,held", [("production", False), ("autokernel", True), ("released", True), ("bogus", True)]
)
def test_gpu_window(tmp_path, holder, held):
    path = tmp_path / "mi210.json"
    path.write_text(json.dumps({"holder": holder, "expected_end": _iso(1800), "parked_roles": ["architect_critic"]}))
    hold = mw.gpu_hold(path, now=NOW.timestamp())
    assert (hold is not None) is held
    if holder == "autokernel":
        assert 1700 <= hold.retry_after_s <= 1800
        assert hold.detail["parked_roles"] == ["architect_critic"]


def test_local_inference_holds_reads_env_paths(tmp_path, monkeypatch):
    gpu = tmp_path / "g.json"
    gpu.write_text(json.dumps({"holder": "production"}))
    cpu = _cpu(tmp_path / "c.json", state="closed", expires_at=_iso(10**9))
    monkeypatch.setenv(mw.GPU_WINDOW_ENV, str(gpu))
    monkeypatch.setenv(mw.CPU_WINDOW_ENV, str(cpu))
    holds = mw.local_inference_holds()
    assert [h.window for h in holds] == ["cpu"]
    assert mw.snapshot()["gpu"]["held"] is False


# ── light calibration ─────────────────────────────────────────────────────


def _fake_inf70(tmp_path: Path) -> tuple[Path, Path]:
    runs = tmp_path / "runs"
    runs.mkdir()
    prompts = []
    for i in range(8):
        cls = ["coding", "math", "general"][i % 3]
        prompts.append({"id": f"p{i}", "class": cls, "prompt": f"Question number {i} about {cls}?"})
    (tmp_path / "prompts.json").write_text(json.dumps(prompts))

    def text(i, variant=""):
        return f"This is a coherent answer to question {i} with several useful words {variant}".strip()

    def write(arm, variant_for):
        rows = [
            {"id": f"p{i}", "verdict": "COHERENT", "text": text(i, variant_for(i))} for i in range(8)
        ]
        (runs / f"{arm}.rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    write("A1_plain", lambda i: "")
    write("A2_plain", lambda i: "")
    write("B1_n4p05", lambda i: "and more" if i % 2 else "")
    write("K_c4_q8_mtp", lambda i: "plus detail" if i % 3 == 0 else "")
    return runs, tmp_path / "prompts.json"


def test_build_set_is_deterministic_and_light(tmp_path):
    runs, prompts = _fake_inf70(tmp_path)
    items, manifest = cal.build_set(runs_dir=runs, prompts_path=prompts)
    again, manifest2 = cal.build_set(runs_dir=runs, prompts_path=prompts)
    assert manifest["sha256"] == manifest2["sha256"] and items == again
    cats = manifest["categories"]
    assert cats["loop"] == 4 and cats["wrong_alternative"] == 4
    assert 0 < cats["mtp_diverged"] <= 6
    assert manifest["n"] <= 32
    mtp = [i for i in items if i["category"] == "mtp_diverged"]
    assert all(i["base_output"] != i["candidate_output"] and i["label_strength"] == "weak" for i in mtp)
    assert all("plain" not in i["candidate_arm"] for i in mtp)
    wrong = [i for i in items if i["category"] == "wrong_alternative"]
    assert all(i["base_output"] != i["candidate_output"] for i in wrong)
    path = cal.write_set(items, manifest, tmp_path / "sets")
    loaded, sha = cal.read_set(path)
    assert loaded == items and sha == manifest["sha256"]


def _verdict(label, key="cjk-test"):
    return {"verdict": label, "judge_key": key, "backend": "local", "model": "worker_general",
            "served_model": "m.gguf", "scoring_mode": "native"}


def test_run_seals_passed_record_usable_by_judge(tmp_path):
    runs, prompts = _fake_inf70(tmp_path)
    items, manifest = cal.build_set(runs_dir=runs, prompts_path=prompts)
    store = cj.CalibrationStore(tmp_path / "cal")
    summary = cal.run_calibration(
        items, set_sha256=manifest["sha256"], set_id=manifest["set_id"],
        call=lambda item: (200, _verdict(item["gold"])), out_dir=tmp_path / "run", store=store,
        backend="local", model="worker_general", scoring="native",
    )
    (rec,) = summary["records"]
    assert rec["passed"] and rec["accuracy_pass_fail"] == 1.0
    assert store.find_passed("cjk-test")["calibration_id"] == rec["calibration_id"]
    rows = (tmp_path / "run" / "rows.jsonl").read_text().splitlines()
    assert len(rows) == len(items)


def test_garbage_passing_fails_calibration(tmp_path):
    runs, prompts = _fake_inf70(tmp_path)
    items, manifest = cal.build_set(runs_dir=runs, prompts_path=prompts)
    leaked = {next(i["id"] for i in items if i["category"] == "salad")}

    def call(item):
        return 200, _verdict(cj.PASS_VERDICT if item["id"] in leaked else item["gold"])

    store = cj.CalibrationStore(tmp_path / "cal")
    summary = cal.run_calibration(items, set_sha256="s", set_id="x", call=call, out_dir=tmp_path / "r",
                                  store=store, backend="local", model=None, scoring="native")
    assert summary["records"][0]["passed"] is False
    assert store.find_passed("cjk-test") is None


def test_window_refusal_aborts_without_sealing(tmp_path):
    items = [{"id": f"i{k}", "category": "loop", "gold": "INCOHERENT", "prompt": "p",
              "base_output": "b", "candidate_output": "c"} for k in range(3)]
    calls = []

    def call(item):
        calls.append(item["id"])
        return 503, {"error": {"type": "measurement_window_held", "message": "held"}}

    store = cj.CalibrationStore(tmp_path / "cal")
    summary = cal.run_calibration(items, set_sha256="s", set_id="x", call=call, out_dir=tmp_path / "r",
                                  store=store, backend="local", model=None, scoring="native")
    assert summary["aborted"]["type"] == "measurement_window_held"
    assert calls == ["i0"] and summary["records"] == [] and store.records() == []


def test_dry_run_makes_no_calls(tmp_path, capsys):
    runs, prompts = _fake_inf70(tmp_path)
    items, manifest = cal.build_set(runs_dir=runs, prompts_path=prompts)
    path = cal.write_set(items, manifest, tmp_path / "sets")
    assert cal.main(["run", "--set", str(path), "--dry-run", "--url", "http://127.0.0.1:1"]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == len(items)


def test_real_inf70_source_builds_when_present():
    if not cal.INF70_RUNS.exists():
        pytest.skip("INF-70 speed-claim runs not on this host")
    items, manifest = cal.build_set()
    assert 28 <= manifest["n"] <= 32
    assert manifest["categories"]["mtp_diverged"] == 6


# ── cloud registry / argv ─────────────────────────────────────────────────


def test_registry_file_parses():
    reg = cloud_judges.load_registry(cloud_judges.DEFAULT_REGISTRY)
    assert reg["codex-luna-low"].transport == "codex_exec"
    assert reg["codex-luna-low"].model == "gpt-5.6-luna" and reg["codex-luna-low"].effort == "low"
    assert reg["sonnet-low"].transport == "claude_cli"


def test_registry_rejects_bad_transport(tmp_path):
    bad = tmp_path / "r.yaml"
    bad.write_text("schema: epyc.orchestrator.cloud_judges.v1\njudges:\n  x: {transport: http, model: m}\n")
    with pytest.raises(ValueError):
        cloud_judges.load_registry(bad)


def test_claude_argv_is_tool_free_structured(tmp_path):
    spec = cloud_judges.CloudJudgeSpec(name="sonnet-low", transport="claude_cli", model="sonnet")
    argv, out = cloud_judges.build_argv(spec, "sonnet", cj.CLOUD_SCHEMA, tmp_path)
    assert out is None
    assert argv[argv.index("--tools") + 1] == ""
    assert "--restricted" in argv and "--json-schema" in argv
    assert argv[argv.index("--effort") + 1] == "low"


def test_claude_envelope_structured_output_parsed():
    spec = cloud_judges.CloudJudgeSpec(name="sonnet-low", transport="claude_cli", model="sonnet")
    envelope = {"is_error": False, "structured_output": {"verdict": "DEGRADED", "reason": "r"},
                "modelUsage": {"claude-sonnet-x": {}}}
    result = cloud_judges.run_cloud_judge(
        spec, "prompt", cj.CLOUD_SCHEMA, run_fn=lambda a, s, t, c: (0, json.dumps(envelope), "")
    )
    assert result.payload["verdict"] == "DEGRADED" and result.resolved_model == "claude-sonnet-x"


def test_cloud_failure_raises():
    spec = cloud_judges.CloudJudgeSpec(name="sonnet-low", transport="claude_cli", model="sonnet")
    with pytest.raises(cloud_judges.CloudJudgeError):
        cloud_judges.run_cloud_judge(spec, "p", cj.CLOUD_SCHEMA, run_fn=lambda a, s, t, c: (1, "", "Not logged in"))


# ── route ─────────────────────────────────────────────────────────────────


class _StubJudge:
    def __init__(self, outcome):
        self.outcome = outcome

    def judge(self, request):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _verdict_obj(calibration_id="cjcal-x"):
    return cj.JudgeVerdict(
        verdict="COHERENT_EQUIVALENT", confidence=0.9, confidence_source="native_token_probs",
        probabilities={v: 0.0 for v in cj.VERDICTS}, probability_source="native_token_probs",
        backend="local", model="worker_general", served_model="m.gguf", scoring_mode="native",
        judge_version=cj.JUDGE_VERSION, prompt_template_sha256="0" * 64, judge_key="cjk-x",
        calibration_id=calibration_id, call_id="cj-1", elapsed_ms=1.0,
    )


@pytest.fixture
def route_client(monkeypatch):
    from fastapi.testclient import TestClient

    from src.api import app
    from src.api.routes import typed_judge

    holder = {}
    monkeypatch.setattr(typed_judge, "_judge_factory", lambda state: _StubJudge(holder["outcome"]))
    monkeypatch.delenv(typed_judge.ENABLE_ENV, raising=False)
    with TestClient(app, raise_server_exceptions=False, client=("127.0.0.1", 50000)) as client:
        yield client, holder


BODY = {"prompt": "p", "base_output": "b", "candidate_output": "c"}


def test_route_returns_verdict(route_client):
    client, holder = route_client
    holder["outcome"] = _verdict_obj()
    resp = client.post("/v1/typed/coherence_judge", json=BODY)
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "COHERENT_EQUIVALENT" and body["passed"] is True
    assert body["calibration_id"] == "cjcal-x" and body["confidence"] == 0.9


def test_route_window_refusal_is_503_with_retry_after(route_client):
    client, holder = route_client
    holder["outcome"] = cj.JudgeRefused("measurement_window_held", "held", status_code=503, retry_after_s=120)
    resp = client.post("/v1/typed/coherence_judge", json=BODY)
    assert resp.status_code == 503 and resp.headers["Retry-After"] == "120"
    assert resp.json()["error"]["type"] == "measurement_window_held"


def test_route_uncalibrated_409_and_failure_502(route_client):
    client, holder = route_client
    holder["outcome"] = cj.JudgeRefused("judge_uncalibrated", "x", status_code=409)
    assert client.post("/v1/typed/coherence_judge", json=BODY).status_code == 409
    holder["outcome"] = cj.JudgeFailed("transport_error", "down")
    assert client.post("/v1/typed/coherence_judge", json=BODY).status_code == 502


def test_route_rejects_non_local_and_kill_switch(route_client, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api import app
    from src.api.routes import typed_judge

    client, holder = route_client
    holder["outcome"] = _verdict_obj()
    with TestClient(app, raise_server_exceptions=False) as remote:  # host "testclient"
        assert remote.post("/v1/typed/coherence_judge", json=BODY).status_code == 403
    monkeypatch.setenv(typed_judge.ENABLE_ENV, "0")
    assert client.post("/v1/typed/coherence_judge", json=BODY).status_code == 404


# ── client (stdlib, loaded by path) ───────────────────────────────────────


@pytest.fixture(scope="module")
def cjc():
    path = Path(__file__).resolve().parents[2] / "scripts" / "coherence_judge_client.py"
    spec = importlib.util.spec_from_file_location("coherence_judge_client", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module
    spec.loader.exec_module(module)
    return module


def test_client_judge_fn_shapes_and_verdict(cjc):
    sent = []

    def post(url, body, timeout):
        sent.append((url, body))
        return 200, _verdict_obj().to_dict()

    judge_fn = cjc.make_judge_fn(url="http://127.0.0.1:8000", post=post, caller="ak")
    for pair in ({"prompt": "p", "base": "b", "candidate": "c"}, ("p", "b", "c")):
        verdict = judge_fn(pair)
        assert verdict.passed and verdict.calibration_id == "cjcal-x"
    assert sent[0][0] == "http://127.0.0.1:8000/v1/typed/coherence_judge"
    assert sent[0][1]["base_output"] == "b" and sent[0][1]["allow_uncalibrated"] is False


def test_client_refuses_uncalibrated_verdict(cjc):
    judge_fn = cjc.make_judge_fn(post=lambda u, b, t: (200, _verdict_obj(None).to_dict()))
    with pytest.raises(cjc.UncalibratedJudge):
        judge_fn(("p", "b", "c"))
    allowed = cjc.make_judge_fn(allow_uncalibrated=True, post=lambda u, b, t: (200, _verdict_obj(None).to_dict()))
    assert allowed(("p", "b", "c")).calibration_id is None


def test_client_maps_refusals(cjc):
    def post_409(u, b, t):
        return 409, {"error": {"type": "judge_uncalibrated", "message": "x"}}

    def post_503(u, b, t):
        return 503, {"error": {"type": "measurement_window_held", "message": "held", "retry_after_s": 60}}

    with pytest.raises(cjc.UncalibratedJudge):
        cjc.make_judge_fn(post=post_409)(("p", "b", "c"))
    with pytest.raises(cjc.JudgeUnavailable) as exc:
        cjc.make_judge_fn(post=post_503)(("p", "b", "c"))
    assert exc.value.kind == "measurement_window_held" and exc.value.retry_after_s == 60


def test_client_rejects_incomplete_pair(cjc):
    with pytest.raises(ValueError):
        cjc.make_judge_fn(post=lambda u, b, t: (200, {}))({"prompt": "p"})
