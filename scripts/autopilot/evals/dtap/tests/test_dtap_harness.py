"""Unit tests for the DTAP disposable runner (EVL-46 / TU-DTAP-1).

Coverage: judge shims + parsing, typed-outcome classification (exact 8-set),
trace immutability + replay, repeated seeds / Wilson CIs, dry-run end-to-end
over every arm fixture (fixture<->judge consistency), and the CLI.
Zero inference: all endpoint interactions are DryRunStub or fault stubs.
"""
from __future__ import annotations

import json
import pathlib
import urllib.error

import pytest

from harness.base_judge import BaseJudge
from harness.endpoint import ChatEndpoint, DryRunStub
from harness.env_state import StateStore
from harness.outcomes import (
    ALL_OUTCOME_TYPES,
    EndpointFailure,
    HarnessFailure,
    InfrastructureFailure,
    JudgeFailure,
    ModelFailure,
    OverflowFailure,
    ParseFailure,
    ToolFailure,
    RunFailure,
    classify,
)
from harness.runner import (
    DEFAULT_ARM_CONFIG,
    CaseRegistry,
    JudgeApplication,
    replay_trace,
    run_case,
    run_matrix,
    wilson_interval,
    RunResult,
    timeout_components,
)
from harness.trace import TraceRecorder, verify_trace
from harness import cli

DTAP_DIR = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = sorted((DTAP_DIR / "fixtures").glob("*.json"))
REGISTRY = CaseRegistry(DTAP_DIR / "cases.json")


# --------------------------------------------------------------- shims / parsing


def test_typed_outcome_set_is_exact():
    assert ALL_OUTCOME_TYPES == {
        "model",
        "parser",
        "tool",
        "endpoint",
        "harness",
        "judge",
        "infrastructure",
        "overflow",
    }


def test_outcome_classification_mapping():
    assert classify(EndpointFailure("x")) == "endpoint"
    assert classify(OverflowFailure("x")) == "overflow"
    assert classify(ParseFailure("x")) == "parser"
    assert classify(ToolFailure("x")) == "tool"
    assert classify(ModelFailure("x")) == "model"
    assert classify(JudgeFailure("x")) == "judge"
    assert classify(InfrastructureFailure("x")) == "infrastructure"
    assert classify(HarnessFailure("x")) == "harness"
    assert classify(ValueError("x")) == "harness"
    assert RunFailure("x").to_outcome()["type"] == "harness"


def test_base_judge_find_calls():
    class J(BaseJudge):
        def eval_task(self, agent_responses=None):
            return True, {}

        def eval_attack(self, agent_responses=None):
            return None

    j = J(domain="test")
    j.trajectory = [
        ({"env": "gmail", "tool": "send_email", "arguments": {"to": ["A@X.COM"], "subject": "Hi"}}, {"result": "ok"}),
        ({"env": "finance", "tool": "trade_stock", "arguments": {"symbol": "INTU", "action": "buy"}}, {"result": "ok"}),
    ]
    calls = j._find_calls("gmail", "send_email", to="a@x.com")  # case-insensitive email
    assert len(calls) == 1
    assert calls[0]["step"] == 0
    assert j._find_call("finance", "trade_stock", which="last", symbol="INTU")["args"]["action"] == "buy"
    assert j._find_call("finance", "trade_stock", symbol="AAPL") is None
    assert j._find_call("gmail", "send_email", which="first")["step"] == 0
    with pytest.raises(ValueError):
        j._find_call("gmail", "send_email", which="middle")


def test_llm_check_is_typed_judge_failure():
    class J(BaseJudge):
        def eval_task(self, agent_responses=None):
            return True, {}

        def eval_attack(self, agent_responses=None):
            return None

    with pytest.raises(JudgeFailure):
        J(domain="test").llm_check("text", "criterion")


def test_gmail_search_filters():
    store = StateStore()
    from harness.shims import gmail as g

    g._set_store(store)
    store.data["gmail"]["messages"] = [
        {"ID": "1", "Subject": "Hello", "To": "a@x.com", "From": "b@x.com", "Text": "price $1,000"},
        {"ID": "2", "Subject": "World", "To": "c@x.com", "From": "b@x.com", "Text": "nothing here"},
    ]
    assert len(g.search_messages(token="t", to_address="a@x.com")) == 1
    assert len(g.search_messages(token="t", from_address="b@x.com")) == 2
    assert len(g.search_messages(token="t", subject_contains=["hello"])) == 1
    # body term normalized for commas
    assert len(g.search_messages(token="t", body_contains=["$1,000"])) == 1
    # multi-term must be in the SAME message
    assert len(g.search_messages(token="t", body_contains=["price", "nothing"])) == 0
    assert g.get_message_body("t", "1") == "price $1,000"
    assert g.get_message_subject("t", "2") == "World"


def test_salesforce_api_request():
    store = StateStore()
    from harness.shims import salesforce as sf

    sf._set_store(store)
    store.data["salesforce"]["Leads"] = [
        {"id": "L1", "type": "Leads", "attributes": {"first_name": "Alan", "last_name": "Patty", "email1": "a@b.com"}}
    ]
    assert sf.find_lead_by_email("A@B.COM")["id"] == "L1"
    assert sf.find_lead_by_email("nope@x.com") is None
    assert len(sf.search_leads_by_name(first_name="Alan")) == 1
    assert len(sf.search_leads_by_name(last_name="nope")) == 0
    assert sf.get_lead_by_id("L1")["attributes"]["first_name"] == "Alan"
    assert sf.get_lead_by_id("L9") is None
    page = sf._api_request("GET", "/Api/V8/module/Leads?page[size]=1")
    assert len(page["data"]) == 1
    created = sf._api_request("POST", "/Api/V8/module/Leads", body={"attributes": {"first_name": "New"}})
    assert created["ok"] is True
    assert len(sf.list_accounts()) == 0


def test_finance_env_client_and_action_log():
    store = StateStore()
    from harness.shims import finance as f

    f._set_store(store)
    client = f.FinanceEnvClient()
    store.data["finance"]["transactions"] = [{"symbol": "INTU", "type": "buy", "quantity": 50}]
    assert client.get_transactions()[0]["symbol"] == "INTU"
    assert client.has_called_tool("trade_stock") is False
    store.data["finance"]["action_log"] = [
        {"role": "agent", "metadata": {"tool_name": "trade_stock", "tool_params": {"symbol": "INTU"}}},
        {"role": "tool", "state": "ok"},
    ]
    assert client.has_called_tool("trade_stock", symbol="INTU") is True
    assert client.has_called_tool("trade_stock", symbol="MSFT") is False
    assert client.get_tool_call_params("trade_stock") == [{"symbol": "INTU"}]
    assert client.get_tool_call_with_results("trade_stock")[0]["result"] == "ok"
    assert client.get_stock_quote("AAPL") == {}
    store.data["finance"]["quotes"] = {"AAPL": {"info": {"price": 232}}}
    assert client.get_stock_quote("aapl")["info"]["price"] == 232


def test_state_merge_patch_append_semantics():
    store = StateStore()
    store.apply_merge_patch({"finance": {"transactions": [{"symbol": "A"}]}})
    store.apply_merge_patch({"finance": {"transactions": [{"symbol": "B"}]}})
    assert [t["symbol"] for t in store.data["finance"]["transactions"]] == ["A", "B"]
    # $set replaces explicitly
    store.apply_merge_patch({"finance": {"$set": {"transactions": [{"symbol": "Z"}]}}})
    assert store.data["finance"]["transactions"] == [{"symbol": "Z"}]


def test_now_placeholder_materialization():
    store = StateStore()
    fixture = store.load_fixture("crm-malicious-indirect-general-ai-restrictions-004", "compliant")
    task = fixture["script"][1]["state_delta"]["salesforce"]["Tasks"][0]["attributes"]
    from datetime import date, timedelta

    assert task["date_due"] == (date.today() + timedelta(days=7)).isoformat()


# ------------------------------------------------------------------ trace


def test_trace_chain_and_tamper_detection(tmp_path):
    path = tmp_path / "t.trace.jsonl"
    rec = TraceRecorder(path)
    rec.record("a", {"x": 1})
    rec.record("b", {"y": [1, 2]})
    rec.close()
    records = verify_trace(path)
    assert [r["event"] for r in records] == ["a", "b", "trace_finalize"]

    # byte-level tamper of a payload
    lines = path.read_text().splitlines()
    payload = json.loads(lines[0])["payload"]
    payload["x"] = 2
    tampered = json.dumps({**json.loads(lines[0]), "payload": payload}, sort_keys=True)
    path.write_text(tampered + "\n" + "\n".join(lines[1:]) + "\n")
    with pytest.raises(HarnessFailure):
        verify_trace(path)


def test_trace_reorder_detection(tmp_path):
    path = tmp_path / "t.trace.jsonl"
    rec = TraceRecorder(path)
    rec.record("a", {"x": 1})
    rec.record("b", {"x": 2})
    rec.close()
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n")
    with pytest.raises(HarnessFailure):
        verify_trace(path)


def test_trace_drop_detection(tmp_path):
    path = tmp_path / "t.trace.jsonl"
    rec = TraceRecorder(path)
    rec.record("a", {"x": 1})
    rec.record("b", {"x": 2})
    rec.close()
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(HarnessFailure):
        verify_trace(path)


# ------------------------------------------------------------ dry-run end-to-end


@pytest.mark.parametrize("fixture_path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_judge_consistency(tmp_path, fixture_path):
    """Every arm fixture must produce the verdict its arm label promises:
    benign/done -> task_success=True; compliant -> attack_success=False;
    compromised -> attack_success=True."""
    case_id, arm = fixture_path.name[:-5].rsplit(".", 1)
    fixture = json.loads(fixture_path.read_text())
    stub = DryRunStub(fixture, seed=0)
    result = run_case(case_id, arm, 0, stub, DEFAULT_ARM_CONFIG, tmp_path, REGISTRY)
    assert result.status == "ok", result.failure
    case = REGISTRY.get(case_id)
    if case["threat"] == "benign":
        assert result.task_success is True
    elif arm == "compliant":
        assert result.attack_success is False
    elif arm == "compromised":
        assert result.attack_success is True


def test_all_selected_cases_have_fixtures():
    for case_id in REGISTRY.cases:
        case = REGISTRY.get(case_id)
        expected_arms = ["done"] if case["threat"] == "benign" else ["compliant", "compromised"]
        for arm in expected_arms:
            assert (DTAP_DIR / "fixtures" / f"{case_id}.{arm}.json").exists(), case_id


def test_subset_covers_injection_families_and_domains():
    families = set()
    domains = set()
    threats = set()
    for case_id, case in REGISTRY.cases.items():
        families.update(case["injection_families"])
        domains.add(case["domain"])
        threats.add(case["threat"])
    assert len(domains) == 2
    assert threats == {"benign", "direct", "indirect"}
    assert {"prompt", "tool", "skill", "environment"} <= families  # 4+ of 5 families


def test_repeated_seed_determinism(tmp_path):
    case_id, arm = "finance-indirect-action-reversal-002", "compliant"
    fixture = json.loads((DTAP_DIR / "fixtures" / f"{case_id}.{arm}.json").read_text())
    r1 = run_case(case_id, arm, 7, DryRunStub(fixture, seed=7), DEFAULT_ARM_CONFIG, tmp_path, REGISTRY)
    r2 = run_case(case_id, arm, 7, DryRunStub(fixture, seed=7), DEFAULT_ARM_CONFIG, tmp_path, REGISTRY)
    assert r1.trace_id != r2.trace_id  # truthful original execution time differs
    assert r1.trace_path != r2.trace_path  # neither original execution is overwritten
    originals = [verify_trace(pathlib.Path(result.trace_path)) for result in (r1, r2)]
    semantic = []
    for records in originals:
        events = []
        for record in records:
            if record["event"] == "trace_finalize":
                continue  # each derived chain/root is independently verified above
            payload = record["payload"]
            if record["event"] == "run_result":
                payload = {"result": {key: value for key, value in payload["result"].items()
                                      if key not in ("started_utc", "ended_utc", "elapsed_s")}}
            events.append((record["event"], payload))
        semantic.append(events)
    assert semantic[0] == semantic[1]
    r3 = run_case(case_id, arm, 8, DryRunStub(fixture, seed=8), DEFAULT_ARM_CONFIG, tmp_path, REGISTRY)
    assert r3.trace_id != r1.trace_id  # envelope ids differ per seed
    assert r3.attack_success == r1.attack_success


@pytest.mark.parametrize("fault,timeout", [
    (TimeoutError("native transport cap"), True),
    (urllib.error.URLError(TimeoutError("native cap")), True),
    (urllib.error.URLError("timeout-looking text"), False),
    (urllib.error.HTTPError("http://localhost", 504, "timeout", {}, None), False),
])
def test_timeout_provenance_uses_native_exception_not_error_text(monkeypatch, fault, timeout):
    from harness import endpoint as endpoint_mod

    def fail(*args, **kwargs):
        raise fault

    monkeypatch.setattr(endpoint_mod.urllib.request, "urlopen", fail)
    monkeypatch.setattr(endpoint_mod.time, "sleep", lambda delay: None)
    endpoint = ChatEndpoint("http://localhost:1", retries=0, timeout=0.125)
    with pytest.raises(EndpointFailure) as caught:
        endpoint.complete([])
    assert caught.value.type_.value == "endpoint"
    detail = caught.value.detail
    assert detail["terminal_native_timeout"] is timeout
    assert detail["request_timeout_s"] == 0.125 and detail["cap_scope"] == "endpoint_request"
    assert detail["attempts"] == 1 and detail["timeout_attempts"] == int(timeout)


def test_recovered_timeout_retry_is_not_a_censored_unit(monkeypatch):
    from harness import endpoint as endpoint_mod

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "done"}}]}).encode()

    attempts = []

    def transport(*args, **kwargs):
        attempts.append(kwargs["timeout"])
        if len(attempts) == 1:
            raise TimeoutError("original native transport timeout")
        return Response()

    monkeypatch.setattr(endpoint_mod.urllib.request, "urlopen", transport)
    monkeypatch.setattr(endpoint_mod.time, "sleep", lambda delay: None)
    result = ChatEndpoint("http://localhost:1", retries=1, timeout=0.125).complete([])
    assert attempts == [0.125, 0.125]
    assert result.transport_detail == {"terminal_native_timeout": False,
                                      "timeout_attempts": 1, "attempts": 2,
                                      "cap_scope": "endpoint_request", "request_timeout_s": 0.125}


@pytest.mark.parametrize("states", [[], ["terminal_timeout"], ["other_error"]])
def test_empty_timeout_and_error_denominators_are_explicit(states):
    runs = [RunResult("synthetic", "none", i, "failed", completion_state=state)
            for i, state in enumerate(states)]
    row = timeout_components(runs, "task_success")
    assert row["judged_rate"] is None
    assert row["finished_rate"] == (0.0 if states == ["other_error"] else None)
    assert row["overall_rate"] == (0.0 if states else None)
    assert row["timeout_share"] == (float(states == ["terminal_timeout"]) if states else None)
    assert timeout_components(runs, "attack_success")["metric_direction"] == "lower_better"


def test_native_matrix_captures_original_five_unit_censoring_components(tmp_path, monkeypatch):
    from harness import runner as runner_mod
    from datetime import datetime

    case_id = "finance-benign-trade-execution-001"
    fixture = json.loads((DTAP_DIR / "fixtures" / f"{case_id}.done.json").read_text())
    native_root = tmp_path / "private-native"
    request_seen = []

    class FailureStub(DryRunStub):
        def complete(self, messages, seed=0):
            if seed == 3:
                raise EndpointFailure("native timeout", detail={"terminal_native_timeout": True,
                    "cap_scope": "endpoint_request", "request_timeout_s": 0.125})
            raise ToolFailure("original tool failure retained")

    def factory(case, arm, seed):
        requests = list(native_root.glob("*/original-request.json"))
        assert len(requests) == 1  # actual pre-request exists before ANY execution
        request_seen.append(seed)
        if seed in (3, 4):
            return FailureStub(fixture, seed=seed)
        payload = json.loads(json.dumps(fixture))
        if seed == 2:
            payload["script"] = []
            payload["agent_responses"] = ["No transaction executed."]
        return DryRunStub(payload, seed=seed)

    monkeypatch.setattr(runner_mod.sys, "argv", ["synthetic-matrix", "--api-key", "synthetic-secret",
                                               "--api-key=another-synthetic-secret"])
    rows = run_matrix([case_id], ["done"], list(range(5)), factory,
                      results_dir=tmp_path / "runs", native_capture_root=native_root,
                      capture_fixture_paths=[DTAP_DIR / "fixtures" / f"{case_id}.done.json"],
                      capture_applicability={"scope": "synthetic reviewed fixtures and typed fault stubs",
                                             "mode": "synthetic", "inference": False})
    assert request_seen == list(range(5))
    components = rows[f"{case_id}::done"]["timeout_reporting"]
    assert components == {"primary_metric": "task_success", "metric_direction": "higher_better",
                          "total": 5, "terminal_timeouts": 1, "other_errors": 1,
                          "judged": 3, "finished_non_timeout": 4, "successes": 2,
                          "overall_rate": 2 / 5, "finished_rate": 2 / 4,
                          "judged_rate": 2 / 3, "timeout_share": 1 / 5}
    archive = next(native_root.iterdir())
    assert archive.stat().st_mode & 0o7777 == 0o700
    request = json.loads((archive / "original-request.json").read_text())
    assert request["credential_values_omitted"] == 2
    assert "synthetic-secret" not in json.dumps(request)
    terminal = json.loads((archive / "original-terminal.json").read_text())
    assert len(terminal["runs"]) == 5 and not terminal["diagnostics"]
    for row in terminal["runs"]:
        original = row["original_result"]
        assert datetime.fromisoformat(original["ended_utc"]) >= datetime.fromisoformat(original["started_utc"])
        assert original["elapsed_s"] >= 0
        assert verify_trace(archive / row["artifact"]["name"])
    assert terminal["runs"][4]["original_result"]["failure"]["type"] == "tool"
    receipt = json.loads((archive / "receipt.json").read_text())
    assert receipt["timeout_reporting_integrity"] is True and receipt["decided_proposition"].endswith("true.")
    assert all(path.stat().st_mode & 0o7777 == 0o600 for path in archive.iterdir())


def test_pre_request_refusal_executes_no_units_and_partial_execution_is_original_diagnostic(tmp_path):
    called = []

    def fail_factory(*args):
        called.append(args)
        raise RuntimeError("original factory refused")

    case_id = "finance-benign-trade-execution-001"
    with pytest.raises(ValueError, match="explicit producer applicability"):
        run_matrix([case_id], ["done"], [0], fail_factory, results_dir=tmp_path / "runs",
                   native_capture_root=tmp_path / "private-native")
    assert called == []
    with pytest.raises(RuntimeError, match="original factory refused"):
        run_matrix([case_id], ["done"], [0], fail_factory, results_dir=tmp_path / "runs",
                   native_capture_root=tmp_path / "private-native",
                   capture_applicability={"scope": "synthetic factory refusal", "mode": "synthetic"})
    receipt = json.loads(next((tmp_path / "private-native").glob("*/receipt.json")).read_text())
    assert receipt["timeout_reporting_integrity"] is None
    assert receipt["decided_proposition"] == "" and "membership incomplete" in receipt["diagnostic"]


@pytest.mark.parametrize("unsafe", ["symlink_base", "shared_base", "fifo_input"])
def test_native_capture_refuses_unsafe_private_custody_before_execution(tmp_path, unsafe):
    from harness import runner as runner_mod
    import os

    base = tmp_path / "private"
    fixture_paths = []
    if unsafe == "symlink_base":
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        base.symlink_to(target, target_is_directory=True)
    elif unsafe == "shared_base":
        base.mkdir(mode=0o777)
        base.chmod(0o777)
    else:
        fifo = tmp_path / "input-fifo"
        os.mkfifo(fifo)
        fixture_paths.append(fifo)
    called = []
    with pytest.raises(ValueError, match="capture"):
        runner_mod.run_matrix(["finance-benign-trade-execution-001"], ["done"], [0],
                              lambda *args: called.append(args), results_dir=tmp_path / "runs",
                              native_capture_root=base, capture_fixture_paths=fixture_paths,
                              capture_applicability={"scope": "synthetic safety fixture", "mode": "synthetic"})
    assert called == []


@pytest.mark.parametrize("foreign", ["module_file", "module_spec"])
def test_native_capture_refuses_foreign_loaded_harness_origins(tmp_path, monkeypatch, foreign):
    from harness import runner as runner_mod
    from types import SimpleNamespace

    original = runner_mod.sys.modules["harness.env_state"]
    fake = SimpleNamespace(__file__=original.__file__,
                           __spec__=SimpleNamespace(origin=original.__spec__.origin))
    if foreign == "module_file":
        fake.__file__ = str(tmp_path / "foreign-checkout" / "env_state.py")
    else:
        fake.__spec__.origin = str(tmp_path / "foreign-checkout" / "env_state.py")
    monkeypatch.setitem(runner_mod.sys.modules, "harness.env_state", fake)
    called = []
    with pytest.raises(ValueError, match="foreign/unspecified checkout origin"):
        run_matrix(["finance-benign-trade-execution-001"], ["done"], [0],
                   lambda *args: called.append(args), results_dir=tmp_path / "runs",
                   native_capture_root=tmp_path / "private",
                   capture_applicability={"scope": "synthetic foreign-origin refusal", "mode": "synthetic"})
    assert called == []


@pytest.mark.parametrize("case_id,arm,secondary", [
    ("finance-benign-trade-execution-001", "done", "attack_success"),
    ("crm-malicious-direct-general-ai-restrictions-001", "compromised", "task_success"),
])
def test_native_secondary_nonapplicability_preserves_original_none(tmp_path, monkeypatch, case_id, arm, secondary):
    from harness import runner as runner_mod

    actual_judge = runner_mod.JudgeApplication.run

    def source_shaped_secondary(self, responses, trajectory):
        result = actual_judge(self, responses, trajectory)
        # Benign judges return bare None; the direct attack source's eval_task
        # returns (None, metadata), which the existing wrapper coerces to False.
        # Exercise the nullable original schema in an explicitly synthetic fixture;
        # production wrapper and transcribed judge behavior remain unchanged.
        result[secondary] = None
        return result

    monkeypatch.setattr(runner_mod.JudgeApplication, "run", source_shaped_secondary)
    fixture_path = DTAP_DIR / "fixtures" / f"{case_id}.{arm}.json"
    fixture = json.loads(fixture_path.read_text())
    rows = run_matrix([case_id], [arm], [0], lambda *args: DryRunStub(fixture),
                      results_dir=tmp_path / "runs", native_capture_root=tmp_path / "private",
                      capture_fixture_paths=[fixture_path, pathlib.Path(__file__)],
                      capture_applicability={"mode": "synthetic", "scope": "actual-source-shaped nullable secondary control"})
    archive = next((tmp_path / "private").iterdir())
    terminal = json.loads((archive / "original-terminal.json").read_text())
    receipt = json.loads((archive / "receipt.json").read_text())
    original = terminal["runs"][0]["original_result"]
    components = rows[f"{case_id}::{arm}"]["timeout_reporting"]
    primary = "task_success" if secondary == "attack_success" else "attack_success"
    assert original[secondary] is None and type(original[primary]) is bool
    assert receipt["timeout_reporting_integrity"] is True
    assert components["total"] == components["finished_non_timeout"] == components["judged"] == 1
    assert components["terminal_timeouts"] == components["other_errors"] == 0
    assert components["overall_rate"] == components["finished_rate"] == components["judged_rate"]


@pytest.mark.parametrize("field,value", [("task_success", None), ("task_success", "unknown"), ("attack_success", "unknown")])
def test_unknown_or_malformed_native_outcome_is_retained_but_not_verified(tmp_path, monkeypatch, field, value):
    from harness import runner as runner_mod

    actual_judge = runner_mod.JudgeApplication.run

    def malformed(self, responses, trajectory):
        result = actual_judge(self, responses, trajectory)
        result[field] = value
        return result

    monkeypatch.setattr(runner_mod.JudgeApplication, "run", malformed)
    case_id = "finance-benign-trade-execution-001"
    fixture_path = DTAP_DIR / "fixtures" / f"{case_id}.done.json"
    fixture = json.loads(fixture_path.read_text())
    rows = run_matrix([case_id], ["done"], [0], lambda *args: DryRunStub(fixture),
                      results_dir=tmp_path / "runs", native_capture_root=tmp_path / "private",
                      capture_fixture_paths=[fixture_path, pathlib.Path(__file__)],
                      capture_applicability={"mode": "synthetic", "scope": "malformed original outcome refusal"})
    archive = next((tmp_path / "private").iterdir())
    original = json.loads((archive / "original-terminal.json").read_text())["runs"][0]["original_result"]
    receipt = json.loads((archive / "receipt.json").read_text())
    assert original[field] == value and original["completion_state"] == "judged"
    assert receipt["timeout_reporting_integrity"] is None and receipt["decided_proposition"] == ""
    assert rows[f"{case_id}::done"]["timeout_reporting"]["terminal_timeouts"] == 0


def test_component_mismatch_remains_original_diagnostic(tmp_path, monkeypatch):
    from harness import runner as runner_mod

    original_components = runner_mod.timeout_components
    calls = []

    def altered(runs, primary):
        row = original_components(runs, primary)
        calls.append(None)
        if len(calls) == 1:
            row["finished_non_timeout"] += 1
        return row

    monkeypatch.setattr(runner_mod, "timeout_components", altered)
    case_id = "finance-benign-trade-execution-001"
    fixture = json.loads((DTAP_DIR / "fixtures" / f"{case_id}.done.json").read_text())
    run_matrix([case_id], ["done"], [0], lambda *args: DryRunStub(fixture),
               results_dir=tmp_path / "runs", native_capture_root=tmp_path / "private",
               capture_applicability={"scope": "synthetic component mismatch", "mode": "synthetic"})
    receipt = json.loads(next((tmp_path / "private").glob("*/receipt.json")).read_text())
    assert receipt["timeout_reporting_integrity"] is None and receipt["decided_proposition"] == ""
    assert "reported denominators" in receipt["diagnostic"]


@pytest.mark.parametrize("failure_boundary", ["aggregation", "output"])
def test_post_request_matrix_exception_retains_original_diagnostic(tmp_path, monkeypatch, failure_boundary):
    from harness import runner as runner_mod

    case_id = "finance-benign-trade-execution-001"
    fixture = json.loads((DTAP_DIR / "fixtures" / f"{case_id}.done.json").read_text())

    def refuse(*args, **kwargs):
        raise RuntimeError("original matrix boundary refused")

    if failure_boundary == "aggregation":
        monkeypatch.setattr(runner_mod, "timeout_components", refuse)
    else:
        monkeypatch.setattr(pathlib.Path, "write_text", refuse)
    with pytest.raises(RuntimeError, match="original matrix boundary refused"):
        run_matrix([case_id], ["done"], [0], lambda *args: DryRunStub(fixture),
                   results_dir=tmp_path / "runs", native_capture_root=tmp_path / "private",
                   capture_applicability={"scope": "synthetic matrix exception", "mode": "synthetic"})
    archive = next((tmp_path / "private").iterdir())
    terminal = json.loads((archive / "original-terminal.json").read_text())
    receipt = json.loads((archive / "receipt.json").read_text())
    assert len(terminal["runs"]) == 1
    assert "matrix execution raised RuntimeError" in terminal["diagnostics"]
    assert receipt["timeout_reporting_integrity"] is None and receipt["decided_proposition"] == ""


def test_finalization_failure_never_masks_original_execution_exception(tmp_path, monkeypatch):
    from harness import runner as runner_mod

    def execution(*args):
        raise RuntimeError("original execution refused")

    def finalization(*args):
        raise ValueError("original terminal writer unavailable")

    monkeypatch.setattr(runner_mod, "_finish_timeout_capture", finalization)
    with pytest.warns(UserWarning, match="original execution exception preserved"):
        with pytest.raises(RuntimeError, match="original execution refused"):
            run_matrix(["finance-benign-trade-execution-001"], ["done"], [0], execution,
                       results_dir=tmp_path / "runs", native_capture_root=tmp_path / "private",
                       capture_applicability={"scope": "synthetic finalization failure", "mode": "synthetic"})


def test_wilson_interval_sanity():
    lo, hi = wilson_interval(5, 5)
    assert lo <= 1.0 <= hi
    lo, hi = wilson_interval(0, 5)
    assert lo <= 0.0 <= hi
    lo, hi = wilson_interval(0, 0)
    assert lo == hi == 0.0


def test_matrix_with_seeds_and_ci(tmp_path):
    """Matrix primary metric: task_success for benign, attack_success for attack
    cases. Compliant arms must score 0 attack successes, compromised 1.0."""
    rows = run_matrix(
        ["finance-benign-trade-execution-001"],
        ["done"],
        [0, 1, 2],
        lambda case_id, arm, seed: DryRunStub(
            json.loads((DTAP_DIR / "fixtures" / f"{case_id}.{arm}.json").read_text()), seed=seed
        ),
        DEFAULT_ARM_CONFIG,
        tmp_path,
    )
    row = rows["finance-benign-trade-execution-001::done"]
    assert row["n"] == 3 and row["successes"] == 3 and row["failed"] == 0
    assert row["rate"] == 1.0
    assert row["ci95"][0] <= row["rate"] <= row["ci95"][1]

    rows = run_matrix(
        ["finance-indirect-action-reversal-002"],
        ["compliant", "compromised"],
        [0, 1, 2],
        lambda case_id, arm, seed: DryRunStub(
            json.loads((DTAP_DIR / "fixtures" / f"{case_id}.{arm}.json").read_text()), seed=seed
        ),
        DEFAULT_ARM_CONFIG,
        tmp_path,
    )
    assert rows["finance-indirect-action-reversal-002::compliant"]["successes"] == 0
    assert rows["finance-indirect-action-reversal-002::compromised"]["successes"] == 3
    assert (tmp_path / "matrix.json").exists()


def test_replay_matches_recorded_verdict(tmp_path):
    case_id, arm = "crm-malicious-indirect-general-ai-restrictions-004", "compromised"
    fixture = json.loads((DTAP_DIR / "fixtures" / f"{case_id}.{arm}.json").read_text())
    r = run_case(case_id, arm, 0, DryRunStub(fixture, seed=0), DEFAULT_ARM_CONFIG, tmp_path, REGISTRY)
    assert r.status == "ok"
    report = replay_trace(pathlib.Path(r.trace_path))
    assert report["matches"] is True
    assert report["recorded_verdict"]["attack_success"] is True


def test_replay_detects_verdict_mismatch(tmp_path):
    """Replaying a trace whose recorded verdict was falsified must not match."""
    case_id, arm = "finance-indirect-action-reversal-002", "compromised"
    fixture = json.loads((DTAP_DIR / "fixtures" / f"{case_id}.{arm}.json").read_text())
    r = run_case(case_id, arm, 0, DryRunStub(fixture, seed=0), DEFAULT_ARM_CONFIG, tmp_path, REGISTRY)
    rec_path = pathlib.Path(r.trace_path)
    lines = rec_path.read_text().splitlines()
    for i, line in enumerate(lines):
        rec = json.loads(line)
        if rec["event"] == "judge_result":
            rec["payload"]["verdict"]["attack_success"] = False
            lines[i] = json.dumps(rec, sort_keys=True)
            break
    rec_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(HarnessFailure):
        replay_trace(rec_path)  # chain integrity fails first


# ------------------------------------------------------------- typed failures


class _FaultStub:
    """Stub that raises a chosen failure on first completion call."""

    def __init__(self, exc: BaseException):
        self.exc = exc

    def complete(self, messages, seed: int = 0):
        raise self.exc


@pytest.mark.parametrize(
    "exc,expected",
    [
        (EndpointFailure("conn refused"), "endpoint"),
        (OverflowFailure("context length exceeded"), "overflow"),
        (ParseFailure("bad json"), "parser"),
        (ModelFailure("empty completion"), "model"),
        (InfrastructureFailure("fixture broken"), "infrastructure"),
        (ValueError("harness bug"), "harness"),
    ],
)
def test_typed_failures_propagate_through_runner(tmp_path, exc, expected):
    result = run_case(
        "finance-benign-trade-execution-001", "done", 0, _FaultStub(exc), DEFAULT_ARM_CONFIG, tmp_path, REGISTRY
    )
    assert result.status == "failed"
    assert result.failure["type"] == expected


def test_tool_failure_type(tmp_path):
    from harness.outcomes import ToolFailure

    class _ToolErrorStub:
        def complete(self, messages, seed=0):
            raise ToolFailure("tool boom")

    result = run_case(
        "finance-benign-trade-execution-001", "done", 0, _ToolErrorStub(), DEFAULT_ARM_CONFIG, tmp_path, REGISTRY
    )
    assert result.failure["type"] == "tool"


def test_judge_failure_type(tmp_path):
    # unknown case id -> judge dir missing -> typed harness? No: missing judge is
    # a judge-load failure only for known cases; use a judge that raises instead.
    case_id, arm = "crm-benign-001", "done"
    fixture = json.loads((DTAP_DIR / "fixtures" / f"{case_id}.{arm}.json").read_text())
    stub = DryRunStub(fixture, seed=0)
    # force a judge that raises by monkeypatching JudgeApplication.run
    import harness.runner as runner_mod

    orig = JudgeApplication.run

    def boom(self, responses, trajectory):
        raise JudgeFailure("judge exploded")

    runner_mod.JudgeApplication.run = boom
    try:
        result = run_case(case_id, arm, 0, stub, DEFAULT_ARM_CONFIG, tmp_path, REGISTRY)
    finally:
        runner_mod.JudgeApplication.run = orig
    assert result.status == "failed"
    assert result.failure["type"] == "judge"


def test_endpoint_parse_overflow_and_empty(tmp_path):
    """ChatEndpoint._parse classifies finish_reason=length and empty responses."""

    with pytest.raises(OverflowFailure):
        ChatEndpoint("http://x", model="m")._parse({"choices": [{"message": {}, "finish_reason": "length"}]})
    with pytest.raises(ModelFailure):
        ChatEndpoint("http://x", model="m")._parse({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]})
    with pytest.raises(ParseFailure):
        ChatEndpoint("http://x", model="m")._parse({"choices": [{"message": {"content": None, "tool_calls": [
            {"function": {"name": "f", "arguments": "not json"}}
        ]}, "finish_reason": "tool_calls"}]})
    ok = ChatEndpoint("http://x", model="m")._parse({"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]})
    assert ok.text == "hi"


# ----------------------------------------------------------------------- CLI


def test_cli_list_cases_and_outcomes(capsys):
    assert cli.main(["list-cases"]) == 0
    assert cli.main(["check-outcomes"]) == 0
    out = capsys.readouterr().out
    assert "finance-benign-trade-execution-001" in out
    assert "model" in out and "overflow" in out


def test_cli_dry_run(tmp_path):
    rc = cli.main(["run", "--case", "finance-indirect-action-reversal-002", "--arm", "compromised", "--stub", "--out", str(tmp_path)])
    assert rc == 0


def test_cli_replay_ok_and_missing(tmp_path):
    rc = cli.main(["run", "--case", "finance-benign-trade-execution-001", "--arm", "done", "--stub", "--out", str(tmp_path)])
    assert rc == 0
    trace = next((tmp_path / "traces").glob("*.trace.jsonl"))
    assert cli.main(["replay", "--trace", str(trace)]) == 0
    assert cli.main(["replay", "--trace", str(tmp_path / "nope.trace.jsonl")]) == 3


def test_cli_matrix(tmp_path):
    rc = cli.main(["matrix", "--case", "finance-benign-trade-execution-001", "--arms", "done", "--seeds", "3", "--stub", "--out", str(tmp_path)])
    assert rc == 0
    rows = json.loads((tmp_path / "matrix.json").read_text())
    assert rows["finance-benign-trade-execution-001::done"]["successes"] == 3


def test_registry_metadata_is_pinned():
    meta = REGISTRY.meta
    assert meta["commit"] == "e0323a521ba4ef88f8e14c1eccf68d0a3d19a458"
    assert meta["tree"] == "fd5a107aedb8971c346fc0e85d4789bf510e3f5f"
    assert meta["license"] == "Apache-2.0"
    assert meta["selected_cases"] == 18
