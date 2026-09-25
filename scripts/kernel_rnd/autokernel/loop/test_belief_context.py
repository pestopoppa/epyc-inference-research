"""The AutoKernel planner's belief-kernel reader, receipt and reliance -- no provider call.

Fixture ledgers are written through ROOT's real write path (`ClaimTuple` -> `to_frames` ->
`Ledger`), and read through ROOT's real `fold` + `gate`, loaded from EPYC_ROOT_REPO (default
/workspace). So the grade a test sees is the one `claim_tuple.grade()` assigned.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from autokernel.loop import actors, belief_context as bc
from autokernel.loop.loop import Abstain, Hypothesis, Outcome

ROOT = bc.root_repo()
VIDYA = ROOT / "scripts" / "vidya"
pytestmark = pytest.mark.skipif(not (VIDYA / "gate.py").is_file(),
                                reason=f"ROOT reader modules absent under {VIDYA}")

KVQ = "kv-quant-27b-v10-measurement"
AS_OF = "2026-09-24T12:00:00+00:00"
WRITTEN = "2026-09-24T10:00:00+00:00"
SHA = "a" * 64
GPU_27B_REGIME = {"model": {"path": "/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf"},
                  "quant": "Q8_0", "backend": "gpu", "measurement_surface": "pp512"}
DS41_CPU_REGIME = {"model": {"path": "/mnt/raid0/llm/models/DeepSeek-V4.1-Flash-Q4.gguf"},
                   "quant": "Q4_K", "backend": "cpu", "measurement_surface": "pp512"}
GPU_27B = bc.target_from_context({"current_regime": GPU_27B_REGIME})
DS41_CPU = bc.target_from_context({"current_regime": DS41_CPU_REGIME})
ARMS = ("A_f16_kv", "B_q8_0_kv", "C_q4_0_kv")
DEPTHS = ("d2k", "d32k")
METRICS = ("gpu_decode_tps", "gpu_prefill_tps")
HYPOTHESIS = {"mechanism_id": "akm-x", "statement": "s", "falsifier": "f",
              "target_surface": "ggml/src/ggml-cuda/fattn.cu", "target_symbol": "k"}


def _root_modules():
    sys.path.insert(0, str(VIDYA))
    try:
        import claim_tuple
        import frames
        import ledger
    finally:
        sys.path.remove(str(VIDYA))
    return claim_tuple, frames, ledger


def _kvq(run, arm, depth, metric, *, date="2026-09-24", protocol="P-GPU-1"):
    claim_tuple, _, _ = _root_modules()
    ident = f"kvq-{run}-{arm}-{depth}-{metric}"
    return claim_tuple.ClaimTuple(
        measurement_id=ident, metric=metric, value=100.0, date=date,
        category="BASELINE" if arm == "A_f16_kv" else "CANDIDATE",
        claim=f"{arm} {metric} at {depth}: 100.0 tokens/s (fixture). K buffer 10 MiB; V buffer 10 MiB.",
        protocol_id=protocol, reps=5, reps_basis="scored", unit="tokens/s",
        attestation_sha256=SHA,
        attestation_locator=f"kvq:{run}:{arm}:{depth}:{metric}|data/run/summary.json#sha256={SHA}",
        attestation_present=True, attestation_verified=True, source_kind=KVQ)


def _run(run="run-20260924T100000Z", **kw):
    return [_kvq(run, a, d, m, **kw) for a in ARMS for d in DEPTHS for m in METRICS]


def _ledger(tmp_path, tuples):
    claim_tuple, _, ledger = _root_modules()
    path = tmp_path / "ledger.jsonl"
    led = ledger.Ledger(path)
    for tup in tuples:
        for frame in claim_tuple.to_frames(tup, as_of=WRITTEN, adapter_id="fixture/v1"):
            led.append(frame)
    return path


def _retract(path, tup, at="2026-09-24T11:00:00+00:00"):
    claim_tuple, frames, ledger = _root_modules()
    evidence = next(f for f in claim_tuple.to_frames(tup, as_of=WRITTEN, adapter_id="fixture/v1")
                    if f["frame_type"].endswith("evidence_supports_claim/v1"))
    ledger.Ledger(path).append(frames.make_frame(
        frame_type="epyc.vidya/frame/retraction/v1",
        assertion={"retracts": evidence["frame_id"], "reason": "fixture"},
        provenance={"method": "fixture/v1"}, actor="fixture/v1",
        authority_scope="measurement", created_at=at))


def _read(target, ledger, **kw):
    return bc.read(target, root=ROOT, ledger_path=ledger, as_of=AS_OF, **kw)


# ------------------------------------------------------------------------ matching


def test_27b_gpu_target_is_presented_the_complete_run(tmp_path):
    ev = _read(GPU_27B, _ledger(tmp_path, _run()))
    assert ev["status"] == "presented" and ev["error"] is None
    assert len(ev["claim_ids"]) == 12 and ev["reasons"] == []
    assert {c["grade"] for c in ev["claims"]} == {"Witnessed/Attested"}
    assert ev["runs"] == [{"source_kind": KVQ, "run": "run-20260924T100000Z",
                           "status": "presented", "reason": "",
                           "claim_ids": sorted(ev["claim_ids"])}]
    assert ev["section"].startswith(bc.SECTION_HEADER) and ev["frontier"] == 36


def test_ds41_cpu_target_gets_nothing_and_never_opens_the_ledger(tmp_path):
    with mock.patch.object(bc, "load_root", side_effect=AssertionError("opened ROOT")):
        ev = _read(DS41_CPU, _ledger(tmp_path, _run()))
    assert (ev["status"], ev["reasons"], ev["section"], ev["claim_ids"]) == (
        "omitted", ["inapplicable_target"], "", [])


@pytest.mark.parametrize("change", [
    {"model_path": "/mnt/raid0/llm/models/mtp-Qwen3.8-27B-Q8_0.gguf"},
    {"model_path": "/mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf"},
    {"model_path": "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-Q8_0.gguf"},
    {"quant": "Q4_K"}, {"backend": "cpu", "device": "epyc-9655"}, {"device": "gfx942"},
])
def test_another_model_quant_backend_or_device_yields_nothing(tmp_path, change):
    ev = _read({**GPU_27B, **change}, _ledger(tmp_path, _run()))
    assert ev["section"] == "" and ev["reasons"] == ["inapplicable_target"]


@pytest.mark.parametrize("missing", ["model_path", "quant", "backend", "device"])
def test_incomplete_target_is_not_a_wildcard(tmp_path, missing):
    ev = _read({k: v for k, v in GPU_27B.items() if k != missing}, _ledger(tmp_path, _run()))
    assert ev["section"] == "" and "target incomplete" in ev["error"]


def test_declared_context_depth_keeps_equal_depth_only(tmp_path):
    ev = _read({**GPU_27B, "context_tokens": 32768}, _ledger(tmp_path, _run()))
    assert len(ev["claim_ids"]) == 6 and all(":d32k:" in c["locator"] for c in ev["claims"])


# ----------------------------------------------------------- decision-readiness


def test_an_incomplete_run_is_never_presented(tmp_path):
    ev = _read(GPU_27B, _ledger(tmp_path, _run()[:11]))
    assert ev["status"] == "omitted" and ev["section"] == ""
    assert ev["reasons"] == ["incomplete_run"]
    assert ev["runs"][0]["status"] == "omitted" and "11 of 12" in ev["omitted"][0]["detail"]


def test_a_retracted_row_makes_its_run_incomplete_and_the_older_complete_run_stands(tmp_path):
    old = _run("run-20260923T100000Z", date="2026-09-23")
    new = _run("run-20260924T100000Z")
    ledger = _ledger(tmp_path, old + new)
    _retract(ledger, new[3])
    ev = _read(GPU_27B, ledger)
    assert ev["status"] == "presented"
    assert {c["run"] for c in ev["claims"]} == {"run-20260923T100000Z"}
    assert "gate_refused:abstain" in ev["reasons"] or any(
        r.startswith("gate_refused:") for r in ev["reasons"])
    assert "incomplete_run" in ev["reasons"]
    assert f"clm_{new[3].measurement_id}" not in ev["claim_ids"]


def test_every_run_retracted_means_nothing_presented(tmp_path):
    run = _run()
    ledger = _ledger(tmp_path, run)
    _retract(ledger, run[0])
    ev = _read(GPU_27B, ledger)
    assert ev["status"] == "omitted" and ev["section"] == "" and ev["claim_ids"] == []


def test_an_older_complete_run_is_superseded(tmp_path):
    ev = _read(GPU_27B, _ledger(tmp_path, _run("run-20260923T100000Z", date="2026-09-23")
                                + _run("run-20260924T100000Z")))
    assert {c["run"] for c in ev["claims"]} == {"run-20260924T100000Z"}
    assert "superseded_run" in ev["reasons"]


def test_observations_are_shown_with_their_grade(tmp_path):
    ev = _read(GPU_27B, _ledger(tmp_path, _run(protocol="")))
    assert {c["grade"] for c in ev["claims"]} == {"Judged/Located"}
    assert "protocol none" in ev["section"] and "grade notes: no protocol citation" in ev["section"]


def test_no_ingested_claims_is_omitted_not_unavailable(tmp_path):
    claim_tuple, _, _ = _root_modules()
    other = claim_tuple.ClaimTuple(
        measurement_id="tale-fixture-1", metric="tale_suite_accuracy", value=0.5,
        date="2026-09-24", category="BASELINE", protocol_id="P-X", reps=5,
        claim="Qwen3.8-27B-Q8_0 on gfx90a scored 0.5 (fixture, source with no declared scope).",
        attestation_locator="tale-budget:run:math", source_kind="tale-budget-measurement")
    ev = _read(GPU_27B, _ledger(tmp_path, [other]))
    assert (ev["status"], ev["reasons"], ev["section"]) == ("omitted", ["no_ingested_claims"], "")


# ---------------------------------------------------------------------- rendering


def test_a_complete_run_is_shown_whole_or_not_at_all(tmp_path):
    ledger = _ledger(tmp_path, _run())
    whole = _read(GPU_27B, ledger)
    assert len(whole["claim_ids"]) == 12 and len(whole["section"].encode()) <= bc.DEFAULT_MAX_BYTES
    for kw in ({"max_claims": 5}, {"max_bytes": 1500}):
        cut = _read(GPU_27B, ledger, **kw)
        assert (cut["status"], cut["section"], cut["claim_ids"]) == ("omitted", "", [])
        assert cut["reasons"] == ["bounded"] and cut["omitted_count"] == 12


def test_claims_without_run_semantics_are_bounded_per_claim(tmp_path, monkeypatch):
    import dataclasses
    scope = bc.SOURCE_SCOPES[KVQ]
    monkeypatch.setitem(bc.SOURCE_SCOPES, KVQ, dataclasses.replace(scope, expected_keys=frozenset()))
    few = _read(GPU_27B, _ledger(tmp_path, _run()), max_claims=5)
    assert len(few["claim_ids"]) == 5 and "Shown 5 of 12" in few["section"]
    assert few["reasons"] == ["bounded"]
    small = _read(GPU_27B, tmp_path / "ledger.jsonl", max_bytes=1500)
    assert len(small["section"].encode()) <= 1500 and 0 < len(small["claim_ids"]) < 12


def test_rendering_is_neutral(tmp_path):
    section = _read(GPU_27B, _ledger(tmp_path, _run()))["section"].lower()
    for steering in ("you should", "consider", "recommend", "prefer", "try ", "must ",
                     "do not", "don't", "use this", "attack", "focus", "authorize"):
        assert steering not in section, steering


# ----------------------------------------------------------------------- failure


def test_reader_failures_are_unavailable_and_never_raise(tmp_path):
    assert _read(GPU_27B, tmp_path / "absent.jsonl")["reasons"] == ["ledger_missing"]
    ledger = _ledger(tmp_path, _run())
    lines = ledger.read_text().splitlines()
    tampered = json.loads(lines[1])
    tampered["frame"]["assertion"]["display_text"] = "tampered"
    lines[1] = json.dumps(tampered)
    ledger.write_text("\n".join(lines) + "\n")
    ev = _read(GPU_27B, ledger)
    assert ev["status"] == "unavailable" and ev["reasons"] == ["ledger_integrity"]
    ev = bc.read(GPU_27B, root=tmp_path / "no-root", ledger_path=ledger, as_of=AS_OF)
    assert (ev["status"], ev["reasons"]) == ("unavailable", ["root_missing"])


def test_planner_evidence_timeout_and_garbage_are_unavailable():
    context = {"current_regime": GPU_27B_REGIME}

    def slow(argv, **kw):
        assert kw["timeout"] == bc.READER_TIMEOUT_S
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    ev = bc.planner_evidence(context, runner=slow)
    assert (ev["status"], ev["reasons"], ev["section"]) == ("unavailable", ["reader_timeout"], "")
    ev = bc.planner_evidence(context, runner=lambda argv, **kw: subprocess.CompletedProcess(
        argv, 1, stdout="Traceback", stderr=""))
    assert (ev["status"], ev["reasons"]) == ("unavailable", ["reader_error"])


def test_planner_evidence_child_process_round_trip(tmp_path):
    """The real child: a ROOT-shaped dir whose scripts/vidya IS ROOT and whose ledger is a fixture."""
    fake_root = tmp_path / "root"
    (fake_root / "scripts").mkdir(parents=True)
    (fake_root / "scripts" / "vidya").symlink_to(VIDYA)
    (fake_root / ".vidya").mkdir()
    _ledger(tmp_path, _run()).rename(fake_root / ".vidya" / "ledger.jsonl")
    ev = bc.planner_evidence({"current_regime": GPU_27B_REGIME}, root=fake_root)
    assert ev["status"] == "presented" and len(ev["claim_ids"]) == 12 and ev["elapsed_s"] < 10
    with mock.patch.object(bc.subprocess, "run", side_effect=AssertionError("spawned")):
        ev = bc.planner_evidence({"current_regime": DS41_CPU_REGIME}, root=fake_root)
    assert (ev["status"], ev["reasons"]) == ("omitted", ["inapplicable_target"])


def test_every_declared_scope_names_its_basis():
    for scope in bc.SOURCE_SCOPES.values():
        assert scope.basis.strip() and scope.limits.strip() and scope.backend in ("cpu", "gpu")


# ------------------------------------------------------------------------ reliance


def test_shown_is_not_relied_on_and_only_presented_ids_count():
    assert bc.reliance(None, ["a", "b"]) == {"declared": [], "accepted": [], "rejected": []}
    got = bc.reliance(["a", "zz", "a", 7], ["a", "b"])
    assert got["accepted"] == ["a"]
    assert [r["claim_id"] for r in got["rejected"]] == ["zz", "7"]
    assert bc.reliance(["a"], [])["accepted"] == []


# ----------------------------------------------------------- planner integration


def _presented(tmp_path):
    ev = _read(GPU_27B, _ledger(tmp_path, _run()))
    return {**ev, "target": GPU_27B, "elapsed_s": 0.1}


def _receipts(workspace: Path) -> list[dict]:
    path = workspace.parent / actors.ACTOR_REPLY_DIR / bc.RECEIPT_LOG
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_on_presents_section_records_receipt_and_keeps_validated_reliance(tmp_path):
    workspace = tmp_path / "lane"
    workspace.mkdir()
    evidence = _presented(tmp_path)
    relied = evidence["claim_ids"][0]
    reply = json.dumps({**HYPOTHESIS, "relies_on_claims": [relied, "clm_not_shown"]})
    with mock.patch.object(bc, "planner_evidence", return_value=evidence), \
         mock.patch.object(actors, "_run_agent", return_value=reply) as run:
        hyp = actors.AgentPlanner(workspace=workspace, belief_context="on").propose(
            {"current_regime": GPU_27B_REGIME})
    prompt, env = run.call_args.args[0], run.call_args.kwargs["env"]
    assert bc.SECTION_HEADER in prompt and relied in prompt
    assert json.loads(env[bc.ENV_KEY])["claim_ids"] == evidence["claim_ids"]
    assert isinstance(hyp, Hypothesis) and hyp.relies_on_claims == (relied,)
    [receipt] = _receipts(workspace)
    assert receipt["schema"] == bc.RECEIPT_SCHEMA and receipt["evidence_status"] == "presented"
    assert receipt["claim_ids"] == evidence["claim_ids"] and receipt["frontier"] == 36
    assert receipt["run"][0]["run"] == "run-20260924T100000Z"
    assert receipt["outcome"] == {"kind": "hypothesis", "mechanism_id": "akm-x"}
    assert receipt["reliance"]["accepted"] == [relied]
    assert receipt["reliance"]["rejected"] == [{"claim_id": "clm_not_shown", "reason": "not_presented"}]
    assert hyp.belief_receipt_id == receipt["receipt_id"]
    import hashlib
    assert receipt["call"]["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    row = Outcome(status="keep", hypothesis=hyp).to_attempt()
    assert row["relies_on_claims"] == [relied] and row["belief_receipt_id"] == receipt["receipt_id"]


def test_exposure_without_declaration_records_no_reliance(tmp_path):
    workspace = tmp_path / "lane"
    workspace.mkdir()
    with mock.patch.object(bc, "planner_evidence", return_value=_presented(tmp_path)), \
         mock.patch.object(actors, "_run_agent", return_value=json.dumps(HYPOTHESIS)):
        hyp = actors.AgentPlanner(workspace=workspace, belief_context="on").propose(
            {"current_regime": GPU_27B_REGIME})
    assert hyp.relies_on_claims == ()
    assert "relies_on_claims" not in hyp.to_dict()
    assert _receipts(workspace)[0]["reliance"] == {"declared": [], "accepted": [], "rejected": []}


def test_ds41_cpu_planner_gets_no_section_and_an_inapplicable_receipt(tmp_path):
    workspace = tmp_path / "lane"
    workspace.mkdir()
    with mock.patch.object(bc.subprocess, "run", side_effect=AssertionError("spawned")), \
         mock.patch.object(actors, "_run_agent",
                           return_value='{"abstain": "no feasible hypothesis"}') as run:
        out = actors.AgentPlanner(workspace=workspace, belief_context="on").propose(
            {"current_regime": DS41_CPU_REGIME})
    assert isinstance(out, Abstain)
    assert "Belief-kernel" not in run.call_args.args[0]
    [receipt] = _receipts(workspace)
    assert (receipt["evidence_status"], receipt["evidence_reasons"], receipt["claim_ids"]) == (
        "omitted", ["inapplicable_target"], [])
    assert receipt["outcome"] == {"kind": "abstain", "abstain_reason": "no feasible hypothesis"}


def test_reader_failure_never_fails_or_changes_the_planner_call(tmp_path):
    workspace = tmp_path / "lane"
    workspace.mkdir()
    with mock.patch.object(bc, "planner_evidence", side_effect=RuntimeError("boom")), \
         mock.patch.object(actors, "_run_agent", return_value=json.dumps(HYPOTHESIS)) as run:
        hyp = actors.AgentPlanner(workspace=workspace, belief_context="on").propose(
            {"current_regime": GPU_27B_REGIME})
    assert isinstance(hyp, Hypothesis) and "Belief-kernel" not in run.call_args.args[0]
    assert _receipts(workspace)[0]["evidence_status"] == "unavailable"


def test_off_is_the_historical_prompt_with_no_receipt(tmp_path):
    workspace = tmp_path / "lane"
    workspace.mkdir()
    with mock.patch.object(bc, "planner_evidence", side_effect=AssertionError("read")), \
         mock.patch.object(actors, "_run_agent", return_value=json.dumps(HYPOTHESIS)) as run:
        hyp = actors.AgentPlanner(workspace=workspace).propose({"current_regime": GPU_27B_REGIME})
    assert "Belief-kernel" not in run.call_args.args[0]
    assert run.call_args.kwargs["env"] is None or bc.ENV_KEY not in run.call_args.kwargs["env"]
    assert _receipts(workspace) == [] and hyp.belief_receipt_id == ""


def test_a_failed_call_still_leaves_an_error_receipt(tmp_path):
    workspace = tmp_path / "lane"
    workspace.mkdir()
    with mock.patch.object(bc, "planner_evidence", return_value=_presented(tmp_path)), \
         mock.patch.object(actors, "_run_agent", return_value='{"mechanism_id": "akm-x"}'), \
         mock.patch.object(actors, "_parse_reply", return_value={"mechanism_id": "akm-x"}):
        with pytest.raises(actors.ProviderTransient):
            actors.AgentPlanner(workspace=workspace, belief_context="on").propose(
                {"current_regime": GPU_27B_REGIME})
    assert _receipts(workspace)[0]["outcome"]["kind"] == "error"


def test_metrics_row_carries_the_presented_half(tmp_path):
    workspace = tmp_path / "lane"
    workspace.mkdir()
    summary = bc.metrics_summary({"status": "presented", "frontier": 36, "claim_ids": ["c1"],
                                  "runs": [{"source_kind": KVQ, "run": "r", "status": "presented"}],
                                  "reasons": []})
    actors._record_metrics(workspace, actors.PLANNER_DEFAULT, role="planner", returncode=0,
                           wall_s=1.0, timed_out=False, before_ids=set(), collect_metrics=False,
                           schema=None, final_text=None, salvaged=False,
                           env={bc.ENV_KEY: summary})
    row = json.loads((workspace.parent / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG)
                     .read_text().splitlines()[-1])
    assert row["belief_context"]["claim_ids"] == ["c1"] and row["belief_context"]["frontier"] == 36


def test_hypothesis_schema_accepts_optional_reliance_only():
    assert actors._schema_valid({**HYPOTHESIS, "relies_on_claims": ["c"]}, actors.HYPOTHESIS_SCHEMA)
    assert actors._schema_valid(HYPOTHESIS, actors.HYPOTHESIS_SCHEMA)
    assert not actors._schema_valid({**HYPOTHESIS, "relies_on_claims": "c"}, actors.HYPOTHESIS_SCHEMA)
    assert "relies_on_claims" not in actors.HYPOTHESIS_SCHEMA["required"]
