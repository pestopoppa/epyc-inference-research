"""Post-sweep ingest of the v10 KV-quant sidecar, end to end into the AutoKernel planner receipt.

The synthetic sweep is written by the PRODUCER's own write path (`write_belief_measurements`), and
ingested by ROOT's own dispatcher (`ingest_sources.ingest` -> adapter `kv_quant_27b_v10` ->
`claim_tuple.to_frames`), loaded from EPYC_ROOT_REPO (default /workspace). Only the adapter's
research-root containment is pointed at the test's tmp dir; its producer-sha pin is real.
No GPU, no process, no real ledger: every ledger here is a tmp file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "kernel_rnd"))
import kv_quant_27b_v10_ingest as ingest  # noqa: E402
import kv_quant_27b_v10_sweep as runner  # noqa: E402
from autokernel.loop import belief_context as bc  # noqa: E402

ROOT = bc.root_repo()
VIDYA = ROOT / "scripts" / "vidya"
pytestmark = pytest.mark.skipif(
    not (VIDYA / "adapters" / "kv_quant_27b_v10.py").is_file(),
    reason=f"ROOT v10 KV-quant adapter absent under {VIDYA} (not yet on ROOT main)")
GPU_27B = {"model_path": str(runner.DEFAULT_TARGET_MODEL), "quant": "Q8_0", "backend": "gpu",
           "device": "gfx90a"}
DS41_CPU = {"model_path": "/mnt/raid0/llm/models/DeepSeek-V4.1-Flash-Q4.gguf", "quant": "Q4_K",
            "backend": "cpu", "device": "epyc-9655"}


def _record(prompt_id, index):
    return {"prompt_id": prompt_id, "prompt_index": index, "finish_reason": "stop",
            "semantic_validation": {"passed": True}, "response_sanity": {"passed": True},
            "request_lifecycle": {"fully_contained_valid": True, "fully_contained_sample_count": 1}}


def _rep(cell, depth, rep):
    return {"cell": cell, "depth": depth, "rep": rep, "status": "ok",
            "records": [_record(n, i) for i, (n, _) in enumerate(runner.common.PROMPT_SPECS, 1)],
            "residency": {"passed": True}, "cleanup": {"dead": True}, "post_cleanup_clean": True,
            "post_cleanup_vram_settled": True, "prompt_ms": 100.0, "decode_ms": 200.0,
            "prompt_tokens": 2000, "completion_tokens": 300, "prompt_tps": 10.0,
            "decode_tps": {"A_f16_kv": 20.0, "B_q8_0_kv": 22.0, "C_q4_0_kv": 18.0}[cell],
            "kv_buffer_total_mib": 2176.0, "kv_k_mib": 1088.0, "kv_v_mib": 1088.0}


def _summary(status="ok", *, native_shape=False):
    """A summary built by the producer's own `summarize_cell`.

    KNOWN CONTRACT DEFECT (reported 2026-09-25): `summarize_cell` returns `prompt_tokens`,
    `kv_k_mib` and `kv_v_mib` as `{n, median, mad}` stats dicts, the producer copies those dicts
    into `extra.arm.prefill_tokens_measured` / `extra.kv_buffer_{k,v}_mib`, and the ROOT adapter
    requires plain numbers there -- so a REAL complete sweep's sidecar is refused wholesale
    ("prefill depth evidence missing"). ROOT's own adapter test hand-builds scalar cell summaries
    and so never sees it. `native_shape=True` keeps the producer's real shape (see the strict
    xfail below); the default reduces those three stats to their medians, i.e. the shape the
    adapter accepts today, so the rest of the path stays exercised."""
    rows = [_rep(c.name, d.name, r) for c in runner.CELLS for d in runner.DEPTHS for r in range(1, 6)]
    summaries = {f"{c.name}|{d.name}": runner.summarize_cell(
        [x for x in rows if x["cell"] == c.name and x["depth"] == d.name], c.name, d.name, 5)
        for c in runner.CELLS for d in runner.DEPTHS}
    if not native_shape:
        for cell_summary in summaries.values():
            for key in ("prompt_tokens", "kv_k_mib", "kv_v_mib"):
                cell_summary[key] = cell_summary[key]["median"]
    return {"schema": ingest.SUMMARY_SCHEMA, "status": status, "n": 5,
            "production_named_kernel": True,
            "candidate": {"binary": {"version_line_matches": True, "binary_sha256": "a" * 64,
                                     "linkage_receipt": {"verdict": "pass"}}},
            "kernel_store": {"resolved_bin_dir": "/mnt/raid0/llm/kernels/builds/gpu-x/bin"},
            "cell_summaries": summaries,
            "kv_cost_by_depth": {d.name: runner.kv_cost_comparison(summaries, d.name, True)
                                 for d in runner.DEPTHS},
            "target_model": {"path": str(runner.DEFAULT_TARGET_MODEL)},
            "hardware_state": {"gpu_product": {}}, "device_claim": {"survived_window": True},
            "warmup_discard_policy": "x", "cpu_interference_policy": "y",
            "exact_plan": {"fixed_recipe": {"context": runner.CONTEXT}}}


def _sweep(parent: Path, name="run-20260925T120000Z", status="ok", native_shape=False) -> Path:
    """What `kv_quant_27b_v10_sweep.main()` leaves behind after --execute."""
    run = parent / name
    run.mkdir(parents=True)
    summary = _summary(status, native_shape=native_shape)
    summary_path = run / "summary.json"
    runner.write_json(summary_path, summary)
    capture = runner.write_belief_measurements(run, summary, summary_path)
    runner.write_json(run / "belief_capture_receipt.json", capture)
    return run


@pytest.fixture
def adapter_root(tmp_path, monkeypatch):
    """Point the ROOT adapter's research-root containment at tmp_path (its only path pin)."""
    sys.path.insert(0, str(VIDYA))
    try:
        import importlib
        module = importlib.import_module("adapters.kv_quant_27b_v10")
    finally:
        sys.path.remove(str(VIDYA))
    monkeypatch.setattr(module, "RESEARCH_ROOT", tmp_path)
    return tmp_path


def _ingest(run, ledger, **kw):
    return ingest.ingest_run(run, root=ROOT, ledger_path=ledger, **kw)


def test_a_complete_sweep_ingests_once_and_reaches_the_planner_receipt(adapter_root):
    run, ledger = _sweep(adapter_root / "data"), adapter_root / "ledger.jsonl"
    dry = _ingest(run, ledger, dry_run=True)
    assert dry["status"] == "dry_run_clean" and not ledger.exists()
    got = _ingest(run, ledger)
    assert got["status"] == "ingested" and got["frames_appended"] == 36
    assert got["report"]["rows_projected"] == 12 and len(got["claim_ids"]) == 12
    receipt = json.loads((run / ingest.INGEST_RECEIPT).read_text())
    assert receipt["claim_ids"] == got["claim_ids"] and receipt["frontier_after"] == 36

    # Retry after a repeated completion: nothing appended, no duplicate evidence.
    again = _ingest(run, ledger)
    assert again["status"] == "already_ingested" and again["frames_appended"] == 0
    assert len(ledger.read_text().splitlines()) == 36

    # The planner reads exactly that run, graded by ROOT.
    ev = bc.read(GPU_27B, root=ROOT, ledger_path=ledger)
    assert ev["status"] == "presented" and sorted(ev["claim_ids"]) == got["claim_ids"]
    assert {c["grade"] for c in ev["claims"]} == {"Witnessed/Attested"}
    assert ev["runs"][0]["run"] == run.name and ev["frontier"] == 36
    sealed = bc.seal_receipt({**ev, "target": GPU_27B}, knob="on", prompt="p", workspace="w",
                             seat_arm=None, outcome={"kind": "hypothesis", "mechanism_id": "m"},
                             relied=bc.reliance([ev["claim_ids"][0]], ev["claim_ids"]))
    assert sealed["run"][0]["run"] == run.name and sealed["reliance"]["accepted"] == [ev["claim_ids"][0]]

    # A nonmatching target: no section, a receipt that says why.
    none = bc.read(DS41_CPU, root=ROOT, ledger_path=ledger)
    assert (none["status"], none["reasons"], none["section"]) == ("omitted", ["inapplicable_target"], "")


def test_a_failed_sweep_is_declined_and_writes_nothing(adapter_root):
    run, ledger = _sweep(adapter_root / "data", status="failed"), adapter_root / "ledger.jsonl"
    with pytest.raises(ingest.Refused) as caught:
        _ingest(run, ledger)
    assert caught.value.status == "declined" and caught.value.code == 3
    assert not ledger.exists()


def test_a_pre_hook_run_is_never_backfilled(adapter_root):
    run = adapter_root / "data" / "run-20260922T090000Z"
    run.mkdir(parents=True)
    runner.write_json(run / "summary.json", _summary())
    with pytest.raises(ingest.Refused) as caught:
        _ingest(run, adapter_root / "ledger.jsonl")
    assert caught.value.status == "declined" and "never backfilled" in caught.value.reason


def test_a_partial_sidecar_is_refused(adapter_root):
    run, ledger = _sweep(adapter_root / "data"), adapter_root / "ledger.jsonl"
    sidecar = run / runner.CAPTURE_SIDECAR_NAME
    sidecar.write_text("".join(sidecar.read_text().splitlines(keepends=True)[:11]))
    with pytest.raises(ingest.Refused, match="11 rows"):
        _ingest(run, ledger)
    assert not ledger.exists()


@pytest.mark.parametrize("mutation", [{"written": False}, {"rows": 11}, {"grades_nothing": False},
                                      {"scored_sha256": "c" * 64}, {"path": "/elsewhere.jsonl"}])
def test_an_invalid_capture_receipt_is_refused(adapter_root, mutation):
    run = _sweep(adapter_root / "data")
    path = run / "belief_capture_receipt.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **mutation}))
    with pytest.raises(ingest.Refused, match="belief_capture_receipt.json invalid"):
        _ingest(run, adapter_root / "ledger.jsonl")


def test_a_tampered_row_is_refused_by_the_producer_contract(adapter_root):
    run = _sweep(adapter_root / "data")
    sidecar = run / runner.CAPTURE_SIDECAR_NAME
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    rows[0]["value"] = 999.0
    sidecar.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ingest.Refused, match="validate_row"):
        _ingest(run, adapter_root / "ledger.jsonl")


def test_a_partial_ledger_state_is_refused_for_a_human(adapter_root):
    run, ledger = _sweep(adapter_root / "data"), adapter_root / "ledger.jsonl"
    _ingest(run, ledger)
    lines = ledger.read_text().splitlines(keepends=True)
    ledger.write_text("".join(lines[:18]))            # six rows' frames survive a crash
    with pytest.raises(ingest.Refused, match="partial_ledger_state"):
        _ingest(run, ledger)
    assert len(ledger.read_text().splitlines()) == 18


def test_retracted_ingested_evidence_is_not_decision_ready(adapter_root):
    run, ledger = _sweep(adapter_root / "data"), adapter_root / "ledger.jsonl"
    _ingest(run, ledger)
    sys.path.insert(0, str(VIDYA))
    try:
        import frames as vframes
        import ledger as vledger
    finally:
        sys.path.remove(str(VIDYA))
    records = vledger.Ledger(ledger).read_all()
    evidence = next(r.frame for r in records
                    if r.frame["frame_type"].endswith("evidence_supports_claim/v1"))
    vledger.Ledger(ledger).append(vframes.make_frame(
        frame_type="epyc.vidya/frame/retraction/v1",
        assertion={"retracts": evidence["frame_id"], "reason": "fixture"},
        provenance={"method": "fixture/v1"}, actor="fixture/v1", authority_scope="measurement",
        created_at=evidence["pubinfo"]["created_at"]))
    ev = bc.read(GPU_27B, root=ROOT, ledger_path=ledger)
    assert ev["status"] == "omitted" and ev["section"] == "" and ev["claim_ids"] == []
    assert "incomplete_run" in ev["reasons"]


def test_cli_exit_codes(adapter_root, capsys):
    run, ledger = _sweep(adapter_root / "data"), adapter_root / "ledger.jsonl"
    argv = [str(run), "--root", str(ROOT), "--ledger", str(ledger), "--json"]
    assert ingest.main(argv) == 0 and json.loads(capsys.readouterr().out)["status"] == "ingested"
    assert ingest.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "already_ingested"
    failed = _sweep(adapter_root / "data", name="run-20260925T130000Z", status="failed")
    assert ingest.main([str(failed), "--root", str(ROOT), "--ledger", str(ledger)]) == 3


@pytest.mark.xfail(strict=True, reason=(
    "producer/adapter contract defect: summarize_cell emits prompt_tokens/kv_k_mib/kv_v_mib as "
    "stats dicts, the ROOT adapter requires numbers; remove this marker when either side is fixed"))
def test_the_producers_native_summary_shape_is_ingestable(adapter_root):
    run = _sweep(adapter_root / "data", native_shape=True)
    assert _ingest(run, adapter_root / "ledger.jsonl")["status"] == "ingested"


def test_the_native_shape_fails_closed_today(adapter_root):
    """Until the defect is fixed, the ingester refuses (nothing half-written), never crashes."""
    run, ledger = _sweep(adapter_root / "data", native_shape=True), adapter_root / "ledger.jsonl"
    with pytest.raises(ingest.Refused, match="prefill depth evidence missing"):
        _ingest(run, ledger)
    assert not ledger.exists()
