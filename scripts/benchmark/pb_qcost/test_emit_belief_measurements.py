"""No-model tests for the PB-QCOST-1 belief emitter (VB-PBQ-1 write side).

Synthetic units exercise the hashing, the refusal paths and the row vocabulary; the real-record tests run only where
the 2026-10-21 / 10-23 run dirs exist (this host) and are skipped elsewhere.
    /usr/bin/python3 -m pytest scripts/benchmark/pb_qcost/test_emit_belief_measurements.py -q
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import emit_belief_measurements as E  # noqa: E402

ORIG_RUN = Path("/mnt/raid0/llm/tmp/pb-qcost-1/runs/20261021T212922Z")
CONFIRM = Path("/mnt/raid0/llm/tmp/f8-launchers-20261023/pb-qcost-1-confirm")
CONFIRM_R = Path("/mnt/raid0/llm/tmp/f8-launchers-20261024/pb-qcost-1-confirm-r")
BIN = "/opt/fake/llama-server"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _original_unit(tmp: Path, *, tamper: float = 0.0) -> Path:
    """A small original-layout unit whose summary.json is what the producer's own statistics state for its rows."""
    P = E.load_pure("original")[0]
    argv = [BIN, "-m", "/m.gguf", "-np", "5", "-b", "512", "-ub", "512", "--port", "18091"]
    rows = [{"kind": "port_ownership", "ok": True, "pid": 1, "detail": "fixture", "argv": argv}]
    for n, a, b in ((1, 760.0, 0.011), (4, 870.0, 0.0138)):
        dn = 512 - 8 * n
        for rep in (1, 2, 3):
            for length in P.TRAIN + P.HELD:
                ivs = []
                for k, c in enumerate(range(dn, length, dn)):
                    wobble = ((k * 7 + rep * 3 + n) % 5 - 2) * 4.0
                    ivs.append({"c": c, "dn": dn, "dt_s": 0.7, "iter_ms": round(a + b * c + wobble, 2)})
                rows.append({"kind": "run", "n_gen": n, "rep": rep, "length": length, "http": 200, "ttft_ms": 1000.0,
                             "intervals": ivs})
    per_n = {}
    for n in P.NS:
        fit = E.interval_fit(P, rows, n)
        per_n[str(n)] = {"verdict": "PASS", "interval_level": {
            "unit": "ms per server iteration (q = 512 tokens, interval = consecutive log lines)",
            "refit_intercept_ms_at_c0": fit["A"] + tamper, "refit_slope_ms_per_cached_token": fit["B"],
            "noise_floor_train_rmse_ms_per_iter": fit["noise_rmse"], "noise_floor_n_intervals": fit["noise_n"],
            "refit_model_heldout": {"n": fit["held_n"], "p99_abs": fit["p99_abs"],
                                    "p99_under_prediction": fit["p99_under"]}}}
    summary = {"schema": E.RUN_SCHEMA, "status": "complete", "overall": "PASS", "np": 5, "binary": BIN,
               "binary_sha256": "ab" * 32, "finished": "2026-10-21T22:01:00Z", "analysis": {"per_n": per_n},
               "server": {"readback": {"proc_cmdline": argv, "proc_cmdline_matches_argv": True,
                                       "exe_matches_binary": True}}}
    unit = tmp / "20261021T000000Z"
    unit.mkdir()
    (unit / "pb_qcost_1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (unit / "summary.json").write_text(json.dumps(summary))
    return unit


def test_vendored_producers_match_their_pins():
    for name, pin in E.PRODUCERS.values():
        assert _sha(HERE / name) == pin


def test_pure_extraction_carries_the_producer_constants():
    P = E.load_pure("confirm")[0]
    assert P.TRAIN == [4000, 16000, 64000] and P.HELD == [8000, 32000] and P.NS == [1, 4]
    assert P.BOOT_SEED == 20261023 and P.B_OFF == 0.9865 and callable(P.analyse_q) and callable(P.usable)


def test_original_synthetic_rows_are_self_hashed_and_scoped(tmp_path):
    unit = _original_unit(tmp_path)
    rows = E.build_rows(unit, emission="backfill", emitted_at="2026-10-24T12:00:00Z")
    assert len(rows) == 8
    assert {(r["n"], r["metric_key"]) for r in rows} == {(n, k) for n in (1, 4)
                                                         for k in ("intercept", "c_slope", "noise_floor", "heldout_p99_abs")}
    for r in rows:
        assert r["schema"] == E.SCHEMA and r["content_sha256"] == E.content_sha256(r)
        assert r["unit"] == E.METRICS[r["metric_key"]][1] and r["np"] == 5 and r["argv"][0] == BIN
        assert r["q_held_fixed"] is True and r["q_tokens_per_iteration"] == 512 - 8 * r["n"]
        assert r["inputs"]["pb_qcost_1.jsonl"] == _sha(unit / "pb_qcost_1.jsonl")
        assert r["grade"] == "UNGRADED" and r["emission"] == "backfill" and r["emitted_at"] == "2026-10-24T12:00:00Z"
    bad = dict(rows[0], value=rows[0]["value"] + 1)
    assert E.content_sha256(bad) != rows[0]["content_sha256"]


def test_refuses_a_stated_value_that_does_not_rederive(tmp_path):
    unit = _original_unit(tmp_path, tamper=0.1)
    with pytest.raises(E.Refused, match="refit_intercept_ms_at_c0"):
        E.build_rows(unit)
    assert E.main(["--run-dir", str(unit), "--out", str(tmp_path / "o.jsonl")]) == 3
    assert not (tmp_path / "o.jsonl").exists()


def test_refuses_without_port_ownership(tmp_path):
    unit = _original_unit(tmp_path)
    rows = [ln for ln in (unit / "pb_qcost_1.jsonl").read_text().splitlines() if "port_ownership" not in ln]
    (unit / "pb_qcost_1.jsonl").write_text("\n".join(rows) + "\n")
    with pytest.raises(E.Refused, match="port_ownership"):
        E.build_rows(unit)


def test_not_a_unit_and_no_silent_overwrite(tmp_path):
    assert E.main(["--run-dir", str(tmp_path), "--out", str(tmp_path / "x.jsonl")]) == 2
    unit = _original_unit(tmp_path)
    out = tmp_path / "out" / "belief_measurements.jsonl"
    assert E.main(["--run-dir", str(unit), "--out", str(out)]) == 0
    first = out.read_bytes()
    assert E.main(["--run-dir", str(unit), "--out", str(out)]) == 2
    assert out.read_bytes() == first
    assert E.main(["--run-dir", str(unit), "--out", str(out), "--overwrite"]) == 0


@pytest.mark.skipif(not (ORIG_RUN / "pb_qcost_1.jsonl").is_file(), reason="2026-10-21 run dir not on this host")
def test_real_original_run():
    rows = E.build_rows(ORIG_RUN, emission="backfill")
    assert len(rows) == 8 and {r["grade"] for r in rows} == {"INDICATIVE"}
    assert "KVU-12" in rows[0]["grade_cause"]
    assert {r["q_tokens_per_iteration"] for r in rows if r["n"] == 1} == {504}
    assert rows[0]["inputs"]["pb_qcost_1.jsonl"].startswith("a82cdf93")


@pytest.mark.skipif(not (CONFIRM / "VERDICT.json").is_file(), reason="2026-10-23 confirm dir not on this host")
def test_real_confirm_run():
    rows = E.build_rows(CONFIRM, emission="backfill")
    assert len(rows) == 18
    assert sum(r["phase"] == "q_axis" for r in rows) == 10
    assert {r["predeclaration_sha256"] for r in rows} == {
        "f45aef1a43976af3d707aea4e7d61875a83e343ec726a61e1725316b385d3e06"}
    assert "path defect" in rows[0]["grade_cause"] and rows[0]["grade"] == "INDICATIVE"
    qb = [r for r in rows if r["metric_key"] == "q_b"]
    assert all(r["unit"] == "ms per token" and r["q_held_fixed"] is False and len(r["ci95"]) == 2 for r in qb)


def _confirm_r_shaped(tmp: Path) -> Path:
    """The 10-23 records under the AMENDMENT-confirm-R verdict (VERDICT.reanalyzed-pathfix.json): confirm-R's file shape."""
    unit = tmp / "pb-qcost-1-confirm-r-sim"
    (unit / "out" / "run").mkdir(parents=True)
    for rel in ("PREDECLARATION.md", "out/sibling_summary.json", "out/sibling_load.csv", "out/run/pb_qcost_1.jsonl",
                "out/run/summary.json", "out/run/graphs_proof_ub256.json", "out/run/graphs_proof_ub512.json",
                "out/run/graphs_proof_ub1024.json"):
        shutil.copyfile(CONFIRM / rel, unit / rel)
    shutil.copyfile(CONFIRM / "VERDICT.reanalyzed-pathfix.json", unit / "VERDICT.json")
    shutil.copyfile(CONFIRM_R / "AMENDMENT-confirm-R.md", unit / "AMENDMENT-confirm-R.md")
    rec = json.loads((CONFIRM / "RUN_RECORD.json").read_text())
    verdict = json.loads((unit / "VERDICT.json").read_text())
    rec.update({"label": verdict["label"], "amendment_sha256": verdict["amendment_sha256"],
                "verdict_q_sha256": _sha(CONFIRM_R / "verdict_q.py")})
    (unit / "RUN_RECORD.json").write_text(json.dumps(rec))
    return unit


@pytest.mark.skipif(not ((CONFIRM / "VERDICT.reanalyzed-pathfix.json").is_file()
                         and (CONFIRM_R / "AMENDMENT-confirm-R.md").is_file()), reason="confirm records not on this host")
def test_confirm_r_shaped_unit(tmp_path):
    unit = _confirm_r_shaped(tmp_path)
    rows = E.build_rows(unit)
    assert len(rows) == 18 and rows[0]["emission"] == "readout"
    assert rows[0]["amendment_sha256"] == _sha(unit / "AMENDMENT-confirm-R.md")
    assert "AMENDMENT-confirm-R.md" in rows[0]["inputs"]
    assert rows[0]["grade"] == "INDICATIVE" and "ub 256 phase" in rows[0]["grade_cause"]
    (unit / "AMENDMENT-confirm-R.md").write_text("edited\n")
    with pytest.raises(E.Refused, match="amendment"):
        E.build_rows(unit)
