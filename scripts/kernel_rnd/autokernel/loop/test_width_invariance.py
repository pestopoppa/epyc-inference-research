"""AKX-ALL-26b: the CPU flash-attn knob-off column of the width-invariance probe driver.

Hermetic: synthetic cell records and a mocked probe process; nothing is built or run."""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from . import width_invariance as wi

REPO = Path(__file__).resolve().parents[4]
CPU_CELLS = REPO / "artifacts" / "akx_all_26_cpu_20261020" / "cells.jsonl.gz"


def _cell(route, nth, typ, n_kv, k, verdict, ns=None):
    reps = [{"equal": verdict == "PASS", "compute_ns": ns} for _ in range(3)]
    return {"schema": "akx-all-26.cell.v1", "route": route, "nth": nth, "op": "FLASH_ATTN_EXT",
            "type": typ.lower(), "shape": str(n_kv), "mode": "-", "k": k, "verdict": verdict,
            "aa_clean": True, "reps": reps}


def _write(out_dir, records, name="g.jsonl"):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_the_full_plan_is_unchanged_and_the_knob_off_plan_pairs_both_routes():
    assert len(wi.plan()) == 74
    assert all(route != wi.FA_SPLIT_KV_OFF_ROUTE for route, *_ in wi.plan())
    groups = wi.fa_split_kv_off_plan()
    for route in ("cpu", wi.FA_SPLIT_KV_OFF_ROUTE):
        for nth in (24, 1):
            assert {t for r, n, op, t in groups if (r, n, op) == (route, nth, "fa")} == set(wi.FA_TYPES)
    assert ("cpu", 24, "ctl", "-") in groups
    assert len(wi.expected_cells("fa-split-kv-off")) == 3 + 2 * 2 * len(wi.FA_TYPES) * len(wi.NKV)
    with pytest.raises(ValueError):
        wi.plan_for("nope")


def test_the_knob_off_route_reaches_the_probe_despite_the_scrub(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, env, capture_output, text):
        seen.update(env)
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setenv("GGML_FA_SPLIT_KV", "1")             # an ambient value is scrubbed ...
    with mock.patch.object(wi.subprocess, "run", side_effect=fake_run):
        args = SimpleNamespace(probe="/probe", build_dir="/b", seed=1)
        with open(tmp_path / "log", "w") as log:
            wi.run_group(args, (wi.FA_SPLIT_KV_OFF_ROUTE, 24, "fa", "F16"), tmp_path, log)
    assert seen["GGML_FA_SPLIT_KV"] == "0" and seen["GGML_IQK"] == "1"   # ... the route sets 0
    seen.clear()
    with mock.patch.object(wi.subprocess, "run", side_effect=fake_run):
        with open(tmp_path / "log", "a") as log:
            wi.run_group(args, ("cpu", 24, "fa", "F16"), tmp_path, log)
    assert "GGML_FA_SPLIT_KV" not in seen                     # the default column stays the default


def test_the_column_reports_whether_f2_clears_and_the_indicative_cost(tmp_path):
    off = wi.FA_SPLIT_KV_OFF_ROUTE
    recs = []
    for k in range(1, wi.KMAX + 1):
        recs.append(_cell("cpu", 24, "F16", 2048, k, "PASS" if k == 1 else "FAIL", ns=1000))
        recs.append(_cell(off, 24, "F16", 2048, k, "PASS", ns=1500))
        recs.append(_cell("cpu", 1, "F16", 2048, k, "PASS", ns=9000))
        recs.append(_cell(off, 1, "F16", 2048, k, "PASS", ns=9000))
        recs.append(_cell("cpu", 24, "Q8_0", 2048, k, "PASS"))   # probe built before 26b: no ns
        recs.append(_cell(off, 24, "Q8_0", 2048, k, "PASS"))
    _write(tmp_path, recs)
    cells, _ = wi.load_cells(tmp_path)
    rows = {(r["type"], r["n_kv"], r["nth"]): r for r in wi.fa_split_kv_off_column(cells)}
    f16 = rows[("F16", "2048", 24)]
    assert (f16["default"], f16["split_kv_off"], f16["f2_clears"]) == ("F@2", "PASS", True)
    assert f16["compute_ns_ratio_k2"] == 1.5 and f16[f"compute_ns_ratio_k{wi.KMAX}"] == 1.5
    assert rows[("F16", "2048", 1)]["f2_clears"] is False         # nothing to clear at 1 thread
    q8 = rows[("Q8_0", "2048", 24)]
    assert q8["f2_clears"] is False and q8["compute_ns_ratio_k2"] is None


def test_table_with_the_knob_off_plan_needs_no_frozen_list_and_reports_missing_cells(tmp_path):
    off = wi.FA_SPLIT_KV_OFF_ROUTE
    recs = [_cell("cpu", 24, "F16", 2048, k, "PASS" if k == 1 else "FAIL", ns=10)
            for k in range(1, wi.KMAX + 1)]
    recs += [_cell(off, 24, "F16", 2048, k, "FAIL" if k >= 3 else "PASS", ns=20)
             for k in range(1, wi.KMAX + 1)]
    _write(tmp_path, recs)
    assert wi.table(SimpleNamespace(out_dir=str(tmp_path), frozen=None, plan="fa-split-kv-off")) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["plan"] == "fa-split-kv-off" and summary["scoring"] == []
    row = next(r for r in summary["fa_split_kv_off"] if r["n_kv"] == "2048" and r["nth"] == 24
               and r["type"] == "F16")
    assert (row["default"], row["split_kv_off"], row["f2_clears"]) == ("F@2", "F@3", False)
    assert summary["missing_groups"], "unrun groups are reported, never omitted"
    md = (tmp_path / "first_fail_tables.md").read_text()
    assert "AKX-ALL-26b" in md and "| F16 | 2048 | 24 | F@2 | F@3 | False |" in md


@pytest.mark.skipif(not CPU_CELLS.exists(), reason="AKX-ALL-26 CPU cells not in this checkout")
def test_the_recorded_cpu_run_has_only_the_default_column():
    """The 2026-10-20 CPU run never set the knob: F16 F@2 at 24 threads, PASS at 1, on the
    default route, and no knob-off cell at all (which is what AKX-ALL-26b adds)."""
    cells = {}
    with gzip.open(CPU_CELLS, "rt") as handle:
        for line in handle:
            r = json.loads(line)
            if r.get("op") == "FLASH_ATTN_EXT":
                cells.setdefault((r["route"], r["nth"], r["op"], r["type"].upper(), r["shape"], r["mode"]),
                                 {})[r["k"]] = r
    column = wi.fa_split_kv_off_column(cells)
    assert column and all(row["split_kv_off"] == "MISSING" for row in column)
    f16_24 = [row for row in column if row["type"] == "F16" and row["nth"] == 24]
    assert f16_24 and all(row["default"] == "F@2" for row in f16_24)
