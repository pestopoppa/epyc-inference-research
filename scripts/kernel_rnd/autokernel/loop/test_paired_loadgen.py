"""AKR-SG-2 acceptance: one paired run on a toy endpoint emits a valid record with delta + floor."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from . import loadgen_selfcheck as lgs
from . import paired_loadgen as pl


def _toy(rate):
    import itertools
    ctr = itertools.count()
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            seed_wobble = (next(ctr) % 3) * 0.001
            out = json.dumps({"timings": {"predicted_per_second": rate * (1 + seed_wobble)}}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers()
            self.wfile.write(out)
        def log_message(self, *a): pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_toy_endpoint_paired_run_emits_valid_record_with_delta_and_floor():
    sa, ua = _toy(100.0)
    sb, ub = _toy(110.0)
    try:
        rec = pl.paired_record(pl.http_arm("A", ua, np=2), pl.http_arm("B", ub, np=2),
                               pairs=4, base_seed=7, aa_pairs=4)
    finally:
        sa.shutdown(); sb.shutdown()
    assert pl.validate_record(rec) == []
    assert rec["effect_pct"] == pytest.approx(10.0, abs=0.5)
    assert rec["noise_floor_pct"] is not None and rec["noise_floor_pct"] < 1.0
    assert rec["decisive"] is True
    assert len(rec["generator"]["pair_delta_pct"]) == 4
    assert rec["generator"]["pair_seeds"] == [7, 8, 9, 10]
    json.dumps(rec)


def test_alternation_order_is_fixed_ABAB():
    seen = []
    a = pl.Arm("A", lambda s: (seen.append(("A", s)) or 100.0, None))
    b = pl.Arm("B", lambda s: (seen.append(("B", s)) or 101.0, None))
    pl.paired_record(a, b, pairs=3, base_seed=5, floor_pct=1.0, floor_unit="session")
    assert seen == [("A", 5), ("B", 5), ("A", 6), ("B", 6), ("A", 7), ("B", 7)]


def test_supplied_floor_requires_unit():
    a = pl.Arm("A", lambda s: (100.0, None)); b = pl.Arm("B", lambda s: (101.0, None))
    with pytest.raises(ValueError):
        pl.paired_record(a, b, pairs=2, base_seed=1, floor_pct=1.0)


def test_voided_sample_is_not_averaged():
    def bad(seed):
        log = lgs.SchedulingLog(); log.record(0.0, 0.05)
        return 100.0, log.require_valid()
    with pytest.raises(lgs.LoadDriverVoided):
        pl.paired_record(pl.Arm("A", bad), pl.Arm("B", bad), pairs=2, base_seed=1,
                         floor_pct=1.0, floor_unit="session")


# --- belief-kernel write side (VB-AK-PAIRED-LOADGEN-WRITE / VB-AKRSG-WRITE) ---------------------

def _belief_record(**extra):
    a = pl.Arm("A", lambda s: (100.0, None), np=2); b = pl.Arm("B", lambda s: (110.0, None), np=2)
    return pl.paired_record(a, b, pairs=3, base_seed=1, floor_pct=1.0, floor_unit="session",
                            belief={"model": "/m/x.gguf", "anchor_build": "/b/a", "candidate_build": "/b/b", **extra})


def test_record_carries_capture_that_rederives_and_exports(tmp_path):
    from . import serving_beliefs as sb
    rec = _belief_record()
    assert pl.validate_record(rec) == []
    cap = rec["belief_capture"]
    assert cap == sb.finish(rec, cap["inputs"])
    p = pl.export_belief(tmp_path, rec, campaign_id="c", epoch="e", recorded_at="2026-10-22T00:00:00Z")
    assert p.is_file() and p.parent == tmp_path / "serving-beliefs"


def test_record_without_capture_is_refused_loudly(tmp_path):
    a = pl.Arm("A", lambda s: (100.0, None)); b = pl.Arm("B", lambda s: (101.0, None))
    rec = pl.paired_record(a, b, pairs=2, base_seed=1, floor_pct=1.0, floor_unit="session")
    assert "belief_capture" not in rec
    with pytest.raises(ValueError, match="no belief_capture"):
        pl.export_belief(tmp_path, rec, campaign_id="c", epoch="e", recorded_at="2026-10-22T00:00:00Z")
    assert not (tmp_path / "serving-beliefs").exists()


def test_recapture_from_stored_record_is_pipeline_only_envelope_and_does_not_touch_samples():
    from . import serving_beliefs as sb
    a = pl.Arm("A", lambda s: (100.0, None), np=2)
    b = pl.Arm("B", lambda s: (110.0, None), np=2)
    stored = pl.paired_record(a, b, pairs=3, base_seed=1, floor_pct=1.0, floor_unit="session")
    assert "belief_capture" not in stored
    rec = pl.recapture_from_stored(stored, run_start=1000.0, run_end=1060.0, np=2,
                                   belief={"model": "/m/x.gguf", "anchor_build": "/b/a", "candidate_build": "/b/a",
                                           "pipeline_only": True, "pipeline_only_reason": "A/A same-build shakedown"})
    assert "belief_capture" not in stored and rec["anchor_samples"] == stored["anchor_samples"]
    assert rec["pipeline_only"] is True and rec["np"] == 2
    assert all(w["window_basis"] == "run_envelope_not_per_round" and (w["window_start"], w["window_end"]) == (1000.0, 1060.0)
               for w in rec["anchor_residency"] + rec["candidate_residency"])
    assert rec["belief_capture"] == sb.finish(rec, rec["belief_capture"]["inputs"])
    with pytest.raises(ValueError, match="already carries"):
        pl.recapture_from_stored(rec, run_start=0, run_end=1, np=2, belief={})
