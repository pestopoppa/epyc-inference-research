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
