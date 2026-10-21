"""AKR-SG shakedown pieces: /metrics mapping, launcher guards/proofs, toy-endpoint driver run. No model."""
import json
import os
from pathlib import Path

import pytest

from . import ab_server_pair as ab
from . import metrics_map as mm
from . import run_paired_serving_ab as drv

PROM = """# HELP llamacpp:tokens_predicted_total x
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 100
llamacpp:tokens_predicted_seconds_total 4.0
llamacpp:requests_processing 2
llamacpp:requests_deferred 1
"""


def test_metrics_parse_and_window_report():
    b = mm.parse(PROM)
    a = mm.parse(PROM.replace("100", "260").replace("4.0", "8.0"))
    rep = mm.window_report(b, a)
    assert rep["counters"]["tokens_predicted_total"] == 160 and rep["server_decode_tok_s"] == pytest.approx(40.0)
    assert rep["kv"] == "not_exposed" and "n_decode_total" in rep["missing"]      # missing is listed, never zero
    assert rep["gauges"]["requests_processing"]["max"] == 2
    assert mm.window_report(a, b)["counter_reset"]                                  # restart is reported


def _spec(tmp, name, port, cl="0-3", build=None):
    build = build or tmp
    (build / "llama-server").write_text("")
    m = tmp / "m.gguf"; m.write_text("")
    return ab.ArmSpec(name, build, port, m, "cpu", cl)


def test_launcher_refuses_production_port_same_region_missing_pieces(tmp_path):
    with pytest.raises(ab.PairError, match="production port"):
        ab.check_specs(_spec(tmp_path, "A", 8070), _spec(tmp_path, "B", 9512, "4-7"))
    with pytest.raises(ab.PairError, match="SAME cpu_list"):
        ab.check_specs(_spec(tmp_path, "A", 9511), _spec(tmp_path, "B", 9512))
    with pytest.raises(ab.PairError, match="different ports"):
        ab.check_specs(_spec(tmp_path, "A", 9511), _spec(tmp_path, "B", 9511, "4-7"))
    bad = _spec(tmp_path, "A", 9511); bad.cpu_list = None
    with pytest.raises(ab.PairError, match="explicit cpu_list"):
        ab.check_specs(bad, _spec(tmp_path, "B", 9512, "4-7"))
    ab.check_specs(_spec(tmp_path, "A", 9511), _spec(tmp_path, "B", 9512, "4-7"))


def test_env_isolates_ld_library_path_and_argv(tmp_path, monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/other/tree/bin")
    s = _spec(tmp_path, "A", 9511)
    assert ab.env_for(s)["LD_LIBRARY_PATH"] == str(tmp_path)
    argv = ab.argv_for(s)
    assert argv[:3] == ["taskset", "-c", "0-3"] and "--metrics" in argv and "-ngl" in argv


def test_linkage_proof_flags_foreign_library(tmp_path):
    res = ab.maps_linkage(os.getpid(), tmp_path, "cpu")      # this process maps no ggml/llama libs
    assert res["ok"] is False and res["libs"] == []           # nothing inspected is NOT a pass


def test_live_requires_window(tmp_path, monkeypatch):
    monkeypatch.delenv("WINDOW_CONFIRMED", raising=False)
    with pytest.raises(SystemExit, match="WINDOW_CONFIRMED=1"):
        drv.main(["--out-dir", str(tmp_path)])


def test_toy_run_emits_valid_record_with_metrics(tmp_path):
    assert drv.main(["--toy", "--out-dir", str(tmp_path), "--pairs", "3", "--aa-pairs", "3", "--np", "2"]) == 0
    m = json.loads((tmp_path / "manifest.json").read_text())
    rec = json.loads((tmp_path / "serving_ab.json").read_text())
    assert m["verdict"] == "OK" and rec["schema"] == "epyc.autokernel.serving_ab.v1"
    assert rec["decisive"] is False and rec["noise_floor_pct"] is not None        # A/A shakedown: nothing to find
    w = rec["generator"]["metrics"]["windows"]
    assert len(w) == 2 * (3 + 3)      # A/B pairs plus A/A floor pairs, two measurements each
    assert w[0]["metrics"]["counters"]["tokens_predicted_total"] > 0


def test_stop_file_honoured(tmp_path):
    (tmp_path / "STOP").write_text("")
    assert drv.main(["--toy", "--out-dir", str(tmp_path), "--pairs", "2", "--aa-pairs", "2"]) == 4


FAKE = """#!/usr/bin/env python3
import sys, http.server
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b = b'{"status":"ok"}' if self.path == "/health" else b"llamacpp:requests_processing 0\\n"
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def test_start_failure_still_tears_down_and_proves_dead(tmp_path):
    """A fake server (no ggml libs mapped) must FAIL the linkage proof, and the pair must be torn down dead."""
    import socket
    for n in "AB":
        d = tmp_path / n; d.mkdir()
        (d / "llama-server").write_text(FAKE); (d / "llama-server").chmod(0o755)
    m = tmp_path / "m.gguf"; m.write_text("")
    ports = []
    for _ in range(2):
        s = socket.socket(); s.bind(("127.0.0.1", 0)); ports.append(s.getsockname()[1]); s.close()
    a = ab.ArmSpec("A", tmp_path / "A", ports[0], m, "cpu", "0")
    b = ab.ArmSpec("B", tmp_path / "B", ports[1], m, "cpu", "1")
    pair = ab.ServerPair(a, b, tmp_path / "logs")
    with pytest.raises(ab.PairError, match="proof failed"):
        pair.start(ready_timeout_s=30)
    assert pair.procs == {} and ab.listening_pid(ports[0]) is None and ab.listening_pid(ports[1]) is None
