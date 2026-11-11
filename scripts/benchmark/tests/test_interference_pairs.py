"""INF-45 artifact (3) interference pairs: design invariants and stop rules (zero inference)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import interference_pairs as ip  # noqa: E402

PAIRS = json.loads(ip.DEFAULT_PAIRS.read_text())


def test_pairs_are_disjoint_private_and_gpu_lane_reused():
    for p in PAIRS["measured"]:
        v = ip.launch_argv(p["victim"])
        g = ip.launch_argv(p["aggressor"])
        assert "--no-mmap" in v and "--no-mmap" in g
        assert p["victim"]["cpu_list"] != p["aggressor"]["cpu_list"]
        assert p["victim"]["model"]["model_path"] != p["aggressor"]["model"]["model_path"]
        assert 19000 <= p["victim"]["port"] <= 19999
    assert [r["id"] for r in PAIRS["reused"]] == ["P3-gpu-lane-vs-cpu"]
    assert {p["id"] for p in PAIRS["measured"]} == {"P1-halfA-vs-halfB-np1", "P1-halfA-vs-halfB-np4", "P2-q0-vs-q1-np1"}


def test_verdicts():
    assert ip.verdict([0.95, 0.951, 0.949, 0.952, 0.948])["verdict"] == "answered-nonzero"
    assert ip.verdict([1.001, 0.999, 1.002, 0.998, 1.0])["verdict"] == "answered-within-floor"
    assert ip.verdict([0.85, 1.0, 0.9, 0.97, 0.99])["verdict"] == "futility"
    assert ip.verdict([0.99, 1.01, 0.97])["verdict"] == "continue"


def test_dry_run_spawns_nothing(capsys):
    assert ip.main(["run", "--pair", "P2-q0-vs-q1-np1", "--out", "/nonexistent-never-created"]) == 0
    out = capsys.readouterr().out
    assert "nothing spawned" in out and "--membind=0" in out
