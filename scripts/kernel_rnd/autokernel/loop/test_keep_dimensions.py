"""G5: the cross-workload keep gate extended across every tracked dimension, plus the
capacity probe over a MIXED request sequence (GPU-POOL-1 scenario)."""
import json

import pytest

from . import capacity_probe as cp
from . import surface_validation as sv

GIB = 1 << 30


def _row(effect_pct, floor_pct=2.0, decisive=False):
    return {"schema": "epyc.autokernel.serving_ab.v1", "effect": effect_pct / 100.0,
            "effect_pct": effect_pct, "noise_floor_pct": floor_pct, "decisive": decisive,
            "floor_unit": "process", "effect_unit": "process"}


def _cap(peaks, limit=62 * GIB):
    return cp.evaluate(peaks, limit_bytes=limit, backend="gpu", ctx=196608, np=2)


def test_parse_dimensions_refuses_unknown_names():
    assert sv.parse_dimensions("") == ()
    assert sv.parse_dimensions("capacity, long_decode,capacity") == ("capacity", "long_decode")
    with pytest.raises(sv.SurfaceValidationRefused, match="unknown keep dimension"):
        sv.parse_dimensions("vram")


def test_primary_dimension_follows_the_serving_concurrency():
    assert sv.primary_dimension(1) == "short_decode"
    assert sv.primary_dimension(4) == "concurrent_aggregate"


def test_undeclared_dimensions_are_skipped_and_recorded():
    record = sv.keep_dimensions(declared=(), primary="short_decode", comparisons={}, capacity=None)
    assert record["passed"] is True
    assert set(record["dimensions"]) == set(sv.KEEP_DIMENSIONS)
    assert all(r["disposition"] == "skipped" for r in record["dimensions"].values())


def test_declared_dimension_without_a_measurement_refuses():
    record = sv.keep_dimensions(declared=("short_decode", "long_decode"), primary="short_decode",
                                comparisons={}, capacity=None)
    assert record["dimensions"]["short_decode"]["disposition"] == "passed"
    assert record["dimensions"]["long_decode"]["disposition"] == "pending"
    assert record["passed"] is False and "long_decode" in record["reason"]


def test_non_primary_throughput_uses_the_cross_workload_rule():
    passed = sv.keep_dimensions(declared=("long_decode", "prefill_at_depth"), primary="short_decode",
                                comparisons={"long_decode": _row(1.0),
                                             "prefill_at_depth": _row(0.0)}, capacity=None)
    assert passed["passed"] is True
    regressed = sv.keep_dimensions(declared=("long_decode",), primary="short_decode",
                                   comparisons={"long_decode": _row(-3.0, decisive=True)},
                                   capacity=None)
    assert regressed["dimensions"]["long_decode"]["disposition"] == "failed"
    errored = sv.keep_dimensions(declared=("long_decode",), primary="short_decode",
                                 comparisons={"long_decode": {"error": "server died"}}, capacity=None)
    assert errored["dimensions"]["long_decode"]["disposition"] == "pending"


def test_capacity_passes_only_over_the_mixed_sequence_and_without_growth():
    fits = sv.keep_dimensions(declared=("capacity",), primary="short_decode", comparisons={},
                              capacity=_cap([52 * GIB, 52 * GIB + (8 << 20), 52 * GIB + (8 << 20)]))
    assert fits["passed"] is True
    over = sv.keep_dimensions(declared=("capacity",), primary="short_decode", comparisons={},
                              capacity=_cap([63 * GIB] * 3))
    assert over["dimensions"]["capacity"]["disposition"] == "failed"
    steady = {"peak_bytes": 52 * GIB, "limit_bytes": 62 * GIB, "growth_bytes": 0}
    single_shape = sv.keep_dimensions(declared=("capacity",), primary="short_decode",
                                      comparisons={}, capacity=steady)
    assert single_shape["dimensions"]["capacity"]["disposition"] == "pending"
    assert "mixed" in single_shape["dimensions"]["capacity"]["reason"]
    failed_probe = sv.keep_dimensions(declared=("capacity",), primary="short_decode",
                                      comparisons={}, capacity={"error": "boot timeout"})
    assert failed_probe["dimensions"]["capacity"]["disposition"] == "pending"


def test_mixed_sequence_alternates_nmax0_and_varies_the_batch():
    body = json.dumps({"prompt": "abc " * 20, "n_predict": 64}).encode()
    cycles = cp.mixed_sequence([("p", body)], cycles=4)
    assert len(cycles) == 4
    labels = [label for label, _ in cycles[0]]
    assert labels == ["decode_nmax0", "decode_draft", "prefill_long", "decode_nmax0_short"]
    parsed = [json.loads(b) for _, b in cycles[0]]
    assert parsed[0]["speculative.n_max"] == 0 and "speculative.n_max" not in parsed[1]
    assert len(parsed[2]["prompt"]) > 10 * len(parsed[1]["prompt"]) and parsed[2]["n_predict"] == 1
    with pytest.raises(ValueError):
        cp.mixed_sequence([("p", body)], cycles=2)


class _FakeServer:
    """VRAM model of the v10 mmvq_q8_1_graph_cache defect (GPU-POOL-1): every n_max
    alternation re-captures the graph and strands ~0.29 GiB of shared-pool buffers."""

    def __init__(self, leaky, base=52 * GIB):
        self.leaky, self.vram, self.peak, self.last_nmax = leaky, base, base, None
        self.pid = 4242

    def post(self, _url, body, _timeout):
        nmax = json.loads(body).get("speculative.n_max", "draft")
        if self.leaky and self.last_nmax is not None and nmax != self.last_nmax:
            self.vram += int(0.29 * GIB)
        self.last_nmax = nmax
        self.peak = max(self.peak, self.vram)

    def poll(self):
        return None

    def send_signal(self, _sig):
        self.terminated = True

    def wait(self, _t=None):
        return 0


def _probe(server, sequence):
    class Sampler:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def watch_pid(self, pid):
            assert pid == server.pid

        @property
        def proof(self):
            return {"own_pid_peak_vram_bytes": server.peak}

    return cp.run(argv=["llama-server"], env={}, port=18190, sequence=sequence, backend="gpu",
                  popen=lambda *a, **k: server, post=server.post,
                  wait_healthy=lambda *a: None, sampler_factory=Sampler, settle=lambda: None)


def test_capacity_catches_the_graph_cache_pool_growth_a_steady_shape_misses():
    body = json.dumps({"prompt": "the quick brown fox", "n_predict": 32}).encode()
    mixed = cp.mixed_sequence([("p", body)], cycles=4)
    steady = [[("decode_draft", body)] * 4 for _ in range(4)]

    leaky_mixed = _probe(_FakeServer(leaky=True), mixed)
    record = sv.keep_dimensions(declared=("capacity",), primary="short_decode", comparisons={},
                                capacity=cp.evaluate(leaky_mixed, limit_bytes=62 * GIB,
                                                     backend="gpu", ctx=196608, np=2))
    capacity = record["dimensions"]["capacity"]
    assert max(leaky_mixed) < 62 * GIB, "the defect hides under the ceiling at first"
    assert capacity["disposition"] == "failed" and "growth" in capacity["reason"]

    leaky_steady = _probe(_FakeServer(leaky=True), steady)
    assert leaky_steady[-1] == leaky_steady[0], "one steady shape never alternates"

    fixed = _probe(_FakeServer(leaky=False), mixed)
    ok = sv.keep_dimensions(declared=("capacity",), primary="short_decode", comparisons={},
                            capacity=cp.evaluate(fixed, limit_bytes=62 * GIB, backend="gpu",
                                                 ctx=196608, np=2))
    assert ok["passed"] is True


def test_dimension_records_are_retained(tmp_path):
    record = sv.keep_dimensions(declared=(), primary="short_decode", comparisons={}, capacity=None)
    path = sv.retain_dimensions(tmp_path, "mech/id", record, now=1.0)
    assert path.parent.name == sv.DIMENSIONS_DIR and json.loads(path.read_text())["passed"] is True
