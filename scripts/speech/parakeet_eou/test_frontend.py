"""Unit tests for the CS-46 frontend/engine (numpy only; no model weights).
run: python -I -m pytest scripts/speech/parakeet_eou/test_frontend.py  (from repo root, PYTHONPATH=scripts/speech)
"""
import numpy as np
import pytest

from parakeet_eou.engine import ParakeetEOUArm
from parakeet_eou.frontend import (HOP_LENGTH, N_MELS, StreamingMel, hann_symmetric, mel_filterbank,
                                   offline_mel)


def _sig(n, seed=0):
    r = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    return (0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * r.standard_normal(n)).astype(np.float32)


@pytest.mark.parametrize("block", [1, 7, 160, 1280, 2560, 4001])
def test_streaming_equals_offline_any_block_size(block):
    x = _sig(16000 + 123)
    ref = offline_mel(x)
    sm = StreamingMel()
    parts = [sm.push(x[i:i + block]) for i in range(0, len(x), block)] + [sm.flush()]
    got = np.concatenate(parts, axis=0)
    assert ref.shape == (len(x) // HOP_LENGTH, N_MELS)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-6)  # same maths; BLAS summation order differs at ULP level


def test_only_new_frames_computed_per_push():
    sm = StreamingMel()
    x = _sig(3200)
    a = sm.push(x[:1600])
    b = sm.push(x[1600:])
    assert sm.frames_emitted == a.shape[0] + b.shape[0]
    # frame i needs samples up to i*hop+200 => 3200 samples finalise frames 0..(3200-200)//160
    assert sm.frames_emitted == (3200 - 200) // 160 + 1


def test_filterbank_shapes_and_slaney_vs_htk_differ():
    s, h = mel_filterbank("slaney"), mel_filterbank("htk")
    assert s.shape == h.shape == (128, 257)
    assert np.abs(s - h).max() > 1e-4


def test_window_is_symmetric_hann():
    w = hann_symmetric()
    assert w[0] == 0 and abs(w[-1]) < 1e-6 and np.allclose(w, w[::-1], atol=1e-6)


class FakeGraphs:
    """encoder: 1 enc frame per 8 new mel frames; decoder: emits EOU on a scripted frame."""
    encoder_path = decoder_path = "fake"

    def __init__(self, script):
        self.script, self.k = script, 0   # script: {frame_index: [token ids...]} then blank
        self.pending = []

    n_out = 2

    def encode(self, feats_ct, length, cache):
        assert feats_ct.shape == (128, 25) and length == 25
        return np.zeros((1, 512, self.n_out), np.float32), cache

    def decode(self, enc_frame, last, h, c, hist=()):
        out = np.full(1027, -5.0, np.float32)
        if not self.pending:
            self.pending = list(self.script.get(self.k, []))
            self.k += 1
        out[self.pending.pop(0) if self.pending else 1026] = 5.0
        return out, h + 1, c + 1


VOCAB = ["<unk>"] + [f"▁w{i}" for i in range(1023)] + ["<EOU>", "<EOB>"]


def _run(arm, n_samples):
    ev = []
    for i in range(0, n_samples, 1280):
        ev += arm.feed(np.zeros(1280, np.float32))
    return ev + arm.finish()


def test_eou_is_surfaced_and_reset_policies():
    for pol in ("none", "decoder", "full"):
        arm = ParakeetEOUArm(FakeGraphs({3: [5, 1024]}), VOCAB, chunk_ms=160, reset=pol)
        assert arm.chunk_frames == 16
        ev = _run(arm, 16000)
        kinds = [e["kind"] for e in ev]
        assert "eou" in kinds
        assert ("reset" in kinds) == (pol != "none")
        assert [e for e in ev if e["kind"] == "token"][0]["id"] == 5


def test_hold_queues_but_does_not_change_tokens():
    a = ParakeetEOUArm(FakeGraphs({0: [7]}), VOCAB, chunk_ms=160, hold_s=0.0)
    b = ParakeetEOUArm(FakeGraphs({0: [7]}), VOCAB, chunk_ms=160, hold_s=1.0)
    ea, eb = _run(a, 32000), _run(b, 32000)
    assert [e["id"] for e in ea if "id" in e] == [e["id"] for e in eb if "id" in e]
    assert all(e["released_t"] >= 1.0 for e in eb)


def test_nemo_va_frontend_runs_and_matches_chunking():
    arm = ParakeetEOUArm(FakeGraphs({}), VOCAB, chunk_ms=80, frontend="nemo_va")
    arm.g.n_out = 1
    for i in range(10):
        arm.feed(np.zeros(1280, np.float32))
    assert arm.step_no == 10 and len(arm.feat_log) == 10
    assert arm.feat_log[0].shape == (25, 128)


def test_feature_buffer_initial_fill_is_log_mel_zero():
    arm = ParakeetEOUArm(FakeGraphs({}), VOCAB, chunk_ms=160)
    assert float(arm.fbuf.max()) == pytest.approx(-16.635)
