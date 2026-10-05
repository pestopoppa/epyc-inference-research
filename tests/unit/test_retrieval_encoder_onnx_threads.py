"""Both KB-RAG ONNX encoders must bound the ORT intra-op pool.

ORIGIN. `src/tools/web/colbert_reranker.py` was fixed on 2026-08-12 (19111369)
after ONNX Runtime's default pool — one thread per visible core, 192 here — was
measured costing 1.8-2.3x on small forward passes. The two `src/retrieval/`
encoders build their sessions the same unbounded way and are **live by default in
KB-RAG**, so they carried the same penalty on the request path.

THE POINT OF THIS FILE, and why the two defaults differ. The obvious move was to
copy the reranker's 8 into both. Measured, that is wrong: the optimum is
CALL-SHAPE dependent.

    colbert_encoder.encode()    -> ONE single-row pass.  best 4t, 8t within 3%.
    cross_encoder.score_pairs() -> N rows in ONE run().  best 16t at BOTH batch
                                   10 and 50; 8t costs ~40% at batch=50.

So `cross_encoder` deliberately uses 16 while its siblings use 8, and
`test_the_two_defaults_are_not_silently_unified` exists to stop a future tidy-up
from "harmonising" them back to one number. That would look like cleanup and cost
40% on the batched path, silently, with every test still green.

WHAT IS PINNED. Behaviour, not spelling: SessionOptions reaches the real
InferenceSession call with the exact measured default (8 for single-row ColBERT,
16 for batched cross-encoder), `inter_op_num_threads=1`, and the CPU provider.
The default must remain identical when the visible CPU count is mocked to 1, 8,
16, or 192; this is a deterministic configuration contract, not a claim about
host topology or performance. The separate benchmark records the production
topology and call-shape measurements.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

MODULES = (
    ("src.retrieval.colbert_encoder", "COLBERT_ENCODE_ONNX_THREADS", 8),
    ("src.retrieval.cross_encoder", "CROSS_ENCODER_ONNX_THREADS", 16),
)


class _FakeSessionOptions:
    def __init__(self) -> None:
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0


def _load_with_stubs(monkeypatch, module_name: str, model_dir):
    """Import the module fresh and drive its loader with a stubbed ORT.

    Captures the kwargs the module actually passes to InferenceSession.
    """
    captured: dict = {"sess_options": None, "providers": None}

    class _FakeSession:
        def __init__(self, path, sess_options=None, providers=None, **kw):
            captured["sess_options"] = sess_options
            captured["providers"] = providers

        def get_inputs(self):
            return [types.SimpleNamespace(name="input_ids"),
                    types.SimpleNamespace(name="attention_mask")]

        def run(self, *a, **k):  # pragma: no cover - not exercised here
            raise AssertionError("not expected in this test")

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.SessionOptions = _FakeSessionOptions
    fake_ort.InferenceSession = _FakeSession

    class _FakeTokenizer:
        @staticmethod
        def from_file(path):
            return types.SimpleNamespace(
                enable_truncation=lambda **k: None,
                enable_padding=lambda **k: None,
                # colbert_encoder resolves the trained [Q]/[D] prefix token ids
                # at load time; a stub lacking this reads as a model that has
                # no prefix tokens, which is a different test.
                token_to_id=lambda token: 50368,
            )

    fake_tok = types.ModuleType("tokenizers")
    fake_tok.Tokenizer = _FakeTokenizer

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tok)
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    (model_dir / "model_int8.onnx").write_bytes(b"")
    (model_dir / "tokenizer.json").write_text("{}")
    monkeypatch.setenv("LATEON_MODEL_PATH", str(model_dir))
    monkeypatch.setenv("KB_RAG_CROSS_ENCODER_PATH", str(model_dir))

    mod = __import__(module_name, fromlist=["ensure_loaded"])
    mod._session = None
    mod._tokenizer = None
    assert mod.ensure_loaded() is True, f"{module_name}.ensure_loaded() failed under stubs"
    return mod, captured


@pytest.mark.parametrize("module_name,env_var,expected", MODULES)
@pytest.mark.parametrize("visible_cpu_count", [1, 8, 16, 192])
def test_encoder_uses_deterministic_intra_op_default(
    monkeypatch, tmp_path, module_name, env_var, expected, visible_cpu_count
):
    """The measured defaults do not vary with the host running the test."""
    monkeypatch.setattr(os, "cpu_count", lambda: visible_cpu_count)
    monkeypatch.delenv(env_var, raising=False)
    mod, captured = _load_with_stubs(monkeypatch, module_name, tmp_path)
    opts = captured["sess_options"]

    assert opts is not None, (
        f"{module_name} built InferenceSession WITHOUT SessionOptions, so ORT uses "
        f"its default pool — one thread per visible core ({os.cpu_count()} here). "
        f"Measured 1.45-2.03x slower, and it is live on the KB-RAG request path."
    )
    assert opts.intra_op_num_threads == mod._DEFAULT_ONNX_THREADS == expected, (
        f"{module_name} default changed from {expected}. If that was deliberate, "
        f"re-measure — do not copy a sibling's value, the optimum is call-shape "
        f"dependent (see the module comment)."
    )
    assert opts.intra_op_num_threads > 0
    assert opts.inter_op_num_threads == 1
    assert captured["providers"] == ["CPUExecutionProvider"]


@pytest.mark.parametrize("module_name,env_var,_expected", MODULES)
def test_env_override_reaches_session_options(monkeypatch, tmp_path, module_name, env_var, _expected):
    """The escape hatch must actually be wired, not just documented."""
    monkeypatch.setenv(env_var, "3")
    _mod, captured = _load_with_stubs(monkeypatch, module_name, tmp_path)
    assert captured["sess_options"].intra_op_num_threads == 3


@pytest.mark.parametrize("module_name,env_var,_expected", MODULES)
def test_bad_env_value_falls_back_instead_of_crashing(monkeypatch, tmp_path, module_name, env_var, _expected):
    """A typo'd env var must not take down the retrieval path."""
    monkeypatch.setenv(env_var, "not-a-number")
    mod, captured = _load_with_stubs(monkeypatch, module_name, tmp_path)
    assert captured["sess_options"].intra_op_num_threads == mod._DEFAULT_ONNX_THREADS


def test_the_two_defaults_are_not_silently_unified(monkeypatch, tmp_path):
    """Guards the REASON the numbers differ, which is the part that looks like a bug.

    A single-row consumer and a batched one have different optima. Someone tidying
    these into one constant would be making a 40%-at-batch-50 regression that no
    other test in this repo would notice.
    """
    from src.retrieval import colbert_encoder, cross_encoder

    assert colbert_encoder._DEFAULT_ONNX_THREADS != cross_encoder._DEFAULT_ONNX_THREADS, (
        "The single-row and batched encoders now share a thread default. These were "
        "measured separately on 2026-08-12: single-row is best near 8, batched is "
        "best at 16 and loses ~40% at batch=50 when forced to 8. If you unified "
        "them, re-measure both shapes first and update the module comments."
    )
    assert cross_encoder._DEFAULT_ONNX_THREADS > colbert_encoder._DEFAULT_ONNX_THREADS, (
        "The batched encoder should carry the LARGER bound — it has more parallel "
        "work per run() to amortise coordination over."
    )
