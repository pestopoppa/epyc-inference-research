#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


SCRIPT = Path(__file__).with_name("check_draft_compatibility.py")


def load_module(monkeypatch):
    fake_gguf = types.ModuleType("gguf")
    fake_gguf.GGUFReader = object
    monkeypatch.setitem(sys.modules, "gguf", fake_gguf)
    spec = importlib.util.spec_from_file_location("check_draft_compatibility_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tokenizer_info(**overrides):
    data = {
        "vocab_size": 248320,
        "bos_token_id": 248044,
        "eos_token_id": 248046,
        "pad_token_id": 248044,
        "tokenizer_model": "gpt2",
        "tokenizer_pre": "qwen35",
        "add_bos": True,
        "add_eos": False,
    }
    data.update(overrides)
    return data


def test_default_mode_warns_but_allows_known_risky_pairings(tmp_path, monkeypatch):
    module = load_module(monkeypatch)
    draft = tmp_path / "draft.gguf"
    target = tmp_path / "target.gguf"
    draft.touch()
    target.touch()

    monkeypatch.setattr(
        module,
        "get_tokenizer_info",
        lambda path: tokenizer_info(vocab_size=151936, tokenizer_pre="qwen2")
        if Path(path).name == "draft.gguf"
        else tokenizer_info(),
    )

    compatible, message = module.check_compatibility(str(draft), str(target), verbose=False)

    assert compatible is True
    assert "COMPATIBLE with warnings" in message
    assert "VOCAB MISMATCH" in message
    assert "TOKENIZER PRE MISMATCH" in message


def test_strict_mode_fails_closed_on_tokenizer_vocab_and_special_mismatch(tmp_path, monkeypatch):
    module = load_module(monkeypatch)
    draft = tmp_path / "draft.gguf"
    target = tmp_path / "target.gguf"
    draft.touch()
    target.touch()

    monkeypatch.setattr(
        module,
        "get_tokenizer_info",
        lambda path: tokenizer_info(
            vocab_size=151936,
            bos_token_id=151643,
            eos_token_id=151645,
            pad_token_id=151643,
            tokenizer_pre="qwen2",
        )
        if Path(path).name == "draft.gguf"
        else tokenizer_info(),
    )

    compatible, message = module.check_compatibility(
        str(draft),
        str(target),
        verbose=False,
        strict=True,
        expected_specials={"bos": 248044, "eos": 248046, "pad": 248044},
    )

    assert compatible is False
    assert "FATAL: TOKENIZER PRE MISMATCH" in message
    assert "FATAL: VOCAB MISMATCH" in message
    assert "FATAL: BOS token mismatch" in message
    assert "FATAL: EOS token mismatch" in message
    assert "FATAL: PAD token mismatch" in message
    assert "BOS token assertion failed for draft" in message


def test_strict_mode_accepts_aligned_qwen35_metadata(tmp_path, monkeypatch):
    module = load_module(monkeypatch)
    draft = tmp_path / "draft.gguf"
    target = tmp_path / "target.gguf"
    draft.touch()
    target.touch()
    monkeypatch.setattr(module, "get_tokenizer_info", lambda _path: tokenizer_info())

    compatible, message = module.check_compatibility(
        str(draft),
        str(target),
        verbose=False,
        strict=True,
        expected_specials={"bos": 248044, "eos": 248046, "pad": 248044},
    )

    assert compatible is True
    assert message == "COMPATIBLE: Vocab sizes match, BOS/EOS tokens match"
