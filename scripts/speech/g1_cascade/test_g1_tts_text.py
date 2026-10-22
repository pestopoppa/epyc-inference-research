"""G1 cascade TTS text preparation (CS-8.L): off = raw text byte-identical, on = orchestrator normalizer."""
import importlib.util
import os
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("g1_cascade_language", HERE / "g1_cascade_language.py")
g1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g1)

NORMALIZER = os.environ.get("G1_TTS_NORMALIZER", g1.TTS_NORMALIZER)


def test_off_is_byte_identical():
    text = "la diga nel 1963, il pH"
    assert g1.tts_text(text, "it", None) is text


@pytest.mark.skipif(not os.path.isfile(NORMALIZER), reason="orchestrator tts_text_normalize.py not present")
def test_on_normalizes_italian_only():
    normalizer = g1.load_tts_normalizer(NORMALIZER)
    assert g1.tts_text("la diga nel 1963, il pH", "it", normalizer) == "la diga nel millenovecentosessantatré, il pi acca"
    english = "the dam in 1963, the pH"
    assert g1.tts_text(english, "en", normalizer) is english


def test_flag_defaults_on():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts-normalize", action=argparse.BooleanOptionalAction, default=True)
    assert ap.parse_args([]).tts_normalize is True
    assert ap.parse_args(["--no-tts-normalize"]).tts_normalize is False
    src = (HERE / "g1_cascade_language.py").read_text()
    assert 'ap.add_argument("--tts-normalize", action=argparse.BooleanOptionalAction, default=True' in src
