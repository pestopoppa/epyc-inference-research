"""Unit tests never write the LIVE production inference tap.

`tests/unit/conftest.py::_hermetic_inference_tap` turns the tap off for every
unit test. Before it existed, `test_edit_transaction.py`'s concurrent
`_real_call` test (and others driving the real `_real_call` with mocked
backends) appended `ROLE=role_a`/`role_b` sections to
`/mnt/raid0/llm/tmp/inference_tap.log` because the on-disk sentinel
`.inference_tap_active` is left in place on this host.

The positive control proves the same call path DOES write a tap when one is
enabled, so the negative assertions are not vacuous.
"""

from __future__ import annotations

import builtins
from pathlib import Path
from unittest.mock import Mock

import src.runtime.inference_tap as tap_mod
from src.llm_primitives import LLMPrimitives
from src.model_server import InferenceResult

_PRODUCTION_TMP = Path("/mnt/raid0/llm/tmp")
_PRODUCTION_PATHS = {
    str(_PRODUCTION_TMP / ".inference_tap_active"),
    str(_PRODUCTION_TMP / "inference_tap.log"),
    str(_PRODUCTION_TMP / "inference_tap_events.jsonl"),
    str(_PRODUCTION_TMP / "inference_tap_events.jsonl.lock"),
}


def _production_sentinel() -> str:
    """The sentinel path the unpatched module would resolve on this host."""
    try:
        from src.config import get_config

        return str(get_config().paths.tmp_dir / ".inference_tap_active")
    except Exception:
        return str(_PRODUCTION_TMP / ".inference_tap_active")


def _prims() -> tuple[LLMPrimitives, Mock]:
    prims = LLMPrimitives(
        mock_mode=False,
        server_urls={"role_a": "http://localhost:9001"},
    )
    backend = Mock(spec=[])
    backend.infer = Mock(return_value=InferenceResult(
        role="role_a", output="tap isolation response", tokens_generated=3,
        generation_speed=1.0, elapsed_time=0.001, success=True, prompt_eval_ms=0.1,
        generation_ms=0.1, http_overhead_ms=0.0, completion_reason="stop",
    ))
    prims._backends["role_a"] = backend
    return prims, backend


def _spy_tap_io(monkeypatch) -> tuple[list[str], list[str]]:
    """Record every open() made by the tap module and every TapWriter path."""
    opened: list[str] = []
    writers: list[str] = []
    real_open = builtins.open

    def _open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    real_writer = tap_mod.TapWriter

    class _SpyWriter(real_writer):  # type: ignore[misc, valid-type]
        def __init__(self, path, metadata=None):
            writers.append(str(path))
            super().__init__(path, metadata=metadata)

    monkeypatch.setattr(tap_mod, "open", _open, raising=False)
    monkeypatch.setattr(tap_mod, "TapWriter", _SpyWriter)
    return opened, writers


def test_fixture_turns_the_tap_off():
    assert tap_mod.is_active() is False
    assert tap_mod._tap_path() == ""
    assert tap_mod._structured_event_path() == ""
    assert tap_mod._get_sentinel() != _production_sentinel()
    assert tap_mod._get_sentinel() not in _PRODUCTION_PATHS


def test_unit_real_call_never_touches_the_production_tap(monkeypatch):
    opened, writers = _spy_tap_io(monkeypatch)
    prims, backend = _prims()

    prims._real_call("tap isolation prompt", "role_a", n_tokens=8)

    assert backend.infer.called, "the call never reached the backend; nothing was tested"
    assert writers == []
    touched = set(opened) & (_PRODUCTION_PATHS | {_production_sentinel()})
    assert touched == set(), f"unit test touched the live tap: {sorted(touched)}"
    assert not any(p.startswith(str(_PRODUCTION_TMP)) for p in opened), opened


def test_positive_control_a_sentinel_enables_the_tap_on_the_same_call(tmp_path, monkeypatch):
    """The leak vector: an on-disk sentinel naming a tap file, no env var at all."""
    tap_file = tmp_path / "inference_tap.log"
    sentinel = tmp_path / ".inference_tap_active"
    sentinel.write_text(str(tap_file))
    monkeypatch.setattr(tap_mod, "_SENTINEL", str(sentinel))
    monkeypatch.setattr(tap_mod, "_sentinel_cache", ("", 0.0))
    opened, writers = _spy_tap_io(monkeypatch)
    prims, backend = _prims()

    prims._real_call("tap isolation prompt", "role_a", n_tokens=8)

    assert backend.infer.called
    assert writers == [str(tap_file)]
    text = tap_file.read_text()
    assert "ROLE=role_a" in text and "tap isolation prompt" in text
    assert (tmp_path / "inference_tap_events.jsonl").exists()
    assert not any(p.startswith(str(_PRODUCTION_TMP)) for p in opened), opened
