"""Unit tests for the control-vector hook preflight (RC-CVEC-HOOKS)."""

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import eval_cvectors as ev  # noqa: E402


def test_known_no_hook_arch_refused():
    with pytest.raises(ev.ControlVectorNoHookError, match="silently ignored"):
        ev.check_cvec_hook("qwen4exp", "/x/v.gguf")


def test_fake_arch_added_to_set_is_refused(monkeypatch):
    monkeypatch.setattr(ev, "NO_CVEC_HOOK_ARCHS", frozenset({"fakearch"}))
    with pytest.raises(ev.ControlVectorNoHookError):
        ev.check_cvec_hook("FakeArch", "/x/v.gguf")


def test_hooked_arch_passes():
    ev.check_cvec_hook("qwen35", "/x/v.gguf")
    ev.check_cvec_hook("qwen35moe", "/x/v.gguf")


def test_no_vector_never_refuses():
    ev.check_cvec_hook("qwen4exp", None)


def test_unknown_arch_warns(capsys):
    ev.check_cvec_hook(None, "/x/v.gguf")
    assert "UNVERIFIED" in capsys.readouterr().err


def _gguf(path, arch):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    body = s("general.name") + struct.pack("<I", 8) + s("n")
    body += s("general.architecture") + struct.pack("<I", 8) + s(arch)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + body)


def test_read_gguf_arch_and_cli_refusal(tmp_path, monkeypatch):
    g = tmp_path / "m.gguf"
    _gguf(g, "qwen4exp")
    assert ev.read_gguf_arch(str(g)) == "qwen4exp"
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--model-port", "1", "--cvector", "v.gguf", "--model-gguf", str(g)],
    )
    with pytest.raises(SystemExit) as e:
        ev.main()
    assert e.value.code == 2
