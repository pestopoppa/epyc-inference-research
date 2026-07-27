import importlib.util
import socket
from pathlib import Path

import pytest


HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("spot", HERE / "ff_mtp_equivalence_spotcheck.py")
spot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(spot)


def test_request_is_greedy_and_seeded():
    body = spot.request_body("test")
    assert body["temperature"] == 0.0
    assert body["top_p"] == 1.0
    assert body["top_k"] == 1
    assert body["seed"] == 42
    assert body["enable_thinking"] is False


def test_extract_content_requires_clean_stop():
    assert spot.extract_content({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}) == "ok"
    try:
        spot.extract_content({"choices": [{"message": {"content": "ok"}, "finish_reason": "length"}]})
    except spot.SpotCheckError:
        pass
    else:
        raise AssertionError("truncation must fail closed")


def test_compare_requires_exact_content_and_token_ids():
    row = {"index": 1, "content": "ok", "output_token_ids": [1], "content_sha256": "a"}
    assert spot.compare_captures([row] * len(spot.PROMPTS), [row] * len(spot.PROMPTS))["exact_equivalent"]
    changed = [dict(row) for _ in spot.PROMPTS]
    changed[0]["output_token_ids"] = [2]
    comparison = spot.compare_captures([row] * len(spot.PROMPTS), changed)
    assert not comparison["exact_equivalent"]
    assert comparison["mismatches"][0]["token_ids_equal"] is False


def test_port_is_free_rejects_an_unhealthy_listener(monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        monkeypatch.setattr(spot, "PORT", listener.getsockname()[1])
        assert spot.port_is_free() is False


def test_assert_server_identity_rejects_wrong_props(monkeypatch):
    class Alive:
        returncode = None
        def poll(self):
            return None

    monkeypatch.setattr(spot, "get_json", lambda _path: {"model_path": "/wrong.gguf"})
    with pytest.raises(spot.SpotCheckError, match="model_path mismatch"):
        spot.assert_server_identity(Alive(), "mtp")


def test_validate_runtime_identity_rejects_manifest_hash_drift(monkeypatch):
    monkeypatch.setattr(
        spot,
        "run_checked",
        lambda argv: "production-consolidated-v8" if "symbolic-ref" in argv else spot.V8_HEAD,
    )
    monkeypatch.setattr(spot, "sha256_file", lambda path: spot.V8_BINARY_SHA if path == spot.BIN else "bad")
    with pytest.raises(spot.SpotCheckError, match="manifest hash drift"):
        spot.validate_runtime_identity()


def test_collect_persists_request_and_raw_response_before_parse_failure(tmp_path, monkeypatch):
    class Alive:
        returncode = None
        def poll(self):
            return None

    monkeypatch.setattr(spot, "assert_server_identity", lambda **_kwargs: None)
    monkeypatch.setattr(spot, "post_raw", lambda _path, _body: b"not-json")
    with pytest.raises(spot.SpotCheckError, match="invalid JSON"):
        spot.collect_arm("non_mtp", tmp_path, Alive())
    raw = tmp_path / "non_mtp" / "raw"
    assert (raw / "01.chat.request.json").is_file()
    assert (raw / "01.chat.response.raw.json").read_bytes() == b"not-json"
