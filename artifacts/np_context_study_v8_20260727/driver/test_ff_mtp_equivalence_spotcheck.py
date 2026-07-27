import importlib.util
from pathlib import Path


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
    try:
        spot.compare_captures([row] * len(spot.PROMPTS), changed)
    except spot.SpotCheckError:
        pass
    else:
        raise AssertionError("token mismatch must fail closed")
