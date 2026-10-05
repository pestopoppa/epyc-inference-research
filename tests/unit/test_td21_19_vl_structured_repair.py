"""Tests for TD-21.19: VLStructuredAnalyzer structured-extraction repair.

Today (pre-fix): a JSON parse failure on the VL server's structured-extraction
reply left ``structured=None`` + ``parse_error`` set, but ``result.success``
stayed ``True`` — a fail-open silent default that made "nothing structured in
the image" indistinguishable from "formatting drift" downstream
(`src/vision/pipeline.py:213` reads ``struct_result.success`` to decide
whether to populate ``structured_data`` or record an error).

Covers: happy path is fished with zero repair calls; a parse miss costs
exactly one repair turn back to the SAME VL server and recovers a value;
terminal repair failure flips ``success`` to ``False`` with a descriptive
``error`` instead of silently keeping ``success=True``; the
(site, status) counters.
"""

from __future__ import annotations

import json
import urllib.error

import pytest
from PIL import Image

from src.structured_output.repair import STRUCTURED_OUTPUT_REPAIR_COUNTS, reset_counts_for_tests
from src.vision.analyzers import vl_describe


@pytest.fixture(autouse=True)
def _use_temporary_vision_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(vl_describe, "VISION_CACHE_DIR", tmp_path / "vision-cache")


@pytest.fixture(autouse=True)
def _clear_counts():
    reset_counts_for_tests()
    yield
    reset_counts_for_tests()


class _FakeResponse:
    status = 200

    def __init__(self, body: dict):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self._body).encode("utf-8")


def _install_fake_urlopen(monkeypatch, replies: list):
    """`replies`: a list of dicts (successful chat-completion bodies) or
    Exception instances, consumed in call order across BOTH the main VL
    request and any repair turn (both go through the same
    `urllib.request.urlopen` singleton)."""
    calls: list[dict] = []
    it = iter(replies)

    def fake_urlopen(request, timeout=None):
        calls.append(json.loads(request.data.decode("utf-8")))
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)

    monkeypatch.setattr(vl_describe.urllib.request, "urlopen", fake_urlopen)
    return calls


def _chat_reply(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _make_analyzer(tmp_path) -> tuple[vl_describe.VLStructuredAnalyzer, Image.Image]:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (4, 4), color="white").save(image_path)
    analyzer = vl_describe.VLStructuredAnalyzer(backend="server")
    analyzer._initialized = True
    return analyzer, Image.open(image_path)


class TestVLStructuredHappyPath:
    def test_fenced_json_fished_with_zero_repair_calls(self, monkeypatch, tmp_path):
        analyzer, image = _make_analyzer(tmp_path)
        calls = _install_fake_urlopen(
            monkeypatch, [_chat_reply('Here you go:\n```json\n{"total": 42}\n```')]
        )

        result = analyzer.analyze(image)

        assert result.success is True
        assert result.data["structured"] == {"total": 42}
        assert "parse_error" not in result.data
        assert len(calls) == 1  # only the main VL request -- no repair call
        assert STRUCTURED_OUTPUT_REPAIR_COUNTS.get(("vision.vl_structured", "repaired"), 0) == 0
        assert STRUCTURED_OUTPUT_REPAIR_COUNTS.get(("vision.vl_structured", "parsed"), 0) == 1

    def test_main_request_carries_the_open_object_schema(self, monkeypatch, tmp_path):
        analyzer, image = _make_analyzer(tmp_path)
        calls = _install_fake_urlopen(monkeypatch, [_chat_reply('{"total": 42}')])

        analyzer.analyze(image)

        payload = calls[0]
        assert payload["response_format"]["type"] == "json_schema"
        schema = payload["response_format"]["json_schema"]["schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is True


class TestVLStructuredRepair:
    def test_unparseable_prose_costs_one_repair_call_and_recovers(self, monkeypatch, tmp_path):
        analyzer, image = _make_analyzer(tmp_path)
        calls = _install_fake_urlopen(
            monkeypatch,
            [
                _chat_reply("The invoice shows a total of 42 dollars from Acme."),
                _chat_reply('{"total": 42, "vendor": "Acme"}'),
            ],
        )

        result = analyzer.analyze(image)

        assert result.success is True
        assert result.data["structured"] == {"total": 42, "vendor": "Acme"}
        assert len(calls) == 2
        assert STRUCTURED_OUTPUT_REPAIR_COUNTS.get(("vision.vl_structured", "repaired"), 0) == 1

    def test_declined_object_free_form_extraction_still_only_promises_object_shape(self, monkeypatch, tmp_path):
        # Sanity: an empty-but-valid object is a legitimate repaired outcome
        # (this analyzer never invents fields), not a failure.
        analyzer, image = _make_analyzer(tmp_path)
        _install_fake_urlopen(
            monkeypatch,
            [
                _chat_reply("Sorry, I cannot make out any text or fields in this image."),
                _chat_reply("{}"),
            ],
        )

        result = analyzer.analyze(image)

        assert result.success is True
        assert result.data["structured"] == {}


class TestVLStructuredTerminalFailure:
    def test_repair_transport_failure_flips_success_false(self, monkeypatch, tmp_path):
        analyzer, image = _make_analyzer(tmp_path)
        calls = _install_fake_urlopen(
            monkeypatch,
            [
                _chat_reply("No structured content, just some prose about the scene."),
                urllib.error.URLError("connection refused"),
            ],
        )

        result = analyzer.analyze(image)

        assert result.success is False
        assert result.data["structured"] is None
        assert "parse_error" in result.data
        assert "VL structured extraction unparseable" in (result.error or "")
        assert len(calls) == 2
        assert STRUCTURED_OUTPUT_REPAIR_COUNTS.get(("vision.vl_structured", "failed"), 0) == 1

    def test_repair_reply_itself_unparseable_flips_success_false(self, monkeypatch, tmp_path):
        analyzer, image = _make_analyzer(tmp_path)
        _install_fake_urlopen(
            monkeypatch,
            [
                _chat_reply("No structured content at all here."),
                _chat_reply("still not JSON, sorry"),
            ],
        )

        result = analyzer.analyze(image)

        assert result.success is False
        assert result.data["structured"] is None
        assert STRUCTURED_OUTPUT_REPAIR_COUNTS.get(("vision.vl_structured", "failed"), 0) == 1


class TestVLDescribeAndOCRUnaffected:
    """VLDescribeAnalyzer / VLOCRAnalyzer are genuinely free-form: no schema,
    no repair. Byte-identical to pre-TD-21.19 behaviour."""

    def test_describe_sends_no_response_format(self, monkeypatch, tmp_path):
        image_path = tmp_path / "sample.jpg"
        Image.new("RGB", (4, 4), color="white").save(image_path)
        calls = _install_fake_urlopen(monkeypatch, [_chat_reply("A white square.")])

        analyzer = vl_describe.VLDescribeAnalyzer(backend="server", prompt="Describe it.")
        analyzer._initialized = True
        result = analyzer.analyze(Image.open(image_path), image_path)

        assert result.success is True
        assert "response_format" not in calls[0]

    def test_ocr_sends_no_response_format(self, monkeypatch, tmp_path):
        image_path = tmp_path / "sample.jpg"
        Image.new("RGB", (4, 4), color="white").save(image_path)
        calls = _install_fake_urlopen(monkeypatch, [_chat_reply("some text")])

        analyzer = vl_describe.VLOCRAnalyzer(backend="server")
        analyzer._initialized = True
        result = analyzer.analyze(Image.open(image_path), image_path)

        assert result.success is True
        assert "response_format" not in calls[0]
