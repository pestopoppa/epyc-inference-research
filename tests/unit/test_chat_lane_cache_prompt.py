"""UFH14-B4: the chat lanes put ``cache_prompt`` on the wire, honouring the override.

Before: only /completion sent it; on /v1/chat/completions a caller's
``cache_prompt=False`` was dropped and the server's own default applied.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from src.backends.llama_server import _cache_prompt
from src.model_server import InferenceRequest
from tests.unit.test_thinking_roles_chat_lane import (  # noqa: F401 - fixtures by import
    REGISTRY_CTK,
    _backend,
    _chat_response,
    _clean_features,
    _FakeStream,
    _sse,
    role_config,
)


@pytest.mark.parametrize("override,expected", [(None, True), (True, True), (False, False)])
def test_cache_prompt_helper(override, expected):
    assert _cache_prompt(InferenceRequest(role="r", prompt="x", cache_prompt=override)) is expected


@pytest.mark.parametrize("override,expected", [(None, True), (False, False)])
def test_chat_lane_non_stream_sends_cache_prompt(role_config, override, expected):
    backend = _backend(True)
    captured: dict[str, Any] = {}

    def _post(path, json=None, timeout=None):
        captured["path"], captured["payload"] = path, json
        return _chat_response("OK", None)

    with patch.object(backend.client, "post", side_effect=_post), patch(
        "src.registry.registry_loader.chat_template_kwargs_for_role", return_value=REGISTRY_CTK
    ):
        backend.infer(role_config, InferenceRequest(role="architect_critic", prompt="q",
                                                    n_tokens=8, cache_prompt=override))
    assert captured["path"] == "/v1/chat/completions"
    assert captured["payload"]["cache_prompt"] is expected
    assert "id_slot" not in captured["payload"]


@pytest.mark.parametrize("override,expected", [(None, True), (False, False)])
def test_chat_lane_stream_sends_cache_prompt(role_config, override, expected):
    backend = _backend(True)
    captured: dict[str, Any] = {}
    lines = [_sse({"role": "assistant", "content": "ok"}, "stop"), "data: [DONE]"]

    def _stream(method, path, json=None, timeout=None):
        captured["path"], captured["payload"] = path, json
        return _FakeStream(lines)

    with patch.object(backend.client, "stream", side_effect=_stream), patch(
        "src.registry.registry_loader.chat_template_kwargs_for_role", return_value=REGISTRY_CTK
    ):
        backend.infer_stream_text(role_config, InferenceRequest(role="architect_critic", prompt="q",
                                                               n_tokens=8, cache_prompt=override),
                                  on_chunk=lambda c: None)
    assert captured["path"] == "/v1/chat/completions"
    assert captured["payload"]["cache_prompt"] is expected
