"""Original direct HTTP bytes, no model or hardware qualification."""
import json

import pytest

from . import measurement_capture as mc, native_server_response as raw
from . import observation_binding as ob, serving
from .test_planned_serving import _prompts
from .test_runtime_treatment import _fixture


def test_direct_capture_uses_existing_http_and_reopens_original_bytes(tmp_path):
    pair, log, pids = _fixture(tmp_path)
    anchor = pair.anchor
    prompts = _prompts(anchor.template)
    context = {"campaign_id": "original-loop", "epoch": "original-epoch",
               "comparison_id": "original-attempt", "arm": "anchor", "launch_index": 0}
    store = mc.ArtifactStore(tmp_path / "captures")
    try:
        capture = raw.ServerResponseCapture.for_direct(
            store=store, context=context, recipe=anchor, prompts=prompts)
        observations = []
        requests = prompts.requests(tuple(row.prompt_id for row in prompts.prompts), anchor.template)
        value = serving._measure_once(anchor.template, anchor.build_dir, anchor.port,
            resolved_recipe=anchor, frozen_requests=requests,
            observation=observations, response_capture=capture)
        assert value == 20.0
        receipt = observations[0]["server_responses"]
        assert receipt["schema"] == raw.DIRECT_UNIT_SCHEMA
        rows = raw.reopen_direct_unit(receipt, store=store, context=context,
            recipe=anchor, prompts=prompts, expected_pid=observations[0]["process_pid"])
        assert len(rows) == 4 and len(pids.read_text().splitlines()) == 1
        assert sorted(json.loads(line) for line in log.read_text().splitlines()) == sorted(
            body.hex() for _, body in requests for _ in raw.PHASES)
        assert all(row["response"]["stop"] is True for row in rows)
        assert "fence" not in receipt["frame"] and "loaded_instrument" not in receipt["frame"]
        assert observations[0]["residency"]["window_end"] >= observations[0]["residency"]["request_end"]
        with pytest.raises(raw.ServerResponseRefused, match="native server unit schema"):
            raw.reopen_unit(receipt, store=store, expected_frame=capture.frame,
                            expected_requests=requests, expected_pid=observations[0]["process_pid"])
        with pytest.raises(raw.ServerResponseRefused, match="parent frame"):
            raw.reopen_direct_unit(receipt, store=store, context={**context, "arm": "candidate"},
                recipe=anchor, prompts=prompts, expected_pid=observations[0]["process_pid"])
        changed = ob._plain(receipt)
        changed["responses"].reverse()
        with pytest.raises((raw.ServerResponseRefused, mc.CaptureError)):
            raw.reopen_direct_unit(changed, store=store, context=context,
                recipe=anchor, prompts=prompts, expected_pid=observations[0]["process_pid"])
    finally:
        store.close()
