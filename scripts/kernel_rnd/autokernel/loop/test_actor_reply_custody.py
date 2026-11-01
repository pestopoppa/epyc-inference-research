"""Completed replies retain every byte and cannot collide within a second."""
import hashlib
import builtins
import io
import os
import sys
import threading
import json
import re
import pytest
from unittest import mock
from pathlib import Path
import subprocess

from . import actors, actor_metrics
from .. import custody


def test_complete_replies_are_distinct_and_untruncated(tmp_path):
    workspace = tmp_path / "lane"
    workspace.mkdir()
    backend = actors.Backend("codex", "model", "low", "binary")
    first = "begin-" + "x" * (4 * 1024 * 1024) + "-end"
    done = subprocess.CompletedProcess([], 0, stdout=first, stderr="err")
    ref1 = actors._persist_reply(workspace, backend, done)
    ref2 = actors._persist_reply(workspace, backend, done)
    target = tmp_path / actors.ACTOR_REPLY_DIR
    assert ref1["stdout"]["path"] != ref2["stdout"]["path"]
    for ref in (ref1, ref2):
        raw = (target / ref["stdout"]["path"]).read_bytes()
        assert raw == first.encode()
        assert ref["stdout"]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert (target / ref["stderr"]["path"]).read_bytes() == b"err"


def test_fallback_call_record_keeps_actual_reply_paths(tmp_path):
    workspace = tmp_path / "lane"
    workspace.mkdir()
    backend = actors.Backend("codex", "model", "low", "binary")
    reply = actors._persist_reply(workspace, backend,
                                  subprocess.CompletedProcess([], 1, "partial", "error"))
    with mock.patch.object(actors, "_call_record_v1", side_effect=RuntimeError("offline")):
        actors._record_call(workspace, backend, "prompt", returncode=1,
                            wall_s=1, env=None, reply=reply)
    row = json.loads((tmp_path / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG).read_text())
    assert row["reply"] == reply
    assert (tmp_path / actors.ACTOR_REPLY_DIR / row["reply"]["stdout"]["path"]).read_text() == "partial"


@pytest.mark.parametrize("failure_site", ["reply", "record", "badrow"])
def test_io_failure_preserves_named_raw_capture_and_stops(tmp_path, failure_site):
    workspace = tmp_path / "lane"
    workspace.mkdir()
    backend = actors.Backend("codex", "model", "low", "binary")

    def child(*args, **kwargs):
        kwargs["stdout"].write("paid-for reply")
        kwargs["stderr"].write("actor diagnostics")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=None, stderr=None)

    target = {"reply": "_persist_reply", "record": "_record_call",
              "badrow": "_call_record_v1"}[failure_site]
    failure = (mock.patch.object(actors, target, return_value={"reply": None})
               if failure_site == "badrow" else
               mock.patch.object(actors, target, side_effect=OSError("disk refused")))
    with mock.patch.object(actors.subprocess, "run", side_effect=child), failure:
        with pytest.raises(actors.RunAborted, match="raw capture retained") as caught:
            actors._run_agent("prompt", workspace=workspace, backend=backend)
    match = re.search(r"raw capture retained at ([^:]+):", str(caught.value))
    assert match
    capture = Path(match.group(1))
    assert (capture / "actor.stdout").read_text() == "paid-for reply"
    assert (capture / "actor.stderr").read_text() == "actor diagnostics"
    marker = actors.scratch.read_marker("dir", capture)
    assert marker["release_blocked"] == "actor reply awaiting durable disposition"


def test_schema_reply_write_failure_keeps_original_http_bytes(tmp_path):
    from . import actor_passthrough
    workspace = tmp_path / "lane"
    workspace.mkdir()
    backend = actors.Backend("opencode", "provider/model", "low", "binary")
    raw = b'{"choices":[{"message":{"content":"{\"a\":1}"}}]}'
    class Response(io.BytesIO):
        pass
    with mock.patch.object(actors, "_provider_base_url", return_value="http://example.invalid"), \
         mock.patch.object(actor_passthrough, "wire_url", return_value=None), \
         mock.patch("urllib.request.urlopen", return_value=Response(raw)), \
         mock.patch.object(actors, "_persist_reply", side_effect=OSError("store full")):
        with pytest.raises(actors.RunAborted, match="raw capture retained") as caught:
            actors._schema_repair("raw", schema={"type": "object"}, backend=backend,
                                  workspace=workspace)
    capture = Path(re.search(r"raw capture retained at ([^:]+):", str(caught.value)).group(1))
    assert (capture / "actor.stdout").read_bytes() == raw
    assert actors.scratch.read_marker("dir", capture)["release_blocked"]


def test_process_crash_leaves_pre_spawn_fence_and_named_reply(tmp_path):
    workspace = tmp_path / "lane"
    workspace.mkdir()
    code = """
import os, subprocess
from pathlib import Path
from unittest import mock
from autokernel.loop import actors

def child(*args, **kwargs):
    output = kwargs["stdout"]
    output.write("finished reply")
    output.flush()
    os.fsync(output.fileno())
    os._exit(17)

backend = actors.Backend("codex", "model", "low", "binary")
with mock.patch.object(actors.subprocess, "run", side_effect=child):
    actors._run_agent("prompt", workspace=Path(os.environ["TEST_ACTOR_LANE"]), backend=backend)
"""
    done = subprocess.run([sys.executable, "-c", code],
                          env={**os.environ, "TEST_ACTOR_LANE": str(workspace)},
                          capture_output=True, text=True)
    assert done.returncode == 17, done.stderr
    captures = list(tmp_path.rglob("actor.stdout"))
    assert len(captures) == 1
    capture = captures[0].parent
    assert captures[0].read_text() == "finished reply"
    assert actors.scratch.read_marker("dir", capture)["release_blocked"]



def test_non_utf8_child_bytes_are_retained_without_changing_text_api(tmp_path):
    workspace = tmp_path / "lane"
    workspace.mkdir()
    backend = actors.Backend("codex", "model", "low", "binary")
    raw = b"begin-\xff-end"

    def child(*args, **kwargs):
        kwargs["stdout"].buffer.write(raw)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=None, stderr=None)

    with mock.patch.object(actors.subprocess, "run", side_effect=child):
        result = actors._run_agent("prompt", workspace=workspace, backend=backend)
    assert result == raw.decode("utf-8", "replace")
    target = tmp_path / actors.ACTOR_REPLY_DIR
    rows = [json.loads(line) for line in (target / actors.ACTOR_CALL_LOG).read_text().splitlines()]
    ref = rows[-1]["reply"]["stdout"]
    assert (target / ref["path"]).read_bytes() == raw
    assert ref["sha256"] == hashlib.sha256(raw).hexdigest()



def test_unrouted_schema_repair_records_reason_and_retained_streams(tmp_path):
    from . import actor_passthrough
    workspace = tmp_path / "lane"
    workspace.mkdir()
    backend = actors.Backend("opencode", "local/model", "low", "binary")
    with mock.patch.object(actors, "_provider_base_url", return_value="http://example.invalid"), \
         mock.patch.object(actor_passthrough, "wire_url",
                           side_effect=actor_passthrough.UnroutedLocalProvider("no route")):
        assert actors._schema_repair("raw", schema={"type": "object"}, backend=backend,
                                     workspace=workspace) is None
    root = tmp_path / actors.ACTOR_REPLY_DIR
    row = json.loads((root / actors.ACTOR_CALL_LOG).read_text().splitlines()[-1])
    assert row["returncode"] == -2
    assert (root / row["reply"]["stderr"]["path"]).read_text() == "no route"



@pytest.mark.parametrize("store_failure", [False, True])
def test_schema_http_error_preserves_exact_status_and_body(tmp_path, store_failure):
    import contextlib
    import urllib.error
    from . import actor_passthrough
    workspace = tmp_path / "lane"
    workspace.mkdir()
    backend = actors.Backend("opencode", "provider/model", "low", "binary")
    body = b"rate-limited-\xff-actual-body"
    http_error = urllib.error.HTTPError("http://example.invalid", 429, "Too Many Requests",
                                        {}, io.BytesIO(body))
    failure = (mock.patch.object(actors, "_persist_reply", side_effect=OSError("store full"))
               if store_failure else contextlib.nullcontext())
    with mock.patch.object(actors, "_provider_base_url", return_value="http://example.invalid"), \
         mock.patch.object(actor_passthrough, "wire_url", return_value=None), \
         mock.patch("urllib.request.urlopen", side_effect=http_error), failure:
        if store_failure:
            with pytest.raises(actors.RunAborted, match="raw capture retained") as caught:
                actors._schema_repair("raw", schema={"type": "object"},
                                      backend=backend, workspace=workspace)
        else:
            assert actors._schema_repair("raw", schema={"type": "object"},
                                         backend=backend, workspace=workspace) is None
    if store_failure:
        capture = Path(re.search(r"raw capture retained at ([^:]+):", str(caught.value)).group(1))
        assert (capture / "actor.stdout").read_bytes() == body
        assert (capture / "actor.stderr").read_text() == "HTTPError status 429: Too Many Requests"
        assert actors.scratch.read_marker("dir", capture)["release_blocked"]
    else:
        root = tmp_path / actors.ACTOR_REPLY_DIR
        row = json.loads((root / actors.ACTOR_CALL_LOG).read_text().splitlines()[-1])
        assert row["returncode"] == 429
        assert (root / row["reply"]["stdout"]["path"]).read_bytes() == body
        assert (root / row["reply"]["stderr"]["path"]).read_text() == "HTTPError status 429: Too Many Requests"
        assert row["reply"]["stdout"]["sha256"] == hashlib.sha256(body).hexdigest()



def test_schema_http_error_capture_write_failure_keeps_prefix_and_status(tmp_path):
    import urllib.error
    from . import actor_passthrough
    workspace = tmp_path / "lane"
    workspace.mkdir()
    backend = actors.Backend("opencode", "provider/model", "low", "binary")
    body = b"a" * 65536 + b"\xfftail"
    http_error = urllib.error.HTTPError("http://example.invalid", 503, "Unavailable",
                                        {}, io.BytesIO(body))
    original_open = Path.open

    class FailingOutput:
        def __init__(self, file):
            self.file = file
            self.writes = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.file.__exit__(*args)
        def __getattr__(self, key):
            return getattr(self.file, key)
        def write(self, data):
            self.writes += 1
            if self.writes == 2:
                raise OSError("capture disk full")
            return self.file.write(data)

    def open_with_failure(path, *args, **kwargs):
        opened = original_open(path, *args, **kwargs)
        return FailingOutput(opened) if path.name == "actor.stdout" and args == ("wb",) else opened

    with mock.patch.object(actors, "_provider_base_url", return_value="http://example.invalid"), \
         mock.patch.object(actor_passthrough, "wire_url", return_value=None), \
         mock.patch("urllib.request.urlopen", side_effect=http_error), \
         mock.patch.object(Path, "open", new=open_with_failure):
        with pytest.raises(actors.RunAborted, match="raw capture retained") as caught:
            actors._schema_repair("raw", schema={"type": "object"},
                                  backend=backend, workspace=workspace)
    capture = Path(re.search(r"raw capture retained at ([^:]+):", str(caught.value)).group(1))
    assert (capture / "actor.stdout").read_bytes() == body[:65536]
    assert (capture / "actor.stderr").read_text() == "HTTPError status 503: Unavailable"
    assert actors.scratch.read_marker("dir", capture)["release_blocked"]


@pytest.mark.parametrize("rollback_fails,complete_write", [(False, False), (True, False), (True, True)])
def test_partial_transfer_then_real_actor_writer_preserves_records(tmp_path, rollback_fails, complete_write):
    workspace = tmp_path / "lane"
    workspace.mkdir()
    target = tmp_path / actors.ACTOR_REPLY_DIR
    target.mkdir()
    log = target / actors.ACTOR_CALL_LOG
    previous = b'{"schema":"existing","value":1}\n'
    raw = b'{"schema":"panel","value":2}\n'
    log.write_bytes(previous)
    source = tmp_path / "member.jsonl"
    source.write_bytes(raw)
    original_open = builtins.open
    class Partial:
        def __init__(self, *args, **kwargs):
            self.file = original_open(*args, **kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.file.close()
        def write(self, data):
            self.file.write(data if complete_write else data[:8])
            self.file.flush()
            raise OSError("partial append")
    def fault(file, mode="r", *args, **kwargs):
        if Path(file) == log and mode == "a":
            return Partial(file, mode, *args, **kwargs)
        if rollback_fails and Path(file) == log and mode == "r+b":
            raise OSError("rollback refused")
        return original_open(file, mode, *args, **kwargs)
    with mock.patch("builtins.open", side_effect=fault):
        with pytest.raises(OSError, match="rollback failed" if rollback_fails else "partial append"):
            custody.transfer_tail(source, log, 0, raw.decode())
    assert source.read_bytes() == raw
    written = raw if complete_write else raw[:8]
    assert log.read_bytes() == (previous + written if rollback_fails else previous)
    backend = actors.Backend("codex", "model", "low", "binary")
    reply = actors._persist_reply(workspace, backend,
                                 subprocess.CompletedProcess([], 0, "actual reply", ""))
    state = {"recorded": False}
    token = actors._reply_custody.set(state)
    try:
        with mock.patch.object(actors, "_call_record_v1", side_effect=RuntimeError("offline")):
            if rollback_fails:
                with pytest.raises(actors.RunAborted, match="File exists"):
                    actors._record_call(workspace, backend, "prompt", returncode=0,
                                        wall_s=1, env=None, reply=reply)
                assert not state["recorded"]
                assert log.read_bytes() == previous + written
            else:
                actors._record_call(workspace, backend, "prompt", returncode=0,
                                    wall_s=1, env=None, reply=reply)
                assert state["recorded"]
                rows = [json.loads(line) for line in log.read_bytes().splitlines()]
                assert rows[0] == {"schema": "existing", "value": 1}
                assert rows[1]["reply"] == reply
    finally:
        actors._reply_custody.reset(token)
    assert source.read_bytes() == raw


def test_corrupt_terminated_tail_refuses_actor_record(tmp_path):
    workspace = tmp_path / "lane"
    workspace.mkdir()
    backend = actors.Backend("codex", "model", "low", "binary")
    reply = actors._persist_reply(workspace, backend,
                                 subprocess.CompletedProcess([], 0, "reply", ""))
    log = tmp_path / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG
    raw = b'{"valid":1}\nnot-json\n'
    log.write_bytes(raw)
    state = {"recorded": False}
    token = actors._reply_custody.set(state)
    try:
        with pytest.raises(actors.RunAborted, match="corrupt row"):
            actors._record_call(workspace, backend, "prompt", returncode=0,
                                wall_s=1, env=None, reply=reply)
        assert not state["recorded"]
        assert log.read_bytes() == raw
    finally:
        actors._reply_custody.reset(token)


def test_actor_fence_stays_held_when_prior_tail_is_corrupt(tmp_path):
    workspace = tmp_path / "lane"
    workspace.mkdir()
    target = tmp_path / actors.ACTOR_REPLY_DIR
    target.mkdir()
    log = target / actors.ACTOR_CALL_LOG
    log.write_bytes(b'{"complete":true}\n{"unfinished"')
    backend = actors.Backend("codex", "model", "low", "binary")
    def child(*args, **kwargs):
        kwargs["stdout"].write("paid reply")
        return subprocess.CompletedProcess([], 0, stdout=None, stderr=None)
    with mock.patch.object(actors.subprocess, "run", side_effect=child):
        with pytest.raises(actors.RunAborted, match="raw capture retained") as caught:
            actors._run_agent("prompt", workspace=workspace, backend=backend)
    capture = Path(re.search(r"raw capture retained at ([^:]+):", str(caught.value)).group(1))
    assert (capture / "actor.stdout").read_text() == "paid reply"
    assert actors.scratch.read_marker("dir", capture)["release_blocked"]
    assert log.read_bytes() == b'{"complete":true}\n{"unfinished"'


@pytest.mark.parametrize("writer", [actor_metrics.record_report_source,
                                    actor_metrics.record_salvage_turn,
                                    actor_metrics.record_answer_protocol])
def test_sibling_writers_do_not_extend_corrupt_tail(tmp_path, writer):
    raw = b'{"complete":1}\n{"unfinished"'
    path = tmp_path / actor_metrics.CALL_LOG_NAME
    path.write_bytes(raw)
    writer(tmp_path, {"reason": "test"})
    assert path.read_bytes() == raw


def test_append_fence_precedes_offset_read_for_concurrent_writers(tmp_path):
    log = tmp_path / "actor-calls.jsonl"
    log.write_bytes(b'{"prior":1}\n')
    entered, release = threading.Event(), threading.Event()
    original = Path.read_bytes
    errors = []
    def blocking_read(path):
        if path == log and threading.current_thread().name == "first-writer":
            entered.set()
            assert release.wait(5)
        return original(path)
    def first():
        try:
            custody.append_rows(log, '{"first":1}\n')
        except Exception as exc:
            errors.append(exc)
    with mock.patch.object(Path, "read_bytes", blocking_read):
        thread = threading.Thread(target=first, name="first-writer")
        thread.start()
        try:
            assert entered.wait(5)
            with pytest.raises(FileExistsError):
                custody.append_rows(log, '{"second":1}\n')
        finally:
            release.set()
            thread.join(5)
    assert not thread.is_alive()
    assert not errors
    assert [json.loads(line) for line in log.read_bytes().splitlines()] == [
        {"prior": 1}, {"first": 1}]
