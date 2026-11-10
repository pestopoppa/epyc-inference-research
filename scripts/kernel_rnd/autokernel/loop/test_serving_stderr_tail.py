"""A calibration failure names its cause: the failing server's stderr tail rides in ServerDied."""
import os
from unittest import mock

import pytest

from . import residency, serving


def _died_popen(text: bytes):
    def popen(argv, **kwargs):
        kwargs["stderr"].write(text)
        kwargs["stderr"].flush()
        process = mock.Mock(pid=4321, returncode=1)
        process.poll.return_value = 1
        return process
    return popen


def _run_failing_launch(text: bytes):
    from .test_resolved_recipe import BUILD, _resolve
    recipe = serving.Recipe(name="gpu", model="/m", np=1)
    with mock.patch.object(serving, "require_free_port"), \
            mock.patch.object(serving, "ServerPortOwner"), \
            mock.patch.object(serving.subprocess, "Popen", side_effect=_died_popen(text)), \
            mock.patch.object(residency.Sampler, "_loop"):
        serving._measure_once(recipe, BUILD, 18311, evidence=[],
                              resolved_recipe=_resolve(recipe))


def test_calibration_launch_failure_keeps_server_stderr_tail(monkeypatch):
    monkeypatch.delenv("AK_SERVING_STDERR_LOG", raising=False)
    with serving._calibration_stderr_log():
        scratch = os.environ["AK_SERVING_STDERR_LOG"]
        with open(scratch, "ab") as earlier:  # a previous launch's output must not leak in
            earlier.write(b"OLD LAUNCH NOISE\n")
        with pytest.raises(serving.ServerDied) as caught:
            _run_failing_launch(b"srv: failed to parse chat output: expected Content-only\n")
    exc = caught.value
    assert "expected Content-only" in str(exc)
    assert "expected Content-only" in exc.record["server_stderr_tail"]
    assert "OLD LAUNCH NOISE" not in exc.record["server_stderr_tail"]
    assert "AK_SERVING_STDERR_LOG" not in os.environ
    assert not os.path.exists(scratch)


def test_caller_provided_log_is_kept_and_not_replaced(monkeypatch, tmp_path):
    log = tmp_path / "server.log"
    monkeypatch.setenv("AK_SERVING_STDERR_LOG", str(log))
    with serving._calibration_stderr_log():
        assert os.environ["AK_SERVING_STDERR_LOG"] == str(log)
    assert os.environ["AK_SERVING_STDERR_LOG"] == str(log)


def test_launch_without_the_option_is_unchanged(monkeypatch):
    monkeypatch.delenv("AK_SERVING_STDERR_LOG", raising=False)
    with mock.patch.object(serving, "require_free_port"), \
            mock.patch.object(serving, "ServerPortOwner"), \
            mock.patch.object(serving.subprocess, "Popen") as popen, \
            mock.patch.object(residency.Sampler, "_loop"):
        popen.return_value = mock.Mock(pid=1, returncode=1, **{"poll.return_value": 1})
        from .test_resolved_recipe import BUILD, _resolve
        recipe = serving.Recipe(name="gpu", model="/m", np=1)
        with pytest.raises(serving.ServerDied) as caught:
            serving._measure_once(recipe, BUILD, 18311, evidence=[],
                                  resolved_recipe=_resolve(recipe))
    assert "server_stderr_tail" not in caught.value.record
    assert popen.call_args.kwargs["stderr"] is serving.subprocess.DEVNULL
