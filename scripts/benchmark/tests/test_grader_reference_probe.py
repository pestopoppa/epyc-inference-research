import json
import os

import pytest

from scripts.benchmark import grader_reference_probe as probe
from scripts.benchmark.code_exec_scorer import SandboxUnavailable


def _record(**changes):
    result = {"direct": {"opened": False, "errno": 2},
              "proc_root": {"opened": False, "errno": 2},
              "environ_self": {"opened": True, "data": "", "truncated": False},
              "environ_init": {"opened": True, "data": "", "truncated": False},
              "fds": [{"target": "pipe:[5]"}], "sent": False, "affinity": [0]}
    result.update(changes)
    return json.dumps(result)


def test_validated_export_is_narrow(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "_run_once", lambda *a, **kw: (True, _record()))
    result = probe.probe("fixture", "0", str(tmp_path))
    assert result["passed"] and all(result["checks"].values())
    assert set(result) == {"checks", "passed", "scope", "image", "cpuset"}


def test_open_reference_and_truncated_environment_fail(monkeypatch, tmp_path):
    response = _record(direct={"opened": True, "data": "", "truncated": False},
                       environ_self={"opened": True, "data": "", "truncated": True})
    monkeypatch.setattr(probe, "_run_once", lambda *a, **kw: (True, response))
    result = probe.probe("fixture", "0", str(tmp_path))
    assert not result["passed"]
    assert not result["checks"]["direct_reference_denied"]
    assert not result["checks"]["reference_absent_from_environment_and_fds"]


def test_unparseable_export_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "_run_once", lambda *a, **kw: (True, "not json"))
    with pytest.raises(SandboxUnavailable, match="invalid probe export"):
        probe.probe("fixture", "0", str(tmp_path))


@pytest.mark.skipif(not os.getenv("CODEEXEC_TEST_DOCKER_IMAGE") or not os.getenv("CODEEXEC_TEST_CPUSET"),
                    reason="no explicit local image or claimed CPU set")
def test_real_grader_reference_boundary(tmp_path):
    result = probe.probe(os.environ["CODEEXEC_TEST_DOCKER_IMAGE"],
                         os.environ["CODEEXEC_TEST_CPUSET"], str(tmp_path))
    assert result["passed"], result
