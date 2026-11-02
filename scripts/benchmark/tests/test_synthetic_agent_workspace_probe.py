import json
import os

import pytest

from scripts.benchmark import synthetic_agent_workspace_probe as probe
from scripts.benchmark.code_exec_scorer import SandboxUnavailable


def _export(**overrides):
    row = {"answer": '{"total":11}', "direct_denied": True, "proc_denied": True,
           "write_denied": True, "environ_self": {"opened": True, "truncated": False, "data": ""},
           "environ_init": {"opened": True, "truncated": False, "data": ""},
           "fds": [{"target": "pipe:[5]"}], "sent": False, "affinity": [0],
           "workspace": "/tmp/workspace"}
    row.update(overrides)
    return json.dumps(row)


def test_synthetic_agent_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "_run_once", lambda *args, **kwargs: (True, _export()))
    receipt = probe.run_probe("fixture", "0", str(tmp_path))
    assert receipt["passed"] and all(receipt["checks"].values())
    assert receipt["schema"] == "synthetic-file-agent-isolation.v1"
    assert "not a /chat or OpenCode" in receipt["scope"]


def test_untrusted_result_and_wrong_oracle_cannot_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "_run_once", lambda *args, **kwargs: (True, _export(answer='{"total":10}')))
    assert not probe.run_probe("fixture", "0", str(tmp_path))["passed"]
    monkeypatch.setattr(probe, "_run_once", lambda *args, **kwargs: (True, _export(answer='{"total":11} extra')))
    with pytest.raises(SandboxUnavailable, match="invalid schema"):
        probe.run_probe("fixture", "0", str(tmp_path))


def test_missing_environment_cannot_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "_run_once", lambda *args, **kwargs: (True, _export(environ_self={"opened": False})))
    result = probe.run_probe("fixture", "0", str(tmp_path))
    assert not result["checks"]["env_fds_no_canary"]


@pytest.mark.skipif(not os.getenv("CODEEXEC_TEST_DOCKER_IMAGE") or not os.getenv("CODEEXEC_TEST_CPUSET"),
                    reason="no explicit local image or claimed CPU set")
def test_actual_synthetic_agent_workspace(tmp_path):
    receipt = probe.run_probe(os.environ["CODEEXEC_TEST_DOCKER_IMAGE"],
                              os.environ["CODEEXEC_TEST_CPUSET"], str(tmp_path))
    assert receipt["passed"], receipt
