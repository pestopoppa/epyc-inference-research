"""Host-isolation for every AutoKernel unit test.

Since 2026-10-04 the CPU/GPU measurement quiet window defaults ON (`run.py`,
`--cpu-measurement-gpu-quiet lock`): a CPU run holds the orchestrator's host-wide
gpu-quiet flock SHARED with its region claim, and a GPU measurement holds it EXCLUSIVE.
A unit test must never take either on the REAL host -- a live GPU bench or CPU
measurement would block it, or it would block them. So:

* `ORCHESTRATOR_TMP_DIR` points into the test's tmp dir, so any real orchestrator
  region or gpu-quiet lock a test reaches is a temporary file;
* `claim.DEVICE_LOCK` points into the test's tmp dir (tests that patch it themselves
  still win: their monkeypatch runs after this one);
* `claim.hold_gpu_quiet_measurement` is a recording no-op (`gpu_quiet_holds`) and
  `claim.gpu_quiet_preflight` a no-op; tests of the window itself inject their own
  `hold`, and `real_gpu_quiet` hands tests the unpatched originals.
"""
from contextlib import contextmanager

import pytest

from .loop import claim

_REAL_GPU_QUIET = {"hold_gpu_quiet_measurement": claim.hold_gpu_quiet_measurement,
                   "gpu_quiet_preflight": claim.gpu_quiet_preflight}


@pytest.fixture(autouse=True)
def _isolate_host_claims(tmp_path_factory):
    # Its OWN MonkeyPatch, not the shared `monkeypatch` fixture: requesting that here
    # would set it up before a test's `tmp_path` and so undo it after tmp_path's
    # teardown, running pytest's cleanup under a test's patched shutil.rmtree.
    root = tmp_path_factory.mktemp("host-claims")
    holds = []

    @contextmanager
    def hold_gpu_quiet_measurement(timeout_s: float = 1.0):
        holds.append("enter")
        try:
            yield {"device_id": "host", "gpu_quiet": "exclusive", "test_double": True}
        finally:
            holds.append("exit")

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ORCHESTRATOR_TMP_DIR", str(root / "orchestrator-locks"))
        patch.setattr(claim, "DEVICE_LOCK", root / "gpu_device.mi210_0.lock")
        patch.setattr(claim, "hold_gpu_quiet_measurement", hold_gpu_quiet_measurement)
        patch.setattr(claim, "gpu_quiet_preflight", lambda: root / "gpu_quiet.lock")
        yield holds


@pytest.fixture
def gpu_quiet_holds(_isolate_host_claims):
    return _isolate_host_claims


@pytest.fixture
def real_gpu_quiet():
    return dict(_REAL_GPU_QUIET)
