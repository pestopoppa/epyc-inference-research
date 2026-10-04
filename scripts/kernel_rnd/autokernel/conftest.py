"""Host-isolation for every AutoKernel unit test.

Since 2026-10-04 the CPU/GPU measurement quiet window defaults ON (`run.py`,
`--cpu-measurement-gpu-quiet q3`): a CPU measurement whose CPU list touches q3 flocks
the MI210 device lock, and a GPU measurement takes the q3 CPU region claim through the
orchestrator's region owner. A unit test must never take either on the REAL host -- a
live CPU measurement window or GPU slot would block it, or it would block them. So:

* `claim.DEVICE_LOCK` points into the test's tmp dir (tests that patch it themselves
  still win: their monkeypatch runs after this one);
* `claim.hold_q3_measurement` is a recording no-op (`q3_measurement_holds`); tests of
  the window itself inject their own `hold`.
"""
from contextlib import contextmanager

import pytest

from .loop import claim


@pytest.fixture(autouse=True)
def _isolate_host_claims(tmp_path_factory):
    # Its OWN MonkeyPatch, not the shared `monkeypatch` fixture: requesting that here
    # would set it up before a test's `tmp_path` and so undo it after tmp_path's
    # teardown, running pytest's cleanup under a test's patched shutil.rmtree.
    root = tmp_path_factory.mktemp("host-claims")
    holds = []

    @contextmanager
    def hold_q3_measurement(timeout_s: float = 1.0):
        holds.append("enter")
        try:
            yield {"device_id": "cpu", "regions": ["q3"], "test_double": True}
        finally:
            holds.append("exit")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(claim, "DEVICE_LOCK", root / "gpu_device.mi210_0.lock")
        patch.setattr(claim, "hold_q3_measurement", hold_q3_measurement)
        yield holds


@pytest.fixture
def q3_measurement_holds(_isolate_host_claims):
    return _isolate_host_claims
