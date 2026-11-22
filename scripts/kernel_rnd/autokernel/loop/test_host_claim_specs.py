"""AKU-11c: build-role and GPU-phase host claims with receipts (fake backend fixture)."""
from __future__ import annotations

import pytest

from . import host_claim_provider as hcp
from . import worker_lifecycle as wl
from .test_host_claim_provider import BOOT, Clock, FakeLockBackend, _request
from .test_worker_lifecycle import MockOwnedContainer


class PhaseBackend(FakeLockBackend):
    """FIXTURE: records the role and adds a gpu-quiet claim for exclusive phases."""

    def __init__(self):
        super().__init__()
        self.specs = []

    def acquire(self, spec, timeout_s):
        self.specs.append(spec)
        handle = super().acquire(spec, timeout_s)
        if spec.gpu_quiet_exclusive:
            handle = hcp.LockHandle(handle.boot_id, handle.holder_pid, handle.holder_start_ticks,
                                    handle.claim_ids + (f"{BOOT}:flock:64768:4242",),
                                    handle.generation, handle.region_fraction,
                                    handle.gpu_device_ids, handle.affinity, handle.token)
            self.acquired[-1] = handle
        return handle


def _provider(tmp_path, backend, clock):
    root = tmp_path / "containers"
    root.mkdir(mode=0o700, exist_ok=True)
    return hcp.HostClaimProvider(backend, specs={
        "cpu": hcp.ClaimSpec("cpu", cpu_list="0-3"),
        "cpu:build": hcp.ClaimSpec("cpu", cpu_list="72-87", role="build", jobs=16),
        "gpu": hcp.ClaimSpec("gpu", device_id="mi210_0", device_lock="/fixture.lock",
                             gpu_quiet_exclusive=True)},
        container_factory=lambda name: MockOwnedContainer(root, name),
        clock=clock, clock_domain=lambda: "linux-monotonic:fixture")


def _stage(request, stage_class):
    binding = request.accounting_binding
    return wl.StageRequest(*[getattr(request, f) for f in request.__dataclass_fields__
                             if f != "accounting_binding"],
                           wl.AccountingBinding(binding.catalog_digest, binding.transition_digest,
                               binding.selection_digest, binding.proposal_digest,
                               binding.proposal_id, binding.backend, stage_class))


def _grant_and_receipt(provider, clock, request, name):
    start = clock()
    auth = provider.authorize(request, name, start + 10)
    assert isinstance(auth, wl.AuthorizedLaunch)
    jobs = provider.build_jobs(auth) if request.accounting_binding.scheduler_stage_class == "build" \
        else None
    assert provider.release(auth, clock() + 5)
    ended = clock()
    receipt = provider.close_held_receipt(
        authorization=auth, request=request, worker_id="w", worker_generation=1,
        container_identity=None, lifecycle_started_at=start, released_at=ended,
        deadline=ended + 5)
    request.accounting_binding.verify_receipt(receipt.receipt)
    return auth, receipt, jobs


def test_build_role_claim_uses_build_role_jobs_cap_and_receipt(tmp_path):
    backend, clock = PhaseBackend(), Clock()
    provider = _provider(tmp_path, backend, clock)
    request = _stage(_request(tmp_path, "cpu"), "build")
    _auth, receipt, jobs = _grant_and_receipt(provider, clock, request, "epyc-autokernel-c7")
    assert backend.specs[-1].cpu_role == "build" and jobs == 16
    assert receipt.receipt.stage_class == "build"
    assert receipt.receipt.affinity_cores == tuple(str(c) for c in range(72, 88))
    search = _stage(_request(tmp_path, "cpu"), "search")
    auth = provider.authorize(search, "epyc-autokernel-c8", clock() + 10)
    assert backend.specs[-1].cpu_role == hcp.CPU_ROLE
    with pytest.raises(hcp.HostClaimRefused, match="build-role"):
        provider.build_jobs(auth)


def test_gpu_phase_claim_holds_gpu_quiet_exclusive_in_its_receipt(tmp_path):
    backend, clock = PhaseBackend(), Clock()
    provider = _provider(tmp_path, backend, clock)
    request = _stage(_request(tmp_path, "gpu"), "search")
    auth, receipt, _ = _grant_and_receipt(provider, clock, request, "epyc-autokernel-c9")
    assert backend.specs[-1].gpu_quiet_exclusive is True
    assert receipt.receipt.gpu_device_ids == ("mi210_0",)
    assert receipt.receipt.physical_claim_ids[-1] == f"{BOOT}:flock:64768:4242"
    assert hcp.parse_grant_id(auth.grant.grant_id)[3] == receipt.receipt.physical_claim_ids


@pytest.mark.parametrize("kwargs", [
    {"backend": "gpu", "device_id": "d", "device_lock": "/l", "role": "build", "jobs": 1},
    {"backend": "cpu", "cpu_list": "0-1", "role": "build", "jobs": 3},
    {"backend": "cpu", "cpu_list": "0-1", "role": "build", "jobs": 0},
    {"backend": "cpu", "cpu_list": "0-1", "jobs": 1},
    {"backend": "cpu", "cpu_list": "0-1", "gpu_quiet_exclusive": True},
    {"backend": "cpu", "cpu_list": "0-1", "role": "other"},
])
def test_claim_spec_refuses_open_role_jobs_or_phase(kwargs):
    with pytest.raises(hcp.HostClaimRefused):
        hcp.ClaimSpec(**kwargs)
