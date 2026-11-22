"""AKU-11 host-claim provider tests. Every lock here is the labelled FakeLockBackend
fixture; no test acquires, inspects or releases a real host claim."""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from . import host_claim_provider as hcp
from . import provider_factories
from . import standalone_inputs as inputs
from . import worker_lifecycle as wl
from .test_standalone_inputs import _document
from .test_worker_lifecycle import Harness, MockOwnedContainer

BOOT = "00000000-0000-4000-8000-000000000000"


class Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


class FakeLockBackend:
    """FIXTURE backend: in-memory claims, labelled fake; not host evidence."""

    label = "fake-lock-backend-fixture"

    def __init__(self) -> None:
        self.busy = False
        self.status = "held"
        self.acquired, self.released, self.timeouts = [], [], []
        self.lock_owners: tuple[int, ...] | None = ()
        self.alive = False
        self.boot = BOOT

    def acquire(self, spec, timeout_s):
        self.timeouts.append(timeout_s)
        if self.busy:
            raise hcp.LockBusy("fixture holder")
        index = len(self.acquired) + 1
        handle = hcp.LockHandle(
            self.boot, 4242, 777, (f"{self.boot}:flock:64768:{9000 + index}",), 1,
            0.25 if spec.backend == "cpu" else 0.0,
            () if spec.backend == "cpu" else (spec.device_id,),
            tuple(str(cpu) for cpu in spec.logical_cpus), token=index)
        self.acquired.append(handle)
        return handle

    def observe(self, handle):
        return self.status

    def release(self, handle):
        self.released.append(handle)
        return True

    def owners(self, claim_ids):
        return self.lock_owners

    def holder_alive(self, pid, start_ticks):
        return self.alive

    def current_boot(self):
        return self.boot


def _provider(tmp_path, backend=None, clock=None):
    root = tmp_path / "containers"
    root.mkdir(mode=0o700, exist_ok=True)
    return hcp.HostClaimProvider(
        backend or FakeLockBackend(),
        specs={"cpu": hcp.ClaimSpec("cpu", cpu_list="0-3"),
               "gpu": hcp.ClaimSpec("gpu", device_id="mi210_0", device_lock="/fixture.lock")},
        container_factory=lambda name: MockOwnedContainer(root, name),
        clock=clock or Clock(), clock_domain=lambda: "linux-monotonic:fixture")


def _binding(backend="cpu"):
    return wl.AccountingBinding("a" * 64, "b" * 64, "c" * 64, "d" * 64,
                                "proposal-fixture", backend, "search")


def _request(tmp_path, backend="cpu", binding=True):
    return wl.StageRequest(
        "request-fixture", "b" * 64, "lineage-fixture", "stage-fixture", "sampling",
        ("/usr/bin/true",), {}, tmp_path, "c" * 64, 5.0, 1.0, 0,
        _binding(backend) if binding else None)


def test_acquire_refresh_release_and_provider_held_receipt(tmp_path):
    backend, clock = FakeLockBackend(), Clock()
    provider = _provider(tmp_path, backend, clock)
    request = _request(tmp_path)
    start = clock()
    auth = provider.authorize(request, "epyc-autokernel-c1", start + 10)
    handle = backend.acquired[0]
    assert isinstance(auth, wl.AuthorizedLaunch)
    assert auth.grant.grant_id == handle.grant_id and auth.grant.generation == 1
    assert hcp.parse_grant_id(auth.grant.grant_id) == (BOOT, 4242, 777, handle.claim_ids)
    assert backend.timeouts == [provider.max_acquire_s]
    refreshed = provider.refresh(auth, clock() + 5)
    assert (refreshed.revoked, refreshed.renewal_ok) == (False, True)
    claim = provider.describe_active_observation_claim(
        authorization=auth, request=request, unit_id="u", process_generation_id="g",
        deadline=clock() + 5)
    body = {key: value for key, value in claim.items() if key != "claim_digest"}
    assert claim["claim_digest"] == wl._digest(body)
    assert claim["held_claim"] == {"logical_cpus": [0, 1, 2, 3], "gpu_devices": []}
    assert provider.release(auth, clock() + 5) is True and backend.released == [handle]
    assert provider.release(auth, clock() + 5) is True  # idempotent for the same grant
    ended = clock()
    receipt = provider.close_held_receipt(
        authorization=auth, request=request, worker_id="worker-1", worker_generation=3,
        container_identity=None, lifecycle_started_at=start, released_at=ended,
        deadline=ended + 5)
    assert isinstance(receipt, wl.TrustedHeldClaimReceipt)
    assert receipt.receipt.physical_claim_ids == handle.claim_ids
    assert (receipt.receipt.ownership_generation, receipt.receipt.allocation_generation) == (3, 1)
    assert (receipt.held_started_at, receipt.held_ended_at) == (start, ended)
    request.accounting_binding.verify_receipt(receipt.receipt)
    with pytest.raises(wl.WaitingAuthority):  # consumed exactly once
        provider.close_held_receipt(
            authorization=auth, request=request, worker_id="worker-1", worker_generation=3,
            container_identity=None, lifecycle_started_at=start, released_at=ended,
            deadline=ended + 5)


def test_denials_bounded_deadline_and_refresh_loss(tmp_path):
    backend, clock = FakeLockBackend(), Clock()
    provider = _provider(tmp_path, backend, clock)
    expired = provider.authorize(_request(tmp_path), "epyc-autokernel-x0", clock() - 1)
    assert isinstance(expired, wl.AuthorizationDenied) and "deadline" in expired.reason
    unbound = provider.authorize(_request(tmp_path, binding=False), "epyc-autokernel-x1",
                                 clock() + 10)
    assert isinstance(unbound, wl.AuthorizationDenied)
    assert backend.timeouts == []  # neither touched the backend
    backend.busy = True
    busy = provider.authorize(_request(tmp_path), "epyc-autokernel-x2", clock() + 10)
    assert isinstance(busy, wl.AuthorizationDenied) and "busy" in busy.reason
    backend.busy = False
    auth = provider.authorize(_request(tmp_path), "epyc-autokernel-x3", clock() + 0.5)
    assert 0 < backend.timeouts[-1] <= 0.5
    with pytest.raises(hcp.HostClaimRefused):
        provider.authorize(_request(tmp_path), "epyc-autokernel-x3", clock() + 10)
    late = provider.refresh(auth, clock() - 1)
    assert (late.revoked, late.renewal_ok) == (False, False)
    backend.status = "lost"
    lost = provider.refresh(auth, clock() + 5)
    assert (lost.revoked, lost.renewal_ok) == (True, False)
    with pytest.raises(wl.WaitingAuthority):
        provider.describe_active_observation_claim(
            authorization=auth, request=_request(tmp_path), unit_id="u",
            process_generation_id="g", deadline=clock() + 5)


def test_receipt_refuses_unreleased_late_or_outside_interval(tmp_path):
    clock = Clock()
    provider = _provider(tmp_path, clock=clock)
    request = _request(tmp_path)
    start = clock()
    auth = provider.authorize(request, "epyc-autokernel-r1", start + 10)
    kwargs = dict(authorization=auth, request=request, worker_id="w", worker_generation=1,
                  container_identity=None)
    with pytest.raises(wl.WaitingAuthority, match="no provider-held"):
        provider.close_held_receipt(**kwargs, lifecycle_started_at=start,
                                    released_at=clock(), deadline=clock() + 5)
    provider.release(auth, clock() + 5)
    with pytest.raises(wl.WaitingAuthority, match="outside"):
        provider.close_held_receipt(**kwargs, lifecycle_started_at=clock(),
                                    released_at=clock(), deadline=clock() + 5)
    with pytest.raises(wl.WaitingAuthority, match="deadline"):
        provider.close_held_receipt(**kwargs, lifecycle_started_at=start,
                                    released_at=clock(), deadline=clock() - 1)


def test_inspect_recovery_present_released_and_unknown_when_unprovable(tmp_path):
    backend, clock = FakeLockBackend(), Clock()
    provider = _provider(tmp_path, backend, clock)
    auth = provider.authorize(_request(tmp_path), "epyc-autokernel-i1", clock() + 10)
    grant = auth.grant
    assert provider.inspect(grant, "epyc-autokernel-i1", clock() + 5).status == "exact"
    assert provider.inspect(grant, "epyc-autokernel-i1", clock() - 1).status == "unknown"
    backend.status = "unavailable"
    assert provider.inspect(grant, "epyc-autokernel-i1", clock() + 5).status == "unknown"
    backend.status = "held"
    provider.release(auth, clock() + 5)
    assert provider.inspect(grant, "epyc-autokernel-i1", clock() + 5).status == "absent_released"

    restarted = _provider(tmp_path, backend, clock)  # fresh incarnation: no in-process state
    def check(): return restarted.inspect(grant, "epyc-autokernel-i1", clock() + 5)
    backend.lock_owners = None
    assert check().status == "unknown"  # kernel lock table unreadable
    backend.lock_owners = (99,)
    assert check().status == "unknown"  # someone holds the inode: release unprovable
    backend.lock_owners, backend.alive = (), True
    assert check().status == "unknown"  # original holder still alive
    backend.alive = False
    assert check().status == "absent_released"
    backend.boot = "11111111-0000-4000-8000-000000000000"
    assert check().status == "absent_released"  # flocks do not survive reboot
    bogus = wl.GrantReceipt("fixture-grant", 1, clock() + 5, "linux-monotonic:fixture")
    assert restarted.inspect(bogus, "epyc-autokernel-i1", clock() + 5).status == "unknown"


def test_inspect_pending_exact_absent_unknown(tmp_path):
    backend, clock = FakeLockBackend(), Clock()
    provider = _provider(tmp_path, backend, clock)

    class Identity:
        container_id = "epyc-autokernel-p1"

    assert provider.inspect_pending(Identity, clock() + 5).status == "unknown"
    auth = provider.authorize(_request(tmp_path), "epyc-autokernel-p1", clock() + 10)
    assert provider.inspect_pending(Identity, clock() + 5).authorization is auth
    provider.release(auth, clock() + 5)
    assert provider.inspect_pending(Identity, clock() + 5).status == "absent"


def test_lifecycle_runs_stage_through_host_claim_provider_with_fake_backend():
    harness = Harness()
    backend = FakeLockBackend()
    provider = hcp.HostClaimProvider(
        backend, specs={"cpu": hcp.ClaimSpec("cpu", cpu_list="0")},
        container_factory=lambda name: MockOwnedContainer(harness.path / "containers", name))
    harness.engine.provider = provider
    try:
        base = harness.request("pass")
        request = wl.StageRequest(**{**base.__dict__, "accounting_binding": _binding()})
        terminal = harness.engine.run_stage(request)
        receipt = harness.engine.trusted_held_claim_receipt(terminal)
        assert terminal.grant_id == backend.acquired[0].grant_id
        assert backend.released == backend.acquired
        assert receipt.physical_claim_ids == backend.acquired[0].claim_ids
    finally:
        harness.close()


def test_off_by_default_registry_stays_unavailable(tmp_path, monkeypatch):
    monkeypatch.delenv(hcp.ACTIVATION_ENV, raising=False)
    assert hcp.activation() == (None, "activation_absent")
    registry = hcp.make_registry()
    assert isinstance(registry, inputs.ProviderRegistry) and registry.get(hcp.PROVIDER_ID) is None
    document = _document(tmp_path)
    document["lifecycle_provider_id"] = hcp.PROVIDER_ID
    document["manifest_digest"] = inputs._digest({
        key: value for key, value in document.items() if key != "manifest_digest"})
    materialized = inputs.materialize(inputs.StartupManifest.from_dict(document))
    report = materialized.preflight(registry)
    assert report["status"] == "unavailable" and report["execution_authorized"] is False
    assert f"lifecycle_provider:{hcp.PROVIDER_ID}:unavailable" in report["missing_prerequisites"]
    installed = inputs.ProviderRegistry({hcp.PROVIDER_ID: inputs.ProviderBinding(
        lifecycle_provider=_provider(tmp_path))})
    assert not any(row.startswith(f"lifecycle_provider:{hcp.PROVIDER_ID}")
                   for row in materialized.preflight(installed)["missing_prerequisites"])
    # The registered entry is exact-source bound and constructs nothing ungated.
    path, name = provider_factories.source(hcp.PROVIDER_ID)
    assert (path.name, name) == ("host_claim_provider.py", "make_registry")
    assert provider_factories.construct(hcp.PROVIDER_ID).get(hcp.PROVIDER_ID) is None


def test_activation_record_is_source_bound_and_owner_only(tmp_path):
    source = hashlib.sha256(open(hcp.__file__, "rb").read()).hexdigest()
    record = {"schema": hcp.ACTIVATION_SCHEMA, "provider_id": hcp.PROVIDER_ID,
              "source_sha256": source, "cpu_list": "0-3", "cgroup_parent": "/sys/fs/cgroup/x"}
    path = tmp_path / "activation.json"
    path.write_text(json.dumps(record))
    os.chmod(path, 0o600)
    assert hcp.activation({hcp.ACTIVATION_ENV: str(path)})[1] == "active"
    path.write_text(json.dumps({**record, "source_sha256": "0" * 64}))
    assert hcp.activation({hcp.ACTIVATION_ENV: str(path)})[1] == "activation_source_differs"
    os.chmod(path, 0o666)
    assert hcp.activation({hcp.ACTIVATION_ENV: str(path)})[1] == "activation_record_untrusted"
    assert hcp.activation({hcp.ACTIVATION_ENV: "relative.json"})[1] != "active"


def test_flock_owner_parser_matches_exclusive_rows_only():
    device = os.makedev(0xfd, 0x02)
    raw = ("1: FLOCK  ADVISORY  WRITE 4242 fd:02:9001 0 EOF\n"
           "2: FLOCK  ADVISORY  READ  5151 fd:02:9001 0 EOF\n"
           "3: POSIX  ADVISORY  WRITE 6161 fd:02:9001 0 EOF\n"
           "4: FLOCK  ADVISORY  WRITE 7171 fd:02:9002 0 EOF\n")
    assert hcp.flock_owners(raw, device, 9001) == (4242,)
    assert hcp.flock_owners(raw, device, 9003) == ()
