"""Hardware-free lifecycle for a persistent, already GPU-resident process.

This is deliberately a small orchestration primitive.  It does not spawn a
server, invoke inference, inspect the host's KFD state, or profile anything.
Those effects belong to its injected loader and proof collector.  Keeping the
state machine here pure enough for fake-process tests makes the safety boundary
auditable: a hot result exists only after an exact typed residency proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..execution import inference_window
from . import gpu_load_admission
from .split_runtime_verifier import HotResidencyIdentity


class HotResidencyError(RuntimeError):
    """The requested persistent-residency transition was refused."""


class HotResidencyLost(HotResidencyError):
    """A previously proved process no longer has the exact hot identity."""


class _Releasable(Protocol):
    def release(self) -> None: ...


class _Window(Protocol):
    def acquire(self) -> _Releasable: ...


@dataclass(frozen=True)
class ResidencyProof:
    """A captured, ownership-attributed positive-VRAM residency observation."""

    identity: HotResidencyIdentity
    owned_kfd_pids: tuple[int, ...]
    foreign_kfd_pids: tuple[int, ...]
    vram_bytes: int

    def __post_init__(self) -> None:
        if (not isinstance(self.identity, HotResidencyIdentity)
                or not isinstance(self.owned_kfd_pids, tuple)
                or not self.owned_kfd_pids
                or any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
                       for pid in self.owned_kfd_pids)
                or len(set(self.owned_kfd_pids)) != len(self.owned_kfd_pids)
                or not isinstance(self.foreign_kfd_pids, tuple)
                or any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
                       for pid in self.foreign_kfd_pids)
                or len(set(self.foreign_kfd_pids)) != len(self.foreign_kfd_pids)
                or set(self.owned_kfd_pids) & set(self.foreign_kfd_pids)
                or isinstance(self.vram_bytes, bool) or not isinstance(self.vram_bytes, int)
                or self.vram_bytes <= 0):
            raise HotResidencyError("residency proof is malformed or non-positive")

    def verify(self) -> None:
        if self.foreign_kfd_pids:
            raise HotResidencyError(
                f"foreign KFD ownership prevents hot residency: {self.foreign_kfd_pids}")
        if self.identity.kfd_pid not in self.owned_kfd_pids:
            raise HotResidencyError("residency proof does not own the identity KFD PID")


DecisionValidator = Callable[[gpu_load_admission.AdmissionDecision], None]
ProcessLoader = Callable[[], Any]
ResidencyProber = Callable[[Any], ResidencyProof]
ClaimAcquirer = Callable[[], _Releasable]
class HotResidencySession:
    """One owned live process and its exact, revalidatable residency identity."""

    def __init__(self, runner: "HotResidencyRunner", *, process: Any,
                 claim: _Releasable, identity: HotResidencyIdentity, mode: str) -> None:
        self._runner = runner
        self.process = process
        self._claim = claim
        self.identity = identity
        self.mode = mode
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def validate_hot(self) -> HotResidencyIdentity:
        """Re-prove the same live PID/runtime/maps before a hot measurement.

        A failed revalidation releases this session's owned claim immediately;
        callers must obtain a fresh cold admission rather than reusing a stale
        process or silently treating a reload as hot.
        """
        if self._closed:
            raise HotResidencyLost("hot residency session is already closed")
        try:
            proof = self._runner._prove(self.process)
            if proof.identity != self.identity:
                raise HotResidencyLost("resident process identity changed")
        except BaseException:
            self.close()
            raise
        return self.identity

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            closer = getattr(self.process, "close", None)
            if callable(closer):
                closer()
        finally:
            self._claim.release()
            self._runner._forget(self)

    def __enter__(self) -> "HotResidencySession":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


class HotResidencyRunner:
    """Admit cold loads and turn a proved process into a reusable hot session.

    ``decision_validator`` is intentionally supplied by the sealed deployment
    layer.  The runner refuses to interpret a mutable dictionary or actor prose
    as overlap authority.  A cold-overlap load is allowed only after that
    validator accepts the exact typed decision.
    """

    def __init__(self, *, window: _Window | None = None,
                 decision_validator: DecisionValidator,
                 claim_acquirer: ClaimAcquirer,
                 loader: ProcessLoader, residency_prober: ResidencyProber) -> None:
        if not callable(decision_validator) or not callable(claim_acquirer) \
                or not callable(loader) or not callable(residency_prober):
            raise TypeError("hot residency runner requires callable typed seams")
        self._window = (inference_window.InferenceCallWindow()
                        if window is None else window)
        if not callable(getattr(self._window, "acquire", None)):
            raise TypeError("hot residency window must expose acquire()")
        self._decision_validator = decision_validator
        self._claim_acquirer = claim_acquirer
        self._loader = loader
        self._residency_prober = residency_prober
        self._session: HotResidencySession | None = None

    @property
    def session(self) -> HotResidencySession | None:
        return self._session

    def start(self, decision: gpu_load_admission.AdmissionDecision) -> HotResidencySession:
        """Return a session for a sealed decision, reusing only exact hot state.

        If a requested hot identity is lost, the stale session is closed and a
        fresh *serialized* cold load is used.  Thus a mismatch cannot continue
        as hot or acquire the overlap fast path by accident.
        """
        if not isinstance(decision, gpu_load_admission.AdmissionDecision):
            raise HotResidencyError("hot residency requires a typed admission decision")
        self._decision_validator(decision)
        mode = decision.mode
        if mode not in gpu_load_admission.MODES:
            raise HotResidencyError("admission decision has an unknown mode")

        existing = self._session
        if existing is not None and not existing.closed:
            if mode == "hot_resident" and self._matches_requested_hot(existing, decision):
                try:
                    existing.validate_hot()
                    existing.mode = "hot_resident"
                    return existing
                except HotResidencyLost:
                    # Exact identity loss is a cold admission boundary, never a
                    # permissive retry of the hot path.
                    pass
            existing.close()

        # A hot decision cannot create a hot process.  It can only reuse the
        # exact existing one above; otherwise restart at the safest cold mode.
        cold_mode = mode if mode in {"cold_serialized", "cold_overlap"} else "cold_serialized"
        return self._cold_start(cold_mode)

    def _matches_requested_hot(self, session: HotResidencySession,
                               decision: gpu_load_admission.AdmissionDecision) -> bool:
        request = decision.request
        expected = request.get("expected_hot_identity_sha256")
        return (isinstance(expected, str)
                and expected == session.identity.identity_sha256
                and request.get("model_path") == str(session.identity.model_path)
                and request.get("model_sha256") == session.identity.model_sha256
                and request.get("device_id") == session.identity.device_id)

    def _cold_start(self, mode: str) -> HotResidencySession:
        claim: _Releasable | None = None
        process: Any = None
        window_lease: _Releasable | None = None
        try:
            claim = self._claim_acquirer()
            if not callable(getattr(claim, "release", None)):
                raise HotResidencyError("claim acquirer returned a non-releasable claim")
            if mode == "cold_serialized":
                window_lease = self._window.acquire()
                if not callable(getattr(window_lease, "release", None)):
                    raise HotResidencyError("window returned a non-releasable lease")
            process = self._loader()
            proof = self._prove(process)
            # The lock protects precisely load plus positive residency proof;
            # release it before returning a session usable for hot measurements.
            if window_lease is not None:
                window_lease.release()
                window_lease = None
            session = HotResidencySession(self, process=process, claim=claim,
                                          identity=proof.identity, mode=mode)
            self._session = session
            return session
        except BaseException:
            if window_lease is not None:
                window_lease.release()
            closer = getattr(process, "close", None)
            if callable(closer):
                closer()
            if claim is not None:
                claim.release()
            raise

    def _prove(self, process: Any) -> ResidencyProof:
        alive = getattr(process, "is_alive", None)
        if callable(alive) and not alive():
            raise HotResidencyLost("resident process is not alive")
        proof = self._residency_prober(process)
        if not isinstance(proof, ResidencyProof):
            raise HotResidencyError("residency prober did not return ResidencyProof")
        proof.verify()
        return proof

    def _forget(self, session: HotResidencySession) -> None:
        if self._session is session:
            self._session = None


__all__ = [
    "ClaimAcquirer", "DecisionValidator", "HotResidencyError", "HotResidencyLost",
    "HotResidencyRunner", "HotResidencySession", "ProcessLoader", "ResidencyProof",
    "ResidencyProber",
]
