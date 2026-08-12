"""Fresh source-prerequisite production inside one held AutoKernel campaign.

This is deliberately an adapter over already-owned capabilities.  It never
acquires a CPU region or GPU device: the campaign supplies the claim objects it
already holds and the out-of-tree ``test-backend-ops`` it just built.  A durable
intent precedes the first invocation; after a crash, a complete package resumes
without execution while an incomplete intent refuses automatic duplication.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from . import schemas, source_prerequisite_package as package_module, storage
from .execution import t0_provider
from .resource import device_claim


SCHEMA = "epyc.autokernel.fresh-source-prerequisite-plan.v1"
PRODUCER_ID = "autokernel.fresh-source-prerequisite-producer/v1"
_FIELDS = frozenset((
    "schema", "campaign_id", "proposal_id", "candidate_id", "suite_seeds",
    "oracle_seed", "backend_filter", "ops", "params_filter", "timeout_s",
    "capture_mode", "plan_sha256",
))


class FreshSourcePrerequisiteError(RuntimeError):
    """Fresh evidence cannot be produced or resumed without ambiguity."""


def _plan_sha256(value: Mapping[str, Any]) -> str:
    return schemas.content_hash({key: value[key] for key in sorted(value)
                                 if key != "plan_sha256"})


@dataclass(frozen=True)
class FreshSourcePrerequisitePlan:
    campaign_id: str
    proposal_id: str
    candidate_id: str
    suite_seeds: tuple[int, ...]
    oracle_seed: int
    backend_filter: str
    ops: tuple[str, ...]
    params_filter: Optional[str]
    timeout_s: float
    capture_mode: str
    plan_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "campaign_id": self.campaign_id,
            "proposal_id": self.proposal_id, "candidate_id": self.candidate_id,
            "suite_seeds": list(self.suite_seeds), "oracle_seed": self.oracle_seed,
            "backend_filter": self.backend_filter, "ops": list(self.ops),
            "params_filter": self.params_filter, "timeout_s": self.timeout_s,
            "capture_mode": self.capture_mode, "plan_sha256": self.plan_sha256,
        }

    @classmethod
    def from_mapping(cls, raw: Any) -> "FreshSourcePrerequisitePlan":
        if not isinstance(raw, Mapping) or set(raw) != _FIELDS:
            got = sorted(raw) if isinstance(raw, Mapping) else type(raw).__name__
            raise FreshSourcePrerequisiteError(
                f"fresh plan fields must be exactly {sorted(_FIELDS)}; got {got}")
        if raw["schema"] != SCHEMA:
            raise FreshSourcePrerequisiteError(f"fresh plan schema must be {SCHEMA!r}")
        for name, prefix in (("campaign_id", "ak-"), ("proposal_id", "akp-"),
                             ("candidate_id", "akc-")):
            if not isinstance(raw[name], str) or not raw[name].startswith(prefix):
                raise FreshSourcePrerequisiteError(f"{name} must start with {prefix!r}")
        seeds = raw["suite_seeds"]
        if (not isinstance(seeds, list) or len(seeds) < 3
                or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                       for seed in seeds) or len(seeds) != len(set(seeds))):
            raise FreshSourcePrerequisiteError(
                "suite_seeds must contain at least three distinct non-negative integers")
        oracle_seed = raw["oracle_seed"]
        if isinstance(oracle_seed, bool) or not isinstance(oracle_seed, int) \
                or oracle_seed < 0:
            raise FreshSourcePrerequisiteError("oracle_seed must be non-negative integer")
        backend = raw["backend_filter"]
        if not isinstance(backend, str) or not backend.strip():
            raise FreshSourcePrerequisiteError("backend_filter must be non-empty")
        ops = raw["ops"]
        if not isinstance(ops, list) or not ops or any(
                not isinstance(op, str) or not op.strip() for op in ops):
            raise FreshSourcePrerequisiteError("ops must be a non-empty string list")
        params = raw["params_filter"]
        if params is not None and (not isinstance(params, str) or not params.strip()):
            raise FreshSourcePrerequisiteError("params_filter must be null or non-empty")
        timeout_s = raw["timeout_s"]
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) \
                or timeout_s <= 0:
            raise FreshSourcePrerequisiteError("timeout_s must be positive")
        if raw["capture_mode"] != "measured":
            raise FreshSourcePrerequisiteError(
                "fresh plans are execute-only and capture_mode must be 'measured'")
        try:
            schemas.require.sha256(raw["plan_sha256"], "plan_sha256",
                                   error=FreshSourcePrerequisiteError)
        except TypeError as exc:
            raise FreshSourcePrerequisiteError(str(exc)) from exc
        if _plan_sha256(raw) != raw["plan_sha256"]:
            raise FreshSourcePrerequisiteError("plan_sha256 does not match fresh plan")
        return cls(
            campaign_id=raw["campaign_id"], proposal_id=raw["proposal_id"],
            candidate_id=raw["candidate_id"], suite_seeds=tuple(seeds),
            oracle_seed=oracle_seed, backend_filter=backend, ops=tuple(ops),
            params_filter=params, timeout_s=float(timeout_s),
            capture_mode=raw["capture_mode"], plan_sha256=raw["plan_sha256"])

    def bind_campaign(self, *, campaign_id: str, proposal: Mapping[str, Any],
                      candidate_id: str) -> None:
        expected = (campaign_id, proposal.get("proposal_id"), candidate_id)
        got = (self.campaign_id, self.proposal_id, self.candidate_id)
        if got != expected:
            raise FreshSourcePrerequisiteError(
                f"fresh plan identity {got!r} != campaign {expected!r}")
        if proposal.get("change_class") == "parameter":
            raise FreshSourcePrerequisiteError(
                "parameter proposals may not carry a fresh source plan")


def load_fresh_source_prerequisite_plan(path: Any) -> FreshSourcePrerequisitePlan:
    """Snapshot a bounded regular plan file before the campaign claim."""
    body = package_module._read_package_bytes(path)  # same O_NOFOLLOW/single-link seam
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict:
        out = {}
        for key, value in pairs:
            if key in out:
                raise FreshSourcePrerequisiteError(f"fresh plan repeats key {key!r}")
            out[key] = value
        return out
    try:
        raw = json.loads(body.decode("utf-8", "strict"), object_pairs_hook=no_duplicates,
                         parse_constant=lambda value: (_ for _ in ()).throw(
                             FreshSourcePrerequisiteError(
                                 f"fresh plan contains non-finite value {value}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreshSourcePrerequisiteError(f"fresh plan is not strict JSON: {exc}") from exc
    return FreshSourcePrerequisitePlan.from_mapping(raw)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdirs_durable(path: Path) -> None:
    pending = []
    current = path
    while not current.exists():
        pending.append(current)
        current = current.parent
    for directory in reversed(pending):
        directory.mkdir(mode=0o755)
        _fsync_dir(directory.parent)


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    body = (schemas.canonical_json(payload) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_dir(path.parent)


def _require_capture(capture: Any, label: str) -> bytes:
    if not isinstance(capture, t0_provider.CompletedProcess):
        raise FreshSourcePrerequisiteError(
            f"{label} runner returned {type(capture).__name__}, expected CompletedProcess")
    if capture.exit_code != 0 or capture.timed_out or capture.signalled or capture.orphans:
        raise FreshSourcePrerequisiteError(
            f"{label} did not complete cleanly (exit={capture.exit_code}, "
            f"timeout={capture.timed_out}, signalled={capture.signalled}, "
            f"orphans={capture.orphans!r})")
    try:
        body = capture.stdout.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise FreshSourcePrerequisiteError(f"{label} stdout is not UTF-8: {exc}") from exc
    if not body.strip():
        raise FreshSourcePrerequisiteError(f"{label} emitted empty CSV")
    return body


class FreshSourcePrerequisiteProducer:
    """Run the three producers through injected claims, runner and built binary."""

    def __init__(self, *, runner: Any) -> None:
        if not hasattr(runner, "run"):
            raise TypeError("runner must implement run(argv, env=, cwd=, timeout_s=)")
        self._runner = runner

    @staticmethod
    def _require_claims(*, cpu_claim: Any, cpu_list: str,
                        held_devices: Sequence[Any], require_device: bool) -> None:
        try:
            t0_provider.require_claim(
                cpu_claim, what="fresh source prerequisite producers", cpu_list=cpu_list)
        except (TypeError, t0_provider.ClaimNotHeld) as exc:
            raise FreshSourcePrerequisiteError(str(exc)) from exc
        if require_device:
            if not held_devices:
                raise FreshSourcePrerequisiteError(
                    "GPU fresh source prerequisites require the campaign's held device claim")
            for held in held_devices:
                if callable(getattr(held, "revocation", None)) and held.revocation() is not None:
                    raise FreshSourcePrerequisiteError(
                        f"device claim {held.claim_id!r} has a pending revocation")
                check = device_claim.check_device_claim_held(held.receipt())
                if check.outcome != schemas.PASS:
                    raise FreshSourcePrerequisiteError(
                        "device claim is not verifiably held: " + "; ".join(check.reasons))

    def produce_or_resume(
            self, *, plan: FreshSourcePrerequisitePlan, journal_root: str,
            candidate: t0_provider.CandidateBuild, candidate_source_sha256: str,
            evaluator_bundle_sha256: str, base_env: Sequence[tuple],
            parameter_env: Sequence[tuple], cpu_claim: Any, cpu_list: str,
            held_devices: Sequence[Any], require_device: bool,
            ) -> package_module.SourcePrerequisitePackage:
        root = Path(storage.assert_not_scratch(
            journal_root, what="fresh source prerequisite journal root"))
        binary_sha256 = storage.hash_file(candidate.test_backend_ops)
        run_identity = schemas.content_hash({
            "producer_id": PRODUCER_ID, "plan_sha256": plan.plan_sha256,
            "candidate_source_sha256": candidate_source_sha256,
            "candidate_binary_sha256": binary_sha256,
            "evaluator_bundle_sha256": evaluator_bundle_sha256,
        })
        evidence_dir = root / "source-prerequisites" / plan.candidate_id / plan.plan_sha256
        _mkdirs_durable(evidence_dir)
        intent_path = evidence_dir / "intent.json"
        package_path = evidence_dir / "package.json"
        identity = {
            "schema": "epyc.autokernel.fresh-source-prerequisite-intent.v1",
            "producer_id": PRODUCER_ID, "campaign_id": plan.campaign_id,
            "proposal_id": plan.proposal_id, "candidate_id": plan.candidate_id,
            "plan": plan.to_mapping(), "plan_sha256": plan.plan_sha256,
            "run_identity_sha256": run_identity,
            "candidate_source_sha256": candidate_source_sha256,
            "candidate_binary_sha256": binary_sha256,
            "evaluator_bundle_sha256": evaluator_bundle_sha256,
        }
        # Resume itself spawns nothing, but the returned package immediately
        # licenses the campaign's remaining T0 work.  A lost/revoked claim may
        # therefore not cross even the package-reuse seam.
        self._require_claims(
            cpu_claim=cpu_claim, cpu_list=cpu_list, held_devices=held_devices,
            require_device=require_device)
        if package_path.exists():
            resumed = package_module.load_source_prerequisite_package(package_path)
            resumed.materialize(
                candidate_source_sha256=candidate_source_sha256,
                candidate_binary_sha256=binary_sha256,
                evaluator_bundle_sha256=evaluator_bundle_sha256)
            return resumed
        if intent_path.exists():
            try:
                prior = json.loads(package_module._read_package_bytes(
                    intent_path).decode("utf-8", "strict"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError,
                    package_module.SourcePrerequisitePackageError) as exc:
                raise FreshSourcePrerequisiteError(
                    f"prior fresh-source intent is unreadable: {exc}") from exc
            if prior != identity:
                raise FreshSourcePrerequisiteError(
                    "prior fresh-source intent identity differs from this build")
            raise FreshSourcePrerequisiteError(
                "prior fresh-source intent has no complete package; refusing automatic "
                "duplicate execution after a partial/crashed attempt")
        _write_exclusive(intent_path, identity)

        documents: dict[str, tuple[tuple[int, bytes], ...]] = {}
        sensitivity_docs = []
        for seed in plan.suite_seeds:
            self._require_claims(cpu_claim=cpu_claim, cpu_list=cpu_list,
                                 held_devices=held_devices, require_device=False)
            invocation = t0_provider.build_backend_ops_invocation(
                binary=candidate.test_backend_ops,
                library_path=candidate.library_path, backend_filter="CPU",
                ops=plan.ops, base_env=base_env, suite_seed=seed,
                value_transform_probe=True, output_format="csv",
                params_filter=plan.params_filter, parameter_env=parameter_env,
                cpu_prefix=True)
            capture = self._runner.run(
                invocation.argv, env=invocation.env_dict(), cwd=candidate.worktree,
                timeout_s=plan.timeout_s)
            sensitivity_docs.append((seed, _require_capture(
                capture, f"input sensitivity seed {seed}")))
        documents["input_sensitivity"] = tuple(sensitivity_docs)

        oracle_docs = {}
        for prerequisite_id, flag in (
                ("hostile_distributions", "--autokernel-hostile-distributions"),
                ("checker_isolation", "--autokernel-properties")):
            self._require_claims(
                cpu_claim=cpu_claim, cpu_list=cpu_list, held_devices=held_devices,
                require_device=require_device)
            invocation = t0_provider.build_backend_ops_invocation(
                binary=candidate.test_backend_ops,
                library_path=candidate.library_path,
                backend_filter=plan.backend_filter, ops=plan.ops,
                base_env=base_env, suite_seed=plan.oracle_seed,
                output_format="csv", params_filter=plan.params_filter,
                parameter_env=parameter_env, cpu_prefix=not require_device)
            argv = tuple(token for token in invocation.argv
                         if token != "--autokernel-properties") + (flag,)
            capture = self._runner.run(
                argv, env=invocation.env_dict(), cwd=candidate.worktree,
                timeout_s=plan.timeout_s)
            oracle_docs[prerequisite_id] = ((plan.oracle_seed, _require_capture(
                capture, prerequisite_id)),)
        documents.update(oracle_docs)

        produced = package_module.build_source_prerequisite_package(
            campaign_id=plan.campaign_id, proposal_id=plan.proposal_id,
            candidate_id=plan.candidate_id,
            candidate_source_sha256=candidate_source_sha256,
            candidate_binary_sha256=binary_sha256,
            evaluator_bundle_sha256=evaluator_bundle_sha256,
            capture_mode=plan.capture_mode,
            documents_by_prerequisite=documents)
        # Re-reduce before acknowledging durable completion.  Partial or
        # malformed output leaves only the intent and is never replayed as PASS.
        produced.materialize(
            candidate_source_sha256=candidate_source_sha256,
            candidate_binary_sha256=binary_sha256,
            evaluator_bundle_sha256=evaluator_bundle_sha256)
        _write_exclusive(package_path, produced.to_mapping())
        return package_module.load_source_prerequisite_package(package_path)


__all__ = [
    "SCHEMA", "PRODUCER_ID", "FreshSourcePrerequisiteError",
    "FreshSourcePrerequisitePlan", "FreshSourcePrerequisiteProducer",
    "load_fresh_source_prerequisite_plan",
]
