#!/usr/bin/env python3
"""Candidate-only AutoKernel discovery controller.

The controller deliberately owns only the state machine.  Existing campaign
code owns source mutation, isolated worktrees, build, resource claims, source
proof, dispatch attribution, screening, cleanup, and frozen-tree proof.  This
module never accepts a command from a planner and never turns a screen into a
promotion.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence

from .. import campaign, journal, source_candidate
from . import codex_container_actor, do_not_repeat, hypotheses
from . import gpu_source_proofs
from scripts.benchmark import autokernel_progression
from scripts.benchmark import run_autokernel_gpu_discovery as gpu_discovery

SCHEMA = "epyc.autokernel.discovery_controller.v2"
AUTHORITY = "nonpromotable_candidate_only_discovery"
HASH = __import__("re").compile(r"^[0-9a-f]{64}$")
SOL = {"provider": "codex", "model": "gpt-5.6-sol", "effort": "high", "role": "planner"}
TERRA = {"provider": "codex", "model": "gpt-5.6-terra", "effort": "high", "role": "critic"}


class DiscoveryControllerError(RuntimeError): pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canon(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: object) -> str: return hashlib.sha256(_canon(value)).hexdigest()


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, sort_keys=True, indent=2) + "\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)


def sealed_roster() -> dict[str, Any]:
    return {"schema": "epyc.autokernel.discovery_roster.v2", "members": [SOL, TERRA], "claude_members": 0, "member_count": 2}


def _require_roster(value: Mapping[str, Any]) -> None:
    if dict(value) != sealed_roster(): raise DiscoveryControllerError("runtime roster is not exact Sol planner + Terra critic, 0 Claude")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value: raise DiscoveryControllerError(f"{label} must be non-empty text")
    return value.strip()


@dataclass(frozen=True)
class PlannedCandidate:
    hypothesis_id: str
    statement: str
    falsifier: str
    regime: Mapping[str, Any]
    proposal: Mapping[str, Any]
    source_manifest: source_candidate.SourcePatchManifest
    source_manifest_sha256: str

    def __post_init__(self) -> None:
        _text(self.hypothesis_id, "hypothesis_id"); _text(self.statement, "statement"); _text(self.falsifier, "falsifier")
        if not self.hypothesis_id.startswith("akh-"): raise DiscoveryControllerError("hypothesis_id must start akh-")
        if not isinstance(self.regime, Mapping) or not isinstance(self.proposal, Mapping): raise DiscoveryControllerError("candidate regime and proposal must be mappings")
        if not isinstance(self.source_manifest, source_candidate.SourcePatchManifest): raise DiscoveryControllerError("candidate requires typed SourcePatchManifest")
        if not HASH.fullmatch(self.source_manifest_sha256): raise DiscoveryControllerError("source manifest hash is required")
        # Planner-owned effect fields are structurally impossible.
        if any("effect" in str(key).lower() or "result" in str(key).lower() for key in self.proposal):
            raise DiscoveryControllerError("planner proposal may not carry measured result fields")


@dataclass(frozen=True)
class Critique:
    decision: str
    reason: str
    def __post_init__(self) -> None:
        if self.decision not in {"accept", "reject", "revise"}: raise DiscoveryControllerError("critic decision must be accept, reject, or revise")
        _text(self.reason, "critic reason")


@dataclass(frozen=True)
class SealedScreen:
    receipt_path: str
    result_sha256: str
    effect_fraction: float
    classification: str
    baseline_sha256: str
    source_proof_sha256: str
    dispatch_proof_sha256: str
    candidate_only: bool = True
    promotion_claim: bool = False
    stages: tuple[str, ...] = ("materialized", "built", "correctness", "attribution", "screen")

    def __post_init__(self) -> None:
        if self.classification not in {"candidate", "screened_out", "inconclusive", "failed"}: raise DiscoveryControllerError("unknown screen class")
        if not isinstance(self.effect_fraction, (int, float)): raise DiscoveryControllerError("screen effect must be measured numeric evidence")
        if not self.candidate_only or self.promotion_claim: raise DiscoveryControllerError("discovery screen must remain nonpromotable")
        if tuple(self.stages) != ("materialized", "built", "correctness", "attribution", "screen"):
            raise DiscoveryControllerError("screen did not prove the required fail-closed stage order")
        for value in (self.result_sha256, self.baseline_sha256, self.source_proof_sha256, self.dispatch_proof_sha256):
            if not HASH.fullmatch(value): raise DiscoveryControllerError("sealed result requires evidence hashes")


class Planner(Protocol):
    def attest(self) -> Mapping[str, Any]: ...
    def plan(self, *, context: Mapping[str, Any], workspace: Path) -> PlannedCandidate: ...

class Critic(Protocol):
    def attest(self) -> Mapping[str, Any]: ...
    def review(self, candidate: PlannedCandidate, *, context: Mapping[str, Any], workspace: Path) -> Critique: ...

class Lease(Protocol):
    def admit(self, candidate: PlannedCandidate) -> Mapping[str, Any]: ...

class Screener(Protocol):
    def screen(self, candidate: PlannedCandidate, authorization: hypotheses.ClaimAuthorization, lease: Mapping[str, Any]) -> SealedScreen: ...


class CodexPlanner:
    """Concrete Sol actor. It may write only a plan and patch manifest in workspace."""
    def __init__(self, *, wrapper: Path, environment: Mapping[str, str]) -> None: self.wrapper, self.environment = wrapper, dict(environment)
    def attest(self) -> Mapping[str, Any]: return {**SOL, "runtime": codex_container_actor.runtime_identity(self.wrapper)}
    def plan(self, *, context: Mapping[str, Any], workspace: Path) -> PlannedCandidate:
        prompt = json.dumps({"role": SOL, "context": context, "output": "Write plan.json and source-patch.json in workspace. plan.json may only name hypothesis_id, statement, falsifier, regime, proposal, source_manifest_path; no commands or results."}, sort_keys=True)
        result = codex_container_actor.run_actor(wrapper=self.wrapper, workspace=workspace, model=SOL["model"], effort=SOL["effort"], prompt=prompt, environment=self.environment)
        if result.returncode: raise DiscoveryControllerError(f"Sol actor failed: {result.stderr[-400:]}")
        return _load_plan(workspace / "plan.json", workspace)


class CodexCritic:
    """Concrete Terra actor. It can bind a veto but never alters the candidate."""
    def __init__(self, *, wrapper: Path, environment: Mapping[str, str]) -> None: self.wrapper, self.environment = wrapper, dict(environment)
    def attest(self) -> Mapping[str, Any]: return {**TERRA, "runtime": codex_container_actor.runtime_identity(self.wrapper)}
    def review(self, candidate: PlannedCandidate, *, context: Mapping[str, Any], workspace: Path) -> Critique:
        prompt = json.dumps({"role": TERRA, "context": context, "candidate": {"hypothesis_id": candidate.hypothesis_id, "statement": candidate.statement, "falsifier": candidate.falsifier, "proposal": candidate.proposal, "source_manifest_sha256": candidate.source_manifest_sha256}, "output": "Write critique.json with exactly decision=accept|reject|revise and reason."}, sort_keys=True)
        result = codex_container_actor.run_actor(wrapper=self.wrapper, workspace=workspace, model=TERRA["model"], effort=TERRA["effort"], prompt=prompt, environment=self.environment)
        if result.returncode: raise DiscoveryControllerError(f"Terra actor failed: {result.stderr[-400:]}")
        value = _read_object(workspace / "critique.json", workspace)
        if set(value) != {"decision", "reason"}: raise DiscoveryControllerError("critic output schema mismatch")
        return Critique(**value)


def _read_object(path: Path, root: Path) -> dict[str, Any]:
    try: path.resolve().relative_to(root.resolve())
    except ValueError as exc: raise DiscoveryControllerError("actor artifact escaped workspace") from exc
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise DiscoveryControllerError(f"invalid actor artifact {path.name}") from exc
    if not isinstance(value, dict): raise DiscoveryControllerError("actor artifact must be object")
    return value


def _load_plan(path: Path, root: Path) -> PlannedCandidate:
    value = _read_object(path, root)
    allowed = {"hypothesis_id", "statement", "falsifier", "regime", "proposal", "source_manifest_path"}
    if set(value) != allowed: raise DiscoveryControllerError("planner output schema mismatch")
    raw_path = Path(_text(value.pop("source_manifest_path"), "source_manifest_path"))
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise DiscoveryControllerError("source manifest path must be a workspace-relative path")
    manifest_path = root / raw_path
    try:
        manifest_path.resolve(strict=True).relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise DiscoveryControllerError("source manifest escaped disposable workspace") from exc
    manifest = source_candidate.load_source_patch_manifest(manifest_path)
    return PlannedCandidate(**value, source_manifest=manifest, source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())


class CampaignScreener:
    """Concrete adapter: call the existing candidate-only campaign transaction."""
    def __init__(self, *, spec_factory: Callable[[PlannedCandidate, hypotheses.ClaimAuthorization], campaign.CampaignSpec], ops_factory: Callable[[], Any]) -> None:
        self.spec_factory, self.ops_factory = spec_factory, ops_factory
    def screen(self, candidate: PlannedCandidate, authorization: hypotheses.ClaimAuthorization, lease: Mapping[str, Any]) -> SealedScreen:
        spec = self.spec_factory(candidate, authorization)
        if not spec.screening_only or spec.source_patch is not candidate.source_manifest or spec.authorization != authorization:
            raise DiscoveryControllerError("campaign adapter must bind typed patch, authorization, and candidate-only screen")
        if spec.source_prerequisite_package is None and spec.fresh_source_prerequisite_plan is None:
            raise DiscoveryControllerError("source candidate requires source correctness and dispatch prerequisite package")
        result = campaign.run_campaign(spec, self.ops_factory())
        return _screen_from_campaign(result)


def _screen_from_campaign(result: campaign.CampaignResult) -> SealedScreen:
    raw = result.to_dict(); report = raw.get("screening_report")
    if not (result.ok and raw.get("state") == "decided" and raw.get("screening_only") is True and isinstance(report, Mapping)):
        raise DiscoveryControllerError("campaign did not produce a sealed candidate-only result")
    required = ("baseline_sha256", "source_prerequisite_package_sha256", "dispatch_attribution_sha256", "result_sha256")
    if not all(isinstance(report.get(key), str) and HASH.fullmatch(report[key]) for key in required):
        raise DiscoveryControllerError("campaign result lacks source proof, exact dispatch proof, baseline, or result hash")
    return SealedScreen(receipt_path=str(report.get("receipt_path", "")), result_sha256=report["result_sha256"], effect_fraction=float(report["median_relative"]), classification=str(report.get("classification", "candidate")), baseline_sha256=report["baseline_sha256"], source_proof_sha256=report["source_prerequisite_package_sha256"], dispatch_proof_sha256=report["dispatch_attribution_sha256"])


@dataclass(frozen=True)
class GpuSourceBuild:
    """A completed isolated build, returned only by a typed source-build seam."""
    anchor_build: Path
    candidate_build: Path
    source_sha256: str
    binary_sha256: str
    def __post_init__(self) -> None:
        for path in (self.anchor_build, self.candidate_build):
            if not path.is_absolute() or not path.is_dir():
                raise DiscoveryControllerError("GPU source build paths must be existing absolute directories")
        for value in (self.source_sha256, self.binary_sha256):
            if not HASH.fullmatch(value): raise DiscoveryControllerError("GPU build identity hash is invalid")


@dataclass(frozen=True)
class ProofReceipt:
    """Hash-bound source or dispatch proof produced before any screen call."""
    path: Path
    sha256: str
    kind: str
    def __post_init__(self) -> None:
        if self.kind not in {"source", "dispatch"} or not self.path.is_absolute() or not self.path.is_file() or not HASH.fullmatch(self.sha256):
            raise DiscoveryControllerError("proof receipt must be an existing typed source/dispatch artifact")
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256:
            raise DiscoveryControllerError("proof receipt bytes differ from its sealed hash")


class GpuSourceScreener:
    """GPU source lane using the existing governed discovery runner.

    This intentionally does not reuse the CPU baseline bank: that bank proves an
    unchanged binary with a parameter delta.  GPU source runs need distinct
    build identities and their own sealed paired receipt.
    """
    def __init__(self, *, build_source: Callable[[PlannedCandidate, hypotheses.ClaimAuthorization, Mapping[str, Any]], GpuSourceBuild],
                 source_proof: Callable[[PlannedCandidate, GpuSourceBuild], ProofReceipt],
                 dispatch_proof: Callable[[PlannedCandidate, GpuSourceBuild], ProofReceipt],
                 args_factory: Callable[[PlannedCandidate, GpuSourceBuild, Mapping[str, Any]], Any]) -> None:
        self.build_source, self.source_proof, self.dispatch_proof, self.args_factory = build_source, source_proof, dispatch_proof, args_factory

    def screen(self, candidate: PlannedCandidate, authorization: hypotheses.ClaimAuthorization, lease: Mapping[str, Any]) -> SealedScreen:
        build = self.build_source(candidate, authorization, lease)
        source = self.source_proof(candidate, build)
        if source.kind != "source": raise DiscoveryControllerError("source gate returned the wrong receipt kind")
        dispatch = self.dispatch_proof(candidate, build)
        if dispatch.kind != "dispatch": raise DiscoveryControllerError("dispatch gate returned the wrong receipt kind")
        args = self.args_factory(candidate, build, lease)
        # The established runner owns KFD/VRAM, device claims, paired samples,
        # and its durable result.  This controller does not spawn a shell.
        if getattr(args, "factor", None) != "source_patch" or Path(getattr(args, "anchor_build", "")).resolve() != build.anchor_build or Path(getattr(args, "candidate_build", "")).resolve() != build.candidate_build:
            raise DiscoveryControllerError("GPU source runner arguments are not bound to the typed build")
        raw = gpu_discovery.run(args)
        result_path = Path(args.output_dir).resolve() / "result.json"
        durable = gpu_source_proofs.require_result_file(result_path, raw)["body"]
        raw = durable
        if not (raw.get("schema") == "epyc.autokernel.gpu_candidate_only_screen.v2" and raw.get("non_promotable") is True and raw.get("promotion_claim") is False and raw.get("hip_residency_proved") is True):
            raise DiscoveryControllerError("GPU runner returned an unsealed or non-resident discovery result")
        projection = autokernel_progression._gpu_screen(result_path, raw)
        if projection is None: raise DiscoveryControllerError("GPU result failed canonical progression validation")
        return SealedScreen(receipt_path=str(result_path), result_sha256=str(raw["result_sha256"]), effect_fraction=float(raw["median_relative"]), classification=str(projection["stage"]), baseline_sha256=str(raw["baseline_sha256"]), source_proof_sha256=source.sha256, dispatch_proof_sha256=dispatch.sha256)


@dataclass(frozen=True)
class ControllerConfig:
    output_root: Path
    max_iterations: int = 1
    nomination_threshold: float = 0.03
    dry_run: bool = False
    def __post_init__(self) -> None:
        if not self.output_root.is_absolute() or not 1 <= self.max_iterations <= 1000 or self.nomination_threshold <= 0: raise DiscoveryControllerError("invalid controller config")


class DurableState:
    def __init__(self, root: Path) -> None:
        self.root=root; self.book=journal.Journal(str(root / "journal")); self.book.initialize(); self.path=root / "state.json"
    def load(self) -> dict[str, Any]:
        if not self.path.exists(): return {"schema": SCHEMA, "authority": AUTHORITY, "roster": sealed_roster(), "iterations": [], "next": 1, "complete": False}
        value=_read_object(self.path, self.root); _require_roster(value.get("roster", {}))
        if value.get("schema") != SCHEMA or value.get("authority") != AUTHORITY: raise DiscoveryControllerError("wrong controller journal")
        return value
    def save(self, state: dict[str, Any], phase: str) -> None:
        state["updated_at"]=_now(); state["state_sha256"]=_sha({k:v for k,v in state.items() if k!="state_sha256"}); _atomic(self.path,state)
        self.book.append(journal.KIND_STOP_STATE,{"state":f"discovery_{phase}","controller_state_sha256":state["state_sha256"]})


def _tracker(store: DurableState) -> hypotheses.HypothesisTracker:
    return hypotheses.HypothesisTracker(journal_=store.book, root=str(store.root / "hypotheses"), campaign_id="ak-discovery")


def _memory_block(tracker: hypotheses.HypothesisTracker, turn: int) -> Mapping[str, Any]:
    ledger=do_not_repeat.compile_for_tracker(tracker); return do_not_repeat.planner_round_block(tracker, ledger, round_id=f"discovery-{turn}")


def _ensure_question(tracker: hypotheses.HypothesisTracker, item: PlannedCandidate) -> None:
    question=hypotheses.Hypothesis(hypothesis_id=item.hypothesis_id, statement=item.statement, falsifier=item.falsifier, origin=hypotheses.ORIGIN_PLANNER, author="gpt-5.6-sol", regime=item.regime, source={"manifest_sha256":item.source_manifest_sha256})
    try: tracker.open_hypothesis(question)
    except hypotheses.HypothesisAlreadyTracked: pass


def _context(state: Mapping[str, Any], tracker: hypotheses.HypothesisTracker, turn: int) -> dict[str, Any]:
    return {"authority": AUTHORITY, "turn":turn, "roster":sealed_roster(), "prior_results":[row.get("result_sha256") for row in state["iterations"] if row.get("result_sha256")], "do_not_repeat":_memory_block(tracker,turn)}


def _append_nomination(root: Path, item: PlannedCandidate, result: SealedScreen, threshold: float) -> None:
    if result.effect_fraction < threshold: return
    path=root / "promotion-queue.jsonl"; key=_sha({"result":result.result_sha256,"manifest":item.source_manifest_sha256})
    existing=path.read_text() if path.exists() else ""
    if key in existing: return
    row={"schema":"epyc.autokernel.discovery_nomination.v1","idempotency_key":key,"receipt_path":result.receipt_path,"result_sha256":result.result_sha256,"source_manifest_sha256":item.source_manifest_sha256,"effect_fraction":result.effect_fraction,"threshold":threshold,"promotion_claim":False,"operator_decision_required":True,"authority":AUTHORITY}
    with path.open("a",encoding="utf-8") as f: f.write(json.dumps(row,sort_keys=True)+"\n"); f.flush(); os.fsync(f.fileno())


def _write_projection(root: Path) -> None:
    # Canonical projection is derived from receipts, not planner text.
    autokernel_progression.export_progression(root=root, output=root / "surface" / "kernel_progression.json")


def run_controller(config: ControllerConfig, *, planner: Planner, critic: Critic, screener: Screener, lease: Lease) -> dict[str, Any]:
    planner_attestation, critic_attestation = dict(planner.attest()), dict(critic.attest())
    if ({k: planner_attestation.get(k) for k in SOL} != SOL
            or {k: critic_attestation.get(k) for k in TERRA} != TERRA
            or not isinstance(planner_attestation.get("runtime"), Mapping)
            or not isinstance(critic_attestation.get("runtime"), Mapping)):
        raise DiscoveryControllerError("actors did not attest the sealed Codex runtime identities")
    _require_roster({"schema":"epyc.autokernel.discovery_roster.v2","members":[SOL,TERRA],"claude_members":0,"member_count":2})
    store=DurableState(config.output_root); state=store.load()
    # A completed state is an acknowledged terminal checkpoint.  Re-entering it
    # must be a read, not another executor opportunity or a timestamp rewrite.
    if state["complete"]: return state
    tracker=_tracker(store)
    while not state["complete"] and state["next"] <= config.max_iterations:
        turn=state["next"]; context=_context(state,tracker,turn)
        with tempfile.TemporaryDirectory(prefix=f"ak-discovery-{turn}-", dir=config.output_root) as temp:
            workspace=Path(temp)
            item=planner.plan(context=context,workspace=workspace)
            review=critic.review(item,context=context,workspace=workspace)
            row={"turn":turn,"hypothesis_id":item.hypothesis_id,"proposal_sha256":_sha(item.proposal),"source_manifest_sha256":item.source_manifest_sha256,"critic":asdict(review),"context_sha256":_sha(context)}
            if review.decision != "accept":
                row["status"]="critic_"+review.decision; state["iterations"].append(row); state["next"]+=1; store.save(state,"critic_refused"); continue
            _ensure_question(tracker,item)
            ledger=do_not_repeat.compile_for_tracker(tracker)
            try: authorization=tracker.authorize_claim(item.hypothesis_id,purpose="candidate_only_discovery",authorized_by="discovery_controller",ledger=ledger)
            except hypotheses.HypothesisError as exc:
                row.update(status="authorization_refused",reason=str(exc)); state["iterations"].append(row); state["next"]+=1; store.save(state,"authorization_refused"); continue
            permit=lease.admit(item)
            if not bool(permit.get("admitted")):
                # Waiting is durable but is not an experiment and cannot spend an
                # iteration budget.  Planning/critique may continue elsewhere;
                # this exact candidate is retried only after a new lease admits it.
                row.update(status="waiting_resource",lease=dict(permit)); state["pending"]=row; store.save(state,"waiting_resource"); break
            try: result=screener.screen(item,authorization,permit)
            except Exception as exc:
                row.update(status="screen_refused",reason=f"{type(exc).__name__}: {exc}"); state["iterations"].append(row); state["next"]+=1; store.save(state,"screen_refused"); continue
            row.update(status=result.classification,result_sha256=result.result_sha256,evidence={"baseline":result.baseline_sha256,"source":result.source_proof_sha256,"dispatch":result.dispatch_proof_sha256},effect_fraction=result.effect_fraction)
            state["iterations"].append(row); state["next"]+=1; _append_nomination(config.output_root,item,result,config.nomination_threshold); _write_projection(config.output_root); store.save(state,"screened")
    state["complete"]=state["next"]>config.max_iterations
    if state["complete"]: state.pop("pending",None)
    store.save(state,"complete" if state["complete"] else "paused"); return state


def _load_factory(reference: str) -> Mapping[str, Any]:
    module,name=reference.split(":",1); value=getattr(importlib.import_module(module),name)()
    if not isinstance(value,Mapping): raise DiscoveryControllerError("adapter factory must return mapping")
    return value


def main(argv: Sequence[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-root",required=True); parser.add_argument("--max-iterations",type=int,default=1); parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--adapter-factory")
    args=parser.parse_args(argv); config=ControllerConfig(Path(args.output_root).resolve(),args.max_iterations,dry_run=args.dry_run)
    if not args.adapter_factory: raise DiscoveryControllerError("concrete adapter factory required; use --dry-run with an explicit simulated factory in tests")
    parts=_load_factory(args.adapter_factory); run_controller(config,planner=parts["planner"],critic=parts["critic"],screener=parts["screener"],lease=parts["lease"]); return 0

if __name__=="__main__": main()
