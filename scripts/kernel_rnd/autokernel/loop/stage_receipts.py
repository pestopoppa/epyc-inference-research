"""Explicit stop/resume controller contract (AK-RSM-1).

A long campaign iteration walks a fixed stage graph:

    assignment -> planner -> critic -> authorization -> build -> correctness
               -> attribution -> runner -> classification

Every completed stage persists a CONTENT-ADDRESSED receipt: the file name is the
sha256 of the canonical receipt body, and the body binds the graph identity, the
deployment identity, the stage, the digest of the stage input and the digest of the
stage output. On restart against the SAME graph a completed stage is never executed
again; its recorded output is replayed from the receipt.

Stopping is a latch, not a kill: `StopLatch.request()` writes an atomic latch file,
and the runner checks it only at stage boundaries, so a stage in flight finishes and
receipts before the iteration exits as `stopped`.

A graph upgrade (a new stage-graph version) never silently copies state. The new
graph refuses every old receipt unless a versioned CARRY-FORWARD receipt names the
old graph digest, the new graph digest, the exact stage receipts it carries, and the
SAME deployment identity. A carry-forward can never change deployment identity.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

RECEIPT_SCHEMA = "epyc.autokernel.stage_receipt.v1"
CARRY_SCHEMA = "epyc.autokernel.stage_carry_forward.v1"
LATCH_SCHEMA = "epyc.autokernel.stop_latch.v1"

STAGES: tuple[str, ...] = (
    "assignment", "planner", "critic", "authorization", "build", "correctness",
    "attribution", "runner", "classification",
)


class ReceiptRefused(RuntimeError):
    """A receipt, graph or carry-forward failed the contract."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class StageGraph:
    """The versioned stage graph plus the deployment it runs under."""
    version: int
    deployment_identity: str
    stages: tuple[str, ...] = STAGES

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int) \
                or self.version < 1:
            raise ReceiptRefused("graph version must be a positive integer")
        if not isinstance(self.deployment_identity, str) or not self.deployment_identity:
            raise ReceiptRefused("deployment identity is required")
        if not self.stages or len(set(self.stages)) != len(self.stages):
            raise ReceiptRefused("graph stages must be non-empty and unique")

    @property
    def digest(self) -> str:
        return digest({"version": self.version, "stages": list(self.stages)})


class StopLatch:
    """File-backed graceful stop latch; checked only at stage boundaries."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def request(self, reason: str) -> None:
        _atomic_write(self.path, _canonical({"schema": LATCH_SCHEMA, "reason": str(reason)}))

    def is_set(self) -> bool:
        return self.path.exists()

    __call__ = is_set  # usable directly as a `stop_requested` callable

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


class ReceiptStore:
    """Content-addressed receipts under <root>/receipts, carries under <root>/carry."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.receipts = self.root / "receipts"
        self.carries = self.root / "carry"

    # -- receipts ---------------------------------------------------------
    def put(self, body: Mapping[str, Any]) -> str:
        if body.get("schema") != RECEIPT_SCHEMA:
            raise ReceiptRefused("receipt schema mismatch")
        name = digest(dict(body))
        path = self.receipts / f"{name}.json"
        if not path.exists():
            _atomic_write(path, _canonical(dict(body)))
        return name

    def get(self, name: str) -> dict:
        path = self.receipts / f"{name}.json"
        raw = path.read_bytes()
        body = json.loads(raw)
        if hashlib.sha256(_canonical(body)).hexdigest() != name:
            raise ReceiptRefused(f"receipt {name} content does not match its address")
        return body

    def all(self) -> list[tuple[str, dict]]:
        if not self.receipts.is_dir():
            return []
        return [(p.stem, self.get(p.stem)) for p in sorted(self.receipts.glob("*.json"))]

    # -- carry-forward ----------------------------------------------------
    def put_carry(self, body: Mapping[str, Any]) -> str:
        if body.get("schema") != CARRY_SCHEMA:
            raise ReceiptRefused("carry-forward schema mismatch")
        name = digest(dict(body))
        _atomic_write(self.carries / f"{name}.json", _canonical(dict(body)))
        return name

    def carries_into(self, graph: StageGraph) -> list[dict]:
        if not self.carries.is_dir():
            return []
        out = []
        for path in sorted(self.carries.glob("*.json")):
            body = json.loads(path.read_bytes())
            if digest(body) != path.stem:
                raise ReceiptRefused(f"carry-forward {path.stem} content mismatch")
            if body.get("to_graph") == graph.digest:
                out.append(body)
        return out


def carry_forward(store: ReceiptStore, *, old: StageGraph, new: StageGraph,
                  iteration: str, stages: Sequence[str], reason: str) -> str:
    """Write a versioned carry-forward receipt from `old` to `new` for one iteration."""
    if new.version <= old.version:
        raise ReceiptRefused("carry-forward must move to a strictly newer graph version")
    if new.deployment_identity != old.deployment_identity:
        raise ReceiptRefused("carry-forward may not change deployment identity")
    if not reason:
        raise ReceiptRefused("carry-forward requires a reason")
    carried = {}
    by_stage = {body["stage"]: name for name, body in store.all()
                if body["graph"] == old.digest and body["iteration"] == iteration
                and body["deployment_identity"] == old.deployment_identity}
    for stage in stages:
        if stage not in new.stages:
            raise ReceiptRefused(f"stage {stage!r} is not in the new graph")
        if stage not in by_stage:
            raise ReceiptRefused(f"no {stage!r} receipt to carry from the old graph")
        carried[stage] = by_stage[stage]
    return store.put_carry({
        "schema": CARRY_SCHEMA, "from_graph": old.digest, "from_version": old.version,
        "to_graph": new.digest, "to_version": new.version,
        "deployment_identity": new.deployment_identity, "iteration": iteration,
        "carried": carried, "reason": reason})


@dataclass(frozen=True)
class IterationOutcome:
    status: str  # "complete" | "stopped"
    executed: tuple[str, ...]
    replayed: tuple[str, ...]
    outputs: Mapping[str, Any]
    stop_reason: str | None = None


StageFn = Callable[[Mapping[str, Any]], Any]


class StageRunner:
    """Runs one iteration's stage graph, receipting each stage and honoring the latch."""

    def __init__(self, store: ReceiptStore, graph: StageGraph, latch: StopLatch):
        self.store, self.graph, self.latch = store, graph, latch

    def _completed(self, iteration: str) -> dict[str, dict]:
        done: dict[str, dict] = {}
        for _, body in self.store.all():
            if body["iteration"] != iteration:
                continue
            if body["deployment_identity"] != self.graph.deployment_identity:
                raise ReceiptRefused("receipt from a foreign deployment identity")
            if body["graph"] == self.graph.digest:
                done[body["stage"]] = body
        for carry in self.store.carries_into(self.graph):
            if carry["iteration"] != iteration:
                continue
            if carry["deployment_identity"] != self.graph.deployment_identity:
                raise ReceiptRefused("carry-forward from a foreign deployment identity")
            for stage, name in carry["carried"].items():
                body = self.store.get(name)
                if body["graph"] != carry["from_graph"] or body["stage"] != stage:
                    raise ReceiptRefused("carry-forward names a mismatched receipt")
                done.setdefault(stage, body)
        return done

    def run(self, iteration: str, stage_fns: Mapping[str, StageFn],
            seed: Mapping[str, Any] | None = None) -> IterationOutcome:
        missing = [s for s in self.graph.stages if s not in stage_fns]
        if missing:
            raise ReceiptRefused(f"no executor for stages {missing}")
        done = self._completed(iteration)
        outputs: dict[str, Any] = {"seed": dict(seed or {})}
        executed: list[str] = []
        replayed: list[str] = []
        for stage in self.graph.stages:
            input_digest = digest(outputs)
            prior = done.get(stage)
            if prior is not None:
                if prior["input_digest"] != input_digest:
                    raise ReceiptRefused(
                        f"{stage} receipt input does not match the replayed chain")
                outputs[stage] = prior["output"]
                replayed.append(stage)
                continue
            if self.latch.is_set():
                return IterationOutcome("stopped", tuple(executed), tuple(replayed),
                                        outputs, f"stop latched before {stage}")
            result = stage_fns[stage](dict(outputs))
            self.store.put({
                "schema": RECEIPT_SCHEMA, "graph": self.graph.digest,
                "graph_version": self.graph.version,
                "deployment_identity": self.graph.deployment_identity,
                "iteration": iteration, "stage": stage, "input_digest": input_digest,
                "output": result, "output_digest": digest(result)})
            outputs[stage] = result
            executed.append(stage)
        return IterationOutcome("complete", tuple(executed), tuple(replayed), outputs)


__all__ = ["CARRY_SCHEMA", "IterationOutcome", "LATCH_SCHEMA", "RECEIPT_SCHEMA",
           "ReceiptRefused", "ReceiptStore", "STAGES", "StageGraph", "StageRunner",
           "StopLatch", "carry_forward", "digest"]
