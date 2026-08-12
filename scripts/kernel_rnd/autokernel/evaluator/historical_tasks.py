"""Held-out historical optimization descriptors and expert-ceiling scoring."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from typing import Mapping, Sequence

from .. import schemas


SCHEMA = "epyc.autokernel.historical_task.v1"
SEALED_HOLDOUT = "sealed_local_holdout"


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _commit(value: object, name: str) -> str:
    text = _nonempty(value, name)
    if len(text) != 40 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a full lowercase git commit")
    return text


def _timestamp(value: object, name: str) -> str:
    text = _nonempty(value, name)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    return text


@dataclass(frozen=True)
class HistoricalTaskDescriptor:
    task_id: str
    source_repo: str
    parent_commit: str
    expert_commit: str
    expert_authored_at: str
    holdout_mode: str
    actor_source_commit: str
    expert_visibility: str
    model_path: str
    model_sha256: str
    benchmark_argv: tuple[str, ...]
    metric_id: str
    metric_direction: str
    minimum_repeats: int
    evidence_refs: tuple[str, ...]
    historical_command_recovered: bool

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "HistoricalTaskDescriptor":
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"historical task schema must be {SCHEMA}")
        parent = _commit(payload.get("parent_commit"), "parent_commit")
        expert = _commit(payload.get("expert_commit"), "expert_commit")
        actor_source = _commit(payload.get("actor_source_commit"), "actor_source_commit")
        if parent == expert:
            raise ValueError("historical parent and expert commits must differ")
        if actor_source != parent:
            raise ValueError("the actor must receive only the pre-optimization parent source")
        if payload.get("holdout_mode") != SEALED_HOLDOUT:
            raise ValueError("historical tasks require a sealed local holdout")
        if payload.get("expert_visibility") != "sealed_until_terminal":
            raise ValueError("the expert patch must remain sealed until candidate terminal state")
        model_sha = _nonempty(payload.get("model_sha256"), "model_sha256")
        if len(model_sha) != 64 or any(char not in "0123456789abcdef" for char in model_sha):
            raise ValueError("model_sha256 must be a lowercase SHA-256")
        argv_raw = payload.get("benchmark_argv")
        if (not isinstance(argv_raw, list) or not argv_raw or
                any(not isinstance(item, str) or not item for item in argv_raw)):
            raise ValueError("benchmark_argv must be a non-empty string list")
        required = {"-c", "-b", "-ub", "-ngl", "-fa", "-npp", "-ntg", "-npl",
                    "--output-format"}
        if not required.issubset(argv_raw):
            raise ValueError("historical benchmark argv does not pin its full exact surface")
        repeats = payload.get("minimum_repeats")
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 3:
            raise ValueError("historical task requires at least three matched repeats")
        refs = payload.get("evidence_refs")
        if (not isinstance(refs, list) or not refs or
                any(not isinstance(item, str) or not item for item in refs)):
            raise ValueError("historical task requires evidence references")
        if payload.get("historical_command_recovered") is not False:
            raise ValueError(
                "this descriptor must preserve that the original command was not recovered")
        return cls(
            task_id=_nonempty(payload.get("task_id"), "task_id"),
            source_repo=_nonempty(payload.get("source_repo"), "source_repo"),
            parent_commit=parent, expert_commit=expert,
            expert_authored_at=_timestamp(
                payload.get("expert_authored_at"), "expert_authored_at"),
            holdout_mode=SEALED_HOLDOUT, actor_source_commit=actor_source,
            expert_visibility="sealed_until_terminal",
            model_path=_nonempty(payload.get("model_path"), "model_path"),
            model_sha256=model_sha, benchmark_argv=tuple(argv_raw),
            metric_id=_nonempty(payload.get("metric_id"), "metric_id"),
            metric_direction=_nonempty(
                payload.get("metric_direction"), "metric_direction"),
            minimum_repeats=repeats, evidence_refs=tuple(refs),
            historical_command_recovered=False)

    def canonical_sha256(self) -> str:
        return hashlib.sha256(json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA, "task_id": self.task_id,
            "source_repo": self.source_repo, "parent_commit": self.parent_commit,
            "expert_commit": self.expert_commit,
            "expert_authored_at": self.expert_authored_at,
            "holdout_mode": self.holdout_mode,
            "actor_source_commit": self.actor_source_commit,
            "expert_visibility": self.expert_visibility,
            "model_path": self.model_path, "model_sha256": self.model_sha256,
            "benchmark_argv": list(self.benchmark_argv),
            "metric_id": self.metric_id, "metric_direction": self.metric_direction,
            "minimum_repeats": self.minimum_repeats,
            "evidence_refs": list(self.evidence_refs),
            "historical_command_recovered": self.historical_command_recovered,
        }


@dataclass(frozen=True)
class ExpertCeilingReport:
    baseline: float
    expert: float
    candidate: float | None
    expert_gain_pct: float
    candidate_gain_pct: float | None
    candidate_vs_expert_pct: float | None
    expert_fraction_recovered: float | None
    check: schemas.Check

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline, "expert": self.expert,
            "candidate": self.candidate, "expert_gain_pct": self.expert_gain_pct,
            "candidate_gain_pct": self.candidate_gain_pct,
            "candidate_vs_expert_pct": self.candidate_vs_expert_pct,
            "expert_fraction_recovered": self.expert_fraction_recovered,
            "check": {"outcome": self.check.outcome,
                      "reasons": list(self.check.reasons)},
        }


def score_expert_ceiling(
        *, baseline_samples: Sequence[float], expert_samples: Sequence[float],
        candidate_samples: Sequence[float] | None, minimum_repeats: int,
        metric_direction: str = "higher_is_better") -> ExpertCeilingReport:
    if metric_direction != "higher_is_better":
        raise ValueError("historical scorer currently requires higher_is_better")
    for name, samples in (("baseline", baseline_samples), ("expert", expert_samples)):
        if len(samples) < minimum_repeats or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
                for value in samples):
            raise ValueError(f"{name} requires {minimum_repeats} positive samples")
    if candidate_samples is not None and (
            len(candidate_samples) < minimum_repeats or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
                for value in candidate_samples)):
        raise ValueError(f"candidate requires {minimum_repeats} positive samples")
    baseline = sum(baseline_samples) / len(baseline_samples)
    expert = sum(expert_samples) / len(expert_samples)
    expert_gain = 100.0 * (expert / baseline - 1.0)
    if candidate_samples is None:
        return ExpertCeilingReport(
            baseline=baseline, expert=expert, candidate=None,
            expert_gain_pct=expert_gain, candidate_gain_pct=None,
            candidate_vs_expert_pct=None, expert_fraction_recovered=None,
            check=schemas.Check(
                schemas.COULD_NOT_CHECK,
                ("no terminal AutoKernel candidate is present for expert-ceiling scoring",)))
    candidate = sum(candidate_samples) / len(candidate_samples)
    candidate_gain = 100.0 * (candidate / baseline - 1.0)
    candidate_vs_expert = 100.0 * (candidate / expert - 1.0)
    denominator = expert - baseline
    fraction = ((candidate - baseline) / denominator
                if abs(denominator) > 1e-12 else None)
    check = (schemas.Check(schemas.PASS) if expert > baseline else
             schemas.Check(schemas.FAIL, ("human patch did not beat the matched baseline",)))
    return ExpertCeilingReport(
        baseline=baseline, expert=expert, candidate=candidate,
        expert_gain_pct=expert_gain, candidate_gain_pct=candidate_gain,
        candidate_vs_expert_pct=candidate_vs_expert,
        expert_fraction_recovered=fraction, check=check)
