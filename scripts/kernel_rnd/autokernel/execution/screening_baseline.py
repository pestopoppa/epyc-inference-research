"""Immutable amortized baseline bank for non-promotable discovery screens.

The bank deliberately carries only an exact-frame anchor vector.  It is never
accepted by strict T1 and cannot be converted into a candidate/archive record.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .. import schemas
from ..evaluator import recipes
from . import microbench
from ..resource import preflight

SCHEMA = "epyc.autokernel.screening_baseline_bank.v2"


class BaselineBankError(ValueError):
    pass


@dataclass(frozen=True)
class BaselineBank:
    frame: Mapping[str, Any]
    anchor_samples: tuple[float, ...]
    sentinel_before: float
    anchor_command: Mapping[str, Any]
    sentinel_after: float | None = None

    def to_dict(self) -> dict[str, Any]:
        body = {"schema": SCHEMA, "frame": dict(self.frame),
                "anchor_samples": list(self.anchor_samples),
                "anchor_command": dict(self.anchor_command),
                "sentinel_before": self.sentinel_before,
                "sentinel_after": self.sentinel_after}
        return {**body, "baseline_sha256": schemas.content_hash(body)}

    def admit(self, frame: Mapping[str, Any]) -> None:
        if dict(frame) != dict(self.frame):
            raise BaselineBankError("screening baseline frame differs from candidate frame")
        if self.sentinel_after is not None:
            raise BaselineBankError("screening baseline is closed; create a fresh bank")

    def nominate(self, candidate_samples: tuple[float, ...]) -> dict[str, Any]:
        """Noise-tolerant directional summary, never a pass/fail decision."""
        if not candidate_samples:
            raise BaselineBankError("screening candidate has no samples")
        center = sum(self.anchor_samples) / len(self.anchor_samples)
        values = tuple((x - center) / center for x in candidate_samples)
        return {"baseline_center": center, "candidate_samples": list(candidate_samples),
                "relative_effects": list(values),
                "median_relative": sorted(values)[len(values) // 2],
                "uncertainty": "screening_noise_unquantified_nonpromotable",
                "nomination": "top_k_candidate_only_not_a_keep"}


def load(path: str | Path) -> BaselineBank:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise BaselineBankError("baseline bank must be an object")
    body = {key: raw.get(key) for key in ("schema", "frame", "anchor_samples", "anchor_command",
                                          "sentinel_before", "sentinel_after")}
    if raw.get("baseline_sha256") != schemas.content_hash(body) or body["schema"] != SCHEMA:
        raise BaselineBankError("baseline bank schema/hash is invalid")
    values = body["anchor_samples"]
    command = body["anchor_command"]
    if not isinstance(body["frame"], Mapping) or not isinstance(values, list) or len(values) < 2:
        raise BaselineBankError("baseline bank needs exact frame and >=2 anchor samples")
    if body["frame"].get("anchor_ggml_iqk") != "0" \
            or not isinstance(command, Mapping) \
            or command.get("arm") != "anchor" \
            or command.get("env", {}).get("GGML_IQK") != "0" \
            or command.get("params", {}).get("ggml_iqk") != "0":
        raise BaselineBankError(
            "baseline bank must seal an anchor command with GGML_IQK=0")
    return BaselineBank(dict(body["frame"]), tuple(float(x) for x in values),
                        float(body["sentinel_before"]),
                        dict(command),
                        None if body["sentinel_after"] is None else float(body["sentinel_after"]))


def create(*, frame: Mapping[str, Any], anchor_command: Mapping[str, Any],
           invoke_anchor, anchor_count: int = 3) -> BaselineBank:
    """Seal O(1) anchor invocations once for a whole discovery batch."""
    if anchor_count < 2:
        raise BaselineBankError("baseline bank needs at least two anchor invocations")
    samples = tuple(float(invoke_anchor()) for _ in range(anchor_count))
    if frame.get("anchor_ggml_iqk") != "0" \
            or anchor_command.get("env", {}).get("GGML_IQK") != "0" \
            or anchor_command.get("params", {}).get("ggml_iqk") != "0":
        raise BaselineBankError("baseline creation requires a bound GGML_IQK=0 anchor")
    return BaselineBank(dict(frame), samples, samples[-1], dict(anchor_command))


def screen(*, bank: BaselineBank, frame: Mapping[str, Any], invoke_candidate,
           competing_inference: bool,
           candidate_command: Mapping[str, Any]) -> dict[str, Any]:
    """Three candidate-only calls; ordinary host load is intentionally not input.

    The caller must provide the claim-scoped competing-inference witness. That
    is the one discovery blocker; service/build/load noise is reflected in the
    uncertainty label, not converted into a false refusal.
    """
    bank.admit(frame)
    if competing_inference:
        raise BaselineBankError("competing model inference occupies claimed screening compute")
    if candidate_command.get("arm") != "candidate" \
            or candidate_command.get("env", {}).get("GGML_IQK") != "1" \
            or candidate_command.get("params", {}).get("ggml_iqk") != "1":
        raise BaselineBankError(
            "screening candidate command must seal candidate GGML_IQK=1")
    anchor_env = dict(bank.anchor_command.get("env", {}))
    candidate_env = dict(candidate_command.get("env", {}))
    # LD_LIBRARY_PATH is binding-specific by design (anchor vs isolated
    # candidate build); every other recipe environment field must be identical
    # except the sole intended GGML_IQK factor.
    for env in (anchor_env, candidate_env):
        env.pop("LD_LIBRARY_PATH", None)
        env.pop("GGML_IQK", None)
    anchor_params = dict(bank.anchor_command.get("params", {}))
    candidate_params = dict(candidate_command.get("params", {}))
    anchor_params.pop("ggml_iqk", None)
    candidate_params.pop("ggml_iqk", None)
    if anchor_env != candidate_env or anchor_params != candidate_params:
        raise BaselineBankError(
            "screening arm commands differ beyond the sole intended GGML_IQK factor")
    samples = tuple(float(invoke_candidate()) for _ in range(3))
    report = bank.nominate(samples)
    report.update({"candidate_invocations": 3, "anchor_invocations": 0,
                   "host_noise_policy": "recorded_not_blocking",
                   "candidate_command_sha256": schemas.content_hash(candidate_command),
                   "sole_intended_factor": {"name": "GGML_IQK",
                                             "anchor": "0", "candidate": "1"},
                   "non_promotable": True})
    return report


def invoke_command(*, command: recipes.ConstructedCommand, spawner: microbench.Spawner,
                   timeout_s: float = 300.0) -> float:
    """Run exactly one bound llama-bench command and reduce its own samples."""
    env = microbench.assemble_env(command.env).env
    spawned = spawner.run(command.argv, env, timeout_s=timeout_s)
    if spawned.timed_out or spawned.returncode != 0:
        raise BaselineBankError("screening invocation failed or timed out")
    rows = microbench.parse_llama_bench_json(spawned.stdout)
    if len(rows) != 1:
        raise BaselineBankError("screening invocation must emit exactly one result row")
    check = microbench.LlamaBenchExpectation.from_command(command).check_row(rows[0])
    if check.outcome != schemas.PASS:
        raise BaselineBankError("screening command/result frame mismatch: " + "; ".join(check.reasons))
    values = rows[0].metric_samples
    return sum(values) / len(values)


def competing_inference_witness() -> dict[str, Any]:
    """Read only model-inference identities; ordinary CPU activity is excluded."""
    try:
        owned = preflight.read_own_scope()
        scan = preflight.interim_process_scan(owned=owned)
    except preflight.PreflightUnavailable as exc:
        raise BaselineBankError("screening inference witness unavailable") from exc
    if scan.unreadable_pids:
        raise BaselineBankError("screening inference witness unreadable")
    findings = [item.to_dict() for item in scan.inference_like()]
    return {"basis": "interim_inference_executable_scan", "competing": bool(findings),
            "findings": findings, "ordinary_processes_ignored": True}
