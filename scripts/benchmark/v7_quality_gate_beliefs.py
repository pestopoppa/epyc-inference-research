"""Prospective ClaimTuple-shaped rows for v7 quality-gate eval runs (SC32).

The 2026-08-12 A1/A3/A4 MMLU-Pro hardened-control panel and the EVL-08
gpqa-cj1 runs predate this hook and deliberately emit zero rows — a tuple
invented on read would claim warrant the native run never captured. Successor
runs pass ``--belief-category`` to the runner; at result-finalize the runner
calls :func:`attach_accuracy_beliefs` so the native summary, rather than a
later reader, records what was scored and under which arm/config identity.

The claim this module authorizes is narrow: "arm X, suite Y, scored accuracy
Z (correct/n) under the recorded serving/sampling identity". It never claims
model quality beyond the scored slice, and the category label (BASELINE for
the anchor arm, CANDIDATE for controls) is the caller's explicit declaration,
never derived here.

Row shape mirrors the house envelope in
``scripts/benchmark/autokernel_gpu_discovery_beliefs.py`` (measurement_id,
metric, value, unit, metric_direction, category, protocol_id, reps,
reps_basis, claim, extra, measurement_sha256).
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROTOCOL_ID = "epyc.v7_quality_gate_runner.accuracy.v1"
PRODUCER_ID = "scripts.benchmark.v7_quality_gate_runner/main"
PRODUCER_PATH = "scripts/benchmark/v7_quality_gate_runner.py"
CATEGORIES = frozenset({"BASELINE", "CANDIDATE"})


class BeliefRefused(ValueError):
    """The native run summary cannot honestly produce a row."""


def content_hash(obj: Any) -> str:
    """SHA-256 over the canonical JSON encoding — byte-identical semantics to
    ``autokernel.schemas.content_hash`` (sort_keys, compact separators,
    ensure_ascii=False, NaN/Infinity refused). Implemented locally because the
    runner executes with the benchmark dir on sys.path, not the repo root."""
    return hashlib.sha256(
        json.dumps(
            obj, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BeliefRefused(message)


def _producer(runner_source_sha256: str) -> Mapping[str, str]:
    _require(
        isinstance(runner_source_sha256, str) and len(runner_source_sha256) == 64,
        "runner source sha256 must be a 64-char hex digest",
    )
    return {
        "producer_id": PRODUCER_ID,
        "path": PRODUCER_PATH,
        "sha256": runner_source_sha256,
    }


def _row(*, measurement_id: str, metric: str, value: float, unit: str,
         category: str, claim: str, reps: int, reps_basis: str,
         extra: Mapping[str, Any]) -> dict:
    row = {
        "measurement_id": measurement_id,
        "metric": metric,
        "value": value,
        "unit": unit,
        "metric_direction": "higher_better",
        "category": category,
        "protocol_id": PROTOCOL_ID,
        "reps": reps,
        "reps_basis": reps_basis,
        "claim": claim,
        "extra": dict(extra),
    }
    row["measurement_sha256"] = content_hash(row)
    return row


def attach_accuracy_beliefs(
    result: Mapping[str, Any],
    *,
    output_path: Path,
    category: str,
    runner_source_sha256: str,
    host: str,
    port: int,
    concurrency: int = 1,
    arm_config: Mapping[str, Any] | None = None,
) -> list[dict]:
    """Build producer-authored rows from the run summary in ``result``.

    The attestation hashes the manifest/summary content (``meta`` + ``suites``)
    written at collect time, in its canonical encoding, so a later reader can
    re-derive exactly what was attested. The scored denominator (``reps``) is
    read from the summary's own ``n`` — never guessed from an expected count.

    Write-once: refuses a result that already carries ``belief_measurements``.
    """
    _require(category in CATEGORIES, f"category must be one of {sorted(CATEGORIES)}")
    _require("belief_measurements" not in result,
             "belief attachment is write-once")
    meta = result.get("meta")
    suites = result.get("suites")
    _require(isinstance(meta, Mapping), "run summary must carry a meta mapping")
    _require(isinstance(suites, list) and suites, "run summary must carry suites")
    for suite in suites:
        _require(isinstance(suite, Mapping), "each suite must be a mapping")

    # The manifest/summary content written at collect time.
    manifest = {"meta": dict(meta), "suites": suites}
    attestation_sha256 = content_hash(manifest)

    pinned = meta.get("questions_pinned")
    pinned_sha256 = None
    if isinstance(pinned, str) and pinned:
        pinned_path = Path(pinned)
        if pinned_path.is_file():
            pinned_sha256 = _sha256_file(pinned_path)
    arm = str(meta.get("arm", ""))
    producer = _producer(runner_source_sha256)

    rows = []
    for suite in suites:
        suite_name = suite.get("suite")
        n = suite.get("n")
        correct = suite.get("correct")
        _require(isinstance(suite_name, str) and suite_name,
                 "suite must carry a non-empty suite name")
        _require(isinstance(n, int) and not isinstance(n, bool) and n >= 1,
                 f"{suite_name}: scored denominator n must be a positive int "
                 "(read from the run's own summary, never guessed)")
        _require(isinstance(correct, int) and not isinstance(correct, bool)
                 and 0 <= correct <= n,
                 f"{suite_name}: correct must be an int within [0, n]")
        accuracy = correct / n
        truncated = suite.get("truncated", 0)
        errors = suite.get("errors", 0)
        id_suffix = f"{arm}_{suite_name}" if arm else suite_name
        extra = {
            "date": meta.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "arm": arm,
            "category": category,
            "kernel": meta.get("kernel", ""),
            "binary": meta.get("binary", ""),
            "models": meta.get("models", ""),
            "serving": {
                "host": host,
                "port": port,
                "endpoint": meta.get("endpoint", ""),
                "template": "server-side; not observable by the runner",
            },
            "sampling": {
                "seed": meta.get("seed"),
                "stratify": meta.get("stratify"),
                "temperature": meta.get("temperature"),
                "top_p": meta.get("top_p"),
                "top_k": meta.get("top_k"),
                "enable_thinking": meta.get("enable_thinking"),
                # REQUESTED values; the completion path pins top_k=1 (greedy)
                # and never sends top_p / enable_thinking. Authority for what
                # was actually sent is `effective_request` on each
                # per-question row, not these fields.
                "sampling_fields_are_requested_not_effective": True,
                "repeats": meta.get("repeats"),
                "max_tokens": meta.get("max_tokens"),
                "concurrency": concurrency,
            },
            "prompt_set": {
                "id": pinned or "fresh_sample_at_seed",
                "path": pinned,
                "sha256": pinned_sha256,
            },
            "scored": {
                "correct": correct,
                "n": n,
                "accuracy": accuracy,
                "truncated": truncated,
                "errors": errors,
                "tier_accuracy": suite.get("per_tier"),
            },
            "capture_schema_version": meta.get("capture_schema_version", ""),
            "producer_id": producer["producer_id"],
            "producer_sha256": producer["sha256"],
            "attestation_path": str(output_path),
            "attestation_sha256": attestation_sha256,
            "attestation_locator": output_path.name,
            "arm_config": dict(arm_config or {}),
        }
        rows.append(_row(
            measurement_id=f"quality_gate_{id_suffix}_accuracy",
            metric=f"{suite_name}_accuracy",
            value=accuracy,
            unit="fraction",
            category=category,
            claim=(
                f"{category} arm {arm or '(unnamed)'} scored "
                f"{accuracy:.4f} accuracy ({correct}/{n}) on {suite_name}"
            ),
            reps=n,
            reps_basis=f"scored:{n} {suite_name} questions "
                       f"(seed {meta.get('seed')}, repeats {meta.get('repeats')})",
            extra=extra,
        ))
    return rows
