#!/usr/bin/env python3
"""Eval-suite discriminability audit (loops-and-dashboards-audit-2026-07-05 P3).

READ-ONLY, deterministic, no-model analysis over already-collected per-question
eval results. It answers one question the operator needs before trusting any
promotion gate: **can each eval suite actually detect a real effect, or is its
quality band an artifact of a suite ceiling / substring scorer / tiny sample?**

For each suite (and, nested, each ``real_task_class`` sub-suite) it computes,
pooling every per-question row it can find on disk:

* **Pass-rate distribution** — accuracy + Wilson score interval
  (``src/llm_primitives/stat_tests.py::wilson_interval``), error rate, and the
  per-run pass-rate spread (a suite whose realized accuracy swings run-to-run
  cannot gate anything).
* **Per-question brittleness** — for every qid that was evaluated in more than
  one run/seed, the variance of its 0/1 outcome across runs; the suite-level
  ``flip_rate`` (fraction of multi-run qids that ever flipped) and mean
  per-question variance.
* **Minimum-detectable-effect (MDE)** — the smallest absolute pass-rate delta a
  two-arm (baseline-vs-candidate) comparison on this suite's sample size can
  distinguish from noise, via the standard normal two-proportion power formula
  ``MDE = (z_alpha/2 + z_power) * sqrt(2 p(1-p) / n_per_arm)`` at the observed
  pass rate ``p``. Also the **effective quantum** ``1/n`` (one question flipping
  moves accuracy by this much) — the handoff's ``z ~= 0.15`` fix-the-suite line.

It flags **saturated** suites (everyone passes -> zero discriminability),
**floored** suites (everyone fails), **tiny-n / underpowered** suites (the n=2
debugbench-style baseline that tripped the -1.5 gate), and **run-unstable**
suites, then ranks all suites by a single ``discriminability_index``.

Where it feeds the pipeline
---------------------------
This audit **gates the OP-1 promotion-reachability sign-off**: a promotion
threshold calibrated against a saturated or underpowered suite is calibrated
against noise, so OP-1 must not certify a suite whose effective quantum exceeds
the target effect (default 0.15) or whose MDE cannot resolve the effect the gate
claims to detect. The output is an OBSERVATION in the MEASUREMENT.md sense
(journal/log-derived, no protocol id): it steers which suites are trustworthy,
it does not itself gate a keep/revert.

Reads (auto-discovered under ``orchestration/reports/``):
* ``**/question_ledger.jsonl`` — schema ``real_suite_v1_eval_question_ledger_row.v1``
  (fields: ``suite``, ``qid``, ``correct``, ``error``, ``scoring_method``,
  ``real_task_class``, ``partition``, ``calibration_id`` ...).
* ``**/question_results.jsonl`` — the compact prompt-free twin (used only when a
  run directory has no ledger, to avoid double-counting the same rows).

Usage:
    python scripts/analysis/eval_suite_discriminability.py            # markdown to stdout
    python scripts/analysis/eval_suite_discriminability.py --json     # + JSON to stdout
    python scripts/analysis/eval_suite_discriminability.py --out-dir orchestration/reports/eval_suite_discriminability_$(date -u +%Y%m%dT%H%M%SZ)
    python scripts/analysis/eval_suite_discriminability.py --input path/to/question_ledger.jsonl --target-effect 0.15
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

# Repo root on path so `from src.llm_primitives.stat_tests import ...` resolves
# whether this runs as a CLI or is imported with `scripts/analysis` on sys.path
# (the fixture test imports it that way). Matches the convention in the sibling
# analysis scripts (reviewer_calibration_report.py, run_paired_ab.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.llm_primitives.stat_tests import (  # noqa: E402
    DEFAULT_WILSON_Z,
    wilson_interval,
)

DEFAULT_REPORTS_ROOT = _REPO_ROOT / "orchestration" / "reports"
# question_ledger is the richer, preferred source; question_results is its
# compact twin over the identical row set — read the twin only when a run dir
# lacks the ledger so the same evaluation is never counted twice.
LEDGER_NAME = "question_ledger.jsonl"
RESULTS_NAME = "question_results.jsonl"
REPORT_SCHEMA = "eval_suite_discriminability_report.v2"
_PRODUCER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_private_once(path: Path, raw: bytes) -> None:
    """Create a private, create-once snapshot; this is not filesystem immutability."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if not path.is_file() or path.is_symlink() or path.read_bytes() != raw:
            raise ValueError("snapshot path already contains different or non-regular bytes")
        path.chmod(0o600)
        return
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"snapshot directory is not a real directory: {path}")
    path.chmod(0o700)


def _seal_report(report: dict[str, Any], inputs: list[dict[str, Any]],
                 report_path: Path) -> dict[str, Any]:
    """Bind the report to private create-once snapshots of parsed inputs/source."""
    if not inputs:
        return report
    producer = Path(__file__).resolve()
    try:
        producer_path = producer.relative_to(_REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("producer must be inside its repository root") from exc
    destination = report_path.resolve()
    producer_bytes = producer.read_bytes()
    producer_sha = hashlib.sha256(producer_bytes).hexdigest()
    if producer_sha != _PRODUCER_SOURCE_SHA256:
        return report  # loaded code and the source being recorded are not the same bytes
    by_path: dict[str, dict[str, Any]] = {}
    for item in inputs:
        path = item["path"]
        prior = by_path.get(path)
        if prior is not None:
            if any(prior.get(key) != item.get(key) for key in
                   ("sha256", "byte_count", "row_count", "normalized_row_count", "_raw_bytes")):
                raise ValueError(f"conflicting bytes or counts for repeated input path: {path}")
            continue
        by_path[path] = item
    raw_inputs = sorted(by_path.values(), key=lambda item: item["path"])
    manifest = [{key: value for key, value in item.items() if key != "_raw_bytes"}
                for item in raw_inputs]
    snapshot_key = _canonical_sha256({"body": report, "inputs": manifest,
                                      "producer_sha256": producer_sha})
    snapshot_dir = destination.with_name(destination.name + ".native") / snapshot_key
    snapshot_root = destination.with_name(destination.name + ".native")
    _ensure_private_dir(snapshot_root)
    _ensure_private_dir(snapshot_dir)
    for index, (entry, raw_entry) in enumerate(zip(manifest, raw_inputs, strict=True)):
        raw_bytes = raw_entry.get("_raw_bytes")
        if not isinstance(raw_bytes, bytes) or hashlib.sha256(raw_bytes).hexdigest() != entry["sha256"]:
            raise ValueError("input bytes retained for snapshot differ from the parsed-byte digest")
        snapshot = snapshot_dir / f"input-{index:04d}-{entry['sha256']}.raw"
        _write_private_once(snapshot, raw_bytes)
        entry["snapshot_path"] = str(snapshot)
    producer_snapshot = snapshot_dir / f"producer-{producer_sha}.py"
    _write_private_once(producer_snapshot, producer_bytes)
    provenance: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "producer_path": producer_path,
        "producer_sha256": producer_sha,
        "producer_snapshot_path": str(producer_snapshot),
        "input_root": "epyc-orchestrator",
        "inputs": manifest,
        "missing_inputs": [],
        "analysis_config_sha256": _canonical_sha256(report.get("config", {})),
        "report_body_sha256": _canonical_sha256(report),
    }
    provenance["record_id"] = _canonical_sha256(provenance)
    report["native_provenance"] = provenance
    provenance["report_sha256"] = _canonical_sha256(report)
    return report


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AuditConfig:
    """Thresholds + power parameters for the discriminability audit."""

    alpha: float = 0.05  # two-sided significance for the MDE
    power: float = 0.80  # 1 - beta for the MDE
    target_effect: float = 0.15  # the pass-rate delta the operator wants to detect
    saturation_high: float = 0.95  # p >= this -> saturated (ceiling, ~0 discriminability)
    saturation_low: float = 0.05  # p <= this -> floored (floor, ~0 discriminability)
    min_n: int = 5  # n_unique below this -> tiny-n / underpowered
    quantum_gate: float = 0.15  # effective quantum (1/n) above this -> underpowered
    run_spread_gate: float = 0.30  # max-min per-run pass-rate above this -> run-unstable
    error_dominated_gate: float = 0.5  # run error_rate >= this -> excluded from run-spread/flip


# ---------------------------------------------------------------------------
# Numerics (stdlib-only; no numpy/scipy)
# ---------------------------------------------------------------------------
def inv_norm(p: float) -> float:
    """Inverse standard-normal CDF (quantile) via Acklam's rational approximation.

    Deterministic, stdlib-only, ~1e-9 absolute accuracy over p in (0, 1).
    Used to turn ``alpha``/``power`` into z-multipliers for the MDE.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"inv_norm domain is (0,1), got {p}")
    a = [
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def two_proportion_mde(
    p: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float | None:
    """Minimum detectable effect (absolute pass-rate delta) for a two-arm test.

    Normal approximation with equal per-arm sample size, evaluated at the
    observed pass rate ``p``::

        MDE = (z_{alpha/2} + z_{power}) * sqrt(2 * p*(1-p) / n_per_arm)

    Returns ``None`` for a degenerate sample where the estimate is
    uninformative: ``n_per_arm < 2`` (no variance to estimate), or ``p`` exactly
    0/1 (zero observed variance — a boundary the *saturation* flag handles, not
    this number). The caller forces such suites to zero discriminability.
    """
    if n_per_arm < 2:
        return None
    if p <= 0.0 or p >= 1.0:
        return None
    z_alpha = inv_norm(1.0 - alpha / 2.0)
    z_power = inv_norm(power)
    return (z_alpha + z_power) * math.sqrt(2.0 * p * (1.0 - p) / n_per_arm)


# ---------------------------------------------------------------------------
# Loading / normalization
# ---------------------------------------------------------------------------
def _as_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "correct", "pass", "passed"}
    return False


def normalize_row(raw: dict[str, Any], run_id: str, source: str) -> dict[str, Any] | None:
    """Reduce a raw per-question ledger/result row to the audit's fields."""
    qid = raw.get("qid") or raw.get("id") or raw.get("question_id")
    if qid is None:
        return None
    suite = raw.get("suite") or "unknown"
    task_class = raw.get("real_task_class") or raw.get("task_class")
    return {
        "suite": str(suite),
        "task_class": str(task_class) if task_class else None,
        "qid": str(qid),
        "correct": _as_bool(raw.get("correct")),
        "error": _as_bool(raw.get("error")),
        "scoring_method": raw.get("scoring_method"),
        "partition": raw.get("partition"),
        "run_id": str(raw.get("calibration_id") or run_id),
        "source": source,
    }


def discover_inputs(reports_root: Path) -> list[Path]:
    """Find per-question JSONL files, one per run directory (ledger preferred)."""
    root = Path(reports_root)
    ledgers = {p.parent: p for p in root.glob(f"**/{LEDGER_NAME}")}
    results = {p.parent for p in root.glob(f"**/{RESULTS_NAME}")}
    chosen: list[Path] = list(ledgers.values())
    # Add compact results only for run dirs that have no ledger (no double-count).
    for parent in sorted(results - set(ledgers)):
        chosen.append(parent / RESULTS_NAME)
    return sorted(chosen)


def _expand_input(spec: str) -> list[Path]:
    p = Path(spec)
    if p.is_dir():
        return discover_inputs(p)
    matches = [Path(m) for m in glob.glob(spec)]
    return sorted(m for m in matches if m.is_file())


def load_rows(paths: list[Path], *, input_manifest: list[dict[str, Any]] | None = None
              ) -> tuple[list[dict[str, Any]], list[str]]:
    """Load + normalize rows from JSONL files. Returns (rows, warnings)."""
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in paths:
        run_id = path.parent.name
        source = str(path)
        try:
            raw_bytes = path.read_bytes()
            text = raw_bytes.decode("utf-8")
        except OSError as exc:  # noqa: BLE001
            warnings.append(f"unreadable: {path} ({exc})")
            continue
        except UnicodeDecodeError as exc:
            warnings.append(f"unreadable: {path} ({exc})")
            continue
        rows_before = len(rows)
        nonempty_lines = 0
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            nonempty_lines += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                warnings.append(f"bad JSON: {path}:{lineno}")
                continue
            norm = normalize_row(raw, run_id, source)
            if norm is None:
                warnings.append(f"no qid: {path}:{lineno}")
                continue
            rows.append(norm)
        if input_manifest is not None:
            resolved = path.resolve()
            try:
                source_path = resolved.relative_to(_REPO_ROOT).as_posix()
            except ValueError:
                source_path = resolved.as_posix()
            input_manifest.append({"path": source_path,
                                   "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                                   "byte_count": len(raw_bytes),
                                   "row_count": nonempty_lines,
                                   "normalized_row_count": len(rows) - rows_before,
                                   "_raw_bytes": raw_bytes})
    return rows, warnings


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def compute_brittleness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-question outcome variance across runs/seeds.

    A qid seen with a genuine (non-error) verdict in >1 run contributes;
    ``flip_rate`` is the fraction of such qids whose 0/1 outcome was not
    constant across runs, ``mean_qid_variance`` the mean population variance
    of those per-qid outcome vectors.

    ``error=True`` rows are excluded from the outcome vectors before flips are
    counted. An errored request never got a model answer at all (backend
    circuit-open, harness no-progress-nudge escalation, timeout, ...), so it
    carries no signal about the *model's* behavior; pooling it in as an
    implicit 0 makes an infrastructure outage in one run look like a
    disagreement with a clean run in another, inflating ``flip_rate`` for a
    reason that has nothing to do with scorer or model nondeterminism. Root
    cause of the RTG-16 "0/50 vs 35/50 on the same 50 qids" instability:
    ``real_suite_v1_eval_20260706T192007Z`` has ``error=True`` on 50/50 rows
    (12 backend-circuit-open + 38 harness no-progress-nudge, at
    ``eval_concurrency=3``) — a total-outage run, not a second independent
    measurement of the suite. ``pass_rate``/MDE accounting is untouched by
    this (an errored task still scores 0 there, per the ETR-1 era
    ``task_failed`` convention) — only the cross-run *flip* diagnostic, whose
    job is specifically to detect scorer/model noise, is corrected to stop
    reading an absent observation as a contradictory one.
    """
    by_qid: dict[str, list[int]] = defaultdict(list)
    n_error_excluded = 0
    for r in rows:
        if r.get("error"):
            n_error_excluded += 1
            continue
        by_qid[r["qid"]].append(1 if r["correct"] else 0)
    multirun = {q: outs for q, outs in by_qid.items() if len(outs) > 1}
    if not multirun:
        return {
            "measured": False,
            "n_multirun_qids": 0,
            "flip_rate": None,
            "mean_qid_variance": None,
            "max_qid_variance": None,
            "n_error_excluded": n_error_excluded,
        }
    variances = []
    flips = 0
    for outs in multirun.values():
        mean = sum(outs) / len(outs)
        var = sum((o - mean) ** 2 for o in outs) / len(outs)  # population variance
        variances.append(var)
        if min(outs) != max(outs):
            flips += 1
    return {
        "measured": True,
        "n_multirun_qids": len(multirun),
        "flip_rate": flips / len(multirun),
        "mean_qid_variance": sum(variances) / len(variances),
        "max_qid_variance": max(variances),
        "n_error_excluded": n_error_excluded,
    }


def _per_run(rows: list[dict[str, Any]], error_dominated_gate: float) -> dict[str, dict[str, Any]]:
    """Per-run counters, including an error-excluded pass rate.

    ``pass_rate`` is the raw, error-inclusive rate (unchanged, still used for
    the group-level accounting). ``pass_rate_excl_errors`` is ``None`` when
    every row in the run errored (no signal at all); ``error_dominated`` marks
    a run whose error rate meets ``error_dominated_gate`` — such a run is
    excluded from the run-spread / run-instability comparison in
    ``analyze_group`` (see its docstring) because it measures backend/harness
    availability, not suite discriminability.
    """
    out: dict[str, dict[str, Any]] = {}
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    for run_id, rr in by_run.items():
        n = len(rr)
        errs = sum(1 for r in rr if r["error"])
        c = sum(1 for r in rr if r["correct"])
        n_scored = n - errs
        error_rate = (errs / n) if n else 0.0
        out[run_id] = {
            "n": n,
            "correct": c,
            "errors": errs,
            "n_scored": n_scored,
            "error_rate": error_rate,
            "pass_rate": (c / n) if n else 0.0,
            "pass_rate_excl_errors": (c / n_scored) if n_scored else None,
            "error_dominated": error_rate >= error_dominated_gate,
        }
    return out


def analyze_group(name: str, rows: list[dict[str, Any]], cfg: AuditConfig) -> dict[str, Any]:
    """Discriminability analysis for one suite / task-class group."""
    n = len(rows)
    unique_qids = {r["qid"] for r in rows}
    n_unique = len(unique_qids)
    correct = sum(1 for r in rows if r["correct"])
    errors = sum(1 for r in rows if r["error"])
    p = (correct / n) if n else 0.0
    ci_lo, ci_hi = wilson_interval(correct, n, z=DEFAULT_WILSON_Z)

    per_run = _per_run(rows, cfg.error_dominated_gate)
    # Run-spread / run-instability is a diagnostic for scorer/model noise
    # across runs, so it must be computed only over runs that actually
    # produced a scored answer for at least one question. A run whose error
    # rate meets error_dominated_gate (e.g. a total backend outage) is
    # excluded here — it is reported separately via error_dominated_runs so
    # it is never silently dropped, only kept out of the noise comparison it
    # cannot speak to. See compute_brittleness docstring / RTG-16.
    scored_runs = {
        rid: v for rid, v in per_run.items()
        if v["pass_rate_excl_errors"] is not None and not v["error_dominated"]
    }
    error_dominated_runs = sorted(rid for rid, v in per_run.items() if v["error_dominated"])
    run_rates = [v["pass_rate_excl_errors"] for v in scored_runs.values()]
    run_spread = (max(run_rates) - min(run_rates)) if len(run_rates) > 1 else 0.0
    run_stability_unmeasurable = len(per_run) > 1 and len(scored_runs) < 2

    # Per-arm sample size for a baseline-vs-candidate comparison = the count of
    # distinct questions the suite offers (each run reuses the same question set).
    n_per_arm = n_unique
    mde = two_proportion_mde(p, n_per_arm, alpha=cfg.alpha, power=cfg.power)
    quantum = (1.0 / n_unique) if n_unique else 1.0

    brittle = compute_brittleness(rows)

    saturated = p >= cfg.saturation_high
    floored = p <= cfg.saturation_low
    degenerate = saturated or floored
    tiny_n = n_unique < cfg.min_n
    underpowered_quantum = quantum > cfg.quantum_gate
    underpowered_mde = (mde is None) or (mde > cfg.target_effect)
    run_unstable = run_spread > cfg.run_spread_gate
    underpowered = tiny_n or underpowered_quantum or underpowered_mde

    # Single ranking index in [0, 1]. Zero when the suite is degenerate
    # (ceiling/floor => cannot distinguish anything) or has no usable MDE;
    # otherwise how well its MDE resolves the target effect, de-rated by
    # cross-run brittleness (a suite that flips run-to-run is untrustworthy).
    if degenerate or mde is None or mde <= 0.0:
        discriminability = 0.0
    else:
        power_to_resolve = min(1.0, cfg.target_effect / mde)
        reliability = 1.0 - brittle["flip_rate"] if brittle["measured"] else 1.0
        discriminability = round(power_to_resolve * reliability, 4)

    flags: list[str] = []
    if saturated:
        flags.append("saturated")
    if floored:
        flags.append("floored")
    if tiny_n:
        flags.append("tiny_n")
    if underpowered_quantum:
        flags.append("underpowered_quantum")
    if underpowered_mde:
        flags.append("underpowered_mde")
    if run_unstable:
        flags.append("run_unstable")
    if brittle["measured"] and brittle["flip_rate"] and brittle["flip_rate"] > 0.2:
        flags.append("brittle")
    if not brittle["measured"]:
        flags.append("brittleness_unmeasured")
    if error_dominated_runs:
        flags.append("error_dominated_runs_excluded")
    if run_stability_unmeasurable:
        flags.append("run_stability_unmeasurable")

    return {
        "group": name,
        "n": n,
        "n_unique_qids": n_unique,
        "n_runs": len(per_run),
        "correct": correct,
        "errors": errors,
        "error_rate": (errors / n) if n else 0.0,
        "pass_rate": p,
        "pass_rate_excl_errors": (correct / (n - errors)) if (n - errors) > 0 else None,
        "wilson_ci": [ci_lo, ci_hi],
        "wilson_width": ci_hi - ci_lo,
        "effective_quantum": quantum,
        "mde": mde,
        "mde_target_effect": cfg.target_effect,
        "n_per_arm": n_per_arm,
        "run_spread": run_spread,
        "error_dominated_runs": error_dominated_runs,
        "run_stability_unmeasurable": run_stability_unmeasurable,
        "brittleness": brittle,
        "per_run": per_run,
        "saturated": saturated,
        "floored": floored,
        "tiny_n": tiny_n,
        "underpowered": underpowered,
        "run_unstable": run_unstable,
        "discriminability_index": discriminability,
        "flags": flags,
    }


def _group(rows: list[dict[str, Any]], key) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


def build_report(
    rows: list[dict[str, Any]],
    cfg: AuditConfig,
    inputs: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """Assemble the full ranked discriminability report."""
    suites = [
        analyze_group(name, grp, cfg)
        for name, grp in _group(rows, lambda r: r["suite"]).items()
    ]
    suites.sort(key=lambda s: (s["discriminability_index"], s["n_unique_qids"]), reverse=True)

    task_classes = [
        analyze_group(name, grp, cfg)
        for name, grp in _group(
            rows,
            lambda r: f"{r['suite']}::{r['task_class'] or 'none'}",
        ).items()
    ]
    task_classes.sort(
        key=lambda s: (s["discriminability_index"], s["n_unique_qids"]), reverse=True
    )

    def _count(groups, pred):
        return sum(1 for g in groups if pred(g))

    summary = {
        "n_rows": len(rows),
        "n_suites": len(suites),
        "n_task_classes": len(task_classes),
        "n_saturated_suites": _count(suites, lambda g: g["saturated"]),
        "n_floored_suites": _count(suites, lambda g: g["floored"]),
        "n_underpowered_suites": _count(suites, lambda g: g["underpowered"]),
        "n_run_unstable_suites": _count(suites, lambda g: g["run_unstable"]),
        "n_underpowered_task_classes": _count(task_classes, lambda g: g["underpowered"]),
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "measurement_class": "OBSERVATION",
        "config": asdict(cfg),
        "inputs": inputs,
        "warnings": warnings,
        "summary": summary,
        "suites": suites,
        "task_classes": task_classes,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _row_md(g: dict[str, Any]) -> str:
    br = g["brittleness"]
    flip = _fmt(br["flip_rate"], 2) if br["measured"] else "n/a"
    ci = f"[{g['wilson_ci'][0]:.2f},{g['wilson_ci'][1]:.2f}]"
    return (
        f"| {g['group']} | {g['n_unique_qids']} | {g['n_runs']} | "
        f"{g['pass_rate']:.3f} | {ci} | {g['effective_quantum']:.3f} | "
        f"{_fmt(g['mde'])} | {flip} | {g['discriminability_index']:.3f} | "
        f"{', '.join(g['flags']) or '-'} |"
    )


def render_markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    s = report["summary"]
    lines: list[str] = []
    lines.append("# Eval-Suite Discriminability Audit")
    lines.append("")
    lines.append(
        f"_Generated {report['generated_at']} · "
        f"class **{report['measurement_class']}** (no protocol id — steers "
        f"attention, does not gate keep/revert)_"
    )
    lines.append("")
    lines.append(
        f"**Config**: alpha={cfg['alpha']}, power={cfg['power']}, "
        f"target_effect={cfg['target_effect']}, "
        f"saturation>={cfg['saturation_high']}, floor<={cfg['saturation_low']}, "
        f"min_n={cfg['min_n']}, quantum_gate={cfg['quantum_gate']}, "
        f"run_spread_gate={cfg['run_spread_gate']}, "
        f"error_dominated_gate={cfg['error_dominated_gate']}."
    )
    lines.append("")
    lines.append(
        f"**Corpus**: {s['n_rows']} per-question rows · {s['n_suites']} suites · "
        f"{s['n_task_classes']} task-classes. "
        f"Flagged: {s['n_saturated_suites']} saturated, "
        f"{s['n_floored_suites']} floored, "
        f"{s['n_underpowered_suites']} underpowered, "
        f"{s['n_run_unstable_suites']} run-unstable."
    )
    lines.append("")
    header = (
        "| suite | n(uq) | runs | pass | Wilson95 | quantum | MDE | flip | "
        "discrim | flags |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"

    lines.append("## Suites (ranked by discriminability)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for g in report["suites"]:
        lines.append(_row_md(g))
    lines.append("")

    lines.append("## Task-class sub-suites (ranked by discriminability)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for g in report["task_classes"]:
        lines.append(_row_md(g))
    lines.append("")

    flagged = [
        g
        for g in report["suites"] + report["task_classes"]
        if g["flags"] and g["flags"] != ["brittleness_unmeasured"]
    ]
    if flagged:
        lines.append("## Flagged — fix before trusting a gate calibrated here")
        lines.append("")
        for g in flagged:
            why = []
            if g["saturated"]:
                why.append(f"saturated (pass={g['pass_rate']:.2f}) -> 0 discriminability")
            if g["floored"]:
                why.append(f"floored (pass={g['pass_rate']:.2f}) -> 0 discriminability")
            if g["tiny_n"]:
                why.append(f"tiny-n (n={g['n_unique_qids']} < {cfg['min_n']})")
            if g["effective_quantum"] > cfg["quantum_gate"]:
                why.append(
                    f"quantum {g['effective_quantum']:.2f} > {cfg['quantum_gate']} "
                    "(one flip moves accuracy more than the target effect)"
                )
            if g["mde"] is None or g["mde"] > cfg["target_effect"]:
                why.append(f"MDE {_fmt(g['mde'])} cannot resolve {cfg['target_effect']}")
            if g["run_unstable"]:
                why.append(f"run-unstable (spread {g['run_spread']:.2f})")
            if g["error_dominated_runs"]:
                why.append(
                    "error-dominated run(s) excluded from run-spread/flip: "
                    + ", ".join(g["error_dominated_runs"])
                )
            if g["run_stability_unmeasurable"]:
                why.append(
                    "run stability unmeasurable (fewer than 2 runs with any scored answer)"
                )
            lines.append(f"- **{g['group']}**: {'; '.join(why)}")
        lines.append("")

    lines.append("## OP-1 promotion-reachability gate")
    lines.append("")
    lines.append(
        "A promotion threshold is only meaningful on a suite whose MDE can "
        "resolve the effect it claims to detect. **OP-1 must not certify** a "
        "suite flagged `saturated`, `floored`, `tiny_n`, or `underpowered_*` "
        "above: its gate would be calibrated against noise. Suites with "
        "`discriminability_index` near 1.0 and no flags are promotion-reachable."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_config(args: argparse.Namespace) -> AuditConfig:
    return AuditConfig(
        alpha=args.alpha,
        power=args.power,
        target_effect=args.target_effect,
        saturation_high=args.saturation_high,
        saturation_low=args.saturation_low,
        min_n=args.min_n,
        quantum_gate=args.quantum_gate,
        run_spread_gate=args.run_spread_gate,
        error_dominated_gate=args.error_dominated_gate,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument(
        "--reports-root",
        default=str(DEFAULT_REPORTS_ROOT),
        help="root scanned for **/question_ledger.jsonl (default: orchestration/reports)",
    )
    ap.add_argument(
        "--input",
        action="append",
        default=None,
        help="explicit file/dir/glob of per-question JSONL (repeatable; overrides scan)",
    )
    ap.add_argument("--alpha", type=float, default=AuditConfig.alpha)
    ap.add_argument("--power", type=float, default=AuditConfig.power)
    ap.add_argument("--target-effect", type=float, default=AuditConfig.target_effect)
    ap.add_argument("--saturation-high", type=float, default=AuditConfig.saturation_high)
    ap.add_argument("--saturation-low", type=float, default=AuditConfig.saturation_low)
    ap.add_argument("--min-n", type=int, default=AuditConfig.min_n)
    ap.add_argument("--quantum-gate", type=float, default=AuditConfig.quantum_gate)
    ap.add_argument("--run-spread-gate", type=float, default=AuditConfig.run_spread_gate)
    ap.add_argument(
        "--error-dominated-gate",
        type=float,
        default=AuditConfig.error_dominated_gate,
        help="run error_rate >= this excludes it from run-spread/flip (default: 0.5)",
    )
    ap.add_argument("--json", action="store_true", help="also print the JSON report to stdout")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="write report.json + report.md here (default: stdout only, no writes)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _build_config(args)

    if args.input:
        paths: list[Path] = []
        for spec in args.input:
            paths.extend(_expand_input(spec))
        paths = sorted(set(paths))
    else:
        paths = discover_inputs(Path(args.reports_root))

    if not paths:
        print("no per-question eval JSONL found", file=sys.stderr)
        return 2

    input_manifest: list[dict[str, Any]] = []
    rows, warnings = load_rows(paths, input_manifest=input_manifest)
    if not rows:
        print("inputs contained no usable per-question rows", file=sys.stderr)
        for w in warnings:
            print(f"  warn: {w}", file=sys.stderr)
        return 2

    report = build_report(rows, cfg, [str(p) for p in paths], warnings)

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        report = _seal_report(report, input_manifest, out / "report.json")
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        md = render_markdown(report)
        (out / "report.md").write_text(md + "\n")
        print(f"wrote {out / 'report.json'}")
        print(f"wrote {out / 'report.md'}")
    else:
        md = render_markdown(report)
        print(md)
        if args.json:
            print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
