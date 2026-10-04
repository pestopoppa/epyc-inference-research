"""Exact original serving-gate rows for a propagated whole-source candidate.

The intended source target uses the loop's existing
``accumulate.classify_serving`` verdict.  Every measured non-author target uses
the cross-workload gate's conservative interim non-inferiority rule.  A target
may be excused from measurement only by an explicit INERT blast-radius witness.
Missing or unit-incompatible calibration never passes a row.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from . import accumulate

SCHEMA = "epyc.autokernel.whole_source_serving_validation.v1"
DEBT_SCHEMA = "epyc.autokernel.whole_source_serving_validation_debt.v1"
REFERENCE_SCHEMA = "epyc.autokernel.whole_source_serving_validation_reference.v1"
MAX_BYTES = 16 << 20


class SurfaceValidationRefused(ValueError):
    pass


def shared_git_commit(target_repo: Path, source_repo: Path,
                      commit: str) -> tuple[Path, str]:
    """Prove a retained commit exists under the target's same Git object owner."""
    target_repo, source_repo = Path(target_repo).resolve(), Path(source_repo).resolve()
    if (not isinstance(commit, str) or len(commit) != 40
            or any(char not in "0123456789abcdef" for char in commit)):
        raise SurfaceValidationRefused("shared source commit is invalid")

    def git(repo: Path, *argv: str) -> str:
        done = subprocess.run(["git", "-C", str(repo), *argv], capture_output=True,
                              text=True, timeout=60)
        if done.returncode:
            raise SurfaceValidationRefused(
                f"shared source Git identity unavailable: {done.stderr[-512:]}")
        return done.stdout.strip()

    def common(repo: Path) -> Path:
        path = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
        return path.resolve(strict=True)

    try:
        target_stat = common(target_repo).stat()
        source_stat = common(source_repo).stat()
    except OSError as exc:
        raise SurfaceValidationRefused("shared source Git directory is unavailable") from exc
    if (target_stat.st_dev, target_stat.st_ino) != (source_stat.st_dev, source_stat.st_ino):
        raise SurfaceValidationRefused("shared source does not belong to target Git ownership")
    tree = git(source_repo, "rev-parse", f"{commit}^{{tree}}")
    if len(tree) != 40 or any(char not in "0123456789abcdef" for char in tree):
        raise SurfaceValidationRefused("shared source tree identity is invalid")
    return source_repo, tree


def shared_source_checkout(target_repo: Path, source_repo: Path,
                           commit: str) -> tuple[Path, str]:
    """Additionally prove the live source checkout is exact and clean before build."""
    source_repo, tree = shared_git_commit(target_repo, source_repo, commit)
    head = subprocess.run(["git", "-C", str(source_repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, timeout=60)
    status = subprocess.run(
        ["git", "-C", str(source_repo), "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True, text=True, timeout=60)
    if head.returncode or head.stdout.strip() != commit:
        raise SurfaceValidationRefused("shared source checkout moved from retained commit")
    if status.returncode or status.stdout:
        raise SurfaceValidationRefused("shared source checkout is not clean")
    return source_repo, tree


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def original_anchor_commit(build: Path, repo: Path) -> str:
    """Recover the existing build owner's original source commit without relabelling it."""
    from . import surface_fold
    build, repo = Path(build).resolve(), Path(repo).resolve()
    provenance = build / "provenance.json"
    identity = build / "IDENTITY.json"
    try:
        if provenance.exists():
            body = json.loads(surface_fold.bounded_regular_bytes(provenance, 1 << 20))
            commit = body.get("champion_commit") if isinstance(body, dict) else None
        else:
            body = json.loads(surface_fold.bounded_regular_bytes(identity, 1 << 20))
            required = {"schema", "kind", "head", "source", "source_status", "build",
                        "build_dir", "files"}
            if (not isinstance(body, dict) or set(body) != required
                    or body.get("schema") != "epyc.champion-candidate-build.v1"
                    or Path(str(body.get("source"))).resolve() != repo):
                raise SurfaceValidationRefused("original experimental build identity differs")
            commit = body.get("head")
    except (OSError, json.JSONDecodeError) as exc:
        raise SurfaceValidationRefused("original anchor source identity is unavailable") from exc
    if (not isinstance(commit, str) or len(commit) != 40
            or any(char not in "0123456789abcdef" for char in commit)):
        raise SurfaceValidationRefused("original anchor source commit is invalid")
    return commit


def _witnesses_inert(blast_radius_row: Mapping[str, Any] | None) -> bool:
    """Recognise only the closed, witnessed structural-excuse record.

    The full blast-radius producer owns deeper semantic validation.  This gate
    deliberately checks the fields on which its decision depends and rejects an
    open or witness-free lookalike.
    """
    if not isinstance(blast_radius_row, Mapping):
        return False
    witnesses = blast_radius_row.get("witnesses")
    tier = blast_radius_row.get("tier")
    if (blast_radius_row.get("schema") != "epyc.autokernel.blast_radius_row.v1"
            or blast_radius_row.get("classification") != "INERT"
            or blast_radius_row.get("decision_class") != "INERT"
            or tier not in {"T0", "T1"}
            or not isinstance(witnesses, Mapping)):
        return False

    def digest(name: str) -> bool:
        value = witnesses.get(name)
        return (isinstance(value, str) and len(value) == 64
                and all(char in "0123456789abcdef" for char in value))

    if not digest("shape_envelope_hash"):
        return False
    if tier == "T0":
        return digest("dso_digests_anchor") and digest("dso_digests_candidate")
    return (digest("coverage_profile_sha256_anchor")
            and digest("coverage_profile_sha256_candidate"))


def classify(comparison: Mapping[str, Any], *, intended_target: bool,
             blast_radius_row: Mapping[str, Any] | None = None) -> str:
    """Return the existing owner's exact three-valued validation disposition."""
    from . import serving
    if not intended_target and _witnesses_inert(blast_radius_row):
        return "passed"
    if not isinstance(comparison, Mapping) or not (
            comparison.get("schema") == "epyc.autokernel.serving_ab.v1"
            or serving.comparison_instrument_matches(comparison,
                instrument=serving.MATCHED_INSTRUMENT, pairs=comparison.get("pairs"))):
        raise SurfaceValidationRefused("validation requires an original serving A/B row")
    effect = comparison.get("effect")
    effect_pct = comparison.get("effect_pct")
    floor = comparison.get("noise_floor_pct")
    decisive = comparison.get("decisive")
    if (not isinstance(effect, (int, float)) or isinstance(effect, bool)
            or not math.isfinite(float(effect))
            or not isinstance(effect_pct, (int, float)) or isinstance(effect_pct, bool)
            or not math.isfinite(float(effect_pct))
            or not math.isclose(float(effect_pct), float(effect) * 100.0,
                                rel_tol=1e-12, abs_tol=1e-12)
            or not isinstance(floor, (int, float)) or isinstance(floor, bool)
            or not math.isfinite(float(floor)) or float(floor) < 0
            or type(decisive) is not bool):
        return "pending"
    if intended_target:
        return ("passed" if accumulate.classify_serving(dict(comparison))
                is accumulate.Outcome.PROMOTE else "failed")
    try:
        serving.check_unit(comparison.get("floor_unit"), comparison.get("effect_unit"),
                           what="cross-workload validation floor")
    except serving.FloorUnitMismatch as exc:
        raise SurfaceValidationRefused(str(exc)) from exc
    # AKX-P2a interim k_delta=1 rule.  A decisive negative still refuses, and
    # the lower-edge clause effect_pct - F < -delta simplifies to effect_pct < 0
    # when delta = k_delta * F and k_delta = 1.
    # This intentionally refuses even a sub-floor negative until the e-process
    # specified by the cross-workload handoff replaces this interim statistic.
    k_delta = 1.0
    delta = k_delta * float(floor)
    decisive_negative = bool(decisive) and float(effect_pct) < 0
    lower_edge_breaches_margin = float(effect_pct) - float(floor) < -delta
    return "failed" if decisive_negative or lower_edge_breaches_margin else "passed"


def row(*, source_commit: str, source_tree: str, source_keep_ids: list[str],
        target: Mapping[str, Any], original_anchor: Mapping[str, str],
        candidate_anchor: Mapping[str, str], request_digest: str,
        recipe_execution_digest: str, comparison: Mapping[str, Any],
        intended_target: bool) -> dict[str, Any]:
    values = {
        "source_commit": source_commit, "source_tree": source_tree,
        "source_keep_ids": list(source_keep_ids), "target": dict(target),
        "original_anchor": dict(original_anchor), "candidate_anchor": dict(candidate_anchor),
        "request_digest": request_digest,
        "recipe_execution_digest": recipe_execution_digest,
        "comparison": dict(comparison), "intended_target": intended_target,
    }
    result = classify(comparison, intended_target=intended_target)
    body = {"schema": SCHEMA, **values, "disposition": result}
    body["row_digest"] = _digest(body)
    return validate(body)


def debt(*, source_commit: str, source_tree: str, source_keep_ids: list[str],
         target: Mapping[str, Any], original_anchor: Mapping[str, Any],
         candidate_anchor: Mapping[str, str], request_digest: str,
         recipe_execution_digest: str, reason: str, failure: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(reason, str) or not reason or len(reason) > 1024 \
            or not isinstance(failure, Mapping):
        raise SurfaceValidationRefused("validation debt reason/evidence is malformed")
    body = {"schema": DEBT_SCHEMA, "source_commit": source_commit,
        "source_tree": source_tree, "source_keep_ids": list(source_keep_ids),
        "target": dict(target), "original_anchor": dict(original_anchor),
        "candidate_anchor": dict(candidate_anchor), "request_digest": request_digest,
        "recipe_execution_digest": recipe_execution_digest, "disposition": "pending",
        "reason": reason, "failure": dict(failure)}
    body["row_digest"] = _digest(body)
    return validate_debt(body)


def validate(value: Any) -> dict[str, Any]:
    fields = {"schema", "source_commit", "source_tree", "source_keep_ids", "target",
              "original_anchor", "candidate_anchor", "request_digest",
              "recipe_execution_digest", "comparison", "intended_target",
              "disposition", "row_digest"}
    if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != SCHEMA:
        raise SurfaceValidationRefused("whole-source validation row has an open schema")
    body = json.loads(json.dumps(value, allow_nan=False))
    for key in ("source_commit", "source_tree", "request_digest", "recipe_execution_digest"):
        raw = body[key]
        expected_length = 40 if key in {"source_commit", "source_tree"} else 64
        if not isinstance(raw, str) or len(raw) != expected_length:
            raise SurfaceValidationRefused(f"invalid {key}")
        if any(char not in "0123456789abcdef" for char in raw):
            raise SurfaceValidationRefused(f"invalid {key}")
    if (not isinstance(body["source_keep_ids"], list)
            or not body["source_keep_ids"]
            or len(body["source_keep_ids"]) > 64
            or any(not isinstance(item, str) or not item for item in body["source_keep_ids"])):
        raise SurfaceValidationRefused("source keep membership is invalid")
    if not isinstance(body["target"], dict) or not body["target"]:
        raise SurfaceValidationRefused("target identity is invalid")
    for key in ("original_anchor", "candidate_anchor"):
        anchor = body[key]
        if (not isinstance(anchor, dict) or set(anchor) != {"path", "commit"}
                or not isinstance(anchor["path"], str) or not Path(anchor["path"]).is_absolute()
                or not isinstance(anchor["commit"], str) or len(anchor["commit"]) != 40):
            raise SurfaceValidationRefused(f"{key} is invalid")
    if type(body["intended_target"]) is not bool:
        raise SurfaceValidationRefused("intended-target marker is invalid")
    comparison = body["comparison"]
    belief = comparison.get("belief_capture") if isinstance(comparison, dict) else None
    inputs = belief.get("inputs") if isinstance(belief, dict) else None
    resolved_arms = inputs.get("resolved_arms") if isinstance(inputs, dict) else None
    candidate_resolved = resolved_arms.get("candidate") \
        if isinstance(resolved_arms, dict) else None
    if (body["candidate_anchor"]["commit"] != body["source_commit"]
            or comparison.get("request_digest") != body["request_digest"]
            or not isinstance(candidate_resolved, dict)
            or candidate_resolved.get("execution_digest") != body["recipe_execution_digest"]):
        raise SurfaceValidationRefused("validation source/request/recipe join differs")
    expected = classify(body["comparison"], intended_target=body["intended_target"])
    unsigned = {key: item for key, item in body.items() if key != "row_digest"}
    if body["disposition"] != expected or body["row_digest"] != _digest(unsigned):
        raise SurfaceValidationRefused("validation disposition or digest changed")
    if len(json.dumps(body, sort_keys=True).encode()) > MAX_BYTES:
        raise SurfaceValidationRefused("whole-source validation row exceeds byte budget")
    return body


def validate_debt(value: Any) -> dict[str, Any]:
    fields = {"schema", "source_commit", "source_tree", "source_keep_ids", "target",
              "original_anchor", "candidate_anchor", "request_digest",
              "recipe_execution_digest", "disposition", "reason", "failure", "row_digest"}
    if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != DEBT_SCHEMA:
        raise SurfaceValidationRefused("whole-source validation debt has an open schema")
    body = json.loads(json.dumps(value, allow_nan=False))
    # Reuse the row's identity checks without inventing a numerical comparison.
    for key in ("source_commit", "source_tree", "request_digest", "recipe_execution_digest"):
        expected_length = 40 if key in {"source_commit", "source_tree"} else 64
        raw = body[key]
        if key == "recipe_execution_digest" and raw is None:
            continue
        if (not isinstance(raw, str) or len(raw) != expected_length
                or any(char not in "0123456789abcdef" for char in raw)):
            raise SurfaceValidationRefused(f"invalid debt {key}")
    if (not isinstance(body["source_keep_ids"], list) or not body["source_keep_ids"]
            or len(body["source_keep_ids"]) > 64
            or any(not isinstance(item, str) or not item for item in body["source_keep_ids"])
            or not isinstance(body["target"], dict) or not body["target"]):
        raise SurfaceValidationRefused("validation debt membership/target is invalid")
    for key in ("original_anchor", "candidate_anchor"):
        anchor = body[key]
        if (not isinstance(anchor, dict) or set(anchor) != {"path", "commit"}
                or not isinstance(anchor["path"], str) or not Path(anchor["path"]).is_absolute()
                or (key == "candidate_anchor" and
                    (not isinstance(anchor["commit"], str) or len(anchor["commit"]) != 40))
                or (key == "original_anchor" and anchor["commit"] is not None and
                    (not isinstance(anchor["commit"], str) or len(anchor["commit"]) != 40))):
            raise SurfaceValidationRefused(f"debt {key} is invalid")
    if (body["candidate_anchor"]["commit"] != body["source_commit"]
            or body["disposition"] != "pending" or not isinstance(body["reason"], str)
            or not body["reason"] or len(body["reason"]) > 1024
            or not isinstance(body["failure"], dict)):
        raise SurfaceValidationRefused("validation debt disposition is malformed")
    if (body["recipe_execution_digest"] is None
            and body["failure"].get("type") != "target_recipe_gate_refused"):
        raise SurfaceValidationRefused(
            "missing candidate execution identity requires an original gate refusal")
    unsigned = {key: item for key, item in body.items() if key != "row_digest"}
    if body["row_digest"] != _digest(unsigned) or len(json.dumps(body).encode()) > MAX_BYTES:
        raise SurfaceValidationRefused("validation debt digest/size differs")
    return body


def retain(directory: Path, value: Any) -> dict[str, Any]:
    """Write one immutable batch-local row and return its exact routing reference."""
    from . import status, surface_fold
    body = validate_debt(value) if isinstance(value, Mapping) \
        and value.get("schema") == DEBT_SCHEMA else validate(value)
    directory = Path(directory)
    path = directory / "whole-source-validation.json"
    expected = json.dumps(body, indent=2, sort_keys=True).encode()
    if path.exists():
        actual = surface_fold.bounded_regular_bytes(path, MAX_BYTES)
        if actual != expected:
            raise SurfaceValidationRefused("validation output path already holds different bytes")
    else:
        status.write_json(directory, path.name, body, prefix=".whole-source-validation-")
        actual = surface_fold.bounded_regular_bytes(path, MAX_BYTES)
    if actual != expected or len(actual) > MAX_BYTES:
        raise SurfaceValidationRefused("validation output readback differs")
    return {"schema": REFERENCE_SCHEMA, "path": str(path.resolve()),
            "sha256": hashlib.sha256(actual).hexdigest()}


def reopen_reference(value: Any) -> dict[str, Any]:
    from . import surface_fold
    if (not isinstance(value, Mapping)
            or set(value) != {"schema", "path", "sha256"}
            or value.get("schema") != REFERENCE_SCHEMA
            or not isinstance(value.get("path"), str)
            or not Path(value["path"]).is_absolute()
            or not isinstance(value.get("sha256"), str)
            or len(value["sha256"]) != 64):
        raise SurfaceValidationRefused("validation reference is malformed")
    path = Path(value["path"])
    raw = surface_fold.bounded_regular_bytes(path, MAX_BYTES)
    if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != value["sha256"]:
        raise SurfaceValidationRefused("validation reference bytes changed")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SurfaceValidationRefused("validation reference is not JSON") from exc
    return validate_debt(body) if body.get("schema") == DEBT_SCHEMA else validate(body)


# ------------------------------------------------------------------ keep dimensions (G5)
#
# AK long-context audit 2026-10-04 §4.2 G5, operator ruling: AutoKernel works across
# EVERY tracked dimension. This EXTENDS the cross-workload gate above rather than adding
# a second one: each declared throughput dimension other than the primary is judged by
# the same interim non-inferiority rule (`classify(..., intended_target=False)`), and the
# primary keeps the existing serving verdict that already admitted the keep. Capacity
# is a fit test: the candidate's peak VRAM (GPU, own-PID KFD when sampled) or peak RSS
# (CPU), sampled by the residency sampler on the launch that carried the recipe's own
# context and slots over a MIXED request sequence (capacity_probe), must fit the declared
# ceiling and must not keep growing. A dimension the target does not
# declare is SKIPPED and recorded so; a declared dimension with no measurement is
# PENDING, which refuses the keep (fail closed, like a missing floor).

KEEP_DIMENSIONS = ("short_decode", "long_decode", "prefill_at_depth",
                   "concurrent_aggregate", "capacity")
THROUGHPUT_DIMENSIONS = KEEP_DIMENSIONS[:-1]
DIMENSIONS_SCHEMA = "epyc.autokernel.keep_dimensions.v1"
DIMENSIONS_DIR = "keep-dimensions"
#: Stack owner (workspace-ec, 2026-10-04, G1 constraint e): AutoKernel GPU runs stay at
#: or under 62 GiB peak on the 64 GiB MI210.
GPU_CAPACITY_CEILING_BYTES = 62 << 30


def parse_dimensions(text: str | None) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(item.strip() for item in (text or "").split(",") if item.strip()))
    unknown = [name for name in names if name not in KEEP_DIMENSIONS]
    if unknown:
        raise SurfaceValidationRefused(
            f"unknown keep dimension(s) {unknown}; known: {list(KEEP_DIMENSIONS)}")
    return names


def primary_dimension(np: int) -> str:
    """The dimension the existing serving gate already measures: aggregate tok/s over
    `np` concurrent frozen requests (np > 1) or single-request decode (np == 1)."""
    return "concurrent_aggregate" if int(np) > 1 else "short_decode"


def _capacity_row(capacity: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fit AND no growth, over the MIXED sequence (`capacity_probe`); else not passed.

    A steady single-shape footprint is not admissible: v10's mmvq_q8_1_graph_cache
    (GPU-POOL-1) passed every steady A/B while growing ~0.29 GiB per n_max:0
    alternation, so only a record taken over the mixed sequence can pass.
    """
    from . import capacity_probe
    cap = dict(capacity or {})
    if cap.get("error"):
        return {"disposition": "pending", **cap, "reason": f"capacity probe failed: {cap['error']}"[:512]}
    if cap.get("sequence") != capacity_probe.SEQUENCE:
        return {"disposition": "pending", **cap,
                "reason": "capacity needs the mixed n_max:0-alternation/varying-batch sequence"}
    peak, limit, growth = cap.get("peak_bytes"), cap.get("limit_bytes"), cap.get("growth_bytes")
    if not peak or not limit or growth is None:
        return {"disposition": "pending", **cap,
                "reason": "peak footprint, growth or ceiling unknown"}
    tolerance = int(cap.get("growth_tolerance_bytes") or capacity_probe.GROWTH_TOLERANCE_BYTES)
    fits, steady = int(peak) <= int(limit), int(growth) <= tolerance
    return {"disposition": "passed" if fits and steady else "failed", **cap,
            "reason": (f"peak {int(peak) / 2**30:.2f} GiB {'<=' if fits else '>'} ceiling "
                       f"{int(limit) / 2**30:.2f} GiB; growth after warm-up "
                       f"{int(growth) / 2**20:.0f} MiB {'<=' if steady else '>'} "
                       f"{tolerance / 2**20:.0f} MiB, at ctx={cap.get('ctx')} np={cap.get('np')}")}


def keep_dimensions(*, declared: Sequence[str], primary: str,
                    comparisons: Mapping[str, Any], capacity: Mapping[str, Any] | None
                    ) -> dict[str, Any]:
    """Per-dimension dispositions for one keep (pure). `passed` iff none failed/pending."""
    declared = tuple(declared)
    rows: dict[str, dict[str, Any]] = {}
    for name in KEEP_DIMENSIONS:
        if name not in declared:
            rows[name] = {"disposition": "skipped", "reason": "not declared by the target"}
        elif name == "capacity":
            rows[name] = _capacity_row(capacity)
        elif name == primary:
            rows[name] = {"disposition": "passed",
                          "reason": "primary serving gate already promoted this keep"}
        else:
            comparison = comparisons.get(name)
            if comparison is None:
                rows[name] = {"disposition": "pending",
                              "reason": "declared but no instrument measured it"}
                continue
            if isinstance(comparison, Mapping) and comparison.get("error"):
                rows[name] = {"disposition": "pending",
                              "reason": f"measurement failed: {comparison['error']}"[:512]}
                continue
            try:
                disposition = classify(comparison, intended_target=False)
            except SurfaceValidationRefused as exc:
                rows[name] = {"disposition": "pending", "reason": str(exc)[:512]}
                continue
            rows[name] = {"disposition": disposition,
                          "effect_pct": comparison.get("effect_pct"),
                          "noise_floor_pct": comparison.get("noise_floor_pct"),
                          "reason": "cross-workload interim non-inferiority rule"}
    blocking = {n: r for n, r in rows.items() if r["disposition"] in ("failed", "pending")}
    return {"schema": DIMENSIONS_SCHEMA, "declared": list(declared), "primary": primary,
            "dimensions": rows, "passed": not blocking,
            "reason": ("; ".join(f"{n}: {r['disposition']} ({r['reason']})"
                                 for n, r in blocking.items())
                       or "no declared dimension regressed")}


def retain_dimensions(store: Path, mechanism_id: str, record: Mapping[str, Any], *,
                      now: float | None = None) -> Path:
    import time as _time
    directory = Path(store) / DIMENSIONS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = int((_time.time() if now is None else now) * 1000)
    path = directory / f"{stamp}-{str(mechanism_id)[:80].replace('/', '_')}.json"
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = ["DEBT_SCHEMA", "DIMENSIONS_SCHEMA", "GPU_CAPACITY_CEILING_BYTES",
           "KEEP_DIMENSIONS", "MAX_BYTES", "REFERENCE_SCHEMA", "SCHEMA",
           "SurfaceValidationRefused", "classify", "debt",
           "keep_dimensions", "original_anchor_commit", "parse_dimensions",
           "primary_dimension", "reopen_reference", "retain", "retain_dimensions", "row",
           "validate", "validate_debt"]
