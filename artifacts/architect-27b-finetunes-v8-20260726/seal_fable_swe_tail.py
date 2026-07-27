#!/usr/bin/env python3
"""Seal the three complete Fable-v4 SWE captures without regenerating output.

This is deliberately not a generic converter.  It is the narrow closure path
for the 2026-07-26 Fable campaign, whose complete v4 captures have verified
row fingerprints but terminal model-contract failures that the generic CLI
must continue to reject fail-closed.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CAPTURE_ROOT = HERE / "live-20260726T1750Z" / "continuation-27b-v8"
INSTRUMENT = CAPTURE_ROOT / "instrument"
CANONICAL = REPO / "artifacts" / "architect-code-eval-20260724"
CONVERTER = CANONICAL / "convert_sr_to_patch.py"
DATASET = CANONICAL / "swebench_verified.json"
QUESTIONS = CANONICAL / "questions_swebench_oracle.json"
RUNNER = INSTRUMENT / "v7_quality_gate_runner.py"
HARNESS = REPO / ".venv-swebench/lib/python3.12/site-packages/swebench/harness/run_evaluation.py"
IDENTITY = INSTRUMENT / "identity.json"
SEALER_SOURCE = Path(__file__).resolve()

CURRENT_CONVERTER_SHA256 = "06a6530570af470cb76999ceb629fa5d280a26469ec75d7bb3e6a980f2c20b9f"
DATASET_SHA256 = "b087b5dad72b3e765a6cf93a9e7d516d8796698a0fd358abb73c6627df19f66e"
QUESTIONS_SHA256 = "f82a5191274048f2fdf432df7a0ebf4017ad982b954d6aa075326a1302df1c3c"
RUNNER_SHA256 = "79721927e95293d070aba294bf422a24b1182dde07310d461d9e3ddaf6c84b0e"
HARNESS_SHA256 = "6959f0b4e4eaf979771f529b88e3e9df1daa7fe86bc4291feec2e7d320bf7f2e"
IDENTITY_SHA256 = "5fd6f333ef82766efcea3bbfc6e90589511c1e4212b9e5546a2c148401bffd1b"
CAPTURE_SCHEMA = "v7_quality_gate_capture.v4"

ARMS = (
    {
        "name": "stock_non_mtp",
        "label": "stock_non_mtp",
        "capture_arm": "A3-ff-quality__stock_non_mtp",
        "raw": CAPTURE_ROOT / "A3-ff-quality__stock_non_mtp/swe_oracle.sealed.jsonl",
        "raw_sha256": "8872966aa162f39587735490927940bd571042ab6a5644c1af7bac2a41354c3a",
    },
    {
        "name": "fable_non_mtp",
        "label": "fable_non_mtp",
        "capture_arm": "A3-ff-quality__fable_non_mtp",
        "raw": CAPTURE_ROOT / "A3-ff-quality__fable_non_mtp/swe_oracle.sealed.jsonl",
        "raw_sha256": "a9f3bf2c65869ef0819416189b2478cfed157dd88bcb4cd62d68d913b034990a",
    },
    {
        "name": "fable_mtp",
        "label": "fable_mtp",
        "capture_arm": "A3-ff-embedded-mtp__fable_mtp",
        "raw": CAPTURE_ROOT / "A3-ff-embedded-mtp__fable_mtp/swe_oracle.sealed.jsonl",
        "raw_sha256": "fb2e265b0189e46c42040f7420ac97bdcf53cc467601fab77c955da1bb02aaa3",
    },
)

# These values make a changed converter or a changed raw capture explicit at
# preflight time; detailed block records remain derived in the sealed ledger.
KNOWN_CURRENT_CLASSIFICATION = {
    "stock_non_mtp": {
        "counts": {"applied": 40, "skipped": 3, "empty": 5},
        "skipped_instance_ids": {"django__django-11087", "django__django-11477", "sphinx-doc__sphinx-10435"},
        "length_empty_ids": {"django__django-11138", "django__django-11211", "matplotlib__matplotlib-20676"},
        "required_nonempty_ids": {"sympy__sympy-12419"},
    },
    "fable_non_mtp": {
        "counts": {"applied": 46, "skipped": 7, "empty": 5},
        "skipped_instance_ids": {"django__django-11292", "django__django-11477", "matplotlib__matplotlib-14623", "scikit-learn__scikit-learn-10297", "sphinx-doc__sphinx-10323"},
        "length_empty_ids": {"django__django-11138", "sympy__sympy-12419"},
        "required_nonempty_ids": set(),
    },
    "fable_mtp": {
        "counts": {"applied": 42, "skipped": 4, "empty": 4},
        "skipped_instance_ids": {"django__django-11292", "django__django-11477", "scikit-learn__scikit-learn-10297", "sympy__sympy-12419"},
        "length_empty_ids": {"django__django-11138"},
        "required_nonempty_ids": set(),
    },
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_fingerprint(value: str) -> dict[str, int | str]:
    encoded = value.encode("utf-8")
    return {"chars": len(value), "utf8_bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pinned_source_hashes() -> dict[Path, tuple[str, str]]:
    """Return and verify every immutable source copied into a sealed tree."""
    bindings = {
        CONVERTER: (CURRENT_CONVERTER_SHA256, "current converter"),
        DATASET: (DATASET_SHA256, "canonical dataset"),
        QUESTIONS: (QUESTIONS_SHA256, "canonical questions"),
        RUNNER: (RUNNER_SHA256, "capture runner"),
        HARNESS: (HARNESS_SHA256, "SWE harness"),
        IDENTITY: (IDENTITY_SHA256, "capture identity"),
    }
    bindings.update({Path(arm["raw"]): (arm["raw_sha256"], f"{arm['name']} frozen raw capture") for arm in ARMS})
    for path, (expected, label) in bindings.items():
        if not path.is_file() or sha256(path) != expected:
            fail(f"{label} SHA-256 drifted")
    return bindings


def source_snapshot() -> dict[Path, str]:
    """Bind static authorities plus this sealer's exact source at seal start."""
    snapshot = {path: expected for path, (expected, _label) in pinned_source_hashes().items()}
    snapshot[SEALER_SOURCE] = sha256(SEALER_SOURCE)
    return snapshot


def verify_source_snapshot(snapshot: dict[Path, str]) -> None:
    for path, expected in snapshot.items():
        if not path.is_file() or sha256(path) != expected:
            fail(f"source-to-sealed TOCTOU drifted: {path}")


def pinned_ids() -> list[str]:
    if sha256(QUESTIONS) != QUESTIONS_SHA256:
        fail("canonical question source SHA-256 drifted")
    rows = json.loads(QUESTIONS.read_text())
    ids = [row.get("id") for row in rows]
    if len(ids) != 40 or len(set(ids)) != 40 or not all(isinstance(item, str) for item in ids):
        fail("canonical question source is not the exact ordered 40-ID denominator")
    return ids


def validate_authorities() -> tuple[list[str], dict[str, dict[str, Any]], Any]:
    ids = pinned_ids()
    pinned_source_hashes()
    for path, expected, label in (
        (INSTRUMENT / "swebench_verified.json", DATASET_SHA256, "frozen dataset copy"),
        (INSTRUMENT / "questions_swe_oracle.json", QUESTIONS_SHA256, "frozen question copy"),
    ):
        if not path.is_file() or sha256(path) != expected:
            fail(f"{label} SHA-256 drifted")
    frozen_questions = json.loads((INSTRUMENT / "questions_swe_oracle.json").read_text())
    if frozen_questions != json.loads(QUESTIONS.read_text()):
        fail("frozen questions differ from canonical questions")
    dataset = json.loads(DATASET.read_text())
    if len(dataset) != 500 or len({row.get("instance_id") for row in dataset}) != 500:
        fail("canonical dataset does not have the pinned 500-instance shape")
    instances = {row["instance_id"]: row for row in dataset}
    if not set(ids) <= set(instances):
        fail("pinned denominator is not contained in canonical dataset")
    identity = json.loads(IDENTITY.read_text())
    if identity.get("capture_schema_version") != CAPTURE_SCHEMA or identity.get("runner_sha256") != RUNNER_SHA256:
        fail("capture identity does not bind the expected v4 runner")
    spec = importlib.util.spec_from_file_location("fable_tail_pinned_converter", CONVERTER)
    if spec is None or spec.loader is None:
        fail("cannot load current converter")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    return ids, instances, converter


def validate_raw_rows(arm: dict[str, Any], ids: list[str], instances: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    raw = Path(arm["raw"])
    if not raw.is_file() or sha256(raw) != arm["raw_sha256"]:
        fail(f"{arm['name']} frozen raw capture SHA-256 drifted")
    rows = load_jsonl(raw)
    if [row.get("id") for row in rows] != ids:
        fail(f"{arm['name']} raw rows do not preserve the ordered 40-ID denominator")
    question_prompts = {row["id"]: row["prompt"] for row in json.loads(QUESTIONS.read_text())}
    for row in rows:
        if row.get("capture_schema_version") != CAPTURE_SCHEMA:
            fail(f"{arm['name']} row {row.get('id')} has a non-v4 schema")
        if row.get("arm") != arm["capture_arm"] or row.get("suite") != "swebench_oracle":
            fail(f"{arm['name']} row {row.get('id')} arm/suite binding drifted")
        if row.get("seed") != 42 or row.get("rep") != 0 or row.get("runner_source_sha256") != RUNNER_SHA256:
            fail(f"{arm['name']} row {row.get('id')} seed/rep/runner binding drifted")
        if row.get("request_error") or row.get("finish_reason") == "request_error":
            fail(f"{arm['name']} row {row.get('id')} is not a completed model draw")
        for field in ("prompt", "response", "reasoning"):
            if not isinstance(row.get(field), str) or row.get(f"{field}_fingerprint") != text_fingerprint(row[field]):
                fail(f"{arm['name']} row {row.get('id')} {field} fingerprint drifted")
        if row["prompt"] != question_prompts.get(row["id"]):
            fail(f"{arm['name']} row {row.get('id')} prompt does not match canonical question")
        sr = row.get("swe_search_replace")
        if not isinstance(sr, dict) or not isinstance(sr.get("state"), str):
            fail(f"{arm['name']} row {row.get('id')} lacks v4 SEARCH/REPLACE capture details")
    return rows


def convert_rows(arm: dict[str, Any], rows: list[dict[str, Any]], converter: Any) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, int]]:
    predictions: list[dict[str, str]] = []
    diagnostics: list[dict[str, Any]] = []
    counts = {"applied": 0, "skipped": 0, "empty": 0, "length_forced_empty": 0}
    for row in rows:
        blocks: list[dict[str, Any]] = []
        if row.get("finish_reason") == "length":
            patch, applied, skipped = "", 0, 0
            counts["length_forced_empty"] += 1
        else:
            patch, applied, skipped = converter.apply_blocks(converter.rows[row["id"]], row["response"], blocks)
        diagnostic = converter.row_diagnostic(row, patch, blocks, RUNNER_SHA256)
        if row.get("finish_reason") == "length":
            diagnostic["empty_patch"] = True
            diagnostic["empty_patch_reason"] = "model_length_forced_empty"
            diagnostic["conversion_disposition"] = "model_length_contract_failure"
        if not diagnostic.get("scoring_eligible"):
            fail(f"{arm['name']} row {row['id']} lost v4 capture-integrity eligibility")
        predictions.append({"instance_id": row["id"], "model_name_or_path": arm["label"], "model_patch": patch})
        diagnostics.append(diagnostic)
        counts["applied"] += applied
        counts["skipped"] += skipped
        counts["empty"] += not bool(patch)
    expected = KNOWN_CURRENT_CLASSIFICATION[arm["name"]]
    if {key: counts[key] for key in expected["counts"]} != expected["counts"]:
        fail(f"{arm['name']} current converter classification drifted: {counts}")
    skipped_ids = {diagnostic["instance_id"] for diagnostic in diagnostics if diagnostic["skipped_block_count"]}
    length_ids = {diagnostic["instance_id"] for diagnostic in diagnostics if diagnostic["finish_reason"] == "length"}
    prediction_by_id = {prediction["instance_id"]: prediction["model_patch"] for prediction in predictions}
    if skipped_ids != expected["skipped_instance_ids"] or length_ids != expected["length_empty_ids"]:
        fail(f"{arm['name']} known current model-contract classification drifted")
    if any(not prediction_by_id[instance_id] for instance_id in expected["required_nonempty_ids"]):
        fail(f"{arm['name']} required safely applied patch disappeared")
    return predictions, diagnostics, counts


def ledger_for(arm: dict[str, Any], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    skipped_blocks = []
    empty_rows = []
    for diagnostic in diagnostics:
        for block in diagnostic.get("blocks", []):
            if str(block.get("outcome", "")).startswith("skipped_"):
                skipped_blocks.append({
                    "instance_id": diagnostic["instance_id"], "block_index": block["block_index"],
                    "outcome": block["outcome"], "search_sha256": block["search_sha256"],
                    "replace_sha256": block["replace_sha256"], "classification": "model_contract_failure",
                    "additional_recovery_attempted": False,
                    "disposition": "preserve_pinned_current_converter_outcome_no_match_broadening",
                })
        if diagnostic["empty_patch"]:
            length = diagnostic.get("finish_reason") == "length"
            if not length and diagnostic.get("parseable_block_count", 0) == 0:
                reason = "stop_no_parseable_search_replace_block"
            elif length:
                reason = "length_forced_empty"
            else:
                reason = str(diagnostic.get("empty_patch_reason"))
            empty_rows.append({
                "instance_id": diagnostic["instance_id"], "finish_reason": diagnostic.get("finish_reason"),
                "empty_patch_reason": reason,
                "classification": "model_length_contract_failure" if length else "model_contract_failure",
                "additional_recovery_attempted": False,
                "disposition": "preserve_pinned_current_converter_outcome_no_match_broadening",
            })
    return {
        "schema_version": "epyc.fable-v4-swe-tail-nonrecovery-ledger.v1",
        "status": "EXHAUSTIVE_MODEL_CONTRACT_FAILURES",
        "arm": arm["name"],
        "scope": "2026-07-26 Fable campaign only; no generic converter exception",
        "skipped_blocks": skipped_blocks,
        "empty_patch_rows": empty_rows,
        "aggregate": {"skipped_block_count": len(skipped_blocks), "empty_patch_row_count": len(empty_rows)},
    }


def validate_conversion(arm: dict[str, Any], ids: list[str], predictions: list[dict[str, str]], diagnostics: list[dict[str, Any]], ledger: dict[str, Any]) -> None:
    if len(predictions) != 40 or [row["instance_id"] for row in predictions] != ids:
        fail(f"{arm['name']} prediction denominator/order drifted")
    if any(set(row) != {"instance_id", "model_name_or_path", "model_patch"} for row in predictions):
        fail(f"{arm['name']} prediction schema drifted")
    if len(diagnostics) != 40 or [row["instance_id"] for row in diagnostics] != ids:
        fail(f"{arm['name']} diagnostic denominator/order drifted")
    actual_skips = {(d["instance_id"], b["block_index"], b["outcome"], b["search_sha256"], b["replace_sha256"])
                    for d in diagnostics for b in d.get("blocks", []) if str(b.get("outcome", "")).startswith("skipped_")}
    ledger_skips = {(row["instance_id"], row["block_index"], row["outcome"], row["search_sha256"], row["replace_sha256"])
                    for row in ledger["skipped_blocks"]}
    if actual_skips != ledger_skips:
        fail(f"{arm['name']} non-recovery ledger is not exhaustive for skipped blocks")
    actual_empty = {row["instance_id"] for row in diagnostics if row["empty_patch"]}
    ledger_empty = {row["instance_id"] for row in ledger["empty_patch_rows"]}
    if actual_empty != ledger_empty:
        fail(f"{arm['name']} non-recovery ledger is not exhaustive for empty patches")
    records = ledger["skipped_blocks"] + ledger["empty_patch_rows"]
    if any(row.get("additional_recovery_attempted") is not False or not str(row.get("classification", "")).startswith("model_") for row in records):
        fail(f"{arm['name']} ledger contains an unsupported recovery or classification")


def copy_and_verify(source: Path, destination: Path, expected_sha256: str) -> str:
    shutil.copyfile(source, destination)
    copied_sha256 = sha256(destination)
    if copied_sha256 != expected_sha256:
        fail(f"copied artifact does not match verified source: {source}")
    return copied_sha256


def copy_authorities(root: Path, snapshot: dict[Path, str]) -> dict[str, dict[str, str]]:
    authority = root / "authority"
    authority.mkdir(parents=True)
    bindings = {}
    for name, source in (
        ("converter.py", CONVERTER), ("swebench_verified.json", DATASET),
        ("questions_swebench_oracle.json", QUESTIONS), ("capture_runner.py", RUNNER),
        ("swebench_harness.py", HARNESS), ("capture_identity.json", IDENTITY),
        ("seal_fable_swe_tail.py", SEALER_SOURCE),
    ):
        destination = authority / name
        bindings[name] = {"source": str(source), "sha256": copy_and_verify(source, destination, snapshot[source])}
    return bindings


def prepare_arms(ids: list[str], instances: dict[str, dict[str, Any]], converter: Any) -> dict[str, dict[str, Any]]:
    prepared = {}
    for arm in ARMS:
        rows = validate_raw_rows(arm, ids, instances)
        predictions, diagnostics, counts = convert_rows(arm, rows, converter)
        ledger = ledger_for(arm, diagnostics)
        validate_conversion(arm, ids, predictions, diagnostics, ledger)
        prepared[arm["name"]] = {
            "rows": rows, "predictions": predictions, "diagnostics": diagnostics,
            "counts": counts, "ledger": ledger,
        }
    return prepared


def observational_provenance() -> dict[str, str]:
    result = {"sealed_at_utc": datetime.now(UTC).isoformat()}
    try:
        result["research_head"] = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            check=True, text=True, capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        result["research_head"] = "unavailable"
    return result


def write_seal_digest(root: Path) -> str:
    """Write deterministic hashes for every sealed file except this digest itself."""
    digest_path = root / "seal.sha256"
    files = [path for path in sorted(root.rglob("*")) if path.is_file() and path != digest_path]
    digest_path.write_text("".join(f"{sha256(path)}  {path.relative_to(root)}\n" for path in files))
    return sha256(digest_path)


def publish_no_replace(stage: Path, output_dir: Path) -> None:
    """Atomically publish a staged directory without clobbering an existing tree."""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        fail("Linux renameat2(RENAME_NOREPLACE) is unavailable; refusing unsafe publish")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(stage), -100, os.fsencode(output_dir), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        fail(f"refusing to clobber output tree created during sealing: {output_dir}")
    fail(f"renameat2(RENAME_NOREPLACE) failed: {os.strerror(error)}")


def seal(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        fail(f"refusing to modify existing output tree: {output_dir}")
    ids, instances, converter = validate_authorities()
    prepared = prepare_arms(ids, instances, converter)
    snapshot = source_snapshot()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        authority = copy_authorities(stage, snapshot)
        manifests = {}
        for arm in ARMS:
            sealed = prepared[arm["name"]]
            arm_dir = stage / arm["name"]
            arm_dir.mkdir()
            copy_and_verify(arm["raw"], arm_dir / "raw_capture.sealed.jsonl", snapshot[Path(arm["raw"])])
            write_json(arm_dir / "predictions.sealed.json", sealed["predictions"])
            (arm_dir / "conversion_diagnostics.sealed.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sealed["diagnostics"]))
            write_json(arm_dir / "nonrecovery_ledger.sealed.json", sealed["ledger"])
            manifests[arm["name"]] = {
                "raw_capture_sha256": sha256(arm_dir / "raw_capture.sealed.jsonl"),
                "predictions_sha256": sha256(arm_dir / "predictions.sealed.json"),
                "diagnostics_sha256": sha256(arm_dir / "conversion_diagnostics.sealed.jsonl"),
                "nonrecovery_ledger_sha256": sha256(arm_dir / "nonrecovery_ledger.sealed.json"),
                "counts": sealed["counts"],
            }
        hashes = {str(path.relative_to(stage)): sha256(path) for path in sorted(stage.rglob("*")) if path.is_file()}
        write_json(stage / "hashes.json", {"schema_version": "epyc.fable-v4-swe-tail-hashes.v1", "files": hashes})
        manifest = {
            "schema_version": "epyc.fable-v4-swe-tail-sealer.v1", "status": "SEALED_FOR_OFFICIAL_SCORING",
            "scope": "stock_non_mtp, fable_non_mtp, fable_mtp only", "no_inference_or_docker": True,
            "requested_ids": ids, "current_converter_sha256": CURRENT_CONVERTER_SHA256,
            "capture_schema_version": CAPTURE_SCHEMA, "runner_sha256": RUNNER_SHA256,
            "authority": authority, "arms": manifests,
            "hashes_json_sha256": sha256(stage / "hashes.json"),
            "conversion_policy": "force finish_reason=length empty; retain safely applied patches; classify skips and stop/no-parseable rows as model contract failures",
            "observational_provenance": observational_provenance(),
            "publish_policy": "output directory is rechecked immediately before Linux renameat2(RENAME_NOREPLACE), which atomically refuses an existing destination",
        }
        write_json(stage / "manifest.json", manifest)
        write_seal_digest(stage)
        verify_source_snapshot(snapshot)
        if output_dir.exists():
            fail(f"refusing to clobber output tree created during sealing: {output_dir}")
        publish_no_replace(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def preflight() -> dict[str, Any]:
    ids, instances, converter = validate_authorities()
    prepared = prepare_arms(ids, instances, converter)
    return {
        "status": "PREFLIGHT_OK", "denominator": ids,
        "arms": {name: {"rows": len(value["rows"]), "counts": value["counts"], "ledger": value["ledger"]["aggregate"]}
                 for name, value in prepared.items()},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.execute and args.output_dir is None:
        parser.error("--execute requires --output-dir")
    if args.preflight and args.output_dir is not None:
        parser.error("--output-dir is only valid with --execute")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = preflight() if args.preflight else seal(args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except RuntimeError as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
