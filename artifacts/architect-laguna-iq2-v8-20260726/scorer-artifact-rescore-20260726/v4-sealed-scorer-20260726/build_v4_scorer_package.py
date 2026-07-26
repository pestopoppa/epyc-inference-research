#!/usr/bin/env python3
"""Build the sealed, v4-only Laguna SWE scorer inputs without scoring them.

The sole admissible capture is the terminal prompt-contract run.  This builder
re-runs the shared converter after pinning both its content and every capture
input, then writes an exact official-SWE invocation for later human-approved
execution.  It never invokes Docker, SWE-bench, or a model endpoint.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/mnt/raid0/llm/epyc-inference-research")
RUN = REPO / (
    "artifacts/architect-laguna-iq2-v8-20260726/scorer-artifact-rescore-20260726/"
    "clean-full40-promptfix-20260726/run-20260726T220759Z"
)
CONVERTER = REPO / "artifacts/architect-code-eval-20260724/convert_sr_to_patch.py"
SWE_PYTHON = REPO / ".venv-swebench/bin/python"
ARM = "Laguna_S_2_1_UD_IQ2_M_v8_clean_full40_promptfix_3072"
RUNNER_SHA256 = "79721927e95293d070aba294bf422a24b1182dde07310d461d9e3ddaf6c84b0e"
CONVERTER_SHA256 = "6bd2302dda3e5139cc6faabcc5639bdcf85b27895f93a9181cbb53dd65749507"
QUESTION_SHA256 = "4b03ad7703bbf2dbaa1eb91b3313cc3cab2892672db87f6242ffd1d489e76375"
PQ_SHA256 = "a2ce92399d87c8f5f15b285ad27eca3cf0328f1b80eec1fec81933ed782cd81a"
CAPTURE_VALIDATION_SHA256 = "7c09a98a487033d1a6b07c477d31b95e1e17a78995fdd74f11d583cf4636c586"
BASE_DIAGNOSTIC_ABORT_RECEIPT_SHA256 = "471f71b5651169ee06a2fb5c7a18bf0a6a7ecd2a626d95aeaef61a79554a282d"
OUT = HERE / "predictions_v4.json"
DIAGNOSTICS = HERE / "predictions_v4.json.diagnostics.jsonl"
SUMMARY = HERE / "predictions_v4.json.diagnostics.summary.json"
POSTSCORE_VALIDATOR = HERE / "validate_official_swebench_report.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require(path: Path, expected_sha: str) -> None:
    actual = sha256(path)
    if actual != expected_sha:
        raise RuntimeError(f"source drift: {path} sha256 {actual}, expected {expected_sha}")


def verify_terminal_capture() -> list[str]:
    require(RUN / "pq.jsonl", PQ_SHA256)
    require(RUN / "questions_pinned_40.json", QUESTION_SHA256)
    require(RUN / "capture.validation.json", CAPTURE_VALIDATION_SHA256)
    require(
        RUN.parent / "BASE_DIAGNOSTIC_SUPERSESSION_ABORT_RECEIPT.json",
        BASE_DIAGNOSTIC_ABORT_RECEIPT_SHA256,
    )
    if sha256(CONVERTER) != CONVERTER_SHA256:
        raise RuntimeError("shared converter drifted; refuse a non-reproducible rescore")

    validation = load_json(RUN / "capture.validation.json")
    if validation != {
        "capture_schema_version": "v7_quality_gate_capture.v4",
        "rows": 40,
        "runner_source_sha256": RUNNER_SHA256,
        "status": "VALID",
    }:
        raise RuntimeError("terminal v4 validation receipt is not the sealed 40-row receipt")
    provenance = load_json(RUN / "provenance.json")
    if provenance.get("arm") != ARM or provenance.get("runner_source_sha256") != RUNNER_SHA256:
        raise RuntimeError("v4 provenance arm or runner source mismatch")

    expected_ids = load_json(RUN / "expected_question_ids.json")
    questions = load_json(RUN / "questions_pinned_40.json")
    rows = [json.loads(line) for line in (RUN / "pq.jsonl").read_text().splitlines() if line]
    ids = [row.get("id") for row in rows]
    if len(expected_ids) != 40 or len(set(expected_ids)) != 40:
        raise RuntimeError("expected IDs must be exactly 40 unique entries")
    if ids != expected_ids or [row.get("id") for row in questions] != expected_ids:
        raise RuntimeError("capture, question source, and expected IDs do not have identical order")
    for row in rows:
        if row.get("arm") != ARM or row.get("runner_source_sha256") != RUNNER_SHA256:
            raise RuntimeError(f"row provenance mismatch: {row.get('id')}")
        if row.get("request_error"):
            raise RuntimeError(f"request error in terminal capture: {row.get('id')}")
    status = load_json(RUN / "pq.live-status.json")
    if status.get("complete") is not True or status.get("artifact_integrity_fail_closed") is not False:
        raise RuntimeError("capture is incomplete or integrity-failed")
    if status.get("swebench_search_replace", {}).get("state_counts", {}).get("prompt_contract_candidate") != 0:
        raise RuntimeError("terminal capture still has prompt-contract candidates")
    return expected_ids


REVIEWED_SKIPS = {
    ("matplotlib__matplotlib-20488", 1): {
        "outcome": "skipped_search_not_found",
        "search_sha256": "aefc8765c94478b50b225536baf52629e03a3f684b4e8f6687e348a1b7adce62",
        "replace_sha256": "e97410b868545a8bd23d81942f59667b5c380d8129df1dd82cd5b7f28b9b3c95",
        "classification": "redundant_duplicate_after_exact_application",
        "reason": "Block 0 applied exactly; block 1 is byte-identical and therefore cannot add a second patch.",
    },
    ("scikit-learn__scikit-learn-11310", 0): {
        "outcome": "skipped_path_normalization_rejected_generic_placeholder",
        "search_sha256": "a6b8c5f645a0a0568a3a9b6e6f49b0bf46402dff9e26b4b3cdb6ebecd516dc52",
        "replace_sha256": "6374568460ed327132b14d9bc0bb1a51cc3c690edfe5a1af0412233210c01cad",
        "classification": "malformed_generic_path_model_output",
        "reason": "The literal generic path/to/file.py has no safe target inference; it remains an empty prediction.",
    },
    ("sphinx-doc__sphinx-10435", 1): {
        "outcome": "skipped_search_not_found",
        "search_sha256": "38fa736f2e812819edf3d6f0bc2bf2914630ee9278d3366b2e614f01793ca1ee",
        "replace_sha256": "1fb63fd9cab6c566f5f650ecd943ebe22f1dc2bc5fdf2a607b76f65e568e406a",
        "classification": "redundant_duplicate_after_exact_application",
        "reason": "Block 0 applied exactly; block 1 is byte-identical and therefore cannot add a second patch.",
    },
}


def reviewed_skip_dispositions() -> list[dict]:
    diagnostics = [json.loads(line) for line in DIAGNOSTICS.read_text().splitlines() if line]
    skipped: list[dict] = []
    for diagnostic in diagnostics:
        for block in diagnostic.get("blocks", []):
            if str(block.get("outcome", "")).startswith("skipped_"):
                key = (diagnostic["instance_id"], block["block_index"])
                review = REVIEWED_SKIPS.get(key)
                if review is None:
                    raise RuntimeError(f"unreviewed v4 converter skip: {key}")
                for field in ("outcome", "search_sha256", "replace_sha256"):
                    if block.get(field) != review[field]:
                        raise RuntimeError(f"skip disposition source drift for {key}: {field}")
                skipped.append({
                    "instance_id": diagnostic["instance_id"],
                    "block_index": block["block_index"],
                    "outcome": block["outcome"],
                    "path": block.get("path"),
                    "search_sha256": block.get("search_sha256"),
                    "replace_sha256": block.get("replace_sha256"),
                    **review,
                    "disposition": "non-recovery: preserve the shared converter's existing patch or empty patch",
                })
    if {(item["instance_id"], item["block_index"]) for item in skipped} != set(REVIEWED_SKIPS):
        raise RuntimeError("reviewed v4 skip set does not exactly match converter diagnostics")
    return skipped


def build_ledger(ids: list[str], skipped: list[dict]) -> None:
    summary = load_json(SUMMARY)
    ledger = {
        "schema": "epyc.laguna-swe-v4-skip-disposition-and-supersession.v1",
        "status": "TERMINAL_V4_REVIEWED_NONRECOVERY_DISPOSITION",
        "authoritative_capture": {
            "path": str(RUN), "pq_sha256": PQ_SHA256,
            "capture_validation_sha256": CAPTURE_VALIDATION_SHA256,
            "runner_source_sha256": RUNNER_SHA256, "question_sha256": QUESTION_SHA256,
            "requested_ids": ids,
        },
        "converter": {"path": str(CONVERTER), "sha256": CONVERTER_SHA256},
        "supersession": {
            "supersedes": [
                "attempt-02-port18089/swe_oracle (legacy tail-capped diagnostic capture)",
                "clean-full40-20260726/run-20260726T215138Z (pre-promptfix v4 candidate)",
                "live-20260726T193048Z/fullcapture5 lineage (non-authoritative; never inherited)",
            ],
            "base_diagnostic_abort_receipt_sha256": BASE_DIAGNOSTIC_ABORT_RECEIPT_SHA256,
            "rule": "Only the named terminal promptfix v4 capture is scorer input; this package contains no fullcapture5 predictions or claims.",
        },
        "converter_result": {
            "prediction_count": summary["prediction_count"],
            "empty_patches": summary["empty_patches"],
            "blocks_applied": summary["blocks_applied"],
            "blocks_skipped": summary["blocks_skipped"],
            "scoring_eligible": summary["scoring_eligible"],
        },
        "skipped_blocks": skipped,
        "review_rule": "This exact three-entry ledger is exhaustive. Any added, removed, or hash-drifted skip fails package construction; no matcher is broadened and no model text is rewritten.",
    }
    atomic_json(HERE / "v4_skip_disposition_and_supersession.json", ledger)


def build_official_argv(ids: list[str]) -> None:
    report_dir = HERE / "official_report"
    argv = [
        str(SWE_PYTHON), "-m", "swebench.harness.run_evaluation",
        "--dataset_name", "princeton-nlp/SWE-bench_Verified",
        "--predictions_path", str(OUT),
        "--instance_ids", *ids,
        "--max_workers", "8", "--cache_level", "env",
        "--run_id", "laguna-iq2-v8-promptfix-v4-20260726",
        "--report_dir", str(report_dir),
    ]
    atomic_json(HERE / "official_swebench_argv.json", {
        "schema": "epyc.official-swebench-argv.v1", "argv": argv,
        "requested_ids": ids, "predictions_sha256": sha256(OUT),
        "report_path": str(HERE / "Laguna_S_2_1_UD_IQ2_M_v8_clean_full40_promptfix_3072.laguna-swe-v4-20260726.json"),
    })


def materialize_reviewed_predictions(ids: list[str]) -> None:
    """Reproduce converter patch semantics after verifying the exhaustive ledger.

    The shared converter deliberately refuses to publish a scorer artifact while
    any skip exists.  This package does not change that policy.  It materializes
    the identical patches only after the ledger proves every v4 skip is either a
    duplicate after successful application or a generic-path malformed output.
    """
    spec = importlib.util.spec_from_file_location("sealed_shared_converter", CONVERTER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sealed shared converter")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    source_rows = [json.loads(line) for line in (RUN / "pq.jsonl").read_text().splitlines() if line]
    predictions = []
    stats = {"empty": 0, "applied": 0, "skipped": 0}
    for row in source_rows:
        block_diagnostics = []
        if row.get("finish_reason") == "length":
            patch, applied, skipped = "", 0, 0
        else:
            patch, applied, skipped = converter.apply_blocks(
                converter.rows[row["id"]], row.get("response", ""), block_diagnostics)
        stats["empty"] += not bool(patch)
        stats["applied"] += applied
        stats["skipped"] += skipped
        predictions.append({"instance_id": row["id"], "model_name_or_path": ARM, "model_patch": patch})
    summary = load_json(SUMMARY)
    if stats != {"empty": summary["empty_patches"], "applied": summary["blocks_applied"], "skipped": summary["blocks_skipped"]}:
        raise RuntimeError(f"materialized predictions diverge from shared converter diagnostics: {stats}")
    if [row["instance_id"] for row in predictions] != ids:
        raise RuntimeError("materialized prediction order drift")
    atomic_json(OUT, predictions)
    atomic_json(HERE / "reviewed_conversion_receipt.json", {
        "schema": "epyc.laguna-swe-v4-reviewed-conversion.v1",
        "shared_converter_summary_sha256": sha256(SUMMARY),
        "prediction_sha256": sha256(OUT), "stats": stats,
        "rule": "Non-recovery dispositions only; shared converter matching is unchanged.",
    })


def main() -> int:
    ids = verify_terminal_capture()
    command = [
        sys.executable, str(CONVERTER), str(RUN / "pq.jsonl"), ARM, str(OUT),
        "--diagnostics-jsonl", str(DIAGNOSTICS), "--diagnostics-summary", str(SUMMARY),
        "--runner-source", str(RUN / "runner_source.py"),
    ]
    result = subprocess.run(command)
    if result.returncode != 1:
        raise RuntimeError("expected shared converter to fail closed pending skip disposition")
    summary = load_json(SUMMARY)
    if summary.get("prediction_count") != 40 or summary.get("artifact_integrity_status") != "verified":
        raise RuntimeError("shared v4 converter did not produce complete verified diagnostics")
    skipped = reviewed_skip_dispositions()
    build_ledger(ids, skipped)
    materialize_reviewed_predictions(ids)
    build_official_argv(ids)
    atomic_json(HERE / "sealed_package_manifest.json", {
        "schema": "epyc.laguna-swe-v4-sealed-package.v1", "status": "READY_FOR_OFFICIAL_SCORING",
        "capture_run": str(RUN), "capture_pq_sha256": PQ_SHA256,
        "converter_sha256": CONVERTER_SHA256, "predictions_sha256": sha256(OUT),
        "diagnostics_summary_sha256": sha256(SUMMARY), "requested_ids": ids,
        "official_argv_sha256": sha256(HERE / "official_swebench_argv.json"),
        "postscore_validator_sha256": sha256(POSTSCORE_VALIDATOR),
    })
    print("v4 sealed scorer package: READY_FOR_OFFICIAL_SCORING (conversion only; no SWE execution)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
