"""Write a prospective, source-bound run record for the Unlimited-OCR arm.

This module records only metadata and digests from the just-completed producer run. It does not
grade a claim, alter predictions, or retrofit earlier run directories.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any

SCHEMA = "epyc.odl_bench.unlimited_ocr_run/v1"
PROTOCOL = "odl-bench/unlimited-ocr-model-gated/v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_file(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"run output cannot be opened as a regular file: {path}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"run output is not a regular file: {path}")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read()
    finally:
        os.close(fd)
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def _tree_files(directory: Path) -> list[dict[str, Any]]:
    directory_info = directory.lstat()
    if not stat.S_ISDIR(directory_info.st_mode):
        raise ValueError(f"run output tree root is not a regular directory: {directory}")
    result = []
    for path in sorted(directory.rglob("*")):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"symlink in run output tree: {path}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if stat.S_ISREG(info.st_mode):
            result.append(_sha_file(path))
        else:
            raise ValueError(f"non-regular entry in run output tree: {path}")
    return result


def capture_source_identity(repository: str | Path) -> dict[str, Any]:
    repository = Path(repository).resolve()
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"],
                                     text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain",
                                      "--untracked-files=all"], text=True)
    return {"repository": "epyc-inference-research", "commit": commit,
            "working_tree_clean": not bool(status)}


def write_unlimited_ocr_record(*, run_dir: str | Path, response_dir: str | Path,
                               prediction_dir: str | Path, gt_json: str | Path,
                               row_set: dict[str, Any], config: Any,
                               source_before: dict[str, Any],
                               prompt_source_sha256: str | None = None) -> Path:
    run = Path(run_dir).resolve(strict=True)
    response = Path(response_dir).resolve(strict=True)
    predictions = Path(prediction_dir).resolve(strict=True)
    input_path = response / "producer_input_manifest.json"
    input_record = json.loads(input_path.read_bytes())
    if input_record.get("schema") != "epyc.odl_bench.model_gated_inputs/v1":
        raise ValueError("model producer input manifest has an unknown schema")
    manifest = next((item for item in row_set.get("run_manifests", [])
                     if item.get("engine") == "unlimited_ocr"), None)
    if manifest is None:
        raise ValueError("Unlimited-OCR run manifest is missing from row set")
    speed_rows = [row for row in row_set.get("metric_rows", [])
                  if row.get("engine") == "unlimited_ocr"
                  and row.get("metric_family") == "speed"
                  and row.get("metric_name") == "latency_ms_median"]
    if len(speed_rows) != 1:
        raise ValueError("expected exactly one native Unlimited-OCR median latency row")
    speed = speed_rows[0]
    value = speed.get("value")
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value)):
        raise ValueError("median latency must be a finite number or unknown")
    attempts = [item for item in manifest.get("artifacts", [])
                if isinstance(item.get("latency_ms"), (int, float))
                and not isinstance(item.get("latency_ms"), bool)
                and item["latency_ms"] > 0]
    source_root = Path(__file__).resolve().parents[3]
    source_after = capture_source_identity(source_root)
    source = {"repository": "epyc-inference-research",
              "commit": source_before.get("commit"),
              "working_tree_clean": source_before.get("working_tree_clean"),
              "checked_after_commit": source_after["commit"],
              "checked_after_clean": source_after["working_tree_clean"]}
    metrics = [{key: row.get(key) for key in
                ("metric_family", "metric_name", "value", "n", "detail")}
               for row in row_set.get("metric_rows", [])]
    payload = {
        "schema": SCHEMA,
        "record_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol_id": PROTOCOL,
        "category": "CANDIDATE",
        "source": source,
        "producer": {"engine": "unlimited_ocr",
                     "entrypoint": "scripts/benchmark/odl_bench/adapter.py run-model",
                     "prompt_profile": config.prompt_profile,
                     "prompt_sha256": hashlib.sha256(config.prompt.encode("utf-8")).hexdigest(),
                     "prompt_source_sha256": prompt_source_sha256,
                     "binary_path": str(config.binary), "binary_sha256": None,
                     "model_path": str(config.model), "model_sha256": None,
                     "mmproj_path": str(config.mmproj), "mmproj_sha256": None,
                     "context": config.context, "threads": config.threads,
                     "parallel": config.parallel, "device": config.device,
                     "gpu_layers": config.gpu_layers, "max_tokens": config.max_tokens},
        "inputs": input_record,
        "measurement": {"metric": "latency_ms_median",
                        "value": value, "unit": "ms/page", "direction": "lower_better",
                        "reps": len(attempts),
                        "reps_basis": "positive per-page latency attempts included in the existing median; errors remain attempts",
                        "claim": "Unlimited-OCR observed median per-page extraction latency for this recorded run"},
        "run_metrics": metrics,
        "locator": {"run_dir": str(run), "row_set": str(run / "model_gated_row_set.json"),
                    "input_manifest": str(input_path),
                    "inference_window": str(response / "inference_window.json")},
        "outputs": {"predictions": _tree_files(predictions),
                    "responses": _tree_files(response),
                    "row_set": _sha_file(run / "model_gated_row_set.json")},
    }
    raw_payload = _canonical(payload)
    envelope = {"payload_sha256": hashlib.sha256(raw_payload).hexdigest(), "payload": payload}
    destination = run / "vidya_measurement_record.json"
    encoded = _canonical(envelope) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    return destination
