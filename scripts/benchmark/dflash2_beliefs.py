#!/usr/bin/env python3
"""Prospective DFlash2 experimental-runtime measurement carrier.

This module never reconstructs claims from the completed DF2-4 panel.  It only
finalizes campaigns produced after this hook exists, and it projects only the
producer-authored ``belief_measurements`` embedded in that sealed final receipt.
Grading remains the belief kernel's responsibility.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


SCHEMA = "epyc.df2.experimental_runtime_campaign.v1"
MANIFEST_SCHEMA = "epyc.df2.experimental_runtime_manifest.v1"
PRODUCER_ID = "scripts.benchmark.dflash2_beliefs/v1"
PRODUCER_PATH = "scripts/benchmark/dflash2_beliefs.py"
PROTOCOL_ID = "DF2-5-QWEN38-NP-GRID-v1"
AUTHORITY = "experimental_runtime_no_kernel_champion_no_promotion"
CAMPAIGN_KIND = "experimental_runtime"
EXPECTED_PROTOCOL = {"np": [2, 4, 8], "context": 32768, "threads": 8, "batch": 2048,
                     "ubatch": 2048, "kv": "f16/f16", "questions": 12,
                     "max_tokens": 2048, "seed": 42, "temperature": 0.6,
                     "top_p": 0.95, "top_k": 20, "enable_thinking": False,
                     "endpoint": "chat", "repeats": 1, "mtp_n_max": 8,
                     "dflash_requested_n_max": 8}
ARMS = tuple(f"{arm}_np{np}" for np in (2, 4, 8) for arm in ("mtp", "dflash2"))
REQUIRED_ARM_FILES = frozenset({
    "acceptance.txt", "claim-open.json", "claim-released.json", "commands.json",
    "pq.jsonl", "resource-samples.json", "result.json", "summary.json", "transport.json",
})
_SHA = re.compile(r"[0-9a-f]{64}")
_ACCEPT = re.compile(
    r"draft acceptance\s*=\s*([0-9.]+)\s*\(\s*(\d+) accepted /\s*(\d+) generated\),\s*"
    r"mean len\s*=\s*([0-9.]+)")


class DFlash2BeliefRefusal(ValueError):
    """The campaign cannot safely become a prospective ClaimTuple carrier."""


def _refuse(message: str) -> None:
    raise DFlash2BeliefRefusal(message)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt_sha(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    return canonical_sha256(payload)


def _measurement_sha(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("measurement_sha256", None)
    return canonical_sha256(payload)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _refuse(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        _refuse(f"{label} keys differ: expected {sorted(keys)}, got {sorted(value)}")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        _refuse(f"{label} must be a lowercase SHA-256")
    return value


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _refuse(f"{label} must be numeric")
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        _refuse(f"{label} must be positive and finite")
    return out


def _int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _refuse(f"{label} must be an integer >= {minimum}")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        _refuse(f"{label} must be a timezone-aware ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DFlash2BeliefRefusal(
            f"{label} must be a timezone-aware ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _refuse(f"{label} must carry a timezone offset")
    return parsed


def _json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 1:
            _refuse(f"{label} is empty")
        return _mapping(json.loads("\n".join(lines)), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DFlash2BeliefRefusal(f"{label} is not UTF-8 JSON") from exc


def _jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    try:
        raw = path.read_bytes()
        if not raw.endswith(b"\n") or b"\n\n" in raw:
            _refuse(f"{label} must be newline-terminated without blank rows")
        return [_mapping(json.loads(line), f"{label}[{i}]")
                for i, line in enumerate(raw.decode("utf-8").splitlines())]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DFlash2BeliefRefusal(f"{label} is not strict JSONL") from exc


def _safe_regular(path: Path, root: Path, label: str) -> None:
    try:
        rel = path.relative_to(root)
    except ValueError:
        _refuse(f"{label} escapes campaign root")
    cursor = root
    for part in rel.parts[:-1]:
        cursor /= part
        st = cursor.lstat()
        if not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode):
            _refuse(f"{label} has a non-directory/symlink parent")
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode) or st.st_nlink != 1:
        _refuse(f"{label} must be a regular single-link file")


def _external_regular(path: Path, label: str) -> None:
    try:
        st = path.lstat()
    except FileNotFoundError:
        _refuse(f"{label} does not exist")
    if not stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode) or st.st_nlink != 1:
        _refuse(f"{label} must be a regular single-link file")


def _producer() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {"producer_id": PRODUCER_ID, "path": PRODUCER_PATH, "sha256": sha256_file(path)}


def _manifest_entries(root: Path) -> list[dict[str, Any]]:
    entries = []
    excluded = {"campaign-manifest.json", "campaign-summary.json"}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if rel in excluded:
            continue
        st = path.lstat()
        if stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode):
            continue
        _safe_regular(path, root, rel)
        entries.append({"path": rel, "size": st.st_size, "sha256": sha256_file(path)})
    paths = {entry["path"] for entry in entries}
    if "preflight.json" not in paths:
        _refuse("campaign lacks preflight.json")
    for arm in ARMS:
        missing = {f"{arm}/{name}" for name in REQUIRED_ARM_FILES} - paths
        if missing:
            _refuse(f"{arm} manifest inputs missing: {sorted(missing)}")
    return entries


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        _refuse(f"refusing to overwrite {path}")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _acceptance(path: Path) -> dict[str, Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ACCEPT.search(line)
        if not match:
            _refuse(f"unrecognized acceptance row in {path}")
        rows.append((float(match.group(1)), int(match.group(2)),
                     int(match.group(3)), float(match.group(4))))
    if len(rows) != 12:
        _refuse(f"{path} must contain exactly 12 acceptance rows")
    accepted = sum(row[1] for row in rows)
    generated = sum(row[2] for row in rows)
    if accepted <= 0 or generated <= 0 or accepted > generated:
        _refuse(f"{path} has impossible acceptance totals")
    return {
        "request_rows": 12, "accepted_tokens": accepted, "generated_tokens": generated,
        "reported_fraction_mean": sum(row[0] for row in rows) / 12,
        "reported_mean_len_mean": sum(row[3] for row in rows) / 12,
        "weighted_fraction": accepted / generated,
    }


def _verify_claims(arm: str, opened: Mapping[str, Any], released: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema", "claim_id", "campaign_id", "device_id", "purpose", "host",
        "holder_pid", "holder_start_ticks", "holder_boot_id", "holder_label", "lock_path",
        "acquired_at", "expires_at", "released_at", "state", "reclaimed_from",
    }
    _exact_keys(opened, expected_keys, f"{arm}.claim_open")
    _exact_keys(released, expected_keys, f"{arm}.claim_released")
    if opened["schema"] != "epyc.autokernel.device_claim_receipt.v1":
        _refuse(f"{arm} claim schema mismatch")
    for key in expected_keys - {"released_at"}:
        if opened[key] != released[key]:
            _refuse(f"{arm} claim {key} drifted across release")
    if opened["released_at"] is not None or not released["released_at"]:
        _refuse(f"{arm} claim release chronology is invalid")
    if opened["device_id"] != "mi210_0" or opened["state"] != "held":
        _refuse(f"{arm} claim is not the exact MI210 lease")
    if not str(opened["campaign_id"]).startswith("df2-5-"):
        _refuse(f"{arm} claim belongs to another campaign")
    if _timestamp(opened["acquired_at"], f"{arm}.claim.acquired_at") >= \
            _timestamp(released["released_at"], f"{arm}.claim.released_at"):
        _refuse(f"{arm} claim release does not follow acquisition")
    return {"claim_id": opened["claim_id"], "campaign_id": opened["campaign_id"],
            "device_id": "mi210_0", "acquired_at": opened["acquired_at"],
            "released_at": released["released_at"]}


def _verify_arm(root: Path, arm: str, preflight: Mapping[str, Any]) -> dict[str, Any]:
    arm_dir = root / arm
    for name in REQUIRED_ARM_FILES:
        _safe_regular(arm_dir / name, root, f"{arm}/{name}")
    opened = _json(arm_dir / "claim-open.json", f"{arm}.claim_open")
    released = _json(arm_dir / "claim-released.json", f"{arm}.claim_released")
    claim = _verify_claims(arm, opened, released)
    transport = _json(arm_dir / "transport.json", f"{arm}.transport")
    if (transport.get("schema") != "epyc.df2.arm_transport.v1" or
            transport.get("arm") != arm or transport.get("runner_returncode") != 0 or
            transport.get("failure") is not None or transport.get("claim_id") != claim["claim_id"] or
            transport.get("claim_released") is not True or
            transport.get("inference_window_released") is not True or
            transport.get("server_returncode") != 0):
        _refuse(f"{arm} transport is not a clean released completion")
    claim_start = _timestamp(claim["acquired_at"], f"{arm}.claim.acquired_at")
    claim_end = _timestamp(claim["released_at"], f"{arm}.claim.released_at")
    run_start = _timestamp(transport.get("started_at"), f"{arm}.transport.started_at")
    run_end = _timestamp(transport.get("finished_at"), f"{arm}.transport.finished_at")
    if not claim_start <= run_start < run_end <= claim_end:
        _refuse(f"{arm} transport does not fit inside its device claim")
    resources = _json(arm_dir / "resource-samples.json", f"{arm}.resources")
    if resources.get("schema") != "epyc.df2.resource_samples.v1":
        _refuse(f"{arm} resource schema mismatch")
    samples = resources.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        _refuse(f"{arm} lacks resource samples")
    for index, row in enumerate(samples):
        if not isinstance(row, Mapping):
            _refuse(f"{arm}.resources.samples[{index}] must be an object")
        observed = _timestamp(row.get("ts"), f"{arm}.resources.samples[{index}].ts")
        if not run_start <= observed <= run_end:
            _refuse(f"{arm} resource sample falls outside the process window")
    server_pid = transport.get("server_pid")
    resident = [row for row in samples if isinstance(row, Mapping) and
                row.get("server_pid") == server_pid and row.get("server_kfd_resident") is True and
                server_pid in row.get("kfd_pids", []) and row.get("kfd_error") is None]
    positive = [row for row in resident if isinstance(row.get("vram_delta_bytes"), int) and
                row["vram_delta_bytes"] > 0 and row.get("vram_error") is None]
    if len(resident) < 2 or len(positive) < 2:
        _refuse(f"{arm} lacks two in-window KFD+positive-VRAM witnesses")
    peak = max(_int(row.get("vram_used_bytes"), f"{arm}.vram_used") for row in positive)
    result = _json(arm_dir / "result.json", f"{arm}.result")
    meta = _mapping(result.get("meta"), f"{arm}.result.meta")
    suites = result.get("suites")
    if not isinstance(suites, list) or len(suites) != 1:
        _refuse(f"{arm} result must have one suite")
    suite = _mapping(suites[0], f"{arm}.result.suites[0]")
    throughput = _mapping(suite.get("throughput"), f"{arm}.throughput")
    np = int(arm.rsplit("np", 1)[1])
    if (suite.get("suite") != "olympiadbench_hard" or suite.get("n") != 12 or
            suite.get("errors") != 0 or throughput.get("concurrency") != np or
            meta.get("arm") != arm or meta.get("n_per_suite") != 12 or
            meta.get("seed") != 42 or meta.get("temperature") != 0.6 or
            meta.get("top_p") != 0.95 or meta.get("top_k") != 20 or
            meta.get("enable_thinking") is not False or meta.get("max_tokens") != 2048 or
            meta.get("runner_source_sha256") != preflight.get("runner_sha256") or
            meta.get("binary") != preflight.get("binary")):
        _refuse(f"{arm} result protocol/identity mismatch")
    expected_models = preflight["target_model"] if arm.startswith("mtp_") else (
        f"{preflight['target_model']};{preflight['draft_model']}")
    if meta.get("models") != expected_models:
        _refuse(f"{arm} result model identity mismatch")
    pq = _jsonl(arm_dir / "pq.jsonl", f"{arm}.pq")
    if len(pq) != 12 or len({row.get("id") for row in pq}) != 12:
        _refuse(f"{arm} must have 12 distinct scored question rows")
    expected_request = {"endpoint": "chat", "request_path": "/v1/chat/completions",
                        "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                        "enable_thinking": False}
    for index, row in enumerate(pq):
        if (row.get("arm") != arm or row.get("suite") != "olympiadbench_hard" or
                row.get("rep") != 0 or row.get("seed") != 42 or row.get("request_error") != "" or
                row.get("effective_request") != expected_request or
                row.get("runner_source_sha256") != preflight.get("runner_sha256")):
            _refuse(f"{arm}.pq[{index}] protocol/producer mismatch")
    acceptance = _acceptance(arm_dir / "acceptance.txt")
    summary = _json(arm_dir / "summary.json", f"{arm}.summary")
    decode = _positive(throughput.get("aggregate_decode_tok_s"), f"{arm}.decode")
    if (summary.get("arm") != arm or summary.get("aggregate_decode_tok_s") != decode or
            summary.get("claim_id") != claim["claim_id"] or
            summary.get("claim_released") is not True or
            summary.get("kfd_resident_samples") != len(resident) or
            summary.get("positive_vram_samples") != len(positive) or
            summary.get("peak_vram_used_bytes") != peak):
        _refuse(f"{arm} summary does not rederive")
    return {"arm": arm, "np": np, "category": "BASELINE" if arm.startswith("mtp_") else "CANDIDATE",
            "decode_tokens_per_s": decode, "acceptance": acceptance, "claim": claim,
            "kfd_resident_samples": len(resident), "positive_vram_samples": len(positive),
            "resource_sample_count": len(samples), "peak_vram_used_bytes": peak,
            "artifact_sha256": {name: sha256_file(arm_dir / name) for name in sorted(REQUIRED_ARM_FILES)}}


def _rows(campaign_id: str, created_at: str, campaign_locator: str,
          manifest_locator: str, manifest_sha: str, producer: Mapping[str, Any],
          preflight: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        record = arms[arm]
        common = {
            "date": created_at[:10], "category": record["category"],
            "metric_direction": "higher_better", "protocol_id": PROTOCOL_ID,
            "reps": 12, "reps_basis": "scored:completed_requests",
            "attestation_locator": manifest_locator, "attestation_sha256": manifest_sha,
            "attestation_present": True,
            "source_kind": "dflash2-experimental-runtime-measurement",
            "extra": {
                "campaign_id": campaign_id, "campaign_locator": campaign_locator,
                "campaign_kind": CAMPAIGN_KIND, "authority": AUTHORITY,
                "experimental_runtime": True, "source_mutation_strategy": False,
                "kernel_champion_authority": False, "promotion_authority": False,
                "production_authority": False, "arm": arm, "np": record["np"],
                "source_commit": preflight["source_commit"],
                "binary_sha256": preflight["binary_sha256"],
                "target_model_sha256": preflight["target_model_sha256"],
                "draft_model_sha256": preflight["draft_model_sha256"],
                "runner_sha256": preflight["runner_sha256"],
                "questions_sha256": preflight["questions_sha256"],
                "claim": record["claim"],
                "kfd_resident_samples": record["kfd_resident_samples"],
                "positive_vram_samples": record["positive_vram_samples"],
                "resource_sample_count": record["resource_sample_count"],
                "peak_vram_used_bytes": record["peak_vram_used_bytes"],
                "artifact_sha256": record["artifact_sha256"],
                "producer_sha256": producer["sha256"],
            },
        }
        identity = canonical_sha256({"campaign_id": campaign_id, "arm": arm,
                                    "manifest_sha256": manifest_sha})[:20]
        throughput = {
            "measurement_id": f"df2_{identity}_decode", "metric": "decode_tokens_per_s",
            "value": record["decode_tokens_per_s"], "unit": "tokens_per_second",
            "claim": f"DFlash2 experimental-runtime {arm} recorded aggregate decode throughput",
            **common,
        }
        throughput["measurement_sha256"] = _measurement_sha(throughput)
        acceptance = {
            "measurement_id": f"df2_{identity}_acceptance", "metric": "draft_acceptance_fraction",
            "value": record["acceptance"]["weighted_fraction"], "unit": "fraction",
            "claim": f"DFlash2 experimental-runtime {arm} recorded weighted draft acceptance",
            **common,
        }
        acceptance["extra"] = {**common["extra"], "accepted_tokens": record["acceptance"]["accepted_tokens"],
                               "generated_tokens": record["acceptance"]["generated_tokens"]}
        acceptance["measurement_sha256"] = _measurement_sha(acceptance)
        rows.extend((throughput, acceptance))
    return rows


def finalize_concurrency(root: str | Path, *, created_at: str | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    if not root.is_dir() or root.is_symlink():
        _refuse("campaign root must be a real directory")
    for output in (root / "campaign-manifest.json", root / "campaign-summary.json"):
        if output.exists():
            _refuse("campaign finalization is write-once")
    preflight = _json(root / "preflight.json", "preflight")
    _exact_keys(preflight, {
        "schema", "campaign_id", "campaign_kind", "authority", "created_at", "source_root",
        "source_commit", "binary", "binary_sha256", "target_model", "target_model_sha256",
        "draft_model", "draft_model_sha256", "questions", "questions_sha256", "runner",
        "runner_sha256", "parity_client", "parity_client_sha256", "protocol", "route_authority",
    }, "preflight")
    if preflight.get("schema") != "epyc.df2.followups_preflight.v2":
        _refuse("preflight predates the prospective DF2 belief hook")
    if (preflight.get("campaign_kind") != CAMPAIGN_KIND or
            preflight.get("authority") != AUTHORITY or
            preflight.get("protocol") != EXPECTED_PROTOCOL):
        _refuse("preflight campaign/protocol authority mismatch")
    for key in ("binary_sha256", "target_model_sha256", "draft_model_sha256",
                "runner_sha256", "questions_sha256"):
        _sha(preflight.get(key), f"preflight.{key}")
    for key in ("binary", "target_model", "draft_model", "runner", "questions"):
        path = Path(str(preflight.get(key, "")))
        _external_regular(path, f"preflight.{key}")
        if sha256_file(path) != preflight[f"{key}_sha256" if key != "runner" else "runner_sha256"]:
            _refuse(f"preflight {key} current-byte identity mismatch")
    if preflight.get("source_commit") != "2046c64e9948671c7557428b198acebc6f416575":
        _refuse("preflight source commit mismatch")
    entries = _manifest_entries(root)
    manifest = {"schema": MANIFEST_SCHEMA, "campaign_id": preflight["campaign_id"],
                "entries": entries, "entries_sha256": canonical_sha256(entries)}
    manifest_path = root / "campaign-manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    arms = {arm: _verify_arm(root, arm, preflight) for arm in ARMS}
    stamp = created_at or datetime.now(timezone.utc).isoformat()
    _timestamp(stamp, "created_at")
    producer = _producer()
    campaign_locator = str(root / "campaign-summary.json")
    rows = _rows(preflight["campaign_id"], stamp, campaign_locator, str(manifest_path),
                 manifest_sha, producer, preflight, arms)
    receipt = {
        "schema": SCHEMA, "campaign_id": preflight["campaign_id"], "created_at": stamp,
        "campaign_kind": CAMPAIGN_KIND, "authority": AUTHORITY, "producer": producer,
        "preflight": {"locator": str(root / "preflight.json"),
                      "sha256": sha256_file(root / "preflight.json")},
        "manifest": {"locator": str(manifest_path), "sha256": manifest_sha,
                     "entries_sha256": manifest["entries_sha256"]},
        "protocol": preflight["protocol"], "arms": arms, "belief_measurements": rows,
    }
    receipt["receipt_sha256"] = _receipt_sha(receipt)
    _write_json(root / "campaign-summary.json", receipt)
    return receipt


def native_rows(source: str | Path | Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(source, Mapping):
        receipt = source
        locator = ""
    else:
        raw_path = Path(source)
        _external_regular(raw_path, "campaign receipt")
        path = raw_path.resolve()
        receipt = _json(path, "campaign")
        locator = str(path)
    # The completed DF2-4 panel and all other pre-hook shapes intentionally project zero rows.
    if receipt.get("schema") != SCHEMA or "belief_measurements" not in receipt:
        return []
    required = {"schema", "campaign_id", "created_at", "campaign_kind", "authority", "producer",
                "preflight", "manifest", "protocol", "arms", "belief_measurements", "receipt_sha256"}
    _exact_keys(receipt, required, "campaign")
    if receipt["receipt_sha256"] != _receipt_sha(receipt):
        _refuse("campaign receipt hash mismatch")
    if receipt["campaign_kind"] != CAMPAIGN_KIND or receipt["authority"] != AUTHORITY:
        _refuse("campaign authority was widened")
    producer = _mapping(receipt["producer"], "producer")
    if producer != _producer():
        _refuse("campaign producer identity mismatch")
    manifest_ref = _mapping(receipt["manifest"], "manifest")
    raw_manifest_path = Path(str(manifest_ref.get("locator", "")))
    root = raw_manifest_path.parent
    if root.is_symlink() or not root.is_dir() or raw_manifest_path.name != "campaign-manifest.json":
        _refuse("campaign manifest locator is not the real campaign root")
    _safe_regular(raw_manifest_path, root, "campaign-manifest.json")
    manifest_path = raw_manifest_path.resolve()
    if sha256_file(manifest_path) != _sha(manifest_ref.get("sha256"), "manifest.sha256"):
        _refuse("campaign manifest byte hash mismatch")
    manifest = _json(manifest_path, "manifest")
    if (manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("campaign_id") != receipt["campaign_id"] or
            manifest.get("entries_sha256") != canonical_sha256(manifest.get("entries")) or
            manifest.get("entries_sha256") != manifest_ref.get("entries_sha256")):
        _refuse("campaign manifest structure mismatch")
    root = manifest_path.parent
    observed_entries = _manifest_entries(root)
    if observed_entries != manifest.get("entries"):
        _refuse("campaign artifact closure changed after finalization")
    preflight_ref = _mapping(receipt["preflight"], "preflight_ref")
    raw_preflight_path = Path(str(preflight_ref.get("locator", "")))
    if raw_preflight_path != root / "preflight.json":
        _refuse("campaign preflight locator mismatch")
    _safe_regular(raw_preflight_path, root, "preflight.json")
    preflight_path = raw_preflight_path.resolve()
    if sha256_file(preflight_path) != _sha(preflight_ref.get("sha256"), "preflight.sha256"):
        _refuse("campaign preflight hash mismatch")
    preflight = _json(preflight_path, "preflight")
    arms = {arm: _verify_arm(root, arm, preflight) for arm in ARMS}
    if receipt["arms"] != arms or receipt["protocol"] != preflight.get("protocol"):
        _refuse("campaign arms/protocol do not rederive")
    expected = _rows(receipt["campaign_id"], receipt["created_at"],
                     locator or str(root / "campaign-summary.json"), str(manifest_path),
                     manifest_ref["sha256"], producer, preflight, arms)
    if receipt["belief_measurements"] != expected:
        _refuse("producer-authored belief measurements do not rederive")
    source_locator = locator or str(root / "campaign-summary.json")
    return [{"source": receipt, "source_locator": source_locator, "measurement": row}
            for row in expected]


def project(native: Mapping[str, Any]) -> dict[str, Any]:
    """Project one validated native row into the ClaimTuple constructor grammar."""
    if not isinstance(native, Mapping):
        _refuse("native measurement must be an object")
    _exact_keys(native, {"source", "source_locator", "measurement"}, "native")
    source_locator = native.get("source_locator")
    if not isinstance(source_locator, str) or not source_locator:
        _refuse("native source_locator is required")
    derived = native_rows(native.get("source"))
    measurement = _mapping(native.get("measurement"), "native.measurement")
    matches = [row for row in derived
               if row["measurement"].get("measurement_id") == measurement.get("measurement_id")]
    if len(matches) != 1 or matches[0]["measurement"] != measurement:
        _refuse("native measurement is not an exact producer-authored row")
    expected_locator = measurement.get("extra", {}).get("campaign_locator")
    if source_locator != expected_locator:
        _refuse("native source locator does not match the campaign locator")
    if measurement.get("measurement_sha256") != _measurement_sha(measurement):
        _refuse("native measurement hash mismatch")
    required = {"measurement_id", "metric", "value", "date", "category", "claim",
                "metric_direction", "protocol_id", "reps", "reps_basis", "unit",
                "attestation_locator", "attestation_sha256", "source_kind", "extra",
                "attestation_present", "measurement_sha256"}
    _exact_keys(measurement, required, "native_measurement")
    if measurement["metric_direction"] != "higher_better":
        _refuse("DF2 measurement direction must be producer-authored higher_better")
    if measurement["category"] not in {"BASELINE", "CANDIDATE"} or measurement["reps"] != 12:
        _refuse("DF2 measurement category/reps mismatch")
    extra = _mapping(measurement["extra"], "native_measurement.extra")
    if (extra.get("campaign_kind") != CAMPAIGN_KIND or extra.get("experimental_runtime") is not True or
            extra.get("source_mutation_strategy") is not False or
            extra.get("kernel_champion_authority") is not False or
            extra.get("promotion_authority") is not False or
            extra.get("production_authority") is not False):
        _refuse("DF2 measurement authority boundary mismatch")
    return {key: measurement[key] for key in required - {"measurement_sha256"}}
