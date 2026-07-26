#!/usr/bin/env python3
"""Prepare and score the bounded MiniCPM-o Phase-1 vision observation.

This tool deliberately performs no inference and makes no network requests.  It
pins the local OCRBench/ChartQA assets used by the future paired run, scores
saved model replies, and computes paired statistics.  The resulting evidence is
an observation until a human ratifies a decision-grade M-1 protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/raid0/llm/epyc-inference-research")
SUITE = ROOT / "benchmarks/prompts/debug/vl.yaml"
IMAGES = ROOT / "benchmarks/images/vl"
SHORT_ANSWER_SUFFIX = "\nReply with only the answer, with no explanation."
SCHEMA = "epyc.minicpm.phase1.m1-observation.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_RECORD_FIELDS = (
    "case_id",
    "raw_content",
    "model_sha256",
    "mmproj_sha256",
    "binary_sha256",
    "endpoint_or_sidecar",
    "started_at",
    "finished_at",
    "request_parameters",
)


# Every expected value is copied from the source-labelled local debug suite.
# The protocol suffix only constrains response formatting for exact scoring.
CASES: tuple[dict[str, Any], ...] = (
    {"role": "worker_vision", "id": "vl_ocr_0001", "image": "ocrbench/ocr_0001.png", "prompt": "what is written in the image?", "answers": ["FRIEND"]},
    {"role": "worker_vision", "id": "vl_ocr_0201", "image": "ocrbench/ocr_0201.png", "prompt": "what is the number in the image?", "answers": ["1056"]},
    {"role": "worker_vision", "id": "vl_ocr_0247", "image": "ocrbench/ocr_0247.png", "prompt": "what is the number in the image?", "answers": ["76961"]},
    {"role": "worker_vision", "id": "vl_ocr_0248", "image": "ocrbench/ocr_0248.png", "prompt": "what is the number in the image?", "answers": ["31000"]},
    {"role": "worker_vision", "id": "vl_chart_test_1311", "image": "chartqa/chart_test_1311.png", "prompt": "What percentage of parents base the amount of pocket money on their child's age?", "answers": ["29"]},
    {"role": "worker_vision", "id": "vl_chart_test_1315", "image": "chartqa/chart_test_1315.png", "prompt": "How many metric tons of CO2 were emitted from coal combustion in 1971?", "answers": ["5230"]},
    {"role": "worker_vision", "id": "vl_chart_test_1441", "image": "chartqa/chart_test_1441.png", "prompt": "What was the main source of petroleum products for the UK in 2019?", "answers": ["Netherlands"]},
    {"role": "worker_vision", "id": "vl_chart_test_2114", "image": "chartqa/chart_test_2114.png", "prompt": "How many new scooters were registered in April 2020?", "answers": ["517"]},
    {"role": "vision_escalation", "id": "vl_ocr_0839", "image": "ocrbench/ocr_0839.png", "prompt": "what is the value for Total carbohydrate of per 100g/ml? Answer this question using the text in the image directly.", "answers": ["41.0g", "41.0 g"]},
    {"role": "vision_escalation", "id": "vl_ocr_0562", "image": "ocrbench/ocr_0562.png", "prompt": "How many patients came from the neighboring state of Mexico?", "answers": ["63086", "63 086", "63,086"]},
    {"role": "vision_escalation", "id": "vl_ocr_0632", "image": "ocrbench/ocr_0632.png", "prompt": "what is the average of all No confidence data?", "answers": ["50.6"]},
    {"role": "vision_escalation", "id": "vl_chart_test_0051", "image": "chartqa/chart_test_0051.png", "prompt": "How many games in the chart have over 40 ratings?", "answers": ["4"]},
    {"role": "vision_escalation", "id": "vl_chart_test_2284", "image": "chartqa/chart_test_2284.png", "prompt": "How many enterprises were in the manufacture of electronic components industry in Sweden in 2013?", "answers": ["282"]},
    {"role": "vision_escalation", "id": "vl_chart_test_2482", "image": "chartqa/chart_test_2482.png", "prompt": "Who was the highest paid actress between June 2017 and June 2018?", "answers": ["Sofia Vergara"]},
    {"role": "vision_escalation", "id": "vl_chart_test_0109", "image": "chartqa/chart_test_0109.png", "prompt": "What's the median value of light blue bar?", "answers": ["37"]},
    {"role": "vision_escalation", "id": "vl_chart_test_0209", "image": "chartqa/chart_test_0209.png", "prompt": "What is the difference in the value of High blood sugar and High Blood pressure?", "answers": ["203"]},
    {"role": "vision_escalation", "id": "vl_chart_test_0285", "image": "chartqa/chart_test_0285.png", "prompt": "What's the average of two smallest bar??", "answers": ["0.235"]},
    {"role": "vision_escalation", "id": "vl_chart_test_0563", "image": "chartqa/chart_test_0563.png", "prompt": "What is the difference between maximum values of International flight and Domestic flight?", "answers": ["0.11"]},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_suite_cases() -> dict[str, dict[str, Any]]:
    """Parse the constrained local debug-suite YAML without a YAML dependency."""
    cases: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    in_alternatives = False
    for line in SUITE.read_text(encoding="utf-8").splitlines():
        if line.startswith("  - id: "):
            if current is not None:
                cases[current["id"]] = current
            current = {"id": line.removeprefix("  - id: ").strip(), "alt_answers": []}
            in_alternatives = False
            continue
        if current is None:
            continue
        if line.startswith("    image_path: "):
            current["image_path"] = json.loads(line.removeprefix("    image_path: "))
            in_alternatives = False
        elif line.startswith("    prompt: "):
            current["prompt"] = json.loads(line.removeprefix("    prompt: "))
            in_alternatives = False
        elif line.startswith("    expected: "):
            current["expected"] = json.loads(line.removeprefix("    expected: "))
            in_alternatives = False
        elif line == "    alt_answers:":
            in_alternatives = True
        elif in_alternatives and line.startswith("      - "):
            current["alt_answers"].append(json.loads(line.removeprefix("      - ")))
        elif line.startswith("    "):
            in_alternatives = False
    if current is not None:
        cases[current["id"]] = current
    return cases


def assert_source_parity() -> None:
    source = source_suite_cases()
    for case in CASES:
        found = source.get(case["id"])
        if found is None:
            raise ValueError(f"source suite missing {case['id']}")
        expected_image = str(IMAGES / case["image"])
        if found.get("image_path") != expected_image:
            raise ValueError(f"image mismatch for {case['id']}")
        if found.get("prompt") != case["prompt"]:
            raise ValueError(f"prompt mismatch for {case['id']}")
        source_answers = [found.get("expected"), *found.get("alt_answers", [])]
        if case["answers"] != source_answers:
            raise ValueError(f"accepted-answer mismatch for {case['id']}")


def normalize_answer(value: str) -> str:
    """Normalize presentation only; never use substring matching."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().strip().split())


def score_response(raw_content: str, accepted_answers: list[str]) -> dict[str, Any]:
    normalized = normalize_answer(raw_content)
    accepted = [normalize_answer(answer) for answer in accepted_answers]
    return {
        "method": "normalized_exact_accepted_alternative",
        "raw_content": raw_content,
        "normalized_content": normalized,
        "accepted_answers": accepted_answers,
        "normalized_accepted_answers": accepted,
        "pass": normalized in accepted,
    }


def atomic_or_verify_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"refusing to overwrite non-identical evidence: {path}")
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def manifest_for_role(role: str) -> dict[str, Any]:
    assert_source_parity()
    fixtures = []
    for case in CASES:
        if case["role"] != role:
            continue
        image = IMAGES / case["image"]
        if not image.is_file():
            raise FileNotFoundError(image)
        fixtures.append(
            {
                "case_id": case["id"],
                "image": str(image),
                "image_sha256": sha256(image),
                "source_dataset": "OCRBench" if case["id"].startswith("vl_ocr_") else "ChartQA",
                "source_suite": str(SUITE),
                "source_suite_sha256": sha256(SUITE),
                "source_prompt": case["prompt"],
                "prompt": case["prompt"] + SHORT_ANSWER_SUFFIX,
                "accepted_answers": case["answers"],
                "scoring": "normalized_exact_accepted_alternative",
            }
        )
    return {
        "schema": SCHEMA,
        "role": role,
        "protocol_status": "observation_only_unratified",
        "decision_use": "No lineup, registry, or deployment decision may use this artifact as a gate.",
        "limitations": [
            "No source-backed spatial-reasoning fixture is included in the local corpus.",
            "This is a narrow local OCR/chart screen, not a broad role-quality certification.",
            "Both arms must use the exact manifest prompt, image bytes, seed, temperature, and max_tokens.",
        ],
        "run_contract": {
            "temperature": 0,
            "seed": 35,
            "max_tokens": 32,
            "response_format": "plain short answer",
            "required_record_fields": [
                "case_id", "raw_content", "model_sha256", "mmproj_sha256", "binary_sha256",
                "endpoint_or_sidecar", "started_at", "finished_at", "request_parameters",
            ],
        },
        "fixtures": fixtures,
    }


def write_manifests(output_dir: Path) -> None:
    for role in ("worker_vision", "vision_escalation"):
        path = output_dir / f"m1_{role}_manifest.json"
        value = manifest_for_role(role)
        atomic_or_verify_json(path, value)


def parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("timestamp must be ISO-8601") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed


def validate_provenance(row: dict[str, Any], contract: dict[str, Any]) -> None:
    for field in REQUIRED_RECORD_FIELDS:
        if field not in row:
            raise ValueError(f"response row missing required field: {field}")
    for field in ("model_sha256", "mmproj_sha256", "binary_sha256"):
        if not isinstance(row[field], str) or not SHA256_RE.fullmatch(row[field]):
            raise ValueError(f"invalid {field}")
    if not isinstance(row["endpoint_or_sidecar"], str) or not row["endpoint_or_sidecar"].strip():
        raise ValueError("endpoint_or_sidecar must be a nonempty string")
    started_at = parse_timestamp(row["started_at"])
    finished_at = parse_timestamp(row["finished_at"])
    if finished_at < started_at:
        raise ValueError("finished_at precedes started_at")
    parameters = row["request_parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("request_parameters must be an object")
    for name in ("temperature", "seed", "max_tokens"):
        if parameters.get(name) != contract[name]:
            raise ValueError(f"request_parameters.{name} does not match manifest")


def index_by_case(rows: list[dict[str, Any]], expected_ids: set[str], contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id not in expected_ids or case_id in indexed:
            raise ValueError(f"invalid, unexpected, or duplicate case_id: {case_id!r}")
        if not isinstance(row.get("raw_content"), str):
            raise ValueError(f"missing raw_content for {case_id}")
        validate_provenance(row, contract)
        indexed[case_id] = row
    if set(indexed) != expected_ids:
        raise ValueError("response set must exactly equal manifest fixture IDs")
    return indexed


def score_saved_responses(manifest: dict[str, Any], rows: list[dict[str, Any]], manifest_sha256: str, arm_id: str) -> dict[str, Any]:
    if not SHA256_RE.fullmatch(manifest_sha256):
        raise ValueError("invalid manifest SHA-256")
    if not isinstance(arm_id, str) or not arm_id.strip():
        raise ValueError("arm_id must be a nonempty string")
    fixtures = manifest["fixtures"]
    by_id = index_by_case(rows, {fixture["case_id"] for fixture in fixtures}, manifest["run_contract"])
    arm_provenance = {
        key: next(iter({row[key] for row in by_id.values()}))
        for key in ("model_sha256", "mmproj_sha256", "binary_sha256", "endpoint_or_sidecar")
    }
    if any(len({row[key] for row in by_id.values()}) != 1 for key in arm_provenance):
        raise ValueError("all response rows in an arm must share model/mmproj/binary hashes and endpoint")
    scored = []
    for fixture in fixtures:
        row = by_id[fixture["case_id"]]
        scored.append({"case_id": fixture["case_id"], "score": score_response(row["raw_content"], fixture["accepted_answers"]), "provenance": {key: value for key, value in row.items() if key != "raw_content"}})
    return {
        "schema": SCHEMA + ".scored-responses.v1",
        "protocol_status": manifest["protocol_status"],
        "role": manifest["role"],
        "manifest_sha256": manifest_sha256,
        "arm_id": arm_id,
        "arm_provenance": arm_provenance,
        "total": len(scored),
        "passed": sum(item["score"]["pass"] for item in scored),
        "rows": scored,
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant pairs b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def validate_scored_output(scored: dict[str, Any]) -> list[dict[str, Any]]:
    scored_schema = SCHEMA + ".scored-responses.v1"
    if scored.get("schema") != scored_schema:
        raise ValueError("paired input must use the scored-response protocol schema")
    rows = scored.get("rows")
    if not isinstance(rows, list):
        raise ValueError("scored rows must be a list")
    if not isinstance(scored.get("total"), int) or isinstance(scored["total"], bool) or scored["total"] != len(rows):
        raise ValueError("stored total must equal the number of scored rows")
    arm_provenance = scored.get("arm_provenance")
    if not isinstance(arm_provenance, dict):
        raise ValueError("scored output missing arm_provenance")
    for key in ("model_sha256", "mmproj_sha256", "binary_sha256"):
        if not isinstance(arm_provenance.get(key), str) or not SHA256_RE.fullmatch(arm_provenance[key]):
            raise ValueError(f"invalid arm provenance {key}")
    if not isinstance(arm_provenance.get("endpoint_or_sidecar"), str) or not arm_provenance["endpoint_or_sidecar"].strip():
        raise ValueError("invalid arm provenance endpoint_or_sidecar")
    case_ids: set[str] = set()
    passed = 0
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str) or row["case_id"] in case_ids:
            raise ValueError("scored rows must have unique string case IDs")
        case_ids.add(row["case_id"])
        score = row.get("score")
        if not isinstance(score, dict) or not isinstance(score.get("pass"), bool):
            raise ValueError("scored row missing boolean pass result")
        provenance = row.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("case_id") != row["case_id"]:
            raise ValueError("scored row has malformed provenance")
        if any(provenance.get(key) != value for key, value in arm_provenance.items()):
            raise ValueError("scored row provenance does not match arm provenance")
        passed += score["pass"]
    if not isinstance(scored.get("passed"), int) or isinstance(scored["passed"], bool) or scored["passed"] != passed:
        raise ValueError("stored passed count is inconsistent with scored rows")
    return rows


def paired_analysis(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_rows = validate_scored_output(baseline)
    candidate_rows = validate_scored_output(candidate)
    if baseline["role"] != candidate["role"]:
        raise ValueError("paired inputs must have the same role")
    if baseline.get("protocol_status") != candidate.get("protocol_status"):
        raise ValueError("paired inputs must have the same protocol status")
    if baseline.get("manifest_sha256") != candidate.get("manifest_sha256"):
        raise ValueError("paired inputs must bind the identical manifest SHA-256")
    if not SHA256_RE.fullmatch(str(baseline.get("manifest_sha256", ""))):
        raise ValueError("paired inputs contain an invalid manifest SHA-256")
    if not isinstance(baseline.get("arm_id"), str) or not isinstance(candidate.get("arm_id"), str) or baseline["arm_id"] == candidate["arm_id"]:
        raise ValueError("paired inputs must declare distinct nonempty arms")
    base = {row["case_id"]: bool(row["score"]["pass"]) for row in baseline_rows}
    cand = {row["case_id"]: bool(row["score"]["pass"]) for row in candidate_rows}
    if set(base) != set(cand):
        raise ValueError("paired inputs must contain the same case IDs")
    both_pass = sum(base[key] and cand[key] for key in base)
    baseline_only = sum(base[key] and not cand[key] for key in base)
    candidate_only = sum(not base[key] and cand[key] for key in base)
    neither = sum(not base[key] and not cand[key] for key in base)
    n = len(base)
    return {
        "schema": SCHEMA + ".paired-analysis.v1",
        "protocol_status": "observation_only_unratified",
        "role": baseline["role"],
        "manifest_sha256": baseline["manifest_sha256"],
        "arms": {"baseline": {"arm_id": baseline["arm_id"], **baseline["arm_provenance"]}, "candidate": {"arm_id": candidate["arm_id"], **candidate["arm_provenance"]}},
        "n": n,
        "baseline_pass_rate": sum(base.values()) / n,
        "candidate_pass_rate": sum(cand.values()) / n,
        "candidate_minus_baseline_pp": 100 * (sum(cand.values()) - sum(base.values())) / n,
        "paired_2x2": {"both_pass": both_pass, "baseline_only": baseline_only, "candidate_only": candidate_only, "neither": neither},
        "mcnemar_exact_two_sided_p": mcnemar_exact(baseline_only, candidate_only),
        "limitation": "Observation only; no decision threshold is asserted.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifests", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--scored-out", type=Path)
    parser.add_argument("--arm-id")
    parser.add_argument("--baseline-scored", type=Path)
    parser.add_argument("--candidate-scored", type=Path)
    parser.add_argument("--paired-out", type=Path)
    args = parser.parse_args(argv)
    write = bool(args.write_manifests)
    score_values = (args.manifest, args.responses, args.scored_out, args.arm_id)
    pair_values = (args.baseline_scored, args.candidate_scored, args.paired_out)
    score = all(score_values)
    pair = all(pair_values)
    if sum((write, score, pair)) != 1 or (any(score_values) and not score) or (any(pair_values) and not pair):
        parser.error("select exactly one complete operation; mixed operation arguments are invalid")
    if write:
        write_manifests(args.write_manifests)
        return 0
    if score:
        manifest_path = args.manifest
        atomic_or_verify_json(args.scored_out, score_saved_responses(json.loads(manifest_path.read_text()), json.loads(args.responses.read_text()), sha256(manifest_path), args.arm_id))
        return 0
    if pair:
        atomic_or_verify_json(args.paired_out, paired_analysis(json.loads(args.baseline_scored.read_text()), json.loads(args.candidate_scored.read_text())))
        return 0
    raise AssertionError("operation validation should have exited")


if __name__ == "__main__":
    raise SystemExit(main())
