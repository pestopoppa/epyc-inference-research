#!/usr/bin/env python3
"""Read RUS3 native capture; legacy RUS2 files/semantics stay untouched."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from rus3_repeated_samples import SAFE_SCORERS, decode_response, digest, load_scorer, request_body

from rus2_trace_length_probe import boot, review_ratio, stat_within, within_questions


def verify_optional_artifact(directory: Path, record: dict, path_key: str, digest_key: str) -> None:
    if path_key not in record and digest_key not in record:
        return
    name = record.get(path_key)
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError(f"invalid {path_key} locator")
    if digest((directory / name).read_bytes()) != record.get(digest_key):
        raise ValueError(f"{path_key} digest mismatch")


def valid_process_identity(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if (type(value.get("pid")) is not int or value["pid"] <= 0
            or type(value.get("pgid")) is not int or value["pgid"] <= 0
            or type(value.get("start_ticks")) is not int or value["start_ticks"] <= 0
            or not isinstance(value.get("boot_id"), str)
            or not re.fullmatch(r"[0-9a-fA-F-]{36}", value["boot_id"])):
        return False
    argv = value.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
        return False
    expected = hashlib.sha256(json.dumps(argv, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()
    return value.get("argv_sha256") == expected


def valid_object_set(rows: object) -> set[tuple[str, str]] | None:
    if not isinstance(rows, list) or not rows:
        return None
    result = set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("path"), str)
                or not Path(row["path"]).is_absolute()
                or not isinstance(row.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            return None
        result.add((row["path"], row["sha256"]))
    return result if result else None


def analyze(directory: Path, scorer_source: Path) -> dict:
    config = json.loads((directory / "config.json").read_bytes())
    config_digest = hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()
    manifest = json.loads((directory / "manifest.json").read_bytes())
    def optional_bytes(name: str) -> bytes | None:
        path = directory / name
        return path.read_bytes() if path.is_file() else None
    grant_raw = optional_bytes("grant-receipt")
    preflight_raw = optional_bytes("preflight-receipt")
    owner_start_raw = optional_bytes("owner-start-receipt")
    expected_metadata = {"config_sha256": config_digest,
                         "questions_sha256": digest((directory / "questions.jsonl").read_bytes()),
                         "launch_identity_sha256": digest((directory / "launch-identity.json").read_bytes()),
                         "scorer_sha256": digest((directory / "scorer-source.py").read_bytes()),
                         "owner_guard_sha256": digest((directory / "owner-guard-source").read_bytes())}
    if not (directory / "launch-identity.json").read_bytes().strip():
        raise ValueError("empty owner receipt")
    if any(manifest.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("capture manifest artifact digest mismatch")
    evidence_metadata = {
        "grant_receipt_sha256": digest(grant_raw) if grant_raw is not None else None,
        "preflight_sha256": digest(preflight_raw) if preflight_raw is not None else None,
        "owner_start_sha256": digest(owner_start_raw) if owner_start_raw is not None else None,
        "source_sha256": manifest.get("producer_sha256"),
    }
    if any(manifest.get(key) != value for key, value in evidence_metadata.items() if key != "source_sha256"):
        raise ValueError("grant/preflight/owner-start evidence digest mismatch")
    complete_path = directory / "COMPLETE.json"
    complete_raw = complete_path.read_bytes() if complete_path.is_file() else b""
    complete_record = json.loads(complete_raw) if complete_raw else None
    samples_raw = (directory / "samples.jsonl").read_bytes()
    complete_marker_matches = bool(
        isinstance(complete_record, dict)
        and complete_record.get("status") == "captured"
        and complete_record.get("run_id") == config.get("run_id")
        and complete_record.get("samples_sha256") == digest(samples_raw)
        and complete_record.get("grant_receipt_sha256") == evidence_metadata["grant_receipt_sha256"]
        and complete_record.get("preflight_sha256") == evidence_metadata["preflight_sha256"]
        and complete_record.get("owner_start_sha256") == evidence_metadata["owner_start_sha256"]
        and complete_record.get("source_sha256") == evidence_metadata["source_sha256"]
        and complete_record.get("config_sha256") == config_digest
        and complete_record.get("questions_sha256") == expected_metadata["questions_sha256"]
        and complete_record.get("scorer_sha256") == expected_metadata["scorer_sha256"]
        and valid_process_identity(complete_record.get("server_identity"))
        and valid_object_set(complete_record.get("ggml_maps_before")) is not None
        and valid_object_set(complete_record.get("ggml_maps_after")) is not None
    )
    # Structural capture completion is distinct from an authenticated measurement
    # warrant. Direct smoke has no grant, and no retroactive warrant is inferred.
    owner_evidence_authenticated = False
    process_evidence_valid = complete_marker_matches
    for key in ("scorer_sha256", "owner_guard_sha256"):
        if config.get(key) != expected_metadata[key]:
            raise ValueError("declared source pin differs from artifact")
    scorer = load_scorer(scorer_source, config["scorer_sha256"])
    if config.get("protocol_id") != "":
        raise ValueError("RUS3 has no ratified protocol citation")
    if type(config.get("k")) is not int or config["k"] < 4:
        raise ValueError("native configuration requires k >=4")
    questions = [json.loads(line) for line in (directory / "questions.jsonl").read_text().splitlines() if line.strip()]
    expected_questions = {(question["suite"], question["id"]) for question in questions}
    if len(expected_questions) != len(questions):
        raise ValueError("duplicate frozen question identity")
    question_by_id = {(question["suite"], question["id"]): question for question in questions}
    if any(question["scoring_method"] not in SAFE_SCORERS for question in questions):
        raise ValueError("unsupported scoring method in frozen manifest")
    question_order = {(question["suite"], question["id"]): index for index, question in enumerate(questions)}
    counts = collections.defaultdict(collections.Counter)
    for suite, _ in expected_questions:
        for key in ("received", "retained", "refused", "truncated", "unscoreable",
                    "scorer_unavailable", "answer_parse_failed", "trace_unavailable"):
            counts[(config["role"], suite)][key] = 0
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    traces = collections.defaultdict(lambda: collections.defaultdict(list))
    seen = set()
    for line in (directory / "samples.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("schema") != "epyc.rus3.native-sample.v1":
            raise ValueError("not RUS3 native source")
        cell = (record["role"], record["suite"])
        count = counts[cell]
        count["received"] += 1
        if any(record.get(key) != value for key, value in manifest.items()):
            raise ValueError("sample differs from capture manifest")
        if complete_marker_matches:
            identity_record = complete_record.get("server_identity")
            sample_window = (record.get("started_monotonic"), record.get("ended_monotonic"))
            outer_window = (complete_record.get("started_monotonic"),
                            complete_record.get("ended_monotonic"))
            if (not isinstance(identity_record, dict)
                    or record.get("server_pid") != identity_record.get("pid")
                    or record.get("server_pgid") != identity_record.get("pgid")
                    or record.get("server_start_ticks") != identity_record.get("start_ticks")
                    or record.get("server_boot_id") != identity_record.get("boot_id")
                    or not all(isinstance(value, (int, float)) for value in sample_window + outer_window)
                    or not outer_window[0] <= sample_window[0] < sample_window[1] <= outer_window[1]):
                process_evidence_valid = False
            row_maps = valid_object_set(record.get("ggml_map_identity"))
            end_maps = valid_object_set(complete_record.get("ggml_maps_after"))
            if row_maps is None or end_maps is None or row_maps != end_maps:
                process_evidence_valid = False
        if (record.get("config_sha256") != config_digest or record.get("protocol_id") != ""
                or record.get("temperature") != config["temperature"]
                or record.get("run_id") != config["run_id"] or record.get("role") != config["role"]):
            raise ValueError("mixed capture configuration")
        identity = (record["suite"], record["question_id"], record["sample_index"])
        if ((record["suite"], record["question_id"]) not in expected_questions
                or type(record["sample_index"]) is not int
                or not 0 <= record["sample_index"] < config["k"]):
            raise ValueError("sample outside frozen question/K manifest")
        if identity in seen:
            raise ValueError("duplicate native sample")
        seen.add(identity)
        request_name = record.get("raw_request_path")
        if not isinstance(request_name, str) or Path(request_name).name != request_name:
            raise ValueError("invalid raw request locator")
        request_raw = (directory / request_name).read_bytes()
        if hashlib.sha256(request_raw).hexdigest() != record.get("request_sha256"):
            raise ValueError("native request digest mismatch")
        request = json.loads(request_raw)
        expected_seed = config["seed"] + question_order[identity[:2]] * config["k"] + record["sample_index"]
        if (record.get("seed") != expected_seed or not 0 <= expected_seed < 4294967295
                or request != request_body(config, question_by_id[identity[:2]], expected_seed)):
            raise ValueError("sample/request sampling configuration mismatch")
        verify_optional_artifact(directory, record, "raw_response_path", "raw_response_sha256")
        verify_optional_artifact(directory, record, "http_error_path", "http_error_sha256")
        if record.get("status") != "native":
            count["refused"] += 1
            continue
        name = record.get("raw_response_path")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("invalid raw response locator")
        raw = (directory / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record.get("raw_response_sha256"):
            raise ValueError("native response digest mismatch")
        response = json.loads(raw)
        native_fields = decode_response(response)
        if any(record.get(key) != value for key, value in native_fields.items()):
            raise ValueError("sample differs from native response")
        if record.get("served_model") != config["model"]:
            raise ValueError("served model differs from frozen configuration")
        finish = record.get("finish_reason")
        if finish == "length":
            count["truncated"] += 1
            continue
        if finish != "stop":
            count["non_stop_or_unknown"] += 1
            continue
        token_count = record.get("tokens_generated")
        if type(token_count) is not int or token_count < 0:
            raise ValueError("invalid native tokens")
        question = question_by_id[identity[:2]]
        verified_passed = None
        scoring_status = "scored"
        try:
            verified_passed = scorer.score_answer(answer=record.get("answer") or "", expected=question["expected"],
                                                 scoring_method=question["scoring_method"],
                                                 scoring_config=question.get("scoring_config"))
        except scorer.AnswerParseError:
            scoring_status = "answer_parse_failed"
            verified_passed = None
        except scorer.ScoringUnavailableError:
            scoring_status = "unavailable"
            verified_passed = None
        if record.get("scoring_status") != scoring_status:
            raise ValueError("recorded scoring status differs from pinned scorer exception class")
        if verified_passed != record.get("passed") or type(verified_passed) is not type(record.get("passed")):
            raise ValueError("recorded passed differs from pinned scorer recomputation")
        if type(verified_passed) is not bool:
            count["unscoreable"] += 1
            if scoring_status == "answer_parse_failed":
                count["answer_parse_failed"] += 1
            elif scoring_status == "unavailable":
                count["scorer_unavailable"] += 1
            else:
                raise ValueError("non-boolean scorer result lacks an exclusion category")
            continue
        qid = record["question_id"]
        passed = record["passed"]
        count["retained"] += 1
        cells[cell][qid].append((token_count, passed, None))
        reasoning = record.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            count["native_trace_samples"] += 1
            ratio = review_ratio(reasoning)
            if ratio is not None:
                traces[cell][qid].append((ratio, passed, None))
                count["review_ratio_samples"] += 1
        else:
            count["trace_unavailable"] += 1
    rows = []
    for cell, count in sorted(counts.items()):
        questions = cells[cell]
        mixed = within_questions(questions)
        auroc, pairs = stat_within(questions)
        ratio, ratio_pairs = stat_within(traces[cell])
        manifest_qids = {qid for suite, qid in expected_questions if suite == cell[1]}
        rows.append({"role": cell[0], "suite": cell[1], "counts": dict(count),
                     "temperature": config["temperature"], "k": config["k"],
                     "questions": len(questions), "mixed_questions": len(mixed),
                     "manifest_questions": len(manifest_qids),
                     "complete_k_recorded_questions": sum(sum(1 for item in seen if item[:2] == (cell[1], qid)) == config["k"] for qid in manifest_qids),
                     "complete_k_retained_questions": sum(len(questions.get(qid, [])) == config["k"] for qid in manifest_qids),
                     "partial_retained_questions": sum(0 < len(questions.get(qid, [])) < config["k"] for qid in manifest_qids),
                     "zero_retained_questions": sum(not questions.get(qid) for qid in manifest_qids),
                     "all_correct_retained_only_questions": sum(bool(s) and all(x[1] for x in s) for s in questions.values()),
                     "all_wrong_retained_only_questions": sum(bool(s) and all(not x[1] for x in s) for s in questions.values()),
                     "within_pooled_auroc": auroc, "within_pairs": pairs,
                     "within_ci95": boot(stat_within, questions) if len(mixed) >= 10 else None,
                     "review_ratio_within_auroc": ratio, "review_ratio_pairs": ratio_pairs})
    markers = {}
    for name in ("INCOMPLETE.json", "NO_RETAINED_SAMPLES.json"):
        marker_path = directory / name
        if marker_path.exists():
            raw = marker_path.read_bytes()
            markers[name] = {"sha256": digest(raw), "record": json.loads(raw)}
    return {"schema": "epyc.rus3.native-analysis.v1", "protocol_id": "", "run_id": config["run_id"],
            "capture_markers": markers,
            "complete_marker_present": bool(complete_record),
            "complete_marker_matches": complete_marker_matches,
            "owner_evidence_authenticated": owner_evidence_authenticated,
            "owner_evidence_status": "UNATTESTED: structural capture is not a measurement warrant",
            "process_evidence_valid": process_evidence_valid,
            "capture_complete": (complete_marker_matches and process_evidence_valid and not markers
                                 and len(seen) == len(expected_questions) * config["k"]),
            "capture_readiness": "OBSERVED_UNATTESTED" if complete_marker_matches and process_evidence_valid else "INCOMPLETE_OR_UNVERIFIED",
            "safe_cancel_or_drain": "NOT_PROVEN",
            "expected_samples": len(expected_questions) * config["k"],
            "received_samples": len(seen),
            "missing_samples": len(expected_questions) * config["k"] - len(seen),
            "sample_file_sha256": hashlib.sha256((directory / "samples.jsonl").read_bytes()).hexdigest(),
            "metric": "P(tokens correct > tokens incorrect), ties 0.5; neither direction universally better",
            "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scorer-source", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.capture, args.scorer_source)
    with args.out.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
