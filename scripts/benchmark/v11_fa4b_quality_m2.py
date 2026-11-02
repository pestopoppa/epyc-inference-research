#!/usr/bin/env python3
"""FA-4b quality successor: correct the frozen LiveCodeBench scorer profile.

M1 and its native records are immutable.  This module links the completed M1
math captures by file digest and makes all new observations under the M2 seal.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Callable
import urllib.request

try:
    from . import v11_fa4b_quality as m1
except ImportError:  # direct script entrypoint
    import v11_fa4b_quality as m1
from answer_scoring import gold_symbolically_parseable, score_response_or_error
from architect_bench_analyze import bootstrap_ci, mcnemar_exact

ORCH_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
ORCH_BENCH = ORCH_ROOT / "scripts/benchmark"
ORACLE = ORCH_ROOT / "data/benchmark/livecodebench_oracle_manifest.json"
NATIVE_FILES = {
    "seeding_scoring.py": ORCH_BENCH / "seeding_scoring.py",
    "seeding_types.py": ORCH_BENCH / "seeding_types.py",
    "debug_scorer.py": ORCH_BENCH / "debug_scorer.py",
    "livecodebench_oracle.py": ORCH_BENCH / "livecodebench_oracle.py",
    "measurement_guards.py": ORCH_ROOT / "src/autopilot_core/measurement_guards.py",
    "livecodebench_oracle_manifest.json": ORACLE,
}
HERE = Path(__file__).resolve().parent
SCHEMA = "v11-fa4b-quality.m2"
ARMS, KVS, DRAFTS, PHASES = m1.ARMS, m1.KVS, m1.DRAFTS, m1.PHASES
BAD_LOG = m1.BAD_LOG
write_new, read_json, file_sha, digest = m1.write_new, m1.read_json, m1.file_sha, m1.digest
validate_declaration = m1.validate_declaration
garbage, exact_tokens, finite_json = m1.garbage, m1.exact_tokens, m1.finite_json
first_divergence, request_payload = m1.first_divergence, m1.request_payload


def _native():
    # Existing seeding API loads its sibling debug_scorer by absolute file path.
    for path in (ORCH_ROOT, ORCH_BENCH):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import seeding_scoring
    import seeding_types
    import src.autopilot_core.measurement_guards as measurement_guards
    expected = NATIVE_FILES["seeding_scoring.py"].resolve()
    if Path(seeding_scoring.__file__).resolve() != expected:
        raise ValueError("native scorer import path drift")
    if Path(seeding_types.__file__).resolve() != NATIVE_FILES["seeding_types.py"].resolve():
        raise ValueError("native scorer types import path drift")
    if Path(measurement_guards.__file__).resolve() != NATIVE_FILES["measurement_guards.py"].resolve():
        raise ValueError("native measurement guards import path drift")
    scorer = seeding_scoring._load_orchestrator_debug_scorer()
    if Path(scorer.__file__).resolve() != NATIVE_FILES["debug_scorer.py"].resolve():
        raise ValueError("native code scorer import path drift")
    return seeding_scoring


def reference_errors(questions: list[dict], oracle: dict) -> dict[str, str]:
    """Check actual gold/oracle fields for every item before any serving."""
    errors = {}
    accepted = oracle.get("oracles", {})
    if oracle.get("schema") != "livecodebench-oracle-manifest/1":
        raise ValueError("unrecognized validated oracle manifest")
    for q in questions:
        qid = q.get("id")
        if q.get("suite") == "olympiadbench_hard":
            if q.get("scoring_method") != "math_symbolic" or not gold_symbolically_parseable(str(q.get("expected", ""))):
                errors[qid] = "no_reference:math_symbolic"
        elif q.get("suite") == "livecodebench":
            slug = qid.removeprefix("leetcode_") if isinstance(qid, str) else ""
            record = accepted.get(slug, {})
            cfg = q.get("scoring_config", {})
            cases = cfg.get("entry_point_cases") if isinstance(cfg, dict) else None
            if (q.get("scoring_method") != "code_execution" or not isinstance(cases, list)
                    or len(cases) < 2 or not cfg.get("entry_point")
                    or q.get("expected") != cfg.get("entry_point")
                    or record.get("scoring_config") != cfg or record.get("n_cases") != len(cases)
                    or any(not isinstance(case, dict) or not isinstance(case.get("args"), list)
                           or "expected" not in case for case in cases)):
                errors[qid] = "no_reference:entry_point_cases"
        else:
            errors[qid] = "unsupported:suite"
    return errors


def _source_hashes() -> dict[str, str]:
    return {"v11_fa4b_quality_m2.py": file_sha(HERE / "v11_fa4b_quality_m2.py"),
            **{name: file_sha(path) for name, path in NATIVE_FILES.items()}}


def freeze_successor(old_manifest_path: Path, old_root: Path, output: Path) -> dict:
    old = read_json(old_manifest_path)
    m1.validate_manifest(old)
    oracle = read_json(ORACLE)
    errors = reference_errors(old["scored"], oracle)
    if errors:
        raise ValueError(f"corpus reference prevalidation failed: {errors}")
    _native()
    links = {}
    # The only completed native observations in M1 are these 100 math items.
    for i, q in enumerate(old["scored"][:100]):
        if q["suite"] != "olympiadbench_hard":
            raise ValueError("historical math prefix changed")
        key = f"{i:03d}-{hashlib.sha256(q['id'].encode()).hexdigest()[:16]}.json"
        path = old_root / "ON/q8_0-off/score" / key
        row = read_json(path)
        m1.validate_record(row, "score", q, old, "ON", "q8_0", "off")
        if row["index"] != i or row["correct"] is None:
            raise ValueError("historical capture index or verdict invalid")
        links[q["id"]] = {"path": str(path), "sha256": file_sha(path),
                          "request_sha256": row["request_sha256"],
                          "response_sha256": row["response_sha256"]}
    result = {k: old[k] for k in ("declaration", "seed", "scored", "greedy_ids", "prompt_sha256", "requests")}
    result.update(schema=SCHEMA, scorer_profile={"olympiadbench_hard": "research.answer_scoring.score_response_or_error",
                                                  "livecodebench": "orchestrator.seeding_scoring.score_answer_or_error"},
                  sources=_source_hashes(), predecessor={"path": str(old_manifest_path),
                  "file_sha256": file_sha(old_manifest_path), "seal_sha256": old["seal_sha256"],
                  "records_root": str(old_root), "links": links})
    result["seal_sha256"] = digest(result)
    write_new(output, result)
    return result


def validate_manifest(m: dict) -> None:
    if m.get("schema") != SCHEMA or m.get("seal_sha256") != digest({k: v for k, v in m.items() if k != "seal_sha256"}):
        raise ValueError("M2 manifest seal/schema mismatch")
    if m.get("sources") != _source_hashes():
        raise ValueError("M2 source or native scorer drift")
    predecessor = m.get("predecessor", {})
    old_path = Path(predecessor.get("path", ""))
    if file_sha(old_path) != predecessor.get("file_sha256"):
        raise ValueError("M1 manifest file digest drift")
    old = read_json(old_path)
    m1.validate_manifest(old)
    if old["seal_sha256"] != predecessor.get("seal_sha256"):
        raise ValueError("M1 internal seal drift")
    for key in ("declaration", "seed", "scored", "greedy_ids", "prompt_sha256", "requests"):
        if m.get(key) != old[key]:
            raise ValueError(f"M1/M2 {key} not identical")
    if m.get("scorer_profile") != {"olympiadbench_hard": "research.answer_scoring.score_response_or_error",
                                   "livecodebench": "orchestrator.seeding_scoring.score_answer_or_error"}:
        raise ValueError("scorer profile drift")
    errors = reference_errors(m["scored"], read_json(ORACLE))
    if errors:
        raise ValueError(f"corpus reference prevalidation failed: {errors}")
    if len(predecessor.get("links", {})) != 100:
        raise ValueError("historical capture link count changed")
    _native()
    for i, q in enumerate(m["scored"][:100]):
        link = predecessor["links"].get(q["id"], {})
        path = Path(link.get("path", ""))
        if path.parent != Path(predecessor["records_root"]) / "ON/q8_0-off/score" or file_sha(path) != link.get("sha256"):
            raise ValueError("historical record file drift")
        row = read_json(path)
        m1.validate_record(row, "score", q, old, "ON", "q8_0", "off")
        if row["index"] != i or row["request_sha256"] != digest(request_payload(m, "score", q)) or row["response_sha256"] != link["response_sha256"]:
            raise ValueError("historical request/response linkage drift")


def historical_row(m: dict, phase: str, arm: str, kv: str, draft: str, q: dict) -> dict | None:
    if (phase, arm, kv, draft) != ("score", "ON", "q8_0", "off"):
        return None
    link = m["predecessor"]["links"].get(q["id"])
    if link is None:
        return None
    path = Path(link["path"])
    if file_sha(path) != link["sha256"]:
        raise ValueError("historical record changed")
    old = read_json(Path(m["predecessor"]["path"]))
    row = read_json(path)
    m1.validate_record(row, phase, q, old, arm, kv, draft)
    if row["request_sha256"] != digest(request_payload(m, phase, q)):
        raise ValueError("historical request differs in M2")
    return row


def parse_response(phase: str, response: dict, question: dict, *, score: bool = True) -> dict:
    if phase == "greedy":
        return {**m1.parse_response(phase, response, question), "undecidable_reason": None}
    if not finite_json(response):
        raise ValueError("nonfinite API response")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("missing scored response choice")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    content = choice.get("message", {}).get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    if not isinstance(content, str) or garbage(content):
        raise ValueError(f"garbage scored content: {garbage(content)}")
    if not score:
        correct, cause = None, None
    elif question["suite"] == "livecodebench":
        api = _native()
        correct, cause = api.score_answer_or_error(content, str(question["expected"]),
                                                     question["scoring_method"], question["scoring_config"])
    else:
        correct, cause = score_response_or_error(content, str(question["expected"]), question)
    if score and cause is None and type(correct) is not bool:
        raise ValueError("scorer returned malformed verdict")
    if cause is not None and correct is not None:
        raise ValueError("scorer returned contradictory verdict")
    return {"tokens": None, "content": content, "correct": correct,
            "undecidable_reason": cause, "truncated": finish_reason == "length",
            "termination_unknown": finish_reason not in ("stop", "length"),
            "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
            "stop_type": None}


def validate_record(row: dict, phase: str, q: dict, m: dict, arm: str, kv: str, draft: str) -> None:
    for key, value in {"schema": SCHEMA, "manifest_sha256": m["seal_sha256"], "phase": phase,
                       "arm": arm, "kv": kv, "draft": draft, "id": q["id"],
                       "prompt_sha256": m["prompt_sha256"][q["id"]]}.items():
        if row.get(key) != value:
            raise ValueError(f"M2 record {key} mismatch")
    if row.get("seal_sha256") != digest({k: v for k, v in row.items() if k != "seal_sha256"}):
        raise ValueError("M2 record seal mismatch")
    if row.get("request_sha256") != digest(request_payload(m, phase, q)):
        raise ValueError("M2 request payload mismatch")
    raw = row.get("raw_response_text")
    if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != row.get("response_sha256"):
        raise ValueError("M2 raw response hash mismatch")
    proof_path = Path(row.get("grant_proof_path", ""))
    log_path = Path(row.get("server_log_path", ""))
    if not proof_path.is_absolute() or not log_path.is_absolute() or file_sha(proof_path) != row.get("grant_proof_sha256"):
        raise ValueError("M2 grant proof linkage mismatch")
    proof = read_json(proof_path)
    ident = proof.get("server_identity", {})
    if (any(proof.get(k) != v for k, v in {
            "manifest_sha256": m["seal_sha256"], "arm": arm, "kv": kv, "draft": draft,
            "port": row.get("server_port"), "actual_grant_ok": True,
            "hip_sha256": m["declaration"]["images"][arm]["hip_sha256"],
            "server_log": str(log_path)}.items()) or
            ident.get("pid") != row.get("server_pid") or ident.get("ticks") != row.get("server_startticks")):
        raise ValueError("M2 server/grant identity mismatch")
    response = json.loads(raw)
    if not isinstance(response, dict) or not finite_json(response):
        raise ValueError("M2 raw response invalid")
    # Validate response shape without executing generated code a second time.
    # The scorer source and full response are pinned to the captured verdict.
    parsed = parse_response(phase, response, q, score=False)
    comparable = ("tokens", "content", "truncated", "termination_unknown", "finish_reason", "stop_type")
    if any(row.get(k) != parsed[k] for k in comparable):
        raise ValueError("M2 record differs from raw response")
    if phase == "score":
        if row.get("correct") is not None and type(row.get("correct")) is not bool:
            raise ValueError("invalid M2 scored verdict")
        if (row.get("correct") is None) != (isinstance(row.get("undecidable_reason"), str)
                                             and bool(row.get("undecidable_reason"))):
            raise ValueError("invalid M2 undecidable classification")
    elif row.get("undecidable_reason") is not None or row.get("correct") is not None:
        raise ValueError("greedy row has score")
    if not (type(row.get("server_log_start")) is int and type(row.get("server_log_end")) is int
            and 0 <= row["server_log_start"] <= row["server_log_end"]):
        raise ValueError("invalid M2 log byte window")
    with log_path.open("rb") as stream:
        stream.seek(row["server_log_start"])
        window = stream.read(row["server_log_end"] - row["server_log_start"])
    if hashlib.sha256(window).hexdigest() != row.get("server_log_window_sha256"):
        raise ValueError("M2 log window linkage mismatch")


def capture(m: dict, phase: str, arm: str, kv: str, draft: str, start: int, count: int,
            port: int, out: Path, timeout: int, grant_proof: Path, server_log: Path,
            before_request: Callable[[], None]) -> None:
    validate_manifest(m)
    if (phase, arm, kv, draft) not in {(p, a, k, d) for p in PHASES for a in ARMS for k in KVS for d in DRAFTS}:
        raise ValueError("invalid M2 cell")
    proof = read_json(grant_proof)
    identity = proof.get("server_identity")
    if (any(proof.get(k) != v for k, v in {
            "manifest_sha256": m["seal_sha256"], "arm": arm, "kv": kv, "draft": draft,
            "port": port, "actual_grant_ok": True,
            "hip_sha256": m["declaration"]["images"][arm]["hip_sha256"],
            "server_log": str(server_log)}.items()) or not isinstance(identity, dict) or
            any(type(identity.get(k)) is not int for k in ("pid", "uid", "sid", "pgid", "ticks")) or
            identity["uid"] != os.getuid() or identity["sid"] != identity["pid"] or
            identity["pgid"] != identity["pid"] or not server_log.is_file() or not callable(before_request)):
        raise ValueError("M2 grant proof or server custody invalid")
    ids = [q["id"] for q in m["scored"]] if phase == "score" else m["greedy_ids"]
    if start < 0 or count < 1 or start + count > len(ids):
        raise ValueError("invalid M2 chunk bounds")
    by_id = {q["id"]: q for q in m["scored"]}
    cell = out / arm / f"{kv}-{draft}" / phase
    cell.mkdir(parents=True, exist_ok=True)
    for i in range(start, start + count):
        q = by_id[ids[i]]
        if historical_row(m, phase, arm, kv, draft, q) is not None:
            continue
        payload = request_payload(m, phase, q)
        key = f"{i:03d}-{hashlib.sha256(q['id'].encode()).hexdigest()[:16]}"
        path = cell / f"{key}.json"
        failure_path = cell / "failures" / f"{key}.json"
        request_sha = digest(payload)
        if failure_path.exists():
            raise ValueError(f"prior M2 request failure requires new attempt root: {failure_path}")
        if path.exists():
            row = read_json(path)
            validate_record(row, phase, q, m, arm, kv, draft)
            continue
        before_request()
        raw = None
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}{m['requests'][phase]['endpoint']}",
                data=m1.canonical(payload), headers={"Content-Type": "application/json"}, method="POST")
            log_start = server_log.stat().st_size
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            time.sleep(.3)
            with server_log.open("rb") as stream:
                stream.seek(log_start)
                window = stream.read()
            if BAD_LOG.search(window.decode("utf-8", "replace")):
                raise ValueError("nonfinite/speculative/server failure in M2 request log window")
            response = json.loads(raw)
            if not isinstance(response, dict):
                raise ValueError("M2 response is not an object")
            parsed = parse_response(phase, response, q)
            row = {"schema": SCHEMA, "manifest_sha256": m["seal_sha256"], "phase": phase,
                   "arm": arm, "kv": kv, "draft": draft, "index": i, "id": q["id"],
                   "prompt_sha256": m["prompt_sha256"][q["id"]], "request_sha256": request_sha,
                   "response_sha256": hashlib.sha256(raw).hexdigest(),
                   "raw_response_text": raw.decode("utf-8"), **parsed,
                   "grant_proof_path": str(grant_proof), "grant_proof_sha256": file_sha(grant_proof),
                   "server_pid": identity["pid"], "server_startticks": identity["ticks"],
                   "server_port": port, "server_log_path": str(server_log),
                   "server_log_start": log_start, "server_log_end": log_start + len(window),
                   "server_log_window_sha256": hashlib.sha256(window).hexdigest()}
            row["seal_sha256"] = digest(row)
            validate_record(row, phase, q, m, arm, kv, draft)
            write_new(path, row)
        except Exception as exc:
            write_new(failure_path, {"schema": SCHEMA, "manifest_sha256": m["seal_sha256"],
                                     "phase": phase, "arm": arm, "kv": kv, "draft": draft,
                                     "index": i, "id": q["id"], "request_sha256": request_sha,
                                     "response_sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
                                     "error_type": type(exc).__name__, "error": str(exc)[:500]})
            raise
    marker = {"manifest_sha256": m["seal_sha256"], "phase": phase, "arm": arm,
              "kv": kv, "draft": draft, "start": start, "end": start + count}
    marker_path = cell / f"chunk-{start:03d}-{start + count:03d}.complete.json"
    if marker_path.exists():
        if read_json(marker_path) != marker:
            raise ValueError("M2 chunk marker mismatch")
    else:
        write_new(marker_path, marker)


def paired_quality(scored: dict, ids: list[str], seed: int) -> dict:
    undecidable = {arm: {qid: row["undecidable_reason"] for qid, row in scored[arm].items()
                         if row.get("undecidable_reason") is not None}
                   for arm in ARMS}
    paired_ids = [qid for qid in ids if all(type(scored[arm][qid]["correct"]) is bool for arm in ARMS)]
    result = {"n_scored_observed_per_arm": {arm: len(scored[arm]) for arm in ARMS},
              "n_decidable_pairs": len(paired_ids), "undecidable": undecidable}
    if len(ids) != 200 or len(paired_ids) < 200:
        result.update(agreement=None, status="INCONCLUSIVE_INSUFFICIENT_DECIDABLE_PAIRS")
        return result
    pairs = [(float(scored["ON"][qid]["correct"]), float(scored["OFF"][qid]["correct"])) for qid in ids]
    delta, lo, hi = bootstrap_ci(pairs, iters=10000, seed=seed)
    on_only = sum(a > b for a, b in pairs)
    off_only = sum(a < b for a, b in pairs)
    if any(not math.isfinite(x) for x in (delta, lo, hi)):
        raise ValueError("nonfinite M2 analysis")
    result.update(accuracy_on=sum(a for a, _ in pairs) / 200,
                  accuracy_off=sum(b for _, b in pairs) / 200,
                  delta_on_minus_off=delta, ci95=[lo, hi],
                  mcnemar={"on_only": on_only, "off_only": off_only,
                           "p_two_sided": mcnemar_exact(on_only, off_only)},
                  agreement=(lo >= -.02 and hi <= .02), status="DECIDABLE")
    return result


def analyze(m: dict, out: Path) -> dict:
    validate_manifest(m)
    by_id = {q["id"]: q for q in m["scored"]}
    report = {"schema": SCHEMA, "manifest_sha256": m["seal_sha256"],
              "predecessor_manifest_file_sha256": m["predecessor"]["file_sha256"],
              "predecessor_manifest_internal_seal": m["predecessor"]["seal_sha256"],
              "cells": {}}
    checked_logs: set[Path] = set()
    for kv in KVS:
        for draft in DRAFTS:
            cell = f"{kv}:{draft}"
            for arm in ARMS:
                if list((out / arm / f"{kv}-{draft}").glob("server_log_failure-*.json")):
                    raise ValueError(f"{arm}/{cell}: whole-server log failure present")
            scored, greedy = {}, {}
            for phase in PHASES:
                ids = [q["id"] for q in m["scored"]] if phase == "score" else m["greedy_ids"]
                for arm in ARMS:
                    folder = out / arm / f"{kv}-{draft}" / phase
                    if list((folder / "failures").glob("*.json")):
                        raise ValueError(f"{arm}/{cell}/{phase}: failed request records present")
                    files = sorted(folder.glob("[0-9][0-9][0-9]-*.json"))
                    rows = {}
                    for path in files:
                        row = read_json(path)
                        q = by_id.get(row.get("id"))
                        if q is None or row["id"] in rows:
                            raise ValueError("unknown or duplicate M2 captured ID")
                        validate_record(row, phase, q, m, arm, kv, draft)
                        if row.get("index") != ids.index(row["id"]):
                            raise ValueError("M2 item order/index mismatch")
                        rows[row["id"]] = row
                    for qid in ids:
                        q = by_id[qid]
                        historical = historical_row(m, phase, arm, kv, draft, q)
                        if historical is not None:
                            if qid in rows:
                                raise ValueError("M1/M2 duplicate observation")
                            rows[qid] = historical
                    if set(rows) != set(ids):
                        raise ValueError(f"{arm}/{cell}/{phase}: {len(rows)}/{len(ids)} records")
                    for row in rows.values():
                        log_path = Path(row["server_log_path"])
                        if log_path in checked_logs:
                            continue
                        health = read_json(log_path.parent / "server-log-health.json")
                        if (health.get("server_log") != str(log_path) or health.get("bad_matches") != [] or
                                health.get("bytes") != log_path.stat().st_size or
                                health.get("sha256") != file_sha(log_path)):
                            raise ValueError("whole-server log health mismatch")
                        checked_logs.add(log_path)
                    (scored if phase == "score" else greedy)[arm] = rows
            ids = [q["id"] for q in m["scored"]]
            parity = {qid: first_divergence(greedy["ON"][qid]["tokens"], greedy["OFF"][qid]["tokens"])
                      for qid in m["greedy_ids"]}
            termination = {arm: {phase: {
                "truncated": sum(row["truncated"] for row in (scored if phase == "score" else greedy)[arm].values()),
                "unknown": sum(row["termination_unknown"] for row in (scored if phase == "score" else greedy)[arm].values())}
                for phase in PHASES} for arm in ARMS}
            result = paired_quality(scored, ids, m["seed"])
            result.update(n_greedy=len(parity), token_identical=sum(v is None for v in parity.values()),
                          first_divergence=parity, termination=termination,
                          historical_M1_links=len(m["predecessor"]["links"]) if (kv, draft) == ("q8_0", "off") else 0)
            report["cells"][cell] = result
    report["agreement_all_configs"] = (all(c["status"] == "DECIDABLE" for c in report["cells"].values())
                                       and all(c["agreement"] for c in report["cells"].values()))
    report["status"] = ("DECIDABLE" if all(c["status"] == "DECIDABLE" for c in report["cells"].values())
                        else "INCONCLUSIVE_INSUFFICIENT_DECIDABLE_PAIRS")
    return report


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--old-manifest", required=True, type=Path)
    freeze.add_argument("--old-output-dir", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--manifest", required=True, type=Path)
    analysis.add_argument("--output-dir", required=True, type=Path)
    analysis.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze_successor(args.old_manifest, args.old_output_dir, args.output)
    else:
        write_new(args.report, analyze(read_json(args.manifest), args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
