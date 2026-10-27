#!/usr/bin/env python3
"""V11-FA-4b paired model quality capture. No server or GPU is started here.

The caller owns the granted GPU window and the two private C5 server processes.
This module freezes inputs, captures bounded resumable slices, and analyzes only
complete paired records. It never changes a scorer or supplies a new grade.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Callable
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from answer_scoring import score_response_or_error  # noqa: E402
from architect_bench_analyze import bootstrap_ci, mcnemar_exact  # noqa: E402
from v7_quality_gate_runner import load_questions  # noqa: E402

SCHEMA = "v11-fa4b-quality.v1"
SUITES = ("olympiadbench_hard", "livecodebench")
ARMS = ("ON", "OFF")
KVS = ("q8_0", "f16")
DRAFTS = ("off", "on")
PHASES = ("score", "greedy")
N_SCORED_PER_SUITE = 100
N_GREEDY = 100
N_PREDICT = 512
BAD_LOG = re.compile(r"non-finite target features|sanitized \d+ isolated non-finite|failed to process speculative batch|GGML_ASSERT|terminate called|Aborted", re.I)


def garbage(text: str) -> str:
    """Established DF2-9 client rule, copied from native/df29/df29_client.py:57."""
    if text is None or not text.strip():
        return "empty"
    t = text.strip()
    if t.count("!") >= max(4, 0.5 * len(t)):
        return "bang-run(NaN-logits-signature)"
    if len(set(t)) <= 2 and len(t) >= 8:
        return "single-char-run"
    return ""


def canonical(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def digest(obj: object) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def protocol_requests(seed: int) -> dict:
    return {"score": {"endpoint": "/v1/chat/completions", "max_tokens": N_PREDICT,
                      "temperature": 0.0, "top_k": 1, "seed": seed,
                      "chat_template_kwargs": {"enable_thinking": False}, "stream": False},
            "greedy": {"endpoint": "/completion", "n_predict": N_PREDICT,
                       "temperature": 0.0, "top_k": 1, "top_p": 1.0,
                       "min_p": 0.0, "typical_p": 1.0,
                       "repeat_penalty": 1.0, "cache_prompt": False,
                       "return_tokens": True, "stream": False, "seed": seed}}


def write_new(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(obj))
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"not a JSON object: {path}")
    return result


def validate_declaration(decl: dict) -> None:
    """Require every serving cell to be explicit and pinned before output."""
    if set(decl.get("images", {})) != set(ARMS):
        raise ValueError("exact ON/OFF images required")
    if set(decl.get("models", {})) != {"target", "draft"}:
        raise ValueError("exact target and draft models required")
    if not {"path", "sha256"}.issubset(decl.get("chat_template", {})):
        raise ValueError("chat template identity required")
    if set(decl.get("configs", {})) != {f"{kv}:{dr}" for kv in KVS for dr in DRAFTS}:
        raise ValueError("all four KV/draft configurations required")
    for arm in ARMS:
        image = decl["images"][arm]
        if not {"binary", "hip_library", "binary_sha256", "hip_sha256", "source_commit"}.issubset(image):
            raise ValueError(f"incomplete image {arm}")
        if image["source_commit"] != decl["images"]["ON"]["source_commit"]:
            raise ValueError("source commit differs across arms")
    for name in ("target", "draft"):
        if not {"path", "sha256"}.issubset(decl["models"][name]):
            raise ValueError(f"missing model identity: {name}")
    for key, cfg in decl["configs"].items():
        kv, draft = key.split(":")
        if not {"ON", "OFF"}.issubset(cfg):
            raise ValueError(f"missing launch argv for {key}")
        for arm in ARMS:
            argv = cfg[arm]
            if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
                raise ValueError(f"invalid launch argv {arm}/{key}")
            def val(*flags: str) -> str | None:
                for flag in flags:
                    if flag in argv:
                        i = argv.index(flag)
                        return argv[i + 1] if i + 1 < len(argv) else None
                return None
            if val("-np", "--parallel") != "1" or val("--cache-ram") != "0":
                raise ValueError(f"-np 1 and --cache-ram 0 required: {arm}/{key}")
            if val("-m", "--model") != decl["models"]["target"]["path"]:
                raise ValueError(f"wrong target model: {arm}/{key}")
            if val("--chat-template-file") != decl["chat_template"]["path"]:
                raise ValueError(f"wrong chat template: {arm}/{key}")
            if val("--flash-attn", "-fa") != "on" or val("--device") != "ROCm0":
                raise ValueError(f"FA/ROCm route missing: {arm}/{key}")
            if val("-ctk", "--cache-type-k") != kv or val("-ctv", "--cache-type-v") != kv:
                raise ValueError(f"wrong KV type: {arm}/{key}")
            has_draft = val("-md", "--model-draft") == decl["models"]["draft"]["path"]
            if has_draft != (draft == "on"):
                raise ValueError(f"wrong draft state: {arm}/{key}")
            if draft == "off" and any(x in argv for x in ("-md", "--model-draft")):
                raise ValueError(f"draft flag on OFF config: {arm}/{key}")
            if draft == "on" and (val("--spec-type") != "draft-dflash"
                                  or val("--device-draft") != "ROCm0"
                                  or not val("--spec-draft-n-max") or not val("-ngld")):
                raise ValueError(f"DFlash launch config missing: {arm}/{key}")
        if cfg["ON"][1:] != cfg["OFF"][1:]:
            raise ValueError(f"ON/OFF launch flags differ: {key}")


def freeze(declaration: Path, output: Path, seed: int = 42) -> dict:
    decl = read_json(declaration)
    validate_declaration(decl)
    for arm in ARMS:
        im = decl["images"][arm]
        if file_sha(Path(im["binary"])) != im["binary_sha256"]:
            raise ValueError(f"binary hash drift: {arm}")
        if file_sha(Path(im["hip_library"])) != im["hip_sha256"]:
            raise ValueError(f"HIP hash drift: {arm}")
    for name in ("target", "draft"):
        item = decl["models"][name]
        if file_sha(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"model hash drift: {name}")
    if file_sha(Path(decl["chat_template"]["path"])) != decl["chat_template"]["sha256"]:
        raise ValueError("chat template hash drift")
    questions = []
    for suite in SUITES:
        rows = load_questions(suite, N_SCORED_PER_SUITE, seed, stratify=True)
        if len(rows) != N_SCORED_PER_SUITE:
            raise ValueError(f"{suite}: expected {N_SCORED_PER_SUITE} scored rows, got {len(rows)}")
        for row in rows:
            if not isinstance(row.get("id"), str) or not row.get("prompt") or not row.get("expected"):
                raise ValueError(f"unscoreable row in {suite}")
            if row.get("suite") != suite or not row.get("scoring_method"):
                raise ValueError(f"scoring profile missing in {suite}")
            questions.append(row)
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question ID")
    manifest = {"schema": SCHEMA, "declaration": decl, "seed": seed,
                "scored": questions, "greedy_ids": ids[:N_GREEDY],
                "prompt_sha256": {q["id"]: hashlib.sha256(q["prompt"].encode()).hexdigest() for q in questions},
                "sources": {name: file_sha(HERE / name) for name in (
                    "v11_fa4b_quality.py", "v7_quality_gate_runner.py", "answer_scoring.py",
                    "dataset_adapters.py", "architect_bench_analyze.py", "code_exec_scorer.py")},
                "requests": protocol_requests(seed)}
    manifest["seal_sha256"] = digest(manifest)
    write_new(output, manifest)
    return manifest


def validate_manifest(m: dict) -> None:
    seal = m.get("seal_sha256")
    if not isinstance(seal, str) or seal != digest({k: v for k, v in m.items() if k != "seal_sha256"}):
        raise ValueError("manifest seal mismatch")
    if m.get("schema") != SCHEMA or len(m.get("scored", [])) != 200 or len(m.get("greedy_ids", [])) < 100:
        raise ValueError("manifest count/schema mismatch")
    if type(m.get("seed")) is not int or m.get("requests") != protocol_requests(m["seed"]):
        raise ValueError("request protocol drift")
    validate_declaration(m["declaration"])
    required_sources = {"v11_fa4b_quality.py", "v7_quality_gate_runner.py", "answer_scoring.py",
                        "dataset_adapters.py", "architect_bench_analyze.py", "code_exec_scorer.py"}
    if set(m.get("sources", {})) != required_sources:
        raise ValueError("source set mismatch")
    for name, expected in m["sources"].items():
        if file_sha(HERE / name) != expected:
            raise ValueError(f"source drift: {name}")
    questions = m["scored"]
    if {q["suite"] for q in questions} != set(SUITES):
        raise ValueError("suite mismatch")
    for suite in SUITES:
        if sum(q["suite"] == suite for q in questions) < N_SCORED_PER_SUITE:
            raise ValueError(f"insufficient {suite}")
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids) or len(set(m["greedy_ids"])) != len(m["greedy_ids"]):
        raise ValueError("duplicate IDs")
    if not set(m["greedy_ids"]).issubset(ids):
        raise ValueError("greedy IDs absent from scored corpus")
    for q in questions:
        if hashlib.sha256(q["prompt"].encode()).hexdigest() != m["prompt_sha256"].get(q["id"]):
            raise ValueError("prompt changed")


def exact_tokens(response: dict) -> list[int]:
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("missing or corrupt returned token IDs")
    return tokens


def finite_json(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_json(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_json(v) for v in value)
    return True


def parse_response(phase: str, response: dict, question: dict) -> dict:
    if not finite_json(response):
        raise ValueError("nonfinite API response")
    if phase == "greedy":
        tokens = exact_tokens(response)
        content = response.get("content")
        if not isinstance(content, str) or garbage(content):
            raise ValueError(f"garbage greedy content: {garbage(content)}")
        stop_type = response.get("stop_type")
        truncated = response.get("truncated")
        if truncated is not None and type(truncated) is not bool:
            raise ValueError("invalid greedy truncation field")
        return {"tokens": tokens, "content": content, "correct": None,
                "truncated": truncated is True or stop_type == "limit",
                "termination_unknown": truncated is None or not isinstance(stop_type, str) or not stop_type,
                "finish_reason": None, "stop_type": stop_type if isinstance(stop_type, str) else None}
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("missing scored response choice")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    content = choice.get("message", {}).get("content")
    if isinstance(content, list):  # same content normalization as v7_quality_gate_runner.query_server_meta
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part)
                          for part in content)
    if not isinstance(content, str) or garbage(content):
        raise ValueError(f"garbage scored content: {garbage(content)}")
    correct, cause = score_response_or_error(content, str(question["expected"]), question)
    if cause is not None or type(correct) is not bool:
        raise ValueError(f"scoring undecidable: {cause}")
    return {"tokens": None, "content": content, "correct": correct,
            "truncated": finish_reason == "length", "termination_unknown": finish_reason not in ("stop", "length"),
            "finish_reason": finish_reason if isinstance(finish_reason, str) else None, "stop_type": None}


def request_payload(m: dict, phase: str, question: dict) -> dict:
    params = dict(m["requests"][phase])
    params.pop("endpoint")
    if phase == "greedy":
        return {"prompt": question["prompt"], **params}
    return {"model": "auto", "messages": [{"role": "user", "content": question["prompt"]}], **params}


def capture(m: dict, phase: str, arm: str, kv: str, draft: str, start: int, count: int,
            port: int, out: Path, timeout: int, grant_proof: Path, server_log: Path,
            before_request: Callable[[], None]) -> None:
    validate_manifest(m)
    if (phase, arm, kv, draft) not in {(p, a, k, d) for p in PHASES for a in ARMS for k in KVS for d in DRAFTS}:
        raise ValueError("invalid cell")
    proof = read_json(grant_proof)
    required_proof = {"manifest_sha256": m["seal_sha256"], "arm": arm, "kv": kv,
                      "draft": draft, "port": port, "actual_grant_ok": True,
                      "hip_sha256": m["declaration"]["images"][arm]["hip_sha256"],
                      "server_log": str(server_log)}
    if any(proof.get(k) != v for k, v in required_proof.items()):
        raise ValueError("grant proof does not match cell")
    identity = proof.get("server_identity")
    if (not isinstance(identity, dict) or
            any(type(identity.get(k)) is not int for k in ("pid", "uid", "sid", "pgid", "ticks")) or
            identity["uid"] != os.getuid() or identity["sid"] != identity["pid"] or
            identity["pgid"] != identity["pid"]):
        raise ValueError("grant proof lacks owned server identity")
    if not server_log.is_file():
        raise ValueError("server log window unavailable")
    if not callable(before_request):
        raise ValueError("grant-bound per-request guard required")
    ids = [q["id"] for q in m["scored"]] if phase == "score" else m["greedy_ids"]
    if start < 0 or count < 1 or start + count > len(ids):
        raise ValueError("invalid chunk bounds")
    by_id = {q["id"]: q for q in m["scored"]}
    cell = out / arm / f"{kv}-{draft}" / phase
    cell.mkdir(parents=True, exist_ok=True)
    for i in range(start, start + count):
        q = by_id[ids[i]]
        payload = request_payload(m, phase, q)
        key = f"{i:03d}-{hashlib.sha256(q['id'].encode()).hexdigest()[:16]}"
        path = cell / f"{key}.json"
        failure_path = cell / "failures" / f"{key}.json"
        request_sha = digest(payload)
        if failure_path.exists():
            raise ValueError(f"prior request failure requires a new attempt root: {failure_path}")
        if path.exists():
            row = read_json(path)
            if row.get("request_sha256") != request_sha or row.get("manifest_sha256") != m["seal_sha256"]:
                raise ValueError(f"resume mismatch: {path}")
            validate_record(row, phase, q, m, arm, kv, draft)
            continue
        before_request()  # outside request-failure capture: normal slice stop preserves completed items
        raw = None
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}{m['requests'][phase]['endpoint']}",
                                         data=canonical(payload), headers={"Content-Type": "application/json"},
                                         method="POST")
            log_start = server_log.stat().st_size
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            time.sleep(0.3)  # existing DF2-9 log-flush attribution interval
            with server_log.open("rb") as stream:
                stream.seek(log_start)
                window_bytes = stream.read()
            log_window = window_bytes.decode("utf-8", "replace")
            if BAD_LOG.search(log_window):
                raise ValueError("nonfinite/speculative/server failure in request log window")
            response = json.loads(raw)
            if not isinstance(response, dict):
                raise ValueError("response is not an object")
            parsed = parse_response(phase, response, q)
            row = {"schema": SCHEMA, "manifest_sha256": m["seal_sha256"], "phase": phase,
                   "arm": arm, "kv": kv, "draft": draft, "index": i, "id": q["id"],
                   "prompt_sha256": m["prompt_sha256"][q["id"]], "request_sha256": request_sha,
                   "response_sha256": hashlib.sha256(raw).hexdigest(),
                   "raw_response_text": raw.decode("utf-8"), **parsed,
                   "grant_proof_path": str(grant_proof),
                   "grant_proof_sha256": file_sha(grant_proof),
                   "server_pid": identity["pid"], "server_startticks": identity["ticks"],
                   "server_port": port, "server_log_path": str(server_log)}
            row["server_log_window_sha256"] = hashlib.sha256(window_bytes).hexdigest()
            row["server_log_start"] = log_start
            row["server_log_end"] = log_start + len(window_bytes)
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
            raise ValueError("chunk marker mismatch")
    else:
        write_new(marker_path, marker)


def validate_record(row: dict, phase: str, q: dict, m: dict, arm: str, kv: str, draft: str) -> None:
    for key, value in {"schema": SCHEMA, "manifest_sha256": m["seal_sha256"], "phase": phase,
                       "arm": arm, "kv": kv, "draft": draft, "id": q["id"],
                       "prompt_sha256": m["prompt_sha256"][q["id"]]}.items():
        if row.get(key) != value:
            raise ValueError(f"record {key} mismatch")
    if row.get("seal_sha256") != digest({k: v for k, v in row.items() if k != "seal_sha256"}):
        raise ValueError("record seal mismatch")
    if row.get("request_sha256") != digest(request_payload(m, phase, q)):
        raise ValueError("request payload mismatch")
    if not isinstance(row.get("request_sha256"), str) or not isinstance(row.get("response_sha256"), str) or not isinstance(row.get("server_log_window_sha256"), str):
        raise ValueError("missing request/response hash")
    if not isinstance(row.get("raw_response_text"), str) or hashlib.sha256(row["raw_response_text"].encode()).hexdigest() != row["response_sha256"]:
        raise ValueError("raw response hash mismatch")
    proof_path = Path(row.get("grant_proof_path", ""))
    log_path = Path(row.get("server_log_path", ""))
    if (not proof_path.is_absolute() or not log_path.is_absolute() or
            file_sha(proof_path) != row.get("grant_proof_sha256")):
        raise ValueError("grant proof linkage mismatch")
    proof = read_json(proof_path)
    identity = proof.get("server_identity", {})
    required = {"manifest_sha256": m["seal_sha256"], "arm": arm, "kv": kv,
                "draft": draft, "port": row.get("server_port"), "actual_grant_ok": True,
                "hip_sha256": m["declaration"]["images"][arm]["hip_sha256"],
                "server_log": str(log_path)}
    if (any(proof.get(k) != v for k, v in required.items()) or
            identity.get("pid") != row.get("server_pid") or
            identity.get("ticks") != row.get("server_startticks")):
        raise ValueError("server/grant identity mismatch")
    response = json.loads(row["raw_response_text"])
    if not isinstance(response, dict) or not finite_json(response):
        raise ValueError("raw response invalid")
    if phase == "greedy":
        if (exact_tokens(response) != row.get("tokens") or response.get("content") != row.get("content")
                or (response.get("truncated") is True or response.get("stop_type") == "limit") != row.get("truncated")
                or (response.get("truncated") is None or not isinstance(response.get("stop_type"), str)
                    or not response.get("stop_type")) != row.get("termination_unknown")):
            raise ValueError("greedy row differs from raw response")
    else:
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("raw scored choice missing")
        choice = choices[0]
        content = choice.get("message", {}).get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part)
                              for part in content)
        reason = choice.get("finish_reason")
        if (content != row.get("content") or (reason == "length") != row.get("truncated")
                or (reason not in ("stop", "length")) != row.get("termination_unknown")):
            raise ValueError("scored row differs from raw response")
    if not (type(row.get("server_log_start")) is int and type(row.get("server_log_end")) is int
            and 0 <= row["server_log_start"] <= row["server_log_end"]):
        raise ValueError("invalid server log byte window")
    with log_path.open("rb") as stream:
        stream.seek(row["server_log_start"])
        window = stream.read(row["server_log_end"] - row["server_log_start"])
    if hashlib.sha256(window).hexdigest() != row["server_log_window_sha256"]:
        raise ValueError("server log window linkage mismatch")
    if not isinstance(row.get("content"), str) or garbage(row["content"]):
        raise ValueError("garbage content")
    if phase == "greedy":
        exact_tokens(row)
    elif type(row.get("correct")) is not bool:
        raise ValueError("missing scored boolean")
    if type(row.get("truncated")) is not bool or type(row.get("termination_unknown")) is not bool:
        raise ValueError("missing termination classification")


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def analyze(m: dict, out: Path) -> dict:
    validate_manifest(m)
    by_id = {q["id"]: q for q in m["scored"]}
    report = {"schema": SCHEMA, "manifest_sha256": m["seal_sha256"], "cells": {}}
    checked_logs: set[Path] = set()
    for kv in KVS:
        for draft in DRAFTS:
            cell = f"{kv}:{draft}"
            for arm in ARMS:
                if list((out / arm / f"{kv}-{draft}").glob("server_log_failure-*.json")):
                    raise ValueError(f"{arm}/{cell}: whole-server log failure present")
            scored = {}
            greedy = {}
            for phase in PHASES:
                expected_ids = [q["id"] for q in m["scored"]] if phase == "score" else m["greedy_ids"]
                for arm in ARMS:
                    folder = out / arm / f"{kv}-{draft}" / phase
                    if list((folder / "failures").glob("*.json")):
                        raise ValueError(f"{arm}/{cell}/{phase}: failed request records present")
                    files = sorted(folder.glob("[0-9][0-9][0-9]-*.json"))
                    if len(files) != len(expected_ids):
                        raise ValueError(f"{arm}/{cell}/{phase}: {len(files)}/{len(expected_ids)} records")
                    rows = {}
                    for path in files:
                        row = read_json(path)
                        q = by_id.get(row.get("id"))
                        if q is None or row["id"] in rows:
                            raise ValueError("unknown or duplicate captured ID")
                        validate_record(row, phase, q, m, arm, kv, draft)
                        log_path = Path(row["server_log_path"])
                        if log_path not in checked_logs:
                            health = read_json(log_path.parent / "server-log-health.json")
                            if (health.get("server_log") != str(log_path) or health.get("bad_matches") != [] or
                                    health.get("bytes") != log_path.stat().st_size or
                                    health.get("sha256") != file_sha(log_path)):
                                raise ValueError("whole-server log health mismatch")
                            checked_logs.add(log_path)
                        if row.get("index") != expected_ids.index(row["id"]):
                            raise ValueError("item order/index mismatch")
                        rows[row["id"]] = row
                    if set(rows) != set(expected_ids):
                        raise ValueError("paired ID set mismatch")
                    (scored if phase == "score" else greedy)[arm] = rows
            pairs = [(float(scored["ON"][qid]["correct"]), float(scored["OFF"][qid]["correct"]))
                     for qid in [q["id"] for q in m["scored"]]]
            delta, lo, hi = bootstrap_ci(pairs, iters=10000, seed=m["seed"])
            b = sum(a > c for a, c in pairs)
            c = sum(a < d for a, d in pairs)
            parity = {qid: first_divergence(greedy["ON"][qid]["tokens"], greedy["OFF"][qid]["tokens"])
                      for qid in m["greedy_ids"]}
            termination = {arm: {phase: {"truncated": sum(row["truncated"] for row in (scored if phase == "score" else greedy)[arm].values()),
                                         "unknown": sum(row["termination_unknown"] for row in (scored if phase == "score" else greedy)[arm].values())}
                                  for phase in PHASES} for arm in ARMS}
            if any(not math.isfinite(x) for x in (delta, lo, hi)):
                raise ValueError("nonfinite analysis")
            report["cells"][cell] = {"n_scored": len(pairs), "n_greedy": len(parity),
                                      "accuracy_on": sum(a for a, _ in pairs) / len(pairs),
                                      "accuracy_off": sum(bv for _, bv in pairs) / len(pairs),
                                      "delta_on_minus_off": delta, "ci95": [lo, hi],
                                      "mcnemar": {"on_only": b, "off_only": c,
                                                  "p_two_sided": mcnemar_exact(b, c)},
                                      "token_identical": sum(v is None for v in parity.values()),
                                      "first_divergence": parity,
                                      "termination": termination,
                                      "agreement": lo >= -0.02 and hi <= 0.02}
    report["agreement_all_configs"] = all(c["agreement"] for c in report["cells"].values())
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--declaration", required=True, type=Path)
    f.add_argument("--output", required=True, type=Path)
    f.add_argument("--seed", type=int, default=42)
    a = sub.add_parser("analyze")
    a.add_argument("--manifest", required=True, type=Path)
    a.add_argument("--output-dir", required=True, type=Path)
    a.add_argument("--report", required=True, type=Path)
    args = p.parse_args()
    if args.mode == "freeze":
        freeze(args.declaration, args.output, args.seed)
    else:
        m = read_json(args.manifest)
        write_new(args.report, analyze(m, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
