#!/usr/bin/env python3
"""V7 quality-gate runner: evaluate MMLU-Pro + GPQA-Diamond on a kernel.

Samples questions from registered suites, queries a llama-server instance,
scores multiple-choice answers, and writes per-suite accuracy JSON ready for
v7_quality_gate_compare.py.

Usage:
    v7_quality_gate_runner.py --port 18072 --output results.json \
        --suites mmlu_pro gpqa --n 200 --seed 42 --endpoint chat

Output JSON shape:
    {
      "meta": {"kernel": "v7-experimental", "binary": "...", "models": "...", "timestamp": "..."},
      "suites": [
        {"suite": "mmlu_pro", "accuracy": 0.82, "n": 200, "correct": 164,
         "per_tier": {"1": {"accuracy": 0.85, "n": 50, "correct": 42}, ...}},
        {"suite": "gpqa", "accuracy": 0.63, "n": 100, "correct": 63, ...}
      ]
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REQUEST_TIMEOUT_S = int(os.environ.get("RUNNER_REQUEST_TIMEOUT_S", "1800"))
# v3 rows are an active historical snapshot which did not retain the prompt.
# Never reinterpret those rows as resumable evidence: v4 is the first schema
# with enough retained material to independently verify every consumed input.
CAPTURE_SCHEMA_VERSION = "v7_quality_gate_capture.v4"
SWE_CAPTURE_STATES = (
    "strict_ready",
    "prompt_contract_candidate",
    "model_truncation_no_patch",
    "model_truncation_partial_patch",
    "request_error",
)
RUNNER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# Detection only. Conversion remains owned by convert_sr_to_patch.py, and this
# runner never turns a model response into a patch or a test verdict.
SEARCH_REPLACE_BLOCK = re.compile(
    r"<<<<<<<+\s*SEARCH\s*\n(.*?)\n?=======\s*\n(.*?)\n?"
    r">>>>>>>+\s*REPLACE\s*(\S*)",
    re.DOTALL,
)


def text_fingerprint(text: str) -> dict[str, int | str]:
    """Return stable UTF-8 size and identity evidence for an unmodified string."""
    encoded = text.encode("utf-8")
    return {
        "chars": len(text),
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def swe_search_replace_diagnostics(
    response: str,
    finish_reason: str = "",
    request_error: str = "",
) -> dict[str, object]:
    """Describe strict SEARCH/REPLACE structure without accepting or applying it."""
    markers = {
        "search": response.count("<<<<<<<"),
        "divider": response.count("======="),
        "replace": response.count(">>>>>>>"),
    }
    parseable_blocks = len(SEARCH_REPLACE_BLOCK.findall(response))
    marker_blocks = markers["search"]
    has_markers = any(markers.values())
    malformed = has_markers and (
        markers["search"] != markers["divider"]
        or markers["search"] != markers["replace"]
        or parseable_blocks != marker_blocks
    )
    if request_error or finish_reason == "request_error":
        state = "request_error"
    elif finish_reason == "length" and parseable_blocks == 0:
        state = "model_truncation_no_patch"
    elif finish_reason == "length":
        state = "model_truncation_partial_patch"
    elif parseable_blocks > 0 and not malformed:
        state = "strict_ready"
    else:
        state = "prompt_contract_candidate"
    return {
        "marker_counts": markers,
        "parseable_block_count": parseable_blocks,
        "has_markers": has_markers,
        "malformed_contract": malformed,
        "state": state,
        "converter_ready": state == "strict_ready",
        # A request error is not evidence of a model failure. It must hold the
        # captured score provisional until the missing draw is recovered.
        "score_provisional": state in {"prompt_contract_candidate", "request_error"},
    }


def runner_source_sha256() -> str:
    return RUNNER_SOURCE_SHA256


def valid_resume_row(row: dict, suite_name: str, question: dict) -> bool:
    """Accept only current, losslessly verifiable captures for resume."""
    if row.get("suite") != suite_name:
        return False
    if row.get("capture_schema_version") != CAPTURE_SCHEMA_VERSION:
        return False
    if row.get("runner_source_sha256") != RUNNER_SOURCE_SHA256:
        return False
    if row.get("request_error") or row.get("finish_reason") == "request_error":
        return False
    if row.get("prompt") != question.get("prompt"):
        return False
    if row.get("expected") != str(question.get("expected", "")).strip():
        return False
    for field in ("prompt", "response", "reasoning"):
        if not isinstance(row.get(field), str):
            return False
    return (
        row.get("prompt_fingerprint") == text_fingerprint(row["prompt"])
        and row.get("response_fingerprint") == text_fingerprint(row["response"])
        and row.get("reasoning_fingerprint") == text_fingerprint(row["reasoning"])
    )


def replace_capture_contents(handle, lines: list[str]) -> None:
    """Atomically replace an append-open capture and rebind its descriptor.

    ``O_APPEND`` makes seek/truncate compaction unsafe.  Write and fsync a
    sibling temporary file, atomically replace the path, then dup an append FD
    for the replacement inode over the caller's existing handle descriptor.
    """
    path = Path(handle.name)
    payload = "\n".join(lines) + ("\n" if lines else "")
    handle.flush()
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
        replacement_fd = os.open(path, os.O_WRONLY | os.O_APPEND)
        try:
            os.dup2(replacement_fd, handle.fileno())
        finally:
            os.close(replacement_fd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_live_capture_status(
    path: Path,
    *,
    suite_name: str,
    arm: str,
    completed_draws: int,
    expected_draws: int,
    capture: dict,
) -> None:
    """Atomically publish the current capture state for a live monitor.

    The sidecar intentionally describes capture health, not a benchmark
    verdict. A length-capped model response is visible but not an artifact
    failure; missing provenance or transport loss is fail-closed.
    """
    swe_capture = capture["swebench_search_replace"]
    states = swe_capture["state_counts"]
    integrity_failure = (
        swe_capture["resumed_rows_without_diagnostics"] > 0
        or swe_capture["resumed_rows_without_provenance"] > 0
        or swe_capture["resumed_rows_source_sha_mismatch"] > 0
        or states["request_error"] > 0
    )
    provisional = (
        integrity_failure
        or states["prompt_contract_candidate"] > 0
        or not swe_capture["summary_complete"]
    )
    if not swe_capture["summary_complete"]:
        live_score_status = "incomplete_capture_diagnostics"
    elif states["request_error"]:
        live_score_status = "provisional_request_error"
    elif states["prompt_contract_candidate"]:
        live_score_status = "provisional_prompt_contract"
    else:
        live_score_status = "terminal_no_prompt_contract_candidate"
    status = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "runner_source_sha256": RUNNER_SOURCE_SHA256,
        "suite": suite_name,
        "arm": arm,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_draws": completed_draws,
        "expected_draws": expected_draws,
        "complete": completed_draws >= expected_draws,
        "request_error_rows": capture["request_error_rows"],
        "length_cap_rows": capture["length_cap_rows"],
        "swebench_search_replace": {
            "applicable": swe_capture["applicable"],
            "state_counts": states,
            "zero_strict_block_rows": swe_capture["zero_strict_block_rows"],
            "partial_strict_block_rows": swe_capture["partial_strict_block_rows"],
            "summary_complete": swe_capture["summary_complete"],
            "score_status": live_score_status,
        },
        "provisional": provisional,
        "artifact_integrity_fail_closed": integrity_failure,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(temporary, path)


def wait_for_server(url: str, timeout: int = 120) -> None:
    """Wait for llama-server /health to return ok."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode().strip().lower()
                if "ok" in body:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Server at {url} did not become healthy within {timeout}s")


def query_server(*args, **kwargs) -> str:
    """Backwards-compatible wrapper: response text only."""
    return query_server_meta(*args, **kwargs)["text"]


def query_server_meta(
    url: str,
    prompt: str,
    max_tokens: int = 64,
    temperature: float = 0.0,
    seed: int = 42,
    endpoint: str = "chat",
    top_p: float | None = None,
    top_k: int | None = None,
    enable_thinking: bool | None = None,
) -> dict:
    """Query llama-server; return text plus why generation stopped.

    finish_reason matters: a response cut off at max_tokens scores wrong for a
    budget reason, not a reasoning reason, and must be counted separately.
    """
    if endpoint == "chat":
        payload = {
            "model": "auto",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if enable_thinking is not None:
            # enable_thinking is only honoured on the /v1/chat/completions path.
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        request_path = "/v1/chat/completions"
        effective = {"endpoint": "chat", "request_path": request_path,
                     "temperature": temperature, "top_p": top_p, "top_k": top_k,
                     "enable_thinking": enable_thinking}
    elif endpoint == "completion":
        payload = {
            "model": "",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "top_k": 1,
            "logprobs": 0,
        }
        request_path = "/v1/completions"
        # The completion path pins top_k=1: this draw is GREEDY, and top_p /
        # enable_thinking are never sent. Recording the requested values here
        # would attest to sampling that did not happen.
        effective = {"endpoint": "completion", "request_path": request_path,
                     "temperature": temperature, "top_p": None, "top_k": 1,
                     "enable_thinking": None, "greedy": True}
    else:
        raise ValueError(f"unsupported endpoint: {endpoint}")

    req = urllib.request.Request(
        f"{url}{request_path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            result = json.loads(resp.read().decode())
            choices = result.get("choices", [])
            if not choices:
                return {"text": "", "reasoning": "", "finish_reason": "no_choices",
                        "completion_tokens": 0, "error": "", "effective": effective}
            choice = choices[0]
            if endpoint == "chat":
                message = choice.get("message", {})
                content = message.get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                text = str(content or "")
            else:
                text = choice.get("text", "")
            reasoning = ""
            if endpoint == "chat":
                reasoning = str(message.get("reasoning_content") or "")
            usage = result.get("usage", {}) or {}
            timings = result.get("timings", {}) or {}
            return {
                "text": text,
                "reasoning": reasoning,
                "finish_reason": choice.get("finish_reason", ""),
                "completion_tokens": usage.get("completion_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                # server-reported per-request rates (single-request view)
                "decode_tok_s": timings.get("predicted_per_second", 0.0),
                "prompt_tok_s": timings.get("prompt_per_second", 0.0),
                "error": "",
                "effective": effective,
            }
    except Exception as e:
        print(f"  [runner] query failed: {e}", file=sys.stderr)
        return {"text": "", "reasoning": "", "finish_reason": "request_error",
                "completion_tokens": 0, "error": str(e)[:300],
                "effective": locals().get("effective")}


# Canonical scoring primitives live in answer_scoring (single source; see
# handoffs/active/scoring-infra-standardization.md). Re-exported here so this
# module's public scoring API is unchanged for existing importers.
from answer_scoring import (  # noqa: F401
    extract_letter_answer, _normalize_numeric, parse_math_number,
    _latex_to_sympy_str, _sympy_expr, _split_top,
    _canon_elem, _is_set_answer, score_math_symbolic,
    gold_symbolically_parseable, extract_boxed, score_math_numeric,
    _first_pattern_match, extract_exact_answer, score_response,
)

def load_questions(
    suite_name: str,
    n: int,
    seed: int,
    stratify: bool = False,
    questions_in: Path | None = None,
    limit: int = 0,
) -> list:
    """Sample a suite's questions, or replay a previously pinned item set.

    Pinning matters for the architect bench: arms are compared paired, and
    the CPU arm runs in a later session. Re-sampling there would silently
    change the item set and break the pairing, so the first arm writes the
    manifest and every later arm replays it verbatim.
    """
    if questions_in is not None:
        pinned = json.loads(Path(questions_in).read_text())
        items = pinned["suites"][suite_name] if "suites" in pinned else pinned
        return items[:limit] if limit else items

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from dataset_adapters import get_adapter
    except ImportError:
        sys.path.insert(
            0,
            str(Path(__file__).resolve().parent.parent / "benchmark"),
        )
        from dataset_adapter_modules.registry import get_adapter

    adapter = get_adapter(suite_name)
    if adapter is None:
        return []
    return adapter.sample(n=n, seed=seed, stratify=stratify)


def run_suite(
    suite_name: str,
    url: str,
    n: int,
    seed: int,
    stratify: bool = False,
    max_tokens: int = 64,
    endpoint: str = "chat",
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    enable_thinking: bool | None = None,
    repeats: int = 1,
    per_question_out=None,
    questions_in: Path | None = None,
    limit: int = 0,
    arm: str = "",
    concurrency: int = 1,
    live_status_out: Path | None = None,
) -> dict:
    """Run eval on a single suite and return per-suite results."""
    questions = load_questions(suite_name, n, seed, stratify, questions_in, limit)
    if not questions:
        return {"suite": suite_name, "accuracy": 0, "n": 0, "correct": 0,
                "error": "no questions sampled"}

    correct = 0
    total = 0
    errors = 0
    truncated = 0
    per_tier: dict[str, dict] = {}
    per_item: dict[str, dict] = {}
    tok_acc = [0, 0]  # [completion, prompt] tokens generated THIS run (excludes resumed)
    capture = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "new_rows": 0,
        "request_error_rows": 0,
        "length_cap_rows": 0,
        "response_chars": 0,
        "response_utf8_bytes": 0,
        "reasoning_chars": 0,
        "reasoning_utf8_bytes": 0,
        "swebench_search_replace": {
            "applicable": suite_name == "swebench_oracle",
            "new_rows": 0,
            "resumed_rows": 0,
            "resumed_rows_without_diagnostics": 0,
            "resumed_rows_without_provenance": 0,
            "resumed_rows_source_sha_mismatch": 0,
            "rows_with_markers": 0,
            "parseable_blocks": 0,
            "malformed_rows": 0,
            "zero_strict_block_rows": 0,
            "partial_strict_block_rows": 0,
            "state_counts": {state: 0 for state in SWE_CAPTURE_STATES},
            "summary_complete": True,
            "converter_contract_ready": False,
            "score_status": "not_applicable",
        },
    }
    suite_t0 = time.monotonic()
    expected_draws = sum(
        1 for question in questions if str(question.get("expected", "")).strip()
    ) * repeats

    def fold_transport_and_length(finish_reason: str, request_error: str) -> None:
        if finish_reason == "length":
            capture["length_cap_rows"] += 1
        if finish_reason == "request_error" or request_error:
            capture["request_error_rows"] += 1

    def fold_swe_diagnostics(diagnostic: dict[str, object] | None, *, resumed: bool) -> None:
        """Fold live or resumed structural evidence without influencing scoring."""
        swe_capture = capture["swebench_search_replace"]
        if resumed:
            swe_capture["resumed_rows"] += 1
        else:
            swe_capture["new_rows"] += 1
        if diagnostic is None:
            swe_capture["resumed_rows_without_diagnostics"] += 1
            swe_capture["summary_complete"] = False
            return
        if diagnostic["has_markers"]:
            swe_capture["rows_with_markers"] += 1
        swe_capture["parseable_blocks"] += diagnostic["parseable_block_count"]
        if diagnostic["malformed_contract"]:
            swe_capture["malformed_rows"] += 1
        state = str(diagnostic["state"])
        swe_capture["state_counts"][state] += 1
        if diagnostic["parseable_block_count"] == 0:
            swe_capture["zero_strict_block_rows"] += 1
        elif state != "strict_ready":
            swe_capture["partial_strict_block_rows"] += 1

    # Idempotent resume: never re-query a (suite, question, seed) already on
    # disk. Lets an interrupted run resume, and an avg@k top-up add only new
    # seeds, without re-spending inference on results already collected. A
    # shared JSONL contains multiple suites, so rows from another suite must
    # neither suppress this suite's work nor be folded into its counters.
    already: set = set()
    if per_question_out is not None:
        done_path = getattr(per_question_out, "name", None)
        if done_path and Path(done_path).exists():
            question_by_id = {
                q.get("id", f"{suite_name}_{index:04d}"): q
                for index, q in enumerate(questions)
            }
            kept_lines: list[str] = []
            rejected_rows: list[dict] = []
            for line in Path(done_path).read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    rejected_rows.append({"raw_line": line, "reason": "malformed_json"})
                    continue
                if r.get("suite") != suite_name:
                    kept_lines.append(line)
                    continue
                key = (r.get("id"), r.get("seed"))
                question = question_by_id.get(r.get("id"))
                if question is None or not valid_resume_row(r, suite_name, question):
                    rejected_rows.append({"row": r, "reason": "resume_validation_failed"})
                    print(
                        f"  [runner] resume: rejecting unverifiable row {key}; re-querying",
                        file=sys.stderr,
                    )
                    continue
                if key in already:
                    rejected_rows.append({"row": r, "reason": "duplicate_resume_key"})
                    continue
                kept_lines.append(line)
                already.add(key)
                tier = str(r.get("tier", 2))
                qid = r.get("id", "")
                per_tier.setdefault(tier, {"correct": 0, "n": 0})
                per_item.setdefault(qid, {"correct": 0, "n": 0})
                per_tier[tier]["n"] += 1
                per_item[qid]["n"] += 1
                total += 1
                if r.get("correct"):
                    correct += 1
                    per_tier[tier]["correct"] += 1
                    per_item[qid]["correct"] += 1
                if r.get("empty_response"):
                    errors += 1
                if r.get("truncated"):
                    truncated += 1
                fold_transport_and_length(
                    str(r.get("finish_reason") or ""),
                    str(r.get("request_error") or ""),
                )
                if suite_name == "swebench_oracle":
                    stored = r.get("swe_search_replace")
                    if isinstance(stored, dict) and "state" in stored:
                        fold_swe_diagnostics(stored, resumed=True)
                    else:
                        fold_swe_diagnostics(
                            swe_search_replace_diagnostics(
                                r["response"], str(r.get("finish_reason") or ""),
                                str(r.get("request_error") or ""),
                            ),
                            resumed=True,
                        )
            if rejected_rows:
                rejected_path = Path(f"{done_path}.rejected.jsonl")
                with rejected_path.open("a") as rejected_handle:
                    for rejected in rejected_rows:
                        rejected_handle.write(json.dumps(rejected) + "\n")
                    rejected_handle.flush()
                    os.fsync(rejected_handle.fileno())
                # Rejected evidence is durable before replacement, preventing
                # duplicate keys after the fresh draw is appended.
                replace_capture_contents(per_question_out, kept_lines)
            if already:
                print(f"  [runner] resume: {len(already)} (id,seed) draws already "
                      f"on disk — folded in, not re-queried", file=sys.stderr)

    # Repeats are the OUTER loop on purpose. Iterating questions outermost
    # would mean an interrupted avg@k run had all k draws for early questions
    # and none for late ones -- a subset-biased score. Sweeping the full
    # question set once per repeat instead means every completed pass is a
    # valid avg@1, and partial work degrades to avg@(k-1) plus a fragment.
    for rep in range(repeats):
        # Distinct seed per repeat so avg@k samples k independent draws
        # instead of re-running one deterministic path k times.
        rep_seed = seed + rep
        if repeats > 1:
            print(f"  [runner] {suite_name}: pass {rep+1}/{repeats} "
                  f"(seed {rep_seed})", file=sys.stderr)

        # Build this pass's work list, skipping (id,seed) already on disk.
        pending = []
        for i, q in enumerate(questions):
            # Do NOT uppercase: fine for A-J / digits, but corrupts LaTeX gold
            # (\frac -> \FRAC). score_response applies case rules per method.
            expected = str(q.get("expected", "")).strip()
            if not expected:
                continue
            qid = q.get("id", f"{suite_name}_{i:04d}")
            if (qid, rep_seed) in already:
                continue  # already collected — never re-query (idempotent resume)
            pending.append((i, q, qid, expected))

        def _work(item):
            """Pure: query + score one question. No shared-state mutation, so it
            runs safely in a thread pool; the main thread applies the record."""
            i, q, qid, expected = item
            meta = query_server_meta(
                url, q["prompt"], max_tokens=max_tokens, temperature=temperature,
                seed=rep_seed, endpoint=endpoint, top_p=top_p, top_k=top_k,
                enable_thinking=enable_thinking)
            response = meta["text"]
            if not response:
                is_correct, got = False, ""
            else:
                is_correct = score_response(response, expected, q)
                _method = q.get("scoring_method", "multiple_choice")
                if _method == "multiple_choice":
                    got = extract_letter_answer(response)
                elif _method in ("math_numeric", "math_symbolic"):
                    got = extract_boxed(response)
                else:
                    got = extract_exact_answer(response, q.get("scoring_config", {}) or {})
            return (i, q, qid, expected, meta, response, is_correct, got)

        def _apply(res):
            """Main-thread only: fold one result into counters + JSONL."""
            nonlocal correct, total, errors, truncated
            i, q, qid, expected, meta, response, is_correct, got = res
            tier = str(q.get("tier", 2))
            per_tier.setdefault(tier, {"correct": 0, "n": 0})
            per_item.setdefault(qid, {"correct": 0, "n": 0})
            per_tier[tier]["n"] += 1
            per_item[qid]["n"] += 1
            total += 1
            if meta.get("finish_reason") == "length":
                truncated += 1
            fold_transport_and_length(
                str(meta.get("finish_reason") or ""),
                str(meta.get("error") or ""),
            )
            if not response:
                errors += 1
            elif is_correct:
                correct += 1
                per_tier[tier]["correct"] += 1
                per_item[qid]["correct"] += 1
            response_fingerprint = text_fingerprint(response)
            prompt = q["prompt"]
            prompt_fingerprint = text_fingerprint(prompt)
            reasoning = str(meta.get("reasoning") or "")
            reasoning_fingerprint = text_fingerprint(reasoning)
            capture["new_rows"] += 1
            capture["response_chars"] += response_fingerprint["chars"]
            capture["response_utf8_bytes"] += response_fingerprint["utf8_bytes"]
            capture["reasoning_chars"] += reasoning_fingerprint["chars"]
            capture["reasoning_utf8_bytes"] += reasoning_fingerprint["utf8_bytes"]
            swe_diag = None
            if suite_name == "swebench_oracle":
                swe_diag = swe_search_replace_diagnostics(
                    response,
                    str(meta.get("finish_reason") or ""),
                    str(meta.get("error") or ""),
                )
                fold_swe_diagnostics(swe_diag, resumed=False)
                if swe_diag["state"] != "strict_ready":
                    anomalies = []
                    if swe_diag["state"] == "request_error":
                        anomalies.append("request_error")
                    if meta.get("finish_reason") == "length":
                        anomalies.append("length_cap")
                    if swe_diag["parseable_block_count"] == 0:
                        anomalies.append("zero_strict_blocks")
                    else:
                        anomalies.append("partial_strict_blocks")
                    print(
                        f"[runner] WARNING swebench_oracle capture not converter-ready "
                        f"id={qid} state={swe_diag['state']} anomalies={','.join(anomalies)} "
                        f"markers={swe_diag['marker_counts']} "
                        f"parseable={swe_diag['parseable_block_count']}",
                        file=sys.stderr,
                    )
            if per_question_out is not None:
                # Written per result, not at completion: an interrupted run keeps
                # everything collected so far.
                row = {
                    "arm": arm, "suite": suite_name, "id": qid, "tier": tier,
                    "capture_schema_version": CAPTURE_SCHEMA_VERSION,
                    "runner_source_sha256": RUNNER_SOURCE_SHA256,
                    "rep": rep, "seed": rep_seed, "expected": expected,
                    "effective_request": meta.get("effective"),
                    "extracted": got, "correct": bool(is_correct),
                    "empty_response": not response,
                    "finish_reason": meta.get("finish_reason", ""),
                    "truncated": meta.get("finish_reason") == "length",
                    "completion_tokens": meta.get("completion_tokens", 0),
                    "prompt_tokens": meta.get("prompt_tokens", 0),
                    "decode_tok_s": round(meta.get("decode_tok_s", 0.0), 2),
                    "request_error": meta.get("error", ""),
                    "prompt": prompt,
                    "prompt_fingerprint": prompt_fingerprint,
                    "response_fingerprint": response_fingerprint,
                    "reasoning": reasoning,
                    "reasoning_fingerprint": reasoning_fingerprint,
                    "reasoning_chars": reasoning_fingerprint["chars"],
                    "empty_content_with_reasoning": (
                        not response and bool(reasoning)),
                    # SWE SEARCH/REPLACE conversion is performed from this artifact.
                    # Truncating it turns a valid model response into a different patch.
                    "response": response,
                }
                if swe_diag is not None:
                    row["swe_search_replace"] = swe_diag
                per_question_out.write(json.dumps(row) + "\n")
                per_question_out.flush()
                if live_status_out is not None:
                    write_live_capture_status(
                        live_status_out,
                        suite_name=suite_name,
                        arm=arm,
                        completed_draws=total,
                        expected_draws=expected_draws,
                        capture=capture,
                    )
            tok_acc[0] += meta.get("completion_tokens", 0)
            tok_acc[1] += meta.get("prompt_tokens", 0)

        if concurrency > 1 and len(pending) > 1:
            # Client-side concurrency; server serves them from its -np slots.
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futs = {pool.submit(_work, it): it for it in pending}
                for n, fut in enumerate(as_completed(futs), 1):
                    _apply(fut.result())  # main thread — no lock needed
                    if n % 25 == 0:
                        print(f"  [runner] {suite_name}: pass {rep+1}/{repeats} "
                              f"{n}/{len(pending)} ({correct}/{total} correct so far)",
                              file=sys.stderr)
        else:
            for n, it in enumerate(pending, 1):
                _apply(_work(it))
                if n % 25 == 0:
                    print(f"  [runner] {suite_name}: pass {rep+1}/{repeats} "
                          f"{n}/{len(pending)} ({correct}/{total} correct so far)",
                          file=sys.stderr)

    accuracy = correct / total if total > 0 else 0.0

    tier_results = {}
    for t, data in sorted(per_tier.items()):
        tier_results[t] = {
            "accuracy": data["correct"] / data["n"] if data["n"] > 0 else 0,
            "n": data["n"],
            "correct": data["correct"],
        }

    suite_wall = time.monotonic() - suite_t0
    if suite_name == "swebench_oracle":
        swe_capture = capture["swebench_search_replace"]
        states = swe_capture["state_counts"]
        captured_rows = swe_capture["new_rows"] + swe_capture["resumed_rows"]
        swe_capture["converter_contract_ready"] = (
            swe_capture["summary_complete"]
            and captured_rows > 0
            and states["strict_ready"] == captured_rows
        )
        if not swe_capture["summary_complete"]:
            swe_capture["score_status"] = "incomplete_capture_diagnostics"
        elif states["request_error"]:
            swe_capture["score_status"] = "provisional_request_error"
        elif states["prompt_contract_candidate"]:
            swe_capture["score_status"] = "provisional_prompt_contract"
        else:
            # Length states remain model-side outcomes. They do not make the
            # SWE test verdict provisional or silently passable.
            swe_capture["score_status"] = "terminal_no_prompt_contract_candidate"
    return {
        "suite": suite_name,
        "accuracy": accuracy,
        "n": total,
        "n_questions": len(per_item),
        "repeats": repeats,
        "correct": correct,
        "errors": errors,
        "truncated": truncated,
        "per_tier": tier_results,
        "per_item": per_item,
        "capture": capture,
        # Throughput (this run only; excludes resumed draws). Aggregate =
        # tokens generated across all concurrent slots / wall-clock.
        "throughput": {
            "concurrency": concurrency,
            "wall_s": round(suite_wall, 1),
            "completion_tokens": tok_acc[0],
            "prompt_tokens": tok_acc[1],
            "aggregate_decode_tok_s": round(tok_acc[0] / suite_wall, 1) if suite_wall > 0 else 0,
            "aggregate_total_tok_s": round((tok_acc[0] + tok_acc[1]) / suite_wall, 1) if suite_wall > 0 else 0,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="V7 quality-gate runner")
    p.add_argument("--port", type=int, default=18072,
                   help="llama-server port (default: 18072)")
    p.add_argument("--host", default="localhost",
                   help="llama-server host (default: localhost)")
    p.add_argument("--output", required=True, type=Path,
                   help="Output JSON path")
    p.add_argument("--suites", nargs="+", default=["mmlu_pro", "gpqa"],
                   help="Suites to evaluate (default: mmlu_pro gpqa)")
    p.add_argument("--n", type=int, default=200,
                   help="Questions per suite (default: 200)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sampling (default: 42)")
    p.add_argument("--stratify", action="store_true",
                   help="Use stratified sampling (equal per tier)")
    p.add_argument("--max-tokens", type=int, default=64,
                   help="Max tokens for model response (default: 64)")
    p.add_argument("--endpoint", choices=["chat", "completion"], default="chat",
                   help="llama-server API endpoint mode (default: chat)")
    p.add_argument("--kernel", default="v7-candidate",
                   help="Kernel label for output metadata")
    p.add_argument("--binary", default="",
                   help="Binary path for output metadata")
    p.add_argument("--models", default="",
                   help="Model path(s) for output metadata")
    p.add_argument("--timeout", type=int, default=120,
                   help="Server health check timeout (seconds)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (default: 0.0). Sampling-sensitive "
                        "suites should use the production temperature, not 0.")
    p.add_argument("--top-p", type=float, default=None,
                   help="Sampling top_p (default: server default)")
    p.add_argument("--top-k", type=int, default=None,
                   help="Sampling top_k (default: server default)")
    p.add_argument("--enable-thinking", dest="enable_thinking",
                   action="store_true", default=None,
                   help="Send chat_template_kwargs.enable_thinking=true")
    p.add_argument("--no-enable-thinking", dest="enable_thinking",
                   action="store_false",
                   help="Send chat_template_kwargs.enable_thinking=false")
    p.add_argument("--repeats", type=int, default=1,
                   help="Draws per question (avg@k). Each repeat uses seed+rep.")
    p.add_argument("--per-question-out", type=Path, default=None,
                   help="JSONL path for canonical per-question captures (default: "
                        "beside --output)")
    p.add_argument("--live-status-out", type=Path, default=None,
                   help="Atomic live capture-status JSON path (default: "
                        "<per-question-out>.live-status.json)")
    p.add_argument("--questions-out", type=Path, default=None,
                   help="Write the sampled item set here so later arms can replay it")
    p.add_argument("--questions-in", type=Path, default=None,
                   help="Replay a pinned item set instead of sampling (paired arms)")
    p.add_argument("--limit", type=int, default=0,
                   help="Use only the first N items of a pinned set (ablations)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Concurrent in-flight requests (client-side); match to the "
                        "server's -np slots. 1 = sequential.")
    p.add_argument("--arm", default="",
                   help="Arm label recorded in per-question records")
    p.add_argument("--belief-category", choices=["BASELINE", "CANDIDATE"],
                   default=None,
                   help="When set, emit producer-authored belief_measurements rows "
                        "into the result receipt at finalize (SC32): BASELINE for the "
                        "anchor arm, CANDIDATE for controls. Absent = no emission; "
                        "pre-hook runs (2026-08-12 panel, gpqa-cj1) stay zero-row.")
    p.add_argument("--belief-config", default="",
                   help="Optional JSON string merged into each belief row's "
                        "extra.arm_config — server-side facts the runner cannot "
                        "observe (template, quant detail)")
    args = p.parse_args()

    # Every non-dry run has a canonical capture.  Keep the optional spelling
    # compatible with older callers while making omission fail closed by default.
    if args.per_question_out is None:
        args.per_question_out = args.output.with_suffix(".per-question.jsonl")

    url = f"http://{args.host}:{args.port}"
    print(f"[runner] Waiting for server at {url}...", file=sys.stderr)
    wait_for_server(url, timeout=args.timeout)
    print("[runner] Server healthy", file=sys.stderr)

    # Determine binary path
    binary = args.binary or os.environ.get("LLAMA_BINARY", "")

    if args.questions_out:
        pinned = {
            s: load_questions(s, args.n, args.seed, args.stratify)
            for s in args.suites
        }
        args.questions_out.write_text(json.dumps({"suites": pinned}, indent=2))
        print(f"[runner] Pinned item set written to {args.questions_out}",
              file=sys.stderr)

    pq_handle = None
    if args.per_question_out:
        args.per_question_out.parent.mkdir(parents=True, exist_ok=True)
        pq_handle = args.per_question_out.open("a")
    live_status_out = (
        args.live_status_out
        or (args.per_question_out.with_suffix(".live-status.json")
            if args.per_question_out else None)
    )

    suites_results = []
    start = time.monotonic()

    questions_in = args.questions_in or args.questions_out

    for suite in args.suites:
        print(f"\n[runner] Evaluating {suite} (n={args.n}, seed={args.seed})...",
              file=sys.stderr)
        result = run_suite(
            suite, url, args.n, args.seed,
            stratify=args.stratify,
            max_tokens=args.max_tokens,
            endpoint=args.endpoint,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            enable_thinking=args.enable_thinking,
            repeats=args.repeats,
            per_question_out=pq_handle,
            questions_in=questions_in,
            limit=args.limit,
            arm=args.arm,
            concurrency=args.concurrency,
            live_status_out=live_status_out,
        )
        suites_results.append(result)
        acc = result.get("accuracy", 0)
        n = result.get("n", 0)
        print(f"[runner] {suite}: {acc:.1%} ({result.get('correct',0)}/{n})",
              file=sys.stderr)

    elapsed = time.monotonic() - start
    if pq_handle is not None:
        pq_handle.close()

    output = {
        "meta": {
            "kernel": args.kernel,
            "binary": binary,
            "models": args.models or "unknown",
            "arm": args.arm,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(elapsed, 1),
            "n_per_suite": args.n,
            "seed": args.seed,
            "stratify": args.stratify,
            "endpoint": args.endpoint,
            # REQUESTED values. The completion path pins top_k=1 (greedy) and
            # never sends top_p / enable_thinking, so these are what was ASKED
            # FOR, not necessarily what was applied. The authority for what was
            # actually sent is `effective_request` on each per-question row.
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "enable_thinking": args.enable_thinking,
            "sampling_fields_are_requested_not_effective": True,
            "repeats": args.repeats,
            "max_tokens": args.max_tokens,
            "questions_pinned": str(questions_in) if questions_in else "",
            "capture_schema_version": CAPTURE_SCHEMA_VERSION,
            "runner_source_sha256": runner_source_sha256(),
        },
        "suites": suites_results,
    }

    args.output.write_text(json.dumps(output, indent=2))

    # Result-finalize belief emission (SC32). The plain manifest above is the
    # content attested at collect time; belief rows are attached to it only
    # when the caller explicitly declares the arm's role. A refusal keeps the
    # manifest on disk (zero belief rows) and exits non-zero so the driver
    # notices the claim was not made.
    belief_exit = 0
    if args.belief_category is not None:
        try:
            if str(Path(__file__).resolve().parent) not in sys.path:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
            import v7_quality_gate_beliefs as beliefs
            arm_config = {}
            if args.belief_config:
                arm_config = json.loads(args.belief_config)
            rows = beliefs.attach_accuracy_beliefs(
                output,
                output_path=args.output,
                category=args.belief_category,
                runner_source_sha256=runner_source_sha256(),
                host=args.host,
                port=args.port,
                concurrency=args.concurrency,
                arm_config=arm_config,
            )
            output["belief_measurements"] = rows
            output["belief_attestation"] = {
                "schema": beliefs.PROTOCOL_ID,
                "attestation_sha256": rows[0]["extra"]["attestation_sha256"],
                "attestation_path": str(args.output),
            }
            args.output.write_text(json.dumps(output, indent=2))
            print(f"[runner] belief_measurements: {len(rows)} row(s) attached "
                  f"(category={args.belief_category})", file=sys.stderr)
        except Exception as exc:  # BeliefRefused or malformed --belief-config
            print(f"[runner] BELIEF REFUSED: {exc}", file=sys.stderr)
            print("[runner] result.json kept with zero belief rows; the claim "
                  "was not emitted", file=sys.stderr)
            belief_exit = 3

    print(f"\n[runner] Results written to {args.output}", file=sys.stderr)
    return belief_exit


if __name__ == "__main__":
    sys.exit(main())
