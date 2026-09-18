#!/usr/bin/env python3
"""Zero-inference equivalence harness for the v7 runner re-pin (2026-09-16).

Question answered: when ``--belief-category`` is absent, does the CURRENT
``scripts/benchmark/v7_quality_gate_runner.py`` (sha256 20a97fbd...) measure
exactly what the two SEALED versions measured?

    79721927...  blob 511f921c  b9ad1008 (2026-07-26)  pinned by the P3 bake-off manifest
    6dea92dd...  blob 5167f570  baf36757 (2026-08-19)  pinned by dflash2_followups
    20a97fbd...  blob b1c27738  da06b371 (2026-08-26)  current origin/main

The runner has no dry-run mode: it always waits for a server. So each version
runs end-to-end, as a real subprocess and with its real CLI, against a
deterministic in-process HTTP stub on 127.0.0.1. The stub is not a model. It
answers from a hash of the request bytes, and uses the expected answer to make
about half the items correct, so both scoring branches run. It also forces
some ``finish_reason=length`` rows and some HTTP 500s, which exercises the
truncation and request-error paths.

For every (version, scenario) pair the harness records:
  * the exact request bytes the runner SENT (path + body), as a sorted multiset
  * the result JSON, and the per-question JSONL sorted by (suite, id, rep)
It then compares each sealed version against current, in two ways:
  * byte comparison of the requests (must be IDENTICAL)
  * comparison of outputs after removing only VOLATILE fields (timestamps,
    wall-clock timings, runner_source_sha256, host-local temp paths). The
    residual diff is reported key by key, so an additive schema change is
    visible as exactly the keys it adds.

Runner versions are read from git blobs, not from the working tree. Each one
runs inside a private copy of the current scripts/benchmark directory, so the
runner file is the only variable.

Usage:
    python3 equivalence_harness.py --repo <research checkout> --out <workdir>
Writes <workdir>/summary.json. Exit 0 only when every request stream is
byte-identical and every output residual is the documented expected set.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import shutil
import socketserver
import subprocess
import sys
import threading
from pathlib import Path

RUNNER_REL = "scripts/benchmark/v7_quality_gate_runner.py"
VERSIONS = {
    "79721927": "511f921c8abd347b32563cc87d407fc0764d8f8a",
    "6dea92dd": "5167f5702dcf218ca500efb3cb98d5a0e07e10ca",
    "20a97fbd": "b1c2773881858e4750edbc022c93af6f650a64b9",
}
CURRENT = "20a97fbd"
FULL_SHA = {
    "79721927": "79721927e95293d070aba294bf422a24b1182dde07310d461d9e3ddaf6c84b0e",
    "6dea92dd": "6dea92dd9e374f79691f5df502fa11035ffd484906754f20190a4189111ae7dc",
    "20a97fbd": "20a97fbd4aecc6f3887299243362d5315b76f819aa1cbc4b8fd92f33b307b45a",
}

LIVE = Path("/mnt/raid0/llm/epyc-inference-research")
INPUTS = {
    # the dflash2 follow-up campaign's own pinned item set (sha 2088d2c0...)
    "olymp": Path("/workspace/tmp/questions_mtp_ab.json"),
    # the P3 bake-off's pinned inputs
    "critic": LIVE / "artifacts/p3-shadow-bakeoff-20260728/manifest/critic_tasks_v1.json",
    "swe": LIVE / "artifacts/architect-code-eval-20260724/questions_swebench_oracle.json",
    "lcb": LIVE / "artifacts/architect-code-eval-20260724/questions_livecodebench_hard.json",
}

# Synthetic rows whose inline scorer is live (the real dflash2 set uses
# scoring_method="exact", which score_response does not dispatch, so it can never
# score True). These make the correct=True branch run for every version.
SYNTHETIC = (
    [{"id": f"mc_{i}", "prompt": f"Synthetic MC question {i}? (A) a (B) b (C) c (D) d",
      "expected": "ABCD"[i % 4], "tier": 1, "scoring_method": "multiple_choice"} for i in range(6)]
    + [{"id": f"sym_{i}", "prompt": f"Synthetic symbolic question {i}", "expected": f"$2^{{k-{i}}}$",
        "tier": 2, "scoring_method": "math_symbolic"} for i in range(4)]
    + [{"id": f"num_{i}", "prompt": f"Synthetic numeric question {i}", "expected": f"0{70 + i}",
        "tier": 3, "scoring_method": "exact_match",
        "scoring_config": {"extract_pattern": r"(\d+)\s*$", "normalize_numeric": True}} for i in range(4)]
)

# (name, input key, suite, extra CLI args)
SCENARIOS = [
    ("olymp_chat_sampled_c1", "olymp", "olympiadbench_hard",
     ["--endpoint", "chat", "--temperature", "0.6", "--top-p", "0.95", "--top-k", "20",
      "--enable-thinking", "--max-tokens", "512", "--concurrency", "1"]),
    ("olymp_completion_greedy_c4", "olymp", "olympiadbench_hard",
     ["--endpoint", "completion", "--temperature", "0.6", "--top-p", "0.95", "--top-k", "20",
      "--max-tokens", "256", "--concurrency", "4"]),
    ("critic_chat_nothink_rep2", "critic", "p3_cocritic_v1",
     ["--endpoint", "chat", "--no-enable-thinking", "--repeats", "2", "--limit", "20",
      "--max-tokens", "256", "--concurrency", "2"]),
    ("swe_chat_c1", "swe", "swebench_oracle",
     ["--endpoint", "chat", "--limit", "4", "--max-tokens", "1024"]),
    ("synthetic_scored_chat_c2", "synthetic", "gpqa",
     ["--endpoint", "chat", "--repeats", "2", "--max-tokens", "128", "--concurrency", "2"]),
    ("lcb_chat_c1", "lcb", "livecodebench_hard",
     ["--endpoint", "chat", "--limit", "4", "--max-tokens", "1024"]),
]

# Keys whose values are wall-clock or host-local by nature. Removing them is the
# ONLY normalisation applied; everything else is compared as-is.
VOLATILE_KEYS = {
    "timestamp", "elapsed_s", "runner_source_sha256", "latency_s", "wall_s",
    "elapsed", "duration_s", "suite_elapsed_s", "started_at", "finished_at",
    "updated_at", "written_at", "t_start", "t_end", "questions_pinned",
    "wall_time_s", "tasks_per_hour", "aggregate_tok_s", "request_wall_s",
    "aggregate_decode_tok_s", "aggregate_total_tok_s", "per_request_tok_s",
    "updated_utc", "created_utc",
}

# Keys that baf36757 (79721927 -> 6dea92dd) ADDED. Expected as the ONLY residual
# for the 79721927 comparison; never expected for 6dea92dd.
BAF36757_ADDED = {"effective_request", "sampling_fields_are_requested_not_effective"}


def git_blob(repo: Path, blob: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), "cat-file", "blob", blob],
                          check=True, capture_output=True).stdout


def expected_by_prompt(rows: list[dict]) -> dict[str, str]:
    return {str(r.get("prompt", "")): str(r.get("expected", "")) for r in rows}


class Stub:
    """Deterministic fake llama-server. Records every POST body verbatim."""

    def __init__(self, expected: dict[str, str]):
        self.expected = expected
        self.requests: list[dict] = []
        self.lock = threading.Lock()

    def respond(self, path: str, body: bytes) -> tuple[int, dict]:
        h = hashlib.sha256(body).digest()
        with self.lock:
            self.requests.append({"path": path, "body_sha256": hashlib.sha256(body).hexdigest(),
                                  "body": body.decode("utf-8")})
        if h[0] % 11 == 0:
            return 500, {"error": {"message": "stub transport failure"}}
        payload = json.loads(body)
        if path.endswith("/chat/completions"):
            prompt = payload["messages"][-1]["content"]
        else:
            prompt = payload["prompt"]
        exp = self.expected.get(prompt, "")
        right = h[1] % 2 == 0
        if right and len(exp) == 1:
            text = f"Working through it step by step.\n\nAnswer: {exp}"
        elif right and exp.isdigit():
            text = f"Working through it step by step.\n\nFinal answer: {int(exp)}"
        elif right and exp:
            text = f"Working through it step by step.\n\nThe final answer is \\boxed{{{exp.strip('$')}}}"
        else:
            text = f"Working through it step by step.\n\nThe final answer is \\boxed{{wrong{h[2]}}}"
        finish = "length" if h[3] % 7 == 0 else "stop"
        usage = {"completion_tokens": 10 + h[4], "prompt_tokens": 100 + h[5]}
        timings = {"predicted_per_second": 50.0 + h[6], "prompt_per_second": 900.0 + h[7]}
        if path.endswith("/chat/completions"):
            choice = {"message": {"content": text, "reasoning_content": f"think-{h[8]}"},
                      "finish_reason": finish}
        else:
            choice = {"text": text, "finish_reason": finish}
        return 200, {"choices": [choice], "usage": usage, "timings": timings}


def serve(stub: Stub) -> socketserver.TCPServer:
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: D401, ANN002
            return

        def do_GET(self):  # noqa: N802
            data = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            code, obj = stub.respond(self.path, body)
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def strip_volatile(obj, removed: set[str], path: str = ""):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in VOLATILE_KEYS:
                removed.add(k)
                continue
            out[k] = strip_volatile(v, removed, f"{path}/{k}")
        return out
    if isinstance(obj, list):
        return [strip_volatile(v, removed, path) for v in obj]
    return obj


def key_residual(a, b, path: str = "") -> list[str]:
    """Paths where a and b differ; a missing key is reported as +key / -key."""
    diffs: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                diffs.append(f"+{path}/{k}")
            elif k not in b:
                diffs.append(f"-{path}/{k}")
            else:
                diffs.extend(key_residual(a[k], b[k], f"{path}/{k}"))
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diffs.extend(key_residual(x, y, f"{path}[]"))
    elif a != b:
        diffs.append(f"~{path}")
    return sorted(set(diffs))


MUTANTS = {
    # request-level: one extra field in every chat payload
    "request": (b'"stream": False,', b'"stream": False, "n_probs": 0,'),
    # scoring-level: every inline verdict inverted
    "scoring": (b'"correct": bool(is_correct),', b'"correct": not bool(is_correct),'),
}


def run_one(repo: Path, work: Path, version: str, scenario, mutant: str | None = None) -> dict:
    name, key, suite, extra = scenario
    base = work / (version + (f"-mutant-{mutant}" if mutant else "")) / name
    if base.exists():
        shutil.rmtree(base)
    bench = base / "bench"
    shutil.copytree(repo / "scripts/benchmark", bench,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    src = git_blob(repo, VERSIONS[version])
    assert hashlib.sha256(src).hexdigest() == FULL_SHA[version], version
    if mutant:
        old, new = MUTANTS[mutant]
        assert src.count(old) == 1, mutant
        src = src.replace(old, new)
    (bench / "v7_quality_gate_runner.py").write_bytes(src)

    raw = {"suites": {suite: SYNTHETIC}} if key == "synthetic" else json.loads(INPUTS[key].read_text())
    rows = raw["suites"][suite] if isinstance(raw, dict) and "suites" in raw else raw
    if isinstance(raw, dict) and "suites" not in raw:
        raise SystemExit(f"unexpected input shape for {key}")
    pinned = base / "pinned.json"
    pinned.write_text(json.dumps({"suites": {suite: rows}}))

    stub = Stub(expected_by_prompt(rows))
    srv = serve(stub)
    out = base / "result.json"
    pq = base / "result.per-question.jsonl"
    cmd = [sys.executable, str(bench / "v7_quality_gate_runner.py"),
           "--host", "127.0.0.1", "--port", str(srv.server_address[1]),
           "--output", str(out), "--suites", suite, "--questions-in", str(pinned),
           "--per-question-out", str(pq), "--arm", "equiv", "--kernel", "equiv",
           "--timeout", "30", *extra]
    try:
        proc = subprocess.run(cmd, cwd=bench, capture_output=True, text=True, timeout=600)
    finally:
        srv.shutdown()
        srv.server_close()
    rows_out = [json.loads(l) for l in pq.read_text().splitlines() if l.strip()] if pq.exists() else []
    rows_out.sort(key=lambda r: (str(r.get("suite")), str(r.get("id")), r.get("rep", 0)))
    reqs = sorted(stub.requests, key=lambda r: (r["path"], r["body_sha256"]))
    return {
        "exit_code": proc.returncode,
        "stderr_tail": proc.stderr.strip().splitlines()[-3:],
        "requests": reqs,
        "result": json.loads(out.read_text()) if out.exists() else None,
        "per_question": rows_out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--negative-control", action="store_true",
                    help="prove the comparison has teeth: run mutants of the 6dea92dd runner "
                         "and exit 0 only if EVERY mutant is detected")
    args = ap.parse_args()
    if args.negative_control:
        sc = next(s for s in SCENARIOS if s[0] == "synthetic_scored_chat_c2")
        cur = run_one(args.repo, args.out, CURRENT, sc)
        detected = {}
        for m in MUTANTS:
            mut = run_one(args.repo, args.out, "6dea92dd", sc, mutant=m)
            removed: set[str] = set()
            req_eq = ([(r["path"], r["body"]) for r in mut["requests"]]
                      == [(r["path"], r["body"]) for r in cur["requests"]])
            out_eq = (json.dumps(strip_volatile(mut["per_question"], removed), sort_keys=True)
                      == json.dumps(strip_volatile(cur["per_question"], removed), sort_keys=True)
                      and json.dumps(strip_volatile(mut["result"], removed), sort_keys=True)
                      == json.dumps(strip_volatile(cur["result"], removed), sort_keys=True))
            detected[m] = {"requests_byte_identical": req_eq,
                           "outputs_identical_after_volatile_strip": out_eq,
                           "detected": not (req_eq and out_eq)}
            print(f"negative control {m}: {detected[m]}")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "negative_control.json").write_text(
            json.dumps({"schema": "v7_runner_repin_negative_control.v1",
                        "mutants": {k: [a.decode(), b.decode()] for k, (a, b) in MUTANTS.items()},
                        "results": detected}, indent=2, sort_keys=True) + "\n")
        return 0 if all(d["detected"] for d in detected.values()) else 1
    for k, p in INPUTS.items():
        if not p.is_file():
            print(f"COULD-NOT-CHECK: input {k} missing at {p}", file=sys.stderr)
            return 2
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {"schema": "v7_runner_repin_equivalence.v1", "versions": FULL_SHA,
               "inputs": {k: {"path": str(p),
                              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for k, p in INPUTS.items()},
               "volatile_keys_removed": sorted(VOLATILE_KEYS),
               "scenarios": {}}
    ok = True
    for sc in SCENARIOS:
        name = sc[0]
        runs = {v: run_one(args.repo, args.out, v, sc) for v in VERSIONS}
        cur = runs[CURRENT]
        entry = {"current": {"exit_code": cur["exit_code"],
                             "n_requests": len(cur["requests"]),
                             "n_rows": len(cur["per_question"]),
                             "request_stream_sha256": hashlib.sha256(json.dumps(
                                 [(r["path"], r["body_sha256"]) for r in cur["requests"]]
                             ).encode()).hexdigest(),
                             "status_codes_forced": "HTTP 500 on ~1/11, length on ~1/7",
                             "correct": [s.get("correct") for s in (cur["result"] or {}).get("suites", [])]},
                 "vs": {}}
        if cur["exit_code"] != 0 or not cur["requests"] or not cur["per_question"]:
            ok = False
            entry["current"]["defect"] = "current run did not exercise the path"
        for v in VERSIONS:
            if v == CURRENT:
                continue
            old = runs[v]
            removed: set[str] = set()
            same_requests = ([(r["path"], r["body"]) for r in old["requests"]]
                             == [(r["path"], r["body"]) for r in cur["requests"]])
            res_old = strip_volatile(old["result"], removed)
            res_cur = strip_volatile(cur["result"], removed)
            pq_old = strip_volatile(old["per_question"], removed)
            pq_cur = strip_volatile(cur["per_question"], removed)
            residual = sorted(set(key_residual(res_old, res_cur, "result"))
                              | set(key_residual(pq_old, pq_cur, "per_question")))
            added_leaf = {p.rsplit("/", 1)[-1] for p in residual}
            expected_residual = BAF36757_ADDED if v == "79721927" else set()
            residual_ok = (all(p.startswith("+") for p in residual)
                           and added_leaf <= expected_residual)
            byte_identical_after_volatile = (json.dumps(res_old, sort_keys=True) == json.dumps(res_cur, sort_keys=True)
                                             and json.dumps(pq_old, sort_keys=True) == json.dumps(pq_cur, sort_keys=True))
            entry["vs"][v] = {
                "exit_code": old["exit_code"],
                "requests_byte_identical": same_requests,
                "outputs_identical_after_volatile_strip": byte_identical_after_volatile,
                "residual_paths": residual,
                "residual_is_expected_additive_set": residual_ok,
                "volatile_keys_seen": sorted(removed),
            }
            if old["exit_code"] != cur["exit_code"] or not same_requests or not residual_ok:
                ok = False
            if v == "6dea92dd" and not byte_identical_after_volatile:
                ok = False
        summary["scenarios"][name] = entry
        print(f"{name}: current rc={cur['exit_code']} reqs={len(cur['requests'])} rows={len(cur['per_question'])} "
              + " ".join(f"{v}:req_eq={e['requests_byte_identical']} out_eq={e['outputs_identical_after_volatile_strip']} "
                         f"residual={e['residual_paths']}" for v, e in entry["vs"].items()))
    summary["verdict"] = "EQUIVALENT" if ok else "NOT-EQUIVALENT"
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"verdict: {summary['verdict']}  -> {args.out / 'summary.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
