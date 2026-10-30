#!/usr/bin/env python3
"""PROPOSED RUS-3 native capture. Never starts a server. No protocol ratification.

Requires owner grant externally plus WINDOW_CONFIRMED=1 and
RUS3_SOURCE_ENROLLMENT_ACCEPTED=1 before transport. The latter records the owner's
acceptance of prospective capture wiring, not a new measurement grading rule.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

SCHEMA = "epyc.rus3.native-sample.v1"
SAFE_SCORERS = {"exact_match", "multiple_choice", "substring"}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("native direct-server capture refuses redirects")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_scorer(path: Path, expected_digest: str):
    path = path.resolve(strict=True)
    if digest(path.read_bytes()) != expected_digest:
        raise ValueError("actual scorer source digest differs from config")
    spec = importlib.util.spec_from_file_location("rus3_pinned_scorer_" + expected_digest, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(inspect.getsourcefile(module.score_answer)).resolve() != path.resolve():
        raise ValueError("scorer source must be actual implementation, not delegation shim")
    return module


def request_body(config: dict, question: dict, seed: int) -> dict:
    return {"model": config["model"], "messages": [{"role": "user", "content": question["prompt"]}],
            "temperature": config["temperature"], "seed": seed,
            "max_tokens": config["max_tokens"], "stream": False,
            "chat_template_kwargs": {"enable_thinking": True}}


def decode_response(data: dict) -> dict:
    """Copy native fields only; unknown/malformed fields refuse this sample."""
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("expected exactly one native choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("missing native message")
    finish = choice.get("finish_reason")
    if not isinstance(finish, str) or not finish:
        raise ValueError("missing native finish_reason")
    tokens = (data.get("usage") or {}).get("completion_tokens")
    if type(tokens) is not int or tokens < 0:
        raise ValueError("missing exact completion_tokens")
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if content is not None and not isinstance(content, str):
        raise ValueError("invalid native content")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError("invalid native reasoning_content")
    model = data.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("missing served model")
    return {"finish_reason": finish, "tokens_generated": tokens,
            "answer": content, "reasoning_content": reasoning,
            "trace_source": "native_reasoning_content" if reasoning else "unavailable",
            "served_model": model}


def validate_config(config: dict, questions: list[dict]) -> None:
    if config.get("protocol_id") != "":
        raise ValueError("no RUS3 protocol ratified: protocol_id must be empty")
    if type(config.get("k")) is not int or config["k"] < 4:
        raise ValueError("k must be >=4")
    temperature = config.get("temperature")
    if type(temperature) not in (float, int) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and >0")
    if type(config.get("seed")) is not int:
        raise ValueError("seed must be an integer")
    if config["seed"] < 0 or config["seed"] + len(questions) * config["k"] - 1 >= 4294967295:
        raise ValueError("all seeds must be uint32 excluding random-seed sentinel 4294967295")
    if type(config.get("max_tokens")) is not int or config["max_tokens"] <= 0:
        raise ValueError("max_tokens must be positive")
    url = urllib.parse.urlsplit(config.get("endpoint", ""))
    if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1")
            or not url.port or url.port == 8000 or url.username or url.password
            or url.path != "/v1/chat/completions" or url.query or url.fragment):
        raise ValueError("require direct loopback llama-server chat endpoint (not :8000)")
    for key in ("run_id", "role", "model", "model_sha256", "binary_sha256", "build_commit"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"missing declared launch identity: {key}")
    for key in ("model_sha256", "binary_sha256", "scorer_sha256", "owner_guard_sha256"):
        if len(config[key]) != 64 or any(c not in "0123456789abcdef" for c in config[key]):
            raise ValueError(f"invalid {key}")
    if not questions:
        raise ValueError("empty question manifest")
    ids = set()
    for question in questions:
        for key in ("id", "suite", "prompt", "expected", "scoring_method"):
            if not isinstance(question.get(key), str) or not question[key]:
                raise ValueError(f"question missing {key}")
        identity = (question["suite"], question["id"])
        if identity in ids:
            raise ValueError("duplicate question identity")
        ids.add(identity)
        if question["scoring_method"] not in SAFE_SCORERS:
            raise ValueError("only deterministic non-transport exact_match/multiple_choice/substring supported")


def owner_boundary(config, guard: Path, deadline: float) -> float:
    guard = guard.resolve(strict=True)
    transport_allowed()
    remaining = deadline - time.time()
    if remaining <= 0:
        raise ValueError("owner deadline reached; no further request authorized")
    if digest(guard.read_bytes()) != config["owner_guard_sha256"]:
        raise ValueError("owner guard digest mismatch")
    subprocess.run([str(guard)], check=True, timeout=min(5.0, remaining),
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    remaining = deadline - time.time()
    if remaining <= 0:
        raise ValueError("owner deadline elapsed during guard")
    return remaining


def transport_allowed() -> None:
    if os.environ.get("WINDOW_CONFIRMED") != "1":
        raise ValueError("WINDOW_CONFIRMED=1 required; owner grant still required externally")
    if (os.environ.get("RUS3_SOURCE_ENROLLMENT_ACCEPTED") != "1"
            and os.environ.get("RUS3_DIRECT_SMOKE") != "1"
            and os.environ.get("RUS3_DIAGNOSTIC") != "1"):
        raise ValueError("prospective source enrollment must be accepted before capture")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--launch-identity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--owner-guard", type=Path, required=True)
    parser.add_argument("--scorer-source", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    question_bytes = args.questions.read_bytes()
    receipt = args.launch_identity.read_bytes()
    if not receipt.strip():
        raise ValueError("empty owner launch receipt")
    launch = json.loads(receipt)
    server_identity = launch["server_identity"]
    expected_maps = launch["ggml_maps"]
    def live_maps():
        pid = server_identity["pid"]
        if Path(f"/proc/{pid}/stat").read_text().split()[21] != str(server_identity["start_ticks"]):
            raise ValueError("server process generation changed")
        paths = {line.split()[-1] for line in Path(f"/proc/{pid}/maps").read_text().splitlines()
                 if line.split()[-1].startswith("/") and "libggml" in line.split()[-1]}
        observed = [{"path": path, "sha256": digest(Path(path).read_bytes())} for path in sorted(paths)]
        if observed != expected_maps:
            raise ValueError("live ggml mappings changed")
        return observed
    config = json.loads(config_bytes)
    questions = [json.loads(line) for line in question_bytes.splitlines() if line.strip()]
    validate_config(config, questions)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    transport_allowed()  # before mkdir, imports with side effects, or any network
    if not math.isfinite(args.deadline_unix) or args.deadline_unix <= time.time():
        raise ValueError("future finite owner deadline required")
    owner_boundary(config, args.owner_guard, args.deadline_unix)
    debug_scorer = load_scorer(args.scorer_source, config["scorer_sha256"])
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_bytes(config_bytes)
    (args.out / "questions.jsonl").write_bytes(question_bytes)
    (args.out / "launch-identity.json").write_bytes(receipt)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect)
    metadata = {"config_sha256": digest(config_bytes), "questions_sha256": digest(question_bytes),
                "launch_identity_sha256": digest(receipt), "producer_sha256": digest(Path(__file__).read_bytes()),
                "scorer_sha256": config["scorer_sha256"], "owner_guard_sha256": config["owner_guard_sha256"],
                "deadline_unix": args.deadline_unix, "protocol_id": ""}
    (args.out / "scorer-source.py").write_bytes(args.scorer_source.read_bytes())
    (args.out / "owner-guard-source").write_bytes(args.owner_guard.read_bytes())
    (args.out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    retained = 0
    with (args.out / "samples.jsonl").open("x") as output:
        for qi, question in enumerate(questions):
            for index in range(config["k"]):
                try:
                    remaining = owner_boundary(config, args.owner_guard, args.deadline_unix)
                except Exception as error:
                    (args.out / "INCOMPLETE.json").write_text(json.dumps({"reason": str(error), "safe_cancel_or_drain": "NOT_PROVEN"}))
                    return 2
                seed = config["seed"] + qi * config["k"] + index
                body = request_body(config, question, seed)
                request_bytes = json.dumps(body, sort_keys=True).encode()
                sample = {"schema": SCHEMA, "run_id": config["run_id"], "role": config["role"],
                          "suite": question["suite"], "question_id": question["id"], "sample_index": index,
                          "temperature": config["temperature"], "seed": seed,
                          "request_sha256": digest(request_bytes), "started_at_unix": time.time(),
                          **metadata}
                sample.update(server_pid=server_identity["pid"], server_pgid=server_identity["pgid"],
                              server_start_ticks=server_identity["start_ticks"],
                              server_boot_id=server_identity["boot_id"], started_monotonic=time.monotonic())
                stem = f"q{qi:05d}-s{index:04d}"
                sample["raw_request_path"] = f"{stem}.request.json"
                (args.out / f"{stem}.request.json").write_bytes(request_bytes)
                try:
                    sample["ggml_map_identity"] = live_maps()
                    request = urllib.request.Request(config["endpoint"], data=request_bytes,
                                                     headers={"Content-Type": "application/json"})
                    with opener.open(request, timeout=min(args.timeout, remaining)) as response:
                        raw = response.read()
                    if live_maps() != sample["ggml_map_identity"]:
                        raise ValueError("live ggml changed across request")
                    raw_path = args.out / f"{stem}.response.json"
                    raw_path.write_bytes(raw)
                    sample.update(raw_response_path=raw_path.name, raw_response_sha256=digest(raw))
                    sample.update(decode_response(json.loads(raw)))
                    if sample["served_model"] != config["model"]:
                        raise ValueError("served model differs from declared model")
                    sample["passed"] = None
                    if sample["finish_reason"] == "stop":
                        try:
                            sample["passed"] = debug_scorer.score_answer(
                                answer=sample["answer"] or "", expected=question["expected"],
                                scoring_method=question["scoring_method"],
                                scoring_config=question.get("scoring_config"))
                            sample["scoring_status"] = "scored"
                        except debug_scorer.AnswerParseError:
                            sample["scoring_status"] = "answer_parse_failed"
                        except debug_scorer.ScoringUnavailableError:
                            sample["scoring_status"] = "unavailable"
                    sample["status"] = "native"
                    if sample["finish_reason"] == "stop" and type(sample["passed"]) is bool:
                        retained += 1
                except Exception as error:
                    if isinstance(error, urllib.error.HTTPError):
                        sample["http_status"] = error.code
                        try:
                            error_bytes = error.read()
                            error_path = args.out / f"{stem}.http-error.body"
                            error_path.write_bytes(error_bytes)
                            sample.update(http_error_path=error_path.name, http_error_sha256=digest(error_bytes))
                        except Exception as capture_error:
                            sample["http_error_body_capture_error"] = str(capture_error)
                    timeout_reason = getattr(error, "reason", error)
                    cut_off = (isinstance(timeout_reason, TimeoutError)
                               and time.time() >= args.deadline_unix)
                    sample.update(status="window_interrupted" if cut_off else "refused",
                                  error_type=type(error).__name__, error=str(error))
                sample["ended_at_unix"] = time.time()
                sample["ended_monotonic"] = time.monotonic()
                output.write(json.dumps(sample, sort_keys=True) + "\n")
                output.flush()
                os.fsync(output.fileno())
                if sample["status"] != "native" or time.time() >= args.deadline_unix:
                    (args.out / "INCOMPLETE.json").write_text(json.dumps({"reason": "refused sample or owner deadline reached", "safe_cancel_or_drain": "NOT_PROVEN"}))
                    return 2
    if retained == 0:
        (args.out / "NO_RETAINED_SAMPLES.json").write_text(json.dumps({"retained": 0, "measurement_acceptance": "NOT_MET"}))
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
