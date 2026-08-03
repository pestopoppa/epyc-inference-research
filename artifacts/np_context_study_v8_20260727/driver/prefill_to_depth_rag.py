#!/usr/bin/env python3
"""Pinned LongHealth document-QA TB-6 decode-at-KV-depth observation runner.

The prompt is a production chat/Jinja conversation made from complete,
immutable LongHealth documents and one original multiple-choice benchmark row.
No synthetic source, repeated packet, or byte-prefix truncation is permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ART = Path("/mnt/raid0/llm/epyc-inference-research/artifacts/np_context_study_v8_20260727")
BIN = Path("/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server")
RUNNER = Path(__file__).resolve()
LONGHEALTH = Path("/mnt/raid0/llm/epyc-inference-research/data/external/compaction/data/longhealth_benchmark_v5.json")
LONGHEALTH_SHA256 = "82d34d9da47ab279d7aa89a6bdf298c0ac79f1e506e1dd0a3ea69a1ad5e2cb45"
CORES, PORT = "184-191", 18072
KERNEL_HEAD = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
GEN_TOKENS, SLOT_HEADROOM, VRAM_LIMIT_GIB = 32, 64, 61
GRID = ((2048, (1, 2, 4, 8, 16, 32)), (8192, (1, 2, 4, 8, 16)),
        (16384, (1, 2, 4, 8)), (32768, (1, 2, 4)))


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, include_hash: bool) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {"path": str(path), "inode": stat.st_ino, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if include_hash:
        result["sha256"] = sha256(path)
    return result


def longhealth_data() -> dict[str, Any]:
    if not LONGHEALTH.is_file() or sha256(LONGHEALTH) != LONGHEALTH_SHA256:
        fail("LongHealth source hash mismatch")
    data = json.loads(LONGHEALTH.read_text())
    if not isinstance(data, dict) or "patient_01" not in data:
        fail("LongHealth source schema mismatch")
    return data


def source_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return complete paragraph passages, never arbitrary byte prefixes.

    LongHealth stores full clinical documents rather than retrieval rows.  A
    blank-line-delimited paragraph is the smallest source-faithful document QA
    passage; it keeps source document, position, and passage hashes explicit.
    """
    question = question_row(data)
    answer = question["correct"].casefold()
    rows = []
    for patient_id, patient in data.items():
        for doc_id, text in patient["texts"].items():
            source_document_sha256 = hashlib.sha256(text.encode()).hexdigest()
            for passage_index, passage in enumerate(re.split(r"\n\s*\n", text)):
                if not passage.strip():
                    continue
                row = {
                    "patient_id": patient_id,
                    "document_id": doc_id,
                    "passage_index": passage_index,
                    "source_document_sha256": source_document_sha256,
                    "contains_pinned_answer": answer in passage.casefold(),
                    "content": passage,
                }
                rows.append({**row, "row_sha256": canonical_hash(row)})
    if not any(row["patient_id"] == "patient_01" and row["contains_pinned_answer"] for row in rows):
        fail("LongHealth source has no passage containing the pinned answer")
    return sorted(
        rows,
        key=lambda row: (
            not (row["patient_id"] == "patient_01" and row["contains_pinned_answer"]),
            row["patient_id"] != "patient_01",
            row["document_id"],
            row["passage_index"],
        ),
    )


def question_row(data: dict[str, Any]) -> dict[str, Any]:
    question = data["patient_01"]["questions"][4]
    if question["No"] != 4 or "Which therapy regime" not in question["question"]:
        fail("pinned LongHealth question row mismatch")
    return question


def question_content(question: dict[str, Any], documents: list[dict[str, Any]]) -> str:
    citations = "\n\n".join(
        f"[retrieved_document rank={rank} patient={row['patient_id']} id={row['document_id']} passage={row['passage_index']}]\n{row['content']}\n[/retrieved_document]"
        for rank, row in enumerate(documents, 1)
    )
    answers = "\n".join(f"{letter}. {question[f'answer_{letter.lower()}']}" for letter in "ABCDE")
    return ("Answer the benchmark question using the retrieved medical documents. Select exactly one option and give one brief evidence sentence.\n\n"
            f"{citations}\n\nQuestion: {question['question']}\n{answers}\nAnswer:")


def source_manifest(data: dict[str, Any]) -> dict[str, Any]:
    question = question_row(data)
    rows = source_rows(data)
    return {"file": {**file_identity(LONGHEALTH, False), "sha256": LONGHEALTH_SHA256},
            "question": {"patient_id": "patient_01", "question_no": question["No"], "row_sha256": canonical_hash(question)},
            "documents": [{key: row[key] for key in ("patient_id", "document_id", "passage_index", "source_document_sha256", "contains_pinned_answer", "row_sha256")} for row in rows]}


def model_spec(label: str, path: Path, mtp: int, cached: dict[str, Any] | None = None) -> dict[str, Any]:
    if mtp < 0:
        fail("MTP depth cannot be negative")
    identity = file_identity(path, False)
    if (
        isinstance(cached, dict)
        and cached.get("label") == label
        and cached.get("mtp_depth") == mtp
        and all(cached.get(key) == identity[key] for key in ("path", "inode", "bytes", "mtime_ns"))
        and isinstance(cached.get("sha256"), str)
    ):
        identity["sha256"] = cached["sha256"]
    else:
        identity["sha256"] = sha256(path)
    return {"label": label, **identity, "mtp_depth": mtp,
            "server_spec": {"type": "none" if mtp == 0 else "draft-mtp", "draft_n_max": mtp},
            "dflash": "off", "kv_cache": {"k": "f16", "v": "f16"}}


def prepare(args: argparse.Namespace) -> int:
    data = longhealth_data()
    previous_models: dict[str, dict[str, Any]] = {}
    if args.prepare_out.is_file():
        try:
            previous = json.loads(args.prepare_out.read_text())
            previous_models = {
                row["label"]: row
                for row in previous.get("models", [])
                if isinstance(row, dict) and isinstance(row.get("label"), str)
            }
        except (OSError, ValueError, TypeError):
            previous_models = {}
    models = []
    for label, model, mtp in args.model:
        path = Path(model)
        if not path.is_file():
            fail(f"missing model: {path}")
        models.append(model_spec(label, path, int(mtp), previous_models.get(label)))
    if len({row["label"] for row in models}) != len(models):
        fail("duplicate prepared model label")
    payload = {"schema": "epyc.tb6.prefill_to_depth.longhealth_chat.prepare.v2",
        "kernel": {"branch": "production-consolidated-v8", "head": KERNEL_HEAD, "binary": str(BIN), "binary_sha256": sha256(BIN)},
        "instrument": {"path": str(RUNNER), "sha256": sha256(RUNNER)},
        "source": source_manifest(data), "models": models, "grid": [{"depth": depth, "np": list(nps)} for depth, nps in GRID],
        "generation_tokens": GEN_TOKENS, "slot_headroom_tokens": SLOT_HEADROOM,
        "serving": {"endpoint": "/v1/chat/completions", "template_endpoint": "/apply-template", "reasoning": "off", "dflash": "off", "thread_fence": CORES}}
    args.prepare_out.parent.mkdir(parents=True, exist_ok=True)
    args.prepare_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def load_prepared(path: Path, label: str, model: Path) -> dict[str, Any]:
    prepared = json.loads(path.read_text())
    if prepared.get("schema") != "epyc.tb6.prefill_to_depth.longhealth_chat.prepare.v2":
        fail("prepared manifest schema mismatch")
    if prepared.get("instrument") != {"path": str(RUNNER), "sha256": sha256(RUNNER)}:
        fail("prepared RAG instrument identity mismatch")
    if prepared.get("kernel", {}).get("head") != KERNEL_HEAD or prepared.get("kernel", {}).get("binary_sha256") != sha256(BIN):
        fail("prepared production binary identity mismatch")
    if prepared.get("source") != source_manifest(longhealth_data()):
        fail("prepared LongHealth row/source identity mismatch")
    rows = [row for row in prepared["models"] if row["label"] == label]
    if len(rows) != 1:
        fail(f"no unique prepared model for {label}")
    selected = rows[0]
    current = file_identity(model, False)
    if any(selected.get(key) != current.get(key) for key in ("path", "inode", "bytes", "mtime_ns")):
        fail(f"model changed after preflight: {model}")
    if selected.get("dflash") != "off" or selected.get("kv_cache") != {"k": "f16", "v": "f16"}:
        fail("prepared cache/speculation invariant mismatch")
    return selected


def http_json(path: str, body: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        fail(f"HTTP {path} failed: {exc}")


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def require_port_unused() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        if sock.connect_ex(("127.0.0.1", PORT)) == 0:
            fail(f"refusing to touch occupied port {PORT}")


def apply_template(messages: list[dict[str, str]]) -> str:
    rendered = http_json("/apply-template", {"messages": messages}).get("prompt")
    if not isinstance(rendered, str) or not rendered:
        fail("/apply-template returned no rendered chat prompt")
    return rendered


def rendered_tokens(messages: list[dict[str, str]]) -> tuple[str, int]:
    rendered = apply_template(messages)
    tokens = http_json("/tokenize", {"content": rendered}).get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        fail("/tokenize returned no integer token list")
    return rendered, len(tokens)


def messages_for_depth(data: dict[str, Any], depth: int) -> tuple[list[dict[str, str]], list[dict[str, Any]], str, int]:
    question, rows = question_row(data), source_rows(data)
    best: tuple[list[dict[str, str]], list[dict[str, Any]], str, int] | None = None
    low, high = 1, len(rows)
    while low <= high:
        count = (low + high) // 2
        selected = rows[:count]
        messages = [{"role": "user", "content": question_content(question, selected)}]
        rendered, token_count = rendered_tokens(messages)
        if token_count > depth:
            high = count - 1
        else:
            best = messages, selected, rendered, token_count
            low = count + 1
    if best is None:
        fail(f"no complete retrieved LongHealth row fits depth={depth}")
    return best


def fence(pid: int, output: Path) -> None:
    applied = []
    for _ in range(2):
        run = subprocess.run(["taskset", "-apc", CORES, str(pid)], text=True, capture_output=True, check=False)
        applied.append(run.stdout + run.stderr)
        if run.returncode:
            fail("taskset thread fence application failed")
        time.sleep(1)
    rows = []
    for task in sorted((Path("/proc") / str(pid) / "task").iterdir()):
        affinity = next((line.split(":", 1)[1].strip() for line in (task / "status").read_text().splitlines() if line.startswith("Cpus_allowed_list:")), "")
        rows.append({"tid": int(task.name), "cpus_allowed_list": affinity})
    output.joinpath("thread_affinity.apply.txt").write_text("".join(applied))
    output.joinpath("thread_affinity.json").write_text(json.dumps({"pid": pid, "expected": CORES, "rows": rows}, indent=2))
    if not rows or any(row["cpus_allowed_list"] != CORES for row in rows):
        fail("thread fence mismatch")


def vram_gib() -> int:
    result = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], text=True, capture_output=True, check=False)
    match = re.search(r"used.*?:\s*(\d+)", result.stdout, re.I)
    return int(match.group(1)) // 1073741824 if match else 0


def run_cell(args: argparse.Namespace, model: dict[str, Any], data: dict[str, Any], depth: int, np: int) -> None:
    slot, directory = depth + SLOT_HEADROOM, args.output_dir / args.label / f"np{np}_D{depth}"
    directory.mkdir(parents=True, exist_ok=True)
    require_port_unused()
    argv = [str(BIN), "-m", str(args.model_path), "--host", "127.0.0.1", "--port", str(PORT), "--metrics", "--slots", "--jinja", "--device", "ROCm0", "-ngl", "all", "-fa", "on", "-np", str(np), "-c", str(np * slot), "-t", "8", "-tb", "8", "-b", "2048", "-ub", "2048", "-ctk", "f16", "-ctv", "f16", "--reasoning", "off"]
    if model["mtp_depth"]:
        argv += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(model["mtp_depth"])]
    directory.joinpath("server.argv").write_text(shlex.join(argv) + "\n")
    environment = {**os.environ, "GGML_IQK": "1", "LD_LIBRARY_PATH": str(BIN.parent)}
    with directory.joinpath("server.stdout").open("w") as stdout, directory.joinpath("server.stderr").open("w") as stderr:
        proc = subprocess.Popen(["taskset", "-c", CORES, *argv], stdout=stdout, stderr=stderr, env=environment, text=True)
        try:
            for _ in range(200):
                if proc.poll() is not None:
                    fail("server exited before health")
                if healthy():
                    break
                time.sleep(3)
            else:
                fail("server health timeout")
            fence(proc.pid, directory)
            found = re.findall(r"n_ctx_(?:per_seq|slot)\s*=\s*(\d+)", directory.joinpath("server.stderr").read_text())
            if not found or int(found[-1]) < slot or vram_gib() > VRAM_LIMIT_GIB:
                directory.joinpath("skip.txt").write_text(f"n_ctx_slot={found[-1] if found else 'missing'} slot={slot} vram={vram_gib()}G\n")
                return
            messages, documents, rendered, observed_depth = messages_for_depth(data, depth)
            body = {"messages": messages, "max_tokens": GEN_TOKENS, "temperature": 0.0, "seed": 42, "enable_thinking": False}
            evidence = {"endpoint": "/v1/chat/completions", "body": body,
                        "prepared_manifest": {"path": str(args.prepared.resolve()), "sha256": sha256(args.prepared)},
                        "instrument": {"path": str(RUNNER), "sha256": sha256(RUNNER)},
                        "model": model, "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                        "rendered_prompt_tokens": observed_depth, "target_depth_ceiling": depth,
                        "retrieved_rows": [{key: row[key] for key in ("patient_id", "document_id", "passage_index", "source_document_sha256", "contains_pinned_answer", "row_sha256")} for row in documents]}
            directory.joinpath("request.json").write_text(json.dumps(evidence, indent=2))
            with ThreadPoolExecutor(max_workers=np) as executor:
                responses = list(executor.map(lambda _: http_json("/v1/chat/completions", body, 600), range(np)))
            rows = []
            for response in responses:
                timings, usage = response.get("timings", {}), response.get("usage", {})
                if int(usage.get("prompt_tokens", timings.get("prompt_n", 0))) < observed_depth:
                    fail("chat response prompt-depth witness is below rendered prompt")
                rows.append({"prompt_tokens": usage.get("prompt_tokens", timings.get("prompt_n", 0)), "completion_tokens": usage.get("completion_tokens", timings.get("predicted_n", 0)), "decode_tok_s": timings.get("predicted_per_second", 0.0), "response": response})
            directory.joinpath("results.json").write_text(json.dumps({
                "target_depth_ceiling": depth,
                "rendered_depth": observed_depth,
                "np": np,
                "prepared_manifest": evidence["prepared_manifest"],
                "instrument": evidence["instrument"],
                "model": model,
                "rows": rows,
            }, indent=2))
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            if proc.poll() is None:
                fail("server process remained live after cleanup")


def execute(args: argparse.Namespace) -> int:
    model = load_prepared(args.prepared, args.label, args.model_path)
    data = longhealth_data()
    for depth, nps in GRID:
        for np in nps:
            run_cell(args, model, data, depth, np)
    (args.output_dir / args.label / "complete.txt").write_text(time.strftime("COMPLETE %Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prepare-out", type=Path, default=ART / "prefill_to_depth_rag.prepared.json")
    parser.add_argument("--model", action="append", nargs=3, metavar=("LABEL", "PATH", "MTP"), default=[])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--prepared", type=Path, default=ART / "prefill_to_depth_rag.prepared.json")
    parser.add_argument("--output-dir", type=Path, default=ART)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prepare == args.execute:
        fail("choose exactly one of --prepare or --execute")
    if args.prepare:
        if not args.model:
            fail("--prepare requires one or more --model LABEL PATH MTP")
        return prepare(args)
    if not args.label or not args.model_path:
        fail("--execute requires --label and --model-path")
    return execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"PREFILL_RAG_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
