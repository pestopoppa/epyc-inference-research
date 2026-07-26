#!/usr/bin/env python3
"""Fail-closed Laguna Q4_K_M CPU quality campaign.

Dry-run is the default and never loads a model. Execution requires a new
timestamped output directory, a released E8 CPU window, exact production
runtime reconciliation, and then runs SWE raw generation -> official
SWE-bench_Verified scoring -> LCB code execution on fresh owned sidecars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
LLAMA_ROOT = Path("/mnt/raid0/llm/llama.cpp")
BINARY = LLAMA_ROOT / "build/bin/llama-server"
MODEL = Path("/mnt/raid0/llm/models/Laguna-S-2.1-GGUF/laguna-s-2.1-Q4_K_M.gguf")
RUNTIME_FACTS = Path("/mnt/raid0/llm/tmp/orchestrator_runtime_facts.json")
STATE = ORCHESTRATOR_ROOT / "orchestration/autopilot_state.json"
SWE_QUESTIONS = (
    RESEARCH_ROOT / "artifacts/architect-code-eval-20260724/questions_swebench_oracle.json"
)
LCB_QUESTIONS = (
    RESEARCH_ROOT / "artifacts/architect-code-eval-20260724/questions_livecodebench_hard.json"
)
CONVERTER = RESEARCH_ROOT / "artifacts/architect-code-eval-20260724/convert_sr_to_patch.py"
SWEBENCH_PYTHON = RESEARCH_ROOT / ".venv-swebench/bin/python"
UV = Path("/usr/bin/uv")
RAW_EVALUATOR = RESEARCH_ROOT / "scripts/benchmark/v7_quality_gate_runner.py"
ANSWER_SCORING = RESEARCH_ROOT / "scripts/benchmark/answer_scoring.py"
CODE_EXEC_SCORER = RESEARCH_ROOT / "scripts/benchmark/code_exec_scorer.py"
EVALUATOR_PYTHON = Path("/usr/bin/python3")
SWEBENCH_PACKAGE = SWEBENCH_PYTHON.parent.parent / "lib/python3.12/site-packages/swebench"
DATASET_ADAPTERS = RESEARCH_ROOT / "scripts/benchmark/dataset_adapters.py"
DOCKER = Path("/usr/bin/docker")
TASKSET = Path("/usr/bin/taskset")
NUMACTL = Path("/usr/bin/numactl")
DRY_RUN_OUTPUT = RESEARCH_ROOT / "artifacts/laguna-q4-cpu-v8-20260726"

EXPECTED_BRANCH = "production-consolidated-v8"
EXPECTED_HEAD = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
EXPECTED_VERSION = "10107"
EXPECTED_BINARY_SHA256 = "a4b667163022aa166ade7c0e00fa4e775b37662e02c10da7642c8c23a4d6b414"
EXPECTED_MODEL_SHA256 = "7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f"
EXPECTED_SWE_SHA256 = "f82a5191274048f2fdf432df7a0ebf4017ad982b954d6aa075326a1302df1c3c"
EXPECTED_LCB_SHA256 = "d51e56f601e3d153910d086b35c6aea94f4d903bab0427c8a49ffe895a6287c4"
EXPECTED_CONVERTER_SHA256 = "ad13519ac6d56d36e6e29938b10fb500440f01780375692f2105a5c4d08202e0"
EXPECTED_SWEBENCH_PYTHON_SHA256 = "9544d2a29138833e6177d45dbc57468d37710b5080c901fbb579d53f251cdd6f"
EXPECTED_RUNTIME_FACTS_SHA256 = "dd7b3a7afd03ee20b5a00ea812d71e28548cd0104be9178c476d70e926a52f98"
EXPECTED_UV_SHA256 = "0c7b00d4f2c10d0bf66b26e4f432388d7782e1071d913f921eae0b8e35cb3c80"
EXPECTED_RAW_EVALUATOR_SHA256 = "18bf1577b531eabde978973aa62b9443a8a4ec9382928f71b5c658c15d2a8125"
EXPECTED_ANSWER_SCORING_SHA256 = "2253e6b1d378e7a929299ebee55ebb7e553a9ec24f2c489504cf2aaef24aabea"
EXPECTED_CODE_EXEC_SCORER_SHA256 = "12b8c9408d4b2f606929e37316c3f1c3d8f6252925dfb7bf6bdea541c3ef23cc"
EXPECTED_SWEBENCH_TREE_SHA256 = "f60e756fd46da88cb9640d5c5e1a4df6476e7b72ab49161f16ce4b493e589608"
EXPECTED_DATASET_ADAPTERS_SHA256 = "f6c60252e9a759017e452b6d101cb6ee410a362469588b09a2475f02cef5c91f"
EXPECTED_DOCKER_SHA256 = "1d7a20313e3a5ed83409f1bf3f06279a2726663f7a127a56354c97beb3261c08"
EXPECTED_TASKSET_SHA256 = "77f0448b62a216931c44f76492bde77fc26c9c32738470b2c4acca9b083922d2"
EXPECTED_NUMACTL_SHA256 = "e88dea788d8b8d2d50d688a55c5982492bb982ea8f5761cf1a5ec2919b981f40"
EXPECTED_EVALUATOR_PYTHON_SHA256 = "efb29ce53d36ebaeee80e3aa44fd6c7f9d71bbded5fe1665240b2ed8ecaeee0e"
EXPECTED_SWEBENCH_RECORD_SHA256 = "e7f2ca868e69b2fc49ff401c1629b6ddc26e33b2e03ae10eef021286f7961079"
EXPECTED_PRODUCTION_PORTS = (
    8070, 8072, 8080, 8082, 8083, 8085, 8086, 8087,
    8090, 8091, 8092, 8093, 8094, 8095,
    8180, 8182, 8185, 8280, 8282, 8285, 8380, 8382, 8385, 8485,
)
OWNED_SIDECARS: dict[int, dict[str, Any]] = {}
OWNED_SIDECARS_LOCK = threading.RLock()
EXPECTED_RUNTIME = {
    LLAMA_ROOT
    / "build/bin/libggml-base.so.0.16.0": "8ab8718efbd7cce0c350e1f096aad735cd0ad5c7b58e5fc7c58b6600f98f2949",
    LLAMA_ROOT
    / "build/bin/libggml-cpu.so.0.16.0": "4c56a1da53cd7e59b487ca4ca592e1bb382d61c487c7972d729c616918d2b214",
    LLAMA_ROOT
    / "build/bin/libggml.so.0.16.0": "ba0a91a85c8b1f1ede0680d6024fcab4c7e560a34f26f27dd832d9ed89a63434",
    LLAMA_ROOT
    / "build/bin/libllama.so.0.0.10107": "dad74a952f42937374f015da30ae3876e363e9d63d130a93dfe88ca81fe29ced",
    LLAMA_ROOT
    / "build/bin/libllama-common.so.0.0.10107": "0fc0b1014d997221effe1777fd247721c63d65ff7cddcde504b4d0f732e18e25",
    LLAMA_ROOT
    / "build/bin/libllama-server-impl.so": "9245e197c5ed332c8e7c362450a401c4d75073589e1f73d45327873c3b649cfc",
    LLAMA_ROOT
    / "build/bin/libmtmd.so.0.0.10107": "70b885f4b68356cddbbe8539131667ab6e2562117f8604b0497aa71e1fcbfce6",
    Path(
        "/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0"
    ): "fa075918dc2eae2dfcaccce487fa65034a9e046d0009cd99c55f0a0cc9314c36",
}
SWE_IDS = (
    "django__django-10999",
    "django__django-11066",
    "django__django-11087",
    "django__django-11095",
    "django__django-11099",
    "django__django-11119",
    "django__django-11133",
    "django__django-11138",
    "django__django-11141",
    "django__django-11149",
    "django__django-11163",
    "django__django-11179",
    "django__django-11206",
    "django__django-11211",
    "django__django-11239",
    "django__django-11265",
    "django__django-11276",
    "django__django-11292",
    "django__django-11299",
    "django__django-11333",
    "django__django-11400",
    "django__django-11433",
    "django__django-11451",
    "django__django-11477",
    "matplotlib__matplotlib-13989",
    "matplotlib__matplotlib-14623",
    "matplotlib__matplotlib-20488",
    "matplotlib__matplotlib-20676",
    "scikit-learn__scikit-learn-10297",
    "scikit-learn__scikit-learn-10844",
    "scikit-learn__scikit-learn-10908",
    "scikit-learn__scikit-learn-11310",
    "sphinx-doc__sphinx-10323",
    "sphinx-doc__sphinx-10435",
    "sphinx-doc__sphinx-10449",
    "sphinx-doc__sphinx-10466",
    "sympy__sympy-11618",
    "sympy__sympy-12096",
    "sympy__sympy-12419",
    "sympy__sympy-12481",
)
LCB_IDS = (
    "lcb_abc343_e",
    "lcb_abc342_e",
    "lcb_abc341_f",
    "lcb_abc340_d",
    "lcb_abc340_e",
    "lcb_abc338_d",
    "lcb_abc338_e",
    "lcb_abc338_f",
    "lcb_abc337_e",
    "lcb_abc334_e",
    "lcb_abc333_e",
    "lcb_abc332_d",
    "lcb_abc331_d",
    "lcb_abc331_e",
    "lcb_abc330_e",
    "lcb_abc329_e",
    "lcb_abc328_e",
    "lcb_abc328_d",
    "lcb_abc327_e",
    "lcb_abc326_e",
    "lcb_abc326_d",
    "lcb_abc325_f",
    "lcb_abc325_e",
    "lcb_abc325_d",
    "lcb_1899_B",
    "lcb_1899_C",
    "lcb_1899_D",
    "lcb_abc324_d",
    "lcb_abc324_f",
    "lcb_abc324_e",
    "lcb_abc323_e",
    "lcb_abc323_d",
    "lcb_abc322_e",
    "lcb_abc320_e",
    "lcb_abc318_e",
    "lcb_abc315_e",
    "lcb_abc315_f",
    "lcb_abc314_f",
    "lcb_abc314_e",
    "lcb_abc312_e",
    "lcb_abc312_f",
    "lcb_abc311_e",
    "lcb_abc310_f",
    "lcb_abc310_e",
    "lcb_abc309_e",
    "lcb_abc308_f",
    "lcb_abc308_e",
    "lcb_abc307_e",
    "lcb_abc305_e",
    "lcb_abc303_e",
    "lcb_abc302_f",
    "lcb_abc301_e",
    "lcb_abc301_f",
)
SUITES = (
    {
        "name": "swe_oracle",
        "external_name": "swebench_oracle",
        "port": 18092,
        "context": 49152,
        "questions": SWE_QUESTIONS,
        "ids": SWE_IDS,
        "question_sha256": EXPECTED_SWE_SHA256,
        "max_tokens": 3072,
    },
    {
        "name": "lcb_hard",
        "external_name": "livecodebench_hard",
        "port": 18093,
        "context": 8192,
        "questions": LCB_QUESTIONS,
        "ids": LCB_IDS,
        "question_sha256": EXPECTED_LCB_SHA256,
        "max_tokens": 4096,
    },
)
MEM_AVAILABLE_MIN_KIB = 100 * 1024 * 1024
RAW_EVALUATOR_TIMEOUT_S = 172800
OFFICIAL_SWE_TIMEOUT_S = 86400
CONVERTER_TIMEOUT_S = 1800
OFFICIAL_SWE_RUN_ID = "laguna-q4-cpu-v8-20260726"
CHILD_ENV = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LD_LIBRARY_PATH": str(BINARY.parent),
    "GGML_IQK": "1",
    "OMP_NUM_THREADS": "96",
    "OMP_PROC_BIND": "spread",
    "OMP_PLACES": "cores",
    "OMP_WAIT_POLICY": "active",
    "OMP_DYNAMIC": "false",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read required JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON {path} is not an object")
    return value


def file_identity(path: Path, expected_sha: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    actual = sha256(resolved)
    if actual != expected_sha:
        raise RuntimeError(f"pinned file SHA mismatch: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": actual,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "bytes": stat.st_size,
    }


def tree_sha256(path: Path) -> str:
    """Hash relative paths and contents, preventing package-file substitution."""
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*.py") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode() + b"\0")
        digest.update(bytes.fromhex(sha256(child)))
    return digest.hexdigest()


def clean_env() -> dict[str, str]:
    return dict(CHILD_ENV)


def verify_model_identity() -> dict[str, Any]:
    return file_identity(MODEL, EXPECTED_MODEL_SHA256)


def server_argv(suite: dict[str, Any]) -> list[str]:
    return [
        str(TASKSET),
        "-c",
        "0-95",
        str(NUMACTL),
        "--interleave=all",
        str(BINARY),
        "-m",
        str(MODEL),
        "--host",
        "127.0.0.1",
        "--port",
        str(suite["port"]),
        "-c",
        str(suite["context"]),
        "-t",
        "96",
        "-tb",
        "96",
        "-b",
        "2048",
        "-ub",
        "2048",
        "-np",
        "1",
        "-ctk",
        "f16",
        "-ctv",
        "f16",
        "-fa",
        "on",
        "-ngl",
        "0",
        "-dev",
        "none",
        "--no-op-offload",
        "--no-mmap",
        "--jinja",
        "--metrics",
        "--slots",
        "--reasoning",
        "off",
    ]


def question_contract(suite: dict[str, Any]) -> dict[str, Any]:
    path = Path(suite["questions"])
    actual_sha = sha256(path)
    rows = json.loads(path.read_text())
    ids = tuple(row.get("id") for row in rows) if isinstance(rows, list) else ()
    if actual_sha != suite["question_sha256"] or ids != suite["ids"]:
        raise RuntimeError(f"pinned {suite['name']} question hash or ordered IDs changed")
    return {"path": str(path), "sha256": actual_sha, "count": len(ids), "ids": list(ids)}


def validate_static(verify_model_sha: bool = False) -> dict[str, Any]:
    branch = subprocess.run(
        ["git", "-C", str(LLAMA_ROOT), "branch", "--show-current"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "-C", str(LLAMA_ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    version_result = subprocess.run(
        [str(BINARY), "--version"], text=True, capture_output=True, check=True
    )
    version = (version_result.stdout + version_result.stderr).strip()
    if branch != EXPECTED_BRANCH or head != EXPECTED_HEAD or EXPECTED_VERSION not in version:
        raise RuntimeError("frozen v8 identity mismatch")
    tracked_status = subprocess.run(
        [
            "git",
            "-C",
            str(LLAMA_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    if tracked_status:
        raise RuntimeError("frozen v8 tree has tracked changes")
    binary = file_identity(BINARY, EXPECTED_BINARY_SHA256)
    runtime = [file_identity(path, expected) for path, expected in EXPECTED_RUNTIME.items()]
    if not MODEL.is_file():
        raise RuntimeError("Laguna Q4 model is missing")
    model_sha = sha256(MODEL) if verify_model_sha else EXPECTED_MODEL_SHA256
    if verify_model_sha and model_sha != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Laguna Q4 model SHA mismatch")
    contracts = [question_contract(suite) for suite in SUITES]
    converter = file_identity(CONVERTER, EXPECTED_CONVERTER_SHA256)
    swebench_python = file_identity(SWEBENCH_PYTHON, EXPECTED_SWEBENCH_PYTHON_SHA256)
    evaluator = {
        "uv": file_identity(UV, EXPECTED_UV_SHA256),
        "runner": file_identity(RAW_EVALUATOR, EXPECTED_RAW_EVALUATOR_SHA256),
        "answer_scoring": file_identity(ANSWER_SCORING, EXPECTED_ANSWER_SCORING_SHA256),
        "code_exec_scorer": file_identity(CODE_EXEC_SCORER, EXPECTED_CODE_EXEC_SCORER_SHA256),
        "dataset_adapters": file_identity(DATASET_ADAPTERS, EXPECTED_DATASET_ADAPTERS_SHA256),
        "python": file_identity(EVALUATOR_PYTHON, EXPECTED_EVALUATOR_PYTHON_SHA256),
        "docker": file_identity(DOCKER, EXPECTED_DOCKER_SHA256),
        "taskset": file_identity(TASKSET, EXPECTED_TASKSET_SHA256),
        "numactl": file_identity(NUMACTL, EXPECTED_NUMACTL_SHA256),
    }
    package_hash = tree_sha256(SWEBENCH_PACKAGE)
    if package_hash != EXPECTED_SWEBENCH_TREE_SHA256:
        raise RuntimeError("pinned swebench distribution SHA mismatch")
    record = file_identity(
        SWEBENCH_PACKAGE.parent / "swebench-4.1.0.dist-info/RECORD", EXPECTED_SWEBENCH_RECORD_SHA256
    )
    return {
        "branch": branch,
        "head": head,
        "version": version,
        "binary": binary,
        "runtime_artifacts": runtime,
        "model_sha256": model_sha,
        "model_sha256_verified": verify_model_sha,
        "model_bytes": MODEL.stat().st_size,
        "questions": contracts,
        "converter": converter,
        "swebench_python": swebench_python,
        "evaluator_artifacts": evaluator,
        "swebench_distribution_sha256": package_hash,
        "swebench_record": record,
    }


def mem_available_kib(meminfo: str | None = None) -> int:
    text = meminfo if meminfo is not None else Path("/proc/meminfo").read_text()
    match = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", text, re.MULTILINE)
    if not match:
        raise RuntimeError("MemAvailable is unavailable")
    return int(match.group(1))


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def proc_cmdline(pid: int, proc_root: Path = Path("/proc")) -> list[str]:
    raw = (proc_root / str(pid) / "cmdline").read_bytes()
    return [part.decode() for part in raw.split(b"\0") if part]


def flag_value(argv: list[str], names: tuple[str, ...]) -> str | None:
    for index, value in enumerate(argv):
        if value in names and index + 1 < len(argv):
            return argv[index + 1]
    return None


def listener_owned(pid: int, port: int, proc_root: Path = Path("/proc")) -> bool:
    hex_port = f"{port:04X}"
    inodes: set[str] = set()
    for name in ("net/tcp", "net/tcp6"):
        for row in (proc_root / name).read_text().splitlines()[1:]:
            fields = row.split()
            if len(fields) >= 10 and fields[1].split(":")[-1] == hex_port and fields[3] == "0A":
                inodes.add(fields[9])
    if len(inodes) != 1:
        return False
    owned = set()
    for fd in (proc_root / str(pid) / "fd").iterdir():
        try:
            target = os.readlink(fd)
        except (FileNotFoundError, OSError):
            continue
        match = re.fullmatch(r"socket:\[(\d+)\]", target)
        if match:
            owned.add(match.group(1))
    return inodes <= owned


def live_llama_rows(proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            comm = (entry / "comm").read_text().strip()
            if comm != "llama-server":
                continue
            argv = proc_cmdline(pid, proc_root)
            port_text = flag_value(argv, ("--port", "-p"))
            model = flag_value(argv, ("-m", "--model"))
            exe = str((entry / "exe").resolve(strict=True))
            if port_text is None or model is None:
                raise RuntimeError(f"live llama-server {pid} lacks port or model")
            port = int(port_text)
            rows.append(
                {
                    "pid": pid,
                    "port": port,
                    "model": model,
                    "exe": exe,
                    "listener_owned": listener_owned(pid, port, proc_root),
                }
            )
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"cannot establish live llama-server identity for {pid}: {exc}"
            ) from exc
    return sorted(rows, key=lambda row: row["pid"])


def runtime_authorizations(facts: dict[str, Any]) -> set[tuple[int, int, str]]:
    state = facts.get("state")
    if not isinstance(state, dict):
        raise RuntimeError("runtime facts state is missing")
    by_port: dict[int, tuple[int, int, str]] = {}
    for key, row in state.items():
        if not key.startswith("server_") or not isinstance(row, dict):
            continue
        pid, port, model = row.get("pid"), row.get("port"), row.get("model_path")
        if (
            not isinstance(pid, int)
            or pid <= 0
            or not isinstance(port, int)
            or not isinstance(model, str)
        ):
            continue
        if model.startswith("preserved:"):
            source = state.get(model.split(":", 1)[1])
            model = source.get("model_path") if isinstance(source, dict) else None
        if not isinstance(model, str) or not model.startswith("/"):
            raise RuntimeError(f"runtime facts cannot resolve exact model for {key}")
        if port not in EXPECTED_PRODUCTION_PORTS:
            continue
        if port in by_port:
            raise RuntimeError(f"runtime facts contain duplicate production port {port}")
        by_port[port] = (pid, port, model)
    missing_ports = sorted(set(EXPECTED_PRODUCTION_PORTS) - set(by_port))
    if missing_ports:
        raise RuntimeError(f"runtime facts missing production ports: {missing_ports}")
    return set(by_port.values())


def runtime_guard(
    live_rows: list[dict[str, Any]] | None = None,
    facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = facts if facts is not None else read_json(RUNTIME_FACTS)
    facts_hash = sha256(RUNTIME_FACTS) if facts is None else EXPECTED_RUNTIME_FACTS_SHA256
    if facts_hash != EXPECTED_RUNTIME_FACTS_SHA256:
        raise RuntimeError("runtime facts SHA changed")
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at:
        raise RuntimeError("runtime facts generated_at is absent")
    live = live_rows if live_rows is not None else live_llama_rows()
    authorized = runtime_authorizations(payload)
    expected_exe = str(BINARY.resolve(strict=True))
    with OWNED_SIDECARS_LOCK:
        owned = {pid: row.copy() for pid, row in OWNED_SIDECARS.items()}
    candidate_rows = [row for row in live if row["pid"] in owned]
    if len(owned) > 1:
        raise RuntimeError("owned candidate registration does not match live sidecars")
    for row in candidate_rows:
        registered = owned[row["pid"]]
        if (
            row["port"] != registered["port"] or row["model"] != str(MODEL)
            or row["exe"] != str(BINARY.resolve(strict=True))
        ):
            raise RuntimeError("owned candidate identity changed")
    production_live = [row for row in live if row["pid"] not in owned]
    live_set = {(row["pid"], row["port"], row["model"]) for row in production_live}
    invalid = [
        row
        for row in production_live
        if (row["pid"], row["port"], row["model"]) not in authorized
        or row.get("exe") != expected_exe
        or row.get("listener_owned") is not True
    ]
    missing = authorized - live_set
    if invalid or missing or len(production_live) != 24 or len(authorized) != 24:
        raise RuntimeError(
            f"unknown or misbound live llama-server rows: invalid={invalid} "
            f"missing={sorted(missing)}"
        )
    return {
        "path": str(RUNTIME_FACTS),
        "sha256": facts_hash,
        "generated_at": generated_at,
        "authorized_live_rows": production_live,
        "owned_sidecars": candidate_rows,
    }


def computed_numeric_trials() -> int:
    code = """import json, sys
sys.path.insert(0, 'scripts/autopilot')
from autopilot import _frontier_rerun_completed_numeric_trials
from experiment_journal import ExperimentJournal
state=json.load(open('orchestration/autopilot_state.json', encoding='utf-8'))
print(_frontier_rerun_completed_numeric_trials(state.get('frontier_rerun_required') or {}, ExperimentJournal()))
"""
    try:
        result = subprocess.run(
            [str(ORCHESTRATOR_ROOT / ".venv/bin/python"), "-c", code],
            cwd=ORCHESTRATOR_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot compute authoritative E8 numeric trial count: {exc}") from exc


def e8_release(state: dict[str, Any], completed: int) -> dict[str, Any]:
    marker = state.get("frontier_rerun_required")
    quality = state.get("e8_quality_rebaseline")
    eras = state.get("active_instrument_eras")
    baseline = state.get("baseline_state")
    required = marker.get("required") if isinstance(marker, dict) else None
    reasons: list[str] = []
    if completed < 16:
        reasons.append("fewer than 16 E8 numeric trials")
    if required is not False:
        reasons.append("frontier rerun remains required")
    if not isinstance(eras, dict) or eras.get("eval_quality") != "E8":
        reasons.append("active eval_quality era is not E8")
    if not isinstance(baseline, dict) or baseline.get("eval_quality_era") != "E8":
        reasons.append("baseline eval_quality era is not E8")
    if not isinstance(quality, dict) or quality.get("status") not in {"closed", "applied"}:
        reasons.append("E8 quality rebaseline hold remains open")
    return {"released": not reasons, "reasons": reasons, "completed_numeric_trials": completed}


def live_autopilot(proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    found = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = proc_cmdline(int(entry.name), proc_root)
        except (FileNotFoundError, OSError):
            continue
        joined = " ".join(argv)
        if "scripts/autopilot/autopilot_supervisor.py" in joined or (
            "scripts/autopilot/autopilot.py" in joined and " start" in f" {joined}"
        ):
            found.append({"pid": int(entry.name), "argv": argv})
    return found


def execution_gates() -> dict[str, Any]:
    state = read_json(STATE)
    release = e8_release(state, computed_numeric_trials())
    memory = mem_available_kib()
    with OWNED_SIDECARS_LOCK:
        owned_ports = {row["port"] for row in OWNED_SIDECARS.values()}
    ports = {
        str(suite["port"]): suite["port"] in owned_ports or port_free(suite["port"])
        for suite in SUITES
    }
    runtime: dict[str, Any]
    runtime_error = ""
    try:
        runtime = runtime_guard()
    except RuntimeError as exc:
        runtime = {"error": repr(exc)}
        runtime_error = str(exc)
    autopilot = live_autopilot()
    failures = list(release["reasons"])
    if memory < MEM_AVAILABLE_MIN_KIB:
        failures.append("MemAvailable below 100 GiB")
    if not all(ports.values()):
        failures.append("required bench port is occupied")
    if runtime_error:
        failures.append("production stack continuity invalid: " + runtime_error)
    if autopilot:
        failures.append("AutoPilot supervisor or child is live")
    return {
        "passed": not failures,
        "failures": failures,
        "e8": release,
        "mem_available_kib": memory,
        "min_mem_available_kib": MEM_AVAILABLE_MIN_KIB,
        "ports_free": ports,
        "runtime_facts": runtime,
        "autopilot_processes": autopilot,
    }


class CampaignMonitor:
    """Continuously attest the release window while any Q4 work is active."""

    def __init__(self, output_dir: Path, interval_s: float = 5.0) -> None:
        self.output_dir, self.interval_s = output_dir, interval_s
        self.stop_event = threading.Event()
        self.failure: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self, phase: str) -> None:
        try:
            gate = execution_gates()
            record: dict[str, Any] = {"at": utc_now(), "phase": phase, "gates": gate}
            if not gate["passed"]:
                self.failure = "; ".join(gate["failures"])
        except Exception as exc:
            record = {"at": utc_now(), "phase": phase, "error": repr(exc)}
            self.failure = repr(exc)
        with (self.output_dir / "runtime_samples.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if self.failure:
            raise RuntimeError(f"campaign continuity gate failed: {self.failure}")

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            try:
                self.sample("monitor")
            except RuntimeError:
                return

    def start(self) -> None:
        self.sample("pre-execution")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval_s + 1)
        self.sample("post-execution")


def run_owned_command(
    command: list[str], *, cwd: Path, env: dict[str, str], timeout_s: int, label: str
) -> subprocess.CompletedProcess[str]:
    """Run an evaluator in an owned session and reap all descendants on timeout."""
    process = subprocess.Popen(
        command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=30)
        if process_group_members(process.pid):
            os.killpg(process.pid, signal.SIGKILL)
        raise RuntimeError(f"{label} timed out and owned process group was terminated") from exc
    if process_group_members(process.pid):
        os.killpg(process.pid, signal.SIGKILL)
        raise RuntimeError(f"{label} left descendant processes")
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def parse_docker_container_rows(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if fields[0] in parsed:
            raise RuntimeError("Docker container listing contains duplicate IDs")
        parsed[fields[0]] = fields[1] if len(fields) == 2 else ""
    return parsed


def validate_docker_container_transition(before_text: str, after_text: str) -> None:
    before = parse_docker_container_rows(before_text)
    after = parse_docker_container_rows(after_text)
    new_ids = set(after) - set(before)
    missing_ids = set(before) - set(after)
    if new_ids:
        raise RuntimeError(f"official SWE left new Docker containers: {sorted(new_ids)}")
    if missing_ids:
        raise RuntimeError(f"official SWE removed pre-existing Docker containers: {sorted(missing_ids)}")
    drifted = [container_id for container_id, value in before.items() if after[container_id] != value]
    if drifted:
        raise RuntimeError(f"official SWE changed pre-existing Docker containers: {sorted(drifted)}")


def campaign_container_owned(inspect_payload: dict[str, Any], run_id: str) -> bool:
    """Require independent name and label bindings before destructive cleanup."""
    name = str(inspect_payload.get("Name", "")).lstrip("/")
    config = inspect_payload.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    if not isinstance(labels, dict):
        return False
    exact_name = name == run_id or name.startswith(run_id + "-") or name.endswith("-" + run_id)
    exact_label = any(
        labels.get(key) == run_id
        for key in ("swebench.run_id", "org.swebench.run_id", "epyc.run_id")
    )
    return exact_name and exact_label


def classify_new_campaign_containers(
    before_text: str,
    after_text: str,
    inspections: dict[str, dict[str, Any]],
    run_id: str,
) -> tuple[list[str], list[str]]:
    before = parse_docker_container_rows(before_text)
    after = parse_docker_container_rows(after_text)
    new_ids = sorted(set(after) - set(before))
    owned = [container_id for container_id in new_ids if campaign_container_owned(inspections.get(container_id, {}), run_id)]
    residual = [container_id for container_id in new_ids if container_id not in owned]
    return owned, residual


def docker_operation(argv: list[str], env: dict[str, str]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "argv": argv,
        "stdout": "",
        "stderr": "",
        "returncode": None,
        "error": "",
    }
    try:
        result = subprocess.run(
            argv, text=True, capture_output=True, check=False, env=env,
        )
        record.update(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
        if result.returncode:
            record["error"] = f"exit {result.returncode}"
    except Exception as exc:
        record["error"] = repr(exc)
    return record


def prepare(
    output_dir: Path, execute: bool, argv: list[str], verify_model_sha: bool = False
) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {output_dir}")
    if execute and (
        output_dir == DRY_RUN_OUTPUT
        or not re.fullmatch(r"laguna-q4-cpu-v8-\d{8}T\d{6}Z", output_dir.name)
    ):
        raise RuntimeError("execution requires a fresh timestamped output directory")
    output_dir.mkdir(parents=True)
    plan = {
        "schema": "epyc.laguna_q4_cpu_v8.plan.v2",
        "created_at": utc_now(),
        "execute": execute,
        "runner_command": [sys.executable, str(Path(__file__).resolve()), *argv],
        "runner_sha256": sha256(Path(__file__).resolve()),
        "static_identity": validate_static(verify_model_sha),
        "gates": execution_gates(),
        "environment": clean_env(),
        "explicitly_omits": ["KMP_BLOCKTIME"],
        "timeouts_s": {
            "raw_evaluator": RAW_EVALUATOR_TIMEOUT_S,
            "converter": CONVERTER_TIMEOUT_S,
            "official_swe": OFFICIAL_SWE_TIMEOUT_S,
        },
        "suites": [
            {
                "name": suite["name"],
                "port": suite["port"],
                "context": suite["context"],
                "max_tokens": suite["max_tokens"],
                "argv": server_argv(suite),
                "question_contract": question_contract(suite),
                "sampling": {"seed": 42, "temperature": 0.6, "top_p": 0.95, "top_k": 20},
            }
            for suite in SUITES
        ],
        "swe_decision_metric": "official SWE-bench_Verified FAIL_TO_PASS resolved/40; empty patches are failures",
        "lcb_decision_metric": "livecodebench_hard code_execution pass@1/53",
    }
    write_json(output_dir / "plan.json", plan)
    (output_dir / "command.txt").write_text(" ".join(plan["runner_command"]) + "\n")
    return plan


def process_group_members(pgid: int) -> list[int]:
    result = subprocess.run(["ps", "-eo", "pid=,pgid="], text=True, capture_output=True, check=True)
    members = []
    for row in result.stdout.splitlines():
        fields = row.split()
        if len(fields) == 2 and int(fields[1]) == pgid:
            members.append(int(fields[0]))
    return sorted(members)


def cleanup_owned(
    process: subprocess.Popen[str], pgid: int, suite: dict[str, Any]
) -> dict[str, Any]:
    before = process_group_members(pgid)
    signals: list[str] = []
    if process.poll() is None:
        signals.append("SIGTERM")
        os.killpg(pgid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            signals.append("SIGKILL")
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=30)
    after_term = process_group_members(pgid)
    if after_term:
        signals.append("SIGKILL-descendants")
        os.killpg(pgid, signal.SIGKILL)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and process_group_members(pgid):
            time.sleep(0.25)
    after_kill = process_group_members(pgid)
    closed = port_free(suite["port"])
    if process.poll() is None or after_kill or not closed:
        raise RuntimeError(
            f"owned {suite['name']} process-group cleanup failed: "
            f"leader={process.poll()} members={after_kill} port_free={closed}"
        )
    return {
        "pgid": pgid,
        "members_before": before,
        "members_after_term": after_term,
        "members_after_kill": after_kill,
        "signals": signals,
        "port_free": closed,
    }


def target_runtime_maps(
    pid: int, static: dict[str, Any], proc_root: Path = Path("/proc")
) -> dict[str, Any]:
    expected = {Path(row["path"]).resolve(strict=True): row for row in static["runtime_artifacts"]}
    mapped: dict[Path, dict[str, Any]] = {}
    for row in (proc_root / str(pid) / "maps").read_text().splitlines():
        fields = row.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path_text = fields[5]
        if path_text.endswith(" (deleted)"):
            raise RuntimeError(f"owned sidecar maps deleted runtime: {path_text}")
        path = Path(path_text)
        relevant = (
            path.name.startswith(("libllama", "libggml", "libmtmd", "libgomp", "libomp"))
            and ".so" in path.name
        )
        if not relevant:
            continue
        resolved = path.resolve(strict=True)
        if resolved not in expected:
            raise RuntimeError(f"owned sidecar maps unpinned runtime: {resolved}")
        identity = expected[resolved]
        stat = resolved.stat()
        if (stat.st_dev, stat.st_ino, stat.st_size) != (
            identity["device"],
            identity["inode"],
            identity["bytes"],
        ):
            raise RuntimeError(f"mapped runtime identity changed: {resolved}")
        if sha256(resolved) != identity["sha256"]:
            raise RuntimeError(f"mapped runtime content changed: {resolved}")
        mapped[resolved] = identity
    missing = set(expected) - set(mapped)
    if missing:
        raise RuntimeError(
            f"owned sidecar is missing pinned runtime maps: {sorted(map(str, missing))}"
        )
    return {"pid": pid, "mapped": [mapped[path] for path in sorted(mapped)]}


def target_executable_identity(
    pid: int, static: dict[str, Any], proc_root: Path = Path("/proc")
) -> dict[str, Any]:
    link = proc_root / str(pid) / "exe"
    if os.readlink(link).endswith(" (deleted)"):
        raise RuntimeError("owned sidecar executable was deleted after launch")
    resolved = link.resolve(strict=True)
    identity = static["binary"]
    if resolved != Path(identity["path"]).resolve(strict=True):
        raise RuntimeError("owned sidecar executable is not frozen v8")
    stat = resolved.stat()
    if (stat.st_dev, stat.st_ino, stat.st_size) != (
        identity["device"],
        identity["inode"],
        identity["bytes"],
    ):
        raise RuntimeError("owned sidecar executable identity changed")
    if sha256(resolved) != identity["sha256"]:
        raise RuntimeError("owned sidecar executable content changed")
    return identity


def wait_ready(
    process: subprocess.Popen[str],
    suite: dict[str, Any],
    static: dict[str, Any],
    timeout_s: int = 900,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"sidecar exited before readiness on {suite['port']}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{suite['port']}/health", timeout=5
            ) as response:  # noqa: S310
                if response.status == 200:
                    if not listener_owned(process.pid, suite["port"]):
                        raise RuntimeError("owned sidecar does not own its listener")
                    return {
                        "pid": process.pid,
                        "listener_owned": True,
                        "executable": target_executable_identity(process.pid, static),
                        "runtime_maps": target_runtime_maps(process.pid, static),
                    }
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise RuntimeError(f"sidecar readiness timed out on {suite['port']}")


def evaluator_argv(suite: dict[str, Any], output_dir: Path) -> list[str]:
    return [
        str(UV),
        "run",
        "--offline",
        "--locked",
        str(EVALUATOR_PYTHON),
        str(RAW_EVALUATOR),
        "--port",
        str(suite["port"]),
        "--host",
        "127.0.0.1",
        "--suites",
        suite["external_name"],
        "--n",
        str(len(suite["ids"])),
        "--limit",
        str(len(suite["ids"])),
        "--seed",
        "42",
        "--max-tokens",
        str(suite["max_tokens"]),
        "--repeats",
        "1",
        "--concurrency",
        "1",
        "--temperature",
        "0.6",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--no-enable-thinking",
        "--endpoint",
        "chat",
        "--arm",
        "laguna_q4_cpu_v8",
        "--binary",
        str(BINARY),
        "--models",
        str(MODEL),
        "--questions-in",
        str(suite["questions"]),
        "--per-question-out",
        str(output_dir / f"{suite['name']}.per_question.jsonl"),
        "--output",
        str(output_dir / f"{suite['name']}.results.json"),
    ]


def validate_raw_artifacts(output_dir: Path, suite: dict[str, Any]) -> dict[str, Any]:
    result_path = output_dir / f"{suite['name']}.results.json"
    row_path = output_dir / f"{suite['name']}.per_question.jsonl"
    try:
        payload = json.loads(result_path.read_text())
        rows = [json.loads(line) for line in row_path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{suite['name']} raw artifacts unreadable: {exc}") from exc
    summaries = payload.get("suites")
    if not isinstance(summaries, list) or len(summaries) != 1:
        raise RuntimeError(f"{suite['name']} raw suite cardinality mismatch")
    summary = summaries[0]
    expected_ids = list(suite["ids"])
    row_ids = [row.get("id") for row in rows]
    row_correct = [row.get("correct") for row in rows]
    if (
        summary.get("suite") != suite["external_name"]
        or summary.get("n") != len(expected_ids)
        or summary.get("n_questions") != len(expected_ids)
        or summary.get("errors") != 0
        or row_ids != expected_ids
        or any(
            row.get("suite") != suite["external_name"] or row.get("request_error") for row in rows
        )
        or not all(isinstance(value, bool) for value in row_correct)
        or summary.get("correct") != sum(row_correct)
        or summary.get("accuracy") != sum(row_correct) / len(expected_ids)
    ):
        raise RuntimeError(f"{suite['name']} raw denominator, IDs, or harness errors invalid")
    return {
        "result_path": str(result_path),
        "result_sha256": sha256(result_path),
        "per_question_path": str(row_path),
        "per_question_sha256": sha256(row_path),
        "count": len(rows),
        "accuracy": sum(row_correct) / len(expected_ids),
        "correct": sum(row_correct),
        "errors": 0,
    }


def run_raw_suite(
    output_dir: Path, suite: dict[str, Any], static: dict[str, Any]
) -> dict[str, Any]:
    gate = execution_gates()
    if not gate["passed"]:
        raise RuntimeError(f"pre-{suite['name']} execution gates failed: {gate['failures']}")
    verify_model_identity()
    stderr = (output_dir / f"{suite['name']}.server.stderr").open("w")
    with OWNED_SIDECARS_LOCK:
        if OWNED_SIDECARS:
            raise RuntimeError("refusing a second owned candidate")
        process = subprocess.Popen(
            server_argv(suite), env=clean_env(), stderr=stderr, stdout=subprocess.DEVNULL,
            text=True, start_new_session=True,
        )
        pgid = process.pid
        OWNED_SIDECARS[process.pid] = {"pgid": pgid, "port": suite["port"]}
    write_json(
        output_dir / f"{suite['name']}.owned_process.json",
        {"pid": process.pid, "pgid": pgid, "argv": server_argv(suite), "prelaunch_gates": gate},
    )
    cleanup: dict[str, Any] | None = None
    try:
        runtime = wait_ready(process, suite, static)
        runtime["model_postready"] = verify_model_identity()
        command = evaluator_argv(suite, output_dir)
        try:
            result = run_owned_command(
                command, cwd=RESEARCH_ROOT, env=clean_env(),
                timeout_s=RAW_EVALUATOR_TIMEOUT_S, label=f"{suite['name']} evaluator",
            )
        except RuntimeError:
            raise
        (output_dir / f"{suite['name']}.evaluator.stdout").write_text(result.stdout)
        (output_dir / f"{suite['name']}.evaluator.stderr").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"{suite['name']} evaluator exited {result.returncode}")
        artifacts = validate_raw_artifacts(output_dir, suite)
        runtime["model_postrun"] = verify_model_identity()
        return {
            "suite": suite["name"],
            "evaluator_argv": command,
            "runtime_identity": runtime,
            "raw_artifacts": artifacts,
        }
    finally:
        with OWNED_SIDECARS_LOCK:
            cleanup = cleanup_owned(process, pgid, suite)
            OWNED_SIDECARS.pop(process.pid, None)
        stderr.close()
        write_json(output_dir / f"{suite['name']}.cleanup.json", cleanup)


def validate_predictions(path: Path) -> dict[str, Any]:
    try:
        predictions = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SWE predictions unreadable: {exc}") from exc
    if not isinstance(predictions, list) or not all(isinstance(row, dict) for row in predictions):
        raise RuntimeError("SWE predictions must be a JSON list of objects")
    ids = [row.get("instance_id") for row in predictions]
    if ids != list(SWE_IDS):
        raise RuntimeError("SWE predictions do not preserve all 40 pinned IDs in order")
    if any(set(row) != {"instance_id", "model_name_or_path", "model_patch"} for row in predictions):
        raise RuntimeError("SWE prediction schema changed")
    if any(
        row["model_name_or_path"] != "laguna_q4_cpu_v8" or not isinstance(row["model_patch"], str)
        for row in predictions
    ):
        raise RuntimeError("SWE prediction model label or patch type changed")
    empty = [row["instance_id"] for row in predictions if not str(row["model_patch"]).strip()]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "count": 40,
        "empty_patch_ids": empty,
        "empty_patch_count": len(empty),
    }


def official_swe_argv(output_dir: Path, predictions: Path) -> list[str]:
    return [
        str(TASKSET),
        "-c",
        "184-191",
        str(SWEBENCH_PYTHON),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        "princeton-nlp/SWE-bench_Verified",
        "--predictions_path",
        str(predictions),
        "--instance_ids",
        *SWE_IDS,
        "--max_workers",
        "8",
        "--cache_level",
        "env",
        "--run_id",
        OFFICIAL_SWE_RUN_ID,
        "--report_dir",
        str(output_dir / "swe_official_logs"),
    ]


def validate_official_swe_report(path: Path, prediction_evidence: dict[str, Any]) -> dict[str, Any]:
    report = read_json(path)
    resolved = report.get("resolved_ids")
    unresolved = report.get("unresolved_ids")
    completed = report.get("completed_ids")
    empty = report.get("empty_patch_ids")
    submitted = report.get("submitted_ids")
    errors = report.get("error_ids")
    incomplete = report.get("incomplete_ids")
    lists = (resolved, unresolved, completed, empty, submitted, errors, incomplete)
    if not all(
        isinstance(value, list) and all(isinstance(item, str) for item in value) for value in lists
    ):
        raise RuntimeError("official SWE report list schema is incomplete")
    if any(len(value) != len(set(value)) for value in lists):
        raise RuntimeError("official SWE report contains duplicate IDs")
    if (
        report.get("schema_version") != 2
        or report.get("total_instances") != 40
        or report.get("submitted_instances") != 40
        or len(submitted) != 40
        or set(submitted) != set(SWE_IDS)
        or report.get("error_instances") != 0
        or errors
        or incomplete
        or set(empty) != set(prediction_evidence["empty_patch_ids"])
        or set(completed) | set(empty) != set(SWE_IDS)
        or set(completed) & set(empty)
        or set(resolved) | set(unresolved) != set(completed)
        or set(resolved) & set(unresolved)
        or report.get("completed_instances") != len(completed)
        or report.get("resolved_instances") != len(resolved)
        or report.get("unresolved_instances") != len(unresolved)
        or report.get("empty_patch_instances") != len(empty)
    ):
        raise RuntimeError("official SWE report has denominator drift or harness errors")
    return {
        "report_path": str(path),
        "report_sha256": sha256(path),
        "resolved": len(resolved),
        "denominator": 40,
        "score": len(resolved) / 40,
        "unresolved_nonempty": len(unresolved),
        "empty_patch_failures": len(empty),
        "harness_errors": 0,
        "resolved_ids": resolved,
    }


def run_official_swe(output_dir: Path) -> dict[str, Any]:
    raw = output_dir / "swe_oracle.per_question.jsonl"
    predictions = output_dir / "swe_predictions.json"
    converter_argv = [
        str(EVALUATOR_PYTHON),
        str(CONVERTER),
        str(raw),
        "laguna_q4_cpu_v8",
        str(predictions),
    ]
    try:
        converted = run_owned_command(
            converter_argv, cwd=RESEARCH_ROOT, env=clean_env(),
            timeout_s=CONVERTER_TIMEOUT_S, label="SWE converter",
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("SWE converter timed out") from exc
    (output_dir / "swe_converter.stdout").write_text(converted.stdout)
    (output_dir / "swe_converter.stderr").write_text(converted.stderr)
    if converted.returncode:
        raise RuntimeError(f"SWE converter exited {converted.returncode}")
    prediction_evidence = validate_predictions(predictions)
    command = official_swe_argv(output_dir, predictions)
    official_env = clean_env()
    official_env["HF_HOME"] = "/mnt/raid0/llm/cache/huggingface"
    docker_images_before = subprocess.run(
        [str(DOCKER), "images", "--no-trunc", "--format", "{{.Repository}}@{{.ID}}"],
        text=True, capture_output=True, check=True, env=official_env,
    ).stdout
    docker_containers_before = subprocess.run(
        [str(DOCKER), "ps", "-a", "--no-trunc", "--format", "{{.ID}} {{.State}} {{.Names}}"],
        text=True, capture_output=True, check=True, env=official_env,
    ).stdout
    result: subprocess.CompletedProcess[str] | None = None
    execution_error: Exception | None = None
    docker_images_after = ""
    docker_containers_after = ""
    docker_containers_final = ""
    inspections: dict[str, dict[str, Any]] = {}
    removed: list[str] = []
    residual: list[str] = []
    cleanup_failed: list[str] = []
    operations: list[dict[str, Any]] = []
    postflight_errors: list[str] = []
    try:
        result = run_owned_command(
            command, cwd=output_dir, env=official_env,
            timeout_s=OFFICIAL_SWE_TIMEOUT_S, label="official SWE harness",
        )
        (output_dir / "swe_official.stdout").write_text(result.stdout)
        (output_dir / "swe_official.stderr").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"official SWE harness exited {result.returncode}")
    except Exception as exc:
        execution_error = exc
    finally:
        terminal: dict[str, Any] = {}
        try:
            images_after_op = docker_operation(
                [str(DOCKER), "images", "--no-trunc", "--format", "{{.Repository}}@{{.ID}}"],
                official_env,
            )
            operations.append(images_after_op)
            docker_images_after = str(images_after_op["stdout"])
            if images_after_op["error"]:
                postflight_errors.append("post images: " + str(images_after_op["error"]))

            containers_after_op = docker_operation(
                [str(DOCKER), "ps", "-a", "--no-trunc", "--format", "{{.ID}} {{.State}} {{.Names}}"],
                official_env,
            )
            operations.append(containers_after_op)
            docker_containers_after = str(containers_after_op["stdout"])
            if containers_after_op["error"]:
                postflight_errors.append("post ps: " + str(containers_after_op["error"]))
                new_ids: list[str] = []
            else:
                try:
                    new_ids = sorted(
                        set(parse_docker_container_rows(docker_containers_after))
                        - set(parse_docker_container_rows(docker_containers_before))
                    )
                except Exception as exc:
                    postflight_errors.append("post ps parse: " + repr(exc))
                    new_ids = []

            for container_id in new_ids:
                inspect_op = docker_operation([str(DOCKER), "inspect", container_id], official_env)
                inspect_op["container_id"] = container_id
                operations.append(inspect_op)
                if inspect_op["error"]:
                    postflight_errors.append(
                        f"inspect {container_id}: {inspect_op['error']}"
                    )
                    residual.append(container_id)
                    continue
                try:
                    payload = json.loads(str(inspect_op["stdout"]))
                    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
                        raise ValueError("inspect output is not a non-empty object list")
                    inspections[container_id] = payload[0]
                except Exception as exc:
                    inspect_op["error"] = "JSON decode/schema: " + repr(exc)
                    postflight_errors.append(f"inspect {container_id}: {inspect_op['error']}")
                    residual.append(container_id)

            owned, unproven = classify_new_campaign_containers(
                docker_containers_before,
                docker_containers_after,
                inspections,
                OFFICIAL_SWE_RUN_ID,
            )
            residual = sorted(set(residual) | set(unproven))
            for container_id in owned:
                remove_op = docker_operation([str(DOCKER), "rm", "-f", container_id], official_env)
                remove_op["container_id"] = container_id
                operations.append(remove_op)
                if remove_op["error"]:
                    cleanup_failed.append(container_id)
                    postflight_errors.append(f"rm {container_id}: {remove_op['error']}")
                else:
                    removed.append(container_id)

            final_ps_op = docker_operation(
                [str(DOCKER), "ps", "-a", "--no-trunc", "--format", "{{.ID}} {{.State}} {{.Names}}"],
                official_env,
            )
            operations.append(final_ps_op)
            docker_containers_final = str(final_ps_op["stdout"])
            if final_ps_op["error"]:
                postflight_errors.append("final ps: " + str(final_ps_op["error"]))
            else:
                try:
                    final_ids = set(parse_docker_container_rows(docker_containers_final))
                    cleanup_failed.extend(
                        container_id
                        for container_id in removed
                        if container_id in final_ids and container_id not in cleanup_failed
                    )
                except Exception as exc:
                    postflight_errors.append("final ps parse: " + repr(exc))
        except Exception as exc:
            postflight_errors.append("postflight internal: " + repr(exc))
        finally:
            terminal = {
                "run_id": OFFICIAL_SWE_RUN_ID,
                "execution_error": repr(execution_error) if execution_error else "",
                "postflight_errors": postflight_errors,
                "images_before": docker_images_before.splitlines(),
                "images_after": docker_images_after.splitlines(),
                "containers_before": docker_containers_before.splitlines(),
                "containers_after": docker_containers_after.splitlines(),
                "containers_final": docker_containers_final.splitlines(),
                "new_container_inspections": inspections,
                "docker_operations": operations,
                "removed_owned_ids": removed,
                "residual_unproven_ids": residual,
                "cleanup_failed_ids": cleanup_failed,
            }
            write_json_atomic(output_dir / "swe_docker_terminal.json", terminal)
        if residual or cleanup_failed or postflight_errors:
            original_error = repr(execution_error) if execution_error else "none"
            execution_error = RuntimeError(
                "official SWE Docker postflight failed: "
                f"execution={original_error} unproven={residual} "
                f"cleanup_failed={cleanup_failed} errors={postflight_errors}"
            )
    if execution_error is not None:
        raise RuntimeError(f"official SWE failed with terminal evidence: {execution_error}") from execution_error
    if docker_images_after != docker_images_before:
        raise RuntimeError("Docker image ID set changed during official SWE scoring")
    validate_docker_container_transition(docker_containers_before, docker_containers_final)
    report = output_dir / f"laguna_q4_cpu_v8.{OFFICIAL_SWE_RUN_ID}.json"
    return {
        "converter_argv": converter_argv,
        "predictions": prediction_evidence,
        "official_argv": command,
        "docker_image_ids_before": docker_images_before.splitlines(),
        "docker_image_ids_after": docker_images_after.splitlines(),
        "docker_containers_before": docker_containers_before.splitlines(),
        "docker_containers_after": docker_containers_after.splitlines(),
        "docker_containers_final": docker_containers_final.splitlines(),
        "decision_score": validate_official_swe_report(report, prediction_evidence),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DRY_RUN_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify-model-sha", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        plan = prepare(args.output_dir, args.execute, raw_argv, args.verify_model_sha)
        if not args.execute:
            print(
                json.dumps(
                    {
                        "status": "prepared_no_inference",
                        "output_dir": str(args.output_dir),
                        "gates": plan["gates"],
                    }
                )
            )
            return 0
        if not args.verify_model_sha:
            raise RuntimeError("execution requires --verify-model-sha")
        if not plan["gates"]["passed"]:
            raise RuntimeError("execution refused: " + "; ".join(plan["gates"]["failures"]))
        static = plan["static_identity"]
        monitor = CampaignMonitor(args.output_dir)
        monitor.start()
        try:
            swe_raw = run_raw_suite(args.output_dir, SUITES[0], static)
            monitor.sample("after-swe-raw")
            swe_official = run_official_swe(args.output_dir)
            monitor.sample("after-official-swe")
            lcb = run_raw_suite(args.output_dir, SUITES[1], static)
        finally:
            monitor.stop()
        summary = {
            "schema": "epyc.laguna_q4_cpu_v8.summary.v2",
            "status": "ok",
            "official_swe": swe_official["decision_score"],
            "lcb_code_execution": lcb["raw_artifacts"],
            "raw_swe_generation": swe_raw["raw_artifacts"],
            "raw_swe_accuracy_decision_use": "forbidden",
            "swe_pipeline": swe_official,
        }
        write_json(args.output_dir / "summary.json", summary)
        return 0
    except Exception as exc:
        if args.output_dir.exists() and args.execute:
            write_json(
                args.output_dir / "summary.json",
                {
                    "schema": "epyc.laguna_q4_cpu_v8.summary.v2",
                    "status": "failed",
                    "error": repr(exc),
                },
            )
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
