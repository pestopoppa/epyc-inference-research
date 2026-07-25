#!/usr/bin/env python3
"""Dry-run-first P-BENCH-PREFILL-1 v7 versus v8 CPU regression matrix.

This runner deliberately owns *pairing and attestation*, not llama-bench command
construction.  Every llama-bench invocation goes through bench_canonical.sh,
which in turn consumes scripts/lib/canonical_recipe.py.  The default mode writes
an immutable preparation manifest only; --execute is required before inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parents[1]
CANONICAL_WRAPPER = SCRIPT_DIR / "bench_canonical.sh"
CANONICAL_RECIPE = RESEARCH_ROOT / "scripts/lib/canonical_recipe.py"
INSTRUMENT_ERAS = Path("/mnt/raid0/llm/epyc-orchestrator/orchestration/instrument_eras.yaml")
PRODUCTION_ROOT = Path("/mnt/raid0/llm/llama.cpp")
CANDIDATE_ROOT = Path("/mnt/raid0/llm/llama.cpp-experimental")
PRODUCTION_BINARY = PRODUCTION_ROOT / "build/bin/llama-bench"
CANDIDATE_BINARY = CANDIDATE_ROOT / "build-v8-cpu/bin/llama-bench"
PRODUCTION_BRANCH = "production-consolidated-v7"
CANDIDATE_BRANCH = "experimental-v8-refresh-20260724"
PRODUCTION_HEAD = "6ad45fa3ff6718c07c000061dbc6e29c1771f6e3"
CANDIDATE_HEAD = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
Q8_WAIVER_PATH = Path(
    "/mnt/raid0/llm/epyc-root/artifacts/operator/waive_q8_cpu_prefill_v8_20260725.json"
)
Q8_WAIVER_SHA256 = "fcd52b61610fcc2782e11f41ffac359343233924805f83d872eeceffbb7522d7"
Q8_WAIVER_SCHEMA = "epyc.cpu_prefill_v8.operator_waiver.v1"
CPU_EXTRA = ("-dev", "none", "-ngl", "0", "--no-op-offload", "1")
# JSON preserves all ten samples; markdown on stderr preserves the mandatory
# result-emitted build witness required by the 2026-07-24 identity erratum.
OUTPUT_EXTRA = ("-o", "json", "-oe", "md")
REPS = 10
PREFILL = 2048
BUILD_LINE_RE = re.compile(r"^\s*build:\s*([0-9a-f]+)\s*\((\d+)\)\s*$", re.MULTILINE)
SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$")
LLAMA_LIBRARY_RE = re.compile(
    r"^\s*(?P<soname>(?:libllama|libggml)\S*\.so(?:\.[0-9.]+)?)\s*=>\s*(?P<target>\S+)",
    re.MULTILINE,
)
OPENMP_LIBRARY_RE = re.compile(
    r"^\s*(?P<soname>lib(?:gomp|omp)\S*\.so(?:\.[0-9.]+)?)\s*=>\s*(?P<target>\S+)",
    re.MULTILINE,
)
WRAPPER_ENV_LINE_RE = re.compile(r"^Env:\s*(?P<assignments>.+?)\s*$", re.MULTILINE)
MAX_UPTIME_SECONDS = 7 * 24 * 60 * 60
CANONICAL_PARENT_ENV = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/rocm/bin",
    "HOME": "/home/node",
    "TMPDIR": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "TZ": "UTC",
    "PYTHONNOUSERSITE": "1",
}
OPTIONAL_SUBPROCESS_ENV_KEYS = {"LD_LIBRARY_PATH"}
LLVM20_LIBDIR = "/usr/lib/llvm-20/lib"
REQUIRED_WRAPPER_OMP_ENV = {
    "OMP_PROC_BIND": "spread",
    "OMP_PLACES": "cores",
    "OMP_WAIT_POLICY": "active",
    "OMP_DYNAMIC": "false",
}
BASE_MEMINFO_KEYS = (
    "MemTotal",
    "MemFree",
    "MemAvailable",
    "Buffers",
    "Cached",
)
THP_HUGEPAGE_MEMINFO_UNITS = {
    "AnonHugePages": "kB",
    "ShmemHugePages": "kB",
    "ShmemPmdMapped": "kB",
    "FileHugePages": "kB",
    "FilePmdMapped": "kB",
    "HugePages_Total": "count",
    "HugePages_Free": "count",
    "HugePages_Rsvd": "count",
    "HugePages_Surp": "count",
    "Hugepagesize": "kB",
    "Hugetlb": "kB",
    "DirectMap2M": "kB",
    "DirectMap1G": "kB",
}
HUGEPAGE_POOL_FILES = (
    "nr_hugepages",
    "free_hugepages",
    "resv_hugepages",
    "surplus_hugepages",
)
HUGEPAGE_POOL_RE = re.compile(r"^hugepages-(\d+)kB$")
MONITOR_INTERVAL_S = 1.0
MIN_MONITOR_SAMPLES = 2
CONTENTION_ACCOUNTING = "sustained-window-v1"
CONFIGURED_CPU_COUNT = 96
MIN_TARGET_CORE_EQUIVALENTS = 0.75 * CONFIGURED_CPU_COUNT
MIN_SUSTAINED_WINDOW_SECONDS = 10.0
MIN_SIGNED_EXTERNAL_CORE_EQUIVALENTS = -1.0
MAX_EXTERNAL_CORE_EQUIVALENTS = 4.0
MAX_HOST_ATTESTATION_AGE_S = 300
CLOCK_TICKS = os.sysconf("SC_CLK_TCK")
if not isinstance(CLOCK_TICKS, int) or CLOCK_TICKS <= 0:
    raise RuntimeError(f"invalid system clock tick rate: {CLOCK_TICKS!r}")
IQK_ATTESTATION_SCHEMA = "epyc.iqk_real_model_correctness.attestation.v1"
HOST_ATTESTATION_SCHEMA = "epyc.cpu_prefill_v8.host_attestation.v1"
IQK_ATTESTATION_MODELS = {"qwen_next_iq2", "glm52_ud_iq2", "hy3_iq1_m"}
IQK_MODEL_BINDINGS = {
    "qwen_next_iq2": "qwen_next_iq2",
    "glm52_ud_iq2": "glm_iq2",
    "hy3_iq1_m": "hy3_iq1",
}
IQK_TASKS = {"exact_json", "math_37_plus_58", "needle", "routing_tradeoffs"}
IQK_NUMERICAL_SCOPE = "real-model completion token logprobs and server stderr only"
IQK_NATIVE_TYPE_CODES = {16, 17, 18, 21, 22}
EXPECTED_NATIVE_TYPES_BY_MODEL = {
    "qwen_next_iq2": {21, 22},
    "glm52_ud_iq2": {16, 18, 22},
    "hy3_iq1_m": {16, 18},
}
IQK_RUNNER_SHA256 = "a7d4c252c1d9083b7ffe6397eefdff2555b1f1f6abcfbfcbfd6ba136ab84f727"
IQK_SERVER_SHA256 = "c6accc0d5bf935e85c56a16a5f837a70774005876d2326628702325e73b6704b"
IQK_SERVER_LIBRARY_SHA256 = {
    "libggml-base.so.0.16.0": "f47cc4ad6ab59ea39de7e5fd302f79ba62626cdf84b48332144dfdfa34af0cde",
    "libggml-cpu.so.0.16.0": "26c3c98a289764c11752751faf380b592ab07b00191d289637b4079d2e1a5e90",
    "libggml.so.0.16.0": "ed67a5d9340c256abdcd9b2729871d3dbab6f979031e752bd4c92c2f73a5dacd",
    "libllama-common.so.0.0.10107": "c80f532417d58c52aee40bb25734661b5c7a3f74763c17d0c57921ad83d72bca",
    "libllama-server-impl.so": "ae40403742641096519c0c336e8d2d4bb8c54a8d7404004d940b42d970eb5d31",
    "libllama.so.0.0.10107": "f1e5a2e0976fa4f96d9f78775634b7e4869af8f6549eed384ebc94435eebee8e",
}

# These are release artifacts, not source-tree claims.  The shared objects are
# precisely the objects resolved by ldd for each frozen llama-bench binary.
RELEASE_ARTIFACTS = {
    "production": {
        "llama_bench_sha256": "6028e410732601b70da470a6880a2d2e4f24c17def48678b4ed1550dfe606541",
        "libraries": {
            "libggml-base.so.0.16.0": "c1b6649b32a8601756d17d11d9ebbb8352e2ffd7be303fd8167677dd2170491c",
            "libggml-cpu.so.0.16.0": "b36a300c027f9bb667812fc280752be9f234d65bd54d91b24ac1d592e91d243d",
            "libggml.so.0.16.0": "594cb1e66106b6d3e198e4dd1e2cc25088aeb059cd4f7b2797853279127613b8",
            "libllama-bench-impl.so": "6cb10daa5c0a9b82bc370559995aa9e6dbbe2ba5092953481de2db0c4834b402",
            "libllama-common.so.0.0.10098": "306068cd63312b28b48ca59d2b4348b2b450edc6780d109c9607d9af8910c401",
            "libllama.so.0.0.10098": "508905d58ba49ff0e286ebc76a10fec4c3933568855fe56faedcb0bee8b72cc2",
        },
    },
    "candidate": {
        "llama_bench_sha256": "8b601a282dd3fb49b4791ccbbd2ab034cd8a723a200db147b22fe05f593b7a98",
        "libraries": {
            "libggml-base.so.0.16.0": "f47cc4ad6ab59ea39de7e5fd302f79ba62626cdf84b48332144dfdfa34af0cde",
            "libggml-cpu.so.0.16.0": "26c3c98a289764c11752751faf380b592ab07b00191d289637b4079d2e1a5e90",
            "libggml.so.0.16.0": "ed67a5d9340c256abdcd9b2729871d3dbab6f979031e752bd4c92c2f73a5dacd",
            "libllama-bench-impl.so": "20aea71e0de1ea50552b67b86f72a4f629e5004ede8a9e5ded43a0ef18470811",
            "libllama-common.so.0.0.10107": "c80f532417d58c52aee40bb25734661b5c7a3f74763c17d0c57921ad83d72bca",
            "libllama.so.0.0.10107": "f1e5a2e0976fa4f96d9f78775634b7e4869af8f6549eed384ebc94435eebee8e",
        },
    },
}
LLVM20_OPENMP_IDENTITY = {
    "reported_target": "/usr/lib/llvm-20/lib/libgomp.so.1",
    "resolved_target": "/usr/lib/llvm-20/lib/libomp.so.5",
    "sha256": "98b1f8225260f138243e8e3e7578b83802e998a240f841dc1944a908bf1aee70",
}


@dataclass(frozen=True)
class ModelCell:
    name: str
    path: Path
    iq: bool


MODELS = (
    ModelCell("gemma_orig_q4", Path("/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf"), False),
    ModelCell("qwen_next_iq2", Path("/mnt/raid0/llm/models/Qwen3-Next-80B-A3B-Instruct.i1-IQ2_M.gguf"), True),
    ModelCell("glm_iq2", Path("/mnt/raid0/llm/models/GLM-5.2-UD-IQ2_M/UD-IQ2_M/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf"), True),
    ModelCell("hy3_iq1", Path("/mnt/raid0/llm/models/hy3-angelslim/Hy3-IQ1_M-mtp.gguf"), True),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_parent_environment() -> dict[str, str]:
    """Return the exact environment inherited by bench_canonical.sh."""
    return dict(CANONICAL_PARENT_ENV)


def environment_identity(environment: dict[str, str]) -> dict[str, Any]:
    encoded = json.dumps(
        environment,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "environment": environment,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def parent_environment_identity() -> dict[str, Any]:
    return environment_identity(canonical_parent_environment())


def exact_subprocess_environment(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return one exact allowlisted environment for every child process."""
    environment = canonical_parent_environment() if env is None else dict(env)
    required_keys = set(CANONICAL_PARENT_ENV)
    actual_keys = set(environment)
    extras = actual_keys - required_keys
    missing = required_keys - actual_keys
    if missing or not extras.issubset(OPTIONAL_SUBPROCESS_ENV_KEYS):
        raise RuntimeError(
            "subprocess environment key set violates the exact allowlist: "
            f"missing={sorted(missing)} extras={sorted(extras)}"
        )
    drift = {
        key: {"actual": environment.get(key), "expected": value}
        for key, value in CANONICAL_PARENT_ENV.items()
        if environment.get(key) != value
    }
    if drift:
        raise RuntimeError(f"subprocess environment base values drifted: {drift}")
    library_path = environment.get("LD_LIBRARY_PATH")
    if library_path is not None:
        entries = library_path.split(":")
        if not entries or any(
            not entry or "\0" in entry or not Path(entry).is_absolute()
            for entry in entries
        ):
            raise RuntimeError(
                f"subprocess LD_LIBRARY_PATH is not an exact absolute-path list: {library_path!r}"
            )
    return environment


def environment_assignments(env: dict[str, str] | None = None) -> list[str]:
    environment = exact_subprocess_environment(env)
    return [f"{key}={value}" for key, value in environment.items()]


def run(
    argv: list[str],
    *,
    input_text: str | None = None,
    check: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    child_environment = exact_subprocess_environment(env)
    result = subprocess.run(
        argv,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        env=child_environment,
    )
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {shlex.join(argv)}\n{result.stderr}")
    return result


def git_value(root: Path, *args: str) -> str:
    return run(
        ["git", "-C", str(root), *args],
        check=True,
        env=canonical_parent_environment(),
    ).stdout.strip()


def arm_spec(name: str) -> dict[str, Any]:
    if name == "production":
        root, binary, branch, head = PRODUCTION_ROOT, PRODUCTION_BINARY, PRODUCTION_BRANCH, PRODUCTION_HEAD
    elif name == "candidate":
        root, binary, branch, head = CANDIDATE_ROOT, CANDIDATE_BINARY, CANDIDATE_BRANCH, CANDIDATE_HEAD
    else:
        raise ValueError(name)
    return {"name": name, "source_root": str(root), "binary": str(binary), "library_path": str(binary.parent), "expected_branch": branch, "expected_head": head}


def wrapper_argv(cell: ModelCell, arm: dict[str, Any], iqk: int, *, dry_run: bool) -> list[str]:
    argv = ["bash", str(CANONICAL_WRAPPER), "-m", str(cell.path), "-p", str(PREFILL), "-n", "0", "-r", str(REPS),
            "--binary", arm["binary"], "--source-root", arm["source_root"], "--library-path", arm["library_path"],
            "--ggml-iqk", str(iqk)]
    if dry_run:
        argv.append("--dry-run")
    return [*argv, "--", *CPU_EXTRA, *OUTPUT_EXTRA]


def build_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for model in MODELS:
        for iqk in ((0, 1) if model.iq else (1,)):
            for metric, prompt, gen in (("tg128", 0, 128), ("pp2048", PREFILL, 0)):
                # tg128 is P-BENCH-1 paired evidence.  The wrapper remains canonical.
                pair_id = f"{model.name}-{metric}-iqk{iqk}"
                for arm in ("production", "candidate"):
                    cells.append({"id": f"{pair_id}-{arm}", "pair_id": pair_id,
                                  "kernel_arm": arm, "model": model.name,
                                  "model_path": str(model.path), "iq": model.iq, "iqk": iqk,
                                  "metric": metric, "n_prompt": prompt, "n_gen": gen, "reps": REPS})
    if len(cells) != 28:
        raise AssertionError(f"matrix drift: expected 28 cells, got {len(cells)}")
    if len({cell["id"] for cell in cells}) != 28:
        raise AssertionError("matrix cell IDs are not unique")
    pair_arms: dict[str, set[str]] = {}
    for cell in cells:
        pair_arms.setdefault(cell["pair_id"], set()).add(cell["kernel_arm"])
    if len(pair_arms) != 14 or any(arms != {"production", "candidate"} for arms in pair_arms.values()):
        raise AssertionError(f"matrix pair cardinality drift: {pair_arms}")
    return cells


def build_pairs(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(cell["pair_id"], []).append(cell)
    pairs: list[dict[str, Any]] = []
    ignored = {"id", "kernel_arm"}
    for pair_id, members in grouped.items():
        if len(members) != 2 or {item["kernel_arm"] for item in members} != {"production", "candidate"}:
            raise RuntimeError(f"pair {pair_id} does not contain exactly one run per arm")
        left = {key: value for key, value in members[0].items() if key not in ignored}
        right = {key: value for key, value in members[1].items() if key not in ignored}
        if left != right:
            raise RuntimeError(f"pair {pair_id} has inconsistent arm metadata")
        pairs.append(left)
    if len(pairs) != 14 or len({pair["pair_id"] for pair in pairs}) != 14:
        raise RuntimeError("expected 14 unique matched pairs")
    return pairs


def argv_for_cell(cell: dict[str, Any], arm: dict[str, Any], *, dry_run: bool) -> list[str]:
    model = ModelCell(cell["model"], Path(cell["model_path"]), bool(cell["iq"]))
    argv = wrapper_argv(model, arm, int(cell["iqk"]), dry_run=dry_run)
    # wrapper_argv is fixed for prefill.  Keep command construction in the wrapper,
    # but replace only profile fields the protocol pairs with decode P-BENCH-1.
    argv[argv.index("-p") + 1] = str(cell["n_prompt"])
    argv[argv.index("-n") + 1] = str(cell["n_gen"])
    return argv


def parse_wrapper_emitted_environment(raw_stderr: str) -> dict[str, str]:
    matches = WRAPPER_ENV_LINE_RE.findall(raw_stderr)
    if len(matches) != 1:
        raise RuntimeError(
            f"canonical wrapper emitted {len(matches)} environment witnesses, expected one"
        )
    environment: dict[str, str] = {}
    for assignment in shlex.split(matches[0]):
        key, separator, value = assignment.partition("=")
        if (
            not separator
            or re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None
            or key in environment
        ):
            raise RuntimeError(
                f"canonical wrapper emitted a malformed environment assignment: {assignment!r}"
            )
        environment[key] = value
    return environment


def canonical_environment_witness(
    raw_stderr: str,
    cell: dict[str, Any],
    arm: dict[str, Any],
) -> dict[str, Any]:
    emitted = parse_wrapper_emitted_environment(raw_stderr)
    expected_emitted = {
        "LD_LIBRARY_PATH": (
            f"{Path(arm['library_path']).resolve()}:{Path(LLVM20_LIBDIR).resolve()}"
        ),
        **REQUIRED_WRAPPER_OMP_ENV,
        "GGML_IQK": str(cell["iqk"]),
    }
    if emitted != expected_emitted:
        raise RuntimeError(
            "canonical wrapper environment drifted: "
            f"expected={expected_emitted} actual={emitted}"
        )
    parent = canonical_parent_environment()
    effective = {**parent, **emitted}
    required_effective = {
        **REQUIRED_WRAPPER_OMP_ENV,
        "GGML_IQK": str(cell["iqk"]),
        "LD_LIBRARY_PATH": expected_emitted["LD_LIBRARY_PATH"],
    }
    drift = {
        key: {"expected": value, "actual": effective.get(key)}
        for key, value in required_effective.items()
        if effective.get(key) != value
    }
    if drift:
        raise RuntimeError(f"effective canonical OpenMP/KMP environment drifted: {drift}")
    return {
        "parent": environment_identity(parent),
        "wrapper_emitted": environment_identity(emitted),
        "effective": environment_identity(effective),
        "required_effective_settings": required_effective,
    }


def file_identity(path: Path) -> dict[str, Any]:
    """Hash a regular file while rejecting replacement or modification races."""
    if not path.is_file():
        raise RuntimeError(f"required file missing: {path}")
    resolved = path.resolve()
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    after = resolved.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"file identity changed while hashing: {resolved}")
    return {
        "path": str(resolved),
        "device": after.st_dev,
        "inode": after.st_ino,
        "bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def instrument_era_attestation() -> dict[str, Any]:
    """Bind this run to the active CPU and evaluation instrument eras."""
    identity = file_identity(INSTRUMENT_ERAS)
    raw = INSTRUMENT_ERAS.read_text(encoding="utf-8")
    raw_sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if raw_sha256 != identity["sha256"]:
        raise RuntimeError(
            "instrument-era registry changed between identity hashing and parsing: "
            f"hashed={identity['sha256']} parsed={raw_sha256}"
        )
    try:
        document = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise RuntimeError("instrument-era registry is not valid YAML") from exc
    eras = document.get("eras") if isinstance(document, dict) else None
    if not isinstance(eras, list) or not all(isinstance(row, dict) for row in eras):
        raise RuntimeError("instrument-era registry lacks a valid eras list")

    rows: dict[str, dict[str, Any]] = {}
    for scope, expected_id in (("cpu_bench", "E6-cpu-kernel"), ("eval_quality", "E7-eval-instrument")):
        scoped = [row for row in eras if row.get("scope") == scope]
        if not scoped or scoped[-1].get("id") != expected_id:
            raise RuntimeError(
                f"instrument-era registry active {scope} row drifted: "
                f"expected={expected_id!r} actual={scoped[-1].get('id') if scoped else None!r}"
            )
        row = scoped[-1]
        if not isinstance(row.get("from"), (str, datetime)):
            raise RuntimeError(f"instrument-era registry {expected_id} lacks a valid from boundary")
        pattern = re.compile(rf"^  - id: {re.escape(expected_id)}\s*$", re.MULTILINE)
        match = pattern.search(raw)
        if match is None:
            raise RuntimeError(f"instrument-era registry cannot locate raw row {expected_id}")
        start_line = raw.count("\n", 0, match.start()) + 1
        next_row = re.compile(r"^  - id: ", re.MULTILINE).search(raw, match.end())
        end_offset = next_row.start() if next_row else len(raw)
        raw_row = raw[match.start():end_offset]
        end_line = raw.count("\n", 0, end_offset)
        rows[scope] = {
            "id": expected_id,
            "scope": scope,
            "from": row["from"].isoformat() if isinstance(row["from"], datetime) else row["from"],
            "parsed_row": row,
            "raw_row_sha256": hashlib.sha256(raw_row.encode("utf-8")).hexdigest(),
            "row_boundaries": {"start_line": start_line, "end_line": end_line},
        }
    return {"source": identity, "active": rows}


def harness_identities() -> dict[str, dict[str, Any]]:
    return {
        "runner": file_identity(Path(__file__)),
        "bench_canonical": file_identity(CANONICAL_WRAPPER),
        "canonical_recipe": file_identity(CANONICAL_RECIPE),
        "parent_environment": parent_environment_identity(),
        "instrument_eras": instrument_era_attestation(),
    }


def attestation_identities(
    paths: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    required_roles = {"host", "correctness", "coherence", "numerical_safety"}
    if set(paths) != required_roles:
        raise RuntimeError(
            f"attestation role binding drift: expected {sorted(required_roles)}, "
            f"got {sorted(paths)}"
        )
    return {role: file_identity(path) for role, path in sorted(paths.items())}


def _strict_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not strict JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be a JSON object")
    return value


def _require_exact_keys(value: dict[str, Any], required: set[str], label: str) -> None:
    missing, extras = required - set(value), set(value) - required
    if missing or extras:
        raise RuntimeError(f"{label} schema keys are invalid: missing={sorted(missing)} extras={sorted(extras)}")


def q8_waiver_attestation() -> dict[str, Any]:
    """Fail closed unless the operator-ratified Q8 campaign waiver is exact."""
    before = file_identity(Q8_WAIVER_PATH)
    if before["sha256"] != Q8_WAIVER_SHA256:
        raise RuntimeError(
            "Q8 waiver SHA256 mismatch: "
            f"expected={Q8_WAIVER_SHA256} actual={before['sha256']}"
        )
    waiver = _strict_json_object(Q8_WAIVER_PATH, "Q8 waiver")
    after = file_identity(Q8_WAIVER_PATH)
    if before != after:
        raise RuntimeError("Q8 waiver changed while validating")
    _require_exact_keys(
        waiver,
        {
            "schema", "decision", "ratified_at", "protocol", "protocol_changed",
            "candidate_head", "production_head", "runner_sha256_before_waiver_implementation",
            "scope", "reason", "consequences",
        },
        "Q8 waiver",
    )
    expected_scope = {
        "excluded_model": "qwen36_q8",
        "excluded_model_path": "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf",
        "excluded_pairs": ["qwen36_q8-tg128-iqk1", "qwen36_q8-pp2048-iqk1"],
        "excluded_arm_runs": 4,
        "remaining_matched_pairs": 14,
        "remaining_arm_runs": 28,
    }
    if (
        waiver["schema"] != Q8_WAIVER_SCHEMA
        or waiver["decision"] != "WAIVE-Q8"
        or waiver["ratified_at"] != "2026-07-25T14:04:16Z"
        or waiver["protocol"] != "P-BENCH-PREFILL-1"
        or waiver["protocol_changed"] is not False
        or waiver["candidate_head"] != CANDIDATE_HEAD
        or waiver["production_head"] != PRODUCTION_HEAD
        or waiver["runner_sha256_before_waiver_implementation"]
        != "2fb0013d2cb71b149a7429995830ac0356048582671ae83428cb1ef15ccfe024"
        or waiver["scope"] != expected_scope
        or waiver["reason"]
        != "The Qwen3.6 Q8 workload naturally sustains about 50-55 target core-equivalents and cannot satisfy the ratified 72-core eligibility floor."
    ):
        raise RuntimeError("Q8 waiver semantic binding is invalid")
    consequences = waiver["consequences"]
    required_consequences = {
        "No v8 Q8 non-regression claim may be made from this campaign.",
        "The ratified 72-core eligibility floor remains unchanged for every remaining arm.",
        "The Gemma Q4 non-IQ B4 pairs remain mandatory.",
        "All retained IQ B3 pairs remain mandatory.",
        "Pre-waiver artifacts remain ineligible and cannot be retro-certified.",
    }
    if (
        not isinstance(consequences, list)
        or len(consequences) != len(required_consequences)
        or set(consequences) != required_consequences
    ):
        raise RuntimeError("Q8 waiver consequences are invalid")
    return {
        "source": after,
        "semantic_binding": {
            "schema": waiver["schema"],
            "decision": waiver["decision"],
            "ratified_at": waiver["ratified_at"],
            "protocol": waiver["protocol"],
            "protocol_changed": waiver["protocol_changed"],
            "candidate_head": waiver["candidate_head"],
            "production_head": waiver["production_head"],
            "runner_sha256_before_waiver_implementation": waiver[
                "runner_sha256_before_waiver_implementation"
            ],
            "scope": waiver["scope"],
            "reason": waiver["reason"],
            "consequences": consequences,
        },
    }


def _require_utc_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.endswith("+00:00"):
        raise RuntimeError(f"{label} must be an ISO-8601 UTC (+00:00) timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(f"{label} is malformed") from exc
    if parsed.tzinfo != timezone.utc:
        raise RuntimeError(f"{label} timezone is not UTC")


def _require_fresh_host_timestamp(value: Any) -> None:
    _require_utc_timestamp(value, "host.created_at")
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(value)).total_seconds()
    if age < 0 or age > MAX_HOST_ATTESTATION_AGE_S:
        raise RuntimeError(f"host attestation is stale or from the future: age_seconds={age}")


def _model_binding(identity: dict[str, Any], label: str) -> dict[str, Any]:
    required = {"entry_path", "shard_count", "total_bytes", "shards"}
    if not isinstance(identity, dict) or not required.issubset(identity):
        raise RuntimeError(f"{label} model identity is malformed")
    shards = identity["shards"]
    if not isinstance(shards, list) or len(shards) != identity["shard_count"]:
        raise RuntimeError(f"{label} model shard cardinality is malformed")
    normalized_shards: list[dict[str, Any]] = []
    for index, shard in enumerate(shards):
        shard_keys = {"path", "device", "inode", "bytes", "mtime_ns", "sha256"}
        if not isinstance(shard, dict) or not shard_keys.issubset(shard):
            raise RuntimeError(f"{label} model shard {index} identity is malformed")
        if (
            isinstance(shard["bytes"], bool)
            or not isinstance(shard["bytes"], int)
            or shard["bytes"] <= 0
            or not isinstance(shard["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", shard["sha256"]) is None
        ):
            raise RuntimeError(f"{label} model shard {index} identity is invalid")
        normalized_shards.append({key: shard[key] for key in sorted(shard_keys)})
    binding = {
        "entry_path": identity["entry_path"],
        "shard_count": identity["shard_count"],
        "total_bytes": identity["total_bytes"],
        "shards": normalized_shards,
    }
    if binding["total_bytes"] != sum(item["bytes"] for item in normalized_shards):
        raise RuntimeError(f"{label} model total bytes disagree with its shards")
    return binding


def _validate_iqk_runtime_identity(
    identity: Any,
    current_models: dict[str, dict[str, Any]],
    label: str,
) -> None:
    if not isinstance(identity, dict):
        raise RuntimeError(f"{label} runtime identity is malformed")
    runner = identity.get("runner")
    if (
        not isinstance(runner, dict)
        or runner.get("sha256") != IQK_RUNNER_SHA256
    ):
        raise RuntimeError(f"{label} IQK runner identity is not pinned")
    attested_models = identity.get("models")
    if not isinstance(attested_models, dict) or set(attested_models) != set(IQK_MODEL_BINDINGS):
        raise RuntimeError(f"{label} IQK model identity set is incomplete")
    for attested_name, current_name in IQK_MODEL_BINDINGS.items():
        attested = _model_binding(
            attested_models[attested_name],
            f"{label}.{attested_name}",
        )
        current = _model_binding(
            current_models[current_name],
            f"current.{current_name}",
        )
        if attested != current:
            raise RuntimeError(
                f"{label} model identity does not match current benchmark input: "
                f"{attested_name}->{current_name}"
            )


def _validate_iqk_arm(row: Any) -> None:
    if not isinstance(row, dict):
        raise RuntimeError("IQK attestation arm is not an object")
    model = row.get("model")
    iqk = row.get("iqk")
    if model not in IQK_ATTESTATION_MODELS or isinstance(iqk, bool) or iqk not in (0, 1):
        raise RuntimeError("IQK attestation arm identity is invalid")
    if row.get("status") != "pass" or row.get("primary_error") is not None:
        raise RuntimeError(f"IQK attestation arm is not a clean pass: {model}/iqk{iqk}")
    cleanup = row.get("cleanup")
    if not isinstance(cleanup, dict) or cleanup.get("status") != "pass":
        raise RuntimeError(f"IQK attestation arm cleanup failed: {model}/iqk{iqk}")
    numerical = row.get("numerical_safety")
    if (
        not isinstance(numerical, dict)
        or numerical.get("status") != "pass"
        or numerical.get("scope") != IQK_NUMERICAL_SCOPE
        or isinstance(numerical.get("logprob_token_count"), bool)
        or not isinstance(numerical.get("logprob_token_count"), int)
        or numerical["logprob_token_count"] <= 0
    ):
        raise RuntimeError(f"IQK attestation arm numerical evidence is invalid: {model}/iqk{iqk}")
    log_evidence = row.get("iqk_log_evidence")
    if (
        not isinstance(log_evidence, dict)
        or log_evidence.get("status") != "pass"
        or log_evidence.get("iqk") != iqk
        or not isinstance(log_evidence.get("active_type_codes"), list)
    ):
        raise RuntimeError(f"IQK attestation arm activation evidence is invalid: {model}/iqk{iqk}")
    active = log_evidence["active_type_codes"]
    if iqk == 0:
        if active:
            raise RuntimeError(f"IQK=0 attestation arm reports active IQK types: {model}")
    else:
        native = log_evidence.get("native_type_codes")
        if (
            not isinstance(native, list)
            or not native
            or not EXPECTED_NATIVE_TYPES_BY_MODEL[model].issubset(set(native))
            or not set(native).issubset(IQK_NATIVE_TYPE_CODES)
            or not set(native).issubset(set(active))
        ):
            raise RuntimeError(f"IQK=1 attestation arm lacks native activation: {model}")
    rows = row.get("rows")
    if not isinstance(rows, list) or len(rows) != len(IQK_TASKS):
        raise RuntimeError(f"IQK attestation arm task evidence is incomplete: {model}/iqk{iqk}")
    tasks: set[str] = set()
    logprob_total = 0
    for task_row in rows:
        if not isinstance(task_row, dict) or task_row.get("task") not in IQK_TASKS:
            raise RuntimeError(f"IQK attestation arm has malformed task evidence: {model}/iqk{iqk}")
        task = task_row["task"]
        semantic = task_row.get("semantic")
        logprobs = task_row.get("logprobs")
        telemetry = task_row.get("telemetry")
        if (
            task in tasks
            or not isinstance(semantic, dict)
            or semantic.get("task") != task
            or semantic.get("status") != "pass"
            or not isinstance(logprobs, dict)
            or logprobs.get("status") != "pass"
            or isinstance(logprobs.get("token_count"), bool)
            or not isinstance(logprobs.get("token_count"), int)
            or logprobs["token_count"] <= 0
            or not isinstance(logprobs.get("tokens"), list)
            or len(logprobs["tokens"]) != logprobs["token_count"]
            or not isinstance(telemetry, dict)
        ):
            raise RuntimeError(f"IQK attestation task evidence is invalid: {model}/iqk{iqk}/{task}")
        for token in logprobs["tokens"]:
            if (
                not isinstance(token, dict)
                or re.fullmatch(r"[0-9a-f]{64}", str(token.get("token_sha256"))) is None
                or isinstance(token.get("logprob"), bool)
                or not isinstance(token.get("logprob"), (int, float))
                or not math.isfinite(float(token["logprob"]))
            ):
                raise RuntimeError(f"IQK attestation token logprob is invalid: {model}/iqk{iqk}/{task}")
        timings = telemetry.get("timings")
        counters = telemetry.get("counters")
        required_timings = {"prompt_n", "predicted_n", "prompt_ms", "predicted_ms"}
        if not isinstance(timings, dict) or not required_timings.issubset(timings) or not isinstance(counters, dict):
            raise RuntimeError(f"IQK attestation telemetry is incomplete: {model}/iqk{iqk}/{task}")
        for name, value in {**timings, **counters}.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(f"IQK attestation telemetry is non-finite: {model}/iqk{iqk}/{task}/{name}")
        predicted_n = timings.get("predicted_n")
        completion_tokens = counters.get("completion_tokens")
        if (
            isinstance(predicted_n, bool)
            or not isinstance(predicted_n, int)
            or predicted_n <= 0
            or isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or completion_tokens <= 0
            or logprobs["token_count"] != predicted_n
            or logprobs["token_count"] != completion_tokens
        ):
            raise RuntimeError(f"IQK attestation logprob/usage/timing token counts drifted: {model}/iqk{iqk}/{task}")
        tasks.add(task)
        logprob_total += logprobs["token_count"]
    if tasks != IQK_TASKS or logprob_total != numerical["logprob_token_count"]:
        raise RuntimeError(f"IQK attestation arm task/logprob totals drifted: {model}/iqk{iqk}")
    runtime = row.get("runtime_identity")
    if (
        not isinstance(runtime, dict)
        or runtime.get("before") != runtime.get("after")
    ):
        raise RuntimeError(f"IQK attestation runtime identity changed: {model}/iqk{iqk}")


def validate_iqk_attestation(
    role: str,
    path: Path,
    candidate: dict[str, Any],
    current_models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    value = _strict_json_object(path, f"{role} attestation")
    _require_exact_keys(value, {"schema", "created_at", "status", "attestation_roles", "decision_gate", "identity", "arms"}, role)
    if value["schema"] != IQK_ATTESTATION_SCHEMA or value["status"] != "pass":
        raise RuntimeError(f"{role} attestation has wrong schema or non-pass status")
    _require_utc_timestamp(value["created_at"], f"{role}.created_at")
    if value["attestation_roles"] != {"correctness": True, "coherence": True, "numerical_safety": True}:
        raise RuntimeError(f"{role} attestation does not prove all semantic roles")
    decision = value["decision_gate"]
    expected_decision = {
        "handoff": "iqk-iquant-enablement B2",
        "b2_gate_passed": True,
        "promotion_decision": False,
        "semantic_contract": "IQK arms are not bit-exact; both independently satisfy fixed tasks",
        "timings": "non-decision observational only",
    }
    if decision != expected_decision:
        raise RuntimeError(f"{role} attestation semantic decision flags are invalid")
    identity = value["identity"]
    if not isinstance(identity, dict) or not isinstance(identity.get("candidate"), dict):
        raise RuntimeError(f"{role} attestation lacks candidate identity")
    attested_candidate = identity["candidate"]
    local_libraries = attested_candidate.get("local_libraries")
    if not isinstance(local_libraries, dict) or not isinstance(attested_candidate.get("binary"), dict):
        raise RuntimeError(f"{role} attestation candidate artifact identity is malformed")
    expected_runtime = candidate["shared_library_identity"]
    if (attested_candidate.get("branch") != CANDIDATE_BRANCH or attested_candidate.get("head") != CANDIDATE_HEAD
            or attested_candidate.get("binary", {}).get("sha256") != IQK_SERVER_SHA256
            or local_libraries.get("filename_sha256") != IQK_SERVER_LIBRARY_SHA256
            or not isinstance(local_libraries.get("openmp_runtime"), dict)
            or local_libraries["openmp_runtime"].get("sha256") != LLVM20_OPENMP_IDENTITY["sha256"]):
        # The IQK server is separately pinned by its own schema; this runner requires
        # the candidate source/runtime binding rather than accepting an arbitrary file.
        raise RuntimeError(f"{role} attestation candidate branch/head/binary/runtime binding is invalid")
    if expected_runtime["openmp_runtime"]["sha256"] != LLVM20_OPENMP_IDENTITY["sha256"]:
        raise RuntimeError("local candidate runtime is not the pinned LLVM 20 artifact")
    _validate_iqk_runtime_identity(identity, current_models, role)
    arms = value["arms"]
    expected = {(name, iqk) for name in IQK_ATTESTATION_MODELS for iqk in (0, 1)}
    actual = {(row.get("model"), row.get("iqk")) for row in arms if isinstance(row, dict)} if isinstance(arms, list) else set()
    if actual != expected or not isinstance(arms, list) or len(arms) != len(expected):
        raise RuntimeError(f"{role} attestation does not contain the complete exact 3x2 IQK arm matrix")
    for row in arms:
        _validate_iqk_arm(row)
        runtime = row["runtime_identity"]["before"]
        _validate_iqk_runtime_identity(
            runtime,
            current_models,
            f"{role}.{row['model']}.iqk{row['iqk']}",
        )
        if runtime != identity:
            raise RuntimeError(
                f"{role} attestation arm runtime identity is not the summary identity"
            )
    return {"artifact": file_identity(path), "verified_by_runner": True, "schema": value["schema"], "created_at": value["created_at"]}


def validate_host_attestation(
    path: Path,
    arms: dict[str, dict[str, Any]],
    *,
    require_fresh: bool,
) -> dict[str, Any]:
    value = _strict_json_object(path, "host attestation")
    required = {"schema", "protocol", "status", "created_at", "candidate", "production", "artifact_binding"}
    _require_exact_keys(value, required, "host attestation")
    if value["schema"] != HOST_ATTESTATION_SCHEMA or value["protocol"] != "P-BENCH-PREFILL-1" or value["status"] != "pass":
        raise RuntimeError("host attestation schema/protocol/status is invalid")
    if require_fresh:
        _require_fresh_host_timestamp(value["created_at"])
    else:
        _require_utc_timestamp(value["created_at"], "host.created_at")
    for name in ("candidate", "production"):
        attested = value[name]
        arm = arms[name]
        if not isinstance(attested, dict) or attested.get("branch") != arm["actual_branch"] or attested.get("head") != arm["actual_head"]:
            raise RuntimeError(f"host attestation {name} branch/head binding is invalid")
    binding = value["artifact_binding"]
    if not isinstance(binding, dict):
        raise RuntimeError("host attestation artifact binding is malformed")
    for name in ("candidate", "production"):
        expected = arms[name]["binary_identity"]["sha256"]
        if not isinstance(binding.get(name), dict) or binding[name].get("binary_sha256") != expected:
            raise RuntimeError(f"host attestation {name} binary artifact binding is invalid")
    # A supplied host file is only corroborating evidence.  The runner performs
    # the live host gate itself immediately before work begins.
    live = host_snapshot()
    require_clean_host(live, "external host-attestation validation")
    return {"artifact": file_identity(path), "verified_by_runner": True, "live_host": live}


def write_new_json_atomic(path: Path, value: Any) -> None:
    """Publish a JSON artifact atomically without replacing an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.parent.resolve() / path.name
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            os.fchmod(output.fileno(), 0o600)
            output.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise RuntimeError(
                f"host attestation target already exists; refusing overwrite: {target}"
            ) from exc
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def write_host_attestation(path: Path) -> dict[str, Any]:
    """Capture exact current arm identities and a clean-host attestation."""
    arms = {
        name: collect_arm_identity(arm_spec(name))
        for name in ("production", "candidate")
    }
    live_host = host_snapshot()
    require_clean_host(live_host, "host-attestation production")
    value = {
        "schema": HOST_ATTESTATION_SCHEMA,
        "protocol": "P-BENCH-PREFILL-1",
        "status": "pass",
        "created_at": utc_now(),
        "candidate": {
            "branch": arms["candidate"]["actual_branch"],
            "head": arms["candidate"]["actual_head"],
        },
        "production": {
            "branch": arms["production"]["actual_branch"],
            "head": arms["production"]["actual_head"],
        },
        "artifact_binding": {
            name: {"binary_sha256": arms[name]["binary_identity"]["sha256"]}
            for name in ("candidate", "production")
        },
    }
    write_new_json_atomic(path, value)
    return value


def validate_external_attestations(
    paths: dict[str, Path],
    arms: dict[str, dict[str, Any]],
    models: dict[str, dict[str, Any]],
    *,
    require_fresh_host: bool,
) -> dict[str, Any]:
    candidate = arms["candidate"]
    validated = {role: validate_iqk_attestation(role, paths[role], candidate, models)
                 for role in ("correctness", "coherence", "numerical_safety")}
    validated["host"] = validate_host_attestation(
        paths["host"],
        arms,
        require_fresh=require_fresh_host,
    )
    return validated


def discover_model_shards(entry_path: Path) -> list[Path]:
    """Resolve an exact split set from a llama.cpp first-shard entry path."""
    if not entry_path.is_file():
        raise RuntimeError(f"required model file missing: {entry_path}")
    match = SHARD_RE.match(entry_path.name)
    if match is None:
        return [entry_path]
    count = int(match.group("count"))
    index = int(match.group("index"))
    if index != 1 or count < 2:
        raise RuntimeError(f"split GGUF entry must be shard 00001 of a multi-shard set: {entry_path}")
    expected = [
        entry_path.parent / f"{match.group('prefix')}-{part:05d}-of-{count:05d}.gguf"
        for part in range(1, count + 1)
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"incomplete split GGUF set: missing {missing}")
    actual = sorted(entry_path.parent.glob(f"{match.group('prefix')}-?????-of-{count:05d}.gguf"))
    if actual != expected:
        raise RuntimeError(
            f"split GGUF set is not exact: expected {[p.name for p in expected]}, "
            f"found {[p.name for p in actual]}"
        )
    return expected


def model_identity(entry_path: Path) -> dict[str, Any]:
    shards = discover_model_shards(entry_path)
    identities = [file_identity(path) for path in shards]
    return {
        "entry_path": str(entry_path),
        "shard_count": len(shards),
        "total_bytes": sum(item["bytes"] for item in identities),
        "shards": identities,
    }


def shared_library_identities(binary: Path, library_path: Path) -> dict[str, Any]:
    env = canonical_parent_environment()
    env["LD_LIBRARY_PATH"] = f"{library_path}:{LLVM20_LIBDIR}"
    capture = run(["ldd", str(binary)], check=True, env=env)
    records: list[dict[str, Any]] = []
    for match in LLAMA_LIBRARY_RE.finditer(capture.stdout):
        target_text = match.group("target")
        if target_text == "not":
            raise RuntimeError(f"unresolved llama.cpp shared library: {match.group(0)}")
        target = Path(target_text).resolve()
        if target.parent != library_path.resolve():
            raise RuntimeError(f"shared library resolves outside selected arm: {match.group(0)}")
        identity = file_identity(target)
        records.append(
            {
                "soname": match.group("soname"),
                "reported_target": target_text,
                "resolved_target": str(target),
                **{key: identity[key] for key in ("device", "inode", "bytes", "mtime_ns", "sha256")},
            }
        )
    if not any(item["soname"].startswith("libllama") for item in records):
        raise RuntimeError("ldd did not resolve a local libllama shared library")
    if not any(item["soname"].startswith("libggml") for item in records):
        raise RuntimeError("ldd did not resolve local libggml shared libraries")
    openmp_matches = list(OPENMP_LIBRARY_RE.finditer(capture.stdout))
    if len(openmp_matches) != 1:
        raise RuntimeError(
            f"ldd resolved {len(openmp_matches)} OpenMP runtimes, expected exactly one"
        )
    openmp_match = openmp_matches[0]
    openmp_target_text = openmp_match.group("target")
    if openmp_target_text == "not":
        raise RuntimeError(f"unresolved OpenMP runtime: {openmp_match.group(0)}")
    reported_openmp = Path(openmp_target_text)
    resolved_openmp = reported_openmp.resolve()
    expected_openmp_dir = Path(LLVM20_LIBDIR).resolve()
    if (
        reported_openmp.parent.resolve() != expected_openmp_dir
        or resolved_openmp.parent != expected_openmp_dir
        or resolved_openmp.name != "libomp.so.5"
    ):
        raise RuntimeError(
            "OpenMP runtime did not resolve to LLVM 20 libomp: "
            f"{openmp_match.group(0)} -> {resolved_openmp}"
        )
    openmp_identity = file_identity(resolved_openmp)
    openmp_runtime = {
        "soname": openmp_match.group("soname"),
        "reported_target": openmp_target_text,
        "resolved_target": str(resolved_openmp),
        **{key: openmp_identity[key] for key in ("device", "inode", "bytes", "mtime_ns", "sha256")},
    }
    return {
        "argv": ["ldd", str(binary)],
        "stdout": capture.stdout,
        "libraries": records,
        "openmp_runtime": openmp_runtime,
    }


def validate_release_artifacts(arm: dict[str, Any], identity: dict[str, Any]) -> None:
    """Reject source-correct but rebuilt/substituted release objects."""
    expected = RELEASE_ARTIFACTS[arm["name"]]
    if identity["binary_identity"]["sha256"] != expected["llama_bench_sha256"]:
        raise RuntimeError(f"{arm['name']} llama-bench SHA256 is not the pinned release artifact")
    actual_libraries = {
        Path(item["resolved_target"]).name: item["sha256"]
        for item in identity["shared_library_identity"]["libraries"]
    }
    if len(actual_libraries) != len(identity["shared_library_identity"]["libraries"]) or actual_libraries != expected["libraries"]:
        raise RuntimeError(
            f"{arm['name']} resolved llama/ggml library SHA256 set is not pinned: "
            f"actual={actual_libraries} expected={expected['libraries']}"
        )
    runtime = identity["shared_library_identity"]["openmp_runtime"]
    for key, value in LLVM20_OPENMP_IDENTITY.items():
        if runtime.get(key) != value:
            raise RuntimeError(f"{arm['name']} LLVM 20 OpenMP runtime identity drifted: {runtime}")


def validate_source_status(arm_name: str, porcelain: str) -> list[str]:
    lines = [line for line in porcelain.splitlines() if line]
    if arm_name == "candidate":
        if lines:
            raise RuntimeError(f"candidate worktree is not completely clean: {lines}")
        return lines
    if arm_name != "production":
        raise RuntimeError(f"unknown arm name: {arm_name}")
    allowed_untracked = {".gitnexusignore", "tools/math-tools/"}
    for line in lines:
        if not line.startswith("?? "):
            raise RuntimeError(f"production tracked/index dirt is forbidden: {line}")
        path = line[3:]
        if path not in allowed_untracked and not path.startswith("tools/math-tools/"):
            raise RuntimeError(f"production has unexpected untracked content: {path}")
    return lines


def collect_arm_identity(arm: dict[str, Any]) -> dict[str, Any]:
    root, binary = Path(arm["source_root"]), Path(arm["binary"])
    head = git_value(root, "rev-parse", "HEAD")
    branch = git_value(root, "branch", "--show-current")
    dirty = git_value(root, "status", "--porcelain=v1")
    if head != arm["expected_head"] or branch != arm["expected_branch"]:
        raise RuntimeError(
            f"invalid {arm['name']} identity: head={head} branch={branch}"
        )
    source_status = validate_source_status(arm["name"], dirty)
    libraries = shared_library_identities(binary, Path(arm["library_path"]))
    identity = {
        **arm,
        "actual_head": head,
        "actual_branch": branch,
        "source_status": source_status,
        "binary_identity": file_identity(binary),
        "shared_library_identity": libraries,
    }
    validate_release_artifacts(arm, identity)
    return identity


def process_ownership(proc_root: Path = Path("/proc")) -> dict[str, Any]:
    exact_llama: list[dict[str, Any]] = []
    autopilot: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    unresolved_ownership: list[dict[str, Any]] = []
    uncertain_relevant: list[dict[str, Any]] = []
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == os.getpid():
            continue
        pid = int(process_dir.name)
        try:
            cmdline = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
            comm = (process_dir / "comm").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        except (PermissionError, OSError) as exc:
            row = {"pid": pid, "error": repr(exc)}
            unreadable.append(row)
            unresolved_ownership.append(row)
            continue
        autopilot_match = (
            "scripts/autopilot/autopilot.py" in cmdline and " start" in f" {cmdline}"
        )
        if autopilot_match:
            autopilot.append({"pid": pid, "comm": comm, "cmdline": cmdline})
        argv0 = cmdline.split(maxsplit=1)[0] if cmdline else ""
        llama_like = (
            comm in {"llama-bench", "llama-server", "llama-cli"}
            or Path(argv0).name in {"llama-bench", "llama-server", "llama-cli"}
        )
        try:
            executable = (process_dir / "exe").resolve(strict=True)
        except FileNotFoundError as exc:
            # A live process whose executable was deleted exposes an unresolved
            # /proc/PID/exe symlink. It is not an exited-process race while the
            # PID directory still exists.
            if process_dir.exists() and llama_like:
                uncertain_relevant.append(
                    {
                        "pid": pid,
                        "comm": comm,
                        "cmdline": cmdline,
                        "error": repr(exc),
                        "reason": "live llama-like PID has missing/deleted executable",
                    }
                )
            continue
        except (PermissionError, OSError) as exc:
            unreadable.append({"pid": pid, "comm": comm, "error": repr(exc)})
            if llama_like:
                uncertain_relevant.append(
                    {"pid": pid, "comm": comm, "cmdline": cmdline, "error": repr(exc)}
                )
            continue
        row = {"pid": pid, "exe": str(executable), "comm": comm, "cmdline": cmdline}
        if executable.name in {"llama-bench", "llama-server", "llama-cli"}:
            exact_llama.append(row)
        if autopilot_match:
            autopilot[-1]["exe"] = str(executable)
    return {
        "exact_llama_processes": exact_llama,
        "autopilot_processes": autopilot,
        "unreadable_processes": unreadable,
        "unresolved_ownership": unresolved_ownership,
        "uncertain_relevant_processes": uncertain_relevant,
    }


def kfd_ownership(proc_root: Path = Path("/proc")) -> dict[str, Any]:
    users: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == os.getpid():
            continue
        pid = int(process_dir.name)
        try:
            descriptors = list((process_dir / "fd").iterdir())
        except FileNotFoundError:
            continue
        except (PermissionError, OSError) as exc:
            unreadable.append({"pid": pid, "error": repr(exc)})
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except FileNotFoundError:
                continue
            except (PermissionError, OSError) as exc:
                unreadable.append({"pid": pid, "fd": descriptor.name, "error": repr(exc)})
                break
            if target == "/dev/kfd":
                users.append({"pid": pid, "fd": descriptor.name})
    fallback: dict[str, Any] | None = None
    if unreadable:
        capture = run(["lsof", "-n", "-P", "-Fpcn", "/dev/kfd"])
        fallback = {
            "argv": ["lsof", "-n", "-P", "-Fpcn", "/dev/kfd"],
            "returncode": capture.returncode,
            "stdout": capture.stdout,
            "stderr": capture.stderr,
        }
        if capture.returncode in (0, 1):
            for line in capture.stdout.splitlines():
                if not line.startswith("p"):
                    continue
                try:
                    pid = int(line[1:])
                except ValueError as exc:
                    raise RuntimeError(f"unparseable lsof KFD owner: {line}") from exc
                if pid != os.getpid() and not any(item["pid"] == pid for item in users):
                    users.append({"pid": pid, "fd": "lsof:/dev/kfd"})
    return {
        "users": users,
        "unreadable_processes": unreadable,
        "lsof_fallback": fallback,
    }


def _read_required(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"required host-state file is unreadable: {path}: {exc}") from exc


def _active_thp(value: str) -> str | None:
    match = re.search(r"\[(\w+)\]", value)
    return match.group(1) if match else None


def _parse_nonnegative_decimal(value: Any, label: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"\d+", value):
        raise RuntimeError(f"{label} is not a non-negative decimal integer: {value!r}")
    return int(value)


def parse_meminfo_evidence(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw:
        raise RuntimeError("meminfo evidence is missing or is not text")
    required_units = {
        **{key: "kB" for key in BASE_MEMINFO_KEYS},
        **THP_HUGEPAGE_MEMINFO_UNITS,
    }
    fields: dict[str, dict[str, Any]] = {}
    for line in raw.splitlines():
        key, separator, value_text = line.partition(":")
        if key not in required_units:
            continue
        if not separator or key in fields:
            raise RuntimeError(f"meminfo field {key!r} is duplicated or malformed")
        match = re.fullmatch(r"\s*(\d+)(?:\s+(kB))?\s*", value_text)
        if not match:
            raise RuntimeError(f"meminfo field {key!r} has a malformed value: {value_text!r}")
        unit = match.group(2) or "count"
        expected_unit = required_units[key]
        if unit != expected_unit:
            raise RuntimeError(
                f"meminfo field {key!r} has unit {unit!r}, expected {expected_unit!r}"
            )
        fields[key] = {"value": int(match.group(1)), "unit": unit}
    missing = sorted(set(required_units) - set(fields))
    if missing:
        raise RuntimeError(f"meminfo evidence is missing required fields: {missing}")

    values = {key: field["value"] for key, field in fields.items()}
    if values["MemFree"] > values["MemTotal"]:
        raise RuntimeError("MemFree exceeds MemTotal")
    if values["MemAvailable"] > values["MemTotal"]:
        raise RuntimeError("MemAvailable exceeds MemTotal")
    if values["HugePages_Free"] > values["HugePages_Total"]:
        raise RuntimeError("HugePages_Free exceeds HugePages_Total")
    if values["HugePages_Rsvd"] > values["HugePages_Free"]:
        raise RuntimeError("HugePages_Rsvd exceeds HugePages_Free")
    if values["HugePages_Surp"] > values["HugePages_Total"]:
        raise RuntimeError("HugePages_Surp exceeds HugePages_Total")
    return {
        "fields": fields,
        "memory_kib": {key: values[key] for key in BASE_MEMINFO_KEYS},
    }


def read_hugepage_pools(
    root: Path = Path("/sys/kernel/mm/hugepages"),
) -> list[dict[str, Any]]:
    try:
        directories = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError as exc:
        raise RuntimeError(f"hugepage pool root is unreadable: {root}: {exc}") from exc
    pools: list[dict[str, Any]] = []
    for directory in directories:
        match = HUGEPAGE_POOL_RE.fullmatch(directory.name)
        if not match:
            continue
        pool: dict[str, Any] = {
            "path": str(directory),
            "page_size_kib": int(match.group(1)),
        }
        for filename in HUGEPAGE_POOL_FILES:
            pool[filename] = _parse_nonnegative_decimal(
                _read_required(directory / filename),
                f"{directory / filename}",
            )
        pools.append(pool)
    if not pools:
        raise RuntimeError(f"no hugepage pools found under {root}")
    return pools


def validate_thp_hugepage_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise RuntimeError("THP/hugepage state is missing or is not an object")
    raw = state.get("meminfo_raw")
    parsed = parse_meminfo_evidence(raw)
    if state.get("meminfo_fields") != parsed["fields"]:
        raise RuntimeError("recorded meminfo fields disagree with raw /proc/meminfo")

    pmd_size = state.get("hpage_pmd_size_bytes")
    if isinstance(pmd_size, bool) or not isinstance(pmd_size, int) or pmd_size <= 0:
        raise RuntimeError(f"THP hpage_pmd_size is invalid: {pmd_size!r}")
    if pmd_size % 1024:
        raise RuntimeError(f"THP hpage_pmd_size is not KiB-aligned: {pmd_size}")

    pools = state.get("pools")
    if not isinstance(pools, list) or not pools:
        raise RuntimeError("THP/hugepage state has no size-specific pools")
    by_size: dict[int, dict[str, Any]] = {}
    for pool in pools:
        if not isinstance(pool, dict):
            raise RuntimeError(f"hugepage pool is not an object: {pool!r}")
        page_size = pool.get("page_size_kib")
        if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size <= 0:
            raise RuntimeError(f"hugepage pool page size is invalid: {page_size!r}")
        if page_size in by_size:
            raise RuntimeError(f"duplicate hugepage pool size: {page_size} kB")
        expected_path = f"/sys/kernel/mm/hugepages/hugepages-{page_size}kB"
        if pool.get("path") != expected_path:
            raise RuntimeError(
                f"hugepage pool path disagrees with page size: {pool.get('path')!r}"
            )
        counts: dict[str, int] = {}
        for filename in HUGEPAGE_POOL_FILES:
            value = pool.get(filename)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(
                    f"hugepage pool {page_size} {filename} is invalid: {value!r}"
                )
            counts[filename] = value
        if counts["free_hugepages"] > counts["nr_hugepages"]:
            raise RuntimeError(f"hugepage pool {page_size} free count exceeds total")
        if counts["resv_hugepages"] > counts["free_hugepages"]:
            raise RuntimeError(f"hugepage pool {page_size} reserved count exceeds free")
        if counts["surplus_hugepages"] > counts["nr_hugepages"]:
            raise RuntimeError(f"hugepage pool {page_size} surplus count exceeds total")
        by_size[page_size] = pool

    fields = parsed["fields"]
    default_size = fields["Hugepagesize"]["value"]
    default_pool = by_size.get(default_size)
    if default_pool is None:
        raise RuntimeError(f"default hugepage pool {default_size} kB is missing")
    comparisons = {
        "HugePages_Total": "nr_hugepages",
        "HugePages_Free": "free_hugepages",
        "HugePages_Rsvd": "resv_hugepages",
        "HugePages_Surp": "surplus_hugepages",
    }
    for meminfo_key, pool_key in comparisons.items():
        if fields[meminfo_key]["value"] != default_pool[pool_key]:
            raise RuntimeError(
                f"{meminfo_key} disagrees with {default_pool['path']}/{pool_key}"
            )
    if pmd_size // 1024 not in by_size:
        raise RuntimeError(f"THP PMD-size pool {pmd_size // 1024} kB is missing")
    hugetlb_kib = sum(
        pool["page_size_kib"] * pool["nr_hugepages"] for pool in pools
    )
    if fields["Hugetlb"]["value"] != hugetlb_kib:
        raise RuntimeError("Hugetlb disagrees with the size-specific hugepage pools")
    return parsed


def host_snapshot() -> dict[str, Any]:
    governors = {
        str(path): _read_required(path)
        for path in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"))
    }
    meminfo_raw = _read_required(Path("/proc/meminfo"))
    meminfo = parse_meminfo_evidence(meminfo_raw)
    thp_hugepage_state = {
        "meminfo_raw": meminfo_raw,
        "meminfo_fields": meminfo["fields"],
        "hpage_pmd_size_bytes": _parse_nonnegative_decimal(
            _read_required(
                Path("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size")
            ),
            "THP hpage_pmd_size",
        ),
        "pools": read_hugepage_pools(),
    }
    validate_thp_hugepage_state(thp_hugepage_state)
    processes = process_ownership()
    kfd = kfd_ownership()
    rocm = run(["rocm-smi", "--showpidgpus"])
    rocm_rows = []
    for line in rocm.stdout.splitlines():
        match = re.match(r"^\s*(\d+)\s+(\S+)\s+", line)
        if match and "pid" not in line.lower():
            rocm_rows.append({"pid": int(match.group(1)), "process": match.group(2), "raw": line})
    return {
        "captured_at": utc_now(),
        "uptime_seconds": float(_read_required(Path("/proc/uptime")).split()[0]),
        "governors": governors,
        "thp_enabled": {
            "raw": _read_required(Path("/sys/kernel/mm/transparent_hugepage/enabled")),
        },
        "thp_defrag": {
            "raw": _read_required(Path("/sys/kernel/mm/transparent_hugepage/defrag")),
        },
        "numa_balancing": _read_required(Path("/proc/sys/kernel/numa_balancing")),
        "memory_kib": meminfo["memory_kib"],
        "thp_hugepage_state": thp_hugepage_state,
        "process_ownership": processes,
        "kfd_ownership": kfd,
        "rocm_ownership": {
            "argv": ["rocm-smi", "--showpidgpus"],
            "returncode": rocm.returncode,
            "stdout": rocm.stdout,
            "stderr": rocm.stderr,
            "owners": rocm_rows,
        },
    }


def host_state_blockers(snapshot: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if snapshot["uptime_seconds"] > MAX_UPTIME_SECONDS:
        blockers.append(f"uptime exceeds one week: {snapshot['uptime_seconds']} seconds")
    governors = snapshot["governors"]
    if not governors or any(value != "performance" for value in governors.values()):
        blockers.append("not every CPU scaling governor is performance")
    if _active_thp(snapshot["thp_enabled"]["raw"]) != "always":
        blockers.append("THP enabled mode is not always")
    if _active_thp(snapshot["thp_defrag"]["raw"]) != "always":
        blockers.append("THP defrag mode is not always")
    if snapshot["numa_balancing"] != "0":
        blockers.append("kernel.numa_balancing is not 0")
    processes = snapshot["process_ownership"]
    if processes["unresolved_ownership"]:
        blockers.append("process ownership could not resolve comm/cmdline identity")
    if processes["uncertain_relevant_processes"]:
        blockers.append("exact llama ownership contains unresolved relevant processes")
    if processes["exact_llama_processes"]:
        blockers.append("residual or concurrent exact llama process")
    if processes["autopilot_processes"]:
        blockers.append("autopilot process is active")
    kfd = snapshot["kfd_ownership"]
    if kfd["unreadable_processes"] and (
        not kfd["lsof_fallback"] or kfd["lsof_fallback"]["returncode"] not in (0, 1)
    ):
        blockers.append("KFD ownership could not be resolved through /proc or lsof")
    if snapshot["kfd_ownership"]["users"]:
        blockers.append("/dev/kfd is owned")
    rocm = snapshot["rocm_ownership"]
    if rocm["returncode"] != 0:
        blockers.append("ROCm ownership query failed")
    if rocm["owners"]:
        blockers.append("ROCm reports active owners")
    try:
        parsed_meminfo = validate_thp_hugepage_state(
            snapshot.get("thp_hugepage_state")
        )
        if snapshot.get("memory_kib") != parsed_meminfo["memory_kib"]:
            raise RuntimeError("memory_kib disagrees with raw /proc/meminfo")
    except (KeyError, RuntimeError, TypeError) as exc:
        blockers.append(f"THP/hugepage pool state is invalid: {exc}")
    return blockers


def require_clean_host(snapshot: dict[str, Any], phase: str) -> None:
    blockers = host_state_blockers(snapshot)
    if blockers:
        raise RuntimeError(f"{phase} strict host check failed: {blockers}")


def prepare_pair(shards: list[Path]) -> list[dict[str, Any]]:
    """Required cache reset and NUMA-aware read.  No benchmark runs here."""
    reset_argv = ["sync"]
    drop_argv = [
        "sudo",
        "-n",
        "/usr/bin/env",
        "-i",
        *environment_assignments(),
        "/usr/bin/tee",
        "/proc/sys/vm/drop_caches",
    ]
    reset = run(reset_argv)
    drop = run(drop_argv, input_text="3\n")
    time.sleep(2)
    records = []
    for name, command, result in (("sync", reset_argv, reset), ("drop_caches", drop_argv, drop)):
        records.append({"step": name, "argv": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    # Rewarm every split shard under the same NUMA-interleaved placement.
    for shard in shards:
        rewarm_argv = ["taskset", "-c", "0-95", "numactl", "--interleave=all", "dd", f"if={shard}", "of=/dev/null", "bs=64M", "iflag=fullblock", "status=none"]
        rewarm = run(rewarm_argv)
        records.append({"step": "numa_rewarm", "shard": str(shard), "argv": rewarm_argv, "returncode": rewarm.returncode, "stdout": rewarm.stdout, "stderr": rewarm.stderr})
    if any(item["returncode"] != 0 for item in records):
        raise RuntimeError(f"pair cache preparation failed: {records}")
    return records


def parse_result(raw_stdout: str, metric: str, expected_head: str) -> dict[str, Any]:
    """Read and validate an exact llama-bench JSON result."""
    try:
        rows = json.loads(raw_stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("canonical stdout was not the requested JSON result") from exc
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError(f"expected exactly one llama-bench JSON row, got {type(rows).__name__}/{len(rows) if isinstance(rows, list) else 'n/a'}")
    row = rows[0]
    expected_prompt, expected_gen = (PREFILL, 0) if metric == "pp2048" else (0, 128)
    if row.get("n_prompt") != expected_prompt or row.get("n_gen") != expected_gen:
        raise RuntimeError(f"result profile drift: expected p={expected_prompt}, n={expected_gen}")
    build_commit = row.get("build_commit")
    build_number = row.get("build_number")
    if (
        not isinstance(build_commit, str)
        or re.fullmatch(r"[0-9a-f]{7,40}", build_commit) is None
        or not expected_head.startswith(build_commit)
    ):
        raise RuntimeError(f"JSON build_commit {build_commit!r} does not match source HEAD {expected_head}")
    if isinstance(build_number, bool) or not isinstance(build_number, int) or build_number <= 0:
        raise RuntimeError(f"invalid JSON build_number: {build_number!r}")
    samples = row.get("samples_ts")
    samples_ns = row.get("samples_ns")
    if not isinstance(samples, list) or len(samples) != REPS:
        raise RuntimeError(f"expected exactly {REPS} numeric samples_ts in canonical JSON")
    if not isinstance(samples_ns, list) or len(samples_ns) != REPS:
        raise RuntimeError(f"expected exactly {REPS} numeric samples_ns in canonical JSON")
    normalized: list[float] = []
    normalized_ns: list[float] = []
    for value, value_ns in zip(samples, samples_ns):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"samples_ts contains a non-numeric value: {value!r}")
        converted = float(value)
        if not math.isfinite(converted) or converted <= 0:
            raise RuntimeError(f"samples_ts contains a nonpositive or non-finite value: {value!r}")
        normalized.append(converted)
        converted_ns = require_finite_positive(value_ns, "samples_ns value")
        normalized_ns.append(converted_ns)
    return {
        "samples_ts": normalized,
        "samples_ns": normalized_ns,
        "build_commit": build_commit,
        "build_number": build_number,
    }


def parse_samples(raw_stdout: str, metric: str, expected_head: str) -> list[float]:
    return parse_result(raw_stdout, metric, expected_head)["samples_ts"]


def build_witness(raw: str, expected_head: str) -> dict[str, str]:
    match = BUILD_LINE_RE.search(raw)
    if not match:
        raise RuntimeError("canonical llama-bench output lacks result-emitted build witness")
    commit, number = match.groups()
    if not expected_head.startswith(commit):
        raise RuntimeError(f"build witness {commit} does not match source HEAD {expected_head}")
    return {"line": match.group(0).strip(), "commit": commit, "build_number": number}


def resolve_build_commit(source_root: Path, abbreviated: str, expected_head: str) -> str:
    if re.fullmatch(r"[0-9a-f]{7,40}", abbreviated) is None:
        raise RuntimeError(f"invalid emitted build commit: {abbreviated!r}")
    resolved = git_value(
        source_root,
        "rev-parse",
        "--verify",
        f"{abbreviated}^{{commit}}",
    )
    if resolved != expected_head:
        raise RuntimeError(
            f"emitted build commit {abbreviated} resolves to {resolved}, "
            f"not pinned HEAD {expected_head}"
        )
    return resolved


def require_finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric: {value!r}")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise RuntimeError(f"{label} must be finite and positive: {value!r}")
    return converted


def safe_ratio(numerator: Any, denominator: Any, label: str) -> float:
    top = require_finite_positive(numerator, f"{label} numerator")
    bottom = require_finite_positive(denominator, f"{label} denominator")
    ratio = top / bottom
    if not math.isfinite(ratio) or ratio <= 0:
        raise RuntimeError(f"{label} ratio overflowed or is invalid: {ratio!r}")
    return ratio


def stats(samples: list[float]) -> dict[str, Any]:
    checked = [
        require_finite_positive(value, f"samples[{index}]")
        for index, value in enumerate(samples)
    ]
    median = statistics.median(checked)
    mad = statistics.median([abs(value - median) for value in checked])
    if not math.isfinite(median) or not math.isfinite(mad):
        raise RuntimeError("median or MAD is non-finite")
    return {"samples_ts": checked, "median_ts": median, "mad_ts": mad}


def measurement_window_observation(
    monitor: dict[str, Any], samples_ns: list[float]
) -> dict[str, Any]:
    """Persist the available overlap facts without inventing per-repetition clocks."""
    sustained = monitor.get("sustained_window")
    samples = monitor.get("samples")
    if not isinstance(sustained, dict) or not isinstance(samples, list) or len(samples) < 2:
        raise RuntimeError("contention monitor lacks raw intervals for measurement coverage")
    total_measured_s = sum(
        require_finite_positive(value, "samples_ns") for value in samples_ns
    ) / 1_000_000_000
    first = require_finite_positive(samples[0].get("monotonic"), "first monitor monotonic")
    last = require_finite_positive(samples[-1].get("monotonic"), "last monitor monotonic")
    selected_start = require_finite_positive(sustained.get("start_monotonic"), "sustained window start")
    selected_end = require_finite_positive(sustained.get("end_monotonic"), "sustained window end")
    selected_duration = require_finite_positive(sustained.get("elapsed_s"), "sustained window duration")
    if last <= first or selected_end <= selected_start or selected_start < first or selected_end > last:
        raise RuntimeError("contention monitor window bounds are invalid for measurement coverage")
    endpoint_duration = selected_end - selected_start
    if not math.isclose(
        selected_duration, endpoint_duration, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise RuntimeError(
            "contention monitor sustained window duration disagrees with endpoints: "
            f"declared={selected_duration} endpoints={endpoint_duration}"
        )
    observed_duration = last - first
    minimum_overlap_s = max(0.0, total_measured_s + selected_duration - observed_duration)
    return {
        "samples_ns": samples_ns,
        "total_measured_repetition_duration_s": total_measured_s,
        "selected_clean_window_duration_s": selected_duration,
        "observed_monitor_duration_s": observed_duration,
        "overlap_basis": "interval-arithmetic-lower-bound; per-repetition timestamps unavailable",
        "minimum_clean_overlap_s": minimum_overlap_s,
        "binding_status": "unavailable",
        "per_repetition_timestamps": "unavailable",
        "raw_monitor_interval_count": len(monitor.get("intervals", [])),
    }


def _proc_stat_cpu() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    if not fields or fields[0] != "cpu" or len(fields) < 6:
        raise RuntimeError("/proc/stat aggregate CPU record is malformed")
    values = [int(item) for item in fields[1:]]
    if any(value < 0 for value in values):
        raise RuntimeError("/proc/stat contains a negative CPU counter")
    total = sum(values)
    idle = values[3] + values[4]
    return total, total - idle


def _swap_counters() -> dict[str, int]:
    values: dict[str, int] = {}
    for row in Path("/proc/vmstat").read_text(encoding="utf-8").splitlines():
        key, *rest = row.split()
        if key in {"pswpin", "pswpout"} and len(rest) == 1:
            values[key] = int(rest[0])
    if set(values) != {"pswpin", "pswpout"}:
        raise RuntimeError("/proc/vmstat lacks required swap I/O counters")
    return values


def _process_stat(pid: int) -> tuple[int, int, int]:
    raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    close = raw.rfind(")")
    if close < 0:
        raise RuntimeError(f"/proc/{pid}/stat is malformed")
    fields = raw[close + 2 :].split()
    if len(fields) < 13:
        raise RuntimeError(f"/proc/{pid}/stat has too few fields")
    # Include reaped descendants so the process-group total remains monotonic
    # when the wrapper shell collects a completed llama-bench child.
    return int(fields[1]), int(fields[2]), sum(int(fields[index]) for index in range(11, 15))


def _target_group_cpu(leader_pid: int, pgid: int) -> dict[str, Any]:
    members: list[int] = []
    total_ticks = 0
    ownership_changed = False
    for child in Path("/proc").iterdir():
        if not child.name.isdigit():
            continue
        try:
            ppid, observed_pgid, ticks = _process_stat(int(child.name))
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"unable to sample benchmark process ownership: {child}") from exc
        if observed_pgid == pgid:
            members.append(int(child.name))
            total_ticks += ticks
            if int(child.name) == leader_pid and ppid == 0:
                ownership_changed = True
    if leader_pid not in members:
        ownership_changed = True
    return {"members": sorted(members), "cpu_ticks": total_ticks, "ownership_changed": ownership_changed}


def monitor_snapshot(leader_pid: int, pgid: int) -> dict[str, Any]:
    cpu_before_started = time.monotonic()
    total_before, busy_before = _proc_stat_cpu()
    cpu_before_finished = time.monotonic()
    target_started = time.monotonic()
    target = _target_group_cpu(leader_pid, pgid)
    target_finished = time.monotonic()
    cpu_after_started = time.monotonic()
    total_after, busy_after = _proc_stat_cpu()
    cpu_after_finished = time.monotonic()
    cpu_before_at = (cpu_before_started + cpu_before_finished) / 2
    target_at = (target_started + target_finished) / 2
    cpu_after_at = (cpu_after_started + cpu_after_finished) / 2
    cpu_bracket_elapsed = cpu_after_at - cpu_before_at
    if cpu_bracket_elapsed <= 0 or not cpu_before_at <= target_at <= cpu_after_at:
        raise RuntimeError("CPU and target counter sampling order is invalid")
    interpolation_fraction = (target_at - cpu_before_at) / cpu_bracket_elapsed
    total = total_before + (total_after - total_before) * interpolation_fraction
    busy = busy_before + (busy_after - busy_before) * interpolation_fraction
    target_members = set(target["members"])
    ownership = process_ownership()
    kfd = kfd_ownership()
    contamination = {
        "exact_llama": [
            row
            for row in ownership["exact_llama_processes"]
            if row["pid"] not in target_members
        ],
        "autopilot": ownership["autopilot_processes"],
        "kfd_users": kfd["users"],
    }
    return {
        "monotonic": target_at,
        "cpu_total_ticks": total,
        "cpu_busy_ticks": busy,
        "cpu_counter_bracket": {
            "before": {
                "monotonic": cpu_before_at,
                "total_ticks": total_before,
                "busy_ticks": busy_before,
            },
            "after": {
                "monotonic": cpu_after_at,
                "total_ticks": total_after,
                "busy_ticks": busy_after,
            },
            "target_scan": {
                "started_monotonic": target_started,
                "finished_monotonic": target_finished,
                "elapsed_s": target_finished - target_started,
            },
            "target_monotonic": target_at,
            "interpolation_fraction": interpolation_fraction,
        },
        "swap": _swap_counters(),
        "target": target,
        "contamination": contamination,
    }


def validate_monitor_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) < MIN_MONITOR_SAMPLES:
        raise RuntimeError(f"insufficient continuous contention-monitor samples: {len(samples)}")

    def failure(index: int, detail: str) -> RuntimeError:
        return RuntimeError(f"contention monitor sampling failure at sample {index}: {detail}")

    def finite_number(value: Any, index: int, label: str) -> float:
        if isinstance(value, bool):
            raise failure(index, f"{label} must be numeric, not boolean")
        try:
            checked = float(value)
        except (TypeError, ValueError) as exc:
            raise failure(index, f"{label} is not numeric") from exc
        if not math.isfinite(checked):
            raise failure(index, f"{label} is not finite")
        return checked

    def exact_keys(value: Any, required: set[str], index: int, label: str) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != required:
            actual = sorted(repr(key) for key in value) if isinstance(value, dict) else type(value).__name__
            raise failure(index, f"{label} keys invalid: expected={sorted(required)} actual={actual}")
        return value

    def validate_sample(index: int, sample: dict[str, Any]) -> None:
        if not isinstance(sample, dict):
            raise failure(index, "sample is not an object")
        required = {
            "monotonic", "cpu_total_ticks", "cpu_busy_ticks", "cpu_counter_bracket",
            "swap", "target", "contamination",
        }
        exact_keys(sample, required, index, "sample")
        monotonic = finite_number(sample["monotonic"], index, "monotonic")
        total = finite_number(sample["cpu_total_ticks"], index, "cpu_total_ticks")
        busy = finite_number(sample["cpu_busy_ticks"], index, "cpu_busy_ticks")
        if total < 0 or busy < 0 or busy > total:
            raise failure(index, "top-level aggregate counters are invalid")

        contamination = exact_keys(
            sample["contamination"], {"exact_llama", "autopilot", "kfd_users"}, index, "contamination",
        )
        if any(not isinstance(value, list) for value in contamination.values()):
            raise failure(index, "contamination witnesses must be lists")

        target = exact_keys(sample["target"], {"members", "cpu_ticks", "ownership_changed"}, index, "target")
        if not isinstance(target["members"], list) or not target["members"] or any(
            type(pid) is not int or pid <= 0 for pid in target["members"]
        ) or len(set(target["members"])) != len(target["members"]):
            raise failure(index, "target members are invalid")
        if type(target["cpu_ticks"]) is not int or target["cpu_ticks"] < 0:
            raise failure(index, "target cpu_ticks is invalid")
        if type(target["ownership_changed"]) is not bool:
            raise failure(index, "target ownership_changed is invalid")

        swap = exact_keys(sample["swap"], {"pswpin", "pswpout"}, index, "swap")
        if any(type(value) is not int or value < 0 for value in swap.values()):
            raise failure(index, "swap counters are invalid")

        bracket = exact_keys(
            sample["cpu_counter_bracket"], {"before", "after", "target_scan", "target_monotonic", "interpolation_fraction"}, index, "cpu_counter_bracket",
        )
        before = exact_keys(bracket["before"], {"monotonic", "total_ticks", "busy_ticks"}, index, "cpu_counter_bracket.before")
        after = exact_keys(bracket["after"], {"monotonic", "total_ticks", "busy_ticks"}, index, "cpu_counter_bracket.after")
        scan = exact_keys(bracket["target_scan"], {"started_monotonic", "finished_monotonic", "elapsed_s"}, index, "cpu_counter_bracket.target_scan")
        before_time = finite_number(before["monotonic"], index, "bracket before monotonic")
        after_time = finite_number(after["monotonic"], index, "bracket after monotonic")
        before_total = finite_number(before["total_ticks"], index, "bracket before total")
        after_total = finite_number(after["total_ticks"], index, "bracket after total")
        before_busy = finite_number(before["busy_ticks"], index, "bracket before busy")
        after_busy = finite_number(after["busy_ticks"], index, "bracket after busy")
        target_started = finite_number(scan["started_monotonic"], index, "target scan start")
        target_finished = finite_number(scan["finished_monotonic"], index, "target scan finish")
        target_elapsed = finite_number(scan["elapsed_s"], index, "target scan elapsed")
        target_time = finite_number(bracket["target_monotonic"], index, "target monotonic")
        fraction = finite_number(bracket["interpolation_fraction"], index, "interpolation fraction")
        if any(
            type(value) is not int
            for value in (
                before["total_ticks"],
                before["busy_ticks"],
                after["total_ticks"],
                after["busy_ticks"],
            )
        ):
            raise failure(index, "cpu counter bracket aggregate counters must be integers")
        if (
            before_time >= after_time
            or target_started > target_finished
            or not math.isclose(target_elapsed, target_finished - target_started, rel_tol=1e-9, abs_tol=1e-9)
            or not before_time <= target_time <= after_time
            or not target_started <= target_time <= target_finished
            or not math.isclose(
                target_time,
                (target_started + target_finished) / 2,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not 0.0 <= fraction <= 1.0
        ):
            raise failure(index, "cpu counter bracket timing is invalid")
        if (
            any(value < 0 for value in (before_total, after_total, before_busy, after_busy))
            or before_busy > before_total
            or after_busy > after_total
            or after_total < before_total
            or after_busy < before_busy
            or after_busy - before_busy > after_total - before_total
        ):
            raise failure(index, "cpu counter bracket aggregate counters are invalid")
        expected_fraction = (target_time - before_time) / (after_time - before_time)
        if not math.isclose(fraction, expected_fraction, rel_tol=1e-9, abs_tol=1e-9):
            raise failure(index, "cpu counter bracket interpolation fraction is invalid")
        expected_total = before_total + (after_total - before_total) * fraction
        expected_busy = before_busy + (after_busy - before_busy) * fraction
        if (
            not math.isclose(monotonic, target_time, rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(total, expected_total, rel_tol=1e-9, abs_tol=1e-6)
            or not math.isclose(busy, expected_busy, rel_tol=1e-9, abs_tol=1e-6)
        ):
            raise failure(index, "top-level counters do not match cpu counter bracket interpolation")

    # These witnesses invalidate the full arm, including intervals that would
    # otherwise be excluded from the sustained throughput window.
    for index, sample in enumerate(samples):
        validate_sample(index, sample)
        if sample["target"]["ownership_changed"]:
            raise RuntimeError("benchmark PID/PGID ownership changed during arm")
        if any(sample["contamination"].values()):
            raise RuntimeError(
                "transient competing llama/AutoPilot/KFD contamination detected: "
                f"sample={index} contamination={sample['contamination']}"
            )

    intervals: list[dict[str, Any]] = []
    for index, (before, after) in enumerate(zip(samples, samples[1:])):
        try:
            elapsed = float(after["monotonic"]) - float(before["monotonic"])
            total_delta = float(after["cpu_total_ticks"]) - float(before["cpu_total_ticks"])
            busy_delta = float(after["cpu_busy_ticks"]) - float(before["cpu_busy_ticks"])
            target_delta = int(after["target"]["cpu_ticks"]) - int(before["target"]["cpu_ticks"])
            swap_delta = {
                key: int(after["swap"][key]) - int(before["swap"][key])
                for key in ("pswpin", "pswpout")
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"contention monitor sampling failure at interval {index}") from exc

        # Aggregate and target counters must be monotonic.  In contrast,
        # target_delta > busy_delta is retained as signed timing telemetry.
        if not all(math.isfinite(value) for value in (elapsed, total_delta, busy_delta)):
            raise RuntimeError(f"contention monitor observed non-finite counter interval: {index}")
        if elapsed <= 0 or total_delta <= 0 or busy_delta < 0 or target_delta < 0:
            raise RuntimeError(f"contention monitor observed invalid counter interval: {index}")
        if busy_delta > total_delta:
            raise RuntimeError(f"contention monitor observed invalid counter interval: {index}")
        if any(value < 0 for value in swap_delta.values()):
            raise RuntimeError(f"contention monitor observed invalid swap counter interval: {index}")
        if any(value != 0 for value in swap_delta.values()):
            raise RuntimeError(f"swap I/O changed during benchmark arm: interval={index} delta={swap_delta}")

        target_core_equivalents = target_delta / (CLOCK_TICKS * elapsed)
        signed_external = (busy_delta - target_delta) / (CLOCK_TICKS * elapsed)
        exclusions: list[str] = []
        if target_core_equivalents < MIN_TARGET_CORE_EQUIVALENTS:
            exclusions.append("target_below_minimum_core_equivalents")
        intervals.append({
            "index": index,
            "start_sample_index": index,
            "end_sample_index": index + 1,
            "elapsed_s": elapsed,
            "aggregate_total_delta_ticks": total_delta,
            "aggregate_busy_delta_ticks": busy_delta,
            "target_group_cpu_delta_ticks": target_delta,
            "target_core_equivalents": target_core_equivalents,
            "signed_external_core_equivalents": signed_external,
            "swap_delta": swap_delta,
            "exclusion_reasons": exclusions,
            "eligible": not exclusions,
        })

    # The qualifying window consists of adjacent eligible intervals.  Iterate
    # in order and retain the first run on equal duration, as ratified.
    best: tuple[int, int, float] | None = None
    run_start: int | None = None
    duration = 0.0
    for index, interval in enumerate(intervals):
        if interval["eligible"]:
            if run_start is None:
                run_start = index
                duration = 0.0
            duration += float(interval["elapsed_s"])
            if best is None or duration > best[2]:
                best = (run_start, index, duration)
        else:
            run_start = None
            duration = 0.0
    if best is None or best[2] < MIN_SUSTAINED_WINDOW_SECONDS:
        raise RuntimeError(
            "contention monitor lacks a qualifying sustained eligible window: "
            f"required_seconds={MIN_SUSTAINED_WINDOW_SECONDS} intervals={intervals}"
        )

    start, end, duration = best
    before = samples[start]
    after = samples[end + 1]
    direct_endpoint_elapsed = float(after["monotonic"]) - float(before["monotonic"])
    if not math.isclose(direct_endpoint_elapsed, duration, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError(
            "contention monitor sustained window endpoint elapsed does not match interval duration: "
            f"endpoint={direct_endpoint_elapsed} summed={duration}"
        )
    endpoint_busy_delta = float(after["cpu_busy_ticks"]) - float(before["cpu_busy_ticks"])
    endpoint_target_delta = int(after["target"]["cpu_ticks"]) - int(before["target"]["cpu_ticks"])
    signed_external = (endpoint_busy_delta - endpoint_target_delta) / (CLOCK_TICKS * duration)
    if not MIN_SIGNED_EXTERNAL_CORE_EQUIVALENTS <= signed_external <= MAX_EXTERNAL_CORE_EQUIVALENTS:
        raise RuntimeError(
            "sustained external CPU contention is outside inclusive bounds "
            f"[{MIN_SIGNED_EXTERNAL_CORE_EQUIVALENTS}, {MAX_EXTERNAL_CORE_EQUIVALENTS}]: "
            f"{signed_external}"
        )
    return {
        "status": "pass",
        "accounting": CONTENTION_ACCOUNTING,
        "thresholds": {
            "configured_cpu_count": CONFIGURED_CPU_COUNT,
            "minimum_target_core_equivalents": MIN_TARGET_CORE_EQUIVALENTS,
            "minimum_window_seconds": MIN_SUSTAINED_WINDOW_SECONDS,
            "signed_external_core_equivalents": {
                "minimum": MIN_SIGNED_EXTERNAL_CORE_EQUIVALENTS,
                "maximum": MAX_EXTERNAL_CORE_EQUIVALENTS,
            },
        },
        "samples": samples,
        "intervals": intervals,
        "sustained_window": {
            "start_interval_index": start,
            "end_interval_index": end,
            "start_sample_index": start,
            "end_sample_index": end + 1,
            "start_monotonic": float(before["monotonic"]),
            "end_monotonic": float(after["monotonic"]),
            "elapsed_s": duration,
            "direct_endpoint_elapsed_s": direct_endpoint_elapsed,
            "aggregate_total_delta_ticks": float(after["cpu_total_ticks"]) - float(before["cpu_total_ticks"]),
            "aggregate_busy_delta_ticks": endpoint_busy_delta,
            "target_group_cpu_delta_ticks": endpoint_target_delta,
            "target_core_equivalents": endpoint_target_delta / (CLOCK_TICKS * duration),
            "signed_external_core_equivalents": signed_external,
        },
    }


def run_monitored(argv: list[str], env: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    selected = exact_subprocess_environment(env)
    proc = subprocess.Popen(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=selected, start_new_session=True)
    pgid = os.getpgid(proc.pid)
    samples: list[dict[str, Any]] = []
    monitor_error: Exception | None = None
    while True:
        try:
            samples.append(monitor_snapshot(proc.pid, pgid))
        except Exception as exc:
            monitor_error = exc
            break
        # Sample before polling: poll() reaps a completed leader, which would
        # destroy the terminal PGID/CPU-accounting witness.
        if proc.poll() is not None:
            break
        time.sleep(MONITOR_INTERVAL_S)
    if monitor_error is not None and proc.poll() is None:
        try:
            os.killpg(pgid, 15)
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, 9)
            proc.wait(timeout=30)
    stdout, stderr = proc.communicate()
    if monitor_error is None:
        try:
            monitor = validate_monitor_samples(samples)
        except Exception as exc:
            monitor_error = exc
            monitor = {"status": "fail", "samples": samples, "error": repr(exc)}
    else:
        monitor = {"status": "fail", "samples": samples, "error": repr(monitor_error)}
    completed = subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
    if monitor_error is not None:
        raise RuntimeError(f"continuous arm contention monitor failed: {monitor}")
    return completed, monitor


def run_arm(cell: dict[str, Any], arm: dict[str, Any], artifact: Path) -> dict[str, Any]:
    argv = argv_for_cell(cell, arm, dry_run=False)
    parent_environment = canonical_parent_environment()
    before_identity = collect_arm_identity(arm)
    completed, monitor = run_monitored(argv, parent_environment)
    monitor_path = artifact.with_suffix(".contention_monitor.json")
    write_json(monitor_path, monitor)
    stdout_path = artifact.with_suffix(".stdout.json")
    stderr_path = artifact.with_suffix(".stderr.md")
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"canonical wrapper failed for {cell.get('pair_id', cell.get('id'))} {arm['name']}"
        )
    after_identity = collect_arm_identity(arm)
    if arm_binding(before_identity) != arm_binding(after_identity):
        raise RuntimeError("release artifacts changed during benchmark arm")
    environment_witness = canonical_environment_witness(
        completed.stderr,
        cell,
        arm,
    )
    parsed = parse_result(completed.stdout, cell["metric"], arm["actual_head"])
    measurement_window = measurement_window_observation(monitor, parsed["samples_ns"])
    witness = build_witness(completed.stderr, arm["actual_head"])
    if parsed["build_commit"] != witness["commit"] or str(parsed["build_number"]) != witness["build_number"]:
        raise RuntimeError(f"JSON and Markdown build witnesses disagree: {parsed} / {witness}")
    json_resolved = resolve_build_commit(
        Path(arm["source_root"]),
        parsed["build_commit"],
        arm["actual_head"],
    )
    markdown_resolved = resolve_build_commit(
        Path(arm["source_root"]),
        witness["commit"],
        arm["actual_head"],
    )
    return {
        "arm": arm["name"],
        "argv": argv,
        "raw_stdout": str(stdout_path),
        "raw_stderr": str(stderr_path),
        "parent_environment_identity": parent_environment_identity(),
        "canonical_environment_witness": environment_witness,
        "contention_monitor": monitor,
        "contention_monitor_path": str(monitor_path),
        "measurement_window_observation": measurement_window,
        "witness": {
            **witness,
            "json_resolved_full_commit": json_resolved,
            "markdown_resolved_full_commit": markdown_resolved,
        },
        **stats(parsed["samples_ts"]),
    }


def verdict(cell: dict[str, Any], production: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ratio = safe_ratio(
        candidate["median_ts"],
        production["median_ts"],
        f"{cell.get('pair_id', 'pair')} initial",
    )
    if cell["iq"] and cell["iqk"] == 1:
        # P-BENCH-PREFILL-1 evaluates a newly enabled IQ path over its paired
        # prefill+decode metrics, not one metric at a time.
        return {"ratio": ratio, "state": "iq_pair_pending", "threshold": "IQ utility paired by model"}
    if ratio >= 0.98:
        state = "pass"
    elif ratio < 0.95:
        state = "fail"
    else:
        state = "gray_retry_required"
    return {
        "ratio": ratio,
        "state": state,
        "threshold": "non-inferiority >= 0.98; < 0.95 fail; gray retry in [0.95, 0.98)",
    }


def run_pair(
    pair: dict[str, Any],
    arms: dict[str, dict[str, Any]],
    model: dict[str, Any],
    output: Path,
    order: tuple[str, str],
) -> dict[str, Any]:
    pair_dir = output / pair["pair_id"] / ("-then-".join(order))
    pair_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "pair_id": pair["pair_id"],
        "order": list(order),
        "started_at": utc_now(),
        "pre_host": host_snapshot(),
    }
    error: Exception | None = None
    try:
        require_clean_host(record["pre_host"], "before pair")
        record["cache_preparation"] = prepare_pair(
            [Path(item["path"]) for item in model["shards"]]
        )
        record["arms"] = {
            name: run_arm(pair, arms[name], pair_dir / f"{name}.log") for name in order
        }
        record["verdict"] = verdict(
            pair, record["arms"]["production"], record["arms"]["candidate"]
        )
    except Exception as exc:  # preserve the pair evidence before propagating
        error = exc
        record["error"] = repr(exc)
    finally:
        try:
            record["post_host"] = host_snapshot()
            require_clean_host(record["post_host"], "after pair")
        except Exception as exc:
            record["post_host_error"] = repr(exc)
            if error is None:
                error = exc
        record["finished_at"] = utc_now()
        write_json(pair_dir / "pair_summary.json", record)
    if error is not None:
        raise RuntimeError(
            f"pair {pair['pair_id']} failed; durable evidence: {pair_dir / 'pair_summary.json'}"
        ) from error
    return record


def apply_gray_retry(initial: dict[str, Any], retry: dict[str, Any]) -> dict[str, Any]:
    p = initial["arms"]["production"]["samples_ts"] + retry["arms"]["production"]["samples_ts"]
    c = initial["arms"]["candidate"]["samples_ts"] + retry["arms"]["candidate"]["samples_ts"]
    production = stats(p)
    candidate = stats(c)
    ratio = safe_ratio(
        candidate["median_ts"],
        production["median_ts"],
        "gray retry pooled",
    )
    return {"production": production, "candidate": candidate, "ratio": ratio, "state": "pass" if ratio >= 0.98 else "fail"}


def evaluate_iq_utility(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the IQ release rule across paired tg128 + pp2048 evidence.

    IQK=0 is retained as a control arm.  The newly enabled route is IQK=1;
    it needs both ratios >= .95 and at least one >= 1.05 for every IQ model.
    """
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        pair = record["pair"]
        if pair["iq"] and pair["iqk"] == 1:
            grouped.setdefault((pair["model"], pair["iqk"]), []).append(record)
    results: list[dict[str, Any]] = []
    for (model, iqk), group in sorted(grouped.items()):
        if {item["pair"]["metric"] for item in group} != {"tg128", "pp2048"}:
            raise RuntimeError(f"incomplete IQ metric pair for {model} iqk={iqk}")
        ratios = {item["pair"]["metric"]: effective_ratio(item) for item in group}
        state = "pass" if min(ratios.values()) >= 0.95 and max(ratios.values()) >= 1.05 else "fail"
        results.append({"model": model, "iqk": iqk, "ratios": ratios, "state": state,
                        "rule": "IQK=1: neither ratio < 0.95 and at least one >= 1.05"})
    return results


def _initial_candidate_attribution_cell(
    record: dict[str, Any],
) -> dict[str, Any]:
    pair = record["pair"]
    try:
        candidate = record["initial"]["arms"]["candidate"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"{pair.get('pair_id')} lacks its initial candidate arm"
        ) from exc
    samples = candidate.get("samples_ts")
    if not isinstance(samples, list) or len(samples) != REPS:
        raise RuntimeError(
            f"{pair.get('pair_id')} candidate attribution requires exactly "
            f"{REPS} initial samples"
        )
    computed = stats(samples)
    recorded_median = require_finite_positive(
        candidate.get("median_ts"),
        f"{pair.get('pair_id')} candidate median",
    )
    recorded_mad = candidate.get("mad_ts")
    if (
        isinstance(recorded_mad, bool)
        or not isinstance(recorded_mad, (int, float))
        or not math.isfinite(float(recorded_mad))
        or float(recorded_mad) < 0
    ):
        raise RuntimeError(
            f"{pair.get('pair_id')} candidate MAD is invalid: {recorded_mad!r}"
        )
    if (
        recorded_median != computed["median_ts"]
        or float(recorded_mad) != computed["mad_ts"]
    ):
        raise RuntimeError(
            f"{pair.get('pair_id')} candidate summary disagrees with its samples"
        )
    return {
        "pair_id": pair["pair_id"],
        "n": len(samples),
        "median_ts": computed["median_ts"],
        "mad_ts": computed["mad_ts"],
        "samples_ts": computed["samples_ts"],
    }


def evaluate_candidate_iqk_attribution(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_keys = {
        (model.name, metric, iqk)
        for model in MODELS
        if model.iq
        for metric in ("pp2048", "tg128")
        for iqk in (0, 1)
    }
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in records:
        pair = record.get("pair")
        if not isinstance(pair, dict):
            raise RuntimeError("attribution record is missing pair metadata")
        if not pair.get("iq"):
            continue
        iqk = pair.get("iqk")
        if isinstance(iqk, bool) or iqk not in (0, 1):
            raise RuntimeError(f"invalid IQ attribution gate value: {iqk!r}")
        key = (pair.get("model"), pair.get("metric"), iqk)
        if key in indexed:
            raise RuntimeError(f"duplicate candidate IQK attribution cell: {key}")
        indexed[key] = record
    if set(indexed) != expected_keys:
        missing = sorted(expected_keys - set(indexed), key=repr)
        unexpected = sorted(set(indexed) - expected_keys, key=repr)
        raise RuntimeError(
            "candidate IQK attribution cardinality drift: "
            f"expected={len(expected_keys)} actual={len(indexed)} "
            f"missing={missing} unexpected={unexpected}"
        )

    cells: list[dict[str, Any]] = []
    for model in sorted(model.name for model in MODELS if model.iq):
        for metric in ("pp2048", "tg128"):
            iqk0 = _initial_candidate_attribution_cell(
                indexed[(model, metric, 0)]
            )
            iqk1 = _initial_candidate_attribution_cell(
                indexed[(model, metric, 1)]
            )
            cells.append(
                {
                    "model": model,
                    "metric": metric,
                    "iqk0": iqk0,
                    "iqk1": iqk1,
                    "ratio_iqk1_over_iqk0": safe_ratio(
                        iqk1["median_ts"],
                        iqk0["median_ts"],
                        f"{model} {metric} candidate IQK1/IQK0",
                    ),
                }
            )
    if len(cells) != 6:
        raise RuntimeError(
            f"candidate IQK attribution emitted {len(cells)} ratios, expected six"
        )
    return {
        "status": "valid",
        "promotion_gate": False,
        "sample_scope": "initial_28_arm_matrix_only",
        "ratio_direction": "candidate_iqk1_median / candidate_iqk0_median",
        "cells": cells,
    }


def effective_ratio(record: dict[str, Any]) -> float:
    return require_finite_positive(
        record["pooled"]["ratio"]
        if "pooled" in record
        else record["initial"]["verdict"]["ratio"],
        f"{record['pair']['pair_id']} effective ratio",
    )


def collect_throughput_failures(
    records: list[dict[str, Any]],
    iq_utility: list[dict[str, Any]],
) -> list[str]:
    failures = [
        item["pair"]["pair_id"]
        for item in records
        if item["pair"]["iqk"] == 0 or not item["pair"]["iq"]
        if item.get("pooled", item["initial"]["verdict"])["state"] != "pass"
    ]
    failures += [
        f"iq-utility:{item['model']}"
        for item in iq_utility
        if item["state"] != "pass"
    ]
    return failures


def manifest() -> dict[str, Any]:
    cells = build_cells()
    pairs = build_pairs(cells)
    q8_waiver = q8_waiver_attestation()
    return {"schema": "cpu-prefill-v8-regression.v3", "protocol": "P-BENCH-PREFILL-1", "created_at": utc_now(),
            "measurement_intent": "decision_gating_throughput_only", "promotion_decision": False,
            "contention_accounting": {
                "id": CONTENTION_ACCOUNTING,
                "configured_cpu_count": CONFIGURED_CPU_COUNT,
                "minimum_target_core_equivalents": MIN_TARGET_CORE_EQUIVALENTS,
                "minimum_window_seconds": MIN_SUSTAINED_WINDOW_SECONDS,
                "signed_external_core_equivalents": {
                    "minimum": MIN_SIGNED_EXTERNAL_CORE_EQUIVALENTS,
                    "maximum": MAX_EXTERNAL_CORE_EQUIVALENTS,
                },
                "prospective_only": True,
            },
            "explicit_exclusion": [
                "qwen3.5-122b",
                "qwen36_q8 (operator-ratified WAIVE-Q8; no v8 Q8 non-regression claim)",
            ],
            "q8_waiver": q8_waiver,
            "parent_environment_identity": parent_environment_identity(),
            "instrument_eras": instrument_era_attestation(),
            "profile": {"prefill": PREFILL, "reps": REPS, "cpu_extra": list(CPU_EXTRA), "output_extra": list(OUTPUT_EXTRA), "initial_order": ["production", "candidate"], "gray_retry_order": ["candidate", "production"], "gray_retry_scope": "non-IQ regression pairs and IQK=0 control pairs only"},
            "arms": {name: arm_spec(name) for name in ("production", "candidate")},
            "models": {model.name: {"path": str(model.path), "iq": model.iq} for model in MODELS},
            "arm_runs": cells, "pairs": pairs,
            "cardinality": {"arm_runs": 28, "unique_pairs": 14}}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False)


def arm_binding(identity: dict[str, Any]) -> dict[str, Any]:
    shared_library_identity = identity["shared_library_identity"]
    libraries = shared_library_identity["libraries"]
    return {
        "name": identity["name"],
        "source_root": identity["source_root"],
        "actual_head": identity["actual_head"],
        "actual_branch": identity["actual_branch"],
        "source_status": identity["source_status"],
        "binary_identity": identity["binary_identity"],
        "openmp_runtime": shared_library_identity["openmp_runtime"],
        "libraries": sorted(
            [
                {
                    key: item[key]
                    for key in ("soname", "resolved_target", "bytes", "sha256")
                }
                for item in libraries
            ],
            key=lambda item: (item["soname"], item["resolved_target"]),
        ),
    }


def collect_bound_inputs(
    plan: dict[str, Any],
    attestation_paths: dict[str, Path],
) -> dict[str, Any]:
    arms = {
        name: collect_arm_identity(spec) for name, spec in plan["arms"].items()
    }
    models = {
        name: model_identity(Path(spec["path"])) for name, spec in plan["models"].items()
    }
    return {
        "arms": arms,
        "models": models,
        "harness": harness_identities(),
        "attestations": attestation_identities(attestation_paths),
        "q8_waiver": q8_waiver_attestation(),
    }


def stable_input_binding(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "arms": {
            name: arm_binding(identity)
            for name, identity in sorted(inputs["arms"].items())
        },
        "models": inputs["models"],
        "harness": inputs["harness"],
        "attestations": inputs["attestations"],
        "q8_waiver": inputs["q8_waiver"],
    }


def require_identical_bound_inputs(
    preflight: dict[str, Any],
    postflight: dict[str, Any],
) -> None:
    before = stable_input_binding(preflight)
    after = stable_input_binding(postflight)
    if before != after:
        raise RuntimeError(
            "bound benchmark inputs mutated during execution: "
            + json.dumps(
                {"preflight": before, "postflight": after},
                sort_keys=True,
                allow_nan=False,
            )
        )


def execute(
    plan: dict[str, Any],
    output: Path,
    attestation_paths: dict[str, Path],
) -> dict[str, Any]:
    current_waiver = q8_waiver_attestation()
    if plan.get("q8_waiver") != current_waiver:
        raise RuntimeError("plan Q8 waiver binding does not match the exact ratified waiver")
    preflight_inputs = collect_bound_inputs(plan, attestation_paths)
    arms = preflight_inputs["arms"]
    models = preflight_inputs["models"]
    verified_attestations = validate_external_attestations(
        attestation_paths,
        arms,
        models,
        require_fresh_host=True,
    )
    pairs = build_pairs(plan["arm_runs"])
    result: dict[str, Any] = {
        "schema": plan["schema"],
        "protocol": plan["protocol"],
        "started_at": utc_now(),
        "plan": plan,
        "attestations": preflight_inputs["attestations"],
        "external_gates": {
            "correctness": verified_attestations["correctness"],
            "coherence": verified_attestations["coherence"],
            "numerical_safety": verified_attestations["numerical_safety"],
            "host": verified_attestations["host"],
        },
        "promotion_decision": False,
        "arm_identity": arms,
        "model_identity": models,
        "harness_identity": preflight_inputs["harness"],
        "pair_results": [],
    }
    for pair in pairs:
        initial = run_pair(
            pair,
            arms,
            models[pair["model"]],
            output,
            ("production", "candidate"),
        )
        record: dict[str, Any] = {"pair": pair, "initial": initial}
        if initial["verdict"]["state"] == "gray_retry_required":
            retry = run_pair(
                pair,
                arms,
                models[pair["model"]],
                output,
                ("candidate", "production"),
            )
            record["gray_retry"] = retry
            record["pooled"] = apply_gray_retry(initial, retry)
        result["pair_results"].append(record)
        write_json(output / "summary.partial.json", result)
    result["candidate_iqk_attribution"] = evaluate_candidate_iqk_attribution(
        result["pair_results"]
    )
    result["iq_utility"] = evaluate_iq_utility(result["pair_results"])
    throughput_failures = collect_throughput_failures(
        result["pair_results"], result["iq_utility"]
    )
    result["throughput_failures"] = throughput_failures
    result["throughput_status"] = "pass" if not throughput_failures else "fail"
    result["postflight_inputs"] = collect_bound_inputs(plan, attestation_paths)
    try:
        result["postflight_external_gates"] = validate_external_attestations(
            attestation_paths,
            result["postflight_inputs"]["arms"],
            result["postflight_inputs"]["models"],
            require_fresh_host=False,
        )
        require_identical_bound_inputs(
            preflight_inputs,
            result["postflight_inputs"],
        )
        result["input_binding_status"] = "identical"
    except Exception as exc:
        result["input_binding_status"] = "mutated"
        result["input_binding_error"] = repr(exc)
        write_json(output / "summary.partial.json", result)
        raise
    result["finished_at"] = utc_now()
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Run cache preparation and all 28 live cells. Default only writes plan.json.")
    mode.add_argument("--write-host-attestation", type=Path)
    parser.add_argument("--host-attestation-path", type=Path)
    parser.add_argument("--correctness-attestation-path", type=Path)
    parser.add_argument("--coherence-attestation-path", type=Path)
    parser.add_argument("--numerical-safety-attestation-path", type=Path)
    args = parser.parse_args(argv)
    if args.write_host_attestation is not None:
        incompatible = {
            "--output-dir": args.output_dir,
            "--host-attestation-path": args.host_attestation_path,
            "--correctness-attestation-path": args.correctness_attestation_path,
            "--coherence-attestation-path": args.coherence_attestation_path,
            "--numerical-safety-attestation-path": args.numerical_safety_attestation_path,
        }
        supplied = [flag for flag, value in incompatible.items() if value is not None]
        if supplied:
            parser.error(
                "--write-host-attestation is a standalone mode and cannot be combined with: "
                + ", ".join(supplied)
            )
    if args.execute:
        required = {
            "--host-attestation-path": args.host_attestation_path,
            "--correctness-attestation-path": args.correctness_attestation_path,
            "--coherence-attestation-path": args.coherence_attestation_path,
            "--numerical-safety-attestation-path": args.numerical_safety_attestation_path,
        }
        missing = [
            flag for flag, value in required.items() if value is None or not value.is_file()
        ]
        if missing:
            parser.error(
                "--execute requires existing durable attestation artifact files: "
                + ", ".join(missing)
            )
    return args


def fresh_output_dir(requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
        requested = (
            RESEARCH_ROOT
            / "data"
            / "kernel-v8-candidate"
            / "cpu-prefill-regression"
            / stamp
        )
    if requested.exists() and any(requested.iterdir()):
        raise RuntimeError(f"output directory is nonempty; refusing mixed artifacts: {requested}")
    requested.mkdir(parents=True, exist_ok=True)
    return requested


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.write_host_attestation is not None:
        write_host_attestation(args.write_host_attestation)
        print(json_text({
            "mode": "host_attestation_written",
            "host_attestation": str(args.write_host_attestation),
            "promotion_decision": False,
        }))
        return 0
    output = fresh_output_dir(args.output_dir)
    plan = manifest()
    write_json(output / "plan.json", plan)
    if not args.execute:
        print(json_text({"mode": "prepared", "plan": str(output / "plan.json"), "arm_runs": len(plan["arm_runs"]), "pairs": len(plan["pairs"]), "execute_required": True, "promotion_decision": False}))
        return 0
    attestation_paths = {
        "host": args.host_attestation_path,
        "correctness": args.correctness_attestation_path,
        "coherence": args.coherence_attestation_path,
        "numerical_safety": args.numerical_safety_attestation_path,
    }
    try:
        result = execute(plan, output, attestation_paths)
    except Exception as exc:
        failure = {
            "schema": plan["schema"],
            "protocol": plan["protocol"],
            "finished_at": utc_now(),
            "throughput_status": "invalid",
            "promotion_decision": False,
            "attestation_paths": {
                role: str(path) for role, path in attestation_paths.items()
            },
            "error": repr(exc),
        }
        write_json(output / "summary.json", failure)
        print(json_text({"throughput_status": "invalid", "promotion_decision": False, "summary": str(output / "summary.json"), "error": repr(exc)}))
        return 2
    write_json(output / "summary.json", result)
    print(json_text({"throughput_status": result["throughput_status"], "promotion_decision": False, "summary": str(output / "summary.json"), "throughput_failures": result["throughput_failures"]}))
    return 1 if result["throughput_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
