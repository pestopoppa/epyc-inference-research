#!/usr/bin/env python3
"""P-GPU-1 KV-cache quantization sweep, Qwen3.8-27B-Q8_0 on MI210, production v10.

Question: does KV-cache quantization cost decode on the MI210 at the v10
production kernel?  Nothing exists at v10 -- the newest GPU KV numbers are v7/v8/v9
era and the v10 promotion bundled a BIOS change (memory interleave ON, 5600 MT/s),
so no prior comparison crosses that boundary.

Matrix: 3 homogeneous KV arms (f16/f16, q8_0/q8_0, q4_0/q4_0)
      x 5 replicates
      x 2 prefill depths (~2k and ~32k)
      = 30 fresh llama-server launches (one per replicate; no resident server).

Two things are held FIXED and the script REFUSES to run if either is violated:

  * ``-fa`` is pinned ``on`` in every arm.  Measured on this host at MI210, the
    same mixed q4_0/f16 KV at pp4096 gave 372.70 t/s at ``-fa 1`` vs 567.23 at
    ``-fa 0`` -- a 1.52x swing from the flag alone
    (``data/gpu-mi210/axa2_mixed_kv_fa_matrix_current_build_20260719T073441Z/summary.md``).
    An arm set that does not hold ``-fa`` fixed is measuring flash attention, not
    KV quantization.
  * K and V types are IDENTICAL within an arm.  ``GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF``
    in the live v10 GPU build makes mixed K/V flash-attn ineligible; on this host
    mixed arms fell back to CPU and were watchdog-killed.

The binary is resolved through the kernel store (``/mnt/raid0/llm/kernels/production/gpu``),
never a hardcoded build path and never the frozen source tree's ``build-hip/`` --
that still holds the v9 binaries.  Being run on the production-named kernel is what
makes the result P-GPU-1 decision-grade instead of observation-only.

Dry-run is the default and touches neither the GPU nor any process: it resolves the
store, hashes the pinned artifacts, and prints the full arm matrix and argv.
``--execute`` is required to launch anything, and it acquires the MI210 device claim
(flock) for the whole window -- a claim is ACQUIRED, never observed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laguna_pgpu1_dflash_runner as common

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RESEARCH_ROOT / "scripts"))

# ---------------------------------------------------------------------------
# Kernel store -- the binary is RESOLVED, never hardcoded.
# ---------------------------------------------------------------------------
KERNEL_STORE_GPU = Path("/mnt/raid0/llm/kernels/production/gpu")
SERVER_BASENAME = "llama-server"
FROZEN_SOURCE_TREE = Path("/mnt/raid0/llm/llama.cpp")
EXPECTED_BRANCH = "production-consolidated-v10"
EXPECTED_COMMIT = "ffc1bac82eeca6f9099e1ccd9ba49703c460a115"
EXPECTED_VERSION_LINE = "version: 10303 (ffc1bac82)"
EXPECTED_SERVER_SHA256 = "2b49713f0e3c022132393bf302dab954a1e2c1974b91c28d01ce2d192abdfa7b"
# Keyed by the SONAME ldd reports; the sha256 is of the resolved real file.
# Source: kernels/builds/gpu-20260921-ffc1bac82/SHA256SUMS (bin/*).
EXPECTED_LIBRARY_SHA256 = {
    "libggml-base.so.0": "92cc3dfd7e1119b7bcc83c18f758099c47363073013a896ee76efcd6ad19145a",
    "libggml-cpu.so.0": "3f4edd2fc0f1e78003735be59db987d43c4bc0e6a8f821bc419d7b6f492346c0",
    "libggml-hip.so.0": "f26a166b8ea3089d19a395ae00afa22e917df8eba71ca6ef4be5235d65ea8079",
    "libggml.so.0": "8709bbacc79b0d4faf43d66370a5db73d9ab378995f1f0964c03df8af6422639",
    "libllama-common.so.0": "0f33ec5cc9967c14c8eeb2d560d6d827301c50cf42dfd701cc1752cc4a7bf9cb",
    "libllama-server-impl.so": "d9f143d7b056c0bc24a1822fb309ddd01c96df2b63b1b1bc61791e97f519e96f",
    "libllama.so.0": "808a4de5ad3f9754f1c3c70bdd87da3418aa73b3b39d26e4dc38ba4c75b5c60a",
    "libmtmd.so.0": "e244e0054e255698a57da7ffab2c4b4d42f98716f543c2388c4e923464743b05",
}
# Non-vacuity: a linkage receipt that inspected nothing is vacuous (P-GPU-1 field 3).
REQUIRED_LIBRARY_SONAMES = (
    "libggml-base.so.0",
    "libggml-cpu.so.0",
    "libggml-hip.so.0",
    "libggml.so.0",
    "libllama.so.0",
)
# The frozen production SOURCE tree's own build dirs still hold v9 binaries.
FORBIDDEN_BINARY_PARENTS = (
    FROZEN_SOURCE_TREE / "build-hip" / "bin",
    FROZEN_SOURCE_TREE / "build" / "bin",
)
# Untracked paths in the frozen production tree that are NOT llama-server build inputs
# and therefore cannot change the binary under test (inherited verbatim from
# laguna_pgpu1_dflash_runner.SOURCE_UNTRACKED_ALLOWLIST).
SOURCE_UNTRACKED_ALLOWLIST = {
    ".gitnexusignore": "local GitNexus configuration; not a llama.cpp build input",
    "tools/math-tools/": "operator-owned unrelated tool subtree; not linked into llama-server",
}
LINKAGE_VERIFIER = RESEARCH_ROOT / "scripts/utils/verify_ggml_linkage.sh"
LINKAGE_VERIFIER_ID = "epyc-inference-research/scripts/utils/verify_ggml_linkage.sh"

# ---------------------------------------------------------------------------
# Device claim -- ACQUIRED, not observed.
# ---------------------------------------------------------------------------
GPU_DEVICE_LOCK = Path("/mnt/raid0/llm/tmp/gpu_device.mi210_0.lock")
GPU_DEVICE_ID = "mi210_0"

# ---------------------------------------------------------------------------
# Model / recipe.
# ---------------------------------------------------------------------------
DEFAULT_TARGET_MODEL = Path("/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf")
TARGET_MODEL_BYTES = 29_047_086_048
# No sha256 for this GGUF is recorded anywhere in the repos as of 2026-09-22, so it
# is CAPTURED at --execute and written into the artifact rather than pre-pinned.
# Once a run has banked one, the owning session may pin it here.
TARGET_MODEL_SHA256: str | None = None

DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / "data/gpu-mi210/kv-quant-27b-v10-sweep"

REPS = 5
CONTEXT = 65536
FLASH_ATTENTION = "on"
MAX_TOKENS = common.DEFAULT_MAX_TOKENS
MIN_COMPLETION_TOKENS = common.DEFAULT_MIN_COMPLETION_TOKENS
SEED = common.DEFAULT_SEED
PORT_BASE = 19960
SETTLEMENT_TIMEOUT_S = 30.0
SETTLEMENT_POLL_INTERVAL_S = 1.0

FA_FIXED_REASON = (
    "-fa is pinned 'on' in every arm and MUST NOT be varied: measured on this host at "
    "MI210, the same mixed q4_0/f16 KV at pp4096 gave 372.70 t/s at -fa 1 vs 567.23 at "
    "-fa 0, a 1.52x swing from the flag alone "
    "(data/gpu-mi210/axa2_mixed_kv_fa_matrix_current_build_20260719T073441Z/summary.md). "
    "An arm set that does not hold -fa fixed is measuring flash attention, not KV "
    "quantization."
)
MIXED_KV_REFUSAL = (
    "mixed K/V cache types are REFUSED: GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF in the live v10 "
    "GPU build (kernels/builds/gpu-20260921-ffc1bac82/CMakeCache.txt:450) makes mixed K/V "
    "flash-attention ineligible; on this host mixed arms fell back to CPU and were "
    "watchdog-killed (0% GPU / 60% VRAM / 0 stdout bytes at 182s). Only homogeneous "
    "cache_k == cache_v arms are admissible here."
)


@dataclass(frozen=True)
class Cell:
    """An arm. It deliberately carries NO flash-attention field: see FA_FIXED_REASON."""

    name: str
    cache_k: str
    cache_v: str


@dataclass(frozen=True)
class Depth:
    name: str
    target_prefill_tokens: int
    min_prefill_tokens: int
    max_prefill_tokens: int


CELLS = (
    Cell("A_f16_kv", "f16", "f16"),
    Cell("B_q8_0_kv", "q8_0", "q8_0"),
    Cell("C_q4_0_kv", "q4_0", "q4_0"),
)

# A short-context-only sweep would miss the regime entirely -- prior runs saw KV
# effects only at depth. Bands are +/-20% of target and gate the measured
# prompt_tokens; the actual count is always recorded.
DEPTHS = (
    Depth("d2k", 2048, 1638, 2458),
    Depth("d32k", 32768, 26214, 39322),
)

TOTAL_LAUNCHES = len(CELLS) * REPS * len(DEPTHS)

CHARS_PER_TOKEN = 4.0
FILLER_WORDS = (
    "alpha", "beacon", "canyon", "delta", "ember", "fathom", "granite", "harbor",
    "ivory", "jasper", "kelvin", "lattice", "meridian", "nimbus", "obsidian", "pylon",
    "quartz", "ridgeline", "sextant", "tundra", "umbral", "vector", "wicket", "xenon",
    "yarrow", "zenith", "anvil", "bramble", "cobalt", "driftwood", "estuary", "flint",
    "gantry", "hollow", "inlet", "junction", "keystone", "lumen", "mantle", "nocturne",
    "orbit", "parapet", "quiver", "rampart", "solstice", "thicket", "underpass", "vellum",
    "windlass", "xylem", "yeoman", "zephyr", "arbor", "basalt", "cistern", "dolmen",
    "echelon", "furrow", "gable", "heliotrope", "isthmus", "jetty", "krill", "lodestone",
)
FILLER_WORDS_PER_LINE = 12
FILLER_HEADER = (
    "REFERENCE LOG (context ballast, machine-generated, deterministic). "
    "It carries no information relevant to the task. Do not summarize it, quote it, "
    "or refer to it. Read past it and answer only the TASK below.\n"
)
FILLER_FOOTER = "\nEND REFERENCE LOG\n\nTASK:\n"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Guards. These refuse; they do not warn.
# ---------------------------------------------------------------------------
def homogeneous_kv_guard(cells: tuple[Cell, ...] = CELLS) -> tuple[bool, str]:
    mixed = [cell.name for cell in cells if cell.cache_k != cell.cache_v]
    if mixed:
        return False, f"{MIXED_KV_REFUSAL} Offending arm(s): {', '.join(mixed)}."
    return True, "ok"


def flash_attention_guard(argvs: list[list[str]]) -> tuple[bool, str]:
    """Every emitted argv must carry exactly one `-fa on`, and never anything else."""
    observed: set[str] = set()
    for argv in argvs:
        if argv.count("-fa") != 1:
            return False, f"{FA_FIXED_REASON} An argv carried {argv.count('-fa')} -fa flags."
        observed.add(argv[argv.index("-fa") + 1])
        if "--flash-attn" in argv or "-fa" in argv[argv.index("-fa") + 1:]:
            return False, f"{FA_FIXED_REASON} An argv carried a second flash-attention flag."
    if observed != {FLASH_ATTENTION}:
        return False, f"{FA_FIXED_REASON} Observed -fa values across the matrix: {sorted(observed)}."
    return True, "ok"


def store_resolution() -> dict[str, Any]:
    """Resolve the production-named kernel through the store. No execution."""
    result: dict[str, Any] = {
        "store_path": str(KERNEL_STORE_GPU),
        "is_symlink": KERNEL_STORE_GPU.is_symlink(),
        "symlink_target": os.readlink(KERNEL_STORE_GPU) if KERNEL_STORE_GPU.is_symlink() else None,
        "forbidden_parents": [str(path) for path in FORBIDDEN_BINARY_PARENTS],
    }
    try:
        resolved_dir = KERNEL_STORE_GPU.resolve(strict=True)
    except OSError as exc:
        return {**result, "resolved": False, "reason": f"kernel store did not resolve: {exc}"}
    binary = resolved_dir / SERVER_BASENAME
    result.update({
        "resolved": True,
        "resolved_bin_dir": str(resolved_dir),
        "build_dir": str(resolved_dir.parent),
        "binary": str(binary),
        "binary_exists": binary.is_file(),
    })
    if resolved_dir in {path.resolve() if path.exists() else path for path in FORBIDDEN_BINARY_PARENTS}:
        result.update({"resolved": False, "reason": (
            "the kernel store resolved into the frozen source tree's own build dir, which still "
            "holds the v9 binaries; a v10 claim cannot be produced there")})
    return result


def resolved_binary() -> Path:
    resolution = store_resolution()
    if not resolution.get("resolved") or not resolution.get("binary_exists"):
        raise RuntimeError(f"kernel store refusal: {resolution.get('reason', 'binary missing')}")
    return Path(str(resolution["binary"]))


def git_state_is_clean(git: dict[str, Any]) -> bool:
    """`git diff --name-only` succeeds with rc=0 whether or not it has output.

    Tracked and staged changes must be empty. Untracked paths are allowed ONLY when
    every one of them is in the inherited allowlist: entries that are not llama-server
    build inputs and cannot change the binary under test.
    """
    for key in ("tracked_diff", "index_diff", "untracked"):
        capture = git.get(key) or {}
        if capture.get("returncode") != 0:
            return False
        if key != "untracked" and str(capture.get("stdout") or "").strip():
            return False
    untracked = [line.strip() for line in str((git.get("untracked") or {}).get("stdout") or "").splitlines() if line.strip()]
    return all(any(entry == allowed or entry.startswith(allowed)
                   for allowed in SOURCE_UNTRACKED_ALLOWLIST) for entry in untracked)


def source_identity() -> dict[str, Any]:
    git = common.git_state(FROZEN_SOURCE_TREE)
    commit = (git.get("commit") or {}).get("stdout", "").strip()
    branch = (git.get("branch") or {}).get("stdout", "").strip()
    return {
        "source_root": str(FROZEN_SOURCE_TREE),
        "expected_branch": EXPECTED_BRANCH,
        "expected_head": EXPECTED_COMMIT,
        "branch": branch,
        "head": commit,
        "branch_matches": branch == EXPECTED_BRANCH,
        "head_matches": commit == EXPECTED_COMMIT,
        "clean": git_state_is_clean(git),
        "git": git,
    }


def runtime_env(binary: Path) -> dict[str, str]:
    """The exact environment the server is launched under: its OWN tree first."""
    return {**common.evidence_env(), "LD_LIBRARY_PATH": f"{binary.parent}:/opt/rocm/lib"}


def identity_command(argv: list[str], env: dict[str, str], timeout: int = 60) -> dict[str, Any]:
    try:
        completed = subprocess.run(argv, text=True, capture_output=True, check=False, env=env, timeout=timeout)
        return {"argv": argv, "environment": env, "returncode": completed.returncode,
                "stdout": completed.stdout, "stderr": completed.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "environment": env, "returncode": None, "stdout": "", "stderr": "",
                "exec_error": repr(exc)}


def local_library_identities(ldd: dict[str, Any]) -> list[dict[str, Any]]:
    if ldd.get("returncode") != 0:
        return []
    identities: list[dict[str, Any]] = []
    for line in str(ldd.get("stdout") or "").splitlines():
        match = re.match(r"^\s*(lib(?:llama|ggml|mtmd)[^\s]*)\s+=>\s+(/[^\s]+)\s+\(", line)
        if match is None:
            continue
        ldd_path = Path(match.group(2))
        try:
            resolved = ldd_path.resolve(strict=True)
            identity = common.stable_file_identity(resolved)
            identity["path"] = str(ldd_path)
        except OSError as exc:
            identity = {"path": str(ldd_path), "resolved_path": str(ldd_path), "stable": False,
                        "identity_error": str(exc)}
        identity["soname"] = match.group(1)
        identities.append(identity)
    return sorted(identities, key=lambda item: (str(item["soname"]), str(item["resolved_path"])))


def linkage_receipt(binary: Path, env: dict[str, str]) -> dict[str, Any]:
    """P-GPU-1 field 3 is satisfied by a VERIFIER-PRODUCED receipt, never an env string."""
    verifier = common.stable_file_identity(LINKAGE_VERIFIER)
    expected_tree = str(binary.parent)
    capture = identity_command(["bash", str(LINKAGE_VERIFIER), str(binary), expected_tree], env)
    ldd = identity_command(["ldd", str(binary)], env)
    libraries = local_library_identities(ldd)
    sonames = {str(item.get("soname")) for item in libraries}
    text = f"{capture.get('stdout') or ''}\n{capture.get('stderr') or ''}"
    verdict = "pass" if capture.get("returncode") == 0 else (
        "vacuous" if capture.get("returncode") == 2 else "fail")
    return {
        "verifier_id": LINKAGE_VERIFIER_ID,
        "verifier_artifact": verifier,
        "verifier_sha256": verifier.get("sha256"),
        "expected_tree": expected_tree,
        "ld_library_path": env.get("LD_LIBRARY_PATH"),
        "capture": capture,
        "verdict": verdict,
        "pass_line_present": "PASS: all linked ggml libraries resolve inside" in text,
        "inspected_libraries": libraries,
        "inspected_count": len(libraries),
        "required_sonames_present": sorted(REQUIRED_LIBRARY_SONAMES),
        "required_sonames_satisfied": set(REQUIRED_LIBRARY_SONAMES).issubset(sonames),
        "ldd": ldd,
    }


def linkage_receipt_valid(receipt: dict[str, Any]) -> tuple[bool, str]:
    if receipt.get("verdict") != "pass" or not receipt.get("pass_line_present"):
        return False, f"linkage verifier verdict is {receipt.get('verdict')!r}, not a pass"
    if not receipt.get("inspected_count") or not receipt.get("required_sonames_satisfied"):
        return False, "linkage receipt is vacuous: it did not inspect the required ggml library set"
    libraries = receipt.get("inspected_libraries") or []
    if any(item.get("stable") is not True for item in libraries):
        return False, "an inspected library identity is unstable"
    for item in libraries:
        soname = str(item.get("soname"))
        if soname not in EXPECTED_LIBRARY_SHA256:
            return False, f"library {soname} is not in the pinned v10 set"
        if item.get("sha256") != EXPECTED_LIBRARY_SHA256[soname]:
            return False, f"library {soname} sha256 differs from the pinned v10 digest"
    return True, "ok"


def binary_identity(binary: Path, *, probe: bool) -> dict[str, Any]:
    """Artifact identity always; execution-derived evidence only when probing."""
    env = runtime_env(binary)
    artifact = common.stable_file_identity(binary)
    identity: dict[str, Any] = {
        "binary": str(binary),
        "binary_sha256": artifact.get("sha256"),
        "expected_binary_sha256": EXPECTED_SERVER_SHA256,
        "binary_sha256_matches": artifact.get("sha256") == EXPECTED_SERVER_SHA256,
        "artifact": artifact,
        "environment": env,
        "scrubbed_parent_env_keys": common.scrubbed_parent_env_keys(),
        "expected_version_line": EXPECTED_VERSION_LINE,
        "probed": probe,
    }
    if not probe:
        identity.update({
            "server_version": {"deferred": "dry run does not execute the binary; captured at --execute"},
            "linkage_receipt": {"deferred": "dry run does not execute the binary; captured at --execute"},
            "version_line_matches": None,
        })
        return identity
    version = identity_command([str(binary), "--version"], env)
    version_text = f"{version.get('stdout') or ''}\n{version.get('stderr') or ''}"
    identity.update({
        "server_version": version,
        "version_line_matches": version.get("returncode") == 0 and EXPECTED_VERSION_LINE in version_text,
        "linkage_receipt": linkage_receipt(binary, env),
    })
    return identity


def harness_identity() -> dict[str, Any]:
    return common.stable_file_identity(Path(__file__).resolve())


def model_identity(path: Path, *, hash_file: bool) -> dict[str, Any]:
    if hash_file:
        return common.stable_file_identity(path)
    try:
        info = path.stat()
        return {"path": str(path), "resolved_path": str(path.resolve()), "bytes": info.st_size,
                "sha256": TARGET_MODEL_SHA256,
                "sha256_deferred": TARGET_MODEL_SHA256 is None,
                "stable": None}
    except OSError as exc:
        return {"path": str(path), "stable": False, "identity_error": str(exc)}


def model_identity_valid(model: dict[str, Any]) -> tuple[bool, str]:
    if Path(str(model.get("path") or "")).resolve() != DEFAULT_TARGET_MODEL.resolve():
        return False, "target model path is not the pinned Qwen3.8-27B-Q8_0 GGUF"
    if model.get("stable") is not True:
        return False, "target model identity is unstable"
    if model.get("bytes") != TARGET_MODEL_BYTES:
        return False, "target model byte length differs from the pinned size"
    if TARGET_MODEL_SHA256 is not None and model.get("sha256") != TARGET_MODEL_SHA256:
        return False, "target model sha256 differs from the pinned digest"
    if not isinstance(model.get("sha256"), str) or len(str(model.get("sha256"))) != 64:
        return False, "target model sha256 was not captured"
    return True, "ok"


def fixed_identities_valid(source: dict[str, Any], binary: dict[str, Any], model: dict[str, Any]) -> tuple[bool, str]:
    resolution = store_resolution()
    if not resolution.get("resolved") or not resolution.get("binary_exists"):
        return False, f"kernel store refusal: {resolution.get('reason', 'binary missing')}"
    if not source.get("branch_matches") or not source.get("head_matches"):
        return False, f"production source tree is not on {EXPECTED_BRANCH}@{EXPECTED_COMMIT[:9]}"
    if not source.get("clean"):
        return False, "production source tree is not clean"
    if (binary.get("artifact") or {}).get("stable") is not True or not binary.get("binary_sha256_matches"):
        return False, "resolved server binary sha256 does not match the pinned v10 digest"
    if binary.get("version_line_matches") is not True:
        return False, f"server version line is not {EXPECTED_VERSION_LINE!r}"
    receipt_ok, receipt_reason = linkage_receipt_valid(binary.get("linkage_receipt") or {})
    if not receipt_ok:
        return False, receipt_reason
    return model_identity_valid(model)


# ---------------------------------------------------------------------------
# Plan.
# ---------------------------------------------------------------------------
def filler_text(target_tokens: int, overhead_tokens: int) -> str:
    """Deterministic ballast. No RNG, no corpus file, byte-identical every run."""
    needed_chars = max(0, int((target_tokens - overhead_tokens) * CHARS_PER_TOKEN))
    words: list[str] = []
    index = 0
    length = 0
    while length < needed_chars:
        word = FILLER_WORDS[(index * 7 + 3) % len(FILLER_WORDS)]
        words.append(word)
        length += len(word) + 1
        index += 1
    lines = [" ".join(words[start:start + FILLER_WORDS_PER_LINE])
             for start in range(0, len(words), FILLER_WORDS_PER_LINE)]
    return "\n".join(lines)


def depth_prompt(depth: Depth, prompt: str) -> str:
    if depth.target_prefill_tokens <= 0:
        return prompt
    overhead = int(len(FILLER_HEADER + FILLER_FOOTER + prompt) / CHARS_PER_TOKEN)
    return FILLER_HEADER + filler_text(depth.target_prefill_tokens, overhead) + FILLER_FOOTER + prompt


def ordered_runs(reps: int) -> list[dict[str, Any]]:
    """Cyclic counterbalance on BOTH axes so no arm or depth is always first/last."""
    runs: list[dict[str, Any]] = []
    for rep in range(1, reps + 1):
        depth_rotation = (rep - 1) % len(DEPTHS)
        for depth_index, depth in enumerate(DEPTHS[depth_rotation:] + DEPTHS[:depth_rotation]):
            cell_rotation = (rep - 1 + depth_index) % len(CELLS)
            for cell in CELLS[cell_rotation:] + CELLS[:cell_rotation]:
                runs.append({"cell": cell.name, "depth": depth.name, "rep": rep,
                             "port": PORT_BASE + len(runs)})
    return runs


def server_argv(binary: Path, cell: Cell, port: int, context: int, seed: int) -> list[str]:
    return [
        str(binary), "-m", str(DEFAULT_TARGET_MODEL), "--host", "127.0.0.1", "--port", str(port),
        "-c", str(context), "-ngl", "all", "-dev", "ROCm0",
        "-fa", FLASH_ATTENTION,
        "--cache-type-k", cell.cache_k, "--cache-type-v", cell.cache_v,
        "--seed", str(seed), "--temp", "0", "--top-k", "1", "--top-p", "1", "--jinja",
        "--reasoning", "off", "--reasoning-budget", "0", "-v",
    ]


def request_body(prompt: str, max_tokens: int, seed: int) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "seed": seed,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "stream": False,
        "cache_prompt": False,
    }


def build_plan(args: argparse.Namespace, binary: Path, model: dict[str, Any], *, probe: bool) -> dict[str, Any]:
    homogeneous_ok, homogeneous_reason = homogeneous_kv_guard()
    if not homogeneous_ok:
        raise RuntimeError(homogeneous_reason)
    runs = ordered_runs(args.reps)
    cells = {cell.name: cell for cell in CELLS}
    argvs = [server_argv(binary, cells[run["cell"]], int(run["port"]), args.context, args.seed) for run in runs]
    fa_ok, fa_reason = flash_attention_guard(argvs)
    if not fa_ok:
        raise RuntimeError(fa_reason)
    for run, argv in zip(runs, argvs, strict=True):
        run["argv"] = argv
    depth_prompts = {
        depth.name: [
            {"prompt_id": prompt_id,
             "chars": len(depth_prompt(depth, text)),
             "estimated_prefill_tokens": int(len(depth_prompt(depth, text)) / CHARS_PER_TOKEN),
             "sha256": common.sha256_text(depth_prompt(depth, text))}
            for prompt_id, text in common.PROMPT_SPECS
        ]
        for depth in DEPTHS
    }
    return {
        "schema": "epyc.kv_quant_27b_v10_sweep.plan.v1",
        "created_at": utc_now(),
        "execute": args.execute,
        "question": "does KV-cache quantization cost decode on the MI210 at the v10 production kernel",
        "protocol": "P-GPU-1",
        "instrument_class": "bench",
        "duty_cycle": "bursty",
        "category": "CANDIDATE",
        "metric_direction": "higher_better",
        "decision_grade_requires": "production-named kernel resolved through the kernel store",
        "kernel_store": store_resolution(),
        "candidate": {"source": source_identity(), "binary": binary_identity(binary, probe=probe),
                      "harness": harness_identity()},
        "target_model": model,
        "fixed_recipe": {
            "device": "ROCm0", "ngl": "all", "context": args.context,
            "flash_attention": FLASH_ATTENTION, "flash_attention_policy": FA_FIXED_REASON,
            "seed": args.seed, "temperature": 0, "top_k": 1, "top_p": 1,
            "reasoning": "off", "cache_prompt": False, "speculative_decoding": "forbidden",
        },
        "kv_policy": {"homogeneous_only": True, "mixed_refusal": MIXED_KV_REFUSAL},
        "cells": [{"name": cell.name, "cache_k": cell.cache_k, "cache_v": cell.cache_v,
                   "flash_attention": FLASH_ATTENTION} for cell in CELLS],
        "depths": [{"name": depth.name, "target_prefill_tokens": depth.target_prefill_tokens,
                    "min_prefill_tokens": depth.min_prefill_tokens,
                    "max_prefill_tokens": depth.max_prefill_tokens} for depth in DEPTHS],
        "fixed_prompt_pack": [{"id": key, "text": value} for key, value in common.PROMPT_SPECS],
        "depth_prompt_manifest": depth_prompts,
        "validators": {"primes_sum": 129, "nested_flatten": [1, 2, 3, 4, 5], "normalize_sum": 1.0},
        "reps_per_cell_per_depth": args.reps,
        "fresh_server_per_replicate": True,
        "total_launches": len(runs),
        "counterbalance": "cyclic on both axes: depth rotates each rep, arms rotate per rep and depth block",
        "warmup_discard_policy": common.PGPU1_WARMUP_POLICY,
        "cpu_interference_policy": common.CPU_INTERFERENCE_POLICY,
        "device_claim": {"mode": "acquired_flock", "lock_path": str(GPU_DEVICE_LOCK),
                         "device_id": GPU_DEVICE_ID,
                         "contract": "a claim is ACQUIRED, not observed; observing the lane looks free is TOCTOU"},
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# Execution.
# ---------------------------------------------------------------------------
def parse_log_residency(log_text: str, cell: Cell) -> dict[str, Any]:
    target = re.search(rf"loading model '{re.escape(str(DEFAULT_TARGET_MODEL))}'", log_text) is not None
    offload = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log_text)
    full_offload = bool(offload) and all(pair[0] == pair[1] and int(pair[1]) > 0 for pair in offload)
    models = [float(value) for value in re.findall(r"ROCm0 model buffer size =\s*([0-9.]+) MiB", log_text)]
    kv_buffers = [
        {"k_mib": float(k), "v_mib": float(v), "total_mib": float(k) + float(v)}
        for k, v in re.findall(
            rf"K \({re.escape(cell.cache_k)}\):\s*([0-9.]+) MiB, V \({re.escape(cell.cache_v)}\):\s*([0-9.]+) MiB",
            log_text,
        )
    ]
    # Arm-identity proof: the server itself must report the REQUESTED K/V types.
    any_kv_line = re.findall(r"K \((\w+)\):\s*[0-9.]+ MiB, V \((\w+)\):\s*[0-9.]+ MiB", log_text)
    wrong_type = [pair for pair in any_kv_line if pair != (cell.cache_k, cell.cache_v)]
    positive_models = [value for value in models if value > 0]
    positive_kv = [row for row in kv_buffers if row["k_mib"] > 0 and row["v_mib"] > 0]
    return {
        "passed": target and full_offload and bool(positive_models) and bool(positive_kv) and not wrong_type,
        "target_model_load_exact": target,
        "offloaded_layer_pairs": offload,
        "full_target_offload": full_offload,
        "rocm0_model_buffers_mib": models,
        "positive_rocm0_model": bool(positive_models),
        "kv_buffers_mib": kv_buffers,
        "kv_buffer_total_mib": max((row["total_mib"] for row in positive_kv), default=None),
        # K and V are reported SEPARATELY: a pooled figure hides the K/V asymmetry.
        "kv_k_mib": max((row["k_mib"] for row in positive_kv), default=None),
        "kv_v_mib": max((row["v_mib"] for row in positive_kv), default=None),
        "requested_kv": {"k": cell.cache_k, "v": cell.cache_v},
        "unexpected_kv_type_lines": wrong_type,
        "kv_types_and_positive_buffers": bool(positive_kv) and not wrong_type,
        "contract": "exact target, full GPU offload, positive ROCm0 model buffer, and server-reported K/V types equal to the requested arm",
    }


def record_from_response(response: dict[str, Any], prompt_id: str, prompt_index: int,
                         lifecycle: dict[str, Any], depth: Depth) -> dict[str, Any]:
    finish = common.finish_reason_from_response(response)
    content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("response lacks nonempty assistant content")
    sanity = common.response_sanity(content)
    semantic = common.semantic_validation(prompt_id, content)
    if not sanity["passed"] or not semantic["passed"]:
        raise RuntimeError(f"sanity/semantic gate failed: {sanity}; {semantic}")
    timings = common.timings_from_response(response, speculative=False)
    if timings["completion_tokens"] < MIN_COMPLETION_TOKENS:
        raise RuntimeError("completion token floor failed")
    if not depth.min_prefill_tokens <= timings["prompt_tokens"] <= depth.max_prefill_tokens:
        raise RuntimeError(
            f"prefill depth {depth.name} out of band: measured {timings['prompt_tokens']} tokens, "
            f"band [{depth.min_prefill_tokens}, {depth.max_prefill_tokens}]")
    return {"prompt_index": prompt_index, "prompt_id": prompt_id, "depth": depth.name,
            "finish_reason": finish, "assistant_content_sha256": common.sha256_text(content),
            "response_sanity": sanity, "semantic_validation": semantic,
            "request_lifecycle": lifecycle, **timings}


def summarize_cell(rows: list[dict[str, Any]], cell_name: str, depth_name: str, reps: int) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]

    def stats(metric: str) -> dict[str, Any]:
        values = [float(row[metric]) for row in ok if isinstance(row.get(metric), (int, float))]
        if not values or any(not math.isfinite(value) for value in values):
            return {"n": len(values), "median": None, "mad": None}
        median = float(statistics.median(values))
        return {"n": len(values), "median": median,
                "mad": float(statistics.median([abs(value - median) for value in values]))}

    return {
        "cell": cell_name, "depth": depth_name, "replicates": len(rows), "ok_replicates": len(ok),
        "all_ok": len(rows) == reps and len(ok) == reps,
        "prompt_ms": stats("prompt_ms"), "decode_ms": stats("decode_ms"),
        "prompt_tps": stats("prompt_tps"), "decode_tps": stats("decode_tps"),
        "prompt_tokens": stats("prompt_tokens"),
        "kv_buffer_total_mib": stats("kv_buffer_total_mib"),
        "kv_k_mib": stats("kv_k_mib"), "kv_v_mib": stats("kv_v_mib"),
    }


def kv_cost_comparison(summaries: dict[str, dict[str, Any]], depth_name: str,
                       post_execution_identity_valid: bool) -> dict[str, Any]:
    """f16 is the reference arm; quantized arms are reported as a ratio against it."""
    baseline_key = f"{CELLS[0].name}|{depth_name}"
    unavailable = {"status": "unavailable", "depth": depth_name, "baseline_cell": CELLS[0].name, "arms": {}}
    if not post_execution_identity_valid:
        return {**unavailable, "reason": "post-execution identity witness is invalid"}
    baseline = summaries.get(baseline_key)
    if not isinstance(baseline, dict) or not baseline.get("all_ok"):
        return {**unavailable, "reason": "the f16 reference arm did not complete every replicate"}
    arms: dict[str, Any] = {}
    for cell in CELLS[1:]:
        summary = summaries.get(f"{cell.name}|{depth_name}")
        if not isinstance(summary, dict) or not summary.get("all_ok"):
            return {**unavailable, "reason": f"{cell.name} did not complete every replicate"}
        row: dict[str, Any] = {}
        for metric in ("decode_tps", "prompt_tps"):
            reference = (baseline.get(metric) or {}).get("median")
            value = (summary.get(metric) or {}).get("median")
            if not all(isinstance(item, (int, float)) and math.isfinite(float(item)) and float(item) > 0
                       for item in (reference, value)):
                return {**unavailable, "reason": f"{cell.name} {metric} median is absent, non-finite, or non-positive"}
            ratio = float(value) / float(reference)
            row[metric] = {"reference_median": float(reference), "arm_median": float(value),
                           "arm_over_reference_ratio": ratio, "arm_vs_reference_percent": (ratio - 1.0) * 100.0}
        row["kv_buffer_total_mib_median"] = (summary.get("kv_buffer_total_mib") or {}).get("median")
        arms[cell.name] = row
    return {"status": "observed", "depth": depth_name, "baseline_cell": CELLS[0].name,
            "direction": "higher_better", "arms": arms}


def identity_witness_matches(pre: dict[str, Any], post: dict[str, Any]) -> tuple[bool, str]:
    """Identity drift invalidates all observations, even after a complete matrix."""
    for key in ("source", "binary", "target_model", "harness", "harness_snapshot", "kernel_store"):
        if pre.get(key) != post.get(key):
            return False, f"post-execution {key} differs from pre-execution witness"
    return True, "ok"


def matrix_valid(rows: list[dict[str, Any]], reps: int) -> tuple[bool, str]:
    if len(rows) != len(CELLS) * len(DEPTHS) * reps:
        return False, "incomplete matrix"
    for cell in CELLS:
        for depth in DEPTHS:
            selected = [row for row in rows if row.get("cell") == cell.name and row.get("depth") == depth.name]
            if sorted(row.get("rep") for row in selected) != list(range(1, reps + 1)):
                return False, f"{cell.name}/{depth.name} is incomplete"
            if any(row.get("status") != "ok" for row in selected):
                return False, f"{cell.name}/{depth.name} has a failed replicate"
            for row in selected:
                records = row.get("records") or []
                expected_ids = [prompt_id for prompt_id, _ in common.PROMPT_SPECS]
                if (len(records) != len(expected_ids)
                        or [record.get("prompt_index") for record in records] != list(range(1, len(expected_ids) + 1))
                        or [record.get("prompt_id") for record in records] != expected_ids
                        or any(record.get("finish_reason") != "stop"
                               or not (record.get("semantic_validation") or {}).get("passed")
                               or not (record.get("response_sanity") or {}).get("passed")
                               or not (record.get("request_lifecycle") or {}).get("fully_contained_valid")
                               or not isinstance((record.get("request_lifecycle") or {}).get("fully_contained_sample_count"), int)
                               or (record.get("request_lifecycle") or {})["fully_contained_sample_count"] < 1
                               for record in records)):
                    return False, f"{cell.name}/{depth.name} failed prompt/residency gates"
                if (not (row.get("residency") or {}).get("passed")
                        or not (row.get("cleanup") or {}).get("dead")
                        or not row.get("post_cleanup_clean")
                        or not row.get("post_cleanup_vram_settled")):
                    return False, f"{cell.name}/{depth.name} lacks residency or cleanup proof"
    return True, "ok"


def poll_vram_settlement(before: dict[str, Any], port: int, *, timeout_s: float = SETTLEMENT_TIMEOUT_S,
                         interval_s: float = SETTLEMENT_POLL_INTERVAL_S) -> tuple[bool, str, list[dict[str, Any]]]:
    """Require a clean process guard and settled valid ROCm evidence by deadline."""
    deadline = time.monotonic() + timeout_s
    samples: list[dict[str, Any]] = []
    while True:
        processes = common.process_snapshot()
        clean, reason = common.process_guard_clean(processes, port)
        rocm = common.collect_rocm_snapshot()
        valid = common.snapshot_is_valid(rocm)
        settled = valid and common.vram_settled(before, rocm)
        samples.append({"attempt": len(samples) + 1, "captured_at": utc_now(), "processes": processes,
                        "process_guard_clean": clean, "process_guard_reason": reason, "rocm": rocm,
                        "rocm_valid": valid, "vram_settled": settled})
        if not clean:
            return False, reason, samples
        if settled:
            return True, "ok", samples
        if time.monotonic() >= deadline:
            return False, "ROCm VRAM did not settle before deadline", samples
        time.sleep(min(interval_s, max(0.0, deadline - time.monotonic())))


def run_replicate(args: argparse.Namespace, binary: Path, cell: Cell, depth: Depth, rep: int, port: int,
                  output_dir: Path, expected_binding: dict[str, Any]) -> dict[str, Any]:
    """Run one fail-closed replicate while preserving evidence for later replicates."""
    rep_dir = output_dir / "runs" / f"{cell.name}_{depth.name}_rep{rep}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    argv = server_argv(binary, cell, port, args.context, args.seed)
    fa_ok, fa_reason = flash_attention_guard([argv])
    if not fa_ok:
        raise RuntimeError(fa_reason)
    homogeneous_ok, homogeneous_reason = homogeneous_kv_guard((cell,))
    if not homogeneous_ok:
        raise RuntimeError(homogeneous_reason)
    write_json(rep_dir / "server_argv.json", argv)
    env = runtime_env(binary)
    write_json(rep_dir / "environment.json",
               {"exact_server_environment": env, "scrubbed_parent_env_keys": common.scrubbed_parent_env_keys()})
    before = common.collect_rocm_snapshot()
    proc: subprocess.Popen[str] | None = None
    log: Any = None
    cleanup: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    residency: dict[str, Any] | None = None
    interrupted: KeyboardInterrupt | None = None
    result: dict[str, Any] = {"cell": cell.name, "depth": depth.name, "rep": rep, "status": "error",
                              "records": records, "residency": residency}
    try:
        clean, reason = common.process_guard_clean(common.process_snapshot(), port)
        if not clean or not common.snapshot_is_valid(before):
            raise RuntimeError(reason if not clean else "pre-launch ROCm evidence failed")
        log = (rep_dir / "server.stderr").open("w", encoding="utf-8")
        proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=log, text=True,
                                start_new_session=True, env=env, cwd=str(rep_dir))
        common.wait_for_health(port, args.startup_timeout, proc)
        for index, (prompt_id, prompt) in enumerate(common.PROMPT_SPECS, 1):
            body = request_body(depth_prompt(depth, prompt), args.max_tokens, args.seed)
            response, _elapsed, lifecycle = common.query_with_live_samples(
                port, body, args.request_timeout, proc.pid, binary, index,
                expected_binding=expected_binding, require_drafter=False)
            records.append(record_from_response(response, prompt_id, index, lifecycle, depth))
        residency = parse_log_residency((rep_dir / "server.stderr").read_text(encoding="utf-8", errors="replace"), cell)
        if not residency["passed"]:
            raise RuntimeError(f"residency proof failed: {residency}")
        prompt_ms = sum(row["prompt_ms"] for row in records)
        decode_ms = sum(row["decode_ms"] for row in records)
        result = {"cell": cell.name, "depth": depth.name, "rep": rep, "status": "ok", "records": records,
                  "residency": residency, "prompt_ms": prompt_ms, "decode_ms": decode_ms,
                  "prompt_tokens": sum(row["prompt_tokens"] for row in records),
                  "completion_tokens": sum(row["completion_tokens"] for row in records),
                  "prompt_tps": sum(row["prompt_tokens"] for row in records) / (prompt_ms / 1000),
                  "decode_tps": sum(row["completion_tokens"] for row in records) / (decode_ms / 1000),
                  "kv_buffer_total_mib": residency.get("kv_buffer_total_mib"),
                  "kv_k_mib": residency.get("kv_k_mib"), "kv_v_mib": residency.get("kv_v_mib")}
    except Exception as exc:  # preserve exact failure evidence
        result = {"cell": cell.name, "depth": depth.name, "rep": rep, "status": "error", "error": repr(exc),
                  "records": records, "residency": residency}
    except KeyboardInterrupt as exc:
        interrupted = exc
        result = {"cell": cell.name, "depth": depth.name, "rep": rep, "status": "interrupted",
                  "error": repr(exc), "records": records, "residency": residency}
    finally:
        cleanup_interrupt: KeyboardInterrupt | None = None
        previous_handler = signal.getsignal(signal.SIGINT)

        def defer_sigint(_signum: int, _frame: Any) -> None:
            nonlocal cleanup_interrupt
            cleanup_interrupt = KeyboardInterrupt()

        signal.signal(signal.SIGINT, defer_sigint)
        try:
            if proc is not None:
                cleanup = common.terminate(proc)
            if log is not None:
                log.close()
            result["cleanup"] = cleanup
            result["rocm_before"] = before
            if cleanup and cleanup.get("dead"):
                settled, settlement_reason, settlement_samples = poll_vram_settlement(before, port)
                result["settlement_samples"] = settlement_samples
                result["post_cleanup_clean"] = settlement_samples[-1]["process_guard_clean"] if settlement_samples else False
                result["post_cleanup_reason"] = settlement_reason
                result["post_cleanup_vram_settled"] = settled
                result["post_cleanup_processes"] = settlement_samples[-1]["processes"] if settlement_samples else None
                result["rocm_after"] = settlement_samples[-1]["rocm"] if settlement_samples else None
            else:
                result["settlement_samples"] = []
                result["post_cleanup_clean"] = False
                result["post_cleanup_reason"] = "server death proof failed"
                result["post_cleanup_vram_settled"] = False
                result["post_cleanup_processes"] = None
                result["rocm_after"] = None
            if not cleanup or not cleanup.get("dead") or not result["post_cleanup_clean"] or not result["post_cleanup_vram_settled"]:
                result["status"] = "cleanup_failed"
        except KeyboardInterrupt:
            cleanup_interrupt = KeyboardInterrupt()
            result["status"] = "cleanup_failed"
            result["cleanup_evidence_error"] = "KeyboardInterrupt while collecting cleanup evidence"
        except Exception as exc:  # cleanup evidence must not prevent durable failure output
            result["status"] = "cleanup_failed"
            result["cleanup_evidence_error"] = repr(exc)
        finally:
            write_json(rep_dir / "result.json", result)
            signal.signal(signal.SIGINT, previous_handler)
        if cleanup_interrupt is not None and interrupted is None:
            interrupted = cleanup_interrupt
    if interrupted is not None:
        raise interrupted
    return result


def execution_binding(binary: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    return {
        "server": {"path": binary.get("binary"), "sha256": binary.get("binary_sha256"),
                   "artifact": binary.get("artifact"),
                   "local_llama_ggml_libraries": (binary.get("linkage_receipt") or {}).get("inspected_libraries")},
        "models": {"target": model,
                   "drafter": {"path": "/__kv_sweep_spec_dec_forbidden__",
                               "resolved_path": "/__kv_sweep_spec_dec_forbidden__"}},
    }


def acquire_gpu_claim() -> Any:
    """Invariant 5: the flock IS the fact. `rocm-smi looks free` is TOCTOU, not exclusion."""
    from kernel_rnd.autokernel.loop import claim as gpu_claim  # noqa: PLC0415

    return gpu_claim


def execute(args: argparse.Namespace, binary: Path, output_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    gpu_claim = acquire_gpu_claim()
    source = source_identity()
    binary_info = binary_identity(binary, probe=True)
    model = model_identity(DEFAULT_TARGET_MODEL, hash_file=True)
    harness = harness_identity()
    kernel_store = store_resolution()
    identities_valid, identity_reason = fixed_identities_valid(source, binary_info, model)
    harness_snapshot = output_dir / "harness_source.py"
    shutil.copy2(Path(__file__).resolve(), harness_snapshot)
    captured_harness = common.stable_file_identity(harness_snapshot)
    harness_valid = (harness.get("stable") is True and captured_harness.get("stable") is True
                     and harness.get("sha256") == captured_harness.get("sha256"))
    expected_binding = execution_binding(binary_info, model)
    pre_execution_identity = {"source": source, "binary": binary_info, "target_model": model,
                              "harness": harness, "harness_snapshot": captured_harness,
                              "kernel_store": kernel_store}
    write_json(output_dir / "identities.json",
               {**pre_execution_identity, "binding": expected_binding, "harness_valid": harness_valid,
                "valid": identities_valid and harness_valid, "reason": identity_reason})
    hardware_state = common.collect_hardware_state()
    results: list[dict[str, Any]] = []
    claim_receipt: dict[str, Any] = {"status": "not_acquired"}
    if identities_valid and harness_valid:
        cells = {cell.name: cell for cell in CELLS}
        depths = {depth.name: depth for depth in DEPTHS}
        with gpu_claim.hold(GPU_DEVICE_LOCK, device_id=GPU_DEVICE_ID) as held:
            claim_receipt = {"status": "acquired", "receipt": dict(held),
                             "opened": held.observe()}
            for run in plan["runs"]:
                results.append(run_replicate(args, binary, cells[run["cell"]], depths[run["depth"]],
                                             int(run["rep"]), int(run["port"]), output_dir, expected_binding))
            claim_receipt["closing"] = held.observe()
        claim_receipt["survived_window"] = claim_receipt["closing"].get("status") == "held"
    else:
        results.append({"status": "error", "error": identity_reason})
    complete, reason = matrix_valid(results, args.reps)
    summaries = {
        f"{cell.name}|{depth.name}": summarize_cell(
            [row for row in results if row.get("cell") == cell.name and row.get("depth") == depth.name],
            cell.name, depth.name, args.reps)
        for cell in CELLS for depth in DEPTHS
    }
    post_identity = {"source": source_identity(), "binary": binary_identity(binary, probe=True),
                     "target_model": model_identity(DEFAULT_TARGET_MODEL, hash_file=True),
                     "harness": harness_identity(),
                     "harness_snapshot": common.stable_file_identity(harness_snapshot),
                     "kernel_store": store_resolution()}
    post_pins_valid, post_pins_reason = fixed_identities_valid(
        post_identity["source"], post_identity["binary"], post_identity["target_model"])
    post_harness_valid = (post_identity["harness"].get("stable") is True
                          and post_identity["harness_snapshot"].get("stable") is True
                          and post_identity["harness"].get("sha256") == post_identity["harness_snapshot"].get("sha256"))
    witness_matches, witness_reason = identity_witness_matches(pre_execution_identity, post_identity)
    claim_held = claim_receipt.get("survived_window") is True
    post_execution_identity_valid = post_pins_valid and post_harness_valid and witness_matches and claim_held
    if post_execution_identity_valid:
        post_reason = "ok"
    elif not post_pins_valid:
        post_reason = post_pins_reason
    elif not post_harness_valid:
        post_reason = "post-execution harness snapshot is invalid"
    elif not witness_matches:
        post_reason = witness_reason
    else:
        post_reason = "the MI210 device claim did not survive the measurement window"
    status = "ok" if identities_valid and harness_valid and complete and post_execution_identity_valid else "failed"
    comparisons = {depth.name: kv_cost_comparison(summaries, depth.name, post_execution_identity_valid)
                   for depth in DEPTHS}
    summary = {
        "schema": "epyc.kv_quant_27b_v10_sweep.summary.v1",
        "created_at": utc_now(),
        "status": status,
        "protocol": "P-GPU-1",
        "instrument_class": "bench",
        "duty_cycle": "bursty",
        "category": "CANDIDATE",
        "metric_direction": "higher_better",
        "production_named_kernel": True,
        "kernel_store": kernel_store,
        "required_branch": EXPECTED_BRANCH,
        "required_commit": EXPECTED_COMMIT,
        "n": args.reps,
        "rep_policy": "n >= 5 per arm per depth; n >= 10 is required before any <=2% claim; this runner makes no <=2% claim",
        "total_launches": len(plan["runs"]),
        "candidate": {"source": source, "binary": binary_info, "harness": harness,
                      "harness_snapshot": captured_harness},
        "target_model": model,
        "hardware_state": hardware_state,
        "device_claim": claim_receipt,
        "exact_plan": plan,
        "results": results,
        "matrix_cardinality_valid": complete,
        "matrix_cardinality_reason": reason,
        "cell_summaries": summaries,
        "kv_cost_by_depth": comparisons,
        "warmup_discard_policy": common.PGPU1_WARMUP_POLICY,
        "cpu_interference_policy": common.CPU_INTERFERENCE_POLICY,
        "post_execution_identity": post_identity,
        "post_execution_identity_valid": post_execution_identity_valid,
        "post_execution_identity_reason": post_reason,
        "cleanup_contract": "PID, process group, port, KFD ownership, and VRAM settlement are required per replicate",
    }
    return summary


# ---------------------------------------------------------------------------
# Belief-kernel WRITE side. Wired BEFORE the run, never retrofitted.
#
# This writes the NATIVE capture rows only. It PROJECTS nothing and GRADES
# nothing: an adapter under epyc-root `scripts/vidya/adapters/` projects a row
# into a `ClaimTuple` and `claim_tuple.grade()` decides. The `measurement`
# source class already has exactly one ladder (`claim_tuple.py`), and the
# registry refuses a second (`register_ladder`), so nothing here may return a
# lattice level. Contract: docs/design/vidya-pilot-spec.md 4.7.
#
# `validate_row` is deliberately importable: the future adapter pins this file's
# sha256 and imports it, so writer and reader share ONE row contract (the
# `dflash2_experimental_runtime` precedent).
# ---------------------------------------------------------------------------
CAPTURE_SCHEMA = "epyc.vidya.kv_quant_27b_v10_capture.v1"
CAPTURE_SOURCE_KIND = "kv-quant-27b-v10-measurement"
CAPTURE_PRODUCER = "epyc-inference-research/scripts/benchmark/kv_quant_27b_v10_sweep.py/v1"
CAPTURE_SIDECAR_NAME = "belief_measurements.jsonl"
CAPTURE_REQUIRED_KEYS = (
    "schema", "run_id", "producer", "emitted_at", "date", "measurement_id", "metric",
    "value", "unit", "metric_direction", "category", "claim", "protocol_id", "reps",
    "reps_basis", "scored_path", "scored_sha256", "extra", "row_sha256",
)
CAPTURE_METRICS = {
    "gpu_decode_tps": ("tokens/s", "higher_better"),
    "gpu_prefill_tps": ("tokens/s", "higher_better"),
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return common.sha256_text(canonical_json(value))


def row_digest(row: dict[str, Any]) -> str:
    return content_hash({key: value for key, value in row.items() if key != "row_sha256"})


def measurement_identity(*, run_id: str, cell: str, depth: str, metric: str, scored_sha256: str) -> str:
    """Identity is derived once, from the identity dimensions -- never invented on read."""
    digest = content_hash({"run_id": run_id, "cell": cell, "depth": depth, "metric": metric,
                           "scored_sha256": scored_sha256})
    return f"kvq_{digest[:24]}"


def validate_row(row: Any) -> list[str]:
    """Shared by the writer (refuse to emit) and the adapter (refuse to project)."""
    problems: list[str] = []
    if not isinstance(row, dict):
        return ["row is not an object"]
    missing = [key for key in CAPTURE_REQUIRED_KEYS if key not in row]
    if missing:
        problems.append(f"missing keys: {', '.join(missing)}")
        return problems
    if row["schema"] != CAPTURE_SCHEMA:
        problems.append(f"schema is {row['schema']!r}, not {CAPTURE_SCHEMA!r}")
    if row["producer"] != CAPTURE_PRODUCER:
        problems.append("producer id is not this harness")
    if row["metric"] not in CAPTURE_METRICS:
        problems.append(f"metric {row['metric']!r} is not one this producer emits")
    else:
        unit, direction = CAPTURE_METRICS[row["metric"]]
        if row["unit"] != unit:
            problems.append(f"unit for {row['metric']} must be {unit!r}")
        if row["metric_direction"] != direction:
            problems.append(f"metric_direction for {row['metric']} must be {direction!r}")
    if row["category"] not in {"OPTIMUM", "BASELINE", "CANDIDATE"}:
        problems.append("category must be one of OPTIMUM, BASELINE, CANDIDATE")
    if not isinstance(row["reps"], int) or isinstance(row["reps"], bool) or row["reps"] < 1:
        problems.append("reps must be a positive int")
    if not str(row["reps_basis"]):
        problems.append("reps_basis must say whether n is scored or attempted")
    if not isinstance(row["value"], (int, float)) or isinstance(row["value"], bool) or not math.isfinite(float(row["value"])):
        problems.append("value must be a finite real number")
    if not re.fullmatch(r"[0-9a-f]{64}", str(row["scored_sha256"] or "")):
        problems.append("scored_sha256 must be 64 hex characters")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(row["date"] or "")):
        problems.append("date must be YYYY-MM-DD")
    if not str(row["claim"]).strip():
        problems.append("claim text must carry the figure: to_frames never emits `value`")
    if not isinstance(row["extra"], dict) or not row["extra"]:
        problems.append("extra must carry the P-GPU-1 evidence block")
    if row["row_sha256"] != row_digest(row):
        problems.append("row_sha256 does not match the row body")
    return problems


def claim_text(cell: Cell, depth: Depth, metric: str, stats: dict[str, Any], reps: int,
               date: str, attest: str, relative: dict[str, Any] | None) -> str:
    label = "decode" if metric == "gpu_decode_tps" else "prefill"
    median = stats.get("median")
    mad = stats.get("mad")
    tail = ""
    if isinstance(relative, dict) and isinstance(relative.get("arm_vs_reference_percent"), (int, float)):
        tail = (f"; {relative['arm_vs_reference_percent']:+.2f}% vs the f16/f16 KV reference arm "
                f"at the same depth")
    return (f"Qwen3.8-27B-Q8_0 on MI210, KV {cell.cache_k}/{cell.cache_v}, -fa on, "
            f"prefill ~{depth.target_prefill_tokens} tok, c={CONTEXT}: {label} "
            f"{median:.3f} t/s (MAD {mad:.3f}, higher better){tail} "
            f"[P-GPU-1, n={reps}, {date}, attest {attest}] "
            f"(instrument_class=bench, duty_cycle=bursty, CANDIDATE/BASELINE arm set; "
            f"K and V reported separately in extra)")


def belief_capture_rows(summary: dict[str, Any], *, run_id: str, scored_path: str,
                        scored_sha256: str, emitted_at: str) -> list[dict[str, Any]]:
    """One native row per (arm, depth, metric). Absence is recorded, never filled."""
    rows: list[dict[str, Any]] = []
    if summary.get("status") != "ok":
        # Never back-fill and never manufacture a protocol route: a failed matrix emits
        # zero rows, permanently.
        return rows
    date = str(emitted_at)[:10]
    attest = f"{scored_path}#sha256={scored_sha256}"
    kernel_proven = bool(summary.get("production_named_kernel")) and bool(
        (summary.get("candidate") or {}).get("binary", {}).get("version_line_matches"))
    protocol_id = "P-GPU-1" if kernel_proven else ""
    for depth in DEPTHS:
        comparison = (summary.get("kv_cost_by_depth") or {}).get(depth.name) or {}
        for cell in CELLS:
            cell_summary = (summary.get("cell_summaries") or {}).get(f"{cell.name}|{depth.name}") or {}
            if not cell_summary.get("all_ok"):
                continue
            relative = ((comparison.get("arms") or {}).get(cell.name) or {})
            for metric, source_metric in (("gpu_decode_tps", "decode_tps"), ("gpu_prefill_tps", "prompt_tps")):
                stats = cell_summary.get(source_metric) or {}
                if not isinstance(stats.get("median"), (int, float)):
                    continue
                unit, direction = CAPTURE_METRICS[metric]
                measurement_id = measurement_identity(run_id=run_id, cell=cell.name, depth=depth.name,
                                                      metric=metric, scored_sha256=scored_sha256)
                row = {
                    "schema": CAPTURE_SCHEMA,
                    "run_id": run_id,
                    "producer": CAPTURE_PRODUCER,
                    "emitted_at": emitted_at,
                    "date": date,
                    "measurement_id": measurement_id,
                    "metric": metric,
                    "value": float(stats["median"]),
                    "unit": unit,
                    "metric_direction": direction,
                    # f16/f16 is the reference arm the quantized arms are judged against.
                    "category": "BASELINE" if cell is CELLS[0] else "CANDIDATE",
                    "claim": claim_text(cell, depth, metric, stats, int(summary.get("n") or 0), date,
                                        attest, relative.get(source_metric)),
                    "protocol_id": protocol_id,
                    "reps": int(stats.get("n") or 0),
                    "reps_basis": "scored:replicates (fresh server per replicate)",
                    "scored_path": scored_path,
                    "scored_sha256": scored_sha256,
                    "extra": {
                        "source_kind": CAPTURE_SOURCE_KIND,
                        "instrument_class": "bench",
                        "duty_cycle": "bursty",
                        "protocol_annex": "measurement/protocols/gpu-cross-device.md",
                        "locator": f"kvq:{run_id}:{cell.name}:{depth.name}:{metric}",
                        "mad": stats.get("mad"),
                        "arm": {"cache_k": cell.cache_k, "cache_v": cell.cache_v,
                                "flash_attention": FLASH_ATTENTION,
                                "flash_attention_policy": FA_FIXED_REASON,
                                "mixed_kv_policy": MIXED_KV_REFUSAL,
                                "context": CONTEXT,
                                "prefill_depth_target_tokens": depth.target_prefill_tokens,
                                "prefill_tokens_measured": cell_summary.get("prompt_tokens")},
                        # K and V separately: a pooled figure hides the asymmetry the sweep exists to find.
                        "kv_buffer_k_mib": cell_summary.get("kv_k_mib"),
                        "kv_buffer_v_mib": cell_summary.get("kv_v_mib"),
                        "kv_buffer_total_mib": cell_summary.get("kv_buffer_total_mib"),
                        "relative_to_reference_arm": relative.get(source_metric),
                        "reference_arm": CELLS[0].name,
                        "kernel": {"branch": EXPECTED_BRANCH, "commit": EXPECTED_COMMIT,
                                   "version_line": EXPECTED_VERSION_LINE,
                                   "production_named": kernel_proven,
                                   "store_path": str(KERNEL_STORE_GPU),
                                   "resolved_bin_dir": (summary.get("kernel_store") or {}).get("resolved_bin_dir"),
                                   "binary_sha256": (summary.get("candidate") or {}).get("binary", {}).get("binary_sha256")},
                        "linkage_receipt": ((summary.get("candidate") or {}).get("binary", {})
                                            .get("linkage_receipt") or {}),
                        "model": summary.get("target_model"),
                        "hardware_state": summary.get("hardware_state"),
                        "device_claim": {"mode": "acquired_flock",
                                         "lock_path": str(GPU_DEVICE_LOCK),
                                         "survived_window": (summary.get("device_claim") or {}).get("survived_window")},
                        "warmup_discard_policy": summary.get("warmup_discard_policy"),
                        "cpu_interference_policy": summary.get("cpu_interference_policy"),
                        "recipe": (summary.get("exact_plan") or {}).get("fixed_recipe"),
                        "caveat": ("bench-surface number on the fixed 3-prompt pack; it is NOT a serving "
                                   "rate and must never be compared across instrument classes"),
                        "protocol_omitted_reason": None if protocol_id else (
                            "the production-named-kernel provenance could not be proven for this run"),
                    },
                }
                row["row_sha256"] = row_digest(row)
                problems = validate_row(row)
                if problems:
                    raise RuntimeError(f"refusing to emit an invalid belief row: {'; '.join(problems)}")
                rows.append(row)
    return rows


def write_belief_measurements(output_dir: Path, summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    """Written AFTER summary.json so the attestation digest is over the real artifact."""
    scored = common.stable_file_identity(summary_path)
    if scored.get("stable") is not True:
        return {"written": False, "reason": "scored artifact identity is unstable", "rows": 0}
    try:
        relative = str(summary_path.resolve().relative_to(RESEARCH_ROOT))
    except ValueError:
        relative = str(summary_path.resolve())
    rows = belief_capture_rows(summary, run_id=output_dir.name, scored_path=relative,
                               scored_sha256=str(scored["sha256"]), emitted_at=utc_now())
    path = output_dir / CAPTURE_SIDECAR_NAME
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)
    return {"written": True, "path": str(path), "rows": len(rows), "schema": CAPTURE_SCHEMA,
            "source_kind": CAPTURE_SOURCE_KIND, "scored_sha256": scored["sha256"],
            "grades_nothing": True}


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def render_matrix(plan: dict[str, Any]) -> str:
    lines = [
        f"planned launches : {plan['total_launches']} "
        f"({len(CELLS)} arms x {plan['reps_per_cell_per_depth']} reps x {len(DEPTHS)} depths, fresh server per replicate)",
        f"flash attention  : -fa {FLASH_ATTENTION} FIXED in every arm (refuses otherwise)",
        f"kv policy        : homogeneous only (mixed k/v refused)",
        f"kernel store     : {plan['kernel_store'].get('store_path')} -> {plan['kernel_store'].get('resolved_bin_dir')}",
        f"binary           : {plan['candidate']['binary'].get('binary')}",
        f"binary sha256    : {plan['candidate']['binary'].get('binary_sha256')} "
        f"(pinned match={plan['candidate']['binary'].get('binary_sha256_matches')})",
        f"expected version : {EXPECTED_VERSION_LINE}",
        f"model            : {plan['target_model'].get('path')} ({plan['target_model'].get('bytes')} bytes)",
        "",
        "arm matrix:",
        f"  {'arm':<12} {'cache_k':<8} {'cache_v':<8} {'-fa':<4}",
    ]
    for cell in CELLS:
        lines.append(f"  {cell.name:<12} {cell.cache_k:<8} {cell.cache_v:<8} {FLASH_ATTENTION:<4}")
    lines.append("")
    lines.append("prefill depths:")
    for depth in DEPTHS:
        manifest = plan["depth_prompt_manifest"][depth.name]
        lines.append(f"  {depth.name:<6} target={depth.target_prefill_tokens:<6} "
                     f"band=[{depth.min_prefill_tokens}, {depth.max_prefill_tokens}] "
                     f"est_prompt_tokens={[row['estimated_prefill_tokens'] for row in manifest]}")
    lines.append("")
    lines.append("run order (counterbalanced) and argv:")
    for index, run in enumerate(plan["runs"], 1):
        lines.append(f"  [{index:02d}/{plan['total_launches']}] rep={run['rep']} depth={run['depth']} "
                     f"arm={run['cell']} port={run['port']}")
        lines.append(f"       {' '.join(run['argv'])}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(f"Launch count: {len(CELLS)} arms x {REPS} reps x {len(DEPTHS)} depths = "
                f"{TOTAL_LAUNCHES} fresh llama-server launches."),
    )
    parser.add_argument("--execute", action="store_true",
                        help="actually run the sweep; without it this is a dry run that executes nothing")
    parser.add_argument("--probe-binary", action="store_true",
                        help="in a dry run, also execute `llama-server --version` and the linkage verifier "
                             "(implied by --execute; off by default so a dry run touches no process)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--context", type=int, default=CONTEXT)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--startup-timeout", type=int, default=common.DEFAULT_STARTUP_TIMEOUT_S)
    parser.add_argument("--request-timeout", type=int, default=common.DEFAULT_REQUEST_TIMEOUT_S)
    args = parser.parse_args(argv)
    if (args.target_model, args.reps, args.context, args.max_tokens, args.seed) != (
            DEFAULT_TARGET_MODEL, REPS, CONTEXT, MAX_TOKENS, SEED):
        parser.error("target identity, five reps, context, max tokens, and replay seed are fixed")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    binary = resolved_binary()
    if args.execute:
        args.output_dir = args.output_dir / f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"output directory is not fresh: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe = bool(args.execute or args.probe_binary)
    model = model_identity(args.target_model, hash_file=bool(args.execute))
    plan = build_plan(args, binary, model, probe=probe)
    write_json(args.output_dir / "plan.json", plan)
    if not args.execute:
        print(render_matrix(plan))
        write_json(args.output_dir / "summary.json", {
            "schema": "epyc.kv_quant_27b_v10_sweep.summary.v1",
            "status": "prepared_no_inference",
            "protocol": "P-GPU-1",
            "instrument_class": "bench",
            "duty_cycle": "bursty",
            "category": "CANDIDATE",
            "exact_plan": plan,
        })
        return 0
    summary = execute(args, binary, args.output_dir, plan)
    summary_path = args.output_dir / "summary.json"
    write_json(summary_path, summary)
    capture = write_belief_measurements(args.output_dir, summary, summary_path)
    write_json(args.output_dir / "belief_capture_receipt.json", capture)
    print(f"belief-kernel capture: {capture['rows']} row(s) -> {capture.get('path')}")
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
