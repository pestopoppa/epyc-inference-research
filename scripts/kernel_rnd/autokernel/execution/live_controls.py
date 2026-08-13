#!/usr/bin/env python3
"""Run and calibrate the five AutoKernel controls on the live CPU cell.

Dry-run is the default.  ``--execute`` additionally requires the operator's
explicit host handoff flag, acquires one q0-q3 claim, measures fresh A/A and
neutral pools, solves the campaign calibration, then drives all five controls
through :mod:`execution.control_runner`'s candidate pipeline.

The frozen production tree is read only. Both arms use the reviewed hardened
measurement overlay, whose commit is a direct child of production v9. A
byte-for-byte copy of its ``llama-bench`` and ggml DSOs is made inside the
evidence bundle so the candidate arm can execute A/A without mislabelling the
serving tree or rebuilding the trusted instrument.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .. import campaign, schemas
from ..evaluator import api, controls, recipes, statistics
from ..resource import device_claim
from . import (control_runner, cpu_region_claim, microbench, physical_bounds,
               powercap_broker, sandbox, screening_baseline)

# ``RECIPE_ID`` remains the process-local active cell for compatibility with
# the recovery helpers below.  ``configure_recipe`` is called once by the CLI
# before it creates any evidence directory; it never changes an in-flight run.
PREFILL_RECIPE_ID = "t1b.llama_cpu.llama_bench_prefill.v1"
DECODE_RECIPE_ID = "t1b.llama_cpu.llama_bench_decode.v1"
RECIPE_ID = PREFILL_RECIPE_ID
CPU_LIST = recipes.CANONICAL_PREFIX[recipes.CANONICAL_PREFIX.index("-c") + 1]
PRODUCTION_ROOT = Path("/mnt/raid0/llm/llama.cpp")
PRODUCTION_COMMIT = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
INSTRUMENT_ROOT = Path(os.environ.get(
    "AUTOKERNEL_INSTRUMENT_ROOT",
    "/mnt/raid0/llm/autokernel/worktrees/ak-final-q6k-20260813"))
INSTRUMENT_BINARY = Path(os.environ.get(
    "AUTOKERNEL_INSTRUMENT_BINARY",
    str(INSTRUMENT_ROOT / "build-ak-t0-cpu-f744cc220/bin/llama-bench")))
INSTRUMENT_BRANCH = "experimental-v9-autokernel-t0-final-q6k-20260813"
INSTRUMENT_COMMIT = "f744cc220e722d1bda93783959471d44f8e118b0"
MODEL = Path(
    "/mnt/raid0/llm/models/lmstudio-community/"
    "Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q4_K_M.gguf")
CALIBRATION_BLOCKS = 200
NEUTRAL_BLOCKS = 60
# A calibration licenses a particular comparison frame, not merely a binary.
# The IQK intervention compares enabled code against the disabled baseline and
# campaign pairs use one llama-bench repetition.  Keep the A/A calibration
# in that baseline frame: calibrating IQK-on against itself would produce an
# apparently healthy noise pool whose anchor band cannot govern IQK-off runs.
CALIBRATION_REPS = campaign.IQK_MATCHED_PAIR_REPS
CALIBRATION_IQK = "0"
# A tg128 observation lasts only about 0.5 s after process startup on the
# 0.5B calibration graph.  r2 showed that one such observation is bimodal even
# when clock, package-power, order, and executable-identity checks all pass:
# random individual arms fell roughly 10--18% below the main mode.  Increasing
# llama-bench ``-r`` is not equivalent -- it reuses one process and aliases the
# within-process P-state trajectory that the prefill calibration already
# rejected.  Decode therefore defines one statistical block as the median of
# five FRESH, alternating candidate/anchor pairs.  The ranked held-out decode
# producer must consume this declared frame; this is aggregation, not a relaxed
# gate or an exclusion of ordinary host noise.
DECODE_FRESH_PAIRS_PER_BLOCK = 5
# This is not a sixth ranked control. It is a fresh A/A trace over the exact
# ranked T1 length. Ordinary host load is measured noise, never a reason to
# sleep or refuse; only a read-only witnessed model-inference process blocks the
# transition into a new leg. The trace licenses only campaigns of this length.
ANCHOR_MOTION_WINDOW_BLOCKS = 15
ANCHOR_MOTION_LABEL = "anchor_motion_calibration"
LEGACY_ANCHOR_MOTION_SETTLING = {
    "schema": "epyc.autokernel.anchor_motion_settling.v1",
    "kind": "non_ranked_post_work_quiet",
    "quiet_barrier_s": campaign.POST_T0_QUIET_BARRIER_S,
    "required_samples": campaign.POST_T0_QUIET_SAMPLES,
    "sample_interval_s": campaign.POST_T0_QUIET_SAMPLE_INTERVAL_S,
}
BETWEEN_LEG_POLICY = {
    "schema": "epyc.autokernel.between_leg_policy.v1",
    "ordinary_load": "recorded_as_measurement_noise_never_waited_or_refused",
    "blocking_condition": "witnessed_competing_model_inference_only",
    "inference_witness": "interim_inference_executable_scan",
}
ANCHOR_MOTION_SETTLING = {
    "schema": "epyc.autokernel.anchor_motion_transition.v2",
    "kind": "claim_held_inference_exclusion",
    "required_samples": 1,
    "ordinary_load_policy": BETWEEN_LEG_POLICY["ordinary_load"],
    "inference_witness": BETWEEN_LEG_POLICY["inference_witness"],
}
RESUME_AMENDMENT_SCHEMA = "epyc.autokernel.live_control_resume_amendment.v1"
RESUME_RECEIPT_SCHEMA = "epyc.autokernel.live_control_resume_receipt.v1"
CONTROL_EXTENSION_ROUNDS = 1
CONTROL_EXTENSION_BLOCKS = 5
CONTRIBUTION_FLOOR = 0.03
PROMPT_TOKENS = 512
WRONG_PROMPT_TOKENS = 2048
DECODE_TOKENS = 128
WRONG_DECODE_TOKENS = 256
NOMINAL_KHZ = 2_500_000
# Conservative work LOWER bounds and hardware UPPER bounds. These make the
# screen permissive: crossing it is evidence of wrong work/unit/timer, while
# staying below it is not a performance claim.
DENSE_ACTIVE_PARAMS_LOWER_BOUND = 400_000_000
MODEL_BYTES_FRACTION_LOWER_BOUND = 0.95
CPU_PEAK_COMPUTE_FLOPS_S_UPPER_BOUND = 110.8e12
CPU_PEAK_MEMORY_BYTES_S_UPPER_BOUND = 614.4e9
WORK_DERIVATION_REF = (
    "Qwen2.5-Coder-0.5B dense model: 400M active-parameter floor; "
    "2 FLOP/parameter/token; GGUF file bytes amortized across declared prompt at 95%"
)
HARDWARE_PEAK_REF = (
    "wiki/hardware-optimization.md:1706: EPYC 9655, 12 DDR5-6400 channels, "
    "614.4 GB/s theoretical; compute ceiling intentionally over-permissive at "
    "110.8 TFLOP/s"
)
CURRENT_SOURCE_CORRECTNESS_REASON = (
    f"frozen v9 source {PRODUCTION_COMMIT} plus exact binary copy; historical v8 "
    "real-model correctness evidence replayed"
)
CONTROL_PROMPT_BY_LABEL = {
    "aa_calibration": PROMPT_TOKENS,
    "neutral_calibration": PROMPT_TOKENS,
    ANCHOR_MOTION_LABEL: PROMPT_TOKENS,
    "positive": PROMPT_TOKENS,
    "historical_win_replay": PROMPT_TOKENS,
    "negative_committed_cell": PROMPT_TOKENS,
    "negative_wrong_cell": WRONG_PROMPT_TOKENS,
}


def configure_recipe(recipe_id: str) -> None:
    """Select one CPU tiny-graph cell before any live-control work.

    Calibration is recipe-local.  The legacy globals are deliberately updated
    as one closed frame because the module's recovery path replays raw vectors
    in a fresh CLI process; no process may switch recipes after writing a
    declaration.  Prefill remains the default byte-for-byte frame.
    """
    global RECIPE_ID, PROMPT_TOKENS, WRONG_PROMPT_TOKENS, CONTROL_PROMPT_BY_LABEL
    if recipe_id == PREFILL_RECIPE_ID:
        prompt, wrong = 512, 2048
    elif recipe_id == DECODE_RECIPE_ID:
        prompt, wrong = DECODE_TOKENS, WRONG_DECODE_TOKENS
    else:
        raise ValueError(f"unsupported live-control CPU recipe: {recipe_id}")
    RECIPE_ID, PROMPT_TOKENS, WRONG_PROMPT_TOKENS = recipe_id, prompt, wrong
    CONTROL_PROMPT_BY_LABEL = {
        "aa_calibration": prompt, "neutral_calibration": prompt,
        ANCHOR_MOTION_LABEL: prompt, "positive": prompt,
        "historical_win_replay": prompt, "negative_committed_cell": prompt,
        "negative_wrong_cell": wrong,
    }


def _token_fields() -> dict[str, int | None]:
    """Closed declaration fields for the active recipe's work dimension."""
    if RECIPE_ID == PREFILL_RECIPE_ID:
        return {"prompt_tokens": PROMPT_TOKENS,
                "wrong_prompt_tokens": WRONG_PROMPT_TOKENS,
                "decode_tokens": None, "wrong_decode_tokens": None}
    return {"prompt_tokens": None, "wrong_prompt_tokens": None,
            "decode_tokens": PROMPT_TOKENS,
            "wrong_decode_tokens": WRONG_PROMPT_TOKENS}


def _calibration_frame() -> dict[str, object]:
    token_key = "prompt_tokens" if RECIPE_ID == PREFILL_RECIPE_ID else "decode_tokens"
    frame = {"recipe_id": RECIPE_ID, token_key: PROMPT_TOKENS,
             "reps": CALIBRATION_REPS,
             "candidate_ggml_iqk": CALIBRATION_IQK,
             "anchor_ggml_iqk": CALIBRATION_IQK}
    if RECIPE_ID == DECODE_RECIPE_ID:
        frame["fresh_pairs_per_block"] = DECODE_FRESH_PAIRS_PER_BLOCK
        frame["aggregation"] = "median_per_arm"
    return frame


def _fresh_pairs_per_block() -> int:
    """Return the predeclared independent-invocation aggregation frame."""
    return (DECODE_FRESH_PAIRS_PER_BLOCK
            if RECIPE_ID == DECODE_RECIPE_ID else 1)
# The controls have intentionally different purposes, but every live control
# uses the same repetition count as an executable campaign.  Only the positive
# and historical controls exercise the IQK intervention; calibration itself
# must remain A/A in the baseline anchor frame.
CONTROL_ARM_IQK = {
    "aa_calibration": (CALIBRATION_IQK, CALIBRATION_IQK),
    "neutral_calibration": (CALIBRATION_IQK, CALIBRATION_IQK),
    ANCHOR_MOTION_LABEL: (CALIBRATION_IQK, CALIBRATION_IQK),
    "positive": ("1", CALIBRATION_IQK),
    "historical_win_replay": ("1", CALIBRATION_IQK),
    "negative_committed_cell": ("1", "1"),
    "negative_wrong_cell": ("1", "1"),
}
INSTRUMENT_BUILD_TARGETS = ("llama-completion", "llama-bench", "test-backend-ops")
REQUIRED_HARDENING_RECEIPTS = (
    b"autokernel_hybrid_ab_complete",
    b"autokernel_thread_set_stable",
    b"autokernel_escape_checks_complete",
    b"autokernel_unsynchronized_samples_ns",
    b"autokernel_thread_set_hashes",
    b"autokernel_device_sync_mode",
)
BELIEF_RECEIPT_SCHEMA = "epyc.autokernel.live_control_beliefs.v1"
BELIEF_PRODUCER_ID = "autokernel.execution.live_controls/v2"


def _ensure_instrument_build() -> None:
    """Materialize the complete read-only T0/T1 instrument before calibration."""
    bindir = INSTRUMENT_BINARY.parent
    required = tuple(bindir / name for name in INSTRUMENT_BUILD_TARGETS)
    if all(path.is_file() and os.access(path, os.X_OK) for path in required):
        return
    build_dir = bindir.parent
    if not build_dir.is_dir():
        raise RuntimeError(f"instrument build directory is missing: {build_dir}")
    subprocess.run(
        ["/usr/bin/cmake", "--build", str(build_dir), "--target",
         *INSTRUMENT_BUILD_TARGETS, "-j", "64"],
        cwd=str(INSTRUMENT_ROOT), check=True)
    missing = [str(path) for path in required
               if not path.is_file() or not os.access(path, os.X_OK)]
    if missing:
        raise RuntimeError("instrument build omitted required tools: " + ", ".join(missing))


@dataclass(frozen=True)
class LiveCampaignIdentity:
    """Fresh operator-chosen id plus every deterministic name derived from it."""

    campaign_id: str
    evidence_ref: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"ak-[a-z0-9][a-z0-9._-]{2,95}", self.campaign_id):
            raise ValueError(
                "campaign_id must match ak-[a-z0-9][a-z0-9._-]{2,95}")
        if not isinstance(self.evidence_ref, str) or not self.evidence_ref.startswith("/"):
            raise ValueError("evidence_ref must be an absolute durable path")

    @property
    def campaign_seed(self) -> str:
        return f"{self.campaign_id}/live-controls/seed-v1"

    @property
    def window_id(self) -> str:
        digest = hashlib.sha256(self.campaign_seed.encode()).hexdigest()[:24]
        return f"akw-{digest}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_file_sha256(value: object) -> str:
    """Digest the exact bytes :func:`_write_json` will persist."""
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
        dirfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _tool_version(executable: Path) -> str:
    """Return version text from the executable actually recorded by CMake."""
    if not executable.is_file():
        raise RuntimeError(f"toolchain executable is not a file: {executable}")
    return subprocess.run((str(executable), "--version"), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=True).stdout.strip()


def _build_toolchain_manifest(output_root: Path) -> str:
    """Persist a hashable receipt of the instrument's measured build inputs.

    Values come from the instrument build's CMake cache and the files it names;
    there are no fallback/toolchain labels.  The manifest is write-once so a
    later phase cannot silently change the provider identity.
    """
    build_root = INSTRUMENT_BINARY.parent.parent
    cache = build_root / "CMakeCache.txt"
    if not cache.is_file():
        raise RuntimeError(f"instrument build metadata is missing: {cache}")
    entries: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.startswith(("CMAKE_C_COMPILER:", "CMAKE_CXX_COMPILER:",
                            "CMAKE_MAKE_PROGRAM:", "CMAKE_BUILD_TYPE:",
                            "GGML_NATIVE:", "GGML_CPU:", "GGML_AVX")):
            key, _, value = line.partition("=")
            entries[key.split(":", 1)[0]] = value
    required = ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_MAKE_PROGRAM",
                "CMAKE_BUILD_TYPE")
    missing = [key for key in required if not entries.get(key)]
    if missing:
        raise RuntimeError("instrument CMake cache lacks: " + ", ".join(missing))
    tools = {}
    for key in ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_MAKE_PROGRAM"):
        path = Path(entries[key]).resolve(strict=True)
        tools[key] = {"path": str(path), "sha256": _sha256_file(path),
                      "version": _tool_version(path)}
    manifest = {
        "schema": "epyc.autokernel.measurement_toolchain_manifest.v1",
        "build_root": str(build_root.resolve()),
        "cmake_cache": {"path": str(cache.resolve()), "sha256": _sha256_file(cache)},
        "cmake_cache_values": entries,
        "tools": tools,
    }
    path = output_root / "measurement-toolchain-manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError("measurement toolchain manifest is immutable")
    else:
        _write_json(path, manifest)
    return _json_file_sha256(manifest)


def _copy_anchor_bundle(root: Path) -> recipes.ToolBinding:
    """Copy the measured tool and every DSO it resolves inside the frozen build."""
    bundle = root / "anchor_binary_copy"
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INSTRUMENT_BINARY, bundle / "llama-bench")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{INSTRUMENT_BINARY.parent}:/usr/lib/llvm-20/lib"
    linked = subprocess.run(
        ("ldd", str(INSTRUMENT_BINARY)), env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
    copied = []
    for line in linked.splitlines():
        fields = line.strip().split()
        if len(fields) < 3 or fields[1] != "=>":
            continue
        name, resolved = fields[0], Path(fields[2])
        try:
            inside = resolved.resolve(strict=True).is_relative_to(
                INSTRUMENT_BINARY.parent.resolve())
        except (OSError, RuntimeError):
            inside = False
        if inside:
            shutil.copy2(resolved.resolve(), bundle / name)
            copied.append(name)
    if not copied:
        raise RuntimeError("ldd found no production-build DSOs to copy with llama-bench")
    if _sha256_file(bundle / "llama-bench") != _sha256_file(INSTRUMENT_BINARY):
        raise RuntimeError("the copied A/A binary is not byte-identical to the instrument")
    return recipes.ToolBinding(
        binary=str(bundle / "llama-bench"), source_root=str(root),
        library_path=str(bundle))


def _linkage(binary: Path, library_path: Path) -> tuple[str, str]:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{library_path}:/usr/lib/llvm-20/lib"
    proc = subprocess.run(("ldd", str(binary)), env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=True)
    text = proc.stdout
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), text


def _normalized_linkage(text: str) -> tuple[str, ...]:
    """Return the stable DSO-resolution identity from ASLR-bearing ``ldd``.

    The persisted raw output remains hash-bound. A later validation cannot
    reproduce its virtual addresses, so equality across invocations must strip
    only the terminal loader address while retaining every soname and path.
    """
    return tuple(
        re.sub(r"\s+\(0x[0-9a-fA-F]+\)$", "", line.rstrip())
        for line in text.splitlines()
    )


def _check_payload(check: schemas.Check) -> dict:
    return {"outcome": check.outcome, "reasons": list(check.reasons)}


def _instrument_receipt_capability(binary: Path) -> schemas.Check:
    """Prove the selected binary bundle can emit every binding T1 receipt."""
    payloads = [binary]
    payloads.extend(sorted(binary.parent.glob("libllama-bench-impl.so*")))
    observed = set()
    for payload in payloads:
        try:
            content = payload.read_bytes()
        except OSError:
            continue
        observed.update(key for key in REQUIRED_HARDENING_RECEIPTS if key in content)
    missing = [key.decode("ascii") for key in REQUIRED_HARDENING_RECEIPTS
               if key not in observed]
    if missing:
        return schemas.Check(
            schemas.FAIL,
            ("measurement instrument cannot emit required hardening receipts: "
             + ", ".join(missing),))
    return schemas.Check(
        schemas.PASS,
        ("measurement instrument contains every required hybrid-sync, thread-set, "
         "escape-check, and device-sync receipt key",))


def _write_declaration(output_root: Path, *, identity: LiveCampaignIdentity,
                       instrument_sha: str, copy_sha: str,
                       instrument_linkage: str, copy_linkage: str,
                       toolchain_manifest_sha256: str,
                       sealed_binding: recipes.ToolBinding) -> str:
    """Commit the fresh campaign inputs before the first measurement."""
    source_manifest = {
        "schema": "epyc.autokernel.runtime_source_label.v1",
        "production_source_commit": PRODUCTION_COMMIT,
        "measurement_instrument_commit": INSTRUMENT_COMMIT,
        "measurement_binary_sha256": instrument_sha,
        "copied_binary_sha256": copy_sha,
        "measurement_linkage_sha256": instrument_linkage,
        "measurement_toolchain_manifest_sha256": toolchain_manifest_sha256,
        "copied_linkage_sha256": copy_linkage,
        "binary_copy_exact": instrument_sha == copy_sha,
        # Path equality is load-bearing for A/A: equal bytes at two different
        # binary/LD_LIBRARY_PATH locations do not measure the same process
        # frame.  Raw receipts below must repeat these exact paths.
        "aa_sealed_binding": {
            "binary_path": sealed_binding.binary,
            "library_path": sealed_binding.library_path,
        },
    }
    source_sha = schemas.content_hash(source_manifest)
    _write_json(output_root / "runtime-source-label.json",
                {**source_manifest, "source_sha256": source_sha})
    declaration = {
        "schema": "epyc.autokernel.live_control_campaign_declaration.v1",
        "declared_at": _utc_now(),
        "campaign_id": identity.campaign_id,
        "campaign_seed_sha256": hashlib.sha256(
            identity.campaign_seed.encode("utf-8")).hexdigest(),
        "window_id": identity.window_id,
        "recipe_id": RECIPE_ID,
        "cpu_list": CPU_LIST,
        "model": str(MODEL),
        "model_sha256": _sha256_file(MODEL) if MODEL.is_file() else None,
        **_token_fields(),
        "calibration_blocks": CALIBRATION_BLOCKS,
        "neutral_blocks": NEUTRAL_BLOCKS,
        "calibration_frame": _calibration_frame(),
        "anchor_motion_window_blocks": ANCHOR_MOTION_WINDOW_BLOCKS,
        "anchor_motion_settling": ANCHOR_MOTION_SETTLING,
        "between_leg_policy": BETWEEN_LEG_POLICY,
        "contribution_floor": CONTRIBUTION_FLOOR,
        "max_candidates": 10,
        "max_blocks_per_candidate": 20,
        "physical_envelopes": {
            label: envelope.to_dict()
            for label, envelope in sorted(_declared_physical_envelopes().items())
        },
        "source_sha256": source_sha,
        "belief_capture_schema": BELIEF_RECEIPT_SCHEMA,
    }
    _write_json(output_root / "campaign_declaration.json", declaration)
    return source_sha


def _build_belief_receipt(
        output_root: Path, *, identity: LiveCampaignIdentity,
        result: control_runner.SweepResult,
        claim_receipt: cpu_region_claim.RegionClaimReceipt,
        measured_at: str) -> dict | None:
    """Capture future control-panel claims without retrofitting old evidence.

    The declaration marker is written before measurement.  Its absence is the
    unambiguous pre-hook boundary: recovery of an older raw bundle may compose
    the old control sweep, but it must never invent a Vidya tuple afterward.
    """
    declaration = _load_json(output_root / "campaign_declaration.json")
    if declaration.get("belief_capture_schema") != BELIEF_RECEIPT_SCHEMA:
        return None
    runtime = _load_json(output_root / "runtime-source-label.json")
    sweep = result.to_dict()
    panel = sweep.get("panel_result")
    outcomes = panel.get("outcomes") if isinstance(panel, Mapping) else None
    observations = panel.get("observations") if isinstance(panel, Mapping) else None
    if not isinstance(outcomes, list) or not isinstance(observations, list):
        raise RuntimeError("live-control sweep lacks terminal outcomes and observations")
    by_control = {
        row.get("control_id"): row for row in observations
        if isinstance(row, Mapping)
    }
    expected = {outcome.get("control_id") for outcome in outcomes
                if isinstance(outcome, Mapping)}
    required = {
        controls.CONTROL_POSITIVE, controls.CONTROL_NEUTRAL,
        controls.CONTROL_DEGRADED_NEGATIVE, controls.CONTROL_AA,
        controls.CONTROL_HISTORICAL_WIN_REPLAY,
    }
    if expected != required or set(by_control) != required:
        raise RuntimeError("live-control belief receipt requires exactly the five controls")
    sweep_sha256 = _json_file_sha256(sweep)
    producer_sha256 = _sha256_file(Path(__file__).resolve())
    producer = {
        "producer_id": BELIEF_PRODUCER_ID,
        "path": "scripts/kernel_rnd/autokernel/execution/live_controls.py",
        "sha256": producer_sha256,
    }
    runtime_body = {key: value for key, value in runtime.items()
                    if key != "source_sha256"}
    if runtime.get("source_sha256") != schemas.content_hash(runtime_body):
        raise RuntimeError("runtime source identity changed before belief finalization")
    source_identity = {
        "production_source_commit": runtime["production_source_commit"],
        "measurement_instrument_commit": runtime["measurement_instrument_commit"],
        "runtime_source_sha256": runtime["source_sha256"],
    }
    source_identity_sha256 = schemas.content_hash(source_identity)
    binary_identity = {
        "path": str(output_root / "anchor_binary_copy" / "llama-bench"),
        "sha256": runtime["copied_binary_sha256"],
        "linkage_sha256": runtime["copied_linkage_sha256"],
        "copy_exact": runtime["binary_copy_exact"],
    }
    model_identity = {
        "path": declaration["model"],
        "sha256": declaration["model_sha256"],
    }
    claim_identity = claim_receipt.to_dict()
    claim_identity_sha256 = schemas.content_hash(claim_identity)
    raw_paths = {
        controls.CONTROL_POSITIVE: ("positive",),
        controls.CONTROL_NEUTRAL: ("neutral_calibration",),
        controls.CONTROL_DEGRADED_NEGATIVE: (
            "negative_committed_cell", "negative_wrong_cell"),
        controls.CONTROL_AA: ("aa_calibration",),
        controls.CONTROL_HISTORICAL_WIN_REPLAY: ("historical_win_replay",),
    }
    raw_sha256 = {
        control_id: {
            label: _sha256_file(output_root / "raw" / f"{label}.json")
            for label in raw_paths[control_id]
        }
        for control_id in required
    }
    measurements = []
    for outcome in sorted(outcomes, key=lambda row: row["ordinal"]):
        control_id = outcome["control_id"]
        observation = by_control[control_id]
        reps = observation.get("abs_effect_count")
        if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
            raise RuntimeError(f"control {control_id} lacks its scored-block count")
        native_verdict = outcome.get("outcome")
        if native_verdict not in {schemas.PASS, schemas.FAIL}:
            raise RuntimeError(f"control {control_id} lacks a native PASS/FAIL verdict")
        evidence_basis = {
            "control_id": control_id,
            "outcome": outcome,
            "observation": observation,
            "raw_vector_sha256": raw_sha256[control_id],
            "control_sweep_sha256": sweep_sha256,
            "source_identity_sha256": source_identity_sha256,
            "binary_sha256": binary_identity["sha256"],
            "model_sha256": model_identity["sha256"],
            "claim_identity_sha256": claim_identity_sha256,
            "producer_sha256": producer_sha256,
        }
        row = {
            "measurement_id": f"live_control_{control_id}_requirement_satisfied",
            "metric": "autokernel_control_requirement_satisfaction",
            "value": 1.0 if native_verdict == schemas.PASS else 0.0,
            "unit": "fraction",
            "metric_direction": "higher_better",
            "category": "BASELINE",
            "protocol_id": api.PROTOCOL_VERSIONED_ID,
            "reps": reps,
            "reps_basis": "scored:paired live-control blocks",
            "claim": (
                f"AutoKernel live control {control_id} requirement outcome is "
                f"{native_verdict} after {reps} scored paired blocks"),
            "native_verdict": native_verdict,
            "extra": {
                "control_id": control_id,
                "native_disposition": outcome.get("disposition"),
                "native_effect_resolution": observation.get("effect_resolution"),
                "source_identity": source_identity,
                "source_identity_sha256": source_identity_sha256,
                "binary_identity": binary_identity,
                "model_identity": model_identity,
                "resource_claim_identity": claim_identity,
                "claim_identity_sha256": claim_identity_sha256,
                "producer_id": BELIEF_PRODUCER_ID,
                "producer_sha256": producer_sha256,
                "evidence_basis": evidence_basis,
                "evidence_sha256": schemas.content_hash(evidence_basis),
            },
        }
        row["measurement_sha256"] = schemas.content_hash(row)
        measurements.append(row)
    payload = {
        "schema": BELIEF_RECEIPT_SCHEMA,
        "status": "complete",
        "campaign_id": identity.campaign_id,
        "created_at": measured_at,
        "ended_at": measured_at,
        "protocol_id": api.PROTOCOL_VERSIONED_ID,
        "producer": producer,
        "source_identity": source_identity,
        "source_identity_sha256": source_identity_sha256,
        "binary_identity": binary_identity,
        "model_identity": model_identity,
        "resource_claim_identity": claim_identity,
        "claim_identity_sha256": claim_identity_sha256,
        "control_sweep_sha256": sweep_sha256,
        "raw_vector_sha256": raw_sha256,
        "control_panel": panel,
        "native_verdict": {
            "marker": panel.get("marker"),
            "may_rank": panel.get("may_rank"),
            "halts_campaign": panel.get("halts_campaign"),
            "voids_window": panel.get("voids_window"),
        },
        "belief_measurements": measurements,
    }
    payload["receipt_sha256"] = schemas.content_hash(payload)
    return payload


def _write_preflight(output_root: Path, *, instrument_sha: str, copy_sha: str,
                     host_state: Callable[..., microbench.HostState]) -> None:
    topology = cpu_region_claim.verify_host_topology()
    free_bytes = shutil.disk_usage(output_root).free
    source_head = subprocess.run(
        ("git", "-C", str(PRODUCTION_ROOT), "rev-parse", "HEAD"),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True).stdout.strip()
    instrument_head = subprocess.run(
        ("git", "-C", str(INSTRUMENT_ROOT), "rev-parse", "HEAD"),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True).stdout.strip()
    instrument_branch = subprocess.run(
        ("git", "-C", str(INSTRUMENT_ROOT), "branch", "--show-current"),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True).stdout.strip()
    instrument_status = subprocess.run(
        ("git", "-C", str(INSTRUMENT_ROOT), "status", "--porcelain"),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True).stdout
    instrument_parents = subprocess.run(
        ("git", "-C", str(INSTRUMENT_ROOT), "rev-list", "--parents", "-n", "1",
         INSTRUMENT_COMMIT),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True).stdout.split()[1:]
    state = host_state(cpu_list=CPU_LIST)
    host_policy = microbench.HostStatePolicy(
        nominal_khz=NOMINAL_KHZ, require_package_power=True)
    checks = {
        "topology": _check_payload(topology),
        "production_commit": _check_payload(schemas.Check(
            schemas.PASS if source_head == PRODUCTION_COMMIT else schemas.FAIL,
            (f"production HEAD is {source_head}; required {PRODUCTION_COMMIT}",))),
        "measurement_instrument": _check_payload(schemas.Check(
            schemas.PASS if (
                instrument_head == INSTRUMENT_COMMIT
                and instrument_branch == INSTRUMENT_BRANCH
                and not instrument_status
                and instrument_parents == [PRODUCTION_COMMIT]
            ) else schemas.FAIL,
            (f"instrument head={instrument_head}, branch={instrument_branch}, "
             f"dirty={bool(instrument_status)}, parents={instrument_parents}; required "
             f"clean {INSTRUMENT_COMMIT} directly on {PRODUCTION_COMMIT}",))),
        "binary_copy": _check_payload(schemas.Check(
            schemas.PASS if instrument_sha == copy_sha else schemas.FAIL,
            (f"instrument and evidence-copy SHA-256 are {instrument_sha}",))),
        "instrument_receipt_capability": _check_payload(
            _instrument_receipt_capability(INSTRUMENT_BINARY)),
        "model_present": _check_payload(schemas.Check(
            schemas.PASS if MODEL.is_file() else schemas.FAIL,
            (f"model path is {MODEL}",))),
        "storage": _check_payload(schemas.Check(
            schemas.PASS if free_bytes >= 200 * 1024 ** 3 else schemas.FAIL,
            (f"{free_bytes} bytes free at campaign open",))),
        "package_power_available": _check_payload(
            host_policy.check_package_power_available(state)),
    }
    _write_json(output_root / "preflight.json", {
        "schema": "epyc.autokernel.live_control_preflight.v1",
        "measured_at": _utc_now(), "checks": checks,
    })
    failed = [name for name, check in checks.items()
              if check["outcome"] != schemas.PASS]
    if failed:
        raise RuntimeError(f"live-control preflight refused: {failed}")


def _write_host_receipt(output_root: Path, materials: Sequence[LiveMaterial],
                        claim_receipt: cpu_region_claim.RegionClaimReceipt,
                        identity: LiveCampaignIdentity) -> None:
    observations = []
    observation_path = output_root / "between_leg_observations.jsonl"
    if observation_path.is_file():
        for lineno, line in enumerate(
                observation_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{observation_path}:{lineno}: malformed observation: {exc}") from exc
            body = dict(record)
            digest = body.pop("observation_sha256", None)
            if digest != schemas.content_hash(body):
                raise ValueError(
                    f"{observation_path}:{lineno}: observation hash does not verify")
            witness = record.get("inference_witness")
            if record.get("claim_attestation", {}).get("outcome") != schemas.PASS \
                    or not isinstance(witness, Mapping) or witness.get("competing") is not False:
                raise ValueError(
                    f"{observation_path}:{lineno}: boundary was not cleanly admitted")
            observations.append(record)
    resume_receipt = (None if not (output_root / "resume_receipt.json").is_file()
                      else _load_json(output_root / "resume_receipt.json"))
    _write_json(output_root / "host.json", {
        "schema": "epyc.autokernel.live_control_host_receipt.v2",
        "measured_at": _utc_now(),
        "claim_receipt": claim_receipt.to_dict(),
        "between_leg_policy": BETWEEN_LEG_POLICY,
        "between_leg_observations": observations,
        "between_leg_observations_sha256": _sha256_file(observation_path),
        "resume_receipt": resume_receipt,
        "legs": [{
            "label": material.label,
            "started_at": material.run.started_at,
            "ended_at": material.run.ended_at,
            "complete": material.run.complete,
            "refusals": list(material.run.refusals),
            "raw_ref": f"{identity.evidence_ref}/raw/{material.label}.json",
        } for material in materials],
    })


@dataclass(frozen=True)
class LiveMaterial:
    label: str
    run: microbench.MicrobenchRun

    @property
    def blocks(self) -> tuple:
        return self.run.paired_blocks()


@dataclass(frozen=True)
class _RecordedRun:
    """Minimal immutable replay of a completed raw vector.

    This is intentionally not a second microbench deserializer.  It exposes only
    the fields deterministic control composition consumes after measurement, and
    construction is available solely through ``_load_recorded_material``'s
    receipt, schedule, arm-parameter, and completeness checks.
    """

    blocks: tuple
    started_at: str
    ended_at: str
    refusals: tuple
    raw_sha256: str

    @property
    def complete(self) -> bool:
        return not self.refusals and bool(self.blocks)

    def paired_blocks(self) -> tuple:
        if not self.complete:
            raise RuntimeError("recorded control material is incomplete")
        return self.blocks


def _paired_block_from_raw(value: Any) -> statistics.PairedBlock:
    if not isinstance(value, list) or len(value) != 9:
        raise ValueError("raw paired_block must be the canonical nine-field list")
    return statistics.PairedBlock(
        block_index=value[0], unit_id=value[1], stratum=value[2], order=value[3],
        segment=value[4], extension_round=value[5], measured_at=value[6],
        anchor_samples=tuple(value[7]), candidate_samples=tuple(value[8]))


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _load_recorded_material(
        output_root: Path, *, identity: LiveCampaignIdentity, label: str,
        expected_blocks: int, prompt: int, candidate_iqk: str,
        anchor_iqk: str, reps: int = CALIBRATION_REPS) -> tuple[LiveMaterial, Mapping[str, Any]]:
    """Rebuild composition-only material from one fully attested raw vector."""
    path = output_root / "raw" / f"{label}.json"
    raw = _load_json(path)
    expected_seed = hashlib.sha256(
        f"{identity.campaign_seed}/{label}".encode("utf-8")).hexdigest()
    expected_candidate = f"akc-control-{label}"
    checks = {
        "schema": raw.get("schema") == "epyc.autokernel.microbench_raw_vector.v1",
        "recipe_id": raw.get("recipe_id") == RECIPE_ID,
        "candidate_id": raw.get("candidate_id") == expected_candidate,
        "campaign_seed": raw.get("campaign_seed_sha256") == expected_seed,
        "complete": raw.get("complete") is True,
        "refusals": raw.get("refusals") == [],
        "order_control": (
            isinstance(raw.get("order_control"), Mapping)
            and raw["order_control"].get("outcome") == schemas.PASS),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"{path}: receipt checks failed: {failed}")
    block_rows = raw.get("blocks")
    if not isinstance(block_rows, list) or len(block_rows) != expected_blocks:
        raise ValueError(
            f"{path}: expected {expected_blocks} blocks, got "
            f"{len(block_rows) if isinstance(block_rows, list) else 'non-list'}")
    blocks = []
    expected_pairs = _fresh_pairs_per_block()
    for index, block_row in enumerate(block_rows):
        if not isinstance(block_row, Mapping) or block_row.get("complete") is not True \
                or block_row.get("refusals") != []:
            raise ValueError(f"{path}: block {index} is not complete and refusal-free")
        block = _paired_block_from_raw(block_row.get("paired_block"))
        if block.block_index != index:
            raise ValueError(f"{path}: block index {block.block_index} != {index}")
        if block.unit_id != _unit_id(label=label, prompt=prompt):
            raise ValueError(f"{path}: block {index} carries the wrong unit id")
        plan = block_row.get("plan")
        invocations = block_row.get("invocations")
        if not isinstance(plan, Mapping) or plan.get("pairs") != expected_pairs \
                or not isinstance(invocations, list) \
                or len(invocations) != 2 * expected_pairs \
                or len(block.anchor_samples) != expected_pairs * reps \
                or len(block.candidate_samples) != expected_pairs * reps:
            raise ValueError(
                f"{path}: block {index} does not match the declared "
                f"{expected_pairs}-fresh-pair aggregation frame")
        blocks.append(block)
    schedule = raw.get("order_schedule")
    if not isinstance(schedule, Mapping) or schedule.get("orders") != [
            block.order for block in blocks]:
        raise ValueError(f"{path}: recorded orders do not match the committed schedule")
    for arm, expected_iqk in (
            ("candidate_receipt", candidate_iqk),
            ("anchor_receipt", anchor_iqk)):
        receipt = raw.get(arm)
        params = receipt.get("params") if isinstance(receipt, Mapping) else None
        token_key = "n_prompt" if RECIPE_ID == PREFILL_RECIPE_ID else "n_gen"
        if not isinstance(params, Mapping) or params.get("ggml_iqk") != expected_iqk \
                or params.get(token_key) != prompt or params.get("reps") != reps \
                or params.get("model") != str(MODEL):
            raise ValueError(f"{path}: {arm} does not match the declared arm parameters")
    run = _RecordedRun(
        blocks=tuple(blocks), started_at=str(raw.get("started_at")),
        ended_at=str(raw.get("ended_at")), refusals=tuple(raw["refusals"]),
        raw_sha256=_sha256_file(path))
    return LiveMaterial(label, run), raw


def _params(*, prompt: int, reps: int = CALIBRATION_REPS) -> dict:
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
        raise ValueError("live-control repetitions must be a positive integer")
    token_key = "n_prompt" if RECIPE_ID == PREFILL_RECIPE_ID else "n_gen"
    return {"model": str(MODEL), token_key: prompt, "reps": reps,
            "autokernel_seed": 2026081101,
            "output_format": "json"}


def _unit_id(*, label: str, prompt: int) -> str:
    unit = "pp" if RECIPE_ID == PREFILL_RECIPE_ID else "tg"
    return f"{MODEL.name}:{unit}{prompt}:{label}"


def _physical_envelope(*, label: str, prompt: int) -> physical_bounds.PhysicalEnvelope:
    params = _params(prompt=prompt)
    return physical_bounds.PhysicalEnvelope(
        shape_id=_unit_id(label=label, prompt=prompt),
        delivered_unit="token",
        flops_per_unit=2.0 * DENSE_ACTIVE_PARAMS_LOWER_BOUND,
        bytes_per_unit=(MODEL_BYTES_FRACTION_LOWER_BOUND * MODEL.stat().st_size / prompt),
        peak_compute_flops_s=CPU_PEAK_COMPUTE_FLOPS_S_UPPER_BOUND,
        peak_memory_bytes_s=CPU_PEAK_MEMORY_BYTES_S_UPPER_BOUND,
        measurement_frame_sha256=physical_bounds.measurement_frame_sha256(
            RECIPE_ID, params),
        work_derivation_ref=WORK_DERIVATION_REF,
        hardware_peak_ref=HARDWARE_PEAK_REF,
    )


def _declared_physical_envelopes() -> dict:
    return {
        label: _physical_envelope(label=label, prompt=prompt)
        for label, prompt in CONTROL_PROMPT_BY_LABEL.items()
    }


def _append_observation(output_root: Path, record: Mapping[str, Any]) -> None:
    """Append and fsync one between-leg observation before proceeding."""
    path = output_root / "between_leg_observations.jsonl"
    payload = (schemas.canonical_json(record) + "\n").encode("utf-8")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o644)
    try:
        if os.write(descriptor, payload) != len(payload):
            raise OSError(f"short write while appending {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _observe_between_legs(
        output_root: Path, *, boundary: str, claim: object) -> dict[str, Any]:
    """Record ordinary load and refuse only witnessed competing inference.

    Load average is a lagging signal which includes this campaign's completed
    leg. It is useful context for interpreting noisy exploratory measurements,
    but never a wait/refusal input. Claim integrity and the read-only inference
    identity scan remain hard gates.
    """
    held = microbench.CpuRegionClaimAdapter(claim, cpu_list=CPU_LIST).attest()
    load1 = None
    load_error = None
    try:
        load1 = float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError) as exc:
        load_error = f"{type(exc).__name__}: {exc}"
    witness_error = None
    try:
        witness = screening_baseline.competing_inference_witness()
    except screening_baseline.BaselineBankError as exc:
        witness = None
        witness_error = str(exc)
    body: dict[str, Any] = {
        "schema": "epyc.autokernel.between_leg_observation.v1",
        "campaign_boundary": boundary,
        "observed_at": _utc_now(),
        "policy": BETWEEN_LEG_POLICY,
        "ordinary_load": {
            "load1": load1, "read_error": load_error,
            "disposition": "recorded_as_noise_not_a_gate",
        },
        "claim_attestation": {
            "claim_id": held.claim_id, "holder": held.holder,
            "cpu_list": held.cpu_list, "observed_at": held.observed_at,
            "outcome": held.check.outcome,
            "reasons": list(held.check.reasons),
        },
        "inference_witness": witness,
        "inference_witness_error": witness_error,
    }
    record = {**body, "observation_sha256": schemas.content_hash(body)}
    _append_observation(output_root, record)
    if held.check.outcome != schemas.PASS:
        raise RuntimeError(
            f"between-leg boundary {boundary!r} lost its CPU claim: "
            f"{'; '.join(held.check.reasons)}")
    if witness is None:
        raise RuntimeError(
            f"between-leg boundary {boundary!r} could not witness inference identity: "
            f"{witness_error}")
    if witness["competing"]:
        raise RuntimeError(
            f"between-leg boundary {boundary!r} witnessed competing model inference")
    return record


def _measure(*, label: str, blocks: int, claim: object,
             candidate_binding: recipes.ToolBinding,
             anchor_binding: recipes.ToolBinding,
             anchor: api.AnchorIdentity, prompt: int | None = None,
             candidate_iqk: str, anchor_iqk: str,
             output_root: Path,
             host_state: Callable[..., microbench.HostState],
             identity: LiveCampaignIdentity) -> LiveMaterial:
    # A Python default captures the object at function-definition time.  The
    # CLI selects its recipe after argument parsing, so ``= PROMPT_TOKENS``
    # here would retain pp512 even after ``configure_recipe`` selects tg128.
    # Resolve the process-local recipe frame at call time instead.
    prompt = PROMPT_TOKENS if prompt is None else prompt
    # An A/A vector measures process/path noise as well as binary bytes.  The
    # sealed copy is therefore BOTH arms whenever their declared factor values
    # are equal.  Only a real IQK intervention is allowed to retain the
    # worktree anchor path as the second executable path.
    if candidate_iqk == anchor_iqk:
        anchor_binding = candidate_binding
    declared_prompt = CONTROL_PROMPT_BY_LABEL.get(label)
    if declared_prompt != prompt:
        unit = "pp" if RECIPE_ID == PREFILL_RECIPE_ID else "tg"
        raise ValueError(
            f"control {label!r} at {unit}{prompt} was not predeclared; expected "
            f"{None if declared_prompt is None else f'{unit}{declared_prompt}'}")
    envelope = _declared_physical_envelopes()[label]
    plan = microbench.MicrobenchPlan(
        recipe_id=RECIPE_ID, candidate_id=f"akc-control-{label}",
        campaign_seed=f"{identity.campaign_seed}/{label}",
        candidate_binding=candidate_binding, anchor_binding=anchor_binding,
        anchor=anchor, params=_params(prompt=prompt, reps=CALIBRATION_REPS),
        candidate_instrument_root=str(INSTRUMENT_ROOT),
        anchor_instrument_root=str(INSTRUMENT_ROOT),
        candidate_param_overrides={"ggml_iqk": candidate_iqk},
        anchor_param_overrides={"ggml_iqk": anchor_iqk},
        base_blocks=blocks, pairs_per_block=_fresh_pairs_per_block(),
        unit_ids=(_unit_id(label=label, prompt=prompt),),
        physical_envelopes={
            _unit_id(label=label, prompt=prompt): envelope},
        stratum=api.STRATUM_SELECTION, timeout_s=300.0)
    sandbox_root = output_root / "candidate-sandbox"
    sandbox_root.mkdir(mode=0o700, exist_ok=True)
    sandbox_policy = sandbox.SandboxPolicy(writable_root=str(sandbox_root))
    runner = microbench.MicrobenchRunner(
        claim=microbench.CpuRegionClaimAdapter(claim, cpu_list=CPU_LIST),
        policy=microbench.HostStatePolicy(
            nominal_khz=NOMINAL_KHZ, require_package_power=True),
        spawner=microbench.SubprocessSpawner(
            workdir_root=str(sandbox_root), sandbox_policy=sandbox_policy),
        host_state=host_state)
    run = runner.run(plan)
    _write_json(output_root / "raw" / f"{label}.json", run.raw_vector())
    if not run.complete:
        raise RuntimeError(f"{label} refused: {'; '.join(run.refusals)}")
    return LiveMaterial(label, run)


def _control_stopping_rule() -> statistics.StoppingRule:
    """The precommitted window that makes the positive control reachable."""
    return statistics.StoppingRule(
        rule_id="ak-stop-live-controls/v1", final_table="t1_paired_block_table",
        decisions=(("evidence_threshold_crossed", "compose_into_champion_lineage"),
                   ("extension_exhausted", "abandon"),
                   ("block_ceiling_reached", "abandon")),
        extension=statistics.BoundedExtension(
            max_rounds=CONTROL_EXTENSION_ROUNDS,
            blocks_per_round=CONTROL_EXTENSION_BLOCKS),
        max_blocks_per_candidate=20)


def _campaign_inputs(aa: LiveMaterial, neutral: LiveMaterial,
                     identity: LiveCampaignIdentity
                     ) -> tuple[api.CampaignControls, statistics.StoppingRule,
                                statistics.EProcessConstruction,
                                statistics.StratumSplitRule,
                                statistics.CalibrationSolve]:
    declared = api.CampaignControls(
        calibration_block_count=CALIBRATION_BLOCKS,
        contribution_floor=CONTRIBUTION_FLOOR, max_candidates=10,
        confirmation_admission_count=2, max_blocks_per_candidate=20,
        storage_floor_bytes_free=200 * 1024 ** 3)
    rule = _control_stopping_rule()
    construction = statistics.select_construction(
        "sign_martingale_predictable_lambda/v1")
    split = statistics.StratumSplitRule(
        rule_id="ak-split-live-controls/v1", campaign_seed=identity.campaign_seed,
        confirmation_fraction=0.3,
        rotation=statistics.RotationSchedule(
            schedule_id="ak-rotation-live-controls/v1", period_campaigns=4))
    anchors = tuple(
        statistics.median(block.anchor_samples) for block in aa.blocks)
    inputs = statistics.CalibrationInputs(
        backend="llama_cpu", phase=recipes.get_recipe(RECIPE_ID).phase,
        cell_class=recipes.CELL_CLASS_TINY_GRAPH,
        campaign_seed=identity.campaign_seed, controls=declared, stopping_rule=rule,
        construction=construction, effect_scale=statistics.EFFECT_SCALE_RELATIVE,
        metric_direction="higher_better",
        hypothesis=statistics.HYPOTHESIS_IMPROVEMENT, margin=0.0,
        aa_blocks=aa.blocks, neutral_blocks=neutral.blocks,
        anchor_calibration_values=anchors,
        samples_ref=f"{identity.evidence_ref}/raw/aa_calibration.json")
    return declared, rule, construction, split, controls.run_calibration_block(inputs)


class _EvidenceGateRunner:
    def __init__(self, degraded_candidate_id: str, evidence_root: str) -> None:
        self.degraded_candidate_id = degraded_candidate_id
        self.evidence_root = evidence_root

    def run_gates(self, request: api.EvaluationRequest) -> tuple:
        mismatch = request.candidate_id == self.degraded_candidate_id
        correctness = schemas.Check(
            schemas.FAIL,
            ("captured candidate row reports pp2048 while the committed control cell is "
             "pp512; the candidate changed the work and is ineligible for a speed rank",)
        ) if mismatch else schemas.Check(
            schemas.PASS,
            (CURRENT_SOURCE_CORRECTNESS_REASON,))
        return (
            api.GateResult(
                gate_id="live.recipe_and_correctness", gate_class=api.GATE_CORRECTNESS,
                check=correctness, requires_anchor=True,
        evidence_ref=f"{self.evidence_root}/raw"),
            api.GateResult(
                gate_id="live.numerical_safety", gate_class=api.GATE_NUMERICAL_SAFETY,
                check=(schemas.Check(schemas.FAIL, correctness.reasons) if mismatch
                       else schemas.Check(schemas.PASS)),
                requires_anchor=True,
                evidence_ref=("data/kernel-v8-candidate/iqk-real-model-correctness/"
                              "run-20260725T102000Z-67a433bf4/")),
        )


def _artifact(*, source_sha: str, binary_sha: str,
              linkage_sha: str) -> api.ArtifactIdentity:
    return api.ArtifactIdentity(
        source_sha256=source_sha,
        binary_sha256=binary_sha, linkage_sha256=linkage_sha)


def _fixture(control_id: str, material: Sequence[statistics.PairedBlock], *,
             tier: str, source_sha: str, binary_sha: str, linkage_sha: str,
             measured_at: str) -> control_runner.ControlFixture:
    definition = next(d for d in controls.CONTROL_DEFINITIONS
                      if d.control_id == control_id)
    return control_runner.ControlFixture(
        fixture_id=definition.fixture_id, control_id=control_id, tier=tier,
        candidate_id=f"akc-control-{control_id.replace('_', '-')}",
        artifact=_artifact(source_sha=source_sha, binary_sha=binary_sha,
                           linkage_sha=linkage_sha),
        determinism=api.DeterminismReport(
            determinism_class="not_measured", same_seed_repeat_runs=0),
        created_at=measured_at, measured_at=measured_at,
        stratum=api.STRATUM_SELECTION,
        anchor_samples=tuple(b.anchor_samples for b in material),
        candidate_samples=tuple(b.candidate_samples for b in material))


def _evaluate_controls(*, solve: statistics.CalibrationSolve,
                       declared: api.CampaignControls,
                       rule: statistics.StoppingRule,
                       construction: statistics.EProcessConstruction,
                       split: statistics.StratumSplitRule,
                       fixtures: tuple, anchor: api.AnchorIdentity,
                       recipe_receipt: api.RecipeReceipt,
                       claim_receipt: cpu_region_claim.RegionClaimReceipt,
                       source_sha: str, measured_at: str,
                       output_root: Path,
                       identity: LiveCampaignIdentity) -> control_runner.SweepResult:
    outputs = solve.require_accepted()
    commitment = statistics.StoppingRuleCommitment.commit(
        rule, campaign_id=identity.campaign_id, committed_at=measured_at)
    campaign_stats = statistics.CampaignStatistics(
        campaign_id=identity.campaign_id, campaign_seed=identity.campaign_seed,
        effect_scale=statistics.EFFECT_SCALE_RELATIVE,
        hypothesis=statistics.HYPOTHESIS_IMPROVEMENT, margin=0.0,
        stopping_rule=rule, stopping_rule_commitment=commitment,
        split_rule=split, construction=construction, calibration=outputs,
        aa_effect_pool=solve.aa_effect_pool,
        anchor_calibration_values=solve.anchor_calibration_values)
    degraded_candidate_id = next(
        f.candidate_id for f in fixtures
        if f.control_id == controls.CONTROL_DEGRADED_NEGATIVE)
    dispatcher = api.TierDispatcher(gate_runners={
        tier: _EvidenceGateRunner(degraded_candidate_id, identity.evidence_ref)
        for tier in ("T0", "T1", "T1b", "T2")})
    pipeline = control_runner.DispatchPipeline(
        dispatcher=dispatcher,
        reducer=statistics.PairedBlockReducer(campaign_stats))
    fixture_digest = schemas.content_hash(control_runner._fixture_payload(fixtures))
    fixture_set = control_runner.resolve_fixture_set(
        fixtures=fixtures, pinned_digest=fixture_digest,
        source_label=f"{identity.campaign_id}/live-fixtures")
    evaluator_hash = schemas.content_hash({
        "controls": controls.CONTROL_DEFINITIONS_DIGEST,
        "runner": control_runner.CONTROL_RUNNER_ID})
    binding = control_runner.CampaignBinding(
        campaign_id=identity.campaign_id, backend="llama_cpu", phase=recipes.get_recipe(RECIPE_ID).phase,
        cell_class=recipes.CELL_CLASS_TINY_GRAPH,
        protocol_id=api.PROTOCOL_VERSIONED_ID,
        evaluator=api.EvaluatorIdentity(
            id="P-AK-SEARCH-1/v1", bundle_sha256=evaluator_hash,
            runtime_source_label_ref=(
                f"{identity.evidence_ref}/runtime-source-label.json")),
        scope_denominator=api.ScopeDenominator(
            machine_subset="full", numa_nodes=(), devices=(), cores=96),
        scope_manifest_sha256=schemas.content_hash({"cpu_list": CPU_LIST}),
        co_residency="single", metric=recipes.get_recipe(RECIPE_ID).metric,
        metric_direction="higher_better", reps=1, change_class="parameter",
        anchor=anchor,
        campaign_controls=declared, calibration=outputs)
    runner = control_runner.ExecutedControlRunner(
        pipeline=pipeline, fixtures=fixture_set, binding=binding,
        campaign_statistics=campaign_stats)
    declaration = controls.HistoricalWinReplayDeclaration(
        win_id=("iqk-prefill-port" if RECIPE_ID == PREFILL_RECIPE_ID else "iqk-decode-control"),
        backend="llama_cpu", phase=recipes.get_recipe(RECIPE_ID).phase,
        reference_direction="higher_better",
        reference_band=controls.ReferenceBand(low=0.03, high=0.60),
        evidence_locator=("data/kernel-v8-candidate/cpu-prefill-regression/"
                          "run-20260725T155655Z-v4-waive-q8-kfd-procrace-swapoff/"
                          "summary.json"), durability_class="carried_in_git")
    bundle = controls.resolve_control_bundle(
        pinned_definitions_digest=controls.CONTROL_DEFINITIONS_DIGEST,
        aa_cadence=controls.AACadence(
            every_n_windows=5, every_n_seconds=3600.0, declared_at=measured_at),
        seed_rotation=controls.SeedRotationSchedule(
            rotate_every_windows=10, declared_at=measured_at),
        historical_win_replays=(declaration,),
        source_label=f"{identity.campaign_id}/evaluator-bundle")
    harness = controls.ControlHarness(bundle=bundle, runner=runner)
    sweep = control_runner.ControlSweep(
        harness=harness, campaign_seed=identity.campaign_seed)
    def passed(reason: str) -> schemas.Check:
        return schemas.Check(schemas.PASS, (reason,))

    claim_open = passed(
        f"claim {claim_receipt.claim_id} acquired at {claim_receipt.acquired_at}")
    claim_close = passed(
        f"claim {claim_receipt.claim_id} released at {claim_receipt.released_at}")
    same_holder = passed(
        f"the immutable receipt binds open and close to pid {claim_receipt.holder_pid}")
    provisional = passed(
        "provisional carrier for the control-evaluation window; the sweep below "
        "derives and exports the measured panel")
    window = api.WindowAttestations(
        resource_claim_receipt=claim_receipt.claim_id,
        resource_claim_open=claim_open, resource_claim_close=claim_close,
        resource_claim_same_holder=same_holder,
        no_concurrent_inference=passed(
            "operator handed the CPU inference lane to this session; every measured "
            "invocation also passed its run-open load and held-claim checks"),
        preflight_attestation_ref=f"{identity.evidence_ref}/preflight.json",
        host_receipt=f"{identity.evidence_ref}/host.json",
        host_health=passed(
            "all raw measurement legs completed with their host-state checks recorded"),
        anchor_at_open=anchor, anchor_at_close=anchor,
        anchor_gate=passed("the fresh calibration solve was accepted"),
        evaluator_bundle=passed(
            f"control definitions resolve to {controls.CONTROL_DEFINITIONS_DIGEST}"),
        runtime_source_label=passed(
            f"runtime source manifest binds production {PRODUCTION_COMMIT} and "
            f"instrument {INSTRUMENT_COMMIT} to {source_sha}"),
        recipe=recipe_receipt,
        storage_open=passed("campaign preflight satisfied the declared storage floor"),
        storage_close=passed("storage remained above the declared floor at close"),
        strata=passed("all control blocks use the declared selection stratum"),
        stopping_rule_id=rule.rule_id,
        rule_immutability=passed(
            f"the evaluated stopping rule hashes to {rule.content_hash()}"),
        order_randomized=passed(
            "control block orders are re-derived from the committed campaign seed"),
        order_seed=identity.campaign_seed,
        aa_cadence=passed("this sweep contains its declared A/A control"),
        controls=api.ControlPanel(
            positive=provisional, neutral=provisional,
            degraded_negative=provisional, aa=provisional,
            historical_replay=provisional),
        calibration=passed("fresh A/A and neutral pools produced an accepted solve"),
        control_definitions_immutable=passed(
            f"the pinned control definitions digest is {controls.CONTROL_DEFINITIONS_DIGEST}"),
        raw_evidence_ref=f"{identity.evidence_ref}/raw/")
    historical = controls.HistoricalWinResolution(
        backend="llama_cpu", available=True, declaration=declaration,
        durability_outcome=schemas.PASS,
        check=schemas.Check(schemas.PASS, ("historical fixture resolves in git",)))
    run_context = controls.ControlRunContext(
        campaign_id=identity.campaign_id, backend="llama_cpu", phase=recipes.get_recipe(RECIPE_ID).phase,
        cell_class=recipes.CELL_CLASS_TINY_GRAPH, window_id=identity.window_id, tier="T1",
        seed="DERIVED-BY-SWEEP", anchor=anchor, declaration=declaration)
    context = controls.ControlContext(
        campaign_id=identity.campaign_id, backend="llama_cpu", phase=recipes.get_recipe(RECIPE_ID).phase,
        cell_class=recipes.CELL_CLASS_TINY_GRAPH, window_id=identity.window_id,
        historical=historical,
        neutral_dispersion=controls.neutral_dispersion_check(solve),
        calibration=outputs)
    return sweep.run(
        run_context=run_context, context=context, window=window,
        aa_cadence=window.aa_cadence, windows_completed=0, last_rotation_epoch=0)


def _compose_measured_controls(
        *, output_root: Path, identity: LiveCampaignIdentity,
        solve: statistics.CalibrationSolve, declared: api.CampaignControls,
        rule: statistics.StoppingRule,
        construction: statistics.EProcessConstruction,
        split: statistics.StratumSplitRule,
        materials: Mapping[str, LiveMaterial], anchor: api.AnchorIdentity,
        recipe_receipt: api.RecipeReceipt,
        claim_receipt: cpu_region_claim.RegionClaimReceipt,
        source_sha: str, binary_sha: str, linkage_sha: str,
        measured_at: str) -> control_runner.SweepResult:
    """Pure post-measurement composition shared by live and recovery paths."""
    aa = materials["aa_calibration"]
    neutral = materials["neutral_calibration"]
    positive = materials["positive"]
    historical = materials["historical_win_replay"]
    negative_anchor = materials["negative_committed_cell"]
    negative_wrong = materials["negative_wrong_cell"]
    control_blocks = rule.max_total_blocks(solve.require_accepted().b_min_blocks)
    negative_blocks = tuple(
        statistics.PairedBlock(
            block_index=i, unit_id=f"negative-work-mismatch-{i}",
            stratum=api.STRATUM_SELECTION, order=a.order,
            anchor_samples=a.anchor_samples,
            candidate_samples=w.candidate_samples, measured_at=measured_at)
        for i, (a, w) in enumerate(zip(negative_anchor.blocks,
                                       negative_wrong.blocks)))
    fixtures = (
        _fixture(controls.CONTROL_POSITIVE, positive.blocks, tier="T1",
                 source_sha=source_sha, binary_sha=binary_sha,
                 linkage_sha=linkage_sha, measured_at=positive.run.ended_at),
        _fixture(controls.CONTROL_NEUTRAL, neutral.blocks[:control_blocks], tier="T1",
                 source_sha=source_sha, binary_sha=binary_sha,
                 linkage_sha=linkage_sha, measured_at=neutral.run.ended_at),
        _fixture(controls.CONTROL_DEGRADED_NEGATIVE, negative_blocks, tier="T1",
                 source_sha=source_sha, binary_sha=binary_sha,
                 linkage_sha=linkage_sha, measured_at=measured_at),
        _fixture(controls.CONTROL_AA, aa.blocks[:control_blocks], tier="T1",
                 source_sha=source_sha, binary_sha=binary_sha,
                 linkage_sha=linkage_sha, measured_at=aa.run.ended_at),
        _fixture(controls.CONTROL_HISTORICAL_WIN_REPLAY, historical.blocks,
                 tier="T2", source_sha=source_sha, binary_sha=binary_sha,
                 linkage_sha=linkage_sha, measured_at=historical.run.ended_at),
    )
    return _evaluate_controls(
        solve=solve, declared=declared, rule=rule,
        construction=construction, split=split, fixtures=fixtures,
        anchor=anchor, recipe_receipt=recipe_receipt,
        claim_receipt=claim_receipt, source_sha=source_sha,
        measured_at=measured_at, output_root=output_root, identity=identity)


def _attest_anchor_motion_transition(
        output_root: Path, *, identity: LiveCampaignIdentity,
        claim: object) -> dict:
    """Witness the claim and absence of competing inference without waiting.

    The immediately preceding neutral leg is expected to remain visible in
    loadavg. That value is recorded by ``_observe_between_legs`` but cannot
    block or delay this non-ranked anchor trace.
    """
    samples = [
        _observe_between_legs(
            output_root, boundary="neutral_to_anchor_motion", claim=claim)
        for _index in range(int(ANCHOR_MOTION_SETTLING["required_samples"]))
    ]
    receipt = {
        "schema": "epyc.autokernel.anchor_motion_transition_receipt.v2",
        "campaign_id": identity.campaign_id, "settling": ANCHOR_MOTION_SETTLING,
        "samples": samples, "completed_at": _utc_now(),
    }
    receipt["receipt_sha256"] = schemas.content_hash(receipt)
    _write_json(output_root / "anchor_motion_settling.json", receipt)
    return receipt


def _anchor_motion_authority(
        output_root: Path, *, material: LiveMaterial,
        settling_receipt: Mapping[str, Any]) -> dict:
    """Derive, rather than type, the current T1-window anchor movement bound."""
    if material.label != ANCHOR_MOTION_LABEL:
        raise ValueError("anchor-motion authority requires its dedicated non-ranked trace")
    anchors = tuple(statistics.median(block.anchor_samples) for block in material.blocks)
    if len(anchors) != ANCHOR_MOTION_WINDOW_BLOCKS:
        raise ValueError("anchor-motion trace does not cover its declared campaign window")
    raw_path = (output_root / "raw" / f"{ANCHOR_MOTION_LABEL}.json").resolve()
    raw = _load_json(raw_path)
    return {
        "schema": "epyc.autokernel.anchor_motion_window.v1",
        "label": ANCHOR_MOTION_LABEL,
        "window_blocks": ANCHOR_MOTION_WINDOW_BLOCKS,
        "settling": ANCHOR_MOTION_SETTLING,
        "settling_receipt_ref": str(output_root / "anchor_motion_settling.json"),
        "settling_receipt_sha256": settling_receipt["receipt_sha256"],
        "raw_ref": str(raw_path), "raw_sha256": schemas.content_hash(raw),
        "anchor_medians": list(anchors),
        "bound": campaign.drift_bound_from(anchors),
    }


def _resume_amendment(
        output_root: Path, *, identity: LiveCampaignIdentity,
        declaration: Mapping[str, Any]) -> dict[str, Any] | None:
    """Bind the pre-policy r2 declaration to the operator's new transition rule.

    The original declaration is immutable. A narrow, self-hashed amendment may
    replace only the obsolete load-wait settling field and add the inference-
    identity policy. Fresh declarations already carry the current fields and
    need no amendment.
    """
    original = declaration.get("anchor_motion_settling")
    if original == ANCHOR_MOTION_SETTLING \
            and declaration.get("between_leg_policy") == BETWEEN_LEG_POLICY:
        return None
    if original != LEGACY_ANCHOR_MOTION_SETTLING \
            or declaration.get("between_leg_policy") is not None:
        raise ValueError(
            "resume refuses a declaration outside the one legacy load-wait policy")
    body = {
        "schema": RESUME_AMENDMENT_SCHEMA,
        "campaign_id": identity.campaign_id,
        "original_declaration_sha256": schemas.content_hash(declaration),
        "replaced_field": "anchor_motion_settling",
        "original_value": LEGACY_ANCHOR_MOTION_SETTLING,
        "replacement_value": ANCHOR_MOTION_SETTLING,
        "added_between_leg_policy": BETWEEN_LEG_POLICY,
        "authority": "operator_policy_20260813_ordinary_load_is_measurement_noise",
    }
    amendment = {**body, "amendment_sha256": schemas.content_hash(body)}
    path = output_root / "resume_amendment.json"
    if path.exists():
        if _load_json(path) != amendment:
            raise ValueError("existing resume amendment differs from the exact policy bridge")
    return amendment


def _validate_resume_existing(
        output_root: Path, *, identity: LiveCampaignIdentity,
) -> dict[str, Any]:
    """Admit exactly one completed AA leg from a dead, interrupted claimant."""
    if not output_root.is_dir() or output_root.is_symlink():
        raise ValueError("--resume-existing requires an existing non-symlink directory")
    forbidden = (
        "summary.json", "control_sweep.json", "calibration.json", "host.json",
        "claim_receipt.json", "resume_receipt.json")
    present = [name for name in forbidden if (output_root / name).exists()]
    if present:
        raise ValueError(f"resume refuses material beyond the AA-only boundary: {present}")
    raw_files = sorted(path.name for path in (output_root / "raw").glob("*.json"))
    if raw_files != ["aa_calibration.json"]:
        raise ValueError(
            f"resume requires exactly raw/aa_calibration.json; found {raw_files}")

    declaration = _load_json(output_root / "campaign_declaration.json")
    checks = {
        "schema": declaration.get("schema")
                  == "epyc.autokernel.live_control_campaign_declaration.v1",
        "campaign_id": declaration.get("campaign_id") == identity.campaign_id,
        "campaign_seed": declaration.get("campaign_seed_sha256") == hashlib.sha256(
            identity.campaign_seed.encode("utf-8")).hexdigest(),
        "window_id": declaration.get("window_id") == identity.window_id,
        "recipe_id": declaration.get("recipe_id") == RECIPE_ID,
        "cpu_list": declaration.get("cpu_list") == CPU_LIST,
        "model": declaration.get("model") == str(MODEL),
        "model_sha256": declaration.get("model_sha256") == _sha256_file(MODEL),
        "token_fields": all(
            declaration.get(key) == value for key, value in _token_fields().items()),
        "calibration_blocks": declaration.get("calibration_blocks")
                              == CALIBRATION_BLOCKS,
        "neutral_blocks": declaration.get("neutral_blocks") == NEUTRAL_BLOCKS,
        "calibration_frame": declaration.get("calibration_frame")
                             == _calibration_frame(),
        "physical_envelopes": declaration.get("physical_envelopes") == {
            label: envelope.to_dict()
            for label, envelope in sorted(_declared_physical_envelopes().items())},
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"resume declaration checks failed: {failed}")
    amendment = _resume_amendment(
        output_root, identity=identity, declaration=declaration)

    runtime = _load_json(output_root / "runtime-source-label.json")
    runtime_body = {key: value for key, value in runtime.items()
                    if key != "source_sha256"}
    source_sha = runtime.get("source_sha256")
    if source_sha != schemas.content_hash(runtime_body) \
            or declaration.get("source_sha256") != source_sha:
        raise ValueError("resume runtime source label does not verify")
    if runtime.get("production_source_commit") != PRODUCTION_COMMIT \
            or runtime.get("measurement_instrument_commit") != INSTRUMENT_COMMIT \
            or runtime.get("binary_copy_exact") is not True:
        raise ValueError("resume runtime identities differ from the live instrument")
    copied_binary = output_root / "anchor_binary_copy" / "llama-bench"
    if _sha256_file(copied_binary) != runtime.get("copied_binary_sha256") \
            or _sha256_file(INSTRUMENT_BINARY) != runtime.get("measurement_binary_sha256"):
        raise ValueError("resume measurement or copied binary hash differs")
    copy_linkage, copy_text = _linkage(copied_binary, copied_binary.parent)
    instrument_linkage, instrument_text = _linkage(
        INSTRUMENT_BINARY, INSTRUMENT_BINARY.parent)
    recorded_copy_text = (output_root / "linkage.copy.txt").read_text(encoding="utf-8")
    recorded_instrument_text = (output_root / "linkage.instrument.txt").read_text(
        encoding="utf-8")
    if _sha256_file(output_root / "linkage.copy.txt") \
            != runtime.get("copied_linkage_sha256") \
            or _sha256_file(output_root / "linkage.instrument.txt") \
            != runtime.get("measurement_linkage_sha256") \
            or _normalized_linkage(copy_text) != _normalized_linkage(recorded_copy_text) \
            or _normalized_linkage(instrument_text) \
            != _normalized_linkage(recorded_instrument_text):
        raise ValueError("resume linkage evidence differs from the recorded runtime")
    if _build_toolchain_manifest(output_root) \
            != runtime.get("measurement_toolchain_manifest_sha256"):
        raise ValueError("resume toolchain identity differs from the recorded runtime")
    preflight = _load_json(output_root / "preflight.json")
    preflight_checks = preflight.get("checks")
    if not isinstance(preflight_checks, Mapping) or not preflight_checks \
            or any(not isinstance(check, Mapping)
                   or check.get("outcome") != schemas.PASS
                   for check in preflight_checks.values()):
        raise ValueError("resume preflight is absent or not all-PASS")

    journal = cpu_region_claim.RegionClaimJournal(output_root / "region_claim.jsonl")
    records = journal.read_all()
    acquired = [record for record in records if record.get("kind") == "claim_acquired"]
    if len(acquired) != 1 or any(record.get("kind") == "claim_released"
                                 for record in records):
        raise ValueError("resume requires one interrupted acquisition and no release")
    previous_payload = acquired[0].get("detail", {}).get("receipt")
    previous = cpu_region_claim.RegionClaimReceipt.from_dict(previous_payload)
    if previous.campaign_id != identity.campaign_id or previous.cpu_list != CPU_LIST:
        raise ValueError("resume previous claim belongs to another campaign or footprint")
    liveness = device_claim.assess_holder_liveness({
        "pid": previous.holder_pid, "start_ticks": previous.holder_start_ticks,
        "boot_id": previous.holder_boot_id, "host": previous.host,
    })
    if liveness.state != device_claim.DEAD:
        raise ValueError(
            f"resume previous claimant is not provably dead: {liveness.state}: "
            f"{liveness.reason}")

    aa, aa_raw = _load_recorded_material(
        output_root, identity=identity, label="aa_calibration",
        expected_blocks=CALIBRATION_BLOCKS, prompt=PROMPT_TOKENS,
        candidate_iqk=CALIBRATION_IQK, anchor_iqk=CALIBRATION_IQK)
    attestations = aa_raw.get("claim_attestations")
    if not isinstance(attestations, list) or not attestations \
            or any(not isinstance(row, Mapping)
                   or row.get("claim_id") != previous.claim_id
                   or row.get("cpu_list") != CPU_LIST
                   or row.get("outcome") != schemas.PASS
                   for row in attestations):
        raise ValueError("resume AA raw vector is not fully bound to the dead held claim")
    expected_binding = runtime.get("aa_sealed_binding")
    arm_receipts = (aa_raw.get("candidate_receipt"), aa_raw.get("anchor_receipt"))
    if not isinstance(expected_binding, Mapping) \
            or any(not isinstance(receipt, Mapping) for receipt in arm_receipts) \
            or any({key: receipt.get(key) for key in ("binary_path", "library_path")}
                   != expected_binding for receipt in arm_receipts):
        raise ValueError("resume AA arms do not use the sealed A/A binding")
    candidate_binding = recipes.ToolBinding(
        binary=str(copied_binary), source_root=str(output_root),
        library_path=str(copied_binary.parent))
    # Validation above is intentionally read-only. Persist the narrow policy
    # bridge only after every immutable input, completed AA receipt and dead
    # claimant check succeeds.
    if amendment is not None and not (output_root / "resume_amendment.json").exists():
        _write_json(output_root / "resume_amendment.json", amendment)
    return {
        "declaration": declaration, "amendment": amendment,
        "runtime": runtime, "source_sha": source_sha,
        "instrument_sha": runtime["measurement_binary_sha256"],
        "copy_sha": runtime["copied_binary_sha256"],
        "instrument_linkage": instrument_linkage,
        "copy_linkage": copy_linkage,
        "candidate_binding": candidate_binding,
        "anchor_binding": recipes.ToolBinding(
            binary=str(INSTRUMENT_BINARY), source_root=str(INSTRUMENT_ROOT),
            library_path=str(INSTRUMENT_BINARY.parent)),
        "aa": aa, "aa_raw_sha256": _sha256_file(
            output_root / "raw" / "aa_calibration.json"),
        "previous_claim": previous, "previous_liveness": liveness,
    }


def execute(output_root: Path, *, campaign_id: str, resume_existing: bool = False,
            host_state: Callable[..., microbench.HostState] =
            microbench.read_host_state) -> dict:
    output_root = output_root.resolve()
    identity = LiveCampaignIdentity(
        campaign_id=campaign_id, evidence_ref=str(output_root))
    resume = (_validate_resume_existing(output_root, identity=identity)
              if resume_existing else None)
    if resume is None:
        output_root.mkdir(parents=True, exist_ok=False)
        _ensure_instrument_build()
        if _sha256_file(INSTRUMENT_BINARY) == "":  # pragma: no cover - explicit read gate
            raise RuntimeError("unreadable hardened measurement binary")
        candidate_binding = _copy_anchor_bundle(output_root)
        anchor_binding = recipes.ToolBinding(
            binary=str(INSTRUMENT_BINARY), source_root=str(INSTRUMENT_ROOT),
            library_path=str(INSTRUMENT_BINARY.parent))
        instrument_linkage, instrument_ldd = _linkage(
            INSTRUMENT_BINARY, INSTRUMENT_BINARY.parent)
        copy_linkage, copy_ldd = _linkage(
            Path(candidate_binding.binary), Path(candidate_binding.library_path))
        (output_root / "linkage.instrument.txt").write_text(
            instrument_ldd, encoding="utf-8")
        (output_root / "linkage.copy.txt").write_text(copy_ldd, encoding="utf-8")
        instrument_sha = _sha256_file(INSTRUMENT_BINARY)
        copy_sha = _sha256_file(Path(candidate_binding.binary))
        source_sha = _write_declaration(
            output_root, identity=identity,
            instrument_sha=instrument_sha, copy_sha=copy_sha,
            instrument_linkage=instrument_linkage, copy_linkage=copy_linkage,
            toolchain_manifest_sha256=_build_toolchain_manifest(output_root),
            sealed_binding=candidate_binding)
        _write_preflight(output_root, instrument_sha=instrument_sha, copy_sha=copy_sha,
                         host_state=host_state)
    else:
        candidate_binding = resume["candidate_binding"]
        anchor_binding = resume["anchor_binding"]
        instrument_linkage = resume["instrument_linkage"]
        copy_linkage = resume["copy_linkage"]
        instrument_sha = resume["instrument_sha"]
        copy_sha = resume["copy_sha"]
        source_sha = resume["source_sha"]
    anchor = api.AnchorIdentity(
        source_commit=INSTRUMENT_COMMIT, binary_sha256=instrument_sha,
        linkage_sha256=instrument_linkage, tool="llama-bench")
    journal = cpu_region_claim.RegionClaimJournal(output_root / "region_claim.jsonl")
    materials = []
    anchor_motion = None
    with cpu_region_claim.acquire_cpu_region_claim(
            CPU_LIST, purpose="AutoKernel five-control calibration block",
            campaign_id=identity.campaign_id, journal=journal, timeout_s=60.0,
            max_hold_s=2 * 3600) as claim:
        if resume is None:
            aa = _measure(
                label="aa_calibration", blocks=CALIBRATION_BLOCKS, claim=claim,
                candidate_binding=candidate_binding, anchor_binding=anchor_binding,
                anchor=anchor, candidate_iqk=CONTROL_ARM_IQK["aa_calibration"][0],
                anchor_iqk=CONTROL_ARM_IQK["aa_calibration"][1],
                output_root=output_root, host_state=host_state,
                identity=identity)
            boundary = "aa_to_neutral"
        else:
            aa = resume["aa"]
            new_claim = claim.receipt().to_dict()
            body = {
                "schema": RESUME_RECEIPT_SCHEMA,
                "campaign_id": identity.campaign_id,
                "resumed_at": _utc_now(),
                "resume_point": "after_completed_aa_before_neutral",
                "aa_raw_sha256": resume["aa_raw_sha256"],
                "previous_claim": resume["previous_claim"].to_dict(),
                "previous_claim_liveness": {
                    "state": resume["previous_liveness"].state,
                    "reason": resume["previous_liveness"].reason,
                },
                "new_claim": new_claim,
                "declaration_sha256": schemas.content_hash(resume["declaration"]),
                "runtime_source_sha256": resume["source_sha"],
                "toolchain_manifest_sha256": resume["runtime"][
                    "measurement_toolchain_manifest_sha256"],
                "policy_amendment_sha256": (
                    None if resume["amendment"] is None else
                    resume["amendment"]["amendment_sha256"]),
                "inference_executed_by_resume_validation": False,
            }
            resume_receipt = {**body, "receipt_sha256": schemas.content_hash(body)}
            journal.append("campaign_resumed", claim.plan.scope_id, {
                "resume_receipt": resume_receipt})
            _write_json(output_root / "resume_receipt.json", resume_receipt)
            boundary = "resume_after_aa_to_neutral"
        materials.append(aa)
        _observe_between_legs(output_root, boundary=boundary, claim=claim)
        neutral = _measure(
            label="neutral_calibration", blocks=NEUTRAL_BLOCKS, claim=claim,
            candidate_binding=candidate_binding, anchor_binding=anchor_binding,
            anchor=anchor, candidate_iqk=CONTROL_ARM_IQK["neutral_calibration"][0],
            anchor_iqk=CONTROL_ARM_IQK["neutral_calibration"][1],
            output_root=output_root, host_state=host_state,
            identity=identity)
        materials.append(neutral)
        declared, rule, construction, split, solve = _campaign_inputs(
            aa, neutral, identity)
        _write_json(output_root / "calibration.json", solve.to_dict())
        if solve.accepted:
            outputs = solve.require_accepted()
            n = outputs.b_min_blocks
            if n > ANCHOR_MOTION_WINDOW_BLOCKS:
                raise RuntimeError(
                    f"accepted B_min={n} exceeds the predeclared anchor-motion window "
                    f"of {ANCHOR_MOTION_WINDOW_BLOCKS}; this calibration cannot license "
                    "the ranked campaign length")
            settling_receipt = _attest_anchor_motion_transition(
                output_root, identity=identity, claim=claim)
            anchor_motion_material = _measure(
                label=ANCHOR_MOTION_LABEL, blocks=ANCHOR_MOTION_WINDOW_BLOCKS,
                claim=claim, candidate_binding=candidate_binding,
                anchor_binding=anchor_binding, anchor=anchor,
                candidate_iqk=CONTROL_ARM_IQK[ANCHOR_MOTION_LABEL][0],
                anchor_iqk=CONTROL_ARM_IQK[ANCHOR_MOTION_LABEL][1],
                output_root=output_root, host_state=host_state, identity=identity)
            materials.append(anchor_motion_material)
            anchor_motion = _anchor_motion_authority(
                output_root, material=anchor_motion_material,
                settling_receipt=settling_receipt)
            control_blocks = rule.max_total_blocks(n)
            _observe_between_legs(
                output_root, boundary="anchor_motion_to_positive", claim=claim)
            positive = _measure(
                label="positive", blocks=control_blocks, claim=claim,
                candidate_binding=candidate_binding, anchor_binding=anchor_binding,
                anchor=anchor, candidate_iqk=CONTROL_ARM_IQK["positive"][0],
                anchor_iqk=CONTROL_ARM_IQK["positive"][1],
                output_root=output_root, host_state=host_state, identity=identity)
            materials.append(positive)
            _observe_between_legs(
                output_root, boundary="positive_to_historical", claim=claim)
            historical = _measure(
                label="historical_win_replay", blocks=control_blocks, claim=claim,
                candidate_binding=candidate_binding, anchor_binding=anchor_binding,
                anchor=anchor, candidate_iqk=CONTROL_ARM_IQK["historical_win_replay"][0],
                anchor_iqk=CONTROL_ARM_IQK["historical_win_replay"][1],
                output_root=output_root, host_state=host_state, identity=identity)
            materials.append(historical)
            _observe_between_legs(
                output_root, boundary="historical_to_negative_committed", claim=claim)
            negative_anchor = _measure(
                label="negative_committed_cell", blocks=control_blocks, claim=claim,
                candidate_binding=candidate_binding, anchor_binding=anchor_binding,
                anchor=anchor, prompt=PROMPT_TOKENS,
                candidate_iqk=CONTROL_ARM_IQK["negative_committed_cell"][0],
                anchor_iqk=CONTROL_ARM_IQK["negative_committed_cell"][1],
                output_root=output_root,
                host_state=host_state, identity=identity)
            materials.append(negative_anchor)
            _observe_between_legs(
                output_root, boundary="negative_committed_to_negative_wrong", claim=claim)
            negative_wrong = _measure(
                label="negative_wrong_cell", blocks=control_blocks, claim=claim,
                candidate_binding=candidate_binding, anchor_binding=anchor_binding,
                anchor=anchor, prompt=WRONG_PROMPT_TOKENS,
                candidate_iqk=CONTROL_ARM_IQK["negative_wrong_cell"][0],
                anchor_iqk=CONTROL_ARM_IQK["negative_wrong_cell"][1],
                output_root=output_root,
                host_state=host_state, identity=identity)
            materials.append(negative_wrong)
            ended = _utc_now()
    claim_receipt = claim.receipt()
    _write_json(output_root / "claim_receipt.json", claim_receipt.to_dict())
    _write_host_receipt(output_root, materials, claim_receipt, identity)
    if not solve.accepted:
        summary = {
            "campaign_id": identity.campaign_id, "measured_at": _utc_now(),
            "state": "calibration_rejected", "controls_started": False,
            "production_source_commit": PRODUCTION_COMMIT,
            "measurement_instrument_commit": INSTRUMENT_COMMIT,
            "measurement_binary_sha256": instrument_sha,
            "copied_binary_sha256": copy_sha,
            "binary_copy_exact": instrument_sha == copy_sha,
            "calibration": solve.to_dict(), "controls": None, "may_rank": False,
            "anchor_motion": None,
        }
        _write_json(output_root / "summary.json", summary)
        return summary

    receipt = positive.run.candidate_receipt
    if receipt is None:
        raise RuntimeError("positive run emitted no recipe receipt")
    result = _compose_measured_controls(
        output_root=output_root, identity=identity, solve=solve,
        declared=declared, rule=rule, construction=construction, split=split,
        materials={material.label: material for material in materials},
        anchor=anchor, recipe_receipt=receipt.recipe_receipt,
        claim_receipt=claim_receipt, source_sha=source_sha,
        binary_sha=copy_sha, linkage_sha=copy_linkage, measured_at=ended)
    belief_receipt = _build_belief_receipt(
        output_root, identity=identity, result=result,
        claim_receipt=claim_receipt, measured_at=ended)
    _write_json(output_root / "control_sweep.json", result.to_dict())
    if belief_receipt is not None:
        _write_json(output_root / "belief_receipt.json", belief_receipt)
    summary = {
        "campaign_id": identity.campaign_id, "measured_at": _utc_now(),
        "state": "controls_complete", "controls_started": True,
        "production_source_commit": PRODUCTION_COMMIT,
        "measurement_instrument_commit": INSTRUMENT_COMMIT,
        "measurement_binary_sha256": instrument_sha,
        "copied_binary_sha256": copy_sha,
        "binary_copy_exact": instrument_sha == copy_sha,
        "calibration": solve.to_dict(), "controls": result.to_dict(),
        "anchor_motion": anchor_motion,
        "may_rank": result.may_rank,
        "belief_receipt_sha256": (
            belief_receipt["receipt_sha256"] if belief_receipt else None),
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def evaluate_existing(output_root: Path, *, campaign_id: str) -> dict:
    """Compose a terminal sweep from completed, receipt-bound raw vectors.

    This path performs no inference, takes no claim, and cannot accept partial
    material.  It exists for the narrow failure mode where all governed legs and
    teardown completed but deterministic post-processing raised.  Re-running the
    benchmark would create a second experiment, not repair the first one.
    """
    output_root = output_root.resolve()
    identity = LiveCampaignIdentity(campaign_id=campaign_id,
                                    evidence_ref=str(output_root))
    if not output_root.is_dir():
        raise ValueError(f"completed evidence directory does not exist: {output_root}")
    for terminal in ("summary.json", "control_sweep.json"):
        if (output_root / terminal).exists():
            raise ValueError(
                f"{terminal} already exists; completed composition is immutable")

    declaration = _load_json(output_root / "campaign_declaration.json")
    declared_checks = {
        "schema": declaration.get("schema")
                  == "epyc.autokernel.live_control_campaign_declaration.v1",
        "campaign_id": declaration.get("campaign_id") == identity.campaign_id,
        "campaign_seed": declaration.get("campaign_seed_sha256") == hashlib.sha256(
            identity.campaign_seed.encode("utf-8")).hexdigest(),
        "window_id": declaration.get("window_id") == identity.window_id,
        "recipe_id": declaration.get("recipe_id") == RECIPE_ID,
        "cpu_list": declaration.get("cpu_list") == CPU_LIST,
        "model": declaration.get("model") == str(MODEL),
        "calibration_blocks": declaration.get("calibration_blocks")
                              == CALIBRATION_BLOCKS,
        "neutral_blocks": declaration.get("neutral_blocks") == NEUTRAL_BLOCKS,
        "calibration_frame": declaration.get("calibration_frame") == _calibration_frame(),
        "contribution_floor": declaration.get("contribution_floor")
                              == CONTRIBUTION_FLOOR,
    }
    failed = sorted(name for name, passed in declared_checks.items() if not passed)
    if failed:
        raise ValueError(f"campaign declaration checks failed: {failed}")

    runtime = _load_json(output_root / "runtime-source-label.json")
    source_sha = runtime.get("source_sha256")
    runtime_body = {key: value for key, value in runtime.items()
                    if key != "source_sha256"}
    if source_sha != schemas.content_hash(runtime_body) \
            or declaration.get("source_sha256") != source_sha:
        raise ValueError("runtime source label does not hash to the declared source")
    copy_sha = str(runtime.get("copied_binary_sha256"))
    linkage_sha = str(runtime.get("copied_linkage_sha256"))
    if runtime.get("binary_copy_exact") is not True \
            or runtime.get("production_source_commit") != PRODUCTION_COMMIT \
            or runtime.get("measurement_instrument_commit") != INSTRUMENT_COMMIT:
        raise ValueError("runtime source label is not the declared v9 instrument")
    copied_binary = output_root / "anchor_binary_copy" / "llama-bench"
    if _sha256_file(copied_binary) != copy_sha \
            or _sha256_file(INSTRUMENT_BINARY) != runtime.get("measurement_binary_sha256"):
        raise ValueError("measurement binary or its evidence copy changed after capture")
    expected_aa_binding = {
        "binary_path": str(copied_binary),
        "library_path": str(copied_binary.parent),
    }
    if runtime.get("aa_sealed_binding") != expected_aa_binding:
        raise ValueError("runtime source label does not declare the sealed A/A path")

    preflight = _load_json(output_root / "preflight.json")
    preflight_checks = preflight.get("checks")
    if not isinstance(preflight_checks, Mapping) or not preflight_checks \
            or any(not isinstance(check, Mapping)
                   or check.get("outcome") != schemas.PASS
                   for check in preflight_checks.values()):
        raise ValueError("recorded live-control preflight is not all-PASS")
    claim_payload = _load_json(output_root / "claim_receipt.json")
    claim_receipt = cpu_region_claim.RegionClaimReceipt.from_dict(claim_payload)
    if claim_receipt.campaign_id != identity.campaign_id \
            or claim_receipt.cpu_list != CPU_LIST or not claim_receipt.released_at:
        raise ValueError("recorded CPU claim is not released and bound to this campaign")

    host = _load_json(output_root / "host.json")
    legs = host.get("legs")
    if host.get("claim_receipt") != claim_payload or not isinstance(legs, list):
        raise ValueError("host receipt does not bind the terminal claim receipt")
    expected_labels = {
        "aa_calibration", "neutral_calibration", "positive",
        "historical_win_replay", "negative_committed_cell", "negative_wrong_cell",
    }
    by_label = {leg.get("label"): leg for leg in legs if isinstance(leg, Mapping)}
    if set(by_label) != expected_labels:
        raise ValueError("host receipt does not cover exactly the six declared legs")
    for label, leg in by_label.items():
        expected_ref = str(output_root / "raw" / f"{label}.json")
        if leg.get("complete") is not True or leg.get("refusals") != [] \
                or leg.get("raw_ref") != expected_ref:
            raise ValueError(f"host receipt leg {label!r} is incomplete or misbound")

    aa, aa_raw = _load_recorded_material(
        output_root, identity=identity, label="aa_calibration",
        expected_blocks=CALIBRATION_BLOCKS, prompt=PROMPT_TOKENS,
        candidate_iqk=CALIBRATION_IQK, anchor_iqk=CALIBRATION_IQK)
    neutral, neutral_raw = _load_recorded_material(
        output_root, identity=identity, label="neutral_calibration",
        expected_blocks=NEUTRAL_BLOCKS, prompt=PROMPT_TOKENS,
        candidate_iqk=CALIBRATION_IQK, anchor_iqk=CALIBRATION_IQK)
    for label, raw in (("aa_calibration", aa_raw),
                       ("neutral_calibration", neutral_raw)):
        for arm in ("candidate_receipt", "anchor_receipt"):
            receipt = raw.get(arm)
            actual = ({key: receipt.get(key) for key in expected_aa_binding}
                      if isinstance(receipt, Mapping) else None)
            if actual != expected_aa_binding:
                raise ValueError(
                    f"{label} {arm} is not bound to the declared sealed A/A path")
    declared, rule, construction, split, solve = _campaign_inputs(
        aa, neutral, identity)
    stored_calibration = _load_json(output_root / "calibration.json")
    if schemas.content_hash(stored_calibration) != schemas.content_hash(solve.to_dict()):
        raise ValueError("stored calibration does not re-derive from the raw A/A pools")
    control_blocks = rule.max_total_blocks(solve.require_accepted().b_min_blocks)
    configs = (
        ("positive", PROMPT_TOKENS, *CONTROL_ARM_IQK["positive"]),
        ("historical_win_replay", PROMPT_TOKENS,
         *CONTROL_ARM_IQK["historical_win_replay"]),
        ("negative_committed_cell", PROMPT_TOKENS,
         *CONTROL_ARM_IQK["negative_committed_cell"]),
        ("negative_wrong_cell", WRONG_PROMPT_TOKENS,
         *CONTROL_ARM_IQK["negative_wrong_cell"]),
    )
    materials = {aa.label: aa, neutral.label: neutral}
    raw_by_label = {aa.label: aa_raw, neutral.label: neutral_raw}
    for label, prompt, candidate_iqk, anchor_iqk in configs:
        material, raw = _load_recorded_material(
            output_root, identity=identity, label=label,
            expected_blocks=control_blocks, prompt=prompt,
            candidate_iqk=candidate_iqk, anchor_iqk=anchor_iqk)
        materials[label] = material
        raw_by_label[label] = raw

    anchor_payload = raw_by_label["positive"].get("anchor_identity")
    if not isinstance(anchor_payload, Mapping):
        raise ValueError("positive raw vector carries no anchor identity")
    if any(raw.get("anchor_identity") != anchor_payload
           for raw in raw_by_label.values()):
        raise ValueError("raw vectors do not share one anchor identity")
    anchor = api.AnchorIdentity(
        source_commit=anchor_payload["source_commit"],
        binary_sha256=anchor_payload["binary_sha256"],
        linkage_sha256=anchor_payload["linkage_sha256"],
        measurement_event_ids=tuple(anchor_payload["measurement_event_ids"]),
        tool=anchor_payload.get("tool"))
    if anchor.source_commit != INSTRUMENT_COMMIT \
            or anchor.binary_sha256 != runtime.get("measurement_binary_sha256"):
        raise ValueError("raw-vector anchor is not the declared hardened instrument")
    candidate_receipt = raw_by_label["positive"].get("candidate_receipt")
    if not isinstance(candidate_receipt, Mapping):
        raise ValueError("positive raw vector carries no candidate recipe receipt")
    recipe_receipt = api.RecipeReceipt(
        constructor_id=candidate_receipt["constructor_id"],
        constructor_sha256=candidate_receipt["constructor_sha256"],
        argv_sha256=candidate_receipt["argv_sha256"])
    measured_at = materials["negative_wrong_cell"].run.ended_at
    result = _compose_measured_controls(
        output_root=output_root, identity=identity, solve=solve,
        declared=declared, rule=rule, construction=construction, split=split,
        materials=materials, anchor=anchor, recipe_receipt=recipe_receipt,
        claim_receipt=claim_receipt, source_sha=str(source_sha),
        binary_sha=copy_sha, linkage_sha=linkage_sha,
        measured_at=measured_at)

    input_paths = (
        "campaign_declaration.json", "runtime-source-label.json", "preflight.json",
        "claim_receipt.json", "host.json", "calibration.json",
        *(f"raw/{label}.json" for label in sorted(expected_labels)),
    )
    evaluator_commit = subprocess.run(
        ("git", "-C", str(Path(__file__).resolve().parents[4]), "rev-parse", "HEAD"),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True).stdout.strip()
    composition = {
        "schema": "epyc.autokernel.control_composition_attestation.v1",
        "campaign_id": identity.campaign_id,
        "composed_at": _utc_now(),
        "mode": "existing_completed_raw_vectors",
        "inference_executed": False,
        "evaluator_commit": evaluator_commit,
        "evaluator_source_sha256": _sha256_file(Path(__file__)),
        "inputs": {name: _sha256_file(output_root / name) for name in input_paths},
    }
    belief_receipt = _build_belief_receipt(
        output_root, identity=identity, result=result,
        claim_receipt=claim_receipt, measured_at=measured_at)
    _write_json(output_root / "composition_attestation.json", composition)
    _write_json(output_root / "control_sweep.json", result.to_dict())
    if belief_receipt is not None:
        _write_json(output_root / "belief_receipt.json", belief_receipt)
    summary = {
        "campaign_id": identity.campaign_id, "measured_at": _utc_now(),
        "state": "controls_complete", "controls_started": True,
        "production_source_commit": PRODUCTION_COMMIT,
        "measurement_instrument_commit": INSTRUMENT_COMMIT,
        "measurement_binary_sha256": runtime.get("measurement_binary_sha256"),
        "copied_binary_sha256": copy_sha,
        "binary_copy_exact": True,
        "composition_mode": composition["mode"],
        "composition_attestation_sha256": schemas.content_hash(composition),
        "calibration": solve.to_dict(), "controls": result.to_dict(),
        "may_rank": result.may_rank,
        "belief_receipt_sha256": (
            belief_receipt["receipt_sha256"] if belief_receipt else None),
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-id", required=True,
        help="fresh ak-* identifier committed into every control receipt")
    parser.add_argument(
        "--output", type=Path, required=True,
        help="fresh absolute evidence directory; execution refuses reuse")
    parser.add_argument(
        "--recipe", choices=(PREFILL_RECIPE_ID, DECODE_RECIPE_ID),
        default=PREFILL_RECIPE_ID,
        help="recipe-local control cell; prefill remains the default")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument(
        "--resume-existing", action="store_true",
        help="strictly resume one completed AA leg from a provably dead claimant")
    mode.add_argument(
        "--evaluate-existing", action="store_true",
        help="compose already-completed receipt-bound raw vectors; runs no inference")
    parser.add_argument("--i-hold-the-host", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_recipe(args.recipe)
    plan = {
        "campaign_id": args.campaign_id, "cpu_list": CPU_LIST,
        "production_source": f"{PRODUCTION_ROOT}@{PRODUCTION_COMMIT}",
        "measurement_binary": str(INSTRUMENT_BINARY),
        "measurement_instrument_commit": INSTRUMENT_COMMIT,
        "recipe_id": RECIPE_ID,
        "model": str(MODEL),
        "calibration_blocks": CALIBRATION_BLOCKS,
        "neutral_blocks": NEUTRAL_BLOCKS,
        "calibration_frame": _calibration_frame(),
        "calibration_fresh_invocations": (
            2 * _fresh_pairs_per_block()
            * (CALIBRATION_BLOCKS + NEUTRAL_BLOCKS)),
        "contribution_floor": CONTRIBUTION_FLOOR,
        "output": str(args.output), "execute": args.execute,
        "resume_existing": args.resume_existing,
        "evaluate_existing": args.evaluate_existing,
    }
    if not args.execute and not args.resume_existing and not args.evaluate_existing:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if (args.execute or args.resume_existing) and not args.i_hold_the_host:
        raise SystemExit("--execute/--resume-existing requires --i-hold-the-host")
    if args.evaluate_existing:
        summary = evaluate_existing(
            args.output.resolve(), campaign_id=args.campaign_id)
    else:
        with powercap_broker.PowercapBroker() as broker:
            summary = execute(
                args.output.resolve(), campaign_id=args.campaign_id,
                resume_existing=args.resume_existing,
                host_state=broker.read_host_state)
    print(json.dumps({
        "campaign_id": summary["campaign_id"],
        "calibration_accepted": summary["calibration"]["accepted"],
        "may_rank": summary["may_rank"],
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0 if summary["calibration"]["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
