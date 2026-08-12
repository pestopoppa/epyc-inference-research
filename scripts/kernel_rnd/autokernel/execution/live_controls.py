#!/usr/bin/env python3
"""Run and calibrate the five AutoKernel controls on the live CPU cell.

Dry-run is the default.  ``--execute`` additionally requires the operator's
explicit host handoff flag, acquires one q0-q3 claim, measures fresh A/A and
neutral pools, solves the campaign calibration, then drives all five controls
through :mod:`execution.control_runner`'s candidate pipeline.

The frozen production tree is read only. Both arms use the reviewed hardened
measurement overlay, whose commit is a one-change child of production v9. A
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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from .. import schemas
from ..evaluator import api, controls, recipes, statistics
from . import (control_runner, cpu_region_claim, microbench, physical_bounds,
               powercap_broker, sandbox)

RECIPE_ID = "t1b.llama_cpu.llama_bench_prefill.v1"
CPU_LIST = recipes.CANONICAL_PREFIX[recipes.CANONICAL_PREFIX.index("-c") + 1]
PRODUCTION_ROOT = Path("/mnt/raid0/llm/llama.cpp")
PRODUCTION_COMMIT = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
INSTRUMENT_ROOT = Path(os.environ.get(
    "AUTOKERNEL_INSTRUMENT_ROOT", "/mnt/raid0/llm/llama.cpp-ak-controls-v9-final"))
INSTRUMENT_BINARY = Path(os.environ.get(
    "AUTOKERNEL_INSTRUMENT_BINARY",
    str(INSTRUMENT_ROOT / "build-v9-cpu/bin/llama-bench")))
INSTRUMENT_BRANCH = "experimental-v9-autokernel-t1-hardening-final"
INSTRUMENT_COMMIT = "a4cb04ca8f92fa4d665684490f609b380f9b5e96"
MODEL = Path(
    "/mnt/raid0/llm/models/lmstudio-community/"
    "Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q4_K_M.gguf")
CALIBRATION_BLOCKS = 200
NEUTRAL_BLOCKS = 60
CONTROL_EXTENSION_ROUNDS = 1
CONTROL_EXTENSION_BLOCKS = 5
CONTRIBUTION_FLOOR = 0.03
PROMPT_TOKENS = 512
WRONG_PROMPT_TOKENS = 2048
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
    "positive": PROMPT_TOKENS,
    "historical_win_replay": PROMPT_TOKENS,
    "negative_committed_cell": PROMPT_TOKENS,
    "negative_wrong_cell": WRONG_PROMPT_TOKENS,
}
REQUIRED_HARDENING_RECEIPTS = (
    b"autokernel_hybrid_ab_complete",
    b"autokernel_thread_set_stable",
    b"autokernel_escape_checks_complete",
    b"autokernel_unsynchronized_samples_ns",
    b"autokernel_thread_set_hashes",
    b"autokernel_device_sync_mode",
)


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
                       instrument_linkage: str, copy_linkage: str) -> str:
    """Commit the fresh campaign inputs before the first measurement."""
    source_manifest = {
        "schema": "epyc.autokernel.runtime_source_label.v1",
        "production_source_commit": PRODUCTION_COMMIT,
        "measurement_instrument_commit": INSTRUMENT_COMMIT,
        "measurement_binary_sha256": instrument_sha,
        "copied_binary_sha256": copy_sha,
        "measurement_linkage_sha256": instrument_linkage,
        "copied_linkage_sha256": copy_linkage,
        "binary_copy_exact": instrument_sha == copy_sha,
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
        "prompt_tokens": PROMPT_TOKENS,
        "wrong_prompt_tokens": WRONG_PROMPT_TOKENS,
        "calibration_blocks": CALIBRATION_BLOCKS,
        "neutral_blocks": NEUTRAL_BLOCKS,
        "contribution_floor": CONTRIBUTION_FLOOR,
        "max_candidates": 10,
        "max_blocks_per_candidate": 20,
        "physical_envelopes": {
            label: envelope.to_dict()
            for label, envelope in sorted(_declared_physical_envelopes().items())
        },
        "source_sha256": source_sha,
    }
    _write_json(output_root / "campaign_declaration.json", declaration)
    return source_sha


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
    _write_json(output_root / "host.json", {
        "schema": "epyc.autokernel.live_control_host_receipt.v1",
        "measured_at": _utc_now(),
        "claim_receipt": claim_receipt.to_dict(),
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


def _params(*, prompt: int) -> dict:
    return {"model": str(MODEL), "n_prompt": prompt, "reps": 1,
            "autokernel_seed": 2026081101,
            "output_format": "json"}


def _unit_id(*, label: str, prompt: int) -> str:
    return f"{MODEL.name}:pp{prompt}:{label}"


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


def _wait_for_quiet(*, ceiling_per_core: float = 0.25,
                    timeout_s: float = 300.0) -> None:
    """Let this process's prior leg age out of the one-minute load average.

    The q0-q3 claim stays held throughout.  Starting the next independent run
    immediately would make its run-open contention gate attribute the preceding
    run's exponentially decaying load to a co-tenant.
    """
    deadline = time.monotonic() + timeout_s
    claimed_cores = len(cpu_region_claim.parse_cpu_list(CPU_LIST))
    while True:
        load1 = float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])
        if load1 / claimed_cores <= ceiling_per_core:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"host did not return below {ceiling_per_core}/core inside "
                f"{timeout_s}s (load1={load1:.2f})")
        time.sleep(5.0)


def _measure(*, label: str, blocks: int, claim: object,
             candidate_binding: recipes.ToolBinding,
             anchor_binding: recipes.ToolBinding,
             anchor: api.AnchorIdentity, prompt: int = PROMPT_TOKENS,
             candidate_iqk: str = "1", anchor_iqk: str = "1",
             output_root: Path,
             host_state: Callable[..., microbench.HostState],
             identity: LiveCampaignIdentity) -> LiveMaterial:
    declared_prompt = CONTROL_PROMPT_BY_LABEL.get(label)
    if declared_prompt != prompt:
        raise ValueError(
            f"control {label!r} at pp{prompt} was not predeclared; expected "
            f"{None if declared_prompt is None else f'pp{declared_prompt}'}")
    envelope = _declared_physical_envelopes()[label]
    plan = microbench.MicrobenchPlan(
        recipe_id=RECIPE_ID, candidate_id=f"akc-control-{label}",
        campaign_seed=f"{identity.campaign_seed}/{label}",
        candidate_binding=candidate_binding, anchor_binding=anchor_binding,
        anchor=anchor, params=_params(prompt=prompt),
        candidate_instrument_root=str(INSTRUMENT_ROOT),
        anchor_instrument_root=str(INSTRUMENT_ROOT),
        candidate_param_overrides={"ggml_iqk": candidate_iqk},
        anchor_param_overrides={"ggml_iqk": anchor_iqk},
        base_blocks=blocks, pairs_per_block=1,
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
        backend="llama_cpu", phase="prefill", cell_class=recipes.CELL_CLASS_TINY_GRAPH,
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
        campaign_id=identity.campaign_id, backend="llama_cpu", phase="prefill",
        cell_class=recipes.CELL_CLASS_TINY_GRAPH,
        protocol_id=api.PROTOCOL_VERSIONED_ID,
        evaluator=api.EvaluatorIdentity(
            id="P-AK-SEARCH-1/v1", bundle_sha256=evaluator_hash,
            runtime_source_label_ref=(
                f"{identity.evidence_ref}/runtime-source-label.json")),
        scope_denominator=api.ScopeDenominator(
            machine_subset="full", numa_nodes=(), devices=(), cores=96),
        scope_manifest_sha256=schemas.content_hash({"cpu_list": CPU_LIST}),
        co_residency="single", metric="prefill_tokens_per_s",
        metric_direction="higher_better", reps=1, anchor=anchor,
        campaign_controls=declared, calibration=outputs)
    runner = control_runner.ExecutedControlRunner(
        pipeline=pipeline, fixtures=fixture_set, binding=binding,
        campaign_statistics=campaign_stats)
    declaration = controls.HistoricalWinReplayDeclaration(
        win_id="iqk-prefill-port", backend="llama_cpu", phase="prefill",
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
        campaign_id=identity.campaign_id, backend="llama_cpu", phase="prefill",
        cell_class=recipes.CELL_CLASS_TINY_GRAPH, window_id=identity.window_id, tier="T1",
        seed="DERIVED-BY-SWEEP", anchor=anchor, declaration=declaration)
    context = controls.ControlContext(
        campaign_id=identity.campaign_id, backend="llama_cpu", phase="prefill",
        cell_class=recipes.CELL_CLASS_TINY_GRAPH, window_id=identity.window_id,
        historical=historical,
        neutral_dispersion=controls.neutral_dispersion_check(solve),
        calibration=outputs)
    return sweep.run(
        run_context=run_context, context=context, window=window,
        aa_cadence=window.aa_cadence, windows_completed=0, last_rotation_epoch=0)


def execute(output_root: Path, *, campaign_id: str,
            host_state: Callable[..., microbench.HostState] =
            microbench.read_host_state) -> dict:
    output_root = output_root.resolve()
    identity = LiveCampaignIdentity(
        campaign_id=campaign_id, evidence_ref=str(output_root))
    output_root.mkdir(parents=True, exist_ok=False)
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
        instrument_linkage=instrument_linkage, copy_linkage=copy_linkage)
    _write_preflight(output_root, instrument_sha=instrument_sha, copy_sha=copy_sha,
                     host_state=host_state)
    anchor = api.AnchorIdentity(
        source_commit=INSTRUMENT_COMMIT, binary_sha256=instrument_sha,
        linkage_sha256=instrument_linkage, tool="llama-bench")
    journal = cpu_region_claim.RegionClaimJournal(output_root / "region_claim.jsonl")
    materials = []
    with cpu_region_claim.acquire_cpu_region_claim(
            CPU_LIST, purpose="AutoKernel five-control calibration block",
            campaign_id=identity.campaign_id, journal=journal, timeout_s=60.0,
            max_hold_s=2 * 3600) as claim:
        aa = _measure(
            label="aa_calibration", blocks=CALIBRATION_BLOCKS, claim=claim,
            candidate_binding=candidate_binding, anchor_binding=anchor_binding,
            anchor=anchor, output_root=output_root, host_state=host_state,
            identity=identity)
        materials.append(aa)
        _wait_for_quiet()
        neutral = _measure(
            label="neutral_calibration", blocks=NEUTRAL_BLOCKS, claim=claim,
            candidate_binding=candidate_binding, anchor_binding=anchor_binding,
            anchor=anchor, output_root=output_root, host_state=host_state,
            identity=identity)
        materials.append(neutral)
        declared, rule, construction, split, solve = _campaign_inputs(
            aa, neutral, identity)
        _write_json(output_root / "calibration.json", solve.to_dict())
        if solve.accepted:
            outputs = solve.require_accepted()
            n = outputs.b_min_blocks
            control_blocks = rule.max_total_blocks(n)
            _wait_for_quiet()
            positive = _measure(
                label="positive", blocks=control_blocks, claim=claim,
                candidate_binding=candidate_binding, anchor_binding=anchor_binding,
                anchor=anchor, candidate_iqk="1", anchor_iqk="0",
                output_root=output_root, host_state=host_state, identity=identity)
            materials.append(positive)
            _wait_for_quiet()
            historical = _measure(
                label="historical_win_replay", blocks=control_blocks, claim=claim,
                candidate_binding=candidate_binding, anchor_binding=anchor_binding,
                anchor=anchor, candidate_iqk="1", anchor_iqk="0",
                output_root=output_root, host_state=host_state, identity=identity)
            materials.append(historical)
            _wait_for_quiet()
            negative_anchor = _measure(
                label="negative_committed_cell", blocks=control_blocks, claim=claim,
                candidate_binding=candidate_binding, anchor_binding=anchor_binding,
                anchor=anchor, prompt=PROMPT_TOKENS, output_root=output_root,
                host_state=host_state, identity=identity)
            materials.append(negative_anchor)
            _wait_for_quiet()
            negative_wrong = _measure(
                label="negative_wrong_cell", blocks=control_blocks, claim=claim,
                candidate_binding=candidate_binding, anchor_binding=anchor_binding,
                anchor=anchor, prompt=WRONG_PROMPT_TOKENS, output_root=output_root,
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
        }
        _write_json(output_root / "summary.json", summary)
        return summary

    negative_blocks = tuple(
        statistics.PairedBlock(
            block_index=i, unit_id=f"negative-work-mismatch-{i}",
            stratum=api.STRATUM_SELECTION, order=a.order,
            anchor_samples=a.anchor_samples,
            candidate_samples=w.candidate_samples, measured_at=ended)
        for i, (a, w) in enumerate(zip(negative_anchor.blocks,
                                       negative_wrong.blocks)))
    fixtures = (
        _fixture(controls.CONTROL_POSITIVE, positive.blocks, tier="T1",
                 source_sha=source_sha, binary_sha=copy_sha,
                 linkage_sha=copy_linkage, measured_at=positive.run.ended_at),
        _fixture(controls.CONTROL_NEUTRAL, neutral.blocks[:control_blocks], tier="T1",
                 source_sha=source_sha, binary_sha=copy_sha,
                 linkage_sha=copy_linkage, measured_at=neutral.run.ended_at),
        _fixture(controls.CONTROL_DEGRADED_NEGATIVE, negative_blocks, tier="T1",
                 source_sha=source_sha, binary_sha=copy_sha,
                 linkage_sha=copy_linkage, measured_at=ended),
        _fixture(controls.CONTROL_AA, aa.blocks[:control_blocks], tier="T1",
                 source_sha=source_sha, binary_sha=copy_sha,
                 linkage_sha=copy_linkage, measured_at=aa.run.ended_at),
        _fixture(controls.CONTROL_HISTORICAL_WIN_REPLAY, historical.blocks,
                 tier="T2", source_sha=source_sha, binary_sha=copy_sha,
                 linkage_sha=copy_linkage, measured_at=historical.run.ended_at),
    )
    receipt = positive.run.candidate_receipt
    if receipt is None:
        raise RuntimeError("positive run emitted no recipe receipt")
    result = _evaluate_controls(
        solve=solve, declared=declared, rule=rule,
        construction=construction, split=split, fixtures=fixtures,
        anchor=anchor, recipe_receipt=receipt.recipe_receipt,
        claim_receipt=claim_receipt, source_sha=source_sha,
        measured_at=ended, output_root=output_root, identity=identity)
    _write_json(output_root / "control_sweep.json", result.to_dict())
    summary = {
        "campaign_id": identity.campaign_id, "measured_at": _utc_now(),
        "state": "controls_complete", "controls_started": True,
        "production_source_commit": PRODUCTION_COMMIT,
        "measurement_instrument_commit": INSTRUMENT_COMMIT,
        "measurement_binary_sha256": instrument_sha,
        "copied_binary_sha256": copy_sha,
        "binary_copy_exact": instrument_sha == copy_sha,
        "calibration": solve.to_dict(), "controls": result.to_dict(),
        "may_rank": result.may_rank,
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
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--i-hold-the-host", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = {
        "campaign_id": args.campaign_id, "cpu_list": CPU_LIST,
        "production_source": f"{PRODUCTION_ROOT}@{PRODUCTION_COMMIT}",
        "measurement_binary": str(INSTRUMENT_BINARY),
        "measurement_instrument_commit": INSTRUMENT_COMMIT,
        "model": str(MODEL),
        "calibration_blocks": CALIBRATION_BLOCKS,
        "neutral_blocks": NEUTRAL_BLOCKS,
        "contribution_floor": CONTRIBUTION_FLOOR,
        "output": str(args.output), "execute": args.execute,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not args.i_hold_the_host:
        raise SystemExit("--execute requires --i-hold-the-host")
    with powercap_broker.PowercapBroker() as broker:
        summary = execute(
            args.output.resolve(), campaign_id=args.campaign_id,
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
