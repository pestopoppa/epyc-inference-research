"""Immutable amortized baseline bank for non-promotable discovery screens.

The bank deliberately carries only an exact-frame anchor vector.  It is never
accepted by strict T1 and cannot be converted into a candidate/archive record.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .. import schemas
from ..evaluator import recipes
from . import microbench
from ..resource import device_claim, preflight

SCHEMA = "epyc.autokernel.screening_baseline_bank.v3"
GPU_DISCOVERY_LIVE_SCHEMA = "epyc.autokernel.gpu_discovery_live_governance.v1"
GPU_DISCOVERY_SCRIPT = "run_autokernel_gpu_discovery.py"
GPU_DISCOVERY_DEVICE = "mi210_0"
GPU_DISCOVERY_MODEL_LIMIT_BYTES = 512 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BaselineBankError(ValueError):
    pass


@dataclass(frozen=True)
class BaselineBank:
    frame: Mapping[str, Any]
    anchor_samples: tuple[float, ...]
    sentinel_before: float
    anchor_command: Mapping[str, Any]
    anchor_artifacts: Mapping[str, Any]
    sentinel_after: float | None = None

    def to_dict(self) -> dict[str, Any]:
        body = {"schema": SCHEMA, "frame": dict(self.frame),
                "anchor_samples": list(self.anchor_samples),
                "anchor_command": dict(self.anchor_command),
                "anchor_artifacts": dict(self.anchor_artifacts),
                "sentinel_before": self.sentinel_before,
                "sentinel_after": self.sentinel_after}
        return {**body, "baseline_sha256": schemas.content_hash(body)}

    def admit(self, frame: Mapping[str, Any]) -> None:
        if dict(frame) != dict(self.frame):
            raise BaselineBankError("screening baseline frame differs from candidate frame")
        if self.sentinel_after is not None:
            raise BaselineBankError("screening baseline is closed; create a fresh bank")

    def nominate(self, candidate_samples: tuple[float, ...]) -> dict[str, Any]:
        """Noise-tolerant directional summary, never a pass/fail decision."""
        if not candidate_samples:
            raise BaselineBankError("screening candidate has no samples")
        center = sum(self.anchor_samples) / len(self.anchor_samples)
        values = tuple((x - center) / center for x in candidate_samples)
        return {"baseline_center": center, "candidate_samples": list(candidate_samples),
                "relative_effects": list(values),
                "median_relative": sorted(values)[len(values) // 2],
                "uncertainty": "screening_noise_unquantified_nonpromotable",
                "nomination": "top_k_candidate_only_not_a_keep"}


def load(path: str | Path) -> BaselineBank:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise BaselineBankError("baseline bank must be an object")
    body = {key: raw.get(key) for key in ("schema", "frame", "anchor_samples", "anchor_command", "anchor_artifacts",
                                          "sentinel_before", "sentinel_after")}
    if raw.get("baseline_sha256") != schemas.content_hash(body) or body["schema"] != SCHEMA:
        raise BaselineBankError("baseline bank schema/hash is invalid")
    values = body["anchor_samples"]
    command = body["anchor_command"]
    artifacts = body["anchor_artifacts"]
    if not isinstance(body["frame"], Mapping) or not isinstance(values, list) or len(values) != 3:
        raise BaselineBankError("baseline bank needs exact frame and exactly three anchor samples")
    if body["frame"].get("anchor_ggml_iqk") != "0" \
            or not isinstance(command, Mapping) \
            or command.get("arm") != "anchor" \
            or command.get("env", {}).get("GGML_IQK") != "0" \
            or command.get("params", {}).get("ggml_iqk") != "0" \
            or not isinstance(artifacts, Mapping):
        raise BaselineBankError(
            "baseline bank must seal an anchor command with GGML_IQK=0")
    return BaselineBank(dict(body["frame"]), tuple(float(x) for x in values),
                        float(body["sentinel_before"]),
                        dict(command),
                        dict(artifacts),
                        None if body["sentinel_after"] is None else float(body["sentinel_after"]))


def create(*, frame: Mapping[str, Any], anchor_command: Mapping[str, Any],
           invoke_anchor, anchor_count: int = 3) -> BaselineBank:
    """Seal O(1) anchor invocations once for a whole discovery batch."""
    if anchor_count != 3:
        raise BaselineBankError("baseline bank requires exactly three anchor invocations")
    samples = tuple(float(invoke_anchor()) for _ in range(anchor_count))
    if frame.get("anchor_ggml_iqk") != "0" \
            or anchor_command.get("env", {}).get("GGML_IQK") != "0" \
            or anchor_command.get("params", {}).get("ggml_iqk") != "0":
        raise BaselineBankError("baseline creation requires a bound GGML_IQK=0 anchor")
    return BaselineBank(dict(frame), samples, samples[-1], dict(anchor_command),
                        command_artifacts(anchor_command))


def command_artifacts(command: Mapping[str, Any]) -> dict[str, Any]:
    binding = command.get("binding", {})
    binary = Path(str(binding.get("binary", "")))
    library_root = Path(str(binding.get("library_path", "")))
    if not binary.is_file() or not library_root.is_dir():
        raise BaselineBankError("screening command artifact paths are unavailable")
    libraries = {}
    for path in sorted(library_root.glob("*.so*")):
        if path.is_file():
            libraries[path.name] = _sha256_file(path)
    return {"binary_sha256": _sha256_file(binary), "libraries": libraries}


def _semantic_command(command: Mapping[str, Any]) -> dict[str, Any]:
    env = dict(command.get("env", {}))
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("GGML_IQK", None)
    params = dict(command.get("params", {}))
    params.pop("ggml_iqk", None)
    params.pop("autokernel_seed", None)
    recipe = command.get("recipe", {})
    return {key: command.get(key) for key in (
        "recipe_id", "registry_id", "backend", "phase", "cell_class",
        "metric", "metric_direction", "tool") } | {
            "recipe": {key: recipe.get(key) for key in (
                "constructor_id", "constructor_sha256")},
            "env": env, "params": params,
        }


def screen(*, bank: BaselineBank, frame: Mapping[str, Any], invoke_candidate,
           competing_inference: bool,
           candidate_command: Mapping[str, Any]) -> dict[str, Any]:
    """Three candidate-only calls; ordinary host load is intentionally not input.

    The caller must provide the claim-scoped competing-inference witness. That
    is the one discovery blocker; service/build/load noise is reflected in the
    uncertainty label, not converted into a false refusal.
    """
    bank.admit(frame)
    if competing_inference:
        raise BaselineBankError("competing model inference occupies claimed screening compute")
    if candidate_command.get("arm") != "candidate" \
            or candidate_command.get("env", {}).get("GGML_IQK") != "1" \
            or candidate_command.get("params", {}).get("ggml_iqk") != "1":
        raise BaselineBankError(
            "screening candidate command must seal candidate GGML_IQK=1")
    anchor_semantic = _semantic_command(bank.anchor_command)
    candidate_semantic = _semantic_command(candidate_command)
    candidate_artifacts = command_artifacts(candidate_command)
    if anchor_semantic != candidate_semantic \
            or dict(bank.anchor_artifacts) != candidate_artifacts:
        raise BaselineBankError(
            "screening arm commands differ beyond the sole intended GGML_IQK factor: "
            f"semantic_equal={anchor_semantic == candidate_semantic}, "
            f"artifact_equal={dict(bank.anchor_artifacts) == candidate_artifacts}")
    samples = tuple(float(invoke_candidate()) for _ in range(3))
    report = bank.nominate(samples)
    report.update({"candidate_invocations": 3, "anchor_invocations": 0,
                   "host_noise_policy": "recorded_not_blocking",
                   "candidate_command_sha256": schemas.content_hash(candidate_command),
                   "sole_intended_factor": {"name": "GGML_IQK",
                                             "anchor": "0", "candidate": "1"},
                   "non_promotable": True})
    return report


def invoke_command(*, command: recipes.ConstructedCommand, spawner: microbench.Spawner,
                   timeout_s: float = 300.0) -> float:
    """Run exactly one bound llama-bench command and reduce its own samples."""
    env = microbench.assemble_env(command.env).env
    spawned = spawner.run(command.argv, env, timeout_s=timeout_s)
    if spawned.timed_out or spawned.returncode != 0:
        raise BaselineBankError("screening invocation failed or timed out")
    rows = microbench.parse_llama_bench_json(spawned.stdout)
    if len(rows) != 1:
        raise BaselineBankError("screening invocation must emit exactly one result row")
    check = microbench.LlamaBenchExpectation.from_command(command).check_row(rows[0])
    if check.outcome != schemas.PASS:
        raise BaselineBankError("screening command/result frame mismatch: " + "; ".join(check.reasons))
    values = rows[0].metric_samples
    return sum(values) / len(values)


def _argv_value(argv: tuple | list, flag: str) -> str | None:
    for index, value in enumerate(argv):
        if value == flag and index + 1 < len(argv):
            return str(argv[index + 1])
        if isinstance(value, str) and value.startswith(flag + "="):
            return value[len(flag) + 1:]
    return None


def _gpu_discovery_noise_admission(observation) -> tuple[dict[str, Any] | None, str]:
    """Admit only a live, receipt-bound, small-model MI210 discovery child."""
    child_argv = tuple(observation.cmdline)
    if _argv_value(child_argv, "-ngl") != "99":
        return None, "inference child is not a fully-offloaded GPU discovery call"
    child_model = _argv_value(child_argv, "-m")
    if not child_model:
        return None, "inference child has no bound model path"
    try:
        child = preflight.describe_pid(observation.pid)
        parent_pid = child.get("ppid")
        if child.get("vanished") or child.get("unreadable") or not isinstance(parent_pid, int):
            return None, "inference child identity/parent is unavailable"
        if child.get("starttime_ticks") != observation.starttime_ticks:
            return None, "inference child PID identity changed after scan"
        runner = preflight.describe_pid(parent_pid)
        runner_argv = tuple(runner.get("cmdline") or ())
        if runner.get("vanished") or runner.get("unreadable"):
            return None, "GPU discovery runner identity is unavailable"
        if not any(Path(arg).name == GPU_DISCOVERY_SCRIPT for arg in runner_argv):
            return None, "inference child is not directly owned by the GPU discovery runner"
        output_value = _argv_value(runner_argv, "--output-dir")
        if not output_value:
            return None, "GPU discovery runner has no output directory"
        output_dir = Path(output_value).resolve()
        governance_path = output_dir / "live-governance.json"
        preflight_path = output_dir / "preflight.json"
        live = json.loads(governance_path.read_text(encoding="utf-8"))
        sealed = json.loads(preflight_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError, preflight.PreflightUnavailable) as exc:
        return None, f"GPU discovery governance is unavailable: {type(exc).__name__}: {exc}"
    if not isinstance(live, Mapping) or not isinstance(sealed, Mapping):
        return None, "GPU discovery governance/preflight is not an object"

    exact_live = {
        "schema": GPU_DISCOVERY_LIVE_SCHEMA,
        "status": "active",
        "runner_pid": parent_pid,
        "authority": "nonpromotable_candidate_only_discovery",
        "cpu_overlap_policy": "allowed_discovery_noise",
        "promotion_claim": False,
        "non_promotable": True,
    }
    for key, expected in exact_live.items():
        if live.get(key) != expected:
            return None, f"GPU discovery governance {key} is not {expected!r}"
    receipt = live.get("device_claim_open")
    if not isinstance(receipt, Mapping):
        return None, "GPU discovery governance lacks a device-claim receipt"
    if receipt.get("device_id") != GPU_DISCOVERY_DEVICE \
            or receipt.get("state") != "held" \
            or receipt.get("holder_pid") != parent_pid \
            or receipt.get("holder_start_ticks") != runner.get("starttime_ticks") \
            or receipt.get("campaign_id") != live.get("campaign_id") \
            or not str(receipt.get("purpose") or "").startswith(
                "AutoKernel GPU candidate-only discovery "):
        return None, "GPU discovery MI210 claim identity does not match its live runner"
    claim_check = device_claim.check_device_claim_held(receipt)
    if claim_check.outcome != schemas.PASS:
        return None, "GPU discovery MI210 claim is not actively held: " + "; ".join(claim_check.reasons)

    if live.get("preflight_sha256") != schemas.content_hash(sealed):
        return None, "GPU discovery preflight hash does not match live governance"
    if sealed.get("schema") != "epyc.autokernel.gpu_discovery_preflight.v1" \
            or sealed.get("inference_executed") is not False:
        return None, "GPU discovery preflight is not a sealed no-inference preflight"
    projection = (
        "campaign_id", "cpu_overlap_policy", "model", "model_sha256",
        "model_size_bytes", "small_model_overlap_max_bytes", "promotion_claim",
    )
    if any(live.get(key) != sealed.get(key) for key in projection):
        return None, "GPU discovery live governance differs from its sealed preflight"
    model_bytes = live.get("model_size_bytes")
    limit = live.get("small_model_overlap_max_bytes")
    if not isinstance(model_bytes, int) or isinstance(model_bytes, bool) \
            or not isinstance(limit, int) or isinstance(limit, bool) \
            or limit != GPU_DISCOVERY_MODEL_LIMIT_BYTES or model_bytes > limit:
        return None, "GPU discovery model exceeds or changes the 512 MiB overlap ceiling"
    if live.get("model") != child_model:
        return None, "inference child model differs from GPU discovery governance"
    try:
        if Path(child_model).stat().st_size != model_bytes:
            return None, "GPU discovery model size changed after preflight"
    except OSError as exc:
        return None, f"GPU discovery model is unavailable: {exc}"
    return ({
        "finding": observation.to_dict(),
        "classification": "allowed_discovery_noise",
        "governance_path": str(governance_path),
        "preflight_path": str(preflight_path),
        "campaign_id": live.get("campaign_id"),
        "device_id": GPU_DISCOVERY_DEVICE,
        "device_claim_id": receipt.get("claim_id"),
        "model_size_bytes": model_bytes,
        "promotion_claim": False,
    }, "")


def competing_inference_witness() -> dict[str, Any]:
    """Read only model-inference identities; ordinary CPU activity is excluded.

    A fully-offloaded MI210 discovery call may overlap only when its direct
    runner publishes the exact active, non-promotable small-model governance
    receipt.  Everything else remains competing inference.
    """
    try:
        owned = preflight.read_own_scope()
        scan = preflight.interim_process_scan(owned=owned)
    except preflight.PreflightUnavailable as exc:
        raise BaselineBankError("screening inference witness unavailable") from exc
    if scan.unreadable_pids:
        raise BaselineBankError("screening inference witness unreadable")
    findings = []
    allowed = []
    for item in scan.inference_like():
        admission, reason = _gpu_discovery_noise_admission(item)
        if admission is not None:
            allowed.append(admission)
        else:
            finding = item.to_dict()
            finding["overlap_admission_refusal"] = reason
            findings.append(finding)
    return {"basis": "interim_inference_executable_scan", "competing": bool(findings),
            "findings": findings, "allowed_discovery_noise": allowed,
            "ordinary_processes_ignored": True}
