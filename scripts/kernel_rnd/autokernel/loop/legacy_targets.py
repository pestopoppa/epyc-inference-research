"""Bind an enrolled target to the existing loop's single-target inputs.

Selection is not artifact verification, resource admission or a serving verdict.
The explicit experimental worktree/build still go through run.py's original owners.
A GPU selection alone is a legacy bench screen; explicit serving launch inputs
are separately joined below. No model bytes are read here.
"""
from __future__ import annotations

from pathlib import Path

from ..execution.cpu_region_claim import parse_cpu_list, render_cpu_list
from . import campaign
from . import claim
from .resolved_recipe import CanonicalResolvedRecipe


class TargetSelectionRefused(ValueError):
    pass


def select_target(resolved: campaign.ResolvedCampaign, target_id: str, *,
                  cpu_serving: bool, model: Path | None = None) -> campaign.TargetRevision:
    """Select one original alias, refusing ambiguous, unavailable or wrong-mode rows."""
    # Reparse the original closed value even when the caller supplied a dataclass.
    resolved = campaign.ResolvedCampaign.from_dict(resolved.to_dict())
    matches = [row for row in resolved.targets if target_id in row.target_ids]
    if len(matches) != 1:
        raise TargetSelectionRefused(
            f"target {target_id!r} is {'unknown' if not matches else 'ambiguous'}")
    selected = matches[0]
    if selected.status != "ready":
        raise TargetSelectionRefused(
            f"target {target_id!r} is {selected.status}: {', '.join(selected.missing)}")
    expected_backend = "cpu" if cpu_serving else "gpu"
    if selected.execution.backend != expected_backend:
        raise TargetSelectionRefused(
            f"target backend {selected.execution.backend!r} differs from "
            f"{'--cpu-serving-launch' if cpu_serving else 'legacy GPU screen'}; "
            "select an explicit single-backend target")
    assert selected.execution.model is not None  # Ready is checked by the owning parser.
    if model is not None and model.resolve() != Path(selected.execution.model.path).resolve():
        raise TargetSelectionRefused("--model differs from the selected target model path")
    return selected


def validate_cpu_workload(target: campaign.TargetRevision,
                          launch: CanonicalResolvedRecipe) -> None:
    """Compatibility entrypoint for the original CPU-only connector."""
    if target.execution.backend != "cpu" or launch.backend != "cpu":
        raise TargetSelectionRefused("CPU workload binding requires two CPU inputs")
    validate_serving_workload(target, launch)


def validate_serving_workload(target: campaign.TargetRevision,
                              launch: CanonicalResolvedRecipe) -> None:
    """Join selected model/workload facts without confusing production and test builds.

The enrolled build/recipe/baseline remain the original declaration. The separately
supplied canonical launch identifies the actual experimental instrument; run.py
checks its anchor build, original requests and startup provenance without a waiver.
"""
    execution = target.execution
    if execution.backend not in ("cpu", "gpu") or execution.backend != launch.backend:
        raise TargetSelectionRefused("serving workload binding requires the selected backend")
    checks = {
        "model": (execution.model is not None
                  and (launch.model.path, launch.model.sha256)
                  == (execution.model.path, execution.model.sha256)),
        "drafter": ((launch.drafter is None and execution.drafter is None)
                    or (launch.drafter is not None and execution.drafter is not None
                        and (launch.drafter.path, launch.drafter.sha256)
                        == (execution.drafter.path, execution.drafter.sha256))),
        "context": launch.template.ctx == execution.context,
        "concurrency": launch.template.np == execution.concurrency,
        "speculation": launch.capability.speculation == execution.speculation,
        "metric": execution.metric == launch.template.metric == "aggregate_tok_s",
        "metric_direction": execution.metric_direction == "higher",
        "environment": {key: value for key, value in launch.launch_env
                        if key != "LD_LIBRARY_PATH"} == dict(execution.env),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise TargetSelectionRefused(f"{launch.backend.upper()} launch differs from selected target: {failed}")


def validate_resources(resources: campaign.ResourceRequest, launch: CanonicalResolvedRecipe | None,
                       *, backend: str, environment=None) -> str:
    """Join the installed single-GPU/CPU owners to explicitly declared resources.

    This describes what the existing owners can acquire; it is not a receipt.
    ROCm0/mi210_0 name the unchanged installed loop route, not a fresh hardware readback.
    """
    cpus = frozenset(resources.cpu_logical)
    if not cpus:
        raise TargetSelectionRefused("selected loop requires declared host CPU resources")
    if launch is not None and launch.template.cpu_list:
        if not parse_cpu_list(launch.template.cpu_list).issubset(cpus):
            raise TargetSelectionRefused("serving affinity exceeds declared resources.cpu_logical")
    if backend == "gpu":
        if not {"ROCm0", claim.DEVICE_ID}.intersection(resources.gpu_ids):
            raise TargetSelectionRefused("GPU loop requires the declared installed ROCm0/mi210_0 route")
        if launch is not None and launch.template.device != "ROCm0":
            raise TargetSelectionRefused("installed GPU oracle/claim supports only the original ROCm0 route")
        for env in (dict(environment or {}), dict(launch.launch_env) if launch else {}):
            if any(key in env and env[key] != "0" for key in (
                    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                    "GPU_DEVICE_ORDINAL")):
                raise TargetSelectionRefused("GPU visibility remapping has no installed physical claim join")
    return render_cpu_list(cpus)
