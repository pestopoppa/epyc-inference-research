"""Bind an enrolled target to the existing loop's single-target inputs.

Selection is not artifact verification, resource admission or a serving verdict.
The explicit experimental worktree/build still go through run.py's original owners.
In particular, a GPU selection is a legacy bench screen, not execution of the
enrolled production server command. No model bytes are read here.
"""
from __future__ import annotations

from pathlib import Path

from . import campaign
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
    """Join selected model/workload facts without confusing production and test builds.

The enrolled build/recipe/baseline remain the original declaration. The separately
supplied canonical launch identifies the actual experimental instrument; run.py
checks its anchor build, original requests and startup provenance without a waiver.
"""
    execution = target.execution
    if execution.backend != "cpu" or launch.backend != "cpu":
        raise TargetSelectionRefused("CPU workload binding requires two CPU inputs")
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
        "environment": {key: value for key, value in launch.launch_env
                        if key != "LD_LIBRARY_PATH"} == dict(execution.env),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise TargetSelectionRefused(f"CPU launch differs from selected target: {failed}")
