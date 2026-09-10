"""Resolve owned launch inputs into the existing serial runner's ordinary argv.

Enrollment describes workloads, not ownership of editable source trees. Only the
missing local ownership/request facts live in the owner map; no launch, build,
model read, resource claim, branch creation or promotion occurs here.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import archive, champion, legacy_targets, planned_serving, resolved_recipe


def build_targets(resolved_path, owned_path, *, target_root, common_path=None):
    from . import campaign_cli, serial_run as sr

    resolved_path = Path(resolved_path).resolve()
    # Bound before the existing owning parser reads the same source document.
    sr._read(resolved_path)
    resolved = campaign_cli.load_previous(resolved_path)
    owners, _ = sr._json(Path(owned_path), limit=64 * 1024)
    if not isinstance(owners, dict) or not 0 < len(owners) <= 64:
        raise sr.SerialRefused("owned targets must be an object with 1–64 target aliases")
    if len(resolved.targets) > 64:
        raise sr.SerialRefused("resolved roster exceeds 64 targets")
    allowed_cpus = frozenset(resolved.resources.cpu_logical)
    if not allowed_cpus or len(allowed_cpus) > 4096:
        raise sr.SerialRefused("owned serving roster requires 1–4096 declared host CPUs")
    aliases = [alias for target in resolved.targets for alias in target.target_ids]
    if len(aliases) != len(set(aliases)):
        raise sr.SerialRefused("resolved roster has ambiguous target aliases")
    if set(owners) - set(aliases):
        raise sr.SerialRefused(f"owned aliases are not enrolled: {sorted(set(owners) - set(aliases))}")
    required = {"worktree", "anchor_build", "branch", "frozen_prompts"}
    optional = {"launch", "store", "calibrate_serving", "allow_unverified_anchor"}
    for alias, owner in owners.items():
        if not isinstance(owner, dict) or required - owner.keys() or owner.keys() - required - optional:
            raise sr.SerialRefused(f"{alias}: expected ownership fields {sorted(required)} plus {sorted(optional)}")
        for key in required | (owner.keys() & {"launch", "store"}):
            value = owner[key]
            if not isinstance(value, str) or not value.strip() or "\0" in value:
                raise sr.SerialRefused(f"{alias}: {key} must be nonempty text")
            if key != "branch" and not Path(value).is_absolute():
                raise sr.SerialRefused(f"{alias}: {key} must be an absolute path")
        count = owner.get("calibrate_serving")
        if count is not None and (type(count) is not int or count < 2):
            raise sr.SerialRefused(f"{alias}: calibrate_serving needs at least two launches")
        if type(owner.get("allow_unverified_anchor", False)) is not bool:
            raise sr.SerialRefused(f"{alias}: allow_unverified_anchor must be boolean")
    common = ["--planner-model", dict(resolved.actors)["planner"],
              "--critic-model", dict(resolved.actors)["critic"]]
    if common_path is not None:
        extra, _ = sr._json(Path(common_path), limit=16 * 1024)
        if not isinstance(extra, list) or len(extra) > 64 or not all(isinstance(x, str) for x in extra):
            raise sr.SerialRefused("common args must be a bounded JSON string array")
        # Target/workload/resource identities are never overridden by common argv.
        valued = {"--workers", "--planner-model", "--planner-effort", "--critic-model",
                  "--critic-effort", "--pairs", "--serving-pairs", "--belief-root-repo",
                  "--shared-history-root"}
        iterator = iter(extra)
        for arg in iterator:
            flag, equals, value = arg.partition("=")
            if flag == "--rank-prior-experiments" and not equals:
                continue
            if flag not in valued:
                raise sr.SerialRefused(f"not a shared actor/measurement option: {flag}")
            value = value if equals else next(iterator, "")
            if not value or value.startswith("--") or "\0" in value:
                raise sr.SerialRefused(f"common argument {flag} has no valid value")
        common += extra
    targets, skipped = [], []
    for target in resolved.targets:
        selected = [alias for alias in target.target_ids if alias in owners]
        reason = (f"{target.status}: {', '.join(target.missing)}" if target.status != "ready"
                  else "no owned source/anchor/request inputs" if not selected
                  else "requires explicit single-backend enrollment" if target.execution.backend not in {"cpu", "gpu"}
                  else None)
        if reason:
            skipped.append({"target_ids": list(target.target_ids), "reason": reason})
            continue
        alias = selected[0]
        owner = owners[alias]
        if any(owners[other] != owner for other in selected[1:]):
            raise sr.SerialRefused(f"aliases for {alias} have conflicting ownership inputs")
        backend = target.execution.backend
        legacy_targets.select_target(resolved, alias, cpu_serving=backend == "cpu")
        launch_path = Path(owner.get("launch", target.execution.recipe.path))
        launch_body, launch_sha = sr._json(launch_path)
        if "launch" not in owner and launch_sha != target.execution.recipe.sha256:
            raise sr.SerialRefused(f"{alias}: enrolled recipe bytes changed")
        launch = resolved_recipe.CanonicalResolvedRecipe.from_dict(launch_body)
        if launch.backend != backend:
            raise sr.SerialRefused(f"{alias}: launch backend differs from enrollment")
        legacy_targets.validate_resources(resolved.resources, launch, backend=backend,
                                          environment=os.environ)
        if Path(launch.build_dir).resolve() != Path(owner["anchor_build"]).resolve():
            raise sr.SerialRefused(f"{alias}: launch differs from original anchor build")
        legacy_targets.validate_serving_workload(target, launch)
        prompts_body, _ = sr._json(Path(owner["frozen_prompts"]))
        try:
            prompts = planned_serving.FrozenPromptManifest.from_dict(prompts_body)
            prompts.requests(tuple(row.prompt_id for row in prompts.prompts), launch.template)
        except planned_serving.PlannedServingError as exc:
            raise sr.SerialRefused(f"{alias}: {exc}") from exc
        if len(prompts.prompts) != launch.template.np:
            raise sr.SerialRefused(f"{alias}: requests differ from selected serving concurrency")
        branch = owner["branch"]
        if branch.startswith("production-") or (backend == "cpu" and branch == champion.CANONICAL_BRANCH):
            raise sr.SerialRefused(f"{alias}: this serving target needs an experimental branch")
        # Full original target, including seed/production provenance and baseline,
        # scopes each automatically generated output namespace; aliases share it.
        root = Path(target_root).resolve() / sr._digest(target.to_dict())
        argv = ["--resolved-campaign", str(resolved_path), "--target-id", alias,
                "--model", target.execution.model.path,
                "--worktree", owner["worktree"], "--anchor-build", owner["anchor_build"],
                "--store", owner.get("store", str(root / "store")),
                "--worker-root", str(root / "workers"), "--worker-build-root", str(root / "builds"),
                f"--{backend}-serving-launch", str(launch_path),
                "--frozen-prompts", owner["frozen_prompts"]]
        if branch != champion.CANONICAL_BRANCH:
            argv += ["--experimental-branch", branch]
        if owner.get("calibrate_serving") is not None:
            argv += [f"--{backend}-calibrate-serving", str(owner["calibrate_serving"])]
        if owner.get("allow_unverified_anchor"):
            argv.append("--allow-unverified-anchor")
        targets.append(argv + common)
    if not targets:
        raise sr.SerialRefused(f"no ready owned executable targets: {skipped}")
    history_roots = [str(archive.CANONICAL_HISTORY_ROOT),
                     *[sr.option(row, "--store") for row in targets]]
    for row in targets:
        own = Path(sr.option(row, "--store")).resolve()
        for history in dict.fromkeys(history_roots):
            if Path(history).resolve() != own:
                row += ["--shared-history-root", history]
    return targets, skipped, tuple(sorted(allowed_cpus))
