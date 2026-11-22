"""Actor configuration and actor budget error helpers for run.py, split verbatim from run.py (code-budget split)."""
from __future__ import annotations
import json
from typing import Any
from . import actor_opencode_config


def _actor_config(args, resolved_campaign=None) -> dict[str, Any]:
    """This launch's actor/backend configuration: provenance only, never identity.

    Recorded on every checkpoint and in loop-run.json; a resumed row carries the keys
    that differ from its checkpoint's (`resume.actor_config_diff`)."""
    get = lambda name, default=None: getattr(args, name, default)  # noqa: E731
    config: dict[str, Any] = {
        "planner_model": get("planner_model"), "planner_effort": get("planner_effort"),
        "critic_model": get("critic_model"), "critic_effort": get("critic_effort"),
        # The author runs on the planner's backend.
        "author_model": get("planner_model"),
        "author_thinking": get("actor_author_thinking"),
        "author_action_rule": get("actor_author_action_rule"),
        "actor_seat": get("actor_seat"), "actor_concise": get("actor_concise"),
        "actor_context_limit": get("actor_context_limit"),
        "actor_output_limit": get("actor_output_limit"),
        "actor_planner_output_limit": get("actor_planner_output_limit"),
        "actor_author_output_limit": get("actor_author_output_limit"),
        "actor_planner_budget_s": get("actor_planner_budget_s"),
        "actor_author_budget_s": get("actor_author_budget_s"),
        "actor_planner_salvage_s": get("actor_planner_salvage_s"),
        "actor_timeout_s": get("actor_timeout_s"),
        "actor_trim_instructions": get("actor_trim_instructions"),
        "actor_trim_tools": get("actor_trim_tools"),
        "actor_lane_guard": get("actor_lane_guard"),
    }
    # Present only when the planner drops its reasoning history, so a keep run records
    # the historical keys.
    history = get("actor_planner_reasoning_history")
    if history and history != actor_opencode_config.DEFAULT_PLANNER_REASONING_HISTORY:
        config["planner_reasoning_history"] = history
    # UFH14-B1: present only when on, so a run with them off records the historical keys.
    if get("actor_serving_f1") == "on":
        config["actor_serving_f1"] = "on"
    if get("actor_answer_protocol") not in (None, "off"):
        config["actor_answer_protocol"] = get("actor_answer_protocol")
        config["actor_answer_force_frac"] = get("actor_answer_force_frac")
    if get("actor_planner_think_budget"):
        config["actor_planner_think_budget"] = get("actor_planner_think_budget")
    if get("actor_local_via_orchestrator") == "on":
        config["actor_local_via_orchestrator"] = get("actor_local_orchestrator_roles")
    # Per-lane planner/author models (`lane_actors`): present only when a lane is
    # overridden, so a run without `--lane-actor-models` records the historical keys.
    lane_spec = get("lane_actor_models")
    if isinstance(lane_spec, str) and lane_spec.strip():
        from . import lane_actors
        config.update(lane_actors.provenance(
            lane_actors.parse(lane_spec, workers=int(get("workers", 1) or 1)),
            get("planner_effort")))
    if resolved_campaign is not None:
        config["manifest_actors"] = {key: value for key, value in resolved_campaign.actors}
        config["manifest_fallbacks"] = {key: list(values)
                                        for key, values in resolved_campaign.fallbacks}
    return json.loads(json.dumps(config, default=str))


def _actor_budget_error(args) -> str | None:
    """Why the OAB-22/23 knobs are unusable, or None."""
    for flag in ("actor_context_limit", "actor_output_limit", "actor_planner_output_limit",
                 "actor_author_output_limit", "actor_planner_budget_s",
                 "actor_author_budget_s"):
        if int(getattr(args, flag)) < 0:
            return f"--{flag.replace('_', '-')} must be >= 0"
    if int(getattr(args, "actor_planner_salvage_s", 0) or 0) < 0:
        return "--actor-planner-salvage-s must be >= 0"
    if int(getattr(args, "actor_planner_think_budget", 0) or 0) < 0:
        return "--actor-planner-think-budget must be >= 0"
    frac = float(getattr(args, "actor_answer_force_frac", 0.65))
    if not 0.0 < frac < 1.0:
        return "--actor-answer-force-frac must be in (0, 1)"
    context = int(args.actor_context_limit)
    if not context:
        return None
    # Operator pool budget (2026-09-26): a FULL unified pool plus MTP crashes
    # llama-server, so the context cap leaves >= 16k of :8083's unified pool free for
    # the other slots, and opencode (which compacts at context - output) keeps >= 32k
    # of context headroom under every role's output limit.
    cap = actor_opencode_config.MAX_CONTEXT_LIMIT
    if context > cap:
        return (f"--actor-context-limit ({context}) must be <= {cap} (the "
                f"{actor_opencode_config.POOL_TOKENS}-token unified pool minus "
                f"{actor_opencode_config.POOL_RESERVE} kept free: a full pool plus MTP "
                "crashes llama-server)")
    headroom = actor_opencode_config.MIN_COMPACTION_HEADROOM
    for flag in ("actor_output_limit", "actor_planner_output_limit",
                 "actor_author_output_limit"):
        output = int(getattr(args, flag))
        if output and output >= context - headroom:
            return (f"--{flag.replace('_', '-')} ({output}) must be below "
                    f"--actor-context-limit - {headroom} ({context - headroom}): "
                    "opencode compacts at context - output and needs that headroom")
    return None
