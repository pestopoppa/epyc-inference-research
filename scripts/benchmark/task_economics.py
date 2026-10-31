"""Report-only arithmetic over exactly the scored draws; no pricing estimates."""

import math


def native_usage(response):
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return {
        "schema": "task_native_usage.v1",
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        # Never substitute judge spend, a price table, or an absent zero.
        "total_cost_usd": response.get("total_cost_usd"),
    }


def summarize(draws):
    def count(value):
        return type(value) is int and value >= 0

    usages = [d.get("usage") for d in draws]
    tokens_known = bool(draws) and all(
        isinstance(u, dict)
        and u.get("schema") == "task_native_usage.v1"
        and count(u.get("prompt_tokens"))
        and count(u.get("completion_tokens"))
        for u in usages
    )
    tokens = (
        sum(u["prompt_tokens"] + u["completion_tokens"] for u in usages)
        if tokens_known
        else None
    )
    costs_known = bool(draws) and all(
        isinstance(u, dict)
        and u.get("schema") == "task_native_usage.v1"
        and type(u.get("total_cost_usd")) in (int, float)
        and math.isfinite(u["total_cost_usd"])
        and u["total_cost_usd"] >= 0
        for u in usages
    )
    cost = sum(u["total_cost_usd"] for u in usages) if costs_known else None
    quality = sum(bool(d["correct"]) for d in draws) / len(draws) if draws else None
    completions = sum(d.get("completed") is True for d in draws)
    return {
        "scope": "all_scored_draws_including_resumed_and_failed",
        "n_draws": len(draws),
        "n_completions": completions,
        "quality": quality,
        "tokens_total": tokens,
        "cost_usd_total": cost,
        "completions_per_million_tokens": completions * 1e6 / tokens
        if tokens
        else None,
        "quality_per_dollar": quality / cost if cost else None,
        "tokens_complete": tokens_known,
        "cost_complete": costs_known,
        "token_denominator": "server_reported_prompt_plus_completion_tokens",
        "cost_denominator": "server_reported_task_generation_total_cost_usd",
    }
