#!/usr/bin/env python3
"""Build an orchestrator X-MAS winner table from research sweep results."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

XMAS_DOMAINS: tuple[str, ...] = (
    "math",
    "code",
    "knowledge",
    "long_context",
    "reasoning",
)

XMAS_FUNCTIONS: tuple[str, ...] = (
    "solve",
    "verify",
    "plan",
    "refine",
    "extract",
)

DEFAULT_FALLBACK_ROLE = "frontdoor"
DOMAIN_PROXY_DERIVATION = "domain_winner_reused_for_function"
FUNCTION_AXIS_DERIVATION = "function_axis_sweep"


def _load_results(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _choose_winner(domain_metrics: dict[str, Any]) -> str:
    if not isinstance(domain_metrics, dict) or not domain_metrics:
        raise ValueError("domain metrics must be a non-empty mapping")

    def score(item: tuple[str, Any]) -> tuple[int, float, str]:
        model_id, metrics = item
        if not isinstance(metrics, dict):
            raise ValueError(f"metrics for {model_id!r} must be a mapping")
        correct = metrics.get("correct")
        total = metrics.get("total")
        wall_mean_s = metrics.get("wall_mean_s")
        if not isinstance(correct, int):
            raise ValueError(f"metrics for {model_id!r} missing integer correct")
        if not isinstance(total, int) or total <= 0:
            raise ValueError(f"metrics for {model_id!r} missing positive integer total")
        if not isinstance(wall_mean_s, int | float):
            raise ValueError(f"metrics for {model_id!r} missing numeric wall_mean_s")
        return (-correct, float(wall_mean_s), model_id)

    return min(domain_metrics.items(), key=score)[0]


def _summary_table(results: dict[str, Any]) -> dict[str, Any]:
    summary = results.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("results missing summary mapping")
    table = summary.get("table")
    if not isinstance(table, dict):
        raise ValueError("results missing summary.table mapping")
    missing = [domain for domain in XMAS_DOMAINS if domain not in table]
    if missing:
        raise ValueError(f"results summary.table missing domains: {', '.join(missing)}")
    return table


def _is_metrics_map(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("correct"), int)
        and isinstance(value.get("total"), int)
        and isinstance(value.get("wall_mean_s"), int | float)
    )


def _is_function_axis_table(table: dict[str, Any]) -> bool:
    first_domain = table.get(XMAS_DOMAINS[0])
    if not isinstance(first_domain, dict):
        return False
    first_function = first_domain.get(XMAS_FUNCTIONS[0])
    return isinstance(first_function, dict) and not _is_metrics_map(first_function)


def build_winner_table(
    results_path: Path,
    *,
    version: str,
    fallback_role: str = DEFAULT_FALLBACK_ROLE,
) -> dict[str, Any]:
    """Return a complete 5x5 orchestrator winner-table payload."""
    results = _load_results(results_path)
    table = _summary_table(results)
    if _is_function_axis_table(table):
        return _build_function_axis_winner_table(
            results,
            results_path,
            version=version,
            fallback_role=fallback_role,
            table=table,
        )
    return _build_domain_proxy_winner_table(
        results,
        results_path,
        version=version,
        fallback_role=fallback_role,
        table=table,
    )


def _build_domain_proxy_winner_table(
    results: dict[str, Any],
    results_path: Path,
    *,
    version: str,
    fallback_role: str,
    table: dict[str, Any],
) -> dict[str, Any]:
    cells: dict[str, dict[str, str]] = {}
    evidence: dict[str, dict[str, dict[str, Any]]] = {}
    domain_winners: dict[str, str] = {}

    for domain in XMAS_DOMAINS:
        domain_metrics = table[domain]
        winner = _choose_winner(domain_metrics)
        domain_winners[domain] = winner
        cells[domain] = {}
        evidence[domain] = {}
        for function in XMAS_FUNCTIONS:
            metrics = domain_metrics[winner]
            cells[domain][function] = winner
            evidence[domain][function] = {
                "cell": f"{domain}:{function}",
                "winner": winner,
                "sample_count": metrics["total"],
                "derivation": DOMAIN_PROXY_DERIVATION,
                "source_domain": domain,
                "source_summary_path": f"summary.table.{domain}.{winner}",
                "candidates": copy.deepcopy(domain_metrics),
            }

    return {
        "version": version,
        "fallback_role": fallback_role,
        "provenance": {
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator": "scripts/research/xmas_winner_table.py",
            "source_results": [str(results_path)],
            "source_started_at": results.get("started_at"),
            "source_n_tasks": results.get("n_tasks"),
            "source_n_models": results.get("n_models"),
            "winner_rule": "correct_desc_then_wall_mean_s_asc",
            "derivation_mode": DOMAIN_PROXY_DERIVATION,
            "domains": list(XMAS_DOMAINS),
            "functions": list(XMAS_FUNCTIONS),
            "domain_winners": domain_winners,
        },
        "cells": cells,
        "evidence": evidence,
    }


def _build_function_axis_winner_table(
    results: dict[str, Any],
    results_path: Path,
    *,
    version: str,
    fallback_role: str,
    table: dict[str, Any],
) -> dict[str, Any]:
    cells: dict[str, dict[str, str]] = {}
    evidence: dict[str, dict[str, dict[str, Any]]] = {}
    cell_winners: dict[str, str] = {}

    for domain in XMAS_DOMAINS:
        function_table = table[domain]
        if not isinstance(function_table, dict):
            raise ValueError(f"results summary.table.{domain} must be a mapping")
        missing_functions = [
            function for function in XMAS_FUNCTIONS if function not in function_table
        ]
        if missing_functions:
            raise ValueError(
                "results summary.table."
                f"{domain} missing functions: {', '.join(missing_functions)}"
            )
        cells[domain] = {}
        evidence[domain] = {}
        for function in XMAS_FUNCTIONS:
            cell_metrics = function_table[function]
            winner = _choose_winner(cell_metrics)
            metrics = cell_metrics[winner]
            cells[domain][function] = winner
            cell_winners[f"{domain}:{function}"] = winner
            evidence[domain][function] = {
                "cell": f"{domain}:{function}",
                "winner": winner,
                "sample_count": metrics["total"],
                "derivation": FUNCTION_AXIS_DERIVATION,
                "source_domain": domain,
                "source_function": function,
                "source_summary_path": f"summary.table.{domain}.{function}.{winner}",
                "candidates": copy.deepcopy(cell_metrics),
            }

    return {
        "version": version,
        "fallback_role": fallback_role,
        "provenance": {
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator": "scripts/research/xmas_winner_table.py",
            "source_results": [str(results_path)],
            "source_started_at": results.get("started_at"),
            "source_n_tasks": results.get("n_tasks"),
            "source_n_models": results.get("n_models"),
            "winner_rule": "correct_desc_then_wall_mean_s_asc",
            "derivation_mode": FUNCTION_AXIS_DERIVATION,
            "domains": list(XMAS_DOMAINS),
            "functions": list(XMAS_FUNCTIONS),
            "cell_winners": cell_winners,
        },
        "cells": cells,
        "evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", default="xmas-v1-domain-proxy")
    parser.add_argument("--fallback-role", default=DEFAULT_FALLBACK_ROLE)
    args = parser.parse_args()

    try:
        payload = build_winner_table(
            args.results,
            version=args.version,
            fallback_role=args.fallback_role,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
