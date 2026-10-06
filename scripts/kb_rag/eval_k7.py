"""K7 KB-RAG retrieval evaluation harness.

This re-implements the Flywheel-style evaluation methodology for this repo's
own markdown corpus. It measures evidence-file recall for curated multi-hop
questions over several query-time signal configurations.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Make src/ importable when invoked directly.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.retrieval.kb_rag import DEFAULT_INDEX_DIR, query as kb_query, stats as kb_stats  # noqa: E402
from src.retrieval.kb_catalog_dependency import (  # noqa: E402
    read_dependency as read_catalog_dependency,
)

DEFAULT_CASES_PATH = _HERE.with_name("k7_seed_cases.json")
DEFAULT_CUTOFFS = (3, 5, 10)


@dataclass(frozen=True)
class EvalConfig:
    name: str
    recency_weight: float = 0.0
    recency_sigma_days: float = 90.0
    rerank: bool = False
    rerank_weight: float = 0.0

    def query_kwargs(self) -> dict[str, Any]:
        return {
            "recency_weight": self.recency_weight,
            "recency_sigma_days": self.recency_sigma_days,
            "rerank": self.rerank,
            "rerank_weight": self.rerank_weight,
        }


DEFAULT_CONFIGS = (
    EvalConfig("maxsim"),
    EvalConfig("recency_w0.1_s90", recency_weight=0.1, recency_sigma_days=90.0),
    EvalConfig("recency_w0.3_s90", recency_weight=0.3, recency_sigma_days=90.0),
    EvalConfig("rerank_w0.3", rerank=True, rerank_weight=0.3),
    EvalConfig("rerank_w0.6", rerank=True, rerank_weight=0.6),
    EvalConfig(
        "recency_w0.1_s90_rerank_w0.3",
        recency_weight=0.1,
        recency_sigma_days=90.0,
        rerank=True,
        rerank_weight=0.3,
    ),
)
CONFIG_BY_NAME = {cfg.name: cfg for cfg in DEFAULT_CONFIGS}


def _resolve_path(raw: str) -> str:
    """Normalize paths for evidence/result matching across symlinks."""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        workspace_candidate = Path("/workspace") / p
        p = workspace_candidate if workspace_candidate.exists() else (_REPO / p)
    return str(p.resolve(strict=False))


def _catalog_metadata(index_dir: Path) -> dict[str, Any]:
    catalog = index_dir / "catalog.sqlite"
    if not catalog.exists():
        return {"exists": False}

    conn = sqlite3.connect(str(catalog))
    try:
        chunks, files, max_mtime, min_mtime = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT file_path), MAX(mtime), MIN(mtime) FROM chunk"
        ).fetchone()
    finally:
        conn.close()

    def _iso(ts: float | None) -> str | None:
        if ts is None:
            return None
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

    return {
        "exists": True,
        "chunks": chunks,
        "files": files,
        "max_mtime": max_mtime,
        "max_mtime_utc": _iso(max_mtime),
        "min_mtime": min_mtime,
        "min_mtime_utc": _iso(min_mtime),
    }


def load_cases(
    path: Path | str = DEFAULT_CASES_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate a K7 case file."""
    path = Path(path)
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        metadata: dict[str, Any] = {}
        raw_cases = payload
    elif isinstance(payload, dict):
        metadata = {k: v for k, v in payload.items() if k != "cases"}
        raw_cases = payload.get("cases", [])
    else:
        raise ValueError(f"case file must contain a JSON object or list: {path}")

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, case in enumerate(raw_cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"case #{idx} is not an object")
        case_id = str(case.get("id") or f"case_{idx:03d}")
        if case_id in seen_ids:
            raise ValueError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)

        query = str(case.get("query") or "").strip()
        evidence_files = case.get("evidence_files") or []
        if not query:
            raise ValueError(f"{case_id}: missing query")
        if not isinstance(evidence_files, list) or not evidence_files:
            raise ValueError(f"{case_id}: evidence_files must be a non-empty list")

        normalized = dict(case)
        normalized["id"] = case_id
        normalized["query"] = query
        normalized["protocol"] = str(case.get("protocol") or "hotpotqa_template")
        normalized["evidence_files"] = [str(p) for p in evidence_files]
        normalized["resolved_evidence_files"] = [_resolve_path(str(p)) for p in evidence_files]
        cases.append(normalized)

    return cases, metadata


def parse_cutoffs(raw: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [int(v) for v in raw]
    if not values or any(v <= 0 for v in values):
        raise ValueError("cutoffs must be positive integers")
    return tuple(sorted(set(values)))


def select_configs(raw: str | Iterable[str] = "default") -> list[EvalConfig]:
    if isinstance(raw, str):
        names = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        names = [str(part).strip() for part in raw if str(part).strip()]

    if not names or names == ["default"] or names == ["all"]:
        return list(DEFAULT_CONFIGS)

    configs: list[EvalConfig] = []
    unknown: list[str] = []
    for name in names:
        cfg = CONFIG_BY_NAME.get(name)
        if cfg is None:
            unknown.append(name)
        else:
            configs.append(cfg)
    if unknown:
        raise ValueError(f"unknown config(s): {', '.join(unknown)}")
    return configs


def score_case(
    evidence_files: Iterable[str],
    results: list[dict[str, Any]],
    cutoffs: Iterable[int] = DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    """Score one ranked result list against evidence-file ground truth."""
    evidence = list(dict.fromkeys(_resolve_path(p) for p in evidence_files))
    first_rank_by_file: dict[str, int] = {}
    ranked_files: list[str] = []
    for rank, row in enumerate(results, start=1):
        file_path = _resolve_path(str(row.get("file", "")))
        ranked_files.append(file_path)
        first_rank_by_file.setdefault(file_path, rank)

    found_ranks = {
        file_path: first_rank_by_file[file_path]
        for file_path in evidence
        if file_path in first_rank_by_file
    }

    metrics: dict[str, Any] = {
        "evidence_count": len(evidence),
        "found_count": len(found_ranks),
        "found_files": sorted(found_ranks),
        "missing_files": sorted(set(evidence) - set(found_ranks)),
        "first_evidence_rank": min(found_ranks.values()) if found_ranks else None,
        "all_evidence_rank": max(found_ranks.values())
        if len(found_ranks) == len(evidence)
        else None,
        "ranked_files": ranked_files,
    }

    for cutoff in parse_cutoffs(cutoffs):
        found_at_k = sum(1 for rank in found_ranks.values() if rank <= cutoff)
        metrics[f"recall@{cutoff}"] = found_at_k / len(evidence) if evidence else 0.0
        metrics[f"perfect@{cutoff}"] = found_at_k == len(evidence)

    return metrics


def summarize_rows(
    rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    configs: list[EvalConfig],
    cutoffs: Iterable[int] = DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    cutoffs = parse_cutoffs(cutoffs)
    protocols = sorted({case["protocol"] for case in cases})
    total_cases = len(cases)
    summary: dict[str, Any] = {
        "case_count": total_cases,
        "protocol_counts": {
            protocol: sum(1 for case in cases if case["protocol"] == protocol)
            for protocol in protocols
        },
        "configs": {},
    }

    for cfg in configs:
        cfg_rows = [row for row in rows if row["config"] == cfg.name]
        cfg_summary: dict[str, Any] = {"overall": _aggregate(cfg_rows, cutoffs), "by_protocol": {}}
        for protocol in protocols:
            cfg_summary["by_protocol"][protocol] = _aggregate(
                [row for row in cfg_rows if row["protocol"] == protocol], cutoffs
            )
        summary["configs"][cfg.name] = cfg_summary

    key = f"mean_recall@{max(cutoffs)}"
    best = sorted(
        ((name, data["overall"].get(key, 0.0)) for name, data in summary["configs"].items()),
        key=lambda x: (x[1], x[0]),
        reverse=True,
    )
    summary["best_config_by_recall"] = {"cutoff": max(cutoffs), "ranking": best}
    return summary


def _aggregate(rows: list[dict[str, Any]], cutoffs: tuple[int, ...]) -> dict[str, Any]:
    if not rows:
        empty: dict[str, Any] = {"n": 0}
        for cutoff in cutoffs:
            empty[f"mean_recall@{cutoff}"] = 0.0
            empty[f"perfect@{cutoff}"] = "0/0"
            empty[f"perfect_rate@{cutoff}"] = 0.0
        return empty

    out: dict[str, Any] = {"n": len(rows)}
    for cutoff in cutoffs:
        recalls = [float(row[f"recall@{cutoff}"]) for row in rows]
        perfects = [bool(row[f"perfect@{cutoff}"]) for row in rows]
        out[f"mean_recall@{cutoff}"] = round(sum(recalls) / len(recalls), 4)
        out[f"perfect@{cutoff}"] = f"{sum(perfects)}/{len(perfects)}"
        out[f"perfect_rate@{cutoff}"] = round(sum(perfects) / len(perfects), 4)

    first_ranks = [row["first_evidence_rank"] for row in rows if row["first_evidence_rank"]]
    out["mean_first_evidence_rank"] = (
        round(sum(first_ranks) / len(first_ranks), 4) if first_ranks else None
    )
    out["missed_all_evidence_count"] = sum(1 for row in rows if row["found_count"] == 0)
    return out


def evaluate(
    cases: list[dict[str, Any]],
    configs: list[EvalConfig],
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    top_k: int = 10,
    cutoffs: Iterable[int] = DEFAULT_CUTOFFS,
    query_fn: Callable[..., list[dict[str, Any]]] = kb_query,
) -> list[dict[str, Any]]:
    """Run all cases through all configs and return per-case rows."""
    cutoffs = parse_cutoffs(cutoffs)
    rows: list[dict[str, Any]] = []
    for case in cases:
        for cfg in configs:
            started = time.perf_counter()
            results = query_fn(
                case["query"],
                top_k=top_k,
                index_dir=index_dir,
                **cfg.query_kwargs(),
            )
            elapsed = time.perf_counter() - started
            metrics = score_case(case["resolved_evidence_files"], results, cutoffs)
            top_files = []
            for result in results:
                resolved = _resolve_path(str(result.get("file", "")))
                if resolved not in top_files:
                    top_files.append(resolved)
            rows.append(
                {
                    "case_id": case["id"],
                    "protocol": case["protocol"],
                    "tags": case.get("tags", []),
                    "config": cfg.name,
                    "query": case["query"],
                    "evidence_files": case["resolved_evidence_files"],
                    "elapsed_sec": round(elapsed, 4),
                    "top_files": top_files,
                    "top_results": results,
                    **metrics,
                }
            )
    return rows


def run_eval(
    cases_path: Path | str = DEFAULT_CASES_PATH,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    output_dir: Path | str | None = None,
    configs: str | Iterable[str] = "default",
    cutoffs: str | Iterable[int] = DEFAULT_CUTOFFS,
    top_k: int | None = None,
    limit_cases: int | None = None,
) -> dict[str, Any]:
    cutoffs_tuple = parse_cutoffs(cutoffs)
    top_k = top_k or max(cutoffs_tuple)
    selected_configs = select_configs(configs)
    cases, metadata = load_cases(cases_path)
    if limit_cases is not None:
        cases = cases[:limit_cases]

    index_dir = Path(index_dir)
    started = time.perf_counter()
    rows = evaluate(
        cases=cases,
        configs=selected_configs,
        index_dir=index_dir,
        top_k=top_k,
        cutoffs=cutoffs_tuple,
    )
    elapsed = time.perf_counter() - started

    evidence_missing = sorted(
        {
            evidence
            for case in cases
            for evidence in case["resolved_evidence_files"]
            if not Path(evidence).exists()
        }
    )
    summary = summarize_rows(rows, cases, selected_configs, cutoffs_tuple)
    summary.update(
        {
            "ok": not evidence_missing,
            "elapsed_sec": round(elapsed, 2),
            "cases_path": str(Path(cases_path).resolve(strict=False)),
            "index_dir": str(index_dir.resolve(strict=False)),
            "cutoffs": list(cutoffs_tuple),
            "top_k": top_k,
            "selected_configs": [cfg.name for cfg in selected_configs],
            "case_file_metadata": metadata,
            "evidence_files_missing_on_disk": evidence_missing,
            "index_stats": kb_stats(index_dir=index_dir),
            "catalog_metadata": _catalog_metadata(index_dir),
            "catalog_dependency_at_report": read_catalog_dependency(index_dir / "catalog.sqlite"),
        }
    )

    output_root = Path(output_dir) if output_dir else _default_output_dir()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    rows_path = output_root / "rows.jsonl"
    cases_out_path = output_root / "cases.json"

    summary["output_dir"] = str(output_root.resolve(strict=False))
    summary["summary_path"] = str(summary_path.resolve(strict=False))
    summary["rows_path"] = str(rows_path.resolve(strict=False))
    summary["cases_out_path"] = str(cases_out_path.resolve(strict=False))

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with rows_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    cases_out_path.write_text(json.dumps({"metadata": metadata, "cases": cases}, indent=2) + "\n")
    return summary


def _default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return _REPO / "data" / "kb_rag" / "eval" / f"k7_{stamp}"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the K7 KB-RAG retrieval eval")
    p.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="case JSON file")
    p.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="KB-RAG index directory")
    p.add_argument("--output-dir", help="directory for summary.json + rows.jsonl")
    p.add_argument(
        "--configs",
        default="default",
        help="comma list of configs, or default/all. Available: " + ", ".join(CONFIG_BY_NAME),
    )
    p.add_argument("--cutoffs", default="3,5,10", help="comma-separated recall cutoffs")
    p.add_argument("--top-k", type=int, help="query depth; defaults to max cutoff")
    p.add_argument("--limit-cases", type=int, help="run only the first N cases")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_eval(
        cases_path=args.cases,
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        configs=args.configs,
        cutoffs=args.cutoffs,
        top_k=args.top_k,
        limit_cases=args.limit_cases,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
