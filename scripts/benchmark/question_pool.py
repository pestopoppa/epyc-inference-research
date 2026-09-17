#!/usr/bin/env python3
"""Pre-extracted question pool for fast sampling.

Instead of loading 16 HuggingFace datasets on every seeding run (~20-30s),
pre-extract all ~45K questions into a single JSONL file. Runtime sampling
then reads this file (~100ms).

Usage:
    # Build the pool (one-time, ~30s)
    python scripts/benchmark/question_pool.py --build

    # Programmatic use
    from question_pool import load_pool, sample_from_pool
    pool = load_pool()
    questions = sample_from_pool(pool, suites=["math"], sample_per_suite=10, seed=42)
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

logger = logging.getLogger(__name__)

POOL_FILE = PROJECT_ROOT / "benchmarks" / "prompts" / "question_pool.jsonl"
# Header sentinel — first line of the JSONL is metadata, not a question
_HEADER_KEY = "__pool_metadata__"
# Warn if pool is older than this
_STALE_DAYS = 30


class PoolBuildInvariantError(RuntimeError):
    """The pool build would silently lose a suite; nothing was written."""


def _empty_suite_reason(summary: dict[str, Any] | None) -> str:
    """The adapter's own recorded explanation for yielding zero rows, or ''."""
    if not summary:
        return ""
    degraded = summary.get("degraded_sources") or []
    if degraded:
        return "degraded_sources: " + "; ".join(
            f"{d.get('source', '?')}: {d.get('error', '?')}" for d in degraded
        )
    return ""


def _check_build_invariant(
    adapter_suites: set[str],
    stats: dict[str, int],
    empty_suites: dict[str, str],
) -> list[str]:
    """A3 build invariant (EVL-12 C2 / LOSS-1/2).

    (a) every registered adapter suite has an entry in the build stats;
    (b) every suite with zero rows carries a recorded reason (extraction error or
        adapter-recorded degraded source). A zero with no reason is a silent loss.
    """
    violations: list[str] = []
    missing = sorted(adapter_suites - set(stats))
    if missing:
        violations.append(f"adapter suites absent from the build: {missing}")
    silent = sorted(
        suite for suite, count in stats.items()
        if count == 0 and not empty_suites.get(suite)
    )
    if silent:
        violations.append(f"suites with zero rows and no recorded reason: {silent}")
    return violations


def build_pool(output_path: Path | None = None) -> dict[str, int]:
    """Extract all questions from all adapters + YAML suites into a JSONL file.

    Returns dict mapping suite_name -> count of questions extracted.

    Raises PoolBuildInvariantError BEFORE writing when a registered adapter suite
    is missing or a suite yields zero rows with no recorded reason; an accounted
    zero (e.g. a gated/absent source) is recorded in the header's
    ``empty_suites`` and does not fail the build (A3, 2026-07-21). The pool is
    written atomically so a failed build never truncates the live file.
    """
    from dataset_adapters import ADAPTER_SUITES, YAML_ONLY_SUITES, get_adapter

    output_path = output_path or POOL_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {}
    empty_suites: dict[str, str] = {}
    adapter_stats: dict[str, dict[str, Any]] = {}
    source_counts: dict[str, dict[str, int]] = {}
    all_questions: list[dict] = []

    # 1. Extract from HF dataset adapters
    for suite_name in sorted(ADAPTER_SUITES):
        adapter = get_adapter(suite_name)
        if adapter is None:
            logger.warning(f"  [{suite_name}] No adapter found, skipping")
            continue
        try:
            questions = adapter.extract_all()
            summary = None
            if hasattr(adapter, "accounting_summary"):
                summary = adapter.accounting_summary()
                adapter_stats[suite_name] = summary
                counts = summary.get("source_counts")
                if isinstance(counts, dict):
                    source_counts[suite_name] = {
                        str(k): int(v) for k, v in counts.items()
                    }
            # Ensure suite field is set
            for q in questions:
                q.setdefault("suite", suite_name)
                q.setdefault("dataset_source", "hf_adapter")
            stats[suite_name] = len(questions)
            if not questions:
                empty_suites[suite_name] = _empty_suite_reason(summary)
            all_questions.extend(questions)
            logger.info(f"  [{suite_name}] Extracted {len(questions)} questions")
        except Exception as e:
            logger.error(f"  [{suite_name}] Extraction failed: {e}")
            stats[suite_name] = 0
            empty_suites[suite_name] = (
                f"extraction failed: {type(e).__name__}: {e}"
            )

    # 2. Extract from YAML-only suites
    try:
        import yaml as _yaml
    except ImportError:
        _yaml = None

    if _yaml:
        debug_dir = PROJECT_ROOT / "benchmarks" / "prompts" / "debug"
        for suite_name in sorted(YAML_ONLY_SUITES):
            yaml_path = debug_dir / f"{suite_name}.yaml"
            if not yaml_path.exists():
                continue
            try:
                with open(yaml_path) as f:
                    data = _yaml.safe_load(f)
                questions = data.get("questions", [])
                default_scoring_method = data.get("scoring_method", "exact_match")
                default_scoring_config = data.get("scoring_config", {})
                converted = []
                for q in questions:
                    converted.append({
                        "id": q["id"],
                        "suite": suite_name,
                        "prompt": q["prompt"].strip(),
                        "context": "",
                        "expected": q.get("expected", ""),
                        "image_path": q.get("image_path", ""),
                        "tier": q.get("tier", 1),
                        "scoring_method": q.get("scoring_method", default_scoring_method),
                        "scoring_config": q.get("scoring_config", default_scoring_config),
                        "dataset_source": "yaml",
                    })
                stats[suite_name] = len(converted)
                all_questions.extend(converted)
                logger.info(f"  [{suite_name}] Extracted {len(converted)} questions (YAML)")
            except Exception as e:
                logger.error(f"  [{suite_name}] YAML extraction failed: {e}")
                stats[suite_name] = 0
                empty_suites[suite_name] = (
                    f"YAML extraction failed: {type(e).__name__}: {e}"
                )
            else:
                if not converted:
                    empty_suites[suite_name] = ""

    violations = _check_build_invariant(set(ADAPTER_SUITES), stats, empty_suites)
    if violations:
        raise PoolBuildInvariantError(
            f"refusing to write {output_path}: " + "; ".join(violations)
        )
    for suite_name, reason in sorted(empty_suites.items()):
        logger.warning(f"  [{suite_name}] 0 rows (accounted): {reason}")

    # 3. Write JSONL with header
    header = {
        _HEADER_KEY: True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "question_pool.py",
        "total_questions": len(all_questions),
        "suites": stats,
        "adapter_stats": adapter_stats,
        "source_counts": source_counts,
        "n_math500": source_counts.get("math", {}).get("math500", 0),
        "empty_suites": dict(sorted(empty_suites.items())),
    }

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with open(tmp_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)

    logger.info(f"Pool written: {output_path} ({len(all_questions)} questions)")
    return stats


def refresh_suites(
    suites: list[str],
    source_path: Path | None = None,
    output_path: Path | None = None,
    *,
    allow_live_overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    """Re-extract only ``suites`` from their adapters; copy every other row verbatim.

    Why (PRB-T4, 2026-09-17): the live pool was built 2026-07-27 and never
    rebuilt, so adapter fixes that landed afterwards (the livecodebench
    executable oracle, 2026-08-12; the MMLU-Pro label range, 2026-09-17) never
    reached the rows every harness actually reads. A full ``--build`` rewrites
    all 38 suites — the eval tower's instrument included — so this splices only
    the named suites. Other rows are byte-preserved and keep their order;
    refreshed rows are appended.

    Refuses to overwrite the live pool unless ``allow_live_overwrite``: swapping
    the tower's input is an instrument-era change for its owner, not a side
    effect of a harness fix. Written atomically.
    """
    from dataset_adapters import get_adapter

    source_path = source_path or POOL_FILE
    output_path = output_path or POOL_FILE
    if output_path.resolve() == POOL_FILE.resolve() and not allow_live_overwrite:
        raise PoolBuildInvariantError(
            f"refusing to overwrite the live pool {POOL_FILE} without allow_live_overwrite"
        )
    targets = set(suites)
    fresh: list[dict] = []
    report: dict[str, dict[str, int]] = {}
    for suite in sorted(targets):
        adapter = get_adapter(suite)
        if adapter is None:
            raise PoolBuildInvariantError(f"no adapter for suite {suite!r}")
        rows = adapter.extract_all()
        if not rows:
            raise PoolBuildInvariantError(f"adapter {suite!r} produced zero rows; refusing")
        for q in rows:
            q.setdefault("suite", suite)
            q.setdefault("dataset_source", "hf_adapter")
        report[suite] = {"added": len(rows), "removed": 0,
                         "dropped_by_adapter": int(getattr(adapter, "dropped_rows", 0))}
        fresh.extend(rows)

    header: dict[str, Any] | None = None
    kept = 0
    with open(source_path, encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            if header is None and _HEADER_KEY in line[:64]:
                header = json.loads(line)
                continue
            suite = json.loads(line).get("suite")
            if suite in targets:
                report[suite]["removed"] += 1
            else:
                kept += 1
    if header is None:
        raise PoolBuildInvariantError(f"{source_path} has no pool header")

    header = dict(header)
    header["suites"] = dict(header.get("suites") or {})
    for suite in targets:
        header["suites"][suite] = report[suite]["added"]
    header["total_questions"] = kept + len(fresh)
    header.setdefault("refreshed_suites", {})
    now = datetime.now(timezone.utc).isoformat()
    for suite in sorted(targets):
        header["refreshed_suites"][suite] = {"refreshed_at": now, **report[suite]}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with open(source_path, encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as out:
        out.write(json.dumps(header) + "\n")
        seen_header = False
        for line in src:
            if not line.strip():
                continue
            if not seen_header and _HEADER_KEY in line[:64]:
                seen_header = True
                continue
            if json.loads(line).get("suite") in targets:
                continue
            out.write(line if line.endswith("\n") else line + "\n")
        for q in fresh:
            out.write(json.dumps(q, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp_path, output_path)
    return report


def load_pool(
    pool_path: Path | None = None, warn_stale: bool = True,
) -> dict[str, list[dict]]:
    """Load the pre-extracted pool, grouped by suite.

    Returns dict mapping suite_name -> list of question dicts.
    Prints warning if pool is older than _STALE_DAYS.
    """
    pool_path = pool_path or POOL_FILE
    if not pool_path.exists():
        return {}

    pool: dict[str, list[dict]] = {}
    header_seen = False
    header: dict[str, Any] | None = None

    with open(pool_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Header line
            if obj.get(_HEADER_KEY):
                header_seen = True
                header = obj
                if warn_stale:
                    _check_staleness(obj)
                continue

            suite = obj.get("suite", "unknown")
            pool.setdefault(suite, []).append(obj)

    if not header_seen:
        logger.warning("Pool file has no header — consider rebuilding with --rebuild-pool")
    elif header is not None:
        _reconcile_loaded_counts(header, pool, pool_path)

    return pool


def _reconcile_loaded_counts(
    header: dict[str, Any], pool: dict[str, list[dict]], pool_path: Path,
) -> None:
    """Warn when loaded row counts do not match pool header statistics."""
    expected_total = header.get("total_questions")
    loaded_total = sum(len(rows) for rows in pool.values())
    if isinstance(expected_total, int) and expected_total != loaded_total:
        logger.warning(
            "Pool loaded count mismatch for %s: header total_questions=%d, loaded=%d",
            pool_path,
            expected_total,
            loaded_total,
        )

    expected_suites = header.get("suites", {})
    if not isinstance(expected_suites, dict):
        return

    for suite, expected in sorted(expected_suites.items()):
        if not isinstance(expected, int):
            continue
        loaded = len(pool.get(suite, []))
        if expected != loaded:
            logger.warning(
                "Pool suite count mismatch for %s [%s]: header=%d, loaded=%d",
                pool_path,
                suite,
                expected,
                loaded,
            )

    unexpected_suites = sorted(set(pool) - set(expected_suites))
    if unexpected_suites:
        logger.warning(
            "Pool contains suite(s) absent from header for %s: %s",
            pool_path,
            ", ".join(unexpected_suites),
        )


def _check_staleness(header: dict) -> None:
    """Warn if pool is older than _STALE_DAYS."""
    generated_at = header.get("generated_at")
    if not generated_at:
        return
    try:
        gen_time = datetime.fromisoformat(generated_at)
        age_days = (datetime.now(timezone.utc) - gen_time).days
        if age_days > _STALE_DAYS:
            logger.warning(
                f"Question pool is {age_days} days old (> {_STALE_DAYS}). "
                "Consider rebuilding: python scripts/benchmark/question_pool.py --build"
            )
    except (ValueError, TypeError):
        pass


_ID_INDEX_TAIL = re.compile(r"_\d+$")


def source_stratum(row: dict[str, Any]) -> str:
    """Source/sub-source key of a pool row, used for stratified sampling.

    The adapters encode provenance in the row id: ``gsm8k_00012`` ->
    ``gsm8k``, ``math500_Algebra_00003`` -> ``math500_Algebra``,
    ``olympiadbench_geometry_00007`` -> ``olympiadbench_geometry``,
    ``mmlu_pro_law_01424`` -> ``mmlu_pro_law``. Ids without a numeric tail
    (``leetcode_two-sum``) collapse to their first token, so a suite of
    slug-keyed rows is one stratum rather than one stratum per row.
    """
    qid = str(row.get("id") or row.get("question_id") or "")
    if not qid:
        return "unknown"
    stripped = _ID_INDEX_TAIL.sub("", qid)
    if stripped != qid and stripped:
        return stripped
    return qid.split("_", 1)[0]


def stratified_sample(
    rows: list[dict[str, Any]], n: int, seed: int,
    stratum: Any = source_stratum,
) -> list[dict[str, Any]]:
    """Seeded sample of ``n`` rows, allocated proportionally across strata.

    Why this exists (PRB-T4, 2026-09-17): harnesses that took the FIRST ``n``
    rows of a suite in file order drew 100% ``gsm8k`` from the ``math`` suite,
    whose file order is 1,319 gsm8k rows followed by 500 MATH-500 rows.

    Deterministic in (row set, n, seed) and independent of file order: rows
    are keyed by id inside each stratum before drawing. Allocation is
    largest-remainder proportional with seeded tie-breaks; every stratum with a
    non-zero share keeps at least its floor. The result is shuffled with the
    same seed so suites interleave their sources.
    """
    if n <= 0 or not rows:
        return []
    if n >= len(rows):
        out = sorted(rows, key=lambda r: str(r.get("id", "")))
        random.Random(seed).shuffle(out)
        return out

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(stratum(row), []).append(row)
    keys = sorted(groups)
    total = len(rows)
    rng = random.Random(seed)
    quotas = {k: n * len(groups[k]) / total for k in keys}
    alloc = {k: int(quotas[k]) for k in keys}
    remaining = n - sum(alloc.values())
    tiebreak = {k: rng.random() for k in keys}
    order = sorted(keys, key=lambda k: (-(quotas[k] - alloc[k]), tiebreak[k]))
    for k in order[:remaining]:
        alloc[k] += 1

    picked: list[dict[str, Any]] = []
    for k in keys:
        members = sorted(groups[k], key=lambda r: str(r.get("id", "")))
        picked.extend(rng.sample(members, min(alloc[k], len(members))))
    rng.shuffle(picked)
    return picked


def stratum_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = source_stratum(row)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def sample_from_pool(
    pool: dict[str, list[dict]],
    suites: list[str],
    sample_per_suite: int,
    seed: int,
    seen: set[str] | None = None,
    allow_reseen: bool = False,
) -> list[dict]:
    """Sample unseen questions from a loaded pool, interleaved across suites.

    Shuffles the full suite list and picks the first ``sample_per_suite``
    unseen questions.  This guarantees we draw from the entire pool instead
    of a tiny 3x window that can overlap almost entirely with the seen set.

    If ``allow_reseen`` is True (debug mode), backfills with seen questions
    when unseen are exhausted.  In normal mode exhausted suites are skipped.
    """
    seen = seen or set()
    per_suite: list[list[dict]] = []

    for suite_name in suites:
        questions = pool.get(suite_name, [])
        if not questions:
            per_suite.append([])
            continue

        # Shuffle full suite, then take first N unseen
        rng = random.Random(seed)
        shuffled = list(questions)
        rng.shuffle(shuffled)

        fresh: list[dict] = []
        reseen: list[dict] = []
        for q in shuffled:
            if q.get("id", "") not in seen:
                fresh.append(q)
                if len(fresh) >= sample_per_suite:
                    break
            elif allow_reseen and len(reseen) < sample_per_suite:
                reseen.append(q)

        # Backfill with seen questions only in debug mode
        if allow_reseen and len(fresh) < sample_per_suite:
            need = sample_per_suite - len(fresh)
            fresh.extend(reseen[:need])

        per_suite.append(fresh)

    # Interleave round-robin
    all_prompts: list[dict] = []
    max_len = max((len(s) for s in per_suite), default=0)
    for i in range(max_len):
        for suite_questions in per_suite:
            if i < len(suite_questions):
                all_prompts.append(suite_questions[i])

    return all_prompts


def pool_header(pool_path: Path | None = None) -> dict | None:
    """Read just the header metadata from a pool file."""
    pool_path = pool_path or POOL_FILE
    if not pool_path.exists():
        return None
    with open(pool_path) as f:
        first_line = f.readline().strip()
        if first_line:
            try:
                obj = json.loads(first_line)
                if obj.get(_HEADER_KEY):
                    return obj
            except json.JSONDecodeError:
                pass
    return None


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Question pool management")
    parser.add_argument(
        "--build", action="store_true",
        help="Build/rebuild the question pool from all adapters",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print pool stats and exit",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=f"Output path (default: {POOL_FILE})",
    )
    parser.add_argument(
        "--refresh-suites", nargs="+", default=None, metavar="SUITE",
        help="Re-extract only these suites into a copy of the pool (needs --output "
             "unless --allow-live-overwrite)",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help=f"Pool to copy non-refreshed suites from (default: {POOL_FILE})",
    )
    parser.add_argument(
        "--allow-live-overwrite", action="store_true",
        help="Permit --refresh-suites to replace the live pool (instrument-era change)",
    )
    args = parser.parse_args()

    if args.refresh_suites:
        report = refresh_suites(
            args.refresh_suites,
            source_path=Path(args.source) if args.source else None,
            output_path=Path(args.output) if args.output else None,
            allow_live_overwrite=args.allow_live_overwrite,
        )
        print(json.dumps(report, indent=2))
        return

    if args.build:
        out = Path(args.output) if args.output else POOL_FILE
        t0 = time.monotonic()
        stats = build_pool(out)
        elapsed = time.monotonic() - t0
        total = sum(stats.values())
        print(f"\nPool built in {elapsed:.1f}s: {total} questions across {len(stats)} suites")
        for suite, count in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"  {suite:25s} {count:>6,d}")
        return

    if args.stats:
        header = pool_header()
        if header is None:
            print("No pool file found. Run with --build first.")
            return
        print(f"Generated: {header.get('generated_at', '?')}")
        print(f"Total:     {header.get('total_questions', '?')}")
        suites = header.get("suites", {})
        for suite, count in sorted(suites.items(), key=lambda x: -x[1]):
            print(f"  {suite:25s} {count:>6,d}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
