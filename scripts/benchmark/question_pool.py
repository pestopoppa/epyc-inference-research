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
import random
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


def build_pool(output_path: Path | None = None) -> dict[str, int]:
    """Extract all questions from all adapters + YAML suites into a JSONL file.

    Returns dict mapping suite_name -> count of questions extracted.
    """
    from dataset_adapters import ADAPTER_SUITES, YAML_ONLY_SUITES, get_adapter

    output_path = output_path or POOL_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {}
    all_questions: list[dict] = []

    # 1. Extract from HF dataset adapters
    for suite_name in sorted(ADAPTER_SUITES):
        adapter = get_adapter(suite_name)
        if adapter is None:
            logger.warning(f"  [{suite_name}] No adapter found, skipping")
            continue
        try:
            questions = adapter.extract_all()
            # Ensure suite field is set
            for q in questions:
                q.setdefault("suite", suite_name)
                q.setdefault("dataset_source", "hf_adapter")
            stats[suite_name] = len(questions)
            all_questions.extend(questions)
            logger.info(f"  [{suite_name}] Extracted {len(questions)} questions")
        except Exception as e:
            logger.error(f"  [{suite_name}] Extraction failed: {e}")
            stats[suite_name] = 0

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
                        "scoring_method": q.get("scoring_method", "exact_match"),
                        "scoring_config": q.get("scoring_config", {}),
                        "dataset_source": "yaml",
                    })
                stats[suite_name] = len(converted)
                all_questions.extend(converted)
                logger.info(f"  [{suite_name}] Extracted {len(converted)} questions (YAML)")
            except Exception as e:
                logger.error(f"  [{suite_name}] YAML extraction failed: {e}")
                stats[suite_name] = 0

    # 3. Write JSONL with header
    header = {
        _HEADER_KEY: True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "question_pool.py",
        "total_questions": len(all_questions),
        "suites": stats,
    }

    with open(output_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    logger.info(f"Pool written: {output_path} ({len(all_questions)} questions)")
    return stats


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
                if warn_stale:
                    _check_staleness(obj)
                continue

            suite = obj.get("suite", "unknown")
            pool.setdefault(suite, []).append(obj)

    if not header_seen:
        logger.warning("Pool file has no header — consider rebuilding with --rebuild-pool")

    return pool


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
    args = parser.parse_args()

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
