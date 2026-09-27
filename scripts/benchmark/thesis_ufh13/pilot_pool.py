"""The pilot pool: MC items from OUTSIDE the frozen 395, so the pilot never sees a scored item.

Sources (both already on disk; nothing is downloaded):

* MMLU-Pro test split, the pinned snapshot ``dataset_adapters.MMLUProAdapter`` reads
  (12,032 rows). The frozen suite's 200 are rows of this file.
* GPQA ``ankner/gpqa`` train (the 448-question main set), which ``GPQAAdapter`` reads. The frozen
  suite's 195 GPQA items are rows of THIS file (a main-set sample; 78 of them are also diamond
  questions). The other 253 (121 also in diamond, 132 main-only) are the only non-suite GPQA on
  disk: the diamond mirror is a subset of main, and no extended split is on disk.

Items are rendered byte-for-byte as the adapters render them. ``build_pool`` proves that by
re-rendering every frozen item from its source row and comparing it with the frozen file; a
mismatch aborts the build. It then excludes the frozen rows and refuses any candidate whose
question hash (normalised stem) or full-prompt hash is in the frozen suite.

The pool is built ONCE (needs pyarrow: run under the research venv) and committed as
``data/ufh13-thesis/pilot_pool.json``; its sha256 is pinned here, so the runner needs no pyarrow
and the pilot's item population is fixed.

MMLU-Pro: 200 items with EXACTLY the frozen suite's category counts. GPQA: all 253 remaining
main-set items; the frozen mix cannot be matched (e.g. every Molecular Biology item of main is
already in the suite), so the pilot sampler allocates by the frozen mix where availability allows.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .suite import RESEARCH_ROOT, SUITE_SHA256, Item, load_suite

MMLU_PRO_PARQUET = Path(
    "/mnt/raid0/llm/cache/huggingface/hub/datasets--TIGER-Lab--MMLU-Pro/snapshots/"
    "b189ec765aa7ed75c8acfea42df31fdae71f97be/data/test-00000-of-00001.parquet"
)
MMLU_PRO_SHA256 = "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"
GPQA_PARQUET = Path(
    "/mnt/raid0/llm/hf-home/hub/datasets--ankner--gpqa/snapshots/"
    "8c23b9cbb7871e81172ef3b36d8903fa3a9b84a1/data/train-00000-of-00001.parquet"
)
GPQA_SHA256 = "f6a75ab58a77dc56c1db56b533dc74d8d6c577172aa1e9cd461f0a5075881015"

POOL_PATH = RESEARCH_ROOT / "data/ufh13-thesis/pilot_pool.json"
POOL_SCHEMA = "ufh13-pilot-pool/v1"
POOL_SEED = 20260927
# Pinned after the one build; a drifted pool is refused.
POOL_SHA256 = "0898013e86b365d89cf941205e19e3843d4d3cd21f869d4db917496f030aa3c5"

MMLU_LABELS = "ABCDEFGHIJ"
GPQA_LABELS = "ABCD"


class PoolError(RuntimeError):
    pass


# ── rendering: byte-identical to dataset_adapters.py ─────────────────────────


def render_mmlu_pro(idx: int, row: dict[str, Any]) -> Item:
    options = list(row["options"])
    answer_index = row["answer_index"]
    if not 0 <= answer_index < len(options):
        raise PoolError(f"mmlu_pro row {idx}: answer_index out of range")
    letter = str(row.get("answer") or "").strip().upper()
    if letter and letter != MMLU_LABELS[answer_index]:
        raise PoolError(f"mmlu_pro row {idx}: answer/answer_index disagree")
    lines = [row["question"], ""]
    lines += [f"{MMLU_LABELS[i]}) {opt}" for i, opt in enumerate(options)]
    lines += ["", "Answer with the letter only (A through J)."]
    category = row.get("category", "other")
    return Item(f"mmlu_pro_{category}_{idx:05d}", "mmlu_pro", "\n".join(lines),
                MMLU_LABELS[answer_index])


def render_gpqa(idx: int, row: dict[str, Any]) -> Item:
    question = row.get("Question", "")
    correct = row.get("Correct Answer", "")
    choices = [c for c in (correct, row.get("Incorrect Answer 1", ""),
                           row.get("Incorrect Answer 2", ""), row.get("Incorrect Answer 3", ""))
               if c]
    rng = random.Random(int(hashlib.sha256(question.encode()).hexdigest()[:8], 16))
    rng.shuffle(choices)
    expected = GPQA_LABELS[choices.index(correct) if correct in choices else 0]
    lines = [question, ""]
    lines += [f"{GPQA_LABELS[i]}) {choice}" for i, choice in enumerate(choices[:4])]
    lines += ["", "Answer with the letter only (A, B, C, or D)."]
    subdomain = row.get("Subdomain", "general")
    return Item(f"gpqa_{subdomain}_{idx:04d}", "gpqa", "\n".join(lines), expected)


# ── the question-hash guard ──────────────────────────────────────────────────


def question_stem(prompt: str) -> str:
    """The question text of a rendered MC prompt (everything before the first option line)."""
    stem = prompt.split("\n\nA) ", 1)[0]
    return re.sub(r"\s+", " ", stem).strip().casefold()


def question_hashes(item: Item) -> tuple[str, str]:
    """(stem hash, full-prompt hash). A collision on EITHER means "the same question"."""
    return (hashlib.sha256(question_stem(item.prompt).encode()).hexdigest(),
            hashlib.sha256(item.prompt.encode()).hexdigest())


def frozen_hashes(frozen: list[Item]) -> set[str]:
    return {h for item in frozen for h in question_hashes(item)}


def refuse_frozen(items: list[Item], frozen: set[str]) -> None:
    """Raise if ANY item's question is in the frozen suite (by stem or full-prompt hash)."""
    hits = [item.item_id for item in items if set(question_hashes(item)) & frozen]
    if hits:
        raise PoolError(f"{len(hits)} pilot item(s) are frozen-suite questions: {hits[:5]}")


# ── build (research venv; once) ──────────────────────────────────────────────


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_parquet(path: Path, expected_sha: str) -> list[dict[str, Any]]:
    actual = _sha256(path)
    if actual != expected_sha:
        raise PoolError(f"{path}: sha256 {actual} != pinned {expected_sha}")
    import pyarrow.parquet as pq  # research venv only

    return pq.read_table(path).to_pylist()


def build_pool(mmlu_rows: list[dict[str, Any]], gpqa_rows: list[dict[str, Any]],
               frozen: list[Item], seed: int = POOL_SEED) -> dict[str, Any]:
    by_id = {item.item_id: item for item in frozen}
    rendered_mmlu = [render_mmlu_pro(i, r) for i, r in enumerate(mmlu_rows)]
    rendered_gpqa = [render_gpqa(i, r) for i, r in enumerate(gpqa_rows)]
    # Provenance proof: every frozen item re-renders byte-identically from its source row.
    frozen_rows = {it.item_id: it for it in rendered_mmlu + rendered_gpqa if it.item_id in by_id}
    missing = sorted(set(by_id) - set(frozen_rows))
    mismatched = sorted(i for i, it in frozen_rows.items() if it != by_id[i])
    if missing or mismatched:
        raise PoolError(f"renderer/source check failed: missing={missing[:3]} "
                        f"mismatched={mismatched[:3]}")
    guard = frozen_hashes(frozen)

    def eligible(items: list[Item]) -> list[Item]:
        seen: set[str] = set()
        out = []
        for item in items:
            hashes = set(question_hashes(item))
            if item.item_id in by_id or hashes & guard or hashes & seen:
                continue  # frozen, a frozen question under another id, or a duplicate
            seen |= hashes
            out.append(item)
        return out

    mmlu_ok = eligible(rendered_mmlu)
    gpqa_ok = eligible(rendered_gpqa)
    rng = random.Random(seed)
    target = Counter(item.stratum for item in frozen if item.suite == "mmlu_pro")
    by_cat: dict[str, list[Item]] = defaultdict(list)
    for item in mmlu_ok:
        by_cat[item.stratum].append(item)
    mmlu_pool: list[Item] = []
    for category in sorted(target):
        if len(by_cat[category]) < target[category]:
            raise PoolError(f"mmlu_pro {category}: only {len(by_cat[category])} non-suite items")
        mmlu_pool += rng.sample(by_cat[category], target[category])
    pool = sorted(mmlu_pool, key=lambda i: i.item_id) + sorted(gpqa_ok, key=lambda i: i.item_id)
    refuse_frozen(pool, guard)
    return {
        "schema": POOL_SCHEMA,
        "seed": seed,
        "frozen_suite_sha256": SUITE_SHA256,
        "sources": {"mmlu_pro": {"path": str(MMLU_PRO_PARQUET), "sha256": MMLU_PRO_SHA256,
                                 "rows": len(mmlu_rows), "eligible": len(mmlu_ok)},
                    "gpqa": {"path": str(GPQA_PARQUET), "sha256": GPQA_SHA256,
                             "split": "ankner/gpqa train (GPQA main, 448)",
                             "rows": len(gpqa_rows), "eligible": len(gpqa_ok)}},
        "renderer_verified_frozen_items": len(frozen_rows),
        "composition": composition(pool),
        "items": [{"id": i.item_id, "suite": i.suite, "prompt": i.prompt, "expected": i.expected}
                  for i in pool],
    }


def composition(items: list[Item]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for suite in sorted({i.suite for i in items}):
        rows = [i for i in items if i.suite == suite]
        out[suite] = {"n": len(rows), "by_subject": dict(sorted(Counter(i.stratum for i in rows).items()))}
    return out


def write_pool(path: Path = POOL_PATH) -> str:
    """Build from the on-disk sources and write the pool file; returns its sha256."""
    pool = build_pool(_read_parquet(MMLU_PRO_PARQUET, MMLU_PRO_SHA256),
                      _read_parquet(GPQA_PARQUET, GPQA_SHA256), load_suite())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    return _sha256(path)


# ── runtime (no pyarrow) ─────────────────────────────────────────────────────


def load_pool(path: Path = POOL_PATH, expected_sha256: str | None = None) -> tuple[list[Item], str]:
    """The pool items and the pool file's sha256. Refuses drift and any frozen question."""
    actual = _sha256(path)
    pinned = POOL_SHA256 if expected_sha256 is None else expected_sha256
    if actual != pinned:
        raise PoolError(f"{path}: sha256 {actual} != pinned {pinned}")
    data = json.loads(path.read_text())
    if data.get("schema") != POOL_SCHEMA or data.get("frozen_suite_sha256") != SUITE_SHA256:
        raise PoolError(f"{path}: wrong schema or built against another frozen suite")
    items = [Item(r["id"], r["suite"], r["prompt"], r["expected"]) for r in data["items"]]
    refuse_frozen(items, frozen_hashes(load_suite()))
    return items, actual


def mix_matched_sample(pool: list[Item], frozen: list[Item], n: int, seed: int) -> list[Item]:
    """``n`` pool items allocated by the FROZEN suite's mix: suites in proportion, then subjects
    in proportion within each suite (largest remainder), capped by what the pool has; any
    shortfall goes to the next-largest frozen subjects, then to other subjects. Deterministic."""
    n = max(0, min(n, len(pool)))
    rng = random.Random(seed)
    pool_by: dict[str, dict[str, list[Item]]] = defaultdict(lambda: defaultdict(list))
    for item in pool:
        pool_by[item.suite][item.stratum].append(item)
    frozen_suite = Counter(i.suite for i in frozen)
    suite_quota = _largest_remainder(frozen_suite, n)
    picked: list[Item] = []
    for suite in sorted(suite_quota):
        want = min(suite_quota[suite], sum(len(v) for v in pool_by[suite].values()))
        mix = Counter(i.stratum for i in frozen if i.suite == suite)
        quota = _largest_remainder(mix, want)
        take = {s: min(q, len(pool_by[suite][s])) for s, q in quota.items()}
        order = [s for s, _ in mix.most_common()] + sorted(set(pool_by[suite]) - set(mix))
        while sum(take.values()) < want:
            for subject in order:
                if sum(take.values()) >= want:
                    break
                if take.get(subject, 0) < len(pool_by[suite][subject]):
                    take[subject] = take.get(subject, 0) + 1
        for subject in sorted(take):
            picked += rng.sample(sorted(pool_by[suite][subject], key=lambda i: i.item_id),
                                 take[subject])
    return picked


def _largest_remainder(weights: Counter, n: int) -> dict[str, int]:
    total = sum(weights.values())
    if not total or n <= 0:
        return {k: 0 for k in weights}
    raw = {k: n * w / total for k, w in weights.items()}
    quota = {k: int(v) for k, v in raw.items()}
    for key in sorted(raw, key=lambda k: (raw[k] - quota[k], weights[k], k),
                      reverse=True)[: n - sum(quota.values())]:
        quota[key] += 1
    return quota


if __name__ == "__main__":  # python -m scripts.benchmark.thesis_ufh13.pilot_pool (research venv)
    print(write_pool())
