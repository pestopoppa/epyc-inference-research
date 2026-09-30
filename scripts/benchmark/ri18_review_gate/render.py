"""Freeze the RI-18 workload: S1 (pilot pool, re-rendered) + S2 (olympiadbench_hard).

S1 = the UFH-13 pilot pool (453 items, GPQA 253 + MMLU-Pro 200), loaded through
``thesis_ufh13.pilot_pool.load_pool`` (sha-pinned, refuses any frozen-suite question) and
re-rendered as ``ri18-brief-justify-v1``: the final letter-only instruction line is replaced by
``RENDER_LINE``. The rewrite is asserted on EVERY item, and the frozen 395 are refused again on
the rendered prompts (stem hash and full-prompt hash) and by id.

S2 = ``olympiadbench_hard`` (155 items, ``math_symbolic``), copied verbatim from the
architect-bench question file (sha-pinned; the file is untracked in git, so its bytes are frozen
into ``items.json`` here).

Outputs (committed): ``data/ri18-review-gate/items.json`` and ``manifest.json``. The manifest pins
the sha256 of the rendered file and of each source, and freezes the seeded (``SEED=18``) 50/50
tune(A)/confirm(B) split stratified by suite plus the seeded 50-item verdict-noise subset.

    python3 -m scripts.benchmark.ri18_review_gate.render            # from the research repo root
    python3 -m scripts.benchmark.ri18_review_gate.render --check    # verify the committed files
No network, no inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "scripts.benchmark.ri18_review_gate"

from ..thesis_ufh13.pilot_pool import (  # noqa: E402
    POOL_PATH,
    POOL_SHA256,
    frozen_hashes,
    load_pool,
    refuse_frozen,
)
from ..thesis_ufh13.suite import SUITE_SHA256, Item, file_sha256, load_suite  # noqa: E402

RESEARCH_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = RESEARCH_ROOT / "data/ri18-review-gate"
ITEMS_PATH = OUT_DIR / "items.json"
MANIFEST_PATH = OUT_DIR / "manifest.json"

RENDER = "ri18-brief-justify-v1"
RENDER_LINE = (
    "Justify briefly (at most three sentences), then end with a final line `Answer: <letter>`."
)
# The pilot pool's final lines (byte-exact, per source suite). Anything else is refused.
LETTER_ONLY_LINES = {
    "mmlu_pro": "Answer with the letter only (A through J).",
    "gpqa": "Answer with the letter only (A, B, C, or D).",
}

S2_SOURCE = Path(
    "/mnt/raid0/llm/epyc-inference-research/artifacts/architect-bench-gpu-20260720/"
    "questions_olympiadbench_hard.json"
)
S2_SHA256 = "7140b7eadfa0a91338cc164cc3a15d65fb5fa4be4a2371240c4dd524c535a5d4"
S2_SUITE = "olympiadbench_hard"
S2_N = 155

ITEMS_SCHEMA = "ri18-review-gate-items/v1"
MANIFEST_SCHEMA = "ri18-review-gate-workload/v1"
SEED = 18
NOISE_N = 50
STRATUM = {"gpqa": "S1", "mmlu_pro": "S1", S2_SUITE: "S2"}


class RenderError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rewrite_last_line(item: Item) -> Item:
    """Replace the final letter-only line; raise unless it is exactly the expected line."""
    head, sep, last = item.prompt.rpartition("\n")
    want = LETTER_ONLY_LINES.get(item.suite)
    if not sep or want is None or last != want:
        raise RenderError(f"{item.item_id}: final line {last!r} is not the letter-only line")
    return Item(item.item_id, item.suite, head + "\n" + RENDER_LINE, item.expected)


def render_s1(pool: list[Item], frozen: list[Item]) -> list[Item]:
    guard = frozen_hashes(frozen)
    refuse_frozen(pool, guard)
    rendered = [rewrite_last_line(item) for item in pool]
    # Asserted on every item: the rewrite happened, nothing else moved.
    for old, new in zip(pool, rendered):
        if not new.prompt.endswith("\n" + RENDER_LINE):
            raise RenderError(f"{new.item_id}: rewrite missing")
        if new.prompt.rpartition("\n")[0] != old.prompt.rpartition("\n")[0]:
            raise RenderError(f"{new.item_id}: body changed by the rewrite")
    refuse_frozen(rendered, guard)  # the stem hash still catches a frozen question
    frozen_ids = {item.item_id for item in frozen}
    overlap = sorted(frozen_ids & {item.item_id for item in rendered})
    if overlap:
        raise RenderError(f"{len(overlap)} frozen-suite ids in S1: {overlap[:5]}")
    return rendered


def load_s2(path: Path = S2_SOURCE, expected_sha256: str = S2_SHA256) -> list[dict[str, Any]]:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise RenderError(f"{path}: sha256 {actual} != pinned {expected_sha256}")
    rows = json.loads(path.read_text())["suites"][S2_SUITE]
    if len(rows) != S2_N:
        raise RenderError(f"{S2_SUITE}: {len(rows)} items, expected {S2_N}")
    for row in rows:
        if row.get("scoring_method") != "math_symbolic" or row.get("image_path"):
            raise RenderError(f"{row['id']}: not a text-only math_symbolic item")
    return rows


def build_items(pool: list[Item], frozen: list[Item], s2_rows: list[dict[str, Any]]
                ) -> list[dict[str, Any]]:
    by_id = {item.item_id: item for item in pool}
    out: list[dict[str, Any]] = []
    for item in render_s1(pool, frozen):
        out.append({
            "id": item.item_id, "suite": item.suite, "stratum": "S1", "prompt": item.prompt,
            "expected": item.expected, "scoring_method": "multiple_choice",
            "scoring_config": {},
            "source_prompt_sha256": sha256_bytes(by_id[item.item_id].prompt.encode()),
        })
    for row in s2_rows:
        out.append({
            "id": row["id"], "suite": S2_SUITE, "stratum": "S2", "prompt": row["prompt"],
            "expected": row["expected"], "scoring_method": "math_symbolic",
            "scoring_config": row.get("scoring_config") or {},
            "source_prompt_sha256": sha256_bytes(row["prompt"].encode()),
        })
    ids = [row["id"] for row in out]
    if len(set(ids)) != len(ids):
        raise RenderError("duplicate item ids")
    return out


def split_and_noise(items: list[dict[str, Any]], seed: int = SEED, noise_n: int = NOISE_N
                    ) -> dict[str, Any]:
    """Seeded 50/50 tune(A)/confirm(B) split stratified by suite, then the noise subset.

    Per suite (sorted): ids sorted, shuffled with ONE ``random.Random(seed)`` stream, the first
    ``n // 2`` are A. The noise subset draws ``noise_n`` ids from the same stream afterwards,
    allocated by suite in proportion (largest remainder), sorted within the draw.
    """
    rng = random.Random(seed)
    by_suite: dict[str, list[str]] = {}
    for row in items:
        by_suite.setdefault(row["suite"], []).append(row["id"])
    split: dict[str, str] = {}
    for suite in sorted(by_suite):
        ids = sorted(by_suite[suite])
        rng.shuffle(ids)
        half = len(ids) // 2
        for i, item_id in enumerate(ids):
            split[item_id] = "A" if i < half else "B"
    total = len(items)
    raw = {s: noise_n * len(v) / total for s, v in by_suite.items()}
    quota = {s: int(v) for s, v in raw.items()}
    for s in sorted(raw, key=lambda k: (raw[k] - quota[k], k), reverse=True)[
            : noise_n - sum(quota.values())]:
        quota[s] += 1
    noise: list[str] = []
    for suite in sorted(by_suite):
        noise += sorted(rng.sample(sorted(by_suite[suite]), quota[suite]))
    order = [row["id"] for row in items]
    return {
        "split": {k: [i for i in order if split[i] == k] for k in ("A", "B")},
        "noise_subset": [i for i in order if i in set(noise)],
        "method": (f"random.Random({seed}); per suite (sorted) shuffle sorted ids, first n//2 = A; "
                   f"then noise: {noise_n} ids by suite, largest-remainder quota, "
                   "rng.sample(sorted ids) from the same stream"),
    }


def build(pool_path: Path = POOL_PATH, s2_path: Path = S2_SOURCE) -> tuple[bytes, dict[str, Any]]:
    pool, pool_sha = load_pool(pool_path)
    frozen = load_suite()
    items = build_items(pool, frozen, load_s2(s2_path))
    items_bytes = (json.dumps({"schema": ITEMS_SCHEMA, "render": RENDER, "items": items},
                              indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode()
    sel = split_and_noise(items)
    counts = Counter((row["stratum"], row["suite"]) for row in items)
    split_counts = {
        k: dict(Counter(row["suite"] for row in items if row["id"] in set(v)))
        for k, v in sel["split"].items()
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "render": RENDER,
        "render_line": RENDER_LINE,
        "replaced_lines": LETTER_ONLY_LINES,
        "items_path": str(ITEMS_PATH.relative_to(RESEARCH_ROOT)),
        "items_sha256": sha256_bytes(items_bytes),
        "n_items": len(items),
        "counts": {f"{s}/{suite}": n for (s, suite), n in sorted(counts.items())},
        "sources": {
            "S1_pilot_pool": {"path": str(pool_path.relative_to(RESEARCH_ROOT)),
                              "sha256": pool_sha, "pinned_sha256": POOL_SHA256},
            "S2_olympiadbench_hard": {"path": str(s2_path), "sha256": file_sha256(s2_path),
                                      "pinned_sha256": S2_SHA256},
            "frozen_suite_sha256": SUITE_SHA256,
        },
        "frozen_excluded": {"frozen_items": len(frozen), "check": "load_pool+refuse_frozen on "
                            "source AND rendered prompts; id overlap asserted empty"},
        "seed": SEED,
        "split_method": sel["method"],
        "split": sel["split"],
        "split_counts": split_counts,
        "noise_subset": sel["noise_subset"],
        "noise_subset_counts": dict(Counter(
            row["suite"] for row in items if row["id"] in set(sel["noise_subset"]))),
        "scorers": {"S1": "answer_scoring.extract_letter_answer (every arm)",
                    "S2": "answer_scoring.score_response -> math_symbolic"},
    }
    return items_bytes, manifest


def write(out_dir: Path = OUT_DIR) -> dict[str, Any]:
    items_bytes, manifest = build()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ITEMS_PATH.name).write_bytes(items_bytes)
    (out_dir / MANIFEST_PATH.name).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


# ── runtime (the driver) ─────────────────────────────────────────────────────


def load_workload(items_path: Path = ITEMS_PATH, manifest_path: Path = MANIFEST_PATH
                  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The frozen items and manifest; refuses a drifted items file."""
    manifest = json.loads(manifest_path.read_text())
    data = items_path.read_bytes()
    if sha256_bytes(data) != manifest["items_sha256"]:
        raise RenderError(f"{items_path}: sha256 drifted from the frozen manifest")
    payload = json.loads(data)
    if payload.get("schema") != ITEMS_SCHEMA or payload.get("render") != RENDER:
        raise RenderError(f"{items_path}: wrong schema/render")
    return payload["items"], manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory and compare with the committed files")
    args = ap.parse_args(argv)
    if args.check:
        items_bytes, manifest = build()
        ok = (ITEMS_PATH.read_bytes() == items_bytes
              and json.loads(MANIFEST_PATH.read_text()) == manifest)
        print("frozen workload reproduces byte-identically" if ok else "DRIFT")
        return 0 if ok else 1
    manifest = write()
    print(json.dumps({k: manifest[k] for k in ("items_sha256", "n_items", "counts",
                                               "split_counts", "noise_subset_counts")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
