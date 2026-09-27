"""The frozen UFH-13 suite: MMLU-Pro 200 + GPQA 195, sha256-pinned.

epyc-root handoff ``handoffs/active/thesis-experiment-orchestrator-vs-strongest-model.md``
(Suite, frozen). The file is the manifest baseline 1 (Flash-Next alone) was scored on. It is
loaded read-only and REFUSED if its bytes moved: a changed item set is a different experiment.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[3]
SUITE_PATH = (
    RESEARCH_ROOT
    / "data/kernel-v8-candidate/quality-gate/run-20260725T204443Z-fullcontract-both-mode/questions.json"
)
SUITE_SHA256 = "1532906b4a754673937027e73e2023d8eee7ed5d08f084c207a60ac81460adb1"
SUITES = ("mmlu_pro", "gpqa")
EXPECTED_COUNTS = {"mmlu_pro": 200, "gpqa": 195}


class SuiteError(RuntimeError):
    """The suite on disk is not the frozen one."""


@dataclass(frozen=True)
class Item:
    item_id: str
    suite: str
    prompt: str
    expected: str

    @property
    def stratum(self) -> str:
        """Subject within the suite (``mmlu_pro_law`` -> ``law``), for stratified sampling."""
        prefix = f"{self.suite}_"
        body = self.item_id[len(prefix):] if self.item_id.startswith(prefix) else self.item_id
        return body.rsplit("_", 1)[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_suite(path: Path = SUITE_PATH, expected_sha256: str = SUITE_SHA256) -> list[Item]:
    """All items in manifest order. Raises ``SuiteError`` on any drift."""
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise SuiteError(f"{path}: sha256 {actual} != frozen {expected_sha256}")
    data = json.loads(path.read_text())["suites"]
    items: list[Item] = []
    for suite in SUITES:
        rows = data[suite]
        if len(rows) != EXPECTED_COUNTS[suite]:
            raise SuiteError(f"{suite}: {len(rows)} items, expected {EXPECTED_COUNTS[suite]}")
        for row in rows:
            if row.get("scoring_method") != "multiple_choice":
                raise SuiteError(f"{row['id']}: scoring_method {row.get('scoring_method')!r}")
            items.append(Item(row["id"], suite, row["prompt"], str(row["expected"]).strip().upper()))
    if len({item.item_id for item in items}) != len(items):
        raise SuiteError("duplicate item ids")
    return items


def stratified_sample(items: list[Item], n: int, seed: int) -> list[Item]:
    """``n`` items spread proportionally over suites, then round-robin over subjects.

    Deterministic for a given (items, n, seed). Used by the pilot only.
    """
    if n <= 0:
        return []
    n = min(n, len(items))
    rng = random.Random(seed)
    by_suite: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        by_suite[item.suite].append(item)
    total = len(items)
    quotas = {suite: round(n * len(rows) / total) for suite, rows in by_suite.items()}
    while sum(quotas.values()) > n:
        quotas[max(quotas, key=quotas.get)] -= 1
    while sum(quotas.values()) < n:
        quotas[max(by_suite, key=lambda s: len(by_suite[s]) - quotas[s])] += 1
    picked: list[Item] = []
    for suite in sorted(by_suite):
        strata: dict[str, list[Item]] = defaultdict(list)
        for item in by_suite[suite]:
            strata[item.stratum].append(item)
        pools = [rng.sample(strata[key], len(strata[key])) for key in sorted(strata)]
        rng.shuffle(pools)
        taken: list[Item] = []
        while len(taken) < quotas[suite] and any(pools):
            for pool in pools:
                if pool and len(taken) < quotas[suite]:
                    taken.append(pool.pop())
        picked.extend(taken)
    return picked
