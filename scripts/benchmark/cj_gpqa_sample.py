#!/usr/bin/env python3
"""CJ-1d — deterministic GPQA-Diamond (CoT) sample for the canonical-judge revamp.

Handoff: ``epyc-root/handoffs/active/canonical-judge-suite-revamp.md`` → CJ-1d.
Design + n justification: ``docs/design/cj1-gpqa-sample-and-cj1e-gpu-pair.md``.

What this module is
-------------------
The CJ suites are not LLM-judged YAML prompts: they are executed by the
canonical ``v7_quality_gate_runner.py`` and scored by the canonical
``answer_scoring.score_response`` (CJ-1c). The runner already replays a pinned
item set through ``--questions-in``. This module builds that pinned manifest
for ``gpqa_diamond_cot`` (never the letter-only ``gpqa_diamond`` framing — it
suppresses reasoning) and stamps a ``sample`` identity block into it, which
``v7_quality_gate_beliefs`` carries into the belief-kernel projection.

Why the population is a pinned FILE, not a fresh adapter load
--------------------------------------------------------------
``GPQADiamondCoTAdapter.sample()`` needs ``datasets``/``pyarrow`` (only the
``ml-training`` venv has them) and draws by *positional* index into the HF
iteration order. Two independent pins (2026-07-20 architect bench and the
2026-08-25 EVL-08 attempt) are byte-identical in id order, prompts and gold,
so the population is stable; this module works from such a pin and selects
by **sorted content-hash id**, so the sample does not depend on file order or
on the HF dataset revision.

Question text is never committed to git (GPQA asks that items not be published
in plain text); the manifest is written next to the run artifacts.

Chosen n (CJ-1d): **198 — the full Diamond population**. The seeded-subset path
exists for the cold-start budget rule, but at the discordance rates measured on
this suite (~6–18% of items) a sign test at n=198 already has only ~0.59 power
for a 7pp paired gap; halving n drops it to ~0.30. See the design doc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

SUITE = "gpqa_diamond_cot"
POPULATION_SIZE = 198
DEFAULT_N = POPULATION_SIZE
DEFAULT_SEED = 42
SAMPLE_SCHEMA = "epyc.cj1.gpqa_sample.v1"
SELECTION_METHOD = "sorted_id+random.Random(seed).sample.v1"
COT_MARKER = "ANSWER: <letter>"
LETTER_ONLY_MARKER = "Answer with the letter only"
ID_PREFIX = "gpqa_diamond_cot_"
DEFAULT_POPULATION = Path(
    "/mnt/raid0/llm/epyc-inference-research/artifacts/"
    "architect-bench-gpu-20260720/questions_gpqa_diamond_cot.json"
)
#: Order-independent digest of the 198 Diamond CoT ids. Verified identical on
#: both on-disk pins (2026-07-20 architect bench, 2026-08-25 EVL-08). Enforced:
#: a pin whose id SET differs is a different population, hence a different
#: benchmark slice, and must not be scored under the same key.
KNOWN_POPULATION_IDS_SHA256 = (
    "381ba36533eb8971e50c18481c6fde69722a97d80762ee61fd8eb0e9fbb23194")


class SampleError(ValueError):
    """The population or request cannot produce an honest CJ-1d sample."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ids_digest(ids: Sequence[str]) -> str:
    """Order-independent digest of an item-id set."""
    return _sha256_bytes("\n".join(sorted(ids)).encode("utf-8"))


def validate_population(items: Sequence[Mapping[str, Any]]) -> None:
    """Refuse anything that is not the 198-item CoT-framed Diamond population."""
    if len(items) != POPULATION_SIZE:
        raise SampleError(
            f"population must hold {POPULATION_SIZE} items, got {len(items)}")
    ids = [it.get("id") for it in items]
    if len(set(ids)) != len(ids):
        raise SampleError("population ids are not unique")
    for it in items:
        iid = it.get("id")
        if not isinstance(iid, str) or not iid.startswith(ID_PREFIX):
            raise SampleError(f"item id {iid!r} is not a {SUITE} id")
        if it.get("suite") != SUITE:
            raise SampleError(f"{iid}: suite {it.get('suite')!r} != {SUITE!r}")
        if it.get("expected") not in {"A", "B", "C", "D"}:
            raise SampleError(f"{iid}: gold {it.get('expected')!r} is not A-D")
        if it.get("scoring_method") != "multiple_choice":
            raise SampleError(f"{iid}: scoring_method must be multiple_choice")
        prompt = it.get("prompt") or ""
        if COT_MARKER not in prompt or LETTER_ONLY_MARKER in prompt:
            raise SampleError(
                f"{iid}: prompt is not the CoT framing (CJ-1c: use "
                f"gpqa_diamond_cot, never the letter-only prompt)")


def load_population(path: Path, *, expected_ids_sha256: str | None = KNOWN_POPULATION_IDS_SHA256,
                    ) -> tuple[list[dict], str]:
    """Return (items, file sha256) from a runner-format pin.

    ``expected_ids_sha256=None`` disables the id-set pin (tests only)."""
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    items = doc["suites"][SUITE] if isinstance(doc, dict) and "suites" in doc else doc
    if not isinstance(items, list):
        raise SampleError(f"{path}: no {SUITE} item list")
    validate_population(items)
    if expected_ids_sha256 is not None:
        got = ids_digest([it["id"] for it in items])
        if got != expected_ids_sha256:
            raise SampleError(
                f"{path}: population id set {got} != pinned {expected_ids_sha256}")
    return [dict(it) for it in items], _sha256_bytes(raw)


def select_sample(items: Sequence[Mapping[str, Any]], n: int, seed: int) -> list[dict]:
    """Deterministic, file-order-independent selection.

    Items are sorted by id first; ``n == len(items)`` returns the whole
    population in sorted-id order (the seed then has no effect, and the
    identity block records that).
    """
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(items):
        raise SampleError(f"n must be an int in [1, {len(items)}], got {n!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SampleError(f"seed must be an int, got {seed!r}")
    ordered = sorted((dict(it) for it in items), key=lambda it: it["id"])
    if n == len(ordered):
        return ordered
    picked = random.Random(seed).sample(range(len(ordered)), n)
    return [ordered[i] for i in sorted(picked)]


def sample_identity(population: Sequence[Mapping[str, Any]], sample: Sequence[Mapping[str, Any]],
                    *, n: int, seed: int, population_path: Path,
                    population_sha256: str) -> dict:
    sample_ids = [it["id"] for it in sample]
    return {
        "schema": SAMPLE_SCHEMA,
        "suite": SUITE,
        "method": SELECTION_METHOD,
        "seed": seed,
        "seed_effective": n != len(population),
        "n": n,
        "n_population": len(population),
        "full_population": n == len(population),
        "population_path": str(population_path),
        "population_sha256": population_sha256,
        "population_ids_sha256": ids_digest([it["id"] for it in population]),
        "sample_ids_sha256": ids_digest(sample_ids),
        "tier_counts": dict(sorted(Counter(str(it.get("tier")) for it in sample).items())),
        "gold_counts": dict(sorted(Counter(it["expected"] for it in sample).items())),
    }


def build_manifest(population_path: Path, *, n: int = DEFAULT_N,
                   seed: int = DEFAULT_SEED,
                   expected_ids_sha256: str | None = KNOWN_POPULATION_IDS_SHA256) -> dict:
    population, pop_sha = load_population(
        population_path, expected_ids_sha256=expected_ids_sha256)
    sample = select_sample(population, n, seed)
    return {
        "suites": {SUITE: sample},
        "sample": sample_identity(population, sample, n=n, seed=seed,
                                  population_path=Path(population_path),
                                  population_sha256=pop_sha),
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", type=Path, required=True,
                   help="runner --questions-in manifest to write (keep it out of git)")
    args = p.parse_args(argv)
    try:
        manifest = build_manifest(args.population, n=args.n, seed=args.seed)
    except (SampleError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"[cj_gpqa_sample] REFUSED: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    ident = manifest["sample"]
    print(f"[cj_gpqa_sample] wrote {args.out}: n={ident['n']}/{ident['n_population']} "
          f"seed={ident['seed']} sample_ids_sha256={ident['sample_ids_sha256']}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
