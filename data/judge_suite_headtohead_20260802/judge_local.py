#!/usr/bin/env python3
"""Score N captured arms with ONE local judge, so the arms are comparable to each other.

WHY THIS EXISTS
---------------
The 2026-05-04 anchor (frontdoor 170/183) was scored by Claude-as-Judge. The
2026-08-02 architect_general capture was scored the same way. Scoring a SECOND
arm with a DIFFERENT judge would make the two numbers incomparable — the thing
we are trying to measure (arm A vs arm B) would be confounded with judge identity.

So this scores EVERY arm with ONE judge, in one pass, under one rubric. The
resulting contrast is internally valid regardless of how it relates to 170/183.
It is NOT a substitute for the Claude-as-Judge anchor and must be reported as a
different instrument.

JUDGE INDEPENDENCE
------------------
The judge is architect_critic (Qwen3.5-122B-A10B, :8074) — deliberately not
either arm, so no arm grades itself.

BLINDING
--------
The judge never sees which arm produced a response, and the per-question order
of arms is shuffled on a fixed seed, so a positional preference cannot
systematically favour one arm.

RUBRIC CONTAMINATION — KNOWN, DELIBERATELY PRESERVED
----------------------------------------------------
rubric_system_prompt carries "calibration examples" that name specific
question_ids together with the score they received on other models, e.g.
  math/t3_q2_combinatorics -> listed under "Score 1 examples"
  coder/t1_q1_algorithm    -> listed under "Score 0 examples"
That primes the judge on identity before it reads the answer. It is preserved
VERBATIM here because it is the instrument the anchor used, and because both
arms receive it identically, so it cannot bias the A-vs-B contrast. It DOES bias
the absolute level, and any absolute number from this harness inherits that.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

JUDGE_URL = "http://127.0.0.1:8074/v1/chat/completions"
PACKET_DIR = Path("/mnt/raid0/llm/tmp/judge-suite-27b/run-20260802/judge_packets")
SEED = 42


def post(url: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def rubric_for(suite: str) -> str:
    p = PACKET_DIR / f"{suite}.json"
    if not p.exists():
        raise SystemExit(f"no judge packet for suite {suite!r} at {p}; refusing to invent a rubric")
    return json.loads(p.read_text())["rubric_system_prompt"]


def read_capture(path: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows[(r["suite"], r["question_id"])] = r
    return rows


def judge_one(rubric: str, item: dict, response: str) -> tuple[int, str]:
    """Score ONE response. Returns (score, reason); score -1 == ineligible."""
    user = (
        f"## Question ({item['suite']}/{item['question_id']}, tier {item.get('tier')})\n\n"
        f"{item['prompt']}\n\n"
        f"## Response to score\n\n{response}\n"
    )
    body = {
        "messages": [
            {"role": "system", "content": rubric},
            {"role": "user", "content": user},
        ],
        "max_tokens": 400,
        "temperature": 0,
        "seed": SEED,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        d = post(JUDGE_URL, body)
    except Exception as exc:  # noqa: BLE001
        return -1, f"judge_transport:{type(exc).__name__}"
    txt = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return -1, "judge_parse: no JSON object in judge output"
    try:
        obj = json.loads(m.group(0))
        score = int(obj["score"])
    except Exception:  # noqa: BLE001
        return -1, "judge_parse: JSON present but no integer score"
    if score not in (0, 1, 2, 3):
        return -1, f"judge_range: score={score} outside 0-3"
    return score, str(obj.get("reason", ""))[:500]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", required=True,
                    help="NAME=/path/to/capture.jsonl (repeatable)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        arms[name] = read_capture(Path(path))

    # Only questions captured by EVERY arm are scored. A question one arm
    # missed cannot contribute to a paired contrast, and averaging over
    # different question sets is the classic way to fake a difference.
    common = set.intersection(*(set(a) for a in arms.values()))
    dropped = {n: sorted(set(a) - common) for n, a in arms.items()}
    keys = sorted(common)
    if args.limit:
        keys = keys[: args.limit]
    print(f"arms={list(arms)}  common questions={len(keys)}")
    for n, d in dropped.items():
        if d:
            print(f"  NOT scored (absent from another arm) for {n}: {len(d)} -> {d[:5]}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    results = {n: [] for n in arms}
    rubrics: dict[str, str] = {}
    t0 = time.time()

    for i, key in enumerate(keys, 1):
        suite, qid = key
        if suite not in rubrics:
            rubrics[suite] = rubric_for(suite)
        order = list(arms)
        rng.shuffle(order)  # blind + positional-bias control
        for name in order:
            row = arms[name][key]
            score, reason = judge_one(rubrics[suite], row, row.get("response") or "")
            results[name].append({
                "suite": suite,
                "question_id": qid,
                "tokens_per_second": row.get("tokens_per_second"),
                "claude_score": score,
                "score_reason": reason,
                "finish_reason": row.get("finish_reason"),
                "failure_class": row.get("failure_class"),
            })
        if i % 10 == 0 or i == len(keys):
            tally = "  ".join(
                f"{n} {sum(r['claude_score'] for r in results[n] if r['claude_score'] >= 0)}"
                for n in arms
            )
            print(f"  {i}/{len(keys)}  {tally}  {int(time.time() - t0)}s", flush=True)

    summary = {"judge": "Qwen3.5-122B-A10B :8074", "seed": SEED,
               "questions_scored": len(keys), "arms": {}}
    for name, rows in results.items():
        with open(outdir / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        elig = [r for r in rows if r["claude_score"] >= 0]
        by_suite: dict[str, list[int]] = {}
        for r in elig:
            by_suite.setdefault(r["suite"], []).append(r["claude_score"])
        summary["arms"][name] = {
            "eligible": len(elig),
            "ineligible": len(rows) - len(elig),
            "score": sum(r["claude_score"] for r in elig),
            "max": 3 * len(elig),
            "pct": round(100.0 * sum(r["claude_score"] for r in elig) / max(1, 3 * len(elig)), 1),
            "by_suite": {s: [sum(v), 3 * len(v)] for s, v in sorted(by_suite.items())},
        }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary["arms"], indent=2))
    print(f"\nwrote {outdir}  ({int(time.time() - t0)}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
