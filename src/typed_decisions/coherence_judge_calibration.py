"""LIGHT calibration for the tier-2 coherence judge (2026-10-04). Built, NOT run.

Scope (operator, 2026-10-04): the mechanism is TD-29 native single-token scoring, which is
already validated. This is not an experiment on the mechanism. It is a cheap sanity check
of the judge on THIS task: one small labelled set, run once. It records one accuracy
number and seals a ``calibration_id``. The judge's "refuse uncalibrated" rule needs that
id.

``build`` makes a deterministic, seeded set of about 32 labelled pairs with no model calls:

  ===========================  ====================  =====  ===============================
  category                     gold                  n      construction
  ===========================  ====================  =====  ===============================
  ``loop``                     INCOHERENT            4      head of a real base output, then
                                                            one of its phrases repeated
  ``salad``                    INCOHERENT            4      seeded random words and symbols
  ``shuffled``                 INCOHERENT            4      the base output's words, permuted
  ``off_task``                 OFF_TASK              3      another prompt's base output
  ``coherent_long_reasoning``  COHERENT_EQUIVALENT   4      short correct answer vs a long
                                                            correct derivation (both orders)
  ``correct_alternative``      COHERENT_EQUIVALENT   3      two correct answers that use
                                                            different methods
  ``wrong_alternative``        DEGRADED              4      a fluent derivation with one
                                                            arithmetic slip
  ``mtp_diverged``             COHERENT_EQUIVALENT   6      REAL INF-70 speed-claim pairs:
                                                            plain arm vs an MTP arm whose
                                                            greedy text diverged, both rows
                                                            classified COHERENT by reason
  ===========================  ====================  =====  ===============================

The ``mtp_diverged`` label is an assumption: no human read every pair. The record
therefore lists every such pair the judge did NOT pass, for a human to look at.

``run`` scores the set once and seals ONE record per judge identity into the calibration
store (``coherence_judge.seal_calibration_record``). The record passes when:

* every item was scored;
* pass/fail accuracy (COHERENT_EQUIVALENT vs the other three labels) is at least
  :data:`MIN_PASS_FAIL_ACCURACY`;
* no ``loop``, ``salad`` or ``shuffled`` item was judged COHERENT_EQUIVALENT.

A judge that lets obvious garbage through is the failure that matters. The 4-way
accuracy is recorded alongside.

Local runs go through the live endpoint (``--url``) because the orchestrator owns the
primitives. Each call passes ``allow_uncalibrated``, since this run IS the calibration,
and pins ``--scoring native`` (TD-29) or ``json``. A ``measurement_window_held`` or
``role_parked`` refusal aborts the run: rows are kept, nothing is sealed. Cloud runs can
use ``--in-process``. Rows are written per item as they arrive.

    python -m src.typed_decisions.coherence_judge_calibration build
    python -m src.typed_decisions.coherence_judge_calibration run --set <set.jsonl> \
        --backend local --model worker_general --scoring native [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.typed_decisions.coherence_judge import (
    JUDGE_VERSION,
    PASS_VERDICT,
    CalibrationStore,
    prompt_template_sha256,
    seal_calibration_record,
)

BUILDER_VERSION = "cjset.light.v1"
SET_SCHEMA = "epyc.orchestrator.coherence_judge_set.v1"
DEFAULT_SEED = 20261004
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SET_DIR = _REPO_ROOT / "artifacts" / "coherence_judge" / "calibration_sets"
DEFAULT_RUN_DIR = _REPO_ROOT / "artifacts" / "coherence_judge" / "calibration_runs"
INF70_RUNS = Path("/mnt/raid0/llm/tmp/inf70/agents/speed-claim/runs")
INF70_PROMPTS = Path("/mnt/raid0/llm/tmp/inf70/agents/e3-alpha/prompts.json")
INF70_BASE_ARM = "A1_plain"

CATEGORY_GOLD = {
    "loop": "INCOHERENT",
    "salad": "INCOHERENT",
    "shuffled": "INCOHERENT",
    "off_task": "OFF_TASK",
    "coherent_long_reasoning": "COHERENT_EQUIVALENT",
    "correct_alternative": "COHERENT_EQUIVALENT",
    "wrong_alternative": "DEGRADED",
    "mtp_diverged": "COHERENT_EQUIVALENT",
}
CATEGORY_COUNTS = {
    "loop": 4,
    "salad": 4,
    "shuffled": 4,
    "off_task": 3,
    "coherent_long_reasoning": 4,
    "correct_alternative": 3,
    "wrong_alternative": 4,
    "mtp_diverged": 6,
}
GARBAGE_CATEGORIES = ("loop", "salad", "shuffled")
MIN_PASS_FAIL_ACCURACY = 0.85


# ---------------------------------------------------------------------------
# Set builder
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _item(category: str, idx: str, prompt: str, base: str, candidate: str, source: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": f"{category}:{idx}",
        "category": category,
        "gold": CATEGORY_GOLD[category],
        "label_strength": "weak" if category == "mtp_diverged" else "strong",
        "prompt": prompt,
        "base_output": base,
        "candidate_output": candidate,
        "source": source,
        **extra,
    }


def load_inf70(runs_dir: Path = INF70_RUNS, prompts_path: Path = INF70_PROMPTS):
    """(prompts by id, {arm: {id: row}}, {source path: sha256})."""
    prompts = {p["id"]: p for p in json.loads(prompts_path.read_text())}
    sources = {str(prompts_path): _sha256(prompts_path.read_bytes())}
    arms: dict[str, dict[str, dict]] = {}
    for path in sorted(runs_dir.glob("*.rows.jsonl")):
        raw = path.read_bytes()
        sources[str(path)] = _sha256(raw)
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        arms[path.name[: -len(".rows.jsonl")]] = {r["id"]: r for r in rows if "id" in r}
    return prompts, arms, sources


def make_loop(base: str, rng: random.Random) -> str:
    words = base.split() or ["the"]
    words = (words * 12)[: max(len(words), 24)]
    head = words[: max(4, len(words) // 3)]
    start = rng.randrange(0, max(1, len(words) - 6))
    phrase = words[start : start + rng.randint(3, 6)]
    out = list(head)
    while len(out) < max(len(words), 60):
        out.extend(phrase)
    return " ".join(out)


def make_salad(vocab: Sequence[str], n_words: int, rng: random.Random) -> str:
    symbols = ["{", "}", "::", "##", "|", ";", "$$", "<<", "0x", "_"]
    return " ".join(
        rng.choice(vocab) if rng.random() < 0.8 else rng.choice(symbols) for _ in range(max(40, n_words))
    )


def make_shuffled(base: str, rng: random.Random) -> str:
    words = base.split() or ["the"]
    words = (words * 4)[: max(len(words), 16)]
    shuffled = list(words)
    for _ in range(5):
        rng.shuffle(shuffled)
        if shuffled != words:
            break
    return " ".join(shuffled)


_NAMES = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fay", "Gus", "Hana"]
_ITEMS = ["apples", "pencils", "stickers", "books", "marbles", "cookies"]


def _arith(rng: random.Random) -> dict[str, Any]:
    boxes, per_box = rng.randint(3, 9), rng.randint(4, 15)
    given, bought = rng.randint(2, boxes * per_box // 2), rng.randint(5, 30)
    p = {"name": rng.choice(_NAMES), "item": rng.choice(_ITEMS), "boxes": boxes, "per_box": per_box,
         "given": given, "bought": bought, "total": boxes * per_box}
    p["answer"] = p["total"] - given + bought
    p["prompt"] = (
        f"{p['name']} has {boxes} boxes with {per_box} {p['item']} in each box. {p['name']} gives "
        f"away {given} {p['item']} and then buys {bought} more. How many {p['item']} does "
        f"{p['name']} have now?"
    )
    return p


def _short(p: Mapping[str, Any]) -> str:
    return f"{p['name']} has {p['answer']} {p['item']} now."


def _long(p: Mapping[str, Any], slip: int = 0) -> str:
    total = p["total"] + slip
    after = total - p["given"]
    final = after + p["bought"]
    return (
        "Let's work through this step by step.\n\n"
        f"1. Boxes: {p['boxes']} x {p['per_box']} = {total} {p['item']}.\n"
        f"2. {p['name']} gives away {p['given']}: {total} - {p['given']} = {after}.\n"
        f"3. {p['name']} buys {p['bought']} more: {after} + {p['bought']} = {final}.\n\n"
        f"**Answer: {final}**"
    )


def _alternative(p: Mapping[str, Any]) -> str:
    net = p["bought"] - p["given"]
    sign = "+" if net >= 0 else "-"
    return (
        f"The net change is {p['bought']} - {p['given']} = {net}. The boxes hold "
        f"{p['boxes']} * {p['per_box']} = {p['total']}, so the count is {p['total']} {sign} "
        f"{abs(net)} = {p['answer']}. Final answer: {p['answer']}."
    )


def build_set(
    *,
    seed: int = DEFAULT_SEED,
    runs_dir: Path = INF70_RUNS,
    prompts_path: Path = INF70_PROMPTS,
    base_arm: str = INF70_BASE_ARM,
    counts: Mapping[str, int] = CATEGORY_COUNTS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(items, manifest). Deterministic for a given seed and given source files."""
    rng = random.Random(seed)
    prompts, arms, sources = load_inf70(runs_dir, prompts_path)
    base_rows = arms.get(base_arm) or {}
    ids = [pid for pid in prompts if (base_rows.get(pid) or {}).get("text", "").strip()]
    if not ids:
        raise ValueError(f"base arm {base_arm!r} has no rows under {runs_dir}")
    vocab = sorted({w for pid in ids for w in base_rows[pid]["text"].split() if re.fullmatch(r"[A-Za-z]{3,12}", w)})
    items: list[dict[str, Any]] = []

    def picks(n: int) -> list[str]:
        return rng.sample(ids, min(n, len(ids)))

    for pid in picks(counts["loop"]):
        items.append(_item("loop", pid, prompts[pid]["prompt"], base_rows[pid]["text"], make_loop(base_rows[pid]["text"], rng), f"inf70:{base_arm}:{pid}"))
    for pid in picks(counts["salad"]):
        base = base_rows[pid]["text"]
        items.append(_item("salad", pid, prompts[pid]["prompt"], base, make_salad(vocab, len(base.split()), rng), f"inf70:{base_arm}:{pid}"))
    for pid in picks(counts["shuffled"]):
        items.append(_item("shuffled", pid, prompts[pid]["prompt"], base_rows[pid]["text"], make_shuffled(base_rows[pid]["text"], rng), f"inf70:{base_arm}:{pid}"))
    for pid in picks(counts["off_task"]):
        others = [o for o in ids if prompts[o].get("class") != prompts[pid].get("class")] or [o for o in ids if o != pid]
        other = rng.choice(others)
        items.append(_item("off_task", pid, prompts[pid]["prompt"], base_rows[pid]["text"], base_rows[other]["text"], f"inf70:{base_arm}:{pid}", candidate_from=other))
    for k in range(counts["coherent_long_reasoning"]):
        p = _arith(rng)
        pair = (_short(p), _long(p)) if k % 2 == 0 else (_long(p), _short(p))
        items.append(_item("coherent_long_reasoning", f"a{k}", p["prompt"], *pair, f"synthetic:{seed}:lr{k}"))
    for k in range(counts["correct_alternative"]):
        p = _arith(rng)
        items.append(_item("correct_alternative", f"a{k}", p["prompt"], _long(p), _alternative(p), f"synthetic:{seed}:ca{k}"))
    for k in range(counts["wrong_alternative"]):
        p = _arith(rng)
        slip = rng.choice([-10, -2, -1, 1, 2, 10])
        items.append(_item("wrong_alternative", f"a{k}", p["prompt"], _long(p), _long(p, slip), f"synthetic:{seed}:wa{k}", slip=slip))
    # Real diverged-but-fine MTP pairs: at most one per prompt, round-robin over arms.
    mtp: list[dict[str, Any]] = []
    mtp_arms = [a for a in sorted(arms) if a != base_arm and "plain" not in a and a != "b7-anchor"]
    for pid in ids:
        if base_rows[pid].get("verdict") != "COHERENT":
            continue
        for arm in mtp_arms:
            row = arms[arm].get(pid) or {}
            text = row.get("text") or ""
            if row.get("verdict") == "COHERENT" and text.strip() and text != base_rows[pid]["text"]:
                mtp.append(_item("mtp_diverged", f"{pid}:{arm}", prompts[pid]["prompt"], base_rows[pid]["text"], text,
                                 f"inf70:{arm}:{pid}", base_arm=base_arm, candidate_arm=arm))
                break
    items.extend(rng.sample(mtp, min(counts["mtp_diverged"], len(mtp))))
    body = "".join(json.dumps(item, sort_keys=True) + "\n" for item in items)
    set_sha = _sha256(body.encode())
    manifest = {
        "schema": SET_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "set_id": "cjset-" + set_sha[:12],
        "sha256": set_sha,
        "seed": seed,
        "n": len(items),
        "categories": dict(Counter(i["category"] for i in items)),
        "sources": sources,
        "base_arm": base_arm,
    }
    return items, manifest


def write_set(items: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], out_dir: Path = DEFAULT_SET_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{manifest['set_id']}.jsonl"
    path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in items))
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def read_set(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()], _sha256(raw)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

#: item -> (http_status, body); body is the verdict object or {"error": {...}}.
JudgeCall = Callable[[Mapping[str, Any]], tuple[int, dict[str, Any]]]
_ABORT_KINDS = ("measurement_window_held", "role_parked")


def _request_body(item: Mapping[str, Any], *, backend: str, model: str | None, scoring: str, set_id: str) -> dict[str, Any]:
    return {
        "prompt": item["prompt"],
        "base_output": item["base_output"],
        "candidate_output": item["candidate_output"],
        "backend": backend,
        "model": model,
        "scoring": scoring,
        "allow_uncalibrated": True,
        "caller": f"calibration:{set_id}",
    }


def http_judge_call(url: str, *, backend: str, model: str | None, scoring: str, set_id: str, timeout_s: float = 600.0) -> JudgeCall:
    endpoint = url.rstrip("/") + "/v1/typed/coherence_judge"

    def call(item: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        body = _request_body(item, backend=backend, model=model, scoring=scoring, set_id=set_id)
        request = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as resp:
                return resp.status, json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read().decode() or "{}")
            except ValueError:
                return exc.code, {"error": {"type": "http_error", "message": str(exc)}}

    return call


def in_process_judge_call(*, backend: str, model: str | None, scoring: str, set_id: str, judge: Any = None) -> JudgeCall:
    from src.typed_decisions.coherence_judge import CoherenceJudge, JudgeFailed, JudgeRefused, JudgeRequest

    judge = judge or CoherenceJudge()

    def call(item: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            verdict = judge.judge(JudgeRequest(**_request_body(item, backend=backend, model=model, scoring=scoring, set_id=set_id)))
        except JudgeRefused as exc:
            return exc.status_code, {"error": exc.to_dict()}
        except JudgeFailed as exc:
            return 502, {"error": exc.to_dict()}
        return 200, verdict.to_dict()

    return call


def score_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The pass/fail accuracy, the 4-way accuracy, per-category counts and the pass decision."""
    scored = [r for r in rows if r.get("verdict")]
    label = {r["id"]: r["verdict"]["verdict"] for r in scored}
    per_cat: dict[str, dict[str, Any]] = {}
    for cat in sorted({r["category"] for r in rows}):
        cat_rows = [r for r in rows if r["category"] == cat]
        per_cat[cat] = {
            "n": len(cat_rows),
            "correct": sum(1 for r in cat_rows if label.get(r["id"]) == r["gold"]),
            "judged_pass": sum(1 for r in cat_rows if label.get(r["id"]) == PASS_VERDICT),
        }
    binary = sum(1 for r in scored if (label[r["id"]] == PASS_VERDICT) == (r["gold"] == PASS_VERDICT))
    metrics = {
        "n_items": len(rows),
        "n_scored": len(scored),
        "accuracy_pass_fail": round(binary / len(rows), 4) if rows else None,
        "accuracy_4way": round(sum(1 for r in scored if label[r["id"]] == r["gold"]) / len(rows), 4) if rows else None,
        "per_category": per_cat,
        "review_queue": [
            {"id": r["id"], "verdict": label[r["id"]]}
            for r in scored
            if r.get("label_strength") == "weak" and label[r["id"]] != r["gold"]
        ],
    }
    reasons = []
    if len(scored) < len(rows):
        reasons.append(f"{len(rows) - len(scored)} item(s) unscored")
    if not rows or metrics["accuracy_pass_fail"] < MIN_PASS_FAIL_ACCURACY:
        reasons.append(f"accuracy_pass_fail {metrics['accuracy_pass_fail']} < {MIN_PASS_FAIL_ACCURACY}")
    leaked = sum(per_cat.get(c, {}).get("judged_pass", 0) for c in GARBAGE_CATEGORIES)
    if leaked:
        reasons.append(f"{leaked} loop/salad/shuffled item(s) judged {PASS_VERDICT}")
    return {"metrics": metrics, "passed": not reasons, "fail_reasons": reasons}


def run_calibration(
    items: Sequence[Mapping[str, Any]],
    *,
    set_sha256: str,
    set_id: str,
    call: JudgeCall,
    out_dir: Path,
    store: CalibrationStore | None = None,
    backend: str,
    model: str | None,
    scoring: str,
) -> dict[str, Any]:
    """Score every item once and persist each row; seal one record per judge_key."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    rows: list[dict[str, Any]] = []
    aborted = None
    with open(rows_path, "a", encoding="utf-8") as handle:
        for item in items:
            status, body = call(item)
            error = body.get("error") if isinstance(body, dict) else None
            row = {
                "id": item["id"],
                "category": item["category"],
                "gold": item["gold"],
                "label_strength": item.get("label_strength", "strong"),
                "status": status,
                "verdict": body if status == 200 and not error else None,
                "error": error,
            }
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            if isinstance(error, dict) and error.get("type") in _ABORT_KINDS:
                aborted = error
                break
    summary: dict[str, Any] = {"set_id": set_id, "set_sha256": set_sha256, "rows_path": str(rows_path), "aborted": aborted, "records": []}
    if aborted:
        return summary  # a partial run is never sealed
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row.get("verdict") or {}).get("judge_key")
        if key:
            by_key[key].append(row)
    unscored = [r for r in rows if not r.get("verdict")]
    store = store or CalibrationStore()
    for key, key_rows in by_key.items():
        result = score_rows(key_rows + unscored)  # failures count against every identity
        sample = key_rows[0]["verdict"]
        record = seal_calibration_record(
            {
                "judge_key": key,
                "judge_identity": {
                    "judge_version": JUDGE_VERSION,
                    "prompt_template_sha256": prompt_template_sha256(),
                    "backend": sample.get("backend"),
                    "model": sample.get("model"),
                    "served_model": sample.get("served_model"),
                    "scoring_mode": sample.get("scoring_mode"),
                },
                "requested": {"backend": backend, "model": model, "scoring": scoring},
                "set": {"set_id": set_id, "sha256": set_sha256, "n": len(items)},
                "rule": {"min_pass_fail_accuracy": MIN_PASS_FAIL_ACCURACY, "no_garbage_pass": list(GARBAGE_CATEGORIES)},
                **result,
                "rows_path": str(rows_path),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        path = store.write(record)
        summary["records"].append(
            {
                "calibration_id": record["calibration_id"],
                "judge_key": key,
                "passed": record["passed"],
                "accuracy_pass_fail": record["metrics"]["accuracy_pass_fail"],
                "path": str(path),
            }
        )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Light calibration for the coherence judge.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the labelled set (no model calls)")
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    b.add_argument("--runs-dir", type=Path, default=INF70_RUNS)
    b.add_argument("--prompts", type=Path, default=INF70_PROMPTS)
    b.add_argument("--out-dir", type=Path, default=DEFAULT_SET_DIR)
    r = sub.add_parser("run", help="score the judge once and seal the calibration record")
    r.add_argument("--set", type=Path, required=True)
    r.add_argument("--backend", default="local")
    r.add_argument("--model", default=None)
    r.add_argument("--scoring", choices=("native", "json"), default="native")
    r.add_argument("--url", default="http://127.0.0.1:8000")
    r.add_argument("--in-process", action="store_true", help="cloud backends only")
    r.add_argument("--out-dir", type=Path, default=None)
    r.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "build":
        items, manifest = build_set(seed=args.seed, runs_dir=args.runs_dir, prompts_path=args.prompts)
        path = write_set(items, manifest, args.out_dir)
        print(json.dumps({"set": str(path), "set_id": manifest["set_id"], "n": manifest["n"], "categories": manifest["categories"]}, indent=2))
        return 0
    items, set_sha = read_set(args.set)
    set_id = args.set.stem
    if args.in_process and not args.backend.startswith("cloud:"):
        print("--in-process is for cloud backends; local runs go through the live endpoint", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps({"dry_run": True, "set_id": set_id, "set_sha256": set_sha, "n": len(items),
                          "categories": dict(Counter(i["category"] for i in items)), "backend": args.backend,
                          "model": args.model, "scoring": args.scoring, "judge_version": JUDGE_VERSION,
                          "prompt_template_sha256": prompt_template_sha256()}, indent=2))
        return 0
    out_dir = args.out_dir or DEFAULT_RUN_DIR / f"{set_id}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    kwargs = dict(backend=args.backend, model=args.model, scoring=args.scoring, set_id=set_id)
    call = in_process_judge_call(**kwargs) if args.in_process else http_judge_call(args.url, **kwargs)
    summary = run_calibration(items, set_sha256=set_sha, set_id=set_id, call=call, out_dir=out_dir,
                              backend=args.backend, model=args.model, scoring=args.scoring)
    print(json.dumps(summary, indent=2))
    if summary["aborted"]:
        return 3
    return 0 if any(r["passed"] for r in summary["records"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
