#!/usr/bin/env python3
"""YARN-JUDGE-1 / YARN-AGG-1(2): score ONE judged HELMET cell end to end with the pinned local judge.

Input is the per-cell result JSON that the pinned HELMET repo
(``princeton-nlp/HELMET`` @ ``aeadacc6593e9a222a84b0660b186a4e23d2efda``) writes for a generation run
against ``--endpoint_url http://localhost:{port}/v1/``: either ``{"data": [...]}`` or a bare list of
rows with ``question``, ``answer`` and ``output`` / ``parsed_output``. (The row keys are remappable;
this adapter has NOT been verified against a real HELMET output file, since none exists yet: the first
window must run ``--selfcheck-schema`` on one real file before judging.)

Output is a judged-cell record carrying everything the claim tuple needs: judge pin, HELMET repo pin,
config file, ``max_test_samples``, the cell task, nominal length, REAL prompt_n stats (when rows carry
``prompt_n`` / ``input_len``) and the unjudged list. The record states ``comparable_to_published: false``.

Judging runs outside the YaRN timing window (``--judge-backend cloud`` needs no server). The runner script ``run_local_judge_window.sh`` wraps this
behind ``WINDOW_CONFIRMED=1``; this module has no inference of its own beyond the localhost judge call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.benchmark.yarn_gate.local_judge import JudgePin, LocalJudge, judge_rows  # noqa: E402

HELMET_REPO = "https://github.com/princeton-nlp/HELMET"
HELMET_PIN = "aeadacc6593e9a222a84b0660b186a4e23d2efda"
SCHEMA = "epyc.yarn_helmet_judged_cell.v1"
#: HELMET categories that need a judge (the hosted GPT-4o judge is replaced by the local pin).
JUDGED_CATEGORIES = ("longqa", "summ")


def load_cell(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("data") if isinstance(obj, dict) else obj
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list or {{'data': [...]}}")
    return rows


def normalise(rows: list[dict], cand_keys=("parsed_output", "output")) -> list[dict]:
    out = []
    for r in rows:
        cand = next((r[k] for k in cand_keys if isinstance(r.get(k), str)), "")
        ans = r.get("answer", r.get("answers", ""))
        out.append({"question": str(r.get("question", "")), "answer": ans, "output": cand,
                    "prompt_n": r.get("prompt_n", r.get("input_len"))})
    return out


def selfcheck_schema(rows: list[dict]) -> list[str]:
    """Problems that would make judging vacuous (all-empty outputs, no references, no questions)."""
    n = normalise(rows)
    probs = []
    if not n:
        probs.append("cell has zero rows")
    if n and not any(r["output"].strip() for r in n):
        probs.append("every output is empty")
    if n and not any(str(r["answer"]).strip() for r in n):
        probs.append("no reference answers")
    if n and not any(r["question"].strip() for r in n):
        probs.append("no question text (summ cells need a different judge prompt)")
    return probs


def score_cell(cell_path: Path, judge: LocalJudge, *, task: str, nominal_length: int,
               config_file: str, max_test_samples: int | None, helmet_pin: str = HELMET_PIN) -> dict:
    raw = load_cell(cell_path)
    probs = selfcheck_schema(raw)
    if probs:
        raise ValueError(f"{cell_path}: " + "; ".join(probs))
    rows = normalise(raw)
    res = judge_rows(judge, rows)
    pn = [r["prompt_n"] for r in rows if isinstance(r["prompt_n"], (int, float))]
    return {"schema": SCHEMA, "task": task, "nominal_length": nominal_length,
            "prompt_n": {"count": len(pn), "min": min(pn) if pn else None, "max": max(pn) if pn else None},
            "helmet_repo": HELMET_REPO, "helmet_pin": helmet_pin, "config_file": config_file,
            "max_test_samples": max_test_samples,
            "cell_sha256": hashlib.sha256(cell_path.read_bytes()).hexdigest(),
            "n_rows": len(rows), "comparable_to_published": False, **res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True, help="HELMET per-cell result JSON")
    ap.add_argument("--task", required=True)
    ap.add_argument("--nominal-length", type=int, required=True)
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--max-test-samples", type=int)
    ap.add_argument("--judge-backend", choices=("served", "cloud"), default="served",
                    help="cloud = pinned cloud judge via the orchestrator cloud_judges transport (no server)")
    ap.add_argument("--cloud-judge", default="sonnet-5-5-low", help="cloud backend: cloud_judges.yaml entry")
    ap.add_argument("--port", type=int, help="served backend")
    ap.add_argument("--judge-model", help="served backend: the served model id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selfcheck-schema", action="store_true", help="only validate the cell file and exit")
    a = ap.parse_args(argv)
    if a.selfcheck_schema:
        probs = selfcheck_schema(load_cell(Path(a.cell)))
        print("OK" if not probs else "PROBLEMS: " + "; ".join(probs))
        return 1 if probs else 0
    caller = None
    if a.judge_backend == "cloud":
        from scripts.benchmark.yarn_gate import cloud_judge as cjm
        cj = cjm.load_cloud_judges()
        caller = cjm.CloudCaller(cj, cjm.resolve_spec(cj, a.cloud_judge, transport="claude_cli"))
        try:
            caller.preflight()
        except cjm.JudgeError as exc:
            print(f"BLOCKED: {exc}", file=sys.stderr)
            return 5
        judge = cjm.CloudJudge(caller)
    else:
        if a.port is None or not a.judge_model:
            ap.error("--judge-backend served needs --port and --judge-model")
        judge = LocalJudge(a.port, JudgePin(model=a.judge_model))
    rec = score_cell(Path(a.cell), judge, task=a.task, nominal_length=a.nominal_length,
                     config_file=a.config_file, max_test_samples=a.max_test_samples)
    if caller is not None:
        rec["cloud_identity"] = caller.identity()
    Path(a.out).write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(f"judged {rec['n_judged']}/{rec['n_rows']} accuracy={rec['accuracy']} unjudged={rec['n_unjudged']}")
    return 0 if rec["n_unjudged"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
