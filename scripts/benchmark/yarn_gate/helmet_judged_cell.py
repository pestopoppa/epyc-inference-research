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

from scripts.benchmark.yarn_gate import retokenize  # noqa: E402
from scripts.benchmark.yarn_gate.local_judge import JudgePin, LocalJudge, judge_rows  # noqa: E402

HELMET_REPO = "https://github.com/princeton-nlp/HELMET"
HELMET_PIN = "aeadacc6593e9a222a84b0660b186a4e23d2efda"
SCHEMA = "epyc.yarn_helmet_judged_cell.v1"
DECISION_SCHEMA = "epyc.yarn_gate_decision.v1"
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


def load_judge_validation(path: Path) -> dict:
    """Read a ``judge_pin.json`` (``epyc.yarn_judge_pin.v1``) -> {"judge_pin": ..., "judge_validation": ...} in the
    shapes epyc.yarn_gate_decision.v1 wants. The pin's ``max_tokens`` lives at the top level of that file."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    v, pin = d["validation"], dict(d["judge_pin"])
    pin.setdefault("max_tokens", d.get("max_tokens"))
    return {"judge_pin": pin,
            "judge_validation": {"n_compared": v["n_compared"], "accuracy": v["accuracy"],
                                 "cohen_kappa": v.get("cohen_kappa"), "sheet_sha256": d["sheet_sha256"]}}


def decision_records(*, rows: list[dict], res: dict, task: str, nominal_length: int, config_file: str,
                     config_sha256: str | None, max_test_samples: int | None, helmet_pin: str, comparison: str,
                     arm_role: str, model_pin: str, prompt_n_source: str | None, validation: dict | None,
                     mock: bool = False) -> tuple[list[dict], list[dict]]:
    """Judged-cell ``epyc.yarn_gate_decision.v1`` records, ONE per (REAL-length bin, zone): a cell that spans zones is
    split, never labelled "mixed". -> (records, excluded-with-reason). Rows without a measured ``prompt_n`` are excluded."""
    excl: list[dict] = []
    if not validation:
        return [], [{"reason": "no judge_validation supplied (judge pin must be validated against the 50-item sheet)"}]
    if (res["judge_pin"].get("model"), res["judge_pin"].get("prompt_sha256")) != (
            validation["judge_pin"].get("model"), validation["judge_pin"].get("prompt_sha256")):
        return [], [{"reason": "judge used differs from the validated judge pin (model/prompt_sha256)"}]
    if prompt_n_source not in retokenize.MEASURED_SOURCES:
        return [], [{"reason": f"prompt_n source {prompt_n_source!r} is not a measured source"}]
    if not config_sha256 or not config_file:
        return [], [{"reason": "config_file/config_sha256 not recorded"}]
    verdict_of = {j["i"]: j["verdict"] for j in res["judged"]}
    groups: dict[tuple, dict] = {}
    for i, r in enumerate(rows):
        n = r.get("prompt_n")
        if not isinstance(n, (int, float)) or isinstance(n, bool) or n <= 0 or int(n) != n:
            excl.append({"row": i, "reason": "no measured prompt_n"})
            continue
        n = int(n)
        g = groups.setdefault((retokenize.bin_label(n), retokenize.label_zone(n, source=prompt_n_source)),
                              {"n": [], "v": [], "unjudged": 0})
        g["n"].append(n)
        if i in verdict_of:
            g["v"].append(verdict_of[i])
        else:
            g["unjudged"] += 1
    out = []
    for (b, zone), g in sorted(groups.items()):
        if not g["v"]:
            excl.append({"group": [b, zone], "reason": "no judged rows in group"})
            continue
        ns_ = sorted(g["n"])
        out.append({
            "schema": DECISION_SCHEMA, "record_kind": "cell", "mock": mock, "comparison": comparison,
            "arm_role": arm_role, "suite": "helmet", "task": task, "nominal_length": nominal_length,
            "model_pin": model_pin,
            "prompt_n": {"source": prompt_n_source, "count": len(ns_), "min": ns_[0],
                         "median": ns_[(len(ns_) - 1) // 2], "max": ns_[-1]},
            "bin": b, "zone": zone,
            "source_pins": {"helmet_pin": helmet_pin, "config_file": config_file, "config_sha256": config_sha256,
                            "max_test_samples": max_test_samples},
            "comparable_to_published": False, "judged": True,
            "judge_pin": {k: res["judge_pin"].get(k) for k in
                          ("model", "prompt_version", "prompt_sha256", "temperature", "seed", "max_tokens")},
            "judge_validation": validation["judge_validation"],
            "metric": "accuracy", "value": sum(g["v"]) / len(g["v"]), "n_units": len(g["v"]), "unit": "items",
            "n_unjudged": g["unjudged"]})
    return out, excl


def score_cell(cell_path: Path, judge: LocalJudge, *, task: str, nominal_length: int,
               config_file: str, max_test_samples: int | None, helmet_pin: str = HELMET_PIN,
               comparison: str | None = None, arm_role: str | None = None, model_pin: str | None = None,
               prompt_n_source: str | None = None, validation: dict | None = None,
               config_sha256: str | None = None, mock: bool = False) -> dict:
    raw = load_cell(cell_path)
    probs = selfcheck_schema(raw)
    if probs:
        raise ValueError(f"{cell_path}: " + "; ".join(probs))
    rows = normalise(raw)
    res = judge_rows(judge, rows)
    pn = [r["prompt_n"] for r in rows if isinstance(r["prompt_n"], (int, float))]
    rec = {"schema": SCHEMA, "task": task, "nominal_length": nominal_length,
           "prompt_n": {"count": len(pn), "min": min(pn) if pn else None, "max": max(pn) if pn else None},
           "helmet_repo": HELMET_REPO, "helmet_pin": helmet_pin, "config_file": config_file,
           "max_test_samples": max_test_samples,
           "cell_sha256": hashlib.sha256(cell_path.read_bytes()).hexdigest(),
           "n_rows": len(rows), "comparable_to_published": False, **res}
    if comparison and arm_role and model_pin:
        if config_sha256 is None and Path(config_file).is_file():
            config_sha256 = hashlib.sha256(Path(config_file).read_bytes()).hexdigest()
        recs, excl = decision_records(
            rows=rows, res=res, task=task, nominal_length=nominal_length, config_file=config_file,
            config_sha256=config_sha256, max_test_samples=max_test_samples, helmet_pin=helmet_pin,
            comparison=comparison, arm_role=arm_role, model_pin=model_pin, prompt_n_source=prompt_n_source,
            validation=validation, mock=mock)
        rec["decision_records"], rec["decision_excluded"] = recs, excl
    return rec


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
    ap.add_argument("--comparison", choices=("A0_vs_A1", "CN_vs_CY2"), help="belief emit: which pair this cell belongs to")
    ap.add_argument("--arm-role", choices=("A0", "A1", "CN", "CY2"))
    ap.add_argument("--model-pin", help="belief emit: served model id (+ GGUF) the cell was generated with")
    ap.add_argument("--prompt-n-source", help="measured source of the rows' prompt_n (served_tokenizer|server_timings|model_tokenizer_json)")
    ap.add_argument("--judge-pin-json", help="judge_pin.json from the judge validation run (judge_pin + 50-item agreement)")
    ap.add_argument("--config-sha256")
    ap.add_argument("--emit-belief-dir", help="run dir: write yarn_gate_agg.beliefs.jsonl here from the judged-cell records")
    ap.add_argument("--run-id")
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
                     config_file=a.config_file, max_test_samples=a.max_test_samples,
                     comparison=a.comparison, arm_role=a.arm_role, model_pin=a.model_pin,
                     prompt_n_source=a.prompt_n_source, config_sha256=a.config_sha256,
                     validation=load_judge_validation(Path(a.judge_pin_json)) if a.judge_pin_json else None)
    if caller is not None:
        rec["cloud_identity"] = caller.identity()
    Path(a.out).write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    if a.emit_belief_dir:
        from scripts.benchmark.yarn_gate import decision_records as dr
        recs = rec.get("decision_records") or []
        if not recs:
            print(f"belief emit skipped: {rec.get('decision_excluded')}", file=sys.stderr)
        else:
            st = dr.emit(Path(a.emit_belief_dir), run_id=a.run_id or f"yarn-helmet-judged:{a.task}:{rec['cell_sha256'][:12]}",
                         cells=recs, pairs={}, rule_id="", alpha=0.05, final=True, excluded=rec["decision_excluded"])
            print(f"belief emit: {st['status']} {st.get('sidecar', '')}")
    print(f"judged {rec['n_judged']}/{rec['n_rows']} accuracy={rec['accuracy']} unjudged={rec['n_unjudged']}")
    return 0 if rec["n_unjudged"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
