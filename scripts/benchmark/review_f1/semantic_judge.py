#!/usr/bin/env python3
"""EV-13b semantic matcher -- implementation of `data/review_f1/SEMANTIC_MATCHER_SPEC.md`.

The deterministic build-leg matcher (`scorer._matches`) can never fire on Augment-v1 goldens
(criterion "unspecified", no location), so an LLM judge decides whether a reviewer finding
describes the same defect as a golden comment. This module:

  * `calibrate`  -- positive/negative judge controls (each must reach >= 95%) -> judge_calibration.json
  * `judge`      -- one judge call per persisted reader finding, per PR per run, atomic + resumable:
                    results/<reader>__<q>/judge/<judge>__<jq>/<case_id>.run<i>.json
  * `score`      -- deterministic maximum-matching assignment over the judge edges, micro P/R/F1 per
                    run, Mean-F1/StdDev (scorer.prf / aggregate protocol), summary fields, and the
                    EV-6 judge-swap delta when a second judge is scored.

Judging runs only over PERSISTED reader outputs (harness.py), so a review is never regenerated to
change the judge. A judge equal to the reader is refused.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import random
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
import scorer  # noqa: E402
from harness import cross_family_ok  # noqa: E402

MATCHER = "semantic-judge.v1"
SPEC_PATH = PKG.parents[2] / "data" / "review_f1" / "SEMANTIC_MATCHER_SPEC.md"
LINE_WINDOW = 10
PARSE_FAIL_MALFUNCTION = 0.05
CONTROL_BAR = 0.95

SYSTEM = (
    "You compare code-review comments. Decide whether a REVIEWER comment points at the SAME\n"
    "underlying defect as any of the numbered GOLDEN comments on the same pull request.\n"
    "Same defect = same root cause in the same code, even if worded differently, at a different\n"
    "line, or described more/less completely. NOT the same: a different bug in the same function,\n"
    "a generic warning (\"add error handling\") vs a specific defect, a style remark vs a bug.\n"
    "If the reviewer comment bundles several defects, list every golden it genuinely covers.\n"
    "Answer with ONLY a JSON object."
)


# --------------------------------------------------------------------------------------------
# diff validity (deterministic, recorded, never decides a match)
# --------------------------------------------------------------------------------------------
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def touched(diff: str) -> dict[str, list[tuple[int, int]]]:
    files: dict[str, list[tuple[int, int]]] = {}
    current = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            current = None if path == "/dev/null" else re.sub(r"^b/", "", path)
            if current is not None:
                files.setdefault(current, [])
            continue
        m = _HUNK.match(line)
        if m and current is not None:
            start = int(m.group(1))
            length = int(m.group(2)) if m.group(2) is not None else 1
            files[current].append((start, start + max(length, 1) - 1))
    return files


def _path_match(a: str, b: str) -> bool:
    a, b = a.strip().lstrip("./"), b.strip().lstrip("./")
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def validity(finding: dict, files: dict[str, list[tuple[int, int]]]) -> tuple[bool, bool]:
    loc = finding.get("location") or {}
    fname = loc.get("file")
    if not fname:
        return False, False
    hit = [f for f in files if _path_match(str(fname), f)]
    if not hit:
        return False, False
    try:
        s = int(loc.get("line_start"))
        e = int(loc.get("line_end") if loc.get("line_end") is not None else s)
    except (TypeError, ValueError):
        return True, False
    for f in hit:
        for hs, he in files[f]:
            if s <= he + LINE_WINDOW and hs - LINE_WINDOW <= e:
                return True, True
    return True, False


# --------------------------------------------------------------------------------------------
# prompt + parse
# --------------------------------------------------------------------------------------------
def golden_order(case: dict, seed: int, finding_index: int) -> list[dict]:
    goldens = list(case["golden_findings"])
    rng = random.Random(f"{case['case_id']}|{seed}|{finding_index}")
    rng.shuffle(goldens)
    return goldens


def judge_messages(case: dict, finding: dict, order: list[dict]) -> list[dict]:
    loc = finding.get("location") or {}
    a, b = loc.get("line_start"), loc.get("line_end")
    lines = f"{a if a is not None else '?'}-{b if b is not None else (a if a is not None else '?')}"
    user = (f"PR: {case.get('pr_ref', {}).get('title', '')}\n"
            f"REVIEWER COMMENT (file {loc.get('file') or '?'}, lines {lines}):\n"
            f"{finding.get('comment', '')}\n\nGOLDEN COMMENTS:\n"
            + "\n".join(f"[{i + 1}] {g['comment']}" for i, g in enumerate(order))
            + '\n\nReturn: {"matches": [<golden numbers>], "confidence": "high"|"medium"|"low",\n'
              '         "rationale": "<= 40 words"}')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_judge(raw: str, order: list[dict]) -> dict:
    obj = None
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            try:
                obj = json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict) or not isinstance(obj.get("matches"), list):
        return {"parse_ok": False, "matches": [], "invalid_index": 0}
    ids, invalid = [], 0
    for m in obj["matches"]:
        try:
            k = int(m)
        except (TypeError, ValueError):
            invalid += 1
            continue
        if 1 <= k <= len(order):
            gid = order[k - 1]["golden_id"]
            if gid not in ids:
                ids.append(gid)
        else:
            invalid += 1
    return {"parse_ok": True, "matches": ids, "invalid_index": invalid,
            "confidence": obj.get("confidence"), "rationale": obj.get("rationale")}


# --------------------------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------------------------
def http_chat(url: str, timeout: int = 600) -> Callable[[list[dict], int], str]:
    def call(messages: list[dict], seed: int) -> str:
        payload = {"messages": messages, "temperature": 0.0, "seed": seed, "max_tokens": 256,
                   "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (local server)
            return json.loads(r.read())["choices"][0]["message"]["content"] or ""
    return call


def ask(call, case, finding, finding_index, seed) -> dict:
    order = golden_order(case, seed, finding_index)
    msgs = judge_messages(case, finding, order)
    raw = call(msgs, seed)
    parsed = parse_judge(raw, order)
    retried = False
    if not parsed["parse_ok"]:
        retried = True
        raw = call(msgs, seed + 1)
        parsed = parse_judge(raw, order)
    return {"golden_order": [g["golden_id"] for g in order], "raw": raw, "retried": retried, **parsed}


# --------------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------------
def paraphrase(comment: str) -> str:
    """Fixed template, not the judge: a light, deterministic rewording."""
    c = comment.strip()
    c = c[:1].lower() + c[1:] if c else c
    return f"Possible defect in this change: {c}"


def controls(cases: list[dict]) -> list[dict]:
    items = []
    by_repo: dict[str, list[dict]] = {}
    for case in cases:
        by_repo.setdefault(case["case_id"].split("__")[0], []).append(case)
    for case in cases:
        for g in case["golden_findings"]:
            if scorer._severity(g) == scorer.LOW_SEVERITY:
                continue
            items.append({"kind": "positive", "case": case, "expect": g["golden_id"],
                          "finding": {"comment": paraphrase(g["comment"]), "location": None}})
        peers = [c for c in by_repo[case["case_id"].split("__")[0]] if c["case_id"] != case["case_id"]]
        if peers:
            rng = random.Random(f"neg|{case['case_id']}")
            other = rng.choice(peers)
            g = rng.choice(other["golden_findings"])
            items.append({"kind": "negative", "case": case, "expect": None,
                          "finding": {"comment": g["comment"], "location": None},
                          "source_golden": g["golden_id"]})
    return items


def calibrate(cases, call, seed, workers=4) -> dict:
    items = controls(cases)

    def one(ix):
        i, it = ix
        rec = ask(call, it["case"], it["finding"], 10_000 + i, seed)
        ok = (it["expect"] in rec["matches"]) if it["kind"] == "positive" else (rec["parse_ok"] and not rec["matches"])
        return {"kind": it["kind"], "case_id": it["case"]["case_id"], "expect": it["expect"],
                "source_golden": it.get("source_golden"), "ok": ok, "matches": rec["matches"],
                "parse_ok": rec["parse_ok"], "raw": rec["raw"]}
    with cf.ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, enumerate(items)))
    pos = [r for r in rows if r["kind"] == "positive"]
    neg = [r for r in rows if r["kind"] == "negative"]
    p = sum(r["ok"] for r in pos) / len(pos)
    n = sum(r["ok"] for r in neg) / len(neg)
    return {"positive_rate": p, "n_positive": len(pos), "negative_rate": n, "n_negative": len(neg),
            "bar": CONTROL_BAR, "valid": p >= CONTROL_BAR and n >= CONTROL_BAR,
            "parse_fail": sum(not r["parse_ok"] for r in rows), "rows": rows}


# --------------------------------------------------------------------------------------------
# judging persisted reader output
# --------------------------------------------------------------------------------------------
def judge_all(cases, reader_root: Path, judge_key: str, call, context_dir: Path | None, workers=4) -> None:
    out = reader_root / "judge" / judge_key
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for case in cases:
        pr = json.loads((reader_root / f"{case['case_id']}.json").read_text())
        for run_index, findings in enumerate(pr["runs"]):
            path = out / f"{case['case_id']}.run{run_index}.json"
            if path.exists():
                continue
            jobs.append((case, run_index, findings, path, pr.get("run_seeds", [None])))

    def diff_of(case):
        dp = case.get("pr_ref", {}).get("diff_path")
        if dp and context_dir and (context_dir / dp).exists():
            return (context_dir / dp).read_text(errors="replace")
        return ""

    def do(job):
        case, run_index, findings, path, _ = job
        files = touched(diff_of(case))
        seed = 42 + run_index
        recs = []
        for fi, finding in enumerate(findings):
            rec = ask(call, case, finding, fi, seed)
            fid, liw = validity(finding, files)
            recs.append({"case_id": case["case_id"], "run_index": run_index, "finding_index": fi,
                         "file_in_diff": fid, "line_in_window": liw, **rec})
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"case_id": case["case_id"], "run_index": run_index,
                                   "judge_key": judge_key, "records": recs}, indent=1))
        tmp.replace(path)
        return case["case_id"], run_index, len(recs)
    with cf.ThreadPoolExecutor(workers) as ex:
        for cid, ri, n in ex.map(do, jobs):
            print(f"  judged {cid} run{ri}: {n} findings", flush=True)


# --------------------------------------------------------------------------------------------
# deterministic assignment + scoring
# --------------------------------------------------------------------------------------------
def assign(case: dict, edges: list[list[str]]) -> dict:
    """Maximum bipartite matching (Kuhn, findings in order, goldens by lowest index) over edges
    to SCORED goldens; then classify each finding per spec step 4."""
    scored = [g["golden_id"] for g in case["golden_findings"] if scorer._severity(g) != scorer.LOW_SEVERITY]
    low = {g["golden_id"] for g in case["golden_findings"] if scorer._severity(g) == scorer.LOW_SEVERITY}
    index = {gid: i for i, gid in enumerate(scored)}
    adj = [sorted(index[g] for g in e if g in index) for e in edges]
    owner = [-1] * len(scored)

    def try_assign(f, seen):
        for gi in adj[f]:
            if gi in seen:
                continue
            seen.add(gi)
            if owner[gi] == -1 or try_assign(owner[gi], seen):
                owner[gi] = f
                return True
        return False
    for f in range(len(edges)):
        try_assign(f, set())
    matched_f = {f for f in owner if f != -1}
    tp = len(matched_f)
    fp = dup = neutral = 0
    for f, e in enumerate(edges):
        if f in matched_f:
            continue
        if adj[f]:
            dup += 1
            fp += 1
        elif any(g in low for g in e):
            neutral += 1
        else:
            fp += 1
    fn = owner.count(-1)
    return {"tp": tp, "fp": fp, "fn": fn, "duplicate_fp": dup, "neutral_low": neutral}


def score(cases, reader_root: Path, judge_key: str) -> dict:
    jdir = reader_root / "judge" / judge_key
    runs: dict[int, dict] = {}
    for case in cases:
        for path in sorted(jdir.glob(f"{case['case_id']}.run*.json")):
            data = json.loads(path.read_text())
            recs = sorted(data["records"], key=lambda r: r["finding_index"])
            edges = [r["matches"] if r["parse_ok"] else [] for r in recs]
            c = assign(case, edges)
            r = runs.setdefault(data["run_index"], {"tp": 0, "fp": 0, "fn": 0, "duplicate_fp": 0,
                                                    "neutral_low": 0, "n_findings": 0, "parse_fail": 0,
                                                    "file_in_diff": 0, "line_in_window": 0, "cases": 0})
            for k in ("tp", "fp", "fn", "duplicate_fp", "neutral_low"):
                r[k] += c[k]
            r["n_findings"] += len(recs)
            r["parse_fail"] += sum(not x["parse_ok"] for x in recs)
            r["file_in_diff"] += sum(bool(x["file_in_diff"]) for x in recs)
            r["line_in_window"] += sum(bool(x["line_in_window"]) for x in recs)
            r["cases"] += 1
    per_run = []
    for ri in sorted(runs):
        r = runs[ri]
        pf = r["parse_fail"] / r["n_findings"] if r["n_findings"] else 0.0
        per_run.append({"run_index": ri, **scorer.prf(r["tp"], r["fp"], r["fn"]),
                        "duplicate_fp": r["duplicate_fp"], "neutral_low": r["neutral_low"],
                        "n_findings": r["n_findings"], "cases": r["cases"],
                        "judge_parse_fail_rate": pf,
                        "location_validity_rate": (r["line_in_window"] / r["n_findings"]) if r["n_findings"] else 0.0,
                        "file_in_diff_rate": (r["file_in_diff"] / r["n_findings"]) if r["n_findings"] else 0.0,
                        "malfunction": pf > PARSE_FAIL_MALFUNCTION or r["cases"] != len(cases)})
    valid = [p for p in per_run if not p["malfunction"]]
    f1s = [p["f1"] for p in valid]
    n = len(f1s)
    mean = sum(f1s) / n if n else 0.0
    std = (sum((x - mean) ** 2 for x in f1s) / n) ** 0.5 if n else 0.0
    return {"matcher": MATCHER, "judge_key": judge_key, "n_runs": n, "mean_f1": mean, "std_f1": std,
            "protocol_ok": n >= 3, "per_run": per_run,
            "note": "internal-only F1; not comparable to Factory/Augment leaderboards"}


def file_sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["calibrate", "judge", "score"])
    ap.add_argument("--golden", required=True)
    ap.add_argument("--reader-root", type=Path, help="results/<reader>__<quant>")
    ap.add_argument("--reader-model", default="")
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--judge-quant", required=True)
    ap.add_argument("--judge-url", default="http://127.0.0.1:18372")
    ap.add_argument("--context-dir", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, help="calibrate: output json")
    ap.add_argument("--swap-judge-key", default=None, help="score: second judge for the EV-6 delta")
    args = ap.parse_args()
    golden = json.loads(Path(args.golden).read_text())
    cases = golden["cases"]
    judge_key = f"{args.judge_model}__{args.judge_quant}"
    if args.reader_model and args.reader_model == args.judge_model:
        print("REFUSED: judge_model == reader model", file=sys.stderr)
        return 2
    if args.mode == "calibrate":
        res = calibrate(cases, http_chat(args.judge_url), args.seed, args.workers)
        res.update(judge_model=args.judge_model, judge_quant=args.judge_quant,
                   generated_at=datetime.now(timezone.utc).isoformat())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
        return 0 if res["valid"] else 1
    if args.mode == "judge":
        judge_all(cases, args.reader_root, judge_key, http_chat(args.judge_url), args.context_dir, args.workers)
        return 0
    summ = score(cases, args.reader_root, judge_key)
    reader_summary = args.reader_root / "_summary.json"
    rs = json.loads(reader_summary.read_text()) if reader_summary.exists() else {}
    manifest = json.loads(args.manifest.read_text()) if args.manifest and args.manifest.exists() else {}
    summ.update(
        spec_sha256=file_sha(SPEC_PATH),
        judge_config={"judge_model": args.judge_model, "judge_quant": args.judge_quant,
                      "reader_model": rs.get("model", args.reader_model), "reader_quant": rs.get("quant"),
                      "judge_distinct_from_reader": args.judge_model != rs.get("model", args.reader_model),
                      "cross_family_ok": cross_family_ok(rs.get("model", args.reader_model), args.judge_model),
                      "swap_tolerance_pp": 2.0},
        golden_manifest_checksum=(manifest.get("golden_set") or {}).get("checksum"),
        golden_checksum=golden.get("checksum"),
        n_findings=sum(p["n_findings"] for p in summ["per_run"]),
        duplicate_fp=sum(p["duplicate_fp"] for p in summ["per_run"]),
        neutral_low=sum(p["neutral_low"] for p in summ["per_run"]),
        judge_parse_fail_rate=max((p["judge_parse_fail_rate"] for p in summ["per_run"]), default=0.0),
        location_validity_rate=(sum(p["location_validity_rate"] for p in summ["per_run"]) / len(summ["per_run"]))
        if summ["per_run"] else 0.0,
        generated_at=datetime.now(timezone.utc).isoformat())
    if args.swap_judge_key:
        other = score(cases, args.reader_root, args.swap_judge_key)
        summ["judge_swap"] = {"other_judge_key": args.swap_judge_key, "other_mean_f1": other["mean_f1"],
                              "other_std_f1": other["std_f1"], "other_n_runs": other["n_runs"],
                              "judge_swap_delta_pp": abs(summ["mean_f1"] - other["mean_f1"]) * 100,
                              "gate_pp": 2.0,
                              "gate_ok": abs(summ["mean_f1"] - other["mean_f1"]) * 100 <= 2.0}
    out = args.reader_root / f"_summary.semantic.{judge_key}.json"
    out.write_text(json.dumps(summ, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in summ.items() if k != "per_run"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
