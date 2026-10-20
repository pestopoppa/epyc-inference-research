#!/usr/bin/env python3
"""AK-HR-2(iii): what did the planner's `## Already tried` block actually show?

Offline, single process, read-only.  Reads the two committed JSONL exports of AK-HR-1
(`history_records.jsonl`, `actor_call_records.jsonl`) -- never an experiments.db -- and,
for every planner call, rebuilds AS OF the call the history rows the planner was handed
and renders them with the REAL `actors.render_context` and the REAL
`run.prior_experiments` / `run._with_pinned_history` / `experiments.rank`, in both modes:

  recency  default (`--rank-prior-experiments` off): newest 40 of the store, plus up to
           500 pinned-status same-epoch rows appended.
  rank     `--rank-prior-experiments`: the 2000 newest rows ranked, truncated to 40,
           no pinned append.

Only `ExperimentStore.recall` is reproduced (`AsOfStore.recall`), because the real one
needs a sqlite file; the test cross-checks it against the real store on a synthetic set.

PROXIES (all echoed into the summary):
  * as-of cut: row.recorded_at < call.started_at, both normalised to 27-char
    `YYYY-MM-DDTHH:MM:SS.ffffffZ` (whole-second stamps are padded with `.000000`),
    because raw `...:48Z` vs `...:48.5Z` compares wrongly ('Z' > '.').  Row ORDER is by
    the raw stored string then rowid (what the SQL ORDER BY does).
  * call epoch / measurement epoch are those of the row the call is time-joined to (the
    NEXT recorded attempt in that store); an epoch change between call and that row
    mislabels the call.
  * same_epoch = row epoch == call epoch OR (row measurement_epoch is not None and
    == call measurement_epoch).  Verified aliases are not exported.
  * `exact_attribution_effect_fraction`, `target_runtime_effect_fraction`, `falsifier`,
    `result_sha256`, `claims` are not exported and are None; only `effect_fraction`
    matters to what is rendered.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autokernel.controller import experiments  # noqa: E402
from autokernel.loop import actors, dispatch_guard, run  # noqa: E402

HARNESS = frozenset({"planner_transient", "lane_error", "bench_failed", "abstained"})
ANSWER = frozenset(dispatch_guard.ANSWER_STATUSES)
FORMATION_GATE = frozenset({
    "refused_at_formation", "hypothesis_rejected", "patch_rejected", "gate_refused",
    "hypothesis_retired", "refused_duplicate", "authoring_failed", "patch_rounds_exhausted",
    "scope_blocked", "integrity_refused", "resume_rejected", "stopped_mid_formation"})
GROUPS = ("HARNESS", "ANSWER", "FORMATION_GATE", "OTHER")
LIMIT = 12
RECENCY_LIMIT = 40


def group_of(status: str | None) -> str:
    if status in HARNESS:
        return "HARNESS"
    if status in ANSWER:
        return "ANSWER"
    if status in FORMATION_GATE:
        return "FORMATION_GATE"
    return "OTHER"


def norm_ts(stamp: str) -> str:
    """Fixed-width ISO-Z: pad/truncate the fraction to 6 digits."""
    base = stamp[:-1] if stamp.endswith("Z") else stamp
    whole, _, frac = base.partition(".")
    return f"{whole}.{(frac + '000000')[:6]}Z"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def static_row(rec: dict) -> dict:
    """An exported experiment line -> the fields recall() reads, plus epoch identity."""
    task = rec.get("task_key") or {}
    outcome = rec.get("outcome") or {}
    return {
        "attempt_id": rec.get("attempt_id"), "recorded_at": rec["recorded_at"],
        "norm": norm_ts(rec["recorded_at"]), "rowid": rec["rowid"],
        "campaign_id": rec.get("campaign_id"), "hypothesis_id": rec.get("hypothesis_id"),
        "mechanism_id": rec.get("mechanism_id"),
        "target_surface": task.get("target_surface"), "target_symbol": task.get("target_symbol"),
        "statement": rec.get("statement"), "status": rec.get("status") or "unknown",
        "effect_fraction": outcome.get("effect_fraction"),
        "refusal_reason": outcome.get("refusal_reason"),
        "epoch": task.get("epoch_sha256"), "meas": task.get("measurement_epoch"),
    }


class AsOfStore:
    """The slice of `ExperimentStore` that `run.prior_experiments` touches, over a list
    of already as-of-filtered static rows.  `recall` mirrors experiments.py's: SELECT with
    optional status IN, ORDER BY recorded_at DESC, rowid DESC, LIMIT pool; same-epoch
    marking (OP-60); fresh row dicts every call (rank() mutates); rank() when authorized."""

    def __init__(self, rows: Iterable[dict]):
        self._rows = sorted(rows, key=lambda r: (r["recorded_at"], r["rowid"]), reverse=True)

    def __enter__(self) -> "AsOfStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def recall(self, *, epoch: str, limit: int = 40, ranking_authorized: bool = False,
               include_claims: bool = False, statuses=None, measurement_epoch=None) -> list[dict]:
        pool = max(int(limit), experiments.RANKING_POOL) if ranking_authorized else int(limit)
        selected = [r for r in self._rows if not statuses or r["status"] in statuses][:pool]
        recalled = []
        for row in selected:
            same_epoch = row["epoch"] == epoch
            row_measurement = row["meas"]
            if row["epoch"] == epoch:
                row_measurement = measurement_epoch
            aliased = (not same_epoch and measurement_epoch is not None
                       and row_measurement == measurement_epoch)
            same_epoch = same_epoch or aliased
            out = {
                "attempt_id": row["attempt_id"], "recorded_at": row["recorded_at"],
                "campaign_id": row["campaign_id"], "hypothesis_id": row["hypothesis_id"],
                "mechanism_id": row["mechanism_id"], "target_surface": row["target_surface"],
                "target_symbol": row["target_symbol"], "statement": row["statement"],
                "falsifier": None, "status": row["status"],
                "effect_fraction": row["effect_fraction"],
                "exact_attribution_effect_fraction": None,
                "target_runtime_effect_fraction": None,
                "refusal_reason": row["refusal_reason"], "result_sha256": None,
                "same_epoch": same_epoch, "stale_epoch": not same_epoch,
                "comparable_measurement": same_epoch,
                "ranking_authorized": bool(ranking_authorized),
            }
            if include_claims:
                out["claims"] = None
            if measurement_epoch is not None:
                out.update(epoch_sha256=row["epoch"], measurement_epoch_sha256=row_measurement,
                           epoch_match=("measurement" if aliased else
                                        "full" if row["epoch"] == epoch else None),
                           research_scope=None)
            recalled.append(out)
        if not ranking_authorized:
            return recalled
        return experiments.rank(recalled)[:int(limit)]


def prior_rows(store: AsOfStore, epoch: str, measurement_epoch: str | None,
               rank: bool) -> list[dict]:
    """The REAL `run.prior_experiments`, with ExperimentStore swapped for `store`."""
    real = experiments.ExperimentStore
    experiments.ExperimentStore = lambda _path: store  # type: ignore[assignment]
    try:
        args = argparse.Namespace(store=Path("."), rank_prior_experiments=rank)
        return run.prior_experiments(args, epoch, measurement_epoch)
    finally:
        experiments.ExperimentStore = real  # type: ignore[assignment]


def already_tried_section(text: str) -> str:
    start = text.index("## Already tried")
    end = text.find("\n## ", start + 1)
    return text[start:] if end < 0 else text[start:end]


def always_shown(prior: list[dict], limit: int = LIMIT) -> tuple[list[dict], int]:
    """Selection logic of the `### Always shown` block, line for line (actors.py)."""
    newest_ids = {id(row) for row in list(prior)[:limit]}
    pinned_kept, seen_kept = [], set()
    for row in prior:
        if id(row) in newest_ids or not row.get("same_epoch"):
            continue
        if row.get("status") not in ("kept", "keep_candidate"):
            continue
        key = row.get("attempt_id") or id(row)
        if key in seen_kept:
            continue
        seen_kept.add(key)
        pinned_kept.append(row)
    pinned_abs, seen_fam = [], set()
    for row in prior:
        if id(row) in newest_ids or not row.get("same_epoch"):
            continue
        if row.get("status") not in ("abstained", "measured_null"):
            continue
        family = (actors._mechanism_family(row) or str(row.get("mechanism_id") or "")
                  or "(unlabeled)")
        if family in seen_fam:
            continue
        seen_fam.add(family)
        pinned_abs.append(row)
    pinned = pinned_kept + pinned_abs
    cap = max(limit * 4, 40)
    omitted = max(0, len(pinned) - cap)
    return (pinned[:cap] if omitted else pinned), omitted


def composition(rows: list[dict]) -> dict:
    counts = {g: 0 for g in GROUPS}
    other: dict[str, int] = {}
    for row in rows:
        group = group_of(row.get("status"))
        counts[group] += 1
        if group == "OTHER":
            other[str(row.get("status"))] = other.get(str(row.get("status")), 0) + 1
    return {"n": len(rows), "counts": counts, "other_statuses": dict(sorted(other.items()))}


def crosscheck(text: str, verbose: list[dict], pinned: list[dict], omitted: int) -> bool:
    """The rendered text must contain the expected lines in order -- proves the
    reimplemented block selection agrees with what render_context printed."""
    pos = 0
    for row in verbose:
        needle = f"- `{row.get('mechanism_id')}` → {row.get('status')} "
        pos = text.find(needle, pos)
        if pos < 0:
            return False
    if ("### Always shown" in text) != bool(pinned):
        return False
    pos = max(pos, text.find("### Always shown") if pinned else pos)
    for row in pinned:
        needle = f"- {row.get('status')} `{row.get('mechanism_id')}` ("
        pos = text.find(needle, pos)
        if pos < 0:
            return False
    return ("more omitted" in text) == bool(omitted)


def analyze_mode(prior: list[dict], answers_before: set[str]) -> dict:
    text = actors.render_context({"prior_experiments": prior}, limit=LIMIT)
    section = already_tried_section(text)
    verbose = list(prior)[:LIMIT]
    pinned, omitted = always_shown(prior)
    visible = verbose + pinned
    visible_ids = {r["attempt_id"] for r in visible}
    mechanisms = {r["mechanism_id"] for r in visible if r.get("mechanism_id")}
    mech_verbose = {r["mechanism_id"] for r in verbose if r.get("mechanism_id")}
    answer_visible = sum(1 for r in visible if r["status"] in ANSWER)
    return {
        "chars": len(section), "n_prior_rows": len(prior),
        "verbose12": composition(verbose), "always_shown": composition(pinned),
        "always_shown_omitted": omitted,
        "distinct_mechanisms_visible": len(mechanisms),
        "distinct_mechanisms_verbose12": len(mech_verbose),
        "any_answer_in_verbose12": any(r["status"] in ANSWER for r in verbose),
        "answers_visible_verbose_plus_always": answer_visible,
        "invisible_answers": len(answers_before - visible_ids),
        "crosscheck_ok": crosscheck(section, verbose, pinned, omitted),
    }


def replay_call(call: dict, rows_by_store: dict[str, list[dict]],
                norms_by_store: dict[str, list[str]],
                index: dict[tuple[str, int], dict]) -> dict:
    join = call.get("join") or {}
    base = {"call_id": call.get("call_id"), "campaign_dir": call.get("campaign_dir"),
            "started_at": call.get("started_at"), "store_label": join.get("store_label"),
            "join_rowid": join.get("rowid"), "prompt_bytes": call.get("prompt_bytes"),
            "reason": None, "modes": None}
    label, rowid = join.get("store_label"), join.get("rowid")
    if not join or label is None or rowid is None:
        base["reason"] = "unjoined"
        return base
    anchor = index.get((label, rowid))
    if anchor is None:
        base["reason"] = "join_row_missing"
        return base
    if anchor["epoch"] is None:
        base["reason"] = "no_call_epoch"
        return base
    cut = bisect.bisect_left(norms_by_store[label], norm_ts(call["started_at"]))
    before = rows_by_store[label][:cut]
    base["n_store_rows_before"] = len(before)
    if not before:
        base["reason"] = "empty_window"
        return base
    epoch, meas = anchor["epoch"], anchor["meas"]
    base["call_epoch12"] = epoch[:12]
    same = [r for r in before if r["epoch"] == epoch or (r["meas"] is not None and r["meas"] == meas)]
    answers_before = {r["attempt_id"] for r in same if r["status"] in ANSWER}
    base["n_same_epoch_before"] = len(same)
    base["n_same_epoch_answers_before"] = len(answers_before)
    store = AsOfStore(before)
    base["reason"] = "rendered"
    base["modes"] = {
        "recency": analyze_mode(prior_rows(store, epoch, meas, rank=False), answers_before),
        "rank": analyze_mode(prior_rows(store, epoch, meas, rank=True), answers_before),
    }
    return base


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "n": 0}
    return {"mean": round(statistics.fmean(values), 6),
            "median": round(statistics.median(values), 6), "n": len(values)}


def summarize_group(per_call: list[dict]) -> dict:
    reasons: dict[str, int] = {}
    for c in per_call:
        reasons[c["reason"]] = reasons.get(c["reason"], 0) + 1
    rendered = [c for c in per_call if c["reason"] == "rendered"]
    out: dict[str, Any] = {"n_calls": len(per_call), "reason_counts": dict(sorted(reasons.items())),
                           "rendered": len(rendered), "empty_window": reasons.get("empty_window", 0),
                           "unjoined": reasons.get("unjoined", 0), "modes": {}}
    for mode in ("recency", "rank"):
        ms = [c["modes"][mode] for c in rendered]
        entry: dict[str, Any] = {}
        for scope in ("verbose12", "always_shown"):
            with_rows = [m[scope] for m in ms if m[scope]["n"] > 0]
            entry[f"{scope}_share"] = {
                g: _stats([s["counts"][g] / s["n"] for s in with_rows]) for g in GROUPS}
            entry[f"{scope}_calls_with_rows"] = len(with_rows)
            totals: dict[str, int] = {}
            for m in ms:
                for s, n in m[scope]["other_statuses"].items():
                    totals[s] = totals.get(s, 0) + n
            entry[f"{scope}_other_status_rows"] = dict(sorted(totals.items()))
        entry["already_tried_chars"] = _stats([m["chars"] for m in ms])
        n = len(ms) or 1
        entry["frac_zero_answer_in_verbose12"] = round(
            sum(1 for m in ms if not m["any_answer_in_verbose12"]) / n, 6)
        entry["frac_zero_answer_in_verbose_plus_always"] = round(
            sum(1 for m in ms if m["answers_visible_verbose_plus_always"] == 0) / n, 6)
        entry["invisible_answers"] = _stats([m["invisible_answers"] for m in ms])
        entry["same_epoch_answers_before"] = _stats(
            [c["n_same_epoch_answers_before"] for c in rendered])
        entry["distinct_mechanisms_visible"] = _stats([m["distinct_mechanisms_visible"] for m in ms])
        entry["distinct_mechanisms_verbose12"] = _stats(
            [m["distinct_mechanisms_verbose12"] for m in ms])
        entry["crosscheck_failures"] = sum(1 for m in ms if not m["crosscheck_ok"])
        out["modes"][mode] = entry
    return out


def build(root: Path, hist: Path, calls: Path, out_dir: Path, argv: list[str]) -> tuple[Path, Path]:
    records = load_jsonl(hist)
    exp = [static_row(r) | {"store_label": r["store_label"]}
           for r in records if r.get("record_kind") == "experiment"]
    rows_by_store: dict[str, list[dict]] = {}
    for r in exp:
        rows_by_store.setdefault(r["store_label"], []).append(r)
    for label, rows in rows_by_store.items():
        rows.sort(key=lambda r: (r["norm"], r["rowid"]))
        ids = [r["attempt_id"] for r in rows]
        assert len(ids) == len(set(ids)), f"duplicate attempt_id in {label}"
    norms_by_store = {k: [r["norm"] for r in v] for k, v in rows_by_store.items()}
    index = {(r["store_label"], r["rowid"]): r for r in exp}
    planner = [c for c in load_jsonl(calls)
               if c.get("record_kind") == "actor_call" and c.get("role") == "planner"]
    planner.sort(key=lambda c: (c["started_at"], c["call_id"]))
    per_call = [replay_call(c, rows_by_store, norms_by_store, index) for c in planner]

    out_dir.mkdir(parents=True, exist_ok=True)
    per_path = out_dir / "render_audit_per_call.jsonl"
    with open(per_path, "w", encoding="utf-8") as handle:
        for c in per_call:
            handle.write(json.dumps(c, sort_keys=True) + "\n")

    by_campaign: dict[str, list[dict]] = {}
    for c in per_call:
        by_campaign.setdefault(c["campaign_dir"], []).append(c)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=False).stdout.strip()
    summary = {
        "config": {
            "argv": argv, "python": sys.version, "git_head": head,
            "input_sha256": {"history_records.jsonl": sha256_file(hist),
                             "actor_call_records.jsonl": sha256_file(calls)},
            "limit": LIMIT, "recency_recall_limit": RECENCY_LIMIT,
            "ranking_pool": experiments.RANKING_POOL,
            "pinned_history_pool": run._PINNED_HISTORY_POOL,
            "pinned_history_statuses": list(run._PINNED_HISTORY_STATUSES),
            "status_groups": {"HARNESS": sorted(HARNESS), "ANSWER": sorted(ANSWER),
                              "FORMATION_GATE": sorted(FORMATION_GATE),
                              "OTHER": "every other status (named per scope in *_other_status_rows)"},
            "proxies": {
                "as_of_cut": "row.recorded_at < call.started_at on 27-char normalised "
                             "YYYY-MM-DDTHH:MM:SS.ffffffZ (whole seconds padded .000000); "
                             "row order = raw recorded_at DESC then rowid DESC (as the SQL)",
                "call_epoch": "task_key.epoch_sha256 / measurement_epoch of the row named by "
                              "call.join (store_label+rowid), i.e. the NEXT recorded attempt "
                              "(time-window heuristic, no key join)",
                "same_epoch": "row epoch == call epoch OR (row measurement_epoch not None and "
                              "== call measurement_epoch); verified aliases not exported",
                "unexported_fields_none": ["falsifier", "result_sha256", "claims",
                                           "exact_attribution_effect_fraction",
                                           "target_runtime_effect_fraction"],
                "shares": "mean/median over rendered calls of per-call (group rows / rows shown); "
                          "always_shown shares only over calls whose block is non-empty",
                "invisible_answers": "same-epoch ANSWER rows before the call that are in neither "
                                     "the verbose 12 nor the visible Always-shown block",
                "render": "real actors.render_context({'prior_experiments': rows}, limit=12); "
                          "Already-tried section = '## Already tried' to next '\\n## '",
                "run_py": "imported; real run.prior_experiments / _with_pinned_history used with "
                          "ExperimentStore swapped for AsOfStore",
            },
        },
        "overall": summarize_group(per_call),
        "per_campaign_dir": {k: summarize_group(v) for k, v in sorted(by_campaign.items())},
    }
    sum_path = out_dir / "render_audit_summary.json"
    with open(sum_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return per_path, sum_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ns = ap.parse_args()
    data = ns.root / "data" / "ak-history-replay-20261020"
    build(ns.root, data / "history_records.jsonl", data / "actor_call_records.jsonl",
          data / "ak-hr-2", sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
