#!/usr/bin/env python3
"""Replay `render_context`'s two diminishing-returns triggers over the closed AK history.

Decision input for 2026-10-24 ("should `abstained` / `refused_at_formation` count as
terminal failures?"). Reads AK-HR-1's `history_records.jsonl` only; writes one JSON.

Per campaign, rows in recorded_at order; before each row the trigger sees the previous
40 rows newest-first (the pre-2026-10-20 recency recall window). Three arms of status
sets: the pre-decision sets, without `abstained`, and without both. Reported: how often
each block fires, how many families it names, and how many family firings exist only
because of the excluded statuses. Reads no magnitude. Limitation: AK-HR-1 records do not
carry the abstention's free-text `mechanism_family`, so a keyword-family match on that
text alone is not replayed.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from autokernel.loop import actors

WINDOW = 40
PRE_DECISION_EXHAUSTION = frozenset({"measured_null", "regression", "refused_at_formation",
                                     "authoring_refused", "screened_out",
                                     "hypothesis_retired", "abstained"})
PRE_DECISION_STAGNATION = frozenset({"measured_null", "regression", "refused_at_formation",
                                     "runtime_refused", "abstained"})
ARMS = {
    "pre_decision": (PRE_DECISION_EXHAUSTION, PRE_DECISION_STAGNATION),
    "no_abstained": (PRE_DECISION_EXHAUSTION - {"abstained"},
                     PRE_DECISION_STAGNATION - {"abstained"}),
    "no_abstained_no_refused_at_formation": (
        PRE_DECISION_EXHAUSTION - {"abstained", "refused_at_formation"},
        PRE_DECISION_STAGNATION - {"abstained", "refused_at_formation"}),
}


def _row(record: dict) -> dict:
    task = record.get("task_key") or {}
    return {"status": record.get("status"), "mechanism_id": record.get("mechanism_id"),
            "statement": record.get("statement"), "recorded_at": record.get("recorded_at"),
            "target_surface": task.get("target_surface"),
            "target_symbol": task.get("target_symbol")}


def audit(records: list[dict]) -> dict:
    by_campaign: dict[str, list[dict]] = collections.defaultdict(list)
    for record in records:
        by_campaign[str(record.get("campaign_dir"))].append(record)
    out = {}
    for campaign, rows in sorted(by_campaign.items()):
        rows = [_row(r) for r in sorted(rows, key=lambda r: r.get("recorded_at") or "")]
        arms = {arm: collections.Counter() for arm in ARMS}
        only_with = collections.Counter()
        for index in range(len(rows)):
            prior = list(reversed(rows[max(0, index - WINDOW):index]))
            result = {arm: actors.diminishing_returns_triggers(
                prior, exhaustion_statuses=sets[0], stagnation_statuses=sets[1])
                for arm, sets in ARMS.items()}
            for arm, (exhausted, stagnant) in result.items():
                arms[arm]["calls"] += 1
                arms[arm]["exhaustion_block_fires"] += bool(exhausted)
                arms[arm]["stagnation_block_fires"] += bool(stagnant)
                arms[arm]["exhaustion_families"] += len(exhausted)
                arms[arm]["stagnation_families"] += len(stagnant)
            base_ex, base_st = result["pre_decision"]
            na_ex, na_st = result["no_abstained"]
            nn_ex, nn_st = result["no_abstained_no_refused_at_formation"]
            only_with["exhaustion_families_needing_abstained"] += sum(f not in na_ex for f in base_ex)
            only_with["stagnation_families_needing_abstained"] += sum(f not in na_st for f in base_st)
            only_with["exhaustion_families_needing_refused_at_formation"] += sum(
                f not in nn_ex for f in na_ex)
            only_with["stagnation_families_needing_refused_at_formation"] += sum(
                f not in nn_st for f in na_st)
        statuses = collections.Counter(r["status"] for r in rows)
        identity = collections.Counter(
            f"{r['status']}:{'mechanism' if r['mechanism_id'] else 'no_mechanism'}:"
            f"{'surface_symbol' if r['target_surface'] and r['target_symbol'] else 'no_target'}"
            for r in rows if r["status"] in ("abstained", "refused_at_formation"))
        out[campaign] = {"rows": len(rows), "arms": {a: dict(c) for a, c in arms.items()},
                         "family_firings_only_with": dict(only_with),
                         "abstained_rows": statuses["abstained"],
                         "refused_at_formation_rows": statuses["refused_at_formation"],
                         "identity": dict(identity)}
    return {"schema": "epyc.autokernel.render_trigger_audit.v1", "window": WINDOW,
            "recall": "recency (pre-2026-10-20 default)", "campaigns": out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line]
    args.out.write_text(json.dumps(audit(records), indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
