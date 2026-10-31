"""Explicit AutoKernel lesson submission; retrieval remains disabled."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .journal import Journal
from .lesson_native import produce


def main(argv=None, *, out=None):
    parser = argparse.ArgumentParser(prog="campaign submit-lesson")
    parser.add_argument("--config", required=True, help="native producer JSON")
    parser.add_argument("--journal", required=True, help="AutoKernel journal root")
    parser.add_argument("--control", action="store_true", help="isolated pipeline control only")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    provenance = config.get("cap_provenance")
    expected_kind = "control" if args.control else "workload_roofline"
    valid = (isinstance(provenance, dict)
             and set(provenance) == {"kind", "source"}
             and provenance.get("kind") == expected_kind
             and isinstance(provenance.get("source"), str)
             and bool(provenance["source"].strip()))
    if args.control and not str(config.get("task_id", "")).startswith("CONTROL-"):
        raise ValueError("control task_id must start CONTROL-")
    if not valid:
        # An unproven ceiling is not an admission policy.  produce records
        # UNKNOWN without starting either native execution.
        config["upper_cap"] = None
        config["cap_provenance"] = None
    entry = produce(Journal(args.journal), **config)
    print(json.dumps({"event_id": entry.event_id, "decision": entry.payload["decision"],
                      "reason": entry.payload["reason"], "control": args.control}),
          file=out or sys.stdout)
    return 0
