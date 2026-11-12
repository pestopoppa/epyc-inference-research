#!/usr/bin/env python3
"""AGG-1.10: zero-model ceiling and pilot selection (PREDECLARED_AGG110.md "Frozen derived files").

For each of the 26 affected questions: compile the reference ``agg110.v1`` statement from the golds, execute it on a read-only
in-memory copy of the AGG-1.1 store (pivot view, null guard), score it with the frozen scorer. Writes ``agg110_ceiling.json``
and ``agg110_pilot.json``. No model, no server, the store file is opened ``mode=ro``.

    python3 -I -B agg110_ceiling.py --orch-root <orchestrator tree> --store <AGG-1.1 store.sqlite> [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "agg_phase0"))
import agg110 as G  # noqa: E402
import phase0_arms as P  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orch-root", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--set-dir", type=Path, default=HERE)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: the set dir (the frozen files)")
    ap.add_argument("--corpus-id", default=P.CORPUS_ID)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(a.orch_root))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    miss = json.loads((a.set_dir / "agg18_miss_table.json").read_text())
    golds = {}
    for line in (a.set_dir / "golds.jsonl").read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            golds[g["id"]] = g
    corpus = json.loads((a.set_dir / "corpus.json").read_text())
    ceil = G.ceiling_table(miss, golds, corpus, a.store, a.corpus_id)
    ceil["reference_on_gold_cells"] = G.gold_cell_check(miss, golds, corpus, a.store, a.corpus_id, a.set_dir / "facts_sheet.csv")
    pilot = G.pilot_selection(ceil)
    out = a.out_dir or a.set_dir
    (out / "agg110_ceiling.json").write_text(json.dumps(ceil, indent=1, sort_keys=True) + "\n")
    (out / "agg110_pilot.json").write_text(json.dumps(pilot, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"n": ceil["n"], "n_ceiling_true": ceil["n_ceiling_true"], "n_valid": ceil["n_valid"],
                      "longest_statement_tokens_est": ceil["longest_statement_tokens_est"],
                      "within_60pct_of_cap": ceil["longest_within_60pct_of_cap"],
                      "reference_on_gold_cells": ceil["reference_on_gold_cells"], "pilot": pilot["qids"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
