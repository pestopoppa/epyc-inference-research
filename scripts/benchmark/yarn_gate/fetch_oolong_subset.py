#!/usr/bin/env python3
"""YARN-AGG-1: fetch ONLY the oolong-synth lengths the YaRN gate uses. OPERATOR-VISIBLE STEP.

The full dataset is a 12 GB download (about 25 min unauthenticated). This tool reads the parquet footers and
the small ``context_len`` column first, then reads only the row groups whose rows have a wanted nominal
length, and ONLY the columns the OOL-1 adapter needs (never ``context_window_text_with_labels``), over HTTP
range requests through ``HfFileSystem``. Output is a JSONL per length under ``--out``.

Guards (nothing touches the network unless confirmed):
  --plan            prints what would happen; no network. Always safe.
  --list-lengths    network, METADATA ONLY (footers + context_len column). Needs OPERATOR_FETCH_CONFIRMED=1.
  --fetch           network, downloads the subset. Needs OPERATOR_FETCH_CONFIRMED=1.
Requires ``pyarrow`` and ``huggingface_hub`` (run in the orchestrator venv). NOT RUN by the lane that wrote it.
After the fetch, run ``retokenize`` over the written contexts with the served tokenizer (server needed) to
record real prompt_n; the acceptance clause "token counts re-measured" is that step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = "oolongbench/oolong-synth"
REVISION = "81d2f8d9349f6ff329d695f9be494051e9f05cb9"
NEEDED = ["id", "context_window_id", "dataset", "question", "answer", "answer_type", "task_group", "task",
          "context_len", "num_labels", "context_window_text"]
DEFAULT_LENGTHS = [131072, 262144, 524288]
DEFAULT_OUT = "/mnt/raid0/llm/tmp/yarn-agg-oolong-subset"


def need_confirmed() -> None:
    if os.environ.get("OPERATOR_FETCH_CONFIRMED") != "1":
        sys.exit("REFUSED: set OPERATOR_FETCH_CONFIRMED=1 (operator-visible network step; one at a time, see "
                 "feedback no-concurrent-downloads)")


def plan(lengths, out) -> dict:
    return {"repo": REPO, "revision": REVISION, "split": "test", "columns": NEEDED,
            "excluded_column": "context_window_text_with_labels", "context_len_values": lengths,
            "out_dir": out, "network": False}


def _shards(fs, split: str):
    base = f"datasets/{REPO}@{REVISION}/data"
    return sorted(p for p in fs.ls(base, detail=False) if f"/{split}-" in p)


def run(args) -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen: dict[int, int] = {}
    writers = {}
    try:
        for shard in _shards(fs, args.split):
            with fs.open(shard, "rb") as f:
                pf = pq.ParquetFile(f)
                for rg in range(pf.num_row_groups):
                    lens = pf.read_row_group(rg, columns=["context_len"]).column("context_len").to_pylist()
                    for L in set(lens):
                        seen[L] = seen.get(L, 0) + lens.count(L)
                    if args.list_lengths or not (set(lens) & set(args.lengths)):
                        continue
                    tbl = pf.read_row_group(rg, columns=NEEDED)
                    for row in tbl.to_pylist():
                        if row["context_len"] in args.lengths:
                            w = writers.setdefault(row["context_len"], (out / f"oolong_synth_{args.split}_{row['context_len']}.jsonl").open("a"))
                            w.write(json.dumps(row) + "\n")
    finally:
        for w in writers.values():
            w.close()
    print(json.dumps({"context_len_counts": dict(sorted(seen.items())), "revision": REVISION}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=DEFAULT_OUT)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--list-lengths", action="store_true")
    g.add_argument("--fetch", action="store_true")
    a = ap.parse_args(argv)
    if a.plan:
        print(json.dumps(plan(a.lengths, a.out), indent=1))
        return 0
    need_confirmed()
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
