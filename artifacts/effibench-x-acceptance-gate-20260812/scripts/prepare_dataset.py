#!/usr/bin/env python
"""Download EffiBench/effibench-x, filter to DATED problems, unpack per-problem JSONs.

Mirrors upstream hf_dataset.py download (which uses a removed `use_auth_token`
kwarg under datasets 5.x) but additionally filters to rows with a non-null
release_timestamp — the 308 DATED problems (intake-952 dive item 6).
"""
import json
from pathlib import Path
from datasets import load_dataset

OUT = Path("/workspace/tmp/effibench-gate/data/dataset")
OUT.mkdir(parents=True, exist_ok=True)

ds = load_dataset("EffiBench/effibench-x", split="test")
print(f"total rows: {len(ds)}")

dated = 0
sources = {}
for row in ds:
    ts = row.get("release_timestamp")
    if ts is None:
        continue
    dated += 1
    src = row.get("source", "")
    sources[src] = sources.get(src, 0) + 1
    filename = f"{src}_{row['id']}_{row['title_slug']}.json"
    with open(OUT / filename, "w") as f:
        json.dump(row, f, indent=2)

print(f"dated rows written: {dated}")
print(f"by source: {sources}")
