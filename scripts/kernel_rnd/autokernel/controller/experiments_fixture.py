#!/usr/bin/env python3
"""Load the REAL experiment slice that `experiments_fixture.json` carries.

WHY A REAL SLICE AND NOT A HAND-BUILT ONE
-----------------------------------------
A fixture invented from the reader's expectations tests the reader, not the store.
The two facts this fixture exists to expose are both *interleaving* facts and neither
survives being hand-written:

  * the newest 40 rows of a 1,002-row store contain 18 of its 313 distinct
    mechanisms, so recency truncation hides 94% of what has already been tried; and
  * for the first ~20 rows of every new epoch, the recency window is dominated by the
    PREVIOUS epoch, and `render_context`'s "Characterised — do NOT re-measure" block
    pooled those cross-epoch magnitudes into one median.

Both were measured against `/mnt/raid0/llm/autokernel/loop-memory/experiments.db` and
both reproduce from this slice, which is 200 contiguous rows straddling the start of
epoch `6a4dccec`. `_provenance` in the JSON records exactly what was elided (the
`payload` blob and prose over 400 characters, for file size) and what was not: no
column that `recall()` reads is altered.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

FIXTURE = Path(__file__).resolve().parent / "experiments_fixture.json"

#: The epoch the slice moves INTO. Recall with this as `epoch` and the older 140 rows
#: are the stale ones -- which is the state a run is in for its first iterations after
#: any rebuild, and therefore the state the conformance property has to hold in.
CURRENT_EPOCH = "6a4dccec34576723b2598d6eafc813489f5a068ebfbf95c66a2277832c6783a8"
PRIOR_EPOCH = "f6d6cca0b891133a560ee8750322e1a4c7d2d187ba185b72e0418bbcfd8efb95"


def rows() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["rows"]


def store(root: Path) -> Any:
    """An `ExperimentStore` at `root` preloaded with the real slice.

    Rows are inserted directly rather than through `record()` because their
    `attempt_id`s are the real ones the loop minted; re-deriving them would quietly
    substitute this file's idea of identity for the store's.
    """
    from . import experiments

    opened = experiments.ExperimentStore(root)
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    columns = document["columns"]
    placeholders = ",".join("?" * len(columns))
    connection: sqlite3.Connection = opened._connection
    connection.executemany(
        f"INSERT OR IGNORE INTO experiments ({','.join(columns)}) "
        f"VALUES ({placeholders})",
        [tuple(row[column] for column in columns) for row in document["rows"]])
    connection.commit()
    return opened


__all__ = ["CURRENT_EPOCH", "FIXTURE", "PRIOR_EPOCH", "rows", "store"]
