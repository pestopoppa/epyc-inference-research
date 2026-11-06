import json
import os
from pathlib import Path
import tempfile

from . import ak_setup1_store_replay as r
from ..controller import build_store_lookup as bsl


def _contract(patch, base="b1"):
    c = {k: {"x": 1} for k in bsl._KEY_FIELDS}
    c.update(patch_sha256=patch, production_base_authority={"c": base}, build_key="bk" + patch, parallelism={"jobs": 1})
    return c


def _entry(root, dep, name, c, t, with_logs=True):
    d = Path(root) / dep / "operations" / "build-cache" / "entries" / name
    (d / "logs").mkdir(parents=True)
    p = d / "intent.json"
    p.write_text(json.dumps({"build_contract": c}))
    os.utime(p, (t, t))
    if with_logs:
        for f, dt in (("akc-anchor.log", 100), ("akc-discovery-1.log", 300)):
            q = d / "logs" / f
            q.write_text("x")
            os.utime(q, (t + dt, t + dt))


def test_hits_and_seconds():
    root = tempfile.mkdtemp()
    _entry(root, "d1", "e1", _contract("p1"), 1000)
    _entry(root, "d2", "e2", _contract("p2"), 2000)           # same anchor, new patch
    _entry(root, "d3", "e3", _contract("p1"), 3000, False)    # same patch AND anchor, no logs
    _entry(root, "d4", "e4", _contract("p3", base="b2"), 4000)  # new base: all miss
    res = r.replay(r.load(root))
    assert res["n_entries"] == 4
    assert res["anchor_tree_hits"] == 2 and res["candidate_tree_hits"] == 1
    assert res["baseline_build_key_hits"] == 1
    # e2 anchor hit: 100 s from logs; e3 anchor+candidate hit: fallback twice
    assert res["skipped_seconds_estimate"] == 100.0 + 2 * r.FALLBACK_TREE_SECONDS


def test_unreadable_entries_are_skipped():
    root = tempfile.mkdtemp()
    d = Path(root) / "d" / "operations" / "build-cache" / "entries" / "e"
    d.mkdir(parents=True)
    (d / "intent.json").write_text("{nope")
    assert r.replay(r.load(root))["n_entries"] == 0
