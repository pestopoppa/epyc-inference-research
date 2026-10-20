#!/usr/bin/env python3
"""Closed-store history exporter and census (AK-HR-1).

Usage:
    python3 scripts/kernel_rnd/autokernel/history_replay/export.py --out-dir DIR [--root ROOT]

Writes into DIR:
    history_records.jsonl     one row per experiments row (+ one per runtime-arms attempt)
    actor_call_records.jsonl  one row per actor call (actor_call.v1 paired with its metrics line)
    census.json               answers (a)-(g)
Plus two files that are NOT part of the byte-identical contract (they hold run-time evidence):
    run_provenance.json       liveness evidence, mtimes before/after, config echo
Outputs 1-3 are deterministic functions of the closed stores (sorted keys, no wall clock).

Read-only contract: SQLite URI `mode=ro&immutable=1` (immutable is only used after the preflight
proves each `-wal` file is empty, so no committed frame can be skipped); nothing is ever opened
for write under the autokernel tree; store/WAL/SHM mtimes are recorded before and after and the
run fails if any moved.
"""
from __future__ import annotations

import argparse
import bisect
import calendar
import glob
import hashlib
import json
import math
import os
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "epyc.autokernel.history_replay.v1"
DEFAULT_ROOT = "/mnt/raid0/llm/autokernel"
LIVE_WINDOW_S = 30 * 60
HARNESS_STATUSES = ("planner_transient", "lane_error", "bench_failed", "abstained")


# --------------------------------------------------------------------------- discovery
def discover(root: Path):
    """Return (candidate_stores, excluded) where excluded are (path, reason)."""
    stores, excluded = [], []
    lm = root / "loop-memory"
    if (lm / "experiments.db").is_file():
        stores.append(lm / "experiments.db")
    for name, why in (("experiments.sqlite", "empty or non-experiments file"),
                      ("experiments.db.pre-retraction", "pre-retraction backup snapshot of loop-memory/experiments.db")):
        p = lm / name
        if p.exists():
            excluded.append((str(p), why + f" (size {p.stat().st_size})"))
    for d in sorted(glob.glob(str(root / "campaigns" / "*" / "store*"))):
        dp = Path(d)
        if not dp.is_dir():
            continue
        db = dp / "experiments.db"
        if db.is_file():
            stores.append(db)
        else:
            excluded.append((str(dp), "store directory without experiments.db"))
    return stores, excluded


def _locked_inodes() -> set[int]:
    out: set[int] = set()
    try:
        with open("/proc/locks") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 6 and ":" in parts[5]:
                    out.add(int(parts[5].split(":")[2]))
    except OSError:
        pass
    return out


def _live_procs(needle: str) -> list[int]:
    pids = []
    me = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                cmd = fh.read().decode("utf-8", "replace")
        except OSError:
            continue
        if needle in cmd:
            pids.append(int(entry))
    return sorted(pids)


def liveness(db: Path, now: float, locked: set[int]) -> dict:
    """Evidence that a store is (not) held by a live run.  Never opens a store file."""
    sd = db.parent
    ev = {"store": str(db), "reasons": []}
    for f in (db, Path(str(db) + "-wal")):
        if f.exists():
            age = now - f.stat().st_mtime
            if age < LIVE_WINDOW_S:
                ev["reasons"].append(f"{f.name} mtime moved {age:.0f}s ago (< {LIVE_WINDOW_S}s)")
    wal = Path(str(db) + "-wal")
    if wal.exists() and wal.stat().st_size > 0:
        ev["reasons"].append(f"-wal non-empty ({wal.stat().st_size} bytes): uncheckpointed frames")
    inodes = []
    for f in sd.iterdir():
        if f.is_file():
            try:
                if f.stat().st_ino in locked:
                    inodes.append(f.name)
            except OSError:
                pass
    if inodes:
        ev["reasons"].append(f"files hold a kernel lock: {sorted(inodes)}")
    cw = sd / "cpu-window.json"
    if cw.exists():
        try:
            with open(cw) as fh:
                j = json.load(fh)
            exp = j.get("expires_at")
            expired = False
            if exp:
                try:
                    expired = calendar.timegm(time.strptime(exp[:19], "%Y-%m-%dT%H:%M:%S")) < now
                except ValueError:
                    expired = False
            if (j.get("phase") != "exited" or j.get("loop_holds_claim")) and not expired:
                ev["reasons"].append(f"cpu-window {cw} phase={j.get('phase')} holds={j.get('loop_holds_claim')} "
                                     f"unexpired")
            elif (j.get("phase") != "exited" or j.get("loop_holds_claim")) and expired:
                ev["stale_window_ignored"] = (f"{cw} says phase={j.get('phase')} holds={j.get('loop_holds_claim')} "
                                              f"but expires_at {exp} has passed (loop dead per the window contract)")
        except (OSError, ValueError):
            ev["reasons"].append(f"{cw} unreadable")
    procs = _live_procs(str(sd))
    camp = sd.parent
    if camp.parent.name == "campaigns":
        procs = sorted(set(procs) | set(_live_procs(str(camp))))
    if procs:
        ev["reasons"].append(f"processes whose argv mentions the store/campaign: {procs}")
    ev["live"] = bool(ev["reasons"])
    return ev


# --------------------------------------------------------------------------- helpers
def jdump(o) -> str:
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def target_group(db: Path) -> str:
    camp = db.parent.parent.name
    if db.parent.parent.name == "loop-memory" or db.parent.name == "loop-memory":
        return "legacy-loop-memory"
    for key in ("ds41-cpu-decode", "q38fn-cpu-decode", "27b-gpu-verify"):
        if key in camp:
            return key
    return camp


def campaign_dir(db: Path) -> str:
    return "loop-memory" if db.parent.name == "loop-memory" else db.parent.parent.name


def store_label(db: Path) -> str:
    return "loop-memory" if db.parent.name == "loop-memory" else f"{db.parent.parent.name}/{db.parent.name}"


def ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)


def failure_class(d: dict):
    for path in (("integrity_screen", "refusal_class"), ("hypothesis_pending", "last_rejection", "class"),
                 ("hypothesis_pending", "class")):
        cur = d
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if isinstance(cur, str):
            return {"source": ".".join(path), "value": cur}
    rg = d.get("refusal_gate")
    if isinstance(rg, str):
        return {"source": "refusal_gate", "value": rg}
    return None


def knob_descriptor(d: dict):
    rp = d.get("runtime_pair")
    if isinstance(rp, dict) and isinstance(rp.get("dimension"), dict):
        dim = rp["dimension"]
        return {"type": "runtime_dimension", "kind": dim.get("kind"), "dimension_id": dim.get("dimension_id"),
                "anchor": dim.get("anchor"), "candidate": dim.get("candidate")}
    return None


#: AK-HR-3 write-side fields (`loop/history_fields.py`). Read only when present: a row
#: written before AK-HR-3 gets NO `history_fields` key, so outputs over old stores stay
#: byte-identical to the AK-HR-1 run.
HR3_KEYS = ("trial_id", "trial_ordinal", "cumulative_tokens", "stage_seconds", "failure_class",
            "mechanism_family_bucket", "knob_descriptor")


def hr3_fields(d: dict):
    """The AK-HR-3 fields of one payload or metrics line, each null when absent; None when
    the line carries none of them (an old row)."""
    if not isinstance(d, dict) or not any(k in d for k in HR3_KEYS):
        return None
    return {k: d.get(k) for k in HR3_KEYS}


def comparison_scalars(cmp):
    if not isinstance(cmp, dict):
        return None
    keep = ("schema", "surface", "metric", "np", "pairs", "effect", "effect_pct", "noise_floor_pct", "decisive",
            "admission", "baseline_scope", "anchor_tok_s", "candidate_tok_s", "recipe_hash", "recipe",
            "device_seconds", "estimator", "drifting", "anchor_drift_pct", "candidate_drift_pct")
    out = {k: cmp[k] for k in keep if k in cmp and not isinstance(cmp[k], (dict, list))}
    for side in ("anchor_samples", "candidate_samples"):
        v = cmp.get(side)
        if isinstance(v, list):
            out["n_" + side] = sum(1 for x in v if num(x) is not None)
    return out or None


def measure_window_s(cmp) -> float | None:
    if not isinstance(cmp, dict):
        return None
    tot, seen = 0.0, False
    for side in ("anchor_residency", "candidate_residency"):
        for r in cmp.get(side) or []:
            w = num(r.get("window_s")) if isinstance(r, dict) else None
            if w is not None:
                tot += w
                seen = True
    return round(tot, 6) if seen else None


# --------------------------------------------------------------------------- store read
def read_store(db: Path):
    st = db.stat()
    conn = ro(db)
    n = conn.execute("select count(*) from experiments").fetchone()[0]
    size = st.st_size
    tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    aliases = {}
    if "epoch_aliases" in tables:
        for full, meas in conn.execute("select full_epoch, measurement_epoch from epoch_aliases"):
            aliases[full] = meas
    have = [r[1] for r in conn.execute("pragma table_info(experiments)")]
    want = ["attempt_id", "recorded_at", "campaign_id", "deployment", "epoch_sha256", "hypothesis_id", "mechanism_id",
            "target_surface", "target_symbol", "statement", "status", "effect_fraction", "exact_effect",
            "target_effect", "refusal_reason", "spawn_parent", "branch_id", "width", "depth", "payload"]
    cols = "rowid, " + ", ".join(c if c in have else "NULL" for c in want)
    rows = []
    for r in conn.execute(f"select {cols} from experiments order by rowid"):
        (rowid, aid, rec, cid, dep, ep, hid, mech, tsurf, tsym, stmt, status, eff, exact, teff, refr, sp, br, w, dp,
         payload) = r
        d = json.loads(payload)
        cmp = d.get("comparison")
        cs = comparison_scalars(cmp)
        surf = tsurf if tsurf is not None else (cs or {}).get("surface")
        rows.append({
            "record_kind": "experiment",
            "store_path": str(db), "store_label": store_label(db), "store_size_bytes": size, "store_row_count": n,
            "campaign_dir": campaign_dir(db), "target_group": target_group(db),
            "rowid": rowid, "attempt_id": aid, "recorded_at": rec, "campaign_id": cid, "deployment": dep,
            "turn_recorded_at": d.get("turn_recorded_at"),
            "task_key": {"target_surface": surf, "target_symbol": tsym, "epoch_sha256": ep,
                         "measurement_epoch": aliases.get(ep)},
            "status": status, "hypothesis_id": hid, "mechanism_id": mech, "statement": stmt,
            "candidate": {"mechanism_id": mech, "knob": knob_descriptor(d),
                          "candidate_diff_sha256": d.get("candidate_diff_sha256")},
            "outcome": {"status": status, "failure_class": failure_class(d), "refusal_reason": refr,
                        "effect_fraction": eff, "exact_effect": exact, "target_effect": teff,
                        "comparison": cs, "measure_window_s_sum": measure_window_s(cmp),
                        "measure_device_seconds": num((cmp or {}).get("device_seconds")) if isinstance(cmp, dict) else None,
                        "stage_seconds": {"planner": None, "critic": None, "build": None, "correctness": None,
                                          "measure": None}},
            "lineage": {"spawn_parent": sp, "branch_id": br, "width": w, "depth": dp,
                        "hypothesis_round": d.get("hypothesis_round"), "patch_round": d.get("patch_round")},
            **({"history_fields": hr3} if (hr3 := hr3_fields(d)) else {}),
            "payload_bytes": len(payload),
            "payload_flags": {"has_call_id_key": '"call_id"' in payload,
                              "has_retained_patch": "retained_patch" in d,
                              "has_candidate_diff_sha256": bool(d.get("candidate_diff_sha256")),
                              "top_level_keys": sorted(d.keys())},
        })
    conn.close()
    return rows, n, size, aliases


# --------------------------------------------------------------------------- actor calls
def read_calls(path: Path, camp: str):
    """Pair each metrics line with the actor_call.v1 line that follows it (same role, ts == finished_at)."""
    lines = []
    with open(path, encoding="utf-8") as f:
        for i, l in enumerate(f, 1):
            l = l.strip()
            if l:
                lines.append((i, json.loads(l)))
    out, unpaired_metrics, unpaired_calls = [], 0, 0
    i = 0
    used = set()
    for idx, (ln, d) in enumerate(lines):
        if d.get("schema") != "epyc.autokernel.actor_call_metrics.v1":
            continue
        call = None
        for j in (idx + 1, idx - 1):
            if 0 <= j < len(lines) and lines[j][1].get("schema") == "epyc.autokernel.actor_call.v1" \
                    and j not in used and lines[j][1].get("role") == d.get("role") \
                    and _near(lines[j][1].get("finished_at"), d.get("ts")):
                call = (j, lines[j][1])
                break
        if call:
            used.add(call[0])
        else:
            unpaired_metrics += 1
        c = call[1] if call else {}
        oc = (d.get("opencode") or {}).get("totals") or {}
        ak = d.get("ak_check") or {}
        out.append({
            "record_kind": "actor_call",
            "file": str(path), "campaign_dir": camp, "metrics_line": ln, "call_line": lines[call[0]][0] if call else None,
            "call_id": c.get("call_id"), "run_id": c.get("run_id"), "role": d.get("role"),
            "backend_kind": d.get("backend_kind"), "backend_model": d.get("backend_model"),
            "started_at": c.get("started_at"), "finished_at": c.get("finished_at") or d.get("ts"),
            "ts": d.get("ts"), "wall_s": num(d.get("wall_s")), "returncode": d.get("returncode"),
            "timed_out": d.get("timed_out"), "failure_class": d.get("failure_class"),
            "prompt_bytes": (c.get("prompt") or {}).get("bytes"),
            "tokens": ({"decoded": oc.get("decoded_tokens"), "prompt": oc.get("prompt_tokens"),
                        "cache_read": oc.get("cache_read_tokens"), "cache_write": oc.get("cache_write_tokens"),
                        "steps": oc.get("steps")} if oc else None),
            "ak_check_seconds": num(ak.get("seconds")) if ak else None,
            "ak_check_compile_seconds": num(((ak.get("by_mode") or {}).get("compile") or {}).get("seconds")),
            "ak_check_op_test_seconds": num(((ak.get("by_mode") or {}).get("op-test") or {}).get("seconds")),
            **({"history_fields": hr3} if (hr3 := hr3_fields(d)) else {}),
            "attempt_key_in_line": any(k in d for k in ("attempt_id", "trial_id")) or
                                   any(k in c for k in ("attempt_id", "trial_id")),
        })
    unpaired_calls = sum(1 for idx, (ln, d) in enumerate(lines)
                         if d.get("schema") == "epyc.autokernel.actor_call.v1" and idx not in used)
    other = Counter(d.get("schema") for _, d in lines)
    return out, {"lines": len(lines), "schemas": dict(sorted(other.items())),
                 "unpaired_metrics": unpaired_metrics, "unpaired_calls": unpaired_calls}


def _near(a, b, tol=60.0):
    x, y = iso_to_epoch(a), iso_to_epoch(b)
    return x is not None and y is not None and abs(x - y) <= tol


def iso_to_epoch(s):
    if not s:
        return None
    try:
        return float(calendar.timegm(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")))
    except ValueError:
        return None


# --------------------------------------------------------------------------- main build
def build(root: Path, out_dir: Path):
    t0 = time.time()
    stores_all, excluded = discover(root)
    locked = _locked_inodes()
    live_evidence, stores = [], []
    for db in stores_all:
        ev = liveness(db, t0, locked)
        live_evidence.append(ev)
        if ev["live"]:
            excluded.append((str(db), "live: " + "; ".join(ev["reasons"])))
        else:
            stores.append(db)

    # mtime snapshot of EVERY file in every store dir we touch (before)
    def snapshot():
        snap = {}
        dirs = {db.parent for db in stores_all}
        for d in sorted(dirs):
            for f in sorted(d.iterdir()):
                if f.is_file():
                    s = f.stat()
                    snap[str(f)] = [s.st_mtime_ns, s.st_size]
        return snap

    before = snapshot()

    rows, store_meta, alias_tables = [], [], {}
    for db in stores:
        r, n, size, aliases = read_store(db)
        rows.extend(r)
        alias_tables[str(db)] = aliases
        store_meta.append({"store_path": str(db), "label": store_label(db), "campaign_dir": campaign_dir(db),
                           "target_group": target_group(db), "size_bytes": size, "row_count": n,
                           "mtime_ns": db.stat().st_mtime_ns,
                           "wal_bytes": (Path(str(db) + "-wal").stat().st_size
                                         if Path(str(db) + "-wal").exists() else None)})
    # reconcile (a) hard
    by_store_status = defaultdict(Counter)
    for r in rows:
        by_store_status[r["store_path"]][r["status"]] += 1
    for m in store_meta:
        tot = sum(by_store_status[m["store_path"]].values())
        if tot != m["row_count"]:
            raise SystemExit(f"RECONCILE FAIL {m['store_path']}: by-status {tot} != count(*) {m['row_count']}")
        m["by_status"] = dict(sorted(by_store_status[m["store_path"]].items()))

    # anchored flag (epoch has anchor_verified in same store)
    anchored_epochs = {(r["store_path"], r["task_key"]["epoch_sha256"]) for r in rows if r["status"] == "anchor_verified"}
    for r in rows:
        r["anchored_epoch"] = (r["store_path"], r["task_key"]["epoch_sha256"]) in anchored_epochs
        cs = r["outcome"]["comparison"] or {}
        r["paired_anchor"] = (num(cs.get("anchor_tok_s")) is not None and num(cs.get("candidate_tok_s")) is not None) \
            or ((cs.get("n_anchor_samples") or 0) > 0 and (cs.get("n_candidate_samples") or 0) > 0)

    # ---- runtime-arms-attempts
    arm_rows, arm_files = [], []
    for sm in store_meta:
        p = Path(sm["store_path"]).parent / "runtime-preparation" / "runtime-arms-attempts.json"
        if not p.is_file():
            continue
        with open(p) as fh:
            j = json.load(fh)
        atts = j.get("attempts") or []
        arm_files.append({"path": str(p), "sha256": sha(p), "attempts": len(atts)})
        for i, a in enumerate(atts):
            dim = (a.get("pair") or {}).get("dimension") or {}
            cmpx = a.get("comparison") or {}
            # join to experiments by mechanism_id == dimension_id and equal effect
            hit = [r for r in rows if r["store_path"] == sm["store_path"] and r["mechanism_id"] == dim.get("dimension_id")
                   and num(r["outcome"]["effect_fraction"]) is not None and num(cmpx.get("effect")) is not None
                   and abs(r["outcome"]["effect_fraction"] - cmpx["effect"]) < 1e-12]
            arm_rows.append({
                "record_kind": "runtime_arm_attempt",
                "store_path": sm["store_path"], "store_label": sm["label"], "store_size_bytes": sm["size_bytes"],
                "store_row_count": sm["row_count"], "campaign_dir": sm["campaign_dir"], "target_group": sm["target_group"],
                "source_file": str(p), "source_index": i, "rowid": None, "attempt_id": None,
                "recorded_at": a.get("recorded_at"), "status": None,
                "task_key": {"target_surface": "runtime", "target_symbol": dim.get("kind"), "epoch_sha256": None,
                             "measurement_epoch": None},
                "mechanism_id": dim.get("dimension_id"),
                "candidate": {"mechanism_id": dim.get("dimension_id"),
                              "knob": {"type": "runtime_dimension", "kind": dim.get("kind"),
                                       "dimension_id": dim.get("dimension_id"), "anchor": dim.get("anchor"),
                                       "candidate": dim.get("candidate")}, "candidate_diff_sha256": None},
                "outcome": {"status": None, "failure_class": None, "refusal_reason": None,
                            "effect_fraction": num(cmpx.get("effect")), "comparison": {
                                k: cmpx.get(k) for k in ("admission", "decisive", "effect_pct", "noise_floor_pct", "pairs")
                                if k in cmpx}, "measure_window_s_sum": None,
                            "stage_seconds": {"planner": None, "critic": None, "build": None, "correctness": None,
                                              "measure": None}},
                "declaration_sha256": a.get("declaration_sha256"),
                "joined_experiments_rowids": sorted(h["rowid"] for h in hit),
                "anchored_epoch": None, "paired_anchor": True,
            })

    # ---- actor calls
    call_rows, call_files = [], []
    for f in sorted(glob.glob(str(root / "campaigns" / "*" / "**" / "actor-calls.jsonl"), recursive=True)):
        p = Path(f)
        rel = p.relative_to(root / "campaigns")
        camp = rel.parts[0]
        recs, info = read_calls(p, camp)
        call_rows.extend(recs)
        call_files.append({"path": f, "campaign_dir": camp, "sha256": sha(p), **info})
    # time-window heuristic join: next recorded attempt in the same campaign (across its stores)
    att_by_camp = defaultdict(list)
    for r in rows:
        t = iso_to_epoch(r["recorded_at"])
        if t is not None and r["campaign_dir"] != "loop-memory":
            att_by_camp[r["campaign_dir"]].append((t, r["store_label"], r["rowid"], r["attempt_id"]))
    for k in att_by_camp:
        att_by_camp[k].sort()
    for c in call_rows:
        lst = att_by_camp.get(c["campaign_dir"], [])
        tf = iso_to_epoch(c["finished_at"])
        c["join"] = {"method": "time_window_next_recorded_attempt", "is_key_join": False,
                     "attempt_id": None, "store_label": None, "rowid": None, "delta_s": None}
        if tf is not None and lst:
            i = bisect.bisect_left([x[0] for x in lst], tf)
            if i < len(lst):
                t, sl, rid, aid = lst[i]
                c["join"].update({"attempt_id": aid, "store_label": sl, "rowid": rid, "delta_s": t - tf})
    # duplicates
    call_id_counts = Counter(c["call_id"] for c in call_rows if c["call_id"])
    dup_call_ids = sum(1 for v in call_id_counts.values() if v > 1)

    # heuristic per-attempt call seconds attached to history rows (labelled heuristic)
    sec_by_attempt = defaultdict(lambda: defaultdict(float))
    for c in call_rows:
        if c["join"]["attempt_id"] and c["wall_s"] is not None:
            sec_by_attempt[(c["join"]["store_label"], c["join"]["rowid"])][c["role"]] += c["wall_s"]
    for r in rows:
        sec = sec_by_attempt.get((r["store_label"], r["rowid"]))
        r["heuristic_call_seconds_by_role"] = ({k: round(v, 3) for k, v in sorted(sec.items())} if sec else None)

    # ------------------------------------------------------------------ census
    census = build_census(root, rows, arm_rows, store_meta, excluded, call_rows, call_files, dup_call_ids, arm_files,
                          alias_tables)

    after = snapshot()
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    prov = {
        "schema": SCHEMA + ".run_provenance", "root": str(root), "liveness": live_evidence,
        "mtime_check": {"files_checked": len(before), "changed": changed, "before_after_equal": not changed,
                        "stores_mtime_ns": {m["store_path"]: m["mtime_ns"] for m in store_meta}},
        "config_echo": {"sqlite_uri": "file:<db>?mode=ro&immutable=1", "live_window_s": LIVE_WINDOW_S,
                        "python": sys.version.split()[0], "argv": sys.argv},
        "elapsed_s": round(time.time() - t0, 2),
    }
    if changed:
        write_outputs(out_dir, rows, arm_rows, call_rows, census, prov)
        raise SystemExit(f"STORE FILE MTIME/SIZE CHANGED: {changed}")
    write_outputs(out_dir, rows, arm_rows, call_rows, census, prov)
    return census


def write_outputs(out_dir: Path, rows, arm_rows, call_rows, census, prov):
    out_dir.mkdir(parents=True, exist_ok=True)

    def order(r):
        return (r["store_path"], r.get("rowid") if r.get("rowid") is not None else 10**12, r.get("source_index") or 0)
    with open(out_dir / "history_records.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(rows + arm_rows, key=order):
            f.write(jdump(r) + "\n")
    with open(out_dir / "actor_call_records.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(call_rows, key=lambda c: (c["file"], c["metrics_line"])):
            f.write(jdump(r) + "\n")
    with open(out_dir / "census.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(census, sort_keys=True, indent=1, ensure_ascii=False) + "\n")
    with open(out_dir / "run_provenance.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(prov, sort_keys=True, indent=1, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- census
def sd_cv(xs):
    if len(xs) < 2:
        return None
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs)
    return {"n": len(xs), "mean": m, "sd": sd, "cv": (sd / abs(m)) if m else None}


def build_census(root, rows, arm_rows, store_meta, excluded, call_rows, call_files, dup_call_ids, arm_files, alias_tables):
    exp = rows
    # ---- (a)
    by_campaign = defaultdict(Counter)
    for r in exp:
        by_campaign[r["campaign_dir"]][r["status"]] += 1
    a = {
        "label": "answerable_now",
        "by_store": {m["label"]: {"row_count_select_count": m["row_count"], "by_status": m["by_status"],
                                  "sum_by_status": sum(m["by_status"].values()),
                                  "reconciles": sum(m["by_status"].values()) == m["row_count"]} for m in store_meta},
        "by_campaign": {k: dict(sorted(v.items())) for k, v in sorted(by_campaign.items())},
        "rows_total": len(exp),
        "all_stores_reconcile": all(sum(m["by_status"].values()) == m["row_count"] for m in store_meta),
    }

    # ---- (b) independent campaigns: union-find over stores sharing an epoch (alias full/measurement or experiments epoch)
    parent = {m["store_path"]: m["store_path"] for m in store_meta}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    epoch_keys = defaultdict(set)
    for r in exp:
        if r["task_key"]["epoch_sha256"]:
            epoch_keys[r["task_key"]["epoch_sha256"]].add(r["store_path"])
            if r["task_key"]["measurement_epoch"]:
                epoch_keys[r["task_key"]["measurement_epoch"]].add(r["store_path"])
    for sp, al in alias_tables.items():
        for full, meas in al.items():
            epoch_keys[full].add(sp)
            epoch_keys[meas].add(sp)
    for ep, sps in epoch_keys.items():
        sps = sorted(sps)
        for s in sps[1:]:
            parent[find(s)] = find(sps[0])
    clusters = defaultdict(list)
    for m in store_meta:
        clusters[find(m["store_path"])].append(m)
    camps, empty_stores = [], []
    for root_s, ms in sorted(clusters.items()):
        sp = {m["store_path"] for m in ms}
        crow = [r for r in exp if r["store_path"] in sp]
        if not crow:
            empty_stores.extend(m["label"] for m in ms)
            continue
        kept = [r["outcome"]["effect_fraction"] for r in crow
                if r["status"] == "kept" and num(r["outcome"]["effect_fraction"]) is not None]
        measured = [r["outcome"]["effect_fraction"] for r in crow if num(r["outcome"]["effect_fraction"]) is not None
                    and r["status"] in ("kept", "keep_candidate", "measured_null", "regression", "runtime_observed")]
        camps.append({
            "stores": sorted(m["label"] for m in ms), "target_group": ms[0]["target_group"],
            "campaign_dirs": sorted({m["campaign_dir"] for m in ms}), "rows": len(crow),
            "n_kept": len(kept), "best_kept_effect_fraction": max(kept) if kept else None,
            "best_measured_effect_fraction": max(measured) if measured else None,
            "continuation_evidence": ("stores share an epoch id (experiments or epoch_aliases)" if len(ms) > 1
                                      else "single store"),
        })
    per_target = defaultdict(list)
    for c in camps:
        per_target[c["target_group"]].append(c)
    b_targets = {}
    for tg, cs in sorted(per_target.items()):
        scored = [c["best_kept_effect_fraction"] for c in cs if c["best_kept_effect_fraction"] is not None]
        b_targets[tg] = {"independent_store_clusters": len(cs), "with_a_kept_best_score": len(scored),
                         "between_run_best_score": sd_cv(scored) if len(scored) >= 2 else None}
    n_ge2_scored = sum(1 for v in b_targets.values() if v["with_a_kept_best_score"] >= 2)
    b = {
        "label": "answerable_now (count: 1 independent campaign per target)",
        "rule": "a store that shares any epoch id with another store is the same campaign (union-find); "
                "campaign = connected store cluster; best score = max effect_fraction over status=kept rows",
        "clusters": camps, "empty_stores_not_counted": sorted(empty_stores), "by_target": b_targets,
        "targets_with_ge2_scored_independent_campaigns": n_ge2_scored,
        "between_run_sd_cv": ("not answerable: no target has two independent campaigns that each hold a kept best "
                              "score (count of targets with >=2 scored = %d)" % n_ge2_scored) if n_ge2_scored == 0
        else "see by_target.between_run_best_score",
        "max_independent_scored_campaigns_for_any_target": max([v["with_a_kept_best_score"] for v in b_targets.values()] or [0]),
    }

    # ---- (c) budget axis, stable trial id, call->attempt key
    with_tokens = [c for c in call_rows if c["tokens"]]
    tok_roles = Counter(c["backend_kind"] for c in with_tokens)
    all_backends = Counter(c["backend_kind"] for c in call_rows)
    no_tok_by_backend = {k: all_backends[k] - tok_roles.get(k, 0) for k in all_backends}
    mono = {}
    for f in call_files:
        ts = [c["ts"] for c in call_rows if c["file"] == f["path"]]
        mono[f["path"]] = sum(1 for x, y in zip(ts, ts[1:]) if y < x)
    ids_seen = Counter(r["attempt_id"] for r in exp)
    dup_attempt_ids = sum(1 for v in ids_seen.values() if v > 1)
    # rowid vs recorded_at monotone per store
    inv = {}
    for m in store_meta:
        rs = [r["recorded_at"] for r in sorted((r for r in exp if r["store_path"] == m["store_path"]),
                                               key=lambda r: r["rowid"])]
        inv[m["label"]] = sum(1 for x, y in zip(rs, rs[1:]) if y < x)
    keyed = sum(1 for c in call_rows if c["attempt_key_in_line"])
    deltas = [c["join"]["delta_s"] for c in call_rows if c["join"]["delta_s"] is not None]
    buckets = Counter()
    for d in deltas:
        buckets["<=60s" if d <= 60 else "<=600s" if d <= 600 else "<=3600s" if d <= 3600 else ">3600s"] += 1
    c_ = {
        "label": "partially_answerable",
        "metrics_lines_total": len(call_rows),
        "metrics_lines_with_decoded_plus_prompt_tokens": len(with_tokens),
        "metrics_lines_without_tokens": len(call_rows) - len(with_tokens),
        "token_lines_by_backend_kind": dict(sorted(tok_roles.items())),
        "lines_without_tokens_by_backend_kind": dict(sorted(no_tok_by_backend.items())),
        "cumulative_token_axis": "not answerable for the full call stream: %d of %d metrics lines carry no token "
                                 "field (hosted codex/claude seats); answerable only over the %d opencode lines"
                                 % (len(call_rows) - len(with_tokens), len(call_rows), len(with_tokens)),
        "cumulative_wall_s_axis": "answerable: wall_s on %d of %d lines; ts goes backwards %d times across %d files "
                                  "(concurrent lanes), so order by started_at/finished_at, not file position"
                                  % (sum(1 for c in call_rows if c["wall_s"] is not None), len(call_rows),
                                     sum(mono.values()), len(mono)),
        "ts_inversions_by_file": mono, "ts_inversions_total": sum(mono.values()),
        "stable_trial_id": {"attempt_id_present_rows": sum(1 for r in exp if r["attempt_id"]),
                            "rows": len(exp), "duplicate_attempt_ids_across_stores": dup_attempt_ids,
                            "verdict": "answerable: attempt_id is a unique per-attempt hash on every experiments row"},
        "per_campaign_monotone_trial_counter": {
            "rowid_is_per_store_and_restarts_at_1_in_each_store": True,
            "recorded_at_inversions_vs_rowid_by_store": inv,
            "verdict": "not answerable: no campaign-wide ordinal; rowid restarts per store and the DS41 campaign spans "
                       "two stores"},
        "call_to_attempt_key": {"lines_carrying_an_attempt_or_trial_key": keyed, "lines": len(call_rows),
                                "verdict": "not answerable by key (count with key = %d of %d); only the time-window "
                                           "heuristic exists" % (keyed, len(call_rows)),
                                "time_window_next_attempt_delta_buckets": dict(sorted(buckets.items())),
                                "calls_with_no_following_attempt": len(call_rows) - len(deltas)},
        "duplicate_call_ids_across_files": dup_call_ids,
    }

    hr3_calls = [c for c in call_rows if "history_fields" in c]
    hr3_rows = [r for r in exp if "history_fields" in r]
    if hr3_calls or hr3_rows:
        # Present only over post-AK-HR-3 stores; absent (so census bytes unchanged) otherwise.
        row_trials = {r["history_fields"]["trial_id"]: r for r in hr3_rows
                      if r["history_fields"]["trial_id"]}
        joined = [c for c in hr3_calls if c["history_fields"]["trial_id"] in row_trials]
        ordinals = sorted(r["history_fields"]["trial_ordinal"] for r in hr3_rows
                          if isinstance(r["history_fields"]["trial_ordinal"], int))
        cumul = [r["history_fields"]["cumulative_tokens"] for r in sorted(
            (r for r in hr3_rows if isinstance(r["history_fields"]["trial_ordinal"], int)),
            key=lambda r: r["history_fields"]["trial_ordinal"])]
        c_["post_change_fields"] = {
            "rows_with_fields": len(hr3_rows), "metrics_lines_with_fields": len(hr3_calls),
            "calls_joined_to_an_attempt_by_trial_id": len(joined),
            "calls_with_trial_id_but_no_recorded_attempt": sum(
                1 for c in hr3_calls if c["history_fields"]["trial_id"]
                and c["history_fields"]["trial_id"] not in row_trials),
            "calls_with_null_trial_id": sum(1 for c in hr3_calls if not c["history_fields"]["trial_id"]),
            "trial_ordinals_strictly_increasing": all(x < y for x, y in zip(ordinals, ordinals[1:])),
            "cumulative_tokens_monotone_non_decreasing": all(
                y >= x for x, y in zip([v for v in cumul if v is not None],
                                       [v for v in cumul if v is not None][1:])),
            "rows_with_family_bucket": sum(1 for r in hr3_rows if r["history_fields"]["mechanism_family_bucket"]),
            "rows_with_knob_descriptor": sum(1 for r in hr3_rows if r["history_fields"]["knob_descriptor"]),
            "rows_with_stage_seconds": sum(1 for r in hr3_rows if r["history_fields"]["stage_seconds"]),
            "verdict": "call->attempt join by trial_id and the budget axis are answerable over these rows only; "
                       "older rows stay time-window-only",
        }

    # ---- (d) wall split
    role_wall = defaultdict(lambda: defaultdict(float))
    role_n = defaultdict(Counter)
    ak_sec = defaultdict(float)
    for c in call_rows:
        if c["wall_s"] is not None:
            role_wall[c["campaign_dir"]][c["role"]] += c["wall_s"]
            role_n[c["campaign_dir"]][c["role"]] += 1
        if c["ak_check_seconds"] is not None:
            ak_sec[c["campaign_dir"]] += c["ak_check_seconds"]
    meas = defaultdict(float)
    meas_n = Counter()
    dev = defaultdict(float)
    dev_n = Counter()
    for r in exp:
        ds = r["outcome"].get("measure_device_seconds")
        if ds is not None:
            dev[r["campaign_dir"]] += ds
            dev_n[r["campaign_dir"]] += 1
        w = r["outcome"]["measure_window_s_sum"]
        if w is not None:
            meas[r["campaign_dir"]] += w
            meas_n[r["campaign_dir"]] += 1
    d_by = {}
    for camp in sorted(set(role_wall) | set(meas) | set(dev)):
        ident = sum(role_wall[camp].values()) + meas[camp]
        d_by[camp] = {
            "calls_by_role": dict(sorted(role_n[camp].items())),
            "call_wall_s_by_role": {k: round(v, 3) for k, v in sorted(role_wall[camp].items())},
            "author_side_ak_check_seconds_sum": round(ak_sec[camp], 3),
            "measure_window_s_sum_rows": meas_n[camp], "measure_window_s_sum": round(meas[camp], 3),
            "identified_seconds_planner_critic_author_measure": round(ident, 3),
            "share_of_identified": {k: round(v / ident, 4) for k, v in
                                    sorted({**role_wall[camp], "measure_windows": meas[camp]}.items())} if ident else None,
            "build_seconds": None,
            "legacy_comparison_device_seconds_rows": dev_n[camp],
            "legacy_comparison_device_seconds_sum": round(dev[camp], 3),
        }
    d_ = {
        "label": "partially_answerable",
        "by_campaign": d_by,
        "planner_critic_author": "answerable now per call (wall_s on actor_call_metrics.v1); per ATTEMPT only via "
                                 "the time-window heuristic (see (c)), carried as heuristic_call_seconds_by_role",
        "measure": "answerable now as instrument window seconds (sum of comparison anchor/candidate residency "
                   "window_s) on rows that carry a comparison; the measure PHASE wall is not recorded",
        "build": "needs fields: no build seconds on any experiments payload or call line (ak_check compile seconds "
                 "cover only the author-side check). Needed: build_s, correctness_s, planner_s, critic_s, measure_s "
                 "per attempt on the candidate record.",
        "cpu_window_events": "phase transitions exist in cpu-window*.events.jsonl but carry no event timestamp "
                             "(only ttl-derived expires_at), so phase durations cannot be derived",
        "caveat": "shares are over IDENTIFIED seconds only; they are not shares of loop wall and do not replace the "
                  "75.7%/3.9% figures until build seconds are recorded",
    }

    # ---- (e) surrogate readiness per target (group x epoch)
    groups = defaultdict(list)
    for r in exp:
        tk = r["task_key"]
        groups[(r["target_group"], r["store_label"])].append(r)
    e_rows = []
    for (tg, sl), rs in sorted(groups.items()):
        num_eff = [r for r in rs if num(r["outcome"]["effect_fraction"]) is not None]
        measured_trials = [r for r in num_eff if r["status"] != "anchor_verified" and r["outcome"]["comparison"]]
        anchored = [r for r in measured_trials if r["paired_anchor"]]
        # comparable = same (surface, metric, np, measurement epoch-or-epoch)
        cg = defaultdict(list)
        for r in anchored:
            cs = r["outcome"]["comparison"]
            key = (cs.get("surface"), cs.get("metric"), cs.get("np"),
                   r["task_key"]["measurement_epoch"] or r["task_key"]["epoch_sha256"])
            cg[key].append(r)
        comparable = [x for v in cg.values() for x in v]
        kept = [r for r in comparable if r["status"] == "kept"]
        typed = [r for r in comparable if r["candidate"]["knob"] is not None]
        e_rows.append({
            "target_group": tg, "store": sl, "rows": len(rs),
            "rows_with_numeric_effect": len(num_eff),
            "anchor_verified_rows": sum(1 for r in rs if r["status"] == "anchor_verified"),
            "measured_trials_with_comparison": len(measured_trials),
            "anchored_trials_paired_anchor_and_numeric_effect": len(anchored),
            "comparable_same_epoch_groups": len(cg),
            "comparable_trials_same_epoch": len(comparable),
            "largest_comparable_group": max([len(v) for v in cg.values()] or [0]),
            "keeps_among_comparable": len(kept),
            "keeps_all": sum(1 for r in rs if r["status"] == "kept"),
            "typed_knob_trials_among_comparable": len(typed),
            "typed_knob_coverage_of_comparable": (round(len(typed) / len(comparable), 4) if comparable else None),
            "meets_count_threshold_200_trials_and_20_keeps": len(comparable) >= 200 and len(kept) >= 20,
            "distinct_measurement_epochs_in_comparable": len({k[3] for k in cg}),
        })
    tgt_tot = defaultdict(lambda: Counter())
    for er in e_rows:
        for k in ("comparable_trials_same_epoch", "keeps_among_comparable", "typed_knob_trials_among_comparable",
                  "rows_with_numeric_effect"):
            tgt_tot[er["target_group"]][k] += er[k]
    trig = {tg: {"count_threshold_met": v["comparable_trials_same_epoch"] >= 200 and v["keeps_among_comparable"] >= 20,
                 "typed_knob_trials": v["typed_knob_trials_among_comparable"],
                 "trigger_met": (v["comparable_trials_same_epoch"] >= 200 and v["keeps_among_comparable"] >= 20
                                 and v["typed_knob_trials_among_comparable"] > 0)}
            for tg, v in sorted(tgt_tot.items())}
    e_ = {
        "label": "answerable_now",
        "AK_RI_SURROGATE_1_trigger_by_target": trig,
        "definition": "numeric effect = effect_fraction not null; measured trial = numeric effect, status != "
                      "anchor_verified, comparison object present; anchored = comparison carries both anchor_tok_s "
                      "and candidate_tok_s (paired matched anchor); comparable = same comparison "
                      "(surface, metric, np) and same measurement epoch; typed knob = payload.runtime_pair.dimension",
        "by_target_store": e_rows,
        "by_target_total": {k: dict(v) for k, v in sorted(tgt_tot.items())},
        "trigger_AK_RI_SURROGATE_1_met_for_any_target": any(v["trigger_met"] for v in trig.values()),
        "note_legacy": "legacy-loop-memory passes the raw count threshold but spans many surfaces and epochs "
                       "(see distinct_measurement_epochs_in_comparable and largest_comparable_group) and carries no "
                       "typed knob descriptor, so the typed-descriptor clause fails",
        "source_candidates_typed_descriptor": "needs fields: no typed knob/edit descriptor on source-patch candidates "
                                              "(only candidate_diff_sha256 where present)",
    }

    # ---- (f) RuntimeArmPair knob space
    kinds = ["threads", "cpu_list", "numa_policy", "env", "batch", "ubatch", "load_threads"]
    obs = Counter()
    obs_vals = defaultdict(set)
    for r in arm_rows:
        k = r["candidate"]["knob"]
        obs[k["kind"]] += 1
        obs_vals[k["kind"]].add(jdump([k["anchor"], k["candidate"]]))
    exp_knobs = Counter()
    for r in exp:
        k = r["candidate"]["knob"]
        if k:
            exp_knobs[k["kind"] + ":" + str(k["dimension_id"])] += 1
    f_ = {
        "label": "answerable_now",
        "kind_vocabulary": kinds,
        "kind_vocabulary_size": len(kinds),
        "value_domains": {
            "threads/batch/ubatch/load_threads": "positive integers (load_threads may be null); unbounded above",
            "cpu_list": "free text", "numa_policy": "free text constrained to a --interleave=/--membind= prefix",
            "env": "free text key plus free text string-or-null value (key fixed within a dimension)"},
        "bounded_decision_token_space": "no at value level (integers and free strings are unbounded); yes at kind "
                                        "level (7 kinds). A bounded space exists only if values are discretised to a "
                                        "pre-declared grid per kind, which the schema does not hold",
        "observed_runtime_arm_attempts": sum(obs.values()),
        "observed_kinds": dict(sorted(obs.items())),
        "observed_distinct_anchor_candidate_pairs_by_kind": {k: len(v) for k, v in sorted(obs_vals.items())},
        "observed_distinct_dimension_ids_in_experiments": dict(sorted(exp_knobs.items())),
        "arm_attempts_joined_to_an_experiments_row": sum(1 for r in arm_rows if r["joined_experiments_rowids"]),
        "arm_attempt_files": arm_files,
        "source": "scripts/kernel_rnd/autokernel/loop/unified_planner.py DIMENSIONS and RuntimeDimension.__post_init__",
    }

    # ---- (g) gap table
    sta = Counter(r["status"] for r in exp)
    fc = sum(1 for r in exp if r["outcome"]["failure_class"])
    harness_rows = sum(sta.get(s, 0) for s in HARNESS_STATUSES)
    nonmeasured = [r for r in exp if num(r["outcome"]["effect_fraction"]) is None]
    g_rows = [
        ("attempt_id (or trial key) on actor_call.v1 / actor_call_metrics.v1 lines", "absent", f"{keyed}/{len(call_rows)} lines"),
        ("experiments-row key back to its planner/critic/author calls", "absent",
         f"{sum(1 for r in exp if r['payload_flags']['has_call_id_key'])}/{len(exp)} payloads contain a call_id key"),
        ("campaign-wide monotone trial ordinal / evaluation counter", "absent", "rowid restarts per store"),
        ("cumulative decoded+prompt token counter on experiments rows", "absent", f"{len(exp)} rows"),
        ("token counts on hosted (codex/claude) call lines", "absent",
         f"{len(call_rows) - len(with_tokens)}/{len(call_rows)} lines carry none"),
        ("failure_class column/field on experiments rows", "partial",
         f"derivable from payload on {fc}/{len(exp)} rows; {len(exp) - fc} rows have only status/refusal_reason"),
        ("per-stage seconds planner/critic/build/correctness/measure on the candidate record", "absent",
         "0 rows; planner/critic/author only on call lines, measure only as instrument windows"),
        ("typed knob/edit descriptor on source-patch candidates", "absent",
         f"{sum(1 for r in exp if r['candidate']['knob'] is None)}/{len(exp)} rows null; typed only for runtime arms"),
        ("runtime-arms-attempts.json entry -> experiments attempt_id", "absent",
         f"{len(arm_rows)} attempts; joined by (dimension_id, effect) heuristic to "
         f"{sum(1 for r in arm_rows if r['joined_experiments_rowids'])}"),
        ("event timestamps on cpu-window phase events", "absent", "phase durations underivable"),
        ("planner prompt text (only digest/bytes stored)", "absent", "prompt {bytes,chars,sha256} only"),
        ("patch bytes retained per candidate", "partial",
         f"retained_patch on {sum(1 for r in exp if r['payload_flags']['has_retained_patch'])}/{len(exp)} rows; "
         f"candidate_diff_sha256 on {sum(1 for r in exp if r['payload_flags']['has_candidate_diff_sha256'])} rows"),
        ("hypothesis text on harness-event rows", "absent",
         f"{harness_rows} harness-event rows ({', '.join(HARNESS_STATUSES)}) mostly carry no statement"),
    ]
    g_ = {"label": "answerable_now",
          "rows": [{"field": a1, "status": a2, "evidence": a3} for a1, a2, a3 in g_rows]}

    questions = {
        "a": "answerable now", "b": "not answerable: 0 targets with >=2 scored independent campaigns (see b)",
        "c": "partially answerable: stable trial id yes; token budget axis only on %d opencode lines; call->attempt key "
             "no (needs attempt_id on the sibling metrics line)" % len(with_tokens),
        "d": "partially answerable: planner/critic/author per call and measure windows yes; build seconds needs fields",
        "e": "answerable now (counts in e.by_target_total)", "f": "answerable now (kind-level bounded, value-level not)",
        "g": "answerable now",
    }
    return {
        "schema": SCHEMA, "root": str(root),
        "stores_included": [m["label"] for m in store_meta], "stores": store_meta,
        "excluded": [{"path": p, "reason": r} for p, r in sorted(excluded)],
        "inputs": {"actor_call_files": call_files},
        "totals": {"experiment_rows": len(exp), "runtime_arm_attempt_rows": len(arm_rows),
                   "actor_call_rows": len(call_rows)},
        "questions": questions,
        "a_rows_by_status_and_campaign": a, "b_independent_campaigns": b, "c_budget_axis_and_keys": c_,
        "d_wall_split": d_, "e_surrogate_readiness": e_, "f_runtime_arm_knob_space": f_, "g_gap_table": g_,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    a = ap.parse_args()
    build(Path(a.root), Path(a.out_dir))
    print("ok", a.out_dir)


if __name__ == "__main__":
    main()
