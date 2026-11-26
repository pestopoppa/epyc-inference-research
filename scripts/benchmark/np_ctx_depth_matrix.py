#!/usr/bin/env python3
"""np x context-depth serving matrix, measured INCREMENTALLY per stream (operator spec 2026-11-26, C22 production receipts).

Grid: depths (tokens of context per stream) x np (concurrent slots), per (model, device), for every kernel arm
(INCUMBENT = production v10, C21, C22; INCUMBENT2 = A/A at np1 only).  One server launch per (arm, np).

Incremental depth (never prefill a depth from scratch): each stream s owns slot s (id_slot) and its OWN document (distinct
per stream, so np streams never share a prefix cache).  For depth d_k the stream sends the first d_k tokens of its
document with cache_prompt=true; the server re-uses the cached prefix d_{k-1} and prefills only the delta.  Per step and
stream we record the server's own timings: incremental prefill = prompt_n / prompt_ms (prompt_n IS the delta, the cached
part is not re-processed), decode at that depth = predicted_n / predicted_ms for a short fixed n_predict, draft
acceptance = draft_n_accepted / draft_n.  Decode tokens are DISCARDED: the next step re-sends the document prefix only,
so the cache diverges from the generated tail and the delta prefill starts at the old depth (policy recorded per row).
For np > 1 the streams of one step run concurrently; aggregate decode = sum predicted_n / the step's wall time span of
the decode phase (max over streams of predicted_ms), recorded beside the per-request values.

Capacity: -c = (deepest fitting depth + n_predict + margin) x np per launch; KV bytes = c x kv_bytes_per_token.  A cell
whose KV (+ non-KV weights) exceeds the device budget is pre-marked SKIPPED_CAPACITY with the computed KV GiB and never
launched.  Long context: every per-stream position stays <= max depth + n_predict, which must be <= the model's native
n_ctx_train; llama.cpp keys RoPE on token positions, and n_ctx_seq (= c if unified, c / np if split) only gates a
warning.  The KV mode per (model, np) is taken from the config's evidence table (most performant recorded setup), is
IDENTICAL across arms, and is recorded per cell with n_ctx_seq, n_ctx_train and the rope/YaRN flags (none needed when
positions stay native; a cell that would need them is flagged REQUIRES_CTX_EXTENSION and refused, not silently run).

Futility: a step whose wall time exceeds step_cap_s stops that launch's deeper depths (recorded as STOPPED_STEP_CAP).

Process rules (CLAUDE.md): the server PID comes from Popen and is killed only by that PID (SIGTERM -> SIGKILL -> verify);
no name-pattern kill.  Run under a rowflow grant; the driver never claims CPUs or the GPU itself.

Output: <out>/capture.jsonl, one self-hashed row per (arm, np, depth) cell (schema epyc.np_ctx_depth_matrix.v1) plus
<out>/plan.json.  `export-catalog` writes the epyc-web data/catalog.json matrix record shape from a capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

SCHEMA = "epyc.np_ctx_depth_matrix.v1"
PLAN_SCHEMA = "epyc.np_ctx_depth_matrix_plan.v1"
DEPTH_NAMES = {"1k": 1024, "8k": 8192, "32k": 32768, "64k": 65536, "100k": 102400}
GIB = 1024 ** 3


# ------------------------------------------------------------------------------------------------ hashing / rows
def canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def seal(row: dict) -> dict:
    r = {k: v for k, v in row.items() if k != "row_sha256"}
    r["schema"] = SCHEMA
    r["row_sha256"] = hashlib.sha256(canon(r).encode()).hexdigest()
    return r


def verify_row(row: dict) -> bool:
    r = {k: v for k, v in row.items() if k != "row_sha256"}
    return hashlib.sha256(canon(r).encode()).hexdigest() == row.get("row_sha256")


# ------------------------------------------------------------------------------------------------ documents
def build_documents(sources: list[str], n_streams: int, min_chars: int) -> list[str]:
    """Deterministic per-stream documents: stream s concatenates the sorted source files starting at a rotation that
    differs per stream (s * len/n), wrapping, until min_chars.  Distinct prefixes => no shared prefix cache."""
    files = sorted(set(sources))
    if not files:
        raise SystemExit("no document sources")
    texts = [Path(f).read_text(errors="replace") for f in files]
    docs = []
    for s in range(n_streams):
        start = (s * len(files)) // max(1, n_streams)
        parts, n, i = [], 0, 0
        while n < min_chars:
            j = (start + i) % len(files)
            chunk = f"\n\n===== {files[j]} =====\n\n" + texts[j]
            parts.append(chunk)
            n += len(chunk)
            i += 1
            if i > 100000:
                raise SystemExit("sources too small to reach min_chars")
        docs.append("".join(parts))
    if len({d[:4096] for d in docs}) != len(docs):
        raise SystemExit("per-stream documents share a prefix; add sources")
    return docs


def doc_manifest(docs: list[str]) -> list[dict]:
    return [{"stream": s, "chars": len(d), "sha256": hashlib.sha256(d.encode()).hexdigest()} for s, d in enumerate(docs)]


# ------------------------------------------------------------------------------------------------ plan
def kv_mode_for(server: dict, np_: int) -> dict:
    table = server.get("kv_mode_by_np") or {}
    ent = table.get(str(np_)) or table.get("default") or {"mode": "split", "evidence": "llama-server default with -np"}
    if ent["mode"] not in ("split", "unified"):
        raise SystemExit(f"{server['id']}: kv mode {ent['mode']!r}")
    return ent


def plan_server(server: dict, depths: list[str], nps: list[int], n_predict: int, margin: int) -> dict:
    """Pure function: every launch and cell for one (model, device), capacity- and context-checked."""
    kvb = float(server["kv_bytes_per_token"])
    budget = float(server["capacity_gib"]) - float(server.get("non_kv_gib", 0.0))
    native = int(server["n_ctx_train"])
    launches, cells = [], []
    for np_ in nps:
        kv = kv_mode_for(server, np_)
        fit = []
        for dn in depths:
            d = DEPTH_NAMES[dn]
            per_slot = d + n_predict + margin
            c = per_slot * np_
            kv_gib = c * kvb / GIB
            pos_max = d + n_predict
            if pos_max > native:
                status = "REQUIRES_CTX_EXTENSION"
            elif kv_gib > budget:
                status = "SKIPPED_CAPACITY"
            else:
                status = "PLANNED"
                fit.append(d)
            cells.append({"server": server["id"], "np": np_, "depth": dn, "depth_tokens": d, "status": status,
                          "kv_gib_at_depth": round(kv_gib, 2), "kv_budget_gib": round(budget, 2)})
        if not fit:
            continue
        per_slot = max(fit) + n_predict + margin
        per_slot = int(math.ceil(per_slot / 256.0) * 256)          # llama.cpp pads n_ctx_seq to 256
        c = per_slot * np_
        n_ctx_seq = c if kv["mode"] == "unified" else per_slot
        launches.append({"server": server["id"], "np": np_, "c": c, "kv_mode": kv["mode"], "kv_evidence": kv.get("evidence"),
                         "n_ctx_seq": n_ctx_seq, "n_ctx_train": native,
                         "n_ctx_seq_over_train": n_ctx_seq > native,
                         "max_position": max(fit) + n_predict,
                         "rope": "native (no YaRN/rope flags): every per-stream position <= n_ctx_train",
                         "kv_gib": round(c * kvb / GIB, 2), "depths": [dn for dn in depths if DEPTH_NAMES[dn] in fit]})
    return {"server": server["id"], "launches": launches, "cells": cells}


def estimate_hours(server: dict, splan: dict, arms: list[str], n_predict: int) -> float:
    """Prior-based wall estimate: per launch load + per step (delta prefill at the slowest-stream rate x np concurrent
    sharing the device, + decode).  Priors come from the config (prefill_tok_s_prior, decode_tok_s_prior)."""
    pp, dec, load = float(server["prefill_tok_s_prior"]), float(server["decode_tok_s_prior"]), float(server.get("load_s_prior", 90))
    tot = 0.0
    for L in splan["launches"]:
        n_arms = sum(1 for a in arms if not (a.endswith("2") and L["np"] != 1))
        prev, t = 0, load
        for dn in L["depths"]:
            d = DEPTH_NAMES[dn]
            t += (d - prev) * L["np"] / pp + n_predict / max(1.0, dec / L["np"])
            prev = d
        tot += t * n_arms
    return tot / 3600.0


# ------------------------------------------------------------------------------------------------ server I/O
def http(port: int, path: str, body=None, timeout=3600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def launch_argv(server: dict, launch: dict, store_bin: str, port: int) -> list[str]:
    base = list(json.load(open(server["argv_file"])))
    a = [x.replace("@@BIN@@", store_bin + "/llama-server") for x in base]
    if not a[0].endswith("llama-server"):
        a[0] = store_bin + "/llama-server"
    else:
        a[0] = store_bin + "/llama-server"

    def setflag(flag, val):
        if flag in a:
            a[a.index(flag) + 1] = str(val)
        else:
            a.extend([flag, str(val)])
    for alias in ("--ctx-size",):
        if alias in a:
            a[a.index(alias)] = "-c"
    for alias in ("--parallel",):
        if alias in a:
            a[a.index(alias)] = "-np"
    setflag("-np", launch["np"])
    setflag("-c", launch["c"])
    setflag("--port", port)
    for f in ("--kv-unified", "--no-kv-unified"):
        while f in a:
            a.remove(f)
    a.append("--kv-unified" if launch["kv_mode"] == "unified" else "--no-kv-unified")
    if "--slot-save-path" in a:
        i = a.index("--slot-save-path")
        del a[i:i + 2]
    return a


class Server:
    def __init__(self, argv, env, log_path):
        self.log = open(log_path, "w")
        self.p = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT, env=env)

    def stop(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(30)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait(30)
        self.log.close()
        return self.p.poll() is not None


def wait_healthy(port, proc, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            if http(port, "/health", timeout=5).get("status") == "ok":
                return True
        except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError):
            pass
        time.sleep(2)
    return False


def run_steps(req_fn, tokens_by_stream, launch, n_predict, step_cap_s, stamp):
    """Incremental sweep for one launch.  req_fn(stream, prompt_tokens, n_predict) -> server response dict.
    Pure w.r.t. the server (injected), so it is unit-tested with a fake."""
    rows, stopped = [], None
    for dn in launch["depths"]:
        d = DEPTH_NAMES[dn]
        res, errs = [None] * launch["np"], [None] * launch["np"]

        def one(s):
            try:
                res[s] = req_fn(s, tokens_by_stream[s][:d], n_predict)
            except Exception as e:  # noqa: BLE001 - recorded, never swallowed silently
                errs[s] = repr(e)[:300]
        t0 = time.time()
        th = [threading.Thread(target=one, args=(s,)) for s in range(launch["np"])]
        [t.start() for t in th]
        [t.join() for t in th]
        wall = time.time() - t0
        per = []
        for s in range(launch["np"]):
            t = (res[s] or {}).get("timings") or {}
            per.append({"stream": s, "ok": errs[s] is None and bool(t), "error": errs[s],
                        "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
                        "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
                        "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
                        "cache_n": t.get("cache_n")})
        ok = [p for p in per if p["ok"]]

        def ratio(n, ms):
            return (n / ms * 1000.0) if ms else None
        pp = [ratio(p["prompt_n"], p["prompt_ms"]) for p in ok if p["prompt_ms"]]
        dec = [ratio(p["predicted_n"], p["predicted_ms"]) for p in ok if p["predicted_ms"]]
        dn_ = sum((p["draft_n"] or 0) for p in ok)
        agg_ms = max((p["predicted_ms"] or 0) for p in ok) if ok else 0
        row = dict(stamp, np=launch["np"], depth=dn, depth_tokens=d, step_wall_s=round(wall, 3), streams=per,
                   request_ok_fraction=len(ok) / launch["np"],
                   prefill_incremental_tok_s_med=_med(pp), decode_tok_s_per_request_med=_med(dec),
                   decode_tok_s_aggregate=(sum(p["predicted_n"] or 0 for p in ok) / agg_ms * 1000.0) if agg_ms else None,
                   draft_acceptance=(sum((p["draft_n_accepted"] or 0) for p in ok) / dn_) if dn_ else None,
                   decode_tokens_policy="discarded: next step re-sends the document prefix only",
                   status="MEASURED" if ok else "ERRORS")
        rows.append(seal(row))
        if wall > step_cap_s:
            stopped = dn
            break
    if stopped:
        later = launch["depths"][launch["depths"].index(stopped) + 1:]
        for dn in later:
            rows.append(seal(dict(stamp, np=launch["np"], depth=dn, depth_tokens=DEPTH_NAMES[dn], status="STOPPED_STEP_CAP",
                                  step_cap_s=step_cap_s, stopped_after=stopped)))
    return rows


def _med(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


# ------------------------------------------------------------------------------------------------ resolution blocks
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201,
       12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120}


def block_order(arms: list[str], b: int) -> list[str]:
    """ABBA-style rotation: block b visits the arms in a rotated order, reversed on odd blocks."""
    k = b // 2 % len(arms)
    o = arms[k:] + arms[:k]
    return o if b % 2 == 0 else list(reversed(o))


def resolve(rows: list[dict], np_: int, depth: str, metric: str, base: str, aa: str, look: int, aa_max: float) -> dict:
    """Paired 95% t-CI over blocks of ln(arm / base) for the repeat cell; A/A guard on ln(aa/base); futility at `look`."""
    import math as m
    rs = [r for r in rows if r.get("kind") == "repeat" and r.get("np") == np_ and r.get("depth") == depth and r.get("status") == "MEASURED"]
    blocks = sorted({r["block"] for r in rs})
    val = {(r["block"], r["arm"]): r.get(metric) for r in rs}
    out = {"np": np_, "depth": depth, "metric": metric, "blocks": len(blocks), "arms": {}}
    arms = sorted({r["arm"] for r in rs} - {base})
    for a in arms:
        lr = [m.log(val[(b, a)] / val[(b, base)]) for b in blocks if val.get((b, a)) and val.get((b, base))]
        n = len(lr)
        if n < 2:
            out["arms"][a] = {"n": n}
            continue
        mu = sum(lr) / n
        sd = (sum((x - mu) ** 2 for x in lr) / (n - 1)) ** 0.5
        h = T95.get(n - 1, 1.96) * sd / n ** 0.5
        out["arms"][a] = {"n": n, "ratio": m.exp(mu), "ci95": [m.exp(mu - h), m.exp(mu + h)]}
    g = out["arms"].get(aa, {})
    out["aa_ok"] = bool(g.get("ratio")) and abs(g["ratio"] - 1) <= aa_max
    if len(blocks) >= look:
        out["futility_look"] = {a: ("decided" if v.get("ci95") and (v["ci95"][0] > 1 or v["ci95"][1] < 1) else "open")
                                for a, v in out["arms"].items() if a != aa}
    return out


# ------------------------------------------------------------------------------------------------ catalog export
def export_catalog(rows: list[dict], record_id: str, title: str, hardware: str, source: str, arms: list[str]) -> dict:
    """The epyc-web data/catalog.json matrix record shape (matrix_columns + matrix rows + study cells), from a capture."""
    for r in rows:
        if not verify_row(r):
            raise SystemExit(f"capture row hash mismatch: {r.get('arm')} np{r.get('np')} {r.get('depth')}")
    meas = [r for r in rows if r.get("status") == "MEASURED"]
    cols = ["context depth", "np"]
    for a in arms:
        cols += [f"{a} per-request decode", f"{a} aggregate decode", f"{a} prefill"]
    keys = sorted({(r["depth_tokens"], r["np"], r["depth"]) for r in rows})
    by = {(r["arm"], r["depth"], r["np"]): r for r in meas}

    def f(x):
        return "" if x is None else f"{x:.1f}"
    matrix = []
    for _, np_, dn in keys:
        line = [dn, str(np_)]
        for a in arms:
            r = by.get((a, dn, np_))
            line += [f(r and r["decode_tok_s_per_request_med"]), f(r and r["decode_tok_s_aggregate"]),
                     f(r and r["prefill_incremental_tok_s_med"])] if r else ["", "", ""]
        matrix.append(line)
    study = [{"arm": r["arm"], "np": str(r["np"]), "L": str(r["depth_tokens"]), "status": r["status"],
              "draft_accept": r.get("draft_acceptance"), "kv_mode": r.get("kv_mode"), "c": r.get("c"),
              "n_ctx_seq": r.get("n_ctx_seq"), "row_sha256": r["row_sha256"]} for r in rows]
    return {"id": record_id, "status": "observed", "title": title, "hardware": hardware,
            "headline": f"{len(meas)} measured cells across {len(arms)} kernels",
            "detail": "incremental-depth matrix (prefix reuse per stream); per-request / aggregate decode and incremental prefill tok/s",
            "matrix_label": "Concurrency x context depth (per stream)",
            "matrix_note": "tok/s from the server's own timings; empty = skipped/stopped (see study)",
            "matrix_columns": cols, "matrix": matrix, "source": source, "study": study}


# ------------------------------------------------------------------------------------------------ main
def server_n_predict(server: dict, grid_n_predict: int) -> int:
    """Decode length per step: >= decode_s_target seconds at the np1 decode prior (acceptance noise averaging)."""
    tgt = server.get("decode_s_target")
    if not tgt:
        return grid_n_predict
    return max(grid_n_predict, int(math.ceil(float(tgt) * float(server["decode_tok_s_prior"]) / 64.0) * 64))


def load_config(p):
    cfg = json.load(open(p))
    for s in cfg["servers"]:
        for k in ("id", "device", "argv_file", "n_ctx_train", "kv_bytes_per_token", "capacity_gib", "arms"):
            if k not in s:
                raise SystemExit(f"config server {s.get('id')}: missing {k}")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--config", required=True)
    pl.add_argument("--out")
    ru = sub.add_parser("run")
    ru.add_argument("--config", required=True)
    ru.add_argument("--server", required=True)
    ru.add_argument("--out", required=True)
    ru.add_argument("--port", type=int, required=True)
    ru.add_argument("--dry-run", action="store_true")
    ex = sub.add_parser("export-catalog")
    ex.add_argument("--capture", required=True)
    ex.add_argument("--record-id", required=True)
    ex.add_argument("--title", required=True)
    ex.add_argument("--hardware", required=True)
    ex.add_argument("--source", required=True)
    ex.add_argument("--arms", default="INCUMBENT,C21,C22")
    rv = sub.add_parser("resolve")
    rv.add_argument("--capture", required=True)
    rv.add_argument("--np", type=int, required=True)
    rv.add_argument("--depth", required=True)
    rv.add_argument("--metric", default="decode_tok_s_aggregate")
    rv.add_argument("--base", default="INCUMBENT")
    rv.add_argument("--aa", default="INCUMBENT2")
    rv.add_argument("--look", type=int, default=5)
    rv.add_argument("--aa-max", type=float, default=0.02)
    a = ap.parse_args(argv)
    if a.cmd == "resolve":
        rows = [json.loads(x) for x in open(a.capture) if x.strip()]
        print(json.dumps(resolve(rows, a.np, a.depth, a.metric, a.base, a.aa, a.look, a.aa_max), indent=1))
        return 0
    if a.cmd == "export-catalog":
        rows = [json.loads(x) for x in open(a.capture) if x.strip()]
        print(json.dumps(export_catalog(rows, a.record_id, a.title, a.hardware, a.source, a.arms.split(",")), indent=1))
        return 0
    cfg = load_config(a.config)
    g = cfg["grid"]
    depths, nps, npred, margin = g["depths"], [int(x) for x in g["nps"]], int(g["n_predict"]), int(g.get("margin", 256))
    if a.cmd == "plan":
        out = {"schema": PLAN_SCHEMA, "grid": g, "servers": []}
        total_h = 0.0
        for s in cfg["servers"]:
            npred_s = server_n_predict(s, npred)
            sp = plan_server(s, depths, nps, npred_s, margin)
            h = estimate_hours(s, sp, list(s["arms"]), npred_s)
            for rep in s.get("repeat_cells") or []:
                rn = int(rep.get("n_predict", npred_s))
                h += int(rep["blocks"]) * len(s["arms"]) * (float(s.get("load_s_prior", 90)) + rn / max(1.0, float(s["decode_tok_s_prior"]) / int(rep["np"]))) / 3600.0
            sp["n_predict"] = npred_s
            sp["estimate_h"] = round(h, 2)
            total_h += h
            out["servers"].append(sp)
            meas = [c for c in sp["cells"] if c["status"] == "PLANNED"]
            print(f"{s['id']:<14} n_predict {npred_s} launches/arm {len(sp['launches'])}  cells planned {len(meas)}/{len(sp['cells'])}  est {h:.1f} h")
            for c in sp["cells"]:
                if c["status"] != "PLANNED":
                    print(f"    {c['status']:<24} np{c['np']} {c['depth']:<5} KV {c['kv_gib_at_depth']} GiB > budget {c['kv_budget_gib']} GiB")
            for L in sp["launches"]:
                print(f"    np{L['np']}: -c {L['c']} {L['kv_mode']} n_ctx_seq {L['n_ctx_seq']} (train {L['n_ctx_train']})"
                      f"{' [n_ctx_seq>train: warning only, positions <= ' + str(L['max_position']) + ']' if L['n_ctx_seq_over_train'] else ''}"
                      f" KV {L['kv_gib']} GiB depths {','.join(L['depths'])}")
        out["estimate_h_total"] = round(total_h, 2)
        print(f"TOTAL est {total_h:.1f} h")
        if a.out:
            Path(a.out).write_text(json.dumps(out, indent=1))
        return 0
    # run
    s = next(x for x in cfg["servers"] if x["id"] == a.server)
    npred = server_n_predict(s, npred)
    sp = plan_server(s, depths, nps, npred, margin)
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    srcs = cfg["documents"]["sources"]
    docs = build_documents([str(p) for pat in srcs for p in sorted(Path("/").glob(pat.lstrip("/")))],
                           max(nps), int(cfg["documents"]["min_chars"]))
    (outd / "documents.json").write_text(json.dumps(doc_manifest(docs), indent=1))
    plan_rows = [seal(dict(c, arm=arm, kind="plan")) for arm in s["arms"] for c in sp["cells"] if c["status"] != "PLANNED"]
    cap = open(outd / "capture.jsonl", "a")
    for r in plan_rows:
        cap.write(canon(r) + "\n")
    for L in sp["launches"]:
        for arm, ad in s["arms"].items():
            if arm.endswith("2") and L["np"] != 1:
                continue                                    # A/A only at np1
            store_bin = ad["store"].rstrip("/") + "/bin"
            argv_ = launch_argv(s, L, store_bin, a.port)
            env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "LD_LIBRARY_PATH": store_bin,
                   **(s.get("env") or {}), **(ad.get("env") or {})}
            stamp = {"server": s["id"], "arm": arm, "store": ad["store"], "c": L["c"], "kv_mode": L["kv_mode"],
                     "kv_evidence": L["kv_evidence"], "n_ctx_seq": L["n_ctx_seq"], "n_ctx_train": L["n_ctx_train"],
                     "rope": L["rope"], "argv": argv_, "env": env, "n_predict": npred}
            if a.dry_run:
                print(canon({"launch": stamp, "depths": L["depths"]})[:600])
                continue
            srv = Server(argv_, env, outd / f"server-{arm}-np{L['np']}.log")
            try:
                if not wait_healthy(a.port, srv.p, float(s.get("ready_timeout_s", 1800))):
                    cap.write(canon(seal(dict(stamp, np=L["np"], status="LAUNCH_FAILED"))) + "\n")
                    continue
                toks = [http(a.port, "/tokenize", {"content": d})["tokens"] for d in docs[:L["np"]]]
                need = max(DEPTH_NAMES[dn] for dn in L["depths"])
                if any(len(t) < need for t in toks):
                    raise SystemExit(f"document shorter than {need} tokens on this tokenizer")

                def req(stream, prompt, n):
                    return http(a.port, "/completion", {"prompt": prompt, "n_predict": n, "cache_prompt": True,
                                                        "id_slot": stream, "temperature": 0.0, "top_k": 1,
                                                        "ignore_eos": True, "seed": 42})
                for r in run_steps(req, toks, L, npred, float(s.get("step_cap_s", 1800)), stamp):
                    cap.write(canon(r) + "\n")
                    cap.flush()
            finally:
                if not srv.stop():
                    raise SystemExit(f"server pid {srv.p.pid} did not exit")
    # resolution blocks (e.g. A3B GPU np4 short context, S5-r2 FUTILITY_HARNESS_AA): one launch per (block, arm), ABBA
    for rep in s.get("repeat_cells") or []:
        Lr = {"np": int(rep["np"]), "depths": [rep["depth"]], "c": (DEPTH_NAMES[rep["depth"]] + npred + margin + 255) // 256 * 256 * int(rep["np"]),
              "kv_mode": kv_mode_for(s, int(rep["np"]))["mode"]}
        arms_r = list(s["arms"])
        for b in range(int(rep["blocks"])):
            for arm in block_order(arms_r, b):
                ad = s["arms"][arm]
                store_bin = ad["store"].rstrip("/") + "/bin"
                argv_ = launch_argv(s, Lr, store_bin, a.port)
                env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "LD_LIBRARY_PATH": store_bin,
                       **(s.get("env") or {}), **(ad.get("env") or {})}
                stamp = {"server": s["id"], "arm": arm, "kind": "repeat", "block": b, "store": ad["store"], "c": Lr["c"],
                         "kv_mode": Lr["kv_mode"], "argv": argv_, "env": env, "n_predict": int(rep.get("n_predict", 256))}
                if a.dry_run:
                    print(canon({"repeat": stamp, "depth": rep["depth"]})[:300])
                    continue
                srv = Server(argv_, env, outd / f"server-rep-b{b}-{arm}.log")
                try:
                    if not wait_healthy(a.port, srv.p, float(s.get("ready_timeout_s", 1800))):
                        cap.write(canon(seal(dict(stamp, np=Lr["np"], depth=rep["depth"], status="LAUNCH_FAILED"))) + "\n")
                        continue
                    toks = [http(a.port, "/tokenize", {"content": d})["tokens"] for d in docs[:Lr["np"]]]

                    def req(stream, prompt, n):
                        return http(a.port, "/completion", {"prompt": prompt, "n_predict": n, "cache_prompt": True,
                                                            "id_slot": stream, "temperature": 0.0, "top_k": 1,
                                                            "ignore_eos": True, "seed": 42})
                    for r in run_steps(req, toks, Lr, int(rep.get("n_predict", 256)), float(s.get("step_cap_s", 1800)), stamp):
                        cap.write(canon(r) + "\n")
                        cap.flush()
                finally:
                    if not srv.stop():
                        raise SystemExit(f"server pid {srv.p.pid} did not exit")
    cap.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
