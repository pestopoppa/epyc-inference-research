#!/usr/bin/env python3
"""YARN-AGG-1 arm runner: drives arms A0 (native RoPE) and A1 (YaRN f2) through a FROZEN plan against an
already-running llama-server, persists every item incrementally, and writes the gate decision record.

It never starts, stops or probes for servers by name: the filler job launches the arm's server, proves its own
PID owns the port, and passes ``--base-url`` / ``--server-pid`` / ``--argv-file`` here.

Subcommands
    plan     --plan native-gate|extension-arms --out DIR      freeze the plan (no inference; tokenizer only)
    dry-run  --plan P --out DIR [--port N]                     resolve data, builders, frozen plan, ports: no load
    status   --out DIR [--json]                                pending items per arm + estimated seconds
    run      --out DIR --arm A0|A1 --base-url URL --deadline-epoch T [--stopfile F] [--server-pid P]
             [--argv-file F] [--build-dir D] [--model PATH]     (refuses unless WINDOW_CONFIRMED=1)
    decide   --out DIR                                         write decision.json (INTERIM until both arms finish)

Gate rule (operator-ratified "Net, paired", 2026-10-21; ``arms.MARGINS``): over the paired items whose real
prompt_n lands in the NATIVE zone on BOTH arms, A1 FAILS iff the one-sided exact McNemar p <= 0.05
(``agg_scoring.net_paired_mcnemar``; RIGHT iff score >= 1.0). The verdict is FINAL only when every planned
item of every arm has a terminal result; before that ``decision.json`` says INTERIM and carries no verdict.

Files under ``--out``
    plan/plan.json, plan/items.jsonl, plan/blobs/<sha256>.txt      the frozen plan (prompts > 16 KiB as blobs)
    results/<arm>.jsonl     one record per item (append + fsync; resume skips ids already present)
    results/<arm>.errors.jsonl   transient failures (an item is retried once; a second failure is terminal)
    runs/<arm>-<ts>.json    one record per runner invocation (server identity, /props, counts, stop reason)
    decision.json           the decision record incl. the belief-kernel claim tuple fields
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.benchmark.yarn_gate import agg_scoring, cells, plans, retokenize  # noqa: E402

SCHEMA = "epyc.yarn_gate.arm_result.v1"
DECISION_SCHEMA = "epyc.yarn_gate.decision.v1"
PROTOCOL_ID = {"native-gate": "YARN-AGG-1/native-gate/net-paired-mcnemar/v1",
               "extension-arms": "YARN-AGG-1/extension-arms/net-paired-mcnemar+report/v1"}
BLOB_MIN = 16384
ARMS = ("A0", "A1")
#: planning defaults until the run has its own measured rates (E1, MI210, Qwen3.8-27B Q8_0: 128K prefill
#: ~400 tok/s, deeper ~180-250 tok/s; decode ~12 tok/s at 128K-240K depth)
PREFILL_LINEAR_TPS = 850.0
PREFILL_QUAD = 9.6e7
DEFAULT_DECODE_TPS = 12.0
DEFAULT_GEN_TOKENS = 64
SAFETY_S = 30.0


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ── plan freeze / load ────────────────────────────────────────────────────────────────────────────────
def _builder(name: str):
    from importlib import import_module
    return import_module(f"scripts.benchmark.yarn_gate.{ {'oolong': 'oolong_local', 'ruler': 'ruler_local', 'helmet': 'helmet_local', 'longbio': 'longbio_local'}[name] }")


def segment_available(seg: dict) -> tuple[bool, str]:
    try:
        mod = _builder(seg["builder"])
    except Exception as e:
        return False, f"builder {seg['builder']} not importable: {type(e).__name__}: {e}"
    a = seg["args"]
    if seg["builder"] == "oolong":
        return mod.available(a["nominal"])
    if seg["builder"] == "helmet":
        return mod.available(a["task"])
    return mod.available()


def build_segment(seg: dict) -> list[dict]:
    mod = _builder(seg["builder"])
    a = dict(seg["args"])
    if seg["builder"] == "oolong":
        tgt = a.pop("context_filter_target", None)
        if tgt is not None:
            from scripts.benchmark.yarn_gate.plans import _within
            a["context_filter"] = _within(tgt)
        return mod.build_cell(**a)
    if seg["builder"] == "ruler":
        return mod.build_cell(a["task"], a["target_real"], a["n"])
    if seg["builder"] == "helmet":
        return mod.build_cell(a["task"], a["target_real"], a["n"])
    return mod.build_cell(a["variant"], a["target_real"], a["n"])


def interleave_units(per_segment: list[list[dict]]) -> list[dict]:
    """Round-robin over context units (adjacent items sharing a context_id) across segments."""
    unit_lists = []
    for items in per_segment:
        units: list[list[dict]] = []
        for it in items:
            if units and units[-1][0]["context_id"] == it["context_id"]:
                units[-1].append(it)
            else:
                units.append([it])
        unit_lists.append(units)
    out: list[dict] = []
    i = 0
    while any(i < len(u) for u in unit_lists):
        for u in unit_lists:
            if i < len(u):
                out.extend(u[i])
        i += 1
    return out


def freeze_plan(plan_name: str, out: Path, *, segments: list[dict] | None = None,
                log: Callable[[str], None] = print) -> dict:
    pdir = out / "plan"
    if (pdir / "plan.json").exists():
        meta = json.loads((pdir / "plan.json").read_text())
        if meta["plan"] != plan_name:
            raise SystemExit(f"refusing: {pdir} holds plan {meta['plan']!r}, not {plan_name!r}")
        spec_sha = sha256_bytes(json.dumps(segments or plans.PLANS[plan_name], sort_keys=True, default=str).encode())
        if meta["spec_sha256"] != spec_sha:
            raise SystemExit(f"refusing: plan spec changed after the freeze ({meta['spec_sha256'][:12]} -> "
                             f"{spec_sha[:12]}); a pre-registered plan is never re-frozen in place")
        log(f"plan already frozen: {meta['plan_sha256']}")
        return meta
    segs = segments if segments is not None else plans.PLANS[plan_name]
    spec_sha = sha256_bytes(json.dumps(segs, sort_keys=True, default=str).encode())
    dropped, built, seg_meta = [], [], []
    for seg in segs:
        ok, why = segment_available(seg)
        if not ok:
            if seg.get("required", True):
                raise SystemExit(f"refusing to freeze: required segment {seg['name']} unavailable: {why}")
            dropped.append({"segment": seg["name"], "reason": why})
            log(f"drop optional segment {seg['name']}: {why}")
            continue
        t0 = time.time()
        items = build_segment(seg)
        for it in items:
            cells.validate_item(it)
            it["segment"] = seg["name"]
            it["arms"] = list(seg["arms"])
            it["gate"] = seg["gate"]
            it["prompt_sha256"] = cells.prompt_sha256(it)
        built.append(items)
        seg_meta.append({"segment": seg["name"], "builder": seg["builder"], "args": seg["args"],
                         "arms": seg["arms"], "gate": seg["gate"], "n_items": len(items),
                         "n_contexts": len({i["context_id"] for i in items}),
                         "build_s": round(time.time() - t0, 2), "pins": items[0].get("pins") if items else None})
        log(f"built {seg['name']}: {len(items)} items in {time.time() - t0:.1f}s")
    ordered = interleave_units(built)
    ids = [i["item_id"] for i in ordered]
    if len(ids) != len(set(ids)):
        raise SystemExit("refusing: duplicate item_id in plan")
    tmp = out / "plan.tmp"
    (tmp / "blobs").mkdir(parents=True, exist_ok=True)
    with (tmp / "items.jsonl").open("w") as f:
        for it in ordered:
            rec = dict(it)
            msgs = []
            for m in it["messages"]:
                c = m["content"]
                if isinstance(c, str) and len(c) > BLOB_MIN:
                    h = sha256_bytes(c.encode())
                    bp = tmp / "blobs" / f"{h}.txt"
                    if not bp.exists():
                        bp.write_text(c)
                    msgs.append({"role": m["role"], "content_ref": h})
                else:
                    msgs.append(m)
            rec["messages"] = msgs
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    plan_sha = sha256_bytes("\n".join(f"{i['item_id']} {i['prompt_sha256']}" for i in ordered).encode())
    meta = {"schema": "epyc.yarn_gate.plan.v1", "plan": plan_name, "frozen_at": utcnow(), "plan_sha256": plan_sha,
            "spec_sha256": spec_sha, "n_items": len(ordered), "segments": seg_meta, "dropped": dropped,
            "margins": __import__("scripts.benchmark.yarn_gate.arms", fromlist=["MARGINS"]).MARGINS,
            "arm_ctx": plans.ARM_CTX, "protocol_id": PROTOCOL_ID.get(plan_name)}
    (tmp / "plan.json").write_text(json.dumps(meta, indent=1, default=str))
    os.replace(tmp, pdir)
    log(f"plan frozen: {plan_sha} ({len(ordered)} items)")
    return meta


class Plan:
    def __init__(self, out: Path):
        self.dir = out / "plan"
        self.meta = json.loads((self.dir / "plan.json").read_text())
        self.items = [json.loads(l) for l in (self.dir / "items.jsonl").read_text().splitlines() if l.strip()]
        self._blob_cache: tuple[str, str] | None = None

    def _blob(self, h: str) -> str:
        if self._blob_cache and self._blob_cache[0] == h:
            return self._blob_cache[1]
        txt = (self.dir / "blobs" / f"{h}.txt").read_text()
        if sha256_bytes(txt.encode()) != h:
            raise RuntimeError(f"blob {h} corrupted")
        self._blob_cache = (h, txt)
        return txt

    def messages(self, item: dict) -> list[dict]:
        return [{"role": m["role"], "content": self._blob(m["content_ref"])} if "content_ref" in m else m
                for m in item["messages"]]

    def for_arm(self, arm: str) -> list[dict]:
        return [i for i in self.items if arm in i["arms"]]


# ── results ───────────────────────────────────────────────────────────────────────────────────────────
def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a torn last line from a hard kill: the item is simply re-run
    return out


def append_jsonl(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def results_by_id(out: Path, arm: str, plan: Plan | None = None) -> dict[str, dict]:
    """Terminal results for ``arm``, keyed by item_id. A result whose prompt sha differs from the frozen plan's
    is refused (raises): it was produced against another plan."""
    res = {}
    want = {i["item_id"]: i["prompt_sha256"] for i in plan.items} if plan else None
    for r in load_jsonl(out / "results" / f"{arm}.jsonl"):
        if want is not None and r["item_id"] in want and r["prompt_sha256"] != want[r["item_id"]]:
            raise SystemExit(f"refusing: {arm} result {r['item_id']} prompt sha differs from the frozen plan")
        res[r["item_id"]] = r
    return res


def error_counts(out: Path, arm: str) -> dict[str, int]:
    c: dict[str, int] = {}
    for r in load_jsonl(out / "results" / f"{arm}.errors.jsonl"):
        c[r["item_id"]] = c.get(r["item_id"], 0) + 1
    return c


# ── client ────────────────────────────────────────────────────────────────────────────────────────────
class ServerError(RuntimeError):
    def __init__(self, msg: str, *, transient: bool = True, status: int | None = None):
        super().__init__(msg)
        self.transient = transient
        self.status = status


class Client:
    def __init__(self, base_url: str, opener: Callable | None = None):
        self.base = base_url.rstrip("/")
        self._open = opener or urllib.request.urlopen

    def _get(self, path: str, timeout: float = 30.0) -> Any:
        with self._open(urllib.request.Request(self.base + path), timeout=timeout) as r:
            return json.loads(r.read())

    def model_id(self) -> str:
        return self._get("/v1/models")["data"][0]["id"]

    def props(self) -> dict:
        try:
            return self._get("/props")
        except Exception as e:  # informational only
            return {"error": f"{type(e).__name__}: {e}"}

    def chat(self, messages: list[dict], *, max_tokens: int, thinking: bool, timeout: float) -> dict:
        payload = {"model": "default", "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
                   "seed": 0, "stream": False, "cache_prompt": True, "id_slot": 0,
                   "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with self._open(req, timeout=timeout) as r:
                body = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read()[:500].decode(errors="replace")
            raise ServerError(f"HTTP {e.code}: {detail}", transient=e.code >= 500, status=e.code) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise ServerError(f"{type(e).__name__}: {e}", transient=True) from None
        ch = body["choices"][0]
        msg = ch.get("message") or {}
        return {"content": msg.get("content") or "", "reasoning_content": msg.get("reasoning_content") or "",
                "finish_reason": ch.get("finish_reason"), "usage": body.get("usage") or {},
                "timings": body.get("timings") or {}}


# ── per-item cost model (for deadline-safe starts) ───────────────────────────────────────────────────
class Rates:
    """Prefill time model t(L) = scale * (L / PREFILL_LINEAR_TPS + L^2 / PREFILL_QUAD), fitted to E1 (MI210,
    Qwen3.8-27B Q8_0, -ub 2048: 3,310 tok in 3.96 s; 127,946 tok in 320.7 s), so a 447K context costs ~43 min
    rather than the ~25 min a flat rate implies. ``scale`` is re-fitted from this run's own full prefills."""

    def __init__(self, results: Iterable[dict]):
        ratios, dec, gen = [], [], []
        for r in results:
            t = r.get("timings") or {}
            pn = t.get("prompt_n") or 0
            if pn > 8192 and t.get("prompt_ms"):
                ratios.append((float(t["prompt_ms"]) / 1000.0) / self.model_s(pn))
            if (t.get("predicted_n") or 0) > 4 and t.get("predicted_per_second"):
                dec.append(float(t["predicted_per_second"]))
            if t.get("predicted_n"):
                gen.append(int(t["predicted_n"]))
        med = lambda xs, d: sorted(xs)[len(xs) // 2] if xs else d
        self.scale = min(max(med(ratios, 1.0), 0.5), 3.0)
        self.decode = med(dec, DEFAULT_DECODE_TPS)
        self.gen = med(gen, DEFAULT_GEN_TOKENS)

    @staticmethod
    def model_s(n: int) -> float:
        return n / PREFILL_LINEAR_TPS + n * n / PREFILL_QUAD

    def estimate(self, item: dict, cached: bool) -> float:
        ctx = 0 if cached else (item.get("real_n_offline") or item.get("nominal_len") or 0)
        gen = item["max_tokens"] if item.get("thinking") else min(item["max_tokens"], max(self.gen * 2, 32))
        return self.scale * self.model_s(ctx) + gen / self.decode + 2.0


# ── run one arm ───────────────────────────────────────────────────────────────────────────────────────
def classify_failure(score: float, output: str, finish_reason: str | None, parse_conf: str | None) -> str | None:
    from scripts.benchmark.oolong_adapter import classify_failure as cf
    return cf(score, output, finish_reason, parse_conf or "high")


def pending_for_arm(plan: Plan, out: Path, arm: str) -> list[dict]:
    done = results_by_id(out, arm, plan)
    other = [a for a in ARMS if a != arm]
    other_done = set()
    for a in other:
        other_done |= set(results_by_id(out, a, plan))
    todo = [i for i in plan.for_arm(arm) if i["item_id"] not in done]
    # catch-up first (items the other arm already has, so pairs complete), plan order otherwise
    return [i for i in todo if i["item_id"] in other_done] + [i for i in todo if i["item_id"] not in other_done]


def run_arm(out: Path, arm: str, client: Client, *, deadline: float, stopfile: str | None = None,
            server: dict | None = None, now: Callable[[], float] = time.time,
            log: Callable[[str], None] = print) -> dict:
    plan = Plan(out)
    model_id = client.model_id()
    props = client.props()
    errs = error_counts(out, arm)
    seen = list(results_by_id(out, arm, plan).values())
    rates = Rates(seen)
    todo = pending_for_arm(plan, out, arm)
    started = utcnow()
    n_ok = n_err = n_skipped = 0
    stop_reason = "plan-exhausted"
    last_ctx = None
    for item in todo:
        if stopfile and os.path.exists(stopfile):
            stop_reason = "stopfile"
            break
        if errs.get(item["item_id"], 0) >= 2:
            continue
        left = deadline - now()
        est = rates.estimate(item, cached=item["context_id"] == last_ctx)
        if est + SAFETY_S > left:
            n_skipped += 1
            continue  # a smaller later item may still fit
        t0 = now()
        try:
            comp = client.chat(plan.messages(item), max_tokens=item["max_tokens"], thinking=item["thinking"],
                               timeout=min(14400.0, max(60.0, left - 5.0)))
        except ServerError as e:
            n_err += 1
            errs[item["item_id"]] = errs.get(item["item_id"], 0) + 1
            rec = {"item_id": item["item_id"], "arm": arm, "time": utcnow(), "error": str(e), "status": e.status,
                   "transient": e.transient}
            append_jsonl(out / "results" / f"{arm}.errors.jsonl", rec)
            log(f"{arm} {item['item_id']} ERROR {e}")
            if errs[item["item_id"]] >= 2 or not e.transient:
                append_jsonl(out / "results" / f"{arm}.jsonl", {
                    "schema": SCHEMA, "item_id": item["item_id"], "arm": arm, "status": "error",
                    "prompt_sha256": item["prompt_sha256"], "cell": item["cell"], "segment": item["segment"],
                    "error": str(e), "time": utcnow()})
            if e.transient and e.status is None:
                stop_reason = "server-unreachable"
                break
            last_ctx = None
            continue
        t1 = now()
        sc = cells.score(item, comp["content"], model=model_id)
        tm = comp["timings"]
        real_n = None
        src = None
        if isinstance(tm.get("prompt_n"), int):
            real_n, src = retokenize.real_prompt_n(tm), "server_timings"
        elif isinstance(comp["usage"].get("prompt_tokens"), int):
            real_n, src = int(comp["usage"]["prompt_tokens"]), "server_timings"
        zone = retokenize.label_zone(real_n, source=src) if real_n else None
        rec = {"schema": SCHEMA, "item_id": item["item_id"], "arm": arm, "status": "ok",
               "prompt_sha256": item["prompt_sha256"], "plan_sha256": plan.meta["plan_sha256"],
               "cell": item["cell"], "segment": item["segment"], "suite": item["suite"], "task": item["task"],
               "context_id": item["context_id"], "context_window_id": item["context_id"], "gate": item["gate"],
               "thinking": item["thinking"], "nominal_len": item["nominal_len"],
               "real_n_offline": item.get("real_n_offline"), "real_n": real_n, "n_source": src, "zone": zone,
               "bin": retokenize.bin_label(real_n) if real_n else None, "num_labels": item.get("num_labels"),
               "score": sc["score"], "scorer": item["scorer"], "scorer_detail": {k: v for k, v in sc.items() if k != "score"},
               "right": sc["score"] >= 1.0, "gold": item["gold"], "output": comp["content"],
               "reasoning_chars": len(comp["reasoning_content"]), "finish_reason": comp["finish_reason"],
               "attempted": True, "answered": bool(comp["content"].strip()) and comp["finish_reason"] != "length",
               "no_answer": not (bool(comp["content"].strip()) and comp["finish_reason"] != "length"),
               "failure_class": classify_failure(sc["score"], comp["content"], comp["finish_reason"],
                                                 sc.get("parse_confidence")),
               "usage": comp["usage"], "timings": tm, "model_id": model_id, "server": server or {},
               "t_start": t0, "t_end": t1, "wall_s": round(t1 - t0, 3), "time": utcnow()}
        append_jsonl(out / "results" / f"{arm}.jsonl", rec)
        n_ok += 1
        last_ctx = item["context_id"]
        seen.append(rec)
        rates = Rates(seen)
        log(f"{arm} {item['item_id']} score={sc['score']:.2f} real_n={real_n} zone={zone} wall={t1 - t0:.1f}s")
    else:
        stop_reason = "plan-exhausted" if n_skipped == 0 else "deadline"
    remaining = len(pending_for_arm(plan, out, arm))
    run = {"schema": "epyc.yarn_gate.run.v1", "arm": arm, "started": started, "ended": utcnow(),
           "model_id": model_id, "props": props, "server": server or {}, "n_ok": n_ok, "n_err": n_err,
           "n_skipped_deadline": n_skipped, "remaining": remaining, "stop_reason": stop_reason,
           "plan_sha256": plan.meta["plan_sha256"]}
    rd = out / "runs"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / f"{arm}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json").write_text(json.dumps(run, indent=1, default=str))
    return run


# ── status / decision ─────────────────────────────────────────────────────────────────────────────────
def status(out: Path) -> dict:
    plan = Plan(out)
    st = {"plan": plan.meta["plan"], "plan_sha256": plan.meta["plan_sha256"], "arms": {}}
    for arm in ARMS:
        planned = plan.for_arm(arm)
        if not planned:
            continue
        done = results_by_id(out, arm, plan)
        errs = error_counts(out, arm)
        pend = [i for i in pending_for_arm(plan, out, arm) if errs.get(i["item_id"], 0) < 2]
        rates = Rates(done.values())
        est, last = 0.0, None
        for i in pend:
            est += rates.estimate(i, cached=i["context_id"] == last)
            last = i["context_id"]
        st["arms"][arm] = {"planned": len(planned), "done": len(done), "pending": len(pend),
                           "est_pending_s": round(est)}
    st["complete"] = all(a["pending"] == 0 for a in st["arms"].values())
    # lagging arm first: fewer done relative to plan; tie -> A0
    st["order"] = sorted(st["arms"], key=lambda a: (st["arms"][a]["done"] / max(1, st["arms"][a]["planned"]), a))
    return st


def _sha_file(p: Path) -> str | None:
    return sha256_bytes(p.read_bytes()) if p.exists() else None


def decide(out: Path, *, identity: dict | None = None) -> dict:
    plan = Plan(out)
    st = status(out)
    res = {a: results_by_id(out, a, plan) for a in ARMS}
    final = st["complete"]
    gate_items = [i for i in plan.items if i["gate"] == "native_mcnemar" and set(ARMS) <= set(i["arms"])]
    a0, a1, excluded = {}, {}, []
    per_cell: dict[str, dict] = {}
    for it in gate_items:
        r0, r1 = res["A0"].get(it["item_id"]), res["A1"].get(it["item_id"])
        if not (r0 and r1 and r0.get("status") == "ok" and r1.get("status") == "ok"):
            if r0 or r1:
                excluded.append({"item_id": it["item_id"], "reason": "unpaired or error"})
            continue
        if r0.get("zone") != "native" or r1.get("zone") != "native":
            excluded.append({"item_id": it["item_id"], "reason": f"zone A0={r0.get('zone')} A1={r1.get('zone')}"})
            continue
        a0[it["item_id"]], a1[it["item_id"]] = r0["score"], r1["score"]
        per_cell.setdefault(it["cell"], {"a0": {}, "a1": {}})
        per_cell[it["cell"]]["a0"][it["item_id"]] = r0["score"]
        per_cell[it["cell"]]["a1"][it["item_id"]] = r1["score"]
    pooled = agg_scoring.net_paired_mcnemar(a0, a1)
    cells_out = {c: agg_scoring.net_paired_mcnemar(v["a0"], v["a1"]) for c, v in sorted(per_cell.items())}
    report = {}
    for arm in ARMS:
        rs = [r for r in res[arm].values() if r.get("status") == "ok"]
        by: dict[str, list] = {}
        for r in rs:
            by.setdefault(r["cell"], []).append(r)
        report[arm] = {c: {**agg_scoring.bucket_stats(v), "zones": sorted({str(x.get("zone")) for x in v}),
                           "real_n_median": sorted(x["real_n"] for x in v if x.get("real_n"))[len(v) // 2]
                           if any(x.get("real_n") for x in v) else None,
                           "failure_classes": {k: sum(1 for x in v if x.get("failure_class") == k)
                                               for k in ("wrong", "no_answer_budget", "refusal", "unparsable")}}
                       for c, v in sorted(by.items())}
    verdict = pooled["verdict"] if final else "INTERIM (no verdict until both arms finish the frozen plan)"
    if final and pooled["n_pairs"] == 0:
        verdict = "NO-EVIDENCE (zero native pairs)"
    rfiles = {a: _sha_file(out / "results" / f"{a}.jsonl") for a in ARMS}
    plan_name = plan.meta["plan"]
    date = utcnow()
    rec = {
        "schema": DECISION_SCHEMA, "status": "FINAL" if final else "INTERIM", "written": date, "plan": plan_name,
        "plan_sha256": plan.meta["plan_sha256"], "protocol_id": plan.meta.get("protocol_id"),
        "rule": plan.meta["margins"]["in_native_rule"], "alpha": plan.meta["margins"]["alpha"],
        "verdict": verdict, "pooled_native": pooled, "per_cell_native": cells_out,
        "excluded_from_gate": excluded, "per_arm_cells": report, "progress": st, "dropped_segments":
            plan.meta.get("dropped", []), "results_sha256": rfiles, "identity": identity or {},
        "power_note": "NOT-FAIL on few discordant pairs is weak evidence; read b, c and the mean difference.",
    }
    # Belief-kernel write side: the producer-authored claim tuple fields (projection lives in scripts/vidya).
    rec["claim_tuple"] = {
        "measurement_id": f"yarn-gate:{plan_name}:{plan.meta['plan_sha256'][:12]}",
        "metric": "mean paired score difference A1 - A0 inside the native window (net-paired McNemar gate)",
        "value": pooled["mean_score_diff_a1_minus_a0"], "unit": "score fraction per item (0-1)",
        "metric_direction": "higher_better", "category": "quality", "date": date,
        "claim": f"YaRN f2 (A1) {'does not fall below' if pooled['verdict'] == 'NOT-FAIL' else 'falls below'} "
                 f"native RoPE (A0) inside the native window on {plan_name} "
                 f"(b={pooled['b_a0right_a1wrong']}, c={pooled['c_a0wrong_a1right']}, p={pooled['p_one_sided']:.4g})",
        "protocol_id": plan.meta.get("protocol_id"), "reps": pooled["n_pairs"], "reps_basis": "scored (paired, native zone)",
        "attestation_path": str(out / "decision.json"), "attestation_locator": "pooled_native",
        "status": "FINAL" if final else "INTERIM", "verdict": verdict,
        "extra": {"b": pooled["b_a0right_a1wrong"], "c": pooled["c_a0wrong_a1right"], "p_one_sided": pooled["p_one_sided"],
                  "alpha": pooled["alpha"], "plan_sha256": plan.meta["plan_sha256"], "results_sha256": rfiles,
                  "identity": identity or {}},
    }
    tmp = out / "decision.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    os.replace(tmp, out / "decision.json")
    return rec


# ── dry run ───────────────────────────────────────────────────────────────────────────────────────────
def dry_run(plan_name: str, out: Path, port: int | None) -> int:
    rc = 0
    print(f"plan {plan_name}: {len(plans.PLANS[plan_name])} segments")
    for seg in plans.PLANS[plan_name]:
        ok, why = segment_available(seg)
        tag = "OK " if ok else ("MISSING" if seg.get("required", True) else "OPTIONAL-MISSING")
        if not ok and seg.get("required", True):
            rc = 3
        print(f"  [{tag}] {seg['name']:<28} arms={','.join(seg['arms'])} gate={seg['gate']}  {why}")
    pj = out / "plan" / "plan.json"
    if pj.exists():
        m = json.loads(pj.read_text())
        print(f"frozen plan: {pj} sha={m['plan_sha256']} items={m['n_items']} dropped={m.get('dropped')}")
        st = status(out)
        for a, v in st["arms"].items():
            print(f"  {a}: planned={v['planned']} done={v['done']} pending={v['pending']} est={v['est_pending_s'] / 60:.0f} min")
    else:
        print(f"plan NOT frozen yet under {out} (the first window freezes it; or run `plan` now)")
    tj = Path(cells.DEFAULT_TOKENIZER_JSON)
    print(f"tokenizer.json: {'OK' if tj.is_file() else 'MISSING'} {tj}")
    if port:
        import socket
        s = socket.socket()
        busy = s.connect_ex(("127.0.0.1", port)) == 0
        s.close()
        print(f"port {port}: {'IN USE (job would refuse)' if busy else 'free'}")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--plan", choices=sorted(plans.PLANS), required=True); p.add_argument("--out", required=True)
    d = sub.add_parser("dry-run"); d.add_argument("--plan", choices=sorted(plans.PLANS), required=True); d.add_argument("--out", required=True); d.add_argument("--port", type=int)
    s = sub.add_parser("status"); s.add_argument("--out", required=True); s.add_argument("--json", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--out", required=True); r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--base-url", required=True); r.add_argument("--deadline-epoch", type=float, required=True)
    r.add_argument("--stopfile"); r.add_argument("--server-pid", type=int); r.add_argument("--argv-file")
    r.add_argument("--build-dir"); r.add_argument("--model")
    c = sub.add_parser("decide"); c.add_argument("--out", required=True); c.add_argument("--identity-file")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.cmd == "plan":
        out.mkdir(parents=True, exist_ok=True)
        freeze_plan(a.plan, out)
        return 0
    if a.cmd == "dry-run":
        return dry_run(a.plan, out, a.port)
    if a.cmd == "status":
        st = status(out)
        print(json.dumps(st) if a.json else json.dumps(st, indent=1))
        return 0
    if a.cmd == "decide":
        ident = json.loads(Path(a.identity_file).read_text()) if a.identity_file and Path(a.identity_file).exists() else None
        rec = decide(out, identity=ident)
        print(json.dumps({k: rec[k] for k in ("status", "verdict", "plan_sha256")}, indent=1))
        print(json.dumps(rec["pooled_native"], indent=1))
        return 0
    if os.environ.get("WINDOW_CONFIRMED") != "1":
        print("refusing: inference needs a granted window (WINDOW_CONFIRMED=1)", file=sys.stderr)
        return 2
    server = {"pid": a.server_pid, "base_url": a.base_url, "build_dir": a.build_dir, "model": a.model}
    if a.argv_file and Path(a.argv_file).exists():
        txt = Path(a.argv_file).read_text()
        server["argv"] = txt.strip()
        server["argv_sha256"] = sha256_bytes(txt.encode())
    run = run_arm(out, a.arm, Client(a.base_url), deadline=a.deadline_epoch, stopfile=a.stopfile, server=server)
    print(json.dumps({k: run[k] for k in ("arm", "n_ok", "n_err", "n_skipped_deadline", "remaining", "stop_reason")}))
    return 0 if run["stop_reason"] != "server-unreachable" else 7


if __name__ == "__main__":
    raise SystemExit(main())
