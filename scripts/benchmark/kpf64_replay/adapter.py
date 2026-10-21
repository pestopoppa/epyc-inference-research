#!/usr/bin/env python3
"""KPF-64 transcript adapter: real agent-trace stores -> ordered per-session request lists.

Store parsing is copied from the 2026-10-21 trace-chunk-reuse study (reuse.py): Claude Code
(~/.claude/projects/*/*.jsonl + subagents), Codex (~/.codex/sessions/Y/M/D/*.jsonl) and
opencode (~/.local/share/opencode/opencode.db). Request model is the study's: request k's prompt
is everything before the k-th generated run; the original assistant output is kept as the
reference (`ref`) and the replay may substitute live output (--history live).

PRIVACY: the per-session files written to --out contain transcript text (caller-given dir, keep it
out of git). manifest.json and everything printed contain aggregates only.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sqlite3
import sys
from pathlib import Path

CHARS_PER_TOKEN = 4  # study's estimate; real token counts come from the server at replay time
STORES = ("claude", "codex", "opencode")


def txt(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(txt(i.get("text") if isinstance(i, dict) and "text" in i
                             else (i if isinstance(i, str) else json.dumps(i, sort_keys=True))) for i in x)
    return json.dumps(x, sort_keys=True)


# ---- item iterators: yield (kind, text, generated) or ("RESET",) --------------------------------
def claude_items(path):
    ids = {}
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = d.get("type")
        if t == "system" and d.get("subtype") == "compact_boundary":
            yield ("RESET",)
            continue
        if t == "attachment":
            yield ("attach", json.dumps(d.get("attachment"), sort_keys=True, default=str), False)
            continue
        if t not in ("user", "assistant"):
            continue
        c = (d.get("message") or {}).get("content")
        gen = t == "assistant"
        role = "asst" if gen else "user"
        if isinstance(c, str):
            yield (role, c, gen)
            continue
        for b in c or []:
            bt = b.get("type")
            if bt == "text":
                yield (role, b.get("text", ""), gen)
            elif bt == "thinking":
                yield ("asst", b.get("thinking", ""), True)
            elif bt == "tool_use":
                ids[b.get("id")] = b.get("name")
                yield ("asst", (b.get("name") or "") + json.dumps(b.get("input"), sort_keys=True), True)
            elif bt == "tool_result":
                yield ("tool:" + str(ids.get(b.get("tool_use_id"), "?")), txt(b.get("content")), False)


def codex_items(path):
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "response_item":
            continue
        p = d.get("payload") or {}
        pt = p.get("type")
        if pt == "message":
            r = p.get("role")
            yield ({"user": "user", "assistant": "asst"}.get(r, "sys"), txt(p.get("content")), r == "assistant")
        elif pt == "reasoning":
            yield ("asst", txt(p.get("summary")) + txt(p.get("content")), True)
        elif pt in ("function_call", "custom_tool_call"):
            yield ("asst", str(p.get("name")) + txt(p.get("arguments") or p.get("input")), True)
        elif pt in ("function_call_output", "custom_tool_call_output"):
            yield ("tool:exec", txt(p.get("output")), False)


def opencode_items(sid, con):
    roles = {}
    for mid, data in con.execute("select id,data from message where session_id=? order by time_created,id", (sid,)):
        try:
            roles[mid] = json.loads(data).get("role")
        except Exception:
            pass
    for mid, data in con.execute("select message_id,data from part where session_id=? order by time_created,id", (sid,)):
        try:
            d = json.loads(data)
        except Exception:
            continue
        t = d.get("type")
        role = roles.get(mid)
        if t == "text":
            yield (("user" if role == "user" else "asst"), d.get("text", ""), role != "user")
        elif t == "reasoning":
            yield ("asst", d.get("text", ""), True)
        elif t == "tool":
            st = d.get("state") or {}
            yield ("asst", str(d.get("tool")) + json.dumps(st.get("input") or {}, sort_keys=True), True)
            if "output" in st:
                yield ("tool:" + str(d.get("tool")), txt(st.get("output")), False)


# ---- items -> turns -----------------------------------------------------------------------------
def render(kind, text):
    return f"<{kind}>\n{text}\n</{kind}>\n"


def items_to_turns(items):
    """Turn k = {pre, ref, reset}. Prompt_k = sum_{i<k}(pre_i + out_i) + pre_k (out_i = ref or live).
    A request is issued at the first generated item that follows non-generated pending items."""
    turns, pre, ref = [], [], []
    st = {"non_gen": False, "reset": False}

    def emit():
        turns.append({"pre": "".join(pre), "ref": "".join(ref), "reset": st["reset"]})
        pre.clear()
        ref.clear()
        st["non_gen"] = False
        st["reset"] = False

    state = "pre"
    for it in items:
        if it[0] == "RESET":
            if state == "gen" or st["non_gen"]:
                emit()
            state = "pre"
            st["reset"] = True
            continue
        kind, text, gen = it
        if not text:
            continue
        s = render(kind, text)
        if gen:
            if state == "pre" and st["non_gen"]:
                state = "gen"
            (ref if state == "gen" else pre).append(s)  # stray leading generated item: context only
        else:
            if state == "gen":
                emit()
                state = "pre"
            pre.append(s)
            st["non_gen"] = True
    if state == "gen" or st["non_gen"]:
        emit()
    return turns


def common_prefix_len(a: str, b: str) -> int:
    if b.startswith(a):
        return len(a)
    if a.startswith(b):
        return len(b)
    n, lo, step = min(len(a), len(b)), 0, 4096
    while lo < n and a[lo:lo + step] == b[lo:lo + step]:
        lo += step
    lo = min(lo, n)
    while lo < n and a[lo] == b[lo]:
        lo += 1
    return lo


def iter_prompts(turns, outputs=None):
    """Yield (k, prompt, prev_text); prev_text = previous prompt + its output (what the slot cache
    holds if it kept the generated tokens), None at a lineage start. outputs=None -> reference."""
    hist, prev = "", None
    for k, t in enumerate(turns):
        if t.get("reset"):
            hist, prev = "", None
        prompt = hist + t["pre"]
        yield k, prompt, prev
        out = t["ref"] if outputs is None or k >= len(outputs) or outputs[k] is None else outputs[k]
        hist = prompt + out
        prev = hist


def ideal_profile(turns):
    """Per request (prompt_chars, ideal_chars): common prefix with the longest previous prompt
    (incl. its output) in the same lineage; a RESET starts a new lineage with ideal 0."""
    rows, lineage = [], []
    for k, prompt, prev in iter_prompts(turns):
        if prev is None:
            lineage = []
        ideal = max((common_prefix_len(prompt, p) for p in (lineage[-1:] + [max(lineage, key=len)] if lineage else [])),
                    default=0)
        rows.append((len(prompt), ideal))
        lineage.append(prompt + turns[k]["ref"])
    return rows


def session_stats(turns):
    prof = ideal_profile(turns)
    pc = sum(p for p, _ in prof)
    ic = sum(i for _, i in prof)
    return {"requests": len(turns), "prompt_chars": pc, "ideal_chars": ic,
            "max_prompt_chars": max((p for p, _ in prof), default=0),
            "ideal_share": (ic / pc) if pc else 0.0}


def truncate(turns, max_chars, max_requests):
    """Keep leading requests whose prompt fits max_chars (prefix structure preserved)."""
    out = []
    for k, prompt, _ in iter_prompts(turns):
        if len(prompt) > max_chars or len(out) >= max_requests:
            break
        out.append(turns[k])
    return out


# ---- discovery + sampling -----------------------------------------------------------------------
def candidates(store, home):
    if store == "claude":
        fs = glob.glob(home + "/.claude/projects/*/*.jsonl") + glob.glob(home + "/.claude/projects/*/*/subagents/*.jsonl")
        return [("claude", f, None) for f in sorted(fs)]
    if store == "codex":
        return [("codex", f, None) for f in sorted(glob.glob(home + "/.codex/sessions/*/*/*/*.jsonl"))]
    db = home + "/.local/share/opencode/opencode.db"
    if not os.path.exists(db):
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return [("opencode", r[0], db) for r in con.execute("select id from session order by id")]
    finally:
        con.close()


def load_turns(cand):
    store, ref, db = cand
    if store == "claude":
        return items_to_turns(claude_items(ref))
    if store == "codex":
        return items_to_turns(codex_items(ref))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return items_to_turns(opencode_items(ref, con))
    finally:
        con.close()


def quotas(n, stores):
    base, extra = divmod(n, len(stores))
    return {s: base + (1 if i < extra else 0) for i, s in enumerate(stores)}


def sample_sessions(home, n=20, seed=64, stores=STORES, min_requests=10,
                    max_prompt_chars=200_000, max_requests=200):
    """Deterministic. Per store: candidates sorted, shuffled with Random(f'{seed}:{store}'); the
    first `quota` sessions passing the filters (after truncation) are taken. A store that cannot
    fill its quota hands the shortfall to the others (round-robin, fixed order)."""
    q = quotas(n, stores)
    pools = {}
    for s in stores:
        c = candidates(s, home)
        random.Random(f"{seed}:{s}").shuffle(c)
        pools[s] = iter(c)
    picked, got, live = [], {s: 0 for s in stores}, list(stores)

    def take(s):
        for cand in pools[s]:
            try:
                turns = truncate(load_turns(cand), max_prompt_chars, max_requests)
            except Exception:
                continue
            if len(turns) >= min_requests:
                picked.append((s, cand, turns, session_stats(turns)))
                got[s] += 1
                return True
        return False

    for s in stores:  # pass 1: own quota
        while got[s] < q[s] and take(s):
            pass
    while len(picked) < n:  # pass 2: shortfall from whichever store still has candidates
        progressed = False
        for s in stores:
            if len(picked) < n and take(s):
                progressed = True
        if not progressed:
            break
    return picked


def totals(sessions):
    pc = sum(s["prompt_chars"] for s in sessions)
    ic = sum(s["ideal_chars"] for s in sessions)
    by = {}
    for st in STORES:
        sub = [s for s in sessions if s["store"] == st]
        if sub:
            p, i = sum(s["prompt_chars"] for s in sub), sum(s["ideal_chars"] for s in sub)
            by[st] = {"sessions": len(sub), "requests": sum(s["requests"] for s in sub),
                      "prompt_tok_est": p // CHARS_PER_TOKEN, "ideal_share": i / p if p else 0.0}
    return {"sessions": len(sessions), "requests": sum(s["requests"] for s in sessions),
            "prompt_tok_est": pc // CHARS_PER_TOKEN, "ideal_tok_est": ic // CHARS_PER_TOKEN,
            "ideal_share": ic / pc if pc else 0.0, "by_store": by}


def build_manifest(picked, params):
    sessions, idx = [], {s: 0 for s in STORES}
    for s, cand, _turns, st in picked:
        sid = f"{s}-{idx[s]:02d}"
        idx[s] += 1
        sessions.append({"id": sid, "store": s,
                         "source_hash": hashlib.blake2b(str(cand[1]).encode(), digest_size=6).hexdigest(), **st})
    return {"schema": "kpf64.sessions.v1", "params": params, "sessions": sessions, "totals": totals(sessions)}


def write_sessions(picked, out_dir, manifest):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for (s, _cand, turns, _st), meta in zip(picked, manifest["sessions"]):
        (out / f"{meta['id']}.json").write_text(json.dumps({"id": meta["id"], "store": s, "turns": turns}))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="dir for per-session request files (contain transcript text)")
    ap.add_argument("--home", default=os.environ.get("HOME", "/home/node"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--stores", default=",".join(STORES))
    ap.add_argument("--min-requests", type=int, default=10)
    ap.add_argument("--max-prompt-chars", type=int, default=200_000, help="~50k tokens; must fit one slot's ctx")
    ap.add_argument("--max-requests", type=int, default=200)
    ap.add_argument("--history", choices=("reference", "live"), default="reference",
                    help="recorded in the manifest; the driver regenerates outputs when 'live'")
    a = ap.parse_args(argv)
    picked = sample_sessions(a.home, a.n, a.seed, tuple(a.stores.split(",")), a.min_requests,
                             a.max_prompt_chars, a.max_requests)
    man = build_manifest(picked, {k: v for k, v in vars(a).items() if k not in ("out", "home")})
    write_sessions(picked, a.out, man)
    t = man["totals"]
    print(json.dumps({"sessions": t["sessions"], "requests": t["requests"], "prompt_tok_est": t["prompt_tok_est"],
                      "ideal_share": round(t["ideal_share"], 5),
                      "by_store": {k: {**v, "ideal_share": round(v["ideal_share"], 5)} for k, v in t["by_store"].items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
