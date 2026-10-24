#!/usr/bin/env python3
"""KPF-64 transcript adapter: real agent-trace stores -> ordered per-session request lists.

Store parsing is copied from the 2026-10-21 trace-chunk-reuse study (reuse.py): Claude Code
(~/.claude/projects/*/*.jsonl + subagents), Codex (~/.codex/sessions/Y/M/D/*.jsonl) and
opencode (the operator's READ-ONLY snapshots /mnt/raid0/llm/opencode-db/{epyc,host-daniele}.db, 2026-10-22;
live stores are refused, see opencode_ro.py). Request model is the study's: request k's prompt
is everything before the k-th generated run; the original assistant output is kept as the
reference (`ref`) and the replay may substitute live output (--history live).

PRIVACY: the per-session files written to --out contain transcript text (caller-given dir, keep it
out of git). manifest.json and everything printed contain aggregates only.

KPF-LB2 amendments (2026-10-22, before any live arm; see PREDECLARED.md "Amendment 1"):
  * sub-agent transcripts are CHILDREN of their parent session, not independent sessions: Claude
    `<proj>/<parent>/subagents/agent-<id>.jsonl` linked through the parent's Task/Agent tool_use ->
    `toolUseResult.agentId` (fallback: time); opencode `session.parent_id` linked through the parent's
    `task` tool part `state.metadata.sessionId`;
  * every turn carries `gap_s` (think/tool time between the previous reply's end and this request) and
    provider `usage`, so the replay can emit think-time gaps and report the provider-realised cache share;
  * opencode layout (V1 message+part, V2 session_message) is detected from sqlite_master and one layout
    is read per session, never both; a store or sample with zero sessions raises instead of shrinking;
  * ideal bounds: within-session (lineage) and cross-session-head (any other session's earlier prompt,
    512-char-block lower bound), plus fan-out stats (child share of requests, peak concurrency).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import random
import sqlite3
import sys
import threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opencode_ro  # noqa: E402

OPENCODE_RO_DBS = opencode_ro.OPENCODE_RO_DBS     # default opencode sources: both read-only snapshots
CHARS_PER_TOKEN = 4  # study's estimate; real token counts come from the server at replay time
STORES = ("claude", "codex", "opencode")
SPAWN_TOOLS = ("Task", "Agent")          # Claude Code tool_use names that launch a sub-agent
HEAD_BLOCK = 1024                        # chars; granularity of the cross-session-head bound (~256 tok)
#: Provider-usage conventions. `exclusive`: input_tokens excludes cache read/write (Anthropic);
#: `inclusive`: input_tokens already includes cached tokens (OpenAI/Codex).
USAGE_CONVENTION = {"claude": "exclusive", "codex": "inclusive", "opencode": "exclusive_assumed"}


class ZeroSessionsError(RuntimeError):
    """A store (or the whole sample) produced zero sessions: fail loudly, never replay less silently."""


def ts_epoch(x):
    """ISO-8601 string (Claude/Codex) or epoch seconds/milliseconds (opencode) -> epoch seconds."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return x / 1000.0 if x > 1e11 else float(x)
    try:
        return dt.datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def usage_norm(raw, convention):
    """-> {'prompt': total prompt tokens, 'cache_read': n, 'cache_write': n} or None."""
    if not isinstance(raw, dict):
        return None
    if convention == "inclusive":
        total = int(raw.get("input_tokens") or 0)
        read, write = int(raw.get("cached_input_tokens") or 0), int(raw.get("cache_write_input_tokens") or 0)
    else:
        read = int(raw.get("cache_read_input_tokens") or 0)
        write = int(raw.get("cache_creation_input_tokens") or 0)
        total = int(raw.get("input_tokens") or 0) + read + write
    if total <= 0:
        return None
    return {"prompt": total, "cache_read": read, "cache_write": write}


def txt(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(txt(i.get("text") if isinstance(i, dict) and "text" in i
                             else (i if isinstance(i, str) else json.dumps(i, sort_keys=True))) for i in x)
    return json.dumps(x, sort_keys=True)


# ---- item iterators: yield (kind, text, generated, meta) | ("RESET", meta) | ("USAGE", usage) ----
# meta: {"ts": epoch seconds|None, "spawn": tool_use id (sub-agent launch)}. Consumers read it[:3] and
# it[3] if present, so 3-tuples from older callers keep working.
def claude_items(path):
    ids, seen_msg = {}, set()
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        t, ts = d.get("type"), ts_epoch(d.get("timestamp"))
        if t == "system" and d.get("subtype") == "compact_boundary":
            yield ("RESET", {"ts": ts})
            continue
        if t == "attachment":
            yield ("attach", json.dumps(d.get("attachment"), sort_keys=True, default=str), False, {"ts": ts})
            continue
        if t not in ("user", "assistant"):
            continue
        msg = d.get("message") or {}
        c = msg.get("content")
        gen = t == "assistant"
        role = "asst" if gen else "user"
        meta = {"ts": ts}
        if isinstance(c, str):
            yield (role, c, gen, meta)
        else:
            for b in c or []:
                bt = b.get("type")
                if bt == "text":
                    yield (role, b.get("text", ""), gen, meta)
                elif bt == "thinking":
                    yield ("asst", b.get("thinking", ""), True, meta)
                elif bt == "tool_use":
                    ids[b.get("id")] = b.get("name")
                    m = {"ts": ts, "spawn": b.get("id")} if b.get("name") in SPAWN_TOOLS else meta
                    yield ("asst", (b.get("name") or "") + json.dumps(b.get("input"), sort_keys=True), True, m)
                elif bt == "tool_result":
                    yield ("tool:" + str(ids.get(b.get("tool_use_id"), "?")), txt(b.get("content")), False, meta)
        if gen and msg.get("id") and msg["id"] not in seen_msg:   # content blocks of one API message repeat usage
            seen_msg.add(msg["id"])
            u = usage_norm(msg.get("usage"), "exclusive")
            if u:
                yield ("USAGE", u)


def claude_links(path):
    """{tool_use_id: {"agent_id", "async"}} from the parent's tool_result lines (`toolUseResult.agentId`)."""
    out = {}
    for line in open(path, errors="replace"):
        if '"agentId"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        tur = d.get("toolUseResult")
        c = (d.get("message") or {}).get("content")
        if not isinstance(tur, dict) or "agentId" not in tur or not isinstance(c, list):
            continue
        for b in c:
            if b.get("type") == "tool_result" and b.get("tool_use_id"):
                out[b["tool_use_id"]] = {"agent_id": tur["agentId"], "async": bool(tur.get("isAsync"))}
    return out


def codex_items(path):
    last_ord = None
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        ts = ts_epoch(d.get("timestamp"))
        p = d.get("payload") or {}
        if d.get("type") == "event_msg" and p.get("type") == "token_count":
            o = d.get("ordinal")
            u = usage_norm((p.get("info") or {}).get("last_token_usage"), "inclusive")
            if u and o != last_ord:                     # token_count repeats; one record per ordinal
                last_ord = o
                yield ("USAGE", u)
            continue
        if d.get("type") != "response_item":
            continue
        pt, meta = p.get("type"), {"ts": ts}
        if pt == "message":
            r = p.get("role")
            yield ({"user": "user", "assistant": "asst"}.get(r, "sys"), txt(p.get("content")), r == "assistant", meta)
        elif pt == "reasoning":
            yield ("asst", txt(p.get("summary")) + txt(p.get("content")), True, meta)
        elif pt in ("function_call", "custom_tool_call"):
            yield ("asst", str(p.get("name")) + txt(p.get("arguments") or p.get("input")), True, meta)
        elif pt in ("function_call_output", "custom_tool_call_output"):
            yield ("tool:exec", txt(p.get("output")), False, meta)


# ---- opencode: V1 (message + part) and V2 (session_message), detected from sqlite_master -----------
def opencode_layouts(con):
    """Which layouts the DB schema declares: subset of {'v1','v2'} (table presence in sqlite_master)."""
    names = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    out = set()
    if {"message", "part"} <= names:
        out.add("v1")
    if "session_message" in names:
        out.add("v2")
    return out


def opencode_has_parent_id(con):
    return any(r[1] == "parent_id" for r in con.execute("pragma table_info(session)"))


def opencode_session_layout(con, sid, layouts=None):
    """The ONE layout to read for this session: V2 if it has session_message rows, else V1 if it has
    message rows, else None. Never both (a migrated session keeps V1 rows beside its V2 rows)."""
    layouts = opencode_layouts(con) if layouts is None else layouts
    if "v2" in layouts and con.execute("select 1 from session_message where session_id=? limit 1", (sid,)).fetchone():
        return "v2"
    if "v1" in layouts and con.execute("select 1 from message where session_id=? limit 1", (sid,)).fetchone():
        return "v1"
    return None


def opencode_items(sid, con, layout=None):
    layout = layout or opencode_session_layout(con, sid)
    if layout == "v2":
        yield from _opencode_items_v2(sid, con)
    elif layout == "v1":
        yield from _opencode_items_v1(sid, con)


def _opencode_items_v1(sid, con):
    roles = {}
    for mid, data in con.execute("select id,data from message where session_id=? order by time_created,id", (sid,)):
        try:
            roles[mid] = json.loads(data).get("role")
        except Exception:
            pass
    for mid, tc, data in con.execute("select message_id,time_created,data from part where session_id=? order by time_created,id", (sid,)):
        try:
            d = json.loads(data)
        except Exception:
            continue
        t, role, ts = d.get("type"), roles.get(mid), ts_epoch(tc)
        if t == "text":
            yield (("user" if role == "user" else "asst"), d.get("text", ""), role != "user", {"ts": ts})
        elif t == "reasoning":
            yield ("asst", d.get("text", ""), True, {"ts": ts})
        elif t == "tool":
            st = d.get("state") or {}
            tm = st.get("time") or {}
            call_meta = {"ts": ts_epoch(tm.get("start")) or ts}
            if d.get("tool") == "task" and (st.get("metadata") or {}).get("sessionId"):
                call_meta["spawn"] = st["metadata"]["sessionId"]       # the child session id itself
            yield ("asst", str(d.get("tool")) + json.dumps(st.get("input") or {}, sort_keys=True), True, call_meta)
            if "output" in st:
                yield ("tool:" + str(d.get("tool")), txt(st.get("output")), False, {"ts": ts_epoch(tm.get("end")) or ts})
        elif t == "step-finish":
            u = usage_norm_opencode(d.get("tokens"))
            if u:
                yield ("USAGE", u)


def usage_norm_opencode(tokens):
    if not isinstance(tokens, dict):
        return None
    cache = tokens.get("cache") or {}
    read, write = int(cache.get("read") or 0), int(cache.get("write") or 0)
    total = int(tokens.get("input") or 0) + read + write
    return {"prompt": total, "cache_read": read, "cache_write": write} if total > 0 else None


def _opencode_items_v2(sid, con):
    """V2 `session_message.data` shapes (opencode session.next projector): user/system/synthetic
    {text}, compaction {summary}, assistant {content: [text|reasoning|tool], tokens}, shell."""
    for typ, tc, data in con.execute("select type,time_created,data from session_message where session_id=? order by seq", (sid,)):
        try:
            d = json.loads(data)
        except Exception:
            continue
        ts = ts_epoch(((d.get("time") or {}).get("created")) or tc)
        if typ in ("user", "system", "synthetic"):
            yield ("user", txt(d.get("text")), False, {"ts": ts})
        elif typ == "compaction":
            yield ("RESET", {"ts": ts})
        elif typ == "shell":
            yield ("tool:shell", txt(d.get("output") or d.get("command")), False, {"ts": ts})
        elif typ == "assistant":
            for c in d.get("content") or []:
                ct = c.get("type")
                if ct in ("text", "reasoning"):
                    yield ("asst", c.get("text", ""), True, {"ts": ts})
                elif ct == "tool":
                    stt = c.get("state") or {}
                    tm = c.get("time") or {}
                    yield ("asst", str(c.get("name")) + json.dumps(stt.get("input") or {}, sort_keys=True), True,
                           {"ts": ts_epoch(tm.get("ran") or tm.get("created")) or ts})
                    out = stt.get("content") if stt.get("content") else stt.get("result", stt.get("error"))
                    if out:
                        yield ("tool:" + str(c.get("name")), txt(out), False, {"ts": ts_epoch(tm.get("completed")) or ts})
            u = usage_norm_opencode(d.get("tokens"))
            if u:
                yield ("USAGE", u)


# ---- items -> turns -----------------------------------------------------------------------------
def render(kind, text):
    return f"<{kind}>\n{text}\n</{kind}>\n"


def items_to_turns(items):
    """Turn k = {pre, ref, reset, gap_s, ts_start, ts_end, spawns, usage}.
    Prompt_k = sum_{i<k}(pre_i + out_i) + pre_k (out_i = ref or live).
    A request is issued at the first generated item that follows non-generated pending items.
    gap_s = time between the previous turn's last generated item and this turn's first pending item
    (tool execution + think time; None when either timestamp is missing)."""
    turns, pre, ref = [], [], []
    st = {"state": "pre", "non_gen": False, "reset": False, "first_pre_ts": None, "last_gen_ts": None, "prev_end": None,
          "spawns": [], "usage": None}

    def emit():
        start = st["first_pre_ts"] if st["first_pre_ts"] is not None else st["last_gen_ts"]
        gap = None
        if start is not None and st["prev_end"] is not None:
            gap = max(0.0, start - st["prev_end"])
        turns.append({"pre": "".join(pre), "ref": "".join(ref), "reset": st["reset"], "gap_s": gap,
                      "ts_start": start, "ts_end": st["last_gen_ts"], "spawns": list(st["spawns"]),
                      "usage": st["usage"]})
        if st["last_gen_ts"] is not None:
            st["prev_end"] = st["last_gen_ts"]
        pre.clear()
        ref.clear()
        st.update(non_gen=False, reset=False, first_pre_ts=None, last_gen_ts=None, spawns=[], usage=None)

    def add_usage(u):
        # usage belongs to the reply being generated; one that arrives after the reply's tool output
        # (opencode step-finish) belongs to the last emitted turn, not to the next request
        tgt = st if st["state"] == "gen" else (turns[-1] if turns else None)
        if tgt is None:
            return
        cur = tgt["usage"] or {"prompt": 0, "cache_read": 0, "cache_write": 0}
        tgt["usage"] = {k: cur[k] + u[k] for k in cur}

    state = "pre"
    for it in items:
        st["state"] = state
        if it[0] == "RESET":
            if state == "gen" or st["non_gen"]:
                emit()
            state = "pre"
            st["reset"] = True
            continue
        if it[0] == "USAGE":
            add_usage(it[1])
            continue
        kind, text, gen = it[:3]
        meta = it[3] if len(it) > 3 else {}
        if not text:
            continue
        s = render(kind, text)
        if gen:
            if state == "pre" and st["non_gen"]:
                state = "gen"
            (ref if state == "gen" else pre).append(s)  # stray leading generated item: context only
            if meta.get("ts") is not None:
                st["last_gen_ts"] = meta["ts"]
            if meta.get("spawn"):
                st["spawns"].append(meta["spawn"])
        else:
            if state == "gen":
                emit()
                state = "pre"
            pre.append(s)
            st["non_gen"] = True
            if st["first_pre_ts"] is None and meta.get("ts") is not None:
                st["first_pre_ts"] = meta["ts"]
    if state == "gen" or st["non_gen"]:
        emit()
    return turns


def default_stores(mode):
    """Stores sampled when --stores is not given. Codex transcripts carry no parent links, so a families sample
    from them is always empty and the zero-session guard would abort the whole run (KPF-64.3 readiness)."""
    return tuple(x for x in STORES if x != "codex") if mode == "families" else STORES


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


def iter_prompts(turns, outputs=None, slot_holds_output=True):
    """Yield (k, prompt, prev_text); prev_text = previous prompt + its output (what the slot cache
    holds if it kept the generated tokens), None at a lineage start. outputs=None -> reference.
    slot_holds_output=False (reference/prefill-only replay, n_predict 1): the slot holds only the previous PROMPT
    (+1 token), so the attainable bound stops at the previous prompt; the replayed recorded output is new text."""
    hist, prev = "", None
    for k, t in enumerate(turns):
        if t.get("reset"):
            hist, prev = "", None
        prompt = hist + t["pre"]
        yield k, prompt, prev
        out = t["ref"] if outputs is None or k >= len(outputs) or outputs[k] is None else outputs[k]
        hist = prompt + out
        prev = hist if slot_holds_output else prompt


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


# ---- cross-session head bound -------------------------------------------------------------------
class HeadIndex:
    """Never-evicted shared prefix store across sessions, at HEAD_BLOCK-char granularity.

    `cross_chars(prompt, sid)` = longest block-aligned prefix of `prompt` that some OTHER session's
    earlier prompt also starts with (a LOWER bound within HEAD_BLOCK chars); `add(prompt, sid)` records
    the prompt. Owner sets are capped at two distinct sessions, enough to answer "someone other than
    sid has it". Thread-safe (the live driver adds from concurrent sessions)."""

    def __init__(self, block=HEAD_BLOCK):
        self.block, self.owners, self.lock = block, {}, threading.Lock()

    def _keys(self, prompt):
        h, b = hashlib.blake2b(digest_size=8), self.block
        for i in range(0, len(prompt) - b + 1, b):
            h.update(prompt[i:i + b].encode("utf-8", "replace"))
            yield i + b, h.digest()

    def cross_chars(self, prompt, sid):
        got = 0
        with self.lock:
            for end, key in self._keys(prompt):
                o = self.owners.get(key)
                if o is None or (o[0] == sid and o[1] is None) or (o[0] == sid and o[1] == sid):
                    break
                got = end
        return got

    def add(self, prompt, sid):
        with self.lock:
            for _end, key in self._keys(prompt):
                o = self.owners.get(key)
                if o is None:
                    self.owners[key] = [sid, None]
                elif o[0] != sid and o[1] is None:
                    o[1] = sid

    def match_and_add(self, prompt, sid):
        c = self.cross_chars(prompt, sid)
        self.add(prompt, sid)
        return c


# ---- discovery + sampling -----------------------------------------------------------------------
def _ro(db):
    """Read-only opencode snapshot connection (?mode=ro, query_only, retry once); live stores are refused."""
    return opencode_ro.open_ro(db)


def candidates(store, home, include_subagents=True, opencode_dbs=None):
    """Candidate sessions of `store`, sorted.

    `include_subagents=True` is the PREDECLARED pool (flat mode): Claude sub-agent transcripts and
    opencode child sessions are sessions in their own right -- and, measured 2026-10-22, they are the
    ONLY replayable Claude material: every one of the 79 live top-level Claude transcripts starts
    above 40k tokens and outgrows the 200k-char cap by its second request. `False` is the families
    pool: top-level parents only (children attach through `children_of`).
    Raises ZeroSessionsError when the store's schema is unknown."""
    if store == "claude":
        fs = glob.glob(home + "/.claude/projects/*/*.jsonl")
        if include_subagents:
            fs += glob.glob(home + "/.claude/projects/*/*/subagents/*.jsonl")
        return [("claude", f, None) for f in sorted(fs)]
    if store == "codex":
        return [("codex", f, None) for f in sorted(glob.glob(home + "/.codex/sessions/*/*/*/*.jsonl"))]
    out = []
    for _src, db in opencode_ro.resolve_dbs(opencode_dbs):      # default: both snapshots; live paths refused
        if not os.path.exists(db):
            continue
        con = _ro(db)
        try:
            if not opencode_layouts(con):
                raise ZeroSessionsError(f"{db}: sqlite_master declares neither opencode layout "
                                        "(V1 message+part, V2 session_message): unknown schema, refusing to guess")
            top_only = not include_subagents and opencode_has_parent_id(con)
            q = "select id from session where parent_id is null order by id" if top_only \
                else "select id from session order by id"
            out += [("opencode", r[0], db) for r in con.execute(q)]
        finally:
            con.close()
    return out


def load_turns(cand):
    store, ref, db = cand
    if store == "claude":
        return items_to_turns(claude_items(ref))
    if store == "codex":
        return items_to_turns(codex_items(ref))
    con = _ro(db)
    try:
        return items_to_turns(opencode_items(ref, con))
    finally:
        con.close()


def _first_ts(turns):
    return next((t["ts_start"] for t in turns if t.get("ts_start") is not None), None)


def _last_ts(turns):
    return next((t["ts_end"] for t in reversed(turns) if t.get("ts_end") is not None), None)


def _parent_turn_by_time(parent_turns, child_first_ts):
    """Parent request that spawned the child, by time (link=time): the latest turn whose reply had ENDED
    by the child's first event (a spawn happens inside a reply, so ts_end <= child start); a child that
    starts inside the first reply falls back to the latest turn that had started."""
    if child_first_ts is None:
        return None
    best = None
    for k, t in enumerate(parent_turns):
        if t.get("ts_end") is not None and t["ts_end"] <= child_first_ts:
            best = k
    if best is None:
        for k, t in enumerate(parent_turns):
            if t.get("ts_start") is not None and t["ts_start"] <= child_first_ts:
                best = k
    return best


def children_of(cand, parent_turns):
    """Child sessions of one parent as [{'cand', 'turns', 'parent_turn', 'async', 'link'}].

    Link order: explicit (Claude `toolUseResult.agentId`; opencode `task` part `metadata.sessionId`),
    else by time. A child whose parent turn cannot be placed keeps parent_turn=None ('unlinked'): it
    counts in the fan-out stats but is never replayed in a burst."""
    store, ref, db = cand
    kids = []
    if store == "claude":
        links = claude_links(ref)
        by_agent = {v["agent_id"]: (tid, v) for tid, v in links.items()}
        spawn_turn = {tid: k for k, t in enumerate(parent_turns) for tid in t["spawns"]}
        for f in sorted(glob.glob(os.path.splitext(ref)[0] + "/subagents/agent-*.jsonl")):
            aid = os.path.basename(f)[len("agent-"):-len(".jsonl")]
            turns = items_to_turns(claude_items(f))
            if not turns:
                continue
            tid, v = by_agent.get(aid, (None, None))
            k, link = (spawn_turn.get(tid), "explicit") if tid in spawn_turn else (None, None)
            if k is None:
                k, link = _parent_turn_by_time(parent_turns, _first_ts(turns)), ("time" if parent_turns else "none")
            kids.append({"cand": ("claude", f, None), "turns": turns, "parent_turn": k,
                         "async": bool(v and v["async"]), "link": link or "none"})
    elif store == "opencode":
        con = _ro(db)
        try:
            spawn_turn = {tid: k for k, t in enumerate(parent_turns) for tid in t["spawns"]}
            child_ids = con.execute("select id from session where parent_id=? order by time_created,id", (ref,)) \
                if opencode_has_parent_id(con) else []
            for (cid,) in child_ids:
                turns = items_to_turns(opencode_items(cid, con))
                if not turns:
                    continue
                if cid in spawn_turn:
                    k, link = spawn_turn[cid], "explicit"
                else:
                    k, link = _parent_turn_by_time(parent_turns, _first_ts(turns)), "time"
                kids.append({"cand": ("opencode", cid, db), "turns": turns, "parent_turn": k, "async": False,
                             "link": link})
        finally:
            con.close()
    return kids


def link_child(cand):
    """For a flat-mode candidate that IS a sub-agent transcript: {'parent_hash', 'parent_turn', 'async',
    'link'} (parent_turn None when it cannot be placed); None for a top-level session."""
    store, ref, db = cand
    if store == "claude" and "/subagents/agent-" in ref:
        parent = os.path.dirname(os.path.dirname(ref)) + ".jsonl"
        aid = os.path.basename(ref)[len("agent-"):-len(".jsonl")]
        info = {"parent_hash": hashlib.blake2b(parent.encode(), digest_size=6).hexdigest(), "parent_turn": None,
                "async": False, "link": "parent_missing"}
        if os.path.exists(parent):
            pt = items_to_turns(claude_items(parent))
            tid, v = next(((t, x) for t, x in claude_links(parent).items() if x["agent_id"] == aid), (None, None))
            spawn_turn = {t: k for k, tt in enumerate(pt) for t in tt["spawns"]}
            if tid in spawn_turn:
                info.update(parent_turn=spawn_turn[tid], link="explicit", **{"async": bool(v["async"])})
            else:
                kid = items_to_turns(claude_items(ref))
                info.update(parent_turn=_parent_turn_by_time(pt, _first_ts(kid)), link="time")
        return info
    if store == "opencode":
        con = _ro(db)
        try:
            if not opencode_has_parent_id(con):
                return None
            row = con.execute("select parent_id from session where id=?", (ref,)).fetchone()
            if not row or not row[0]:
                return None
            pid = row[0]
            pt = items_to_turns(opencode_items(pid, con))
            spawn_turn = {t: k for k, tt in enumerate(pt) for t in tt["spawns"]}
            if ref in spawn_turn:
                return {"parent_hash": hashlib.blake2b(pid.encode(), digest_size=6).hexdigest(),
                        "parent_turn": spawn_turn[ref], "async": False, "link": "explicit"}
            kid = items_to_turns(opencode_items(ref, con))
            return {"parent_hash": hashlib.blake2b(pid.encode(), digest_size=6).hexdigest(),
                    "parent_turn": _parent_turn_by_time(pt, _first_ts(kid)), "async": False, "link": "time"}
        finally:
            con.close()
    return None


def child_count(cand):
    """Cheap upper bound on a top-level candidate's children (directory listing / one indexed query),
    used to skip parents that cannot qualify before any transcript is parsed."""
    store, ref, db = cand
    if store == "claude":
        return len(glob.glob(os.path.splitext(ref)[0] + "/subagents/agent-*.jsonl"))
    if store == "opencode":
        con = _ro(db)
        try:
            if not opencode_has_parent_id(con):
                return 0
            return con.execute("select count(*) from session where parent_id=?", (ref,)).fetchone()[0]
        finally:
            con.close()
    return 0


def quotas(n, stores):
    base, extra = divmod(n, len(stores))
    return {s: base + (1 if i < extra else 0) for i, s in enumerate(stores)}


def sample_sessions(home, n=20, seed=64, stores=STORES, min_requests=10,
                    max_prompt_chars=200_000, max_requests=200, allow_empty_store=False,
                    mode="flat", min_children=2, min_child_requests=3, max_children=8, opencode_dbs=None):
    """Deterministic. Per store: candidates sorted, shuffled with Random(f'{seed}:{store}'); the first
    `quota` sessions passing the filters (after truncation) are taken. A store that cannot fill its
    quota hands the shortfall to the others (round-robin, fixed order).

    mode='flat' (PREDECLARED arms A-D): the candidate pool, order and filters are unchanged, so the
    same 20 sessions are drawn as before; a sampled sub-agent transcript additionally records its link
    to its parent (`link_child`). mode='families' (burst arms): candidates are top-level parents with
    >= `min_children` linked children of >= `min_child_requests` requests; a parent that fails the
    replay filter (every live Claude parent does) is kept as a NON-replayable anchor whose children
    still replay as bursts.

    Fails loudly (ZeroSessionsError) when a requested store has no candidate at all -- unless
    `allow_empty_store` -- or when nothing at all was picked: a silently shrunken sample is how a
    parser that stopped matching a new on-disk layout passes unnoticed. Each picked entry is
    (store, cand, turns, stats, extra); extra = {'children': [...], 'link': {...}|None,
    'replayable': bool}."""
    if mode not in ("flat", "families"):
        raise ValueError(f"mode must be flat|families, got {mode!r}")
    q = quotas(n, stores)
    pools = {}
    for s in stores:
        c = candidates(s, home, include_subagents=(mode == "flat"), opencode_dbs=opencode_dbs)
        if mode == "families" and c:
            c = [x for x in c if child_count(x) >= min_children]
        if not c and not allow_empty_store:
            raise ZeroSessionsError(f"store {s!r} has no sessions under {home} (pass allow_empty_store / "
                                    "--allow-empty-store only if the study really excludes it)")
        random.Random(f"{seed}:{s}").shuffle(c)
        pools[s] = iter(c)
    picked, got = [], {s: 0 for s in stores}

    def take(s):
        for cand in pools[s]:
            try:
                full = load_turns(cand)
                turns = truncate(full, max_prompt_chars, max_requests)
                if mode == "flat":
                    if len(turns) >= min_requests:
                        picked.append((s, cand, turns, session_stats(turns),
                                       {"children": [], "link": link_child(cand), "replayable": True}))
                        got[s] += 1
                        return True
                    continue
                kids = []
                for k in children_of(cand, full):
                    kt = truncate(k["turns"], max_prompt_chars, max_requests)
                    if len(kt) >= min_child_requests and k["parent_turn"] is not None:
                        kids.append({**k, "turns": kt})
                kids = sorted(kids, key=lambda k: k["parent_turn"])[:max_children]
                if len(kids) >= min_children:
                    ok = len(turns) >= min_requests
                    picked.append((s, cand, turns if ok else [], session_stats(turns if ok else []),
                                   {"children": kids, "link": None, "replayable": ok}))
                    got[s] += 1
                    return True
            except ZeroSessionsError:
                raise
            except Exception:
                continue
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
    if not picked:
        raise ZeroSessionsError(f"no session passed the filters (mode={mode}, min_requests={min_requests}, "
                                f"max_prompt_chars={max_prompt_chars}) in stores {list(stores)}")
    return picked


# ---- provider-realised cache share, bounds, fan-out stats ------------------------------------------
def provider_share(turns):
    """Provider-realised prefix-cache share of recorded usage: sum cache_read / sum prompt tokens over
    turns that carry usage (convention-normalised by usage_norm). None when no turn has usage."""
    p = c = n = 0
    for t in turns:
        u = t.get("usage")
        if u:
            p, c, n = p + u["prompt"], c + u["cache_read"], n + 1
    return None if not p else {"prompt_tokens": p, "cache_read_tokens": c, "turns_with_usage": n,
                               "share": c / p}


def fanout_stats(records):
    """Fan-out statistics of a set of sessions.

    records: [{'id', 'parent' (id|None), 'start', 'end' (epoch s|None), 'requests' (int)}].
      child_request_share  = child requests / all requests
      children_per_parent  = mean and max over parents that have children (and the count of parents)
      peak_children_per_parent = max children whose [start,end] spans overlap under ONE parent
      peak_active_lineages = max sessions (parents and children) whose spans overlap anywhere
    Spans run first event -> last event, so they include idle gaps: peaks are UPPER bounds on
    in-flight requests (the driver measures the realised peak, see run_meta)."""
    reqs = sum(r["requests"] for r in records)
    kid = [r for r in records if r["parent"]]
    by_parent = {}
    for r in kid:
        by_parent.setdefault(r["parent"], []).append(r)

    def peak(rs):
        ev = []
        for r in rs:
            if r["start"] is not None and r["end"] is not None:
                ev += [(r["start"], 1), (r["end"], -1)]
        cur = best = 0
        for _, d in sorted(ev, key=lambda e: (e[0], e[1])):   # ends before starts at the same instant
            cur += d
            best = max(best, cur)
        return best

    return {"sessions": len(records), "parents_with_children": len(by_parent), "child_sessions": len(kid),
            "requests": reqs, "child_requests": sum(r["requests"] for r in kid),
            "child_request_share": (sum(r["requests"] for r in kid) / reqs) if reqs else 0.0,
            "children_per_parent_mean": (len(kid) / len(by_parent)) if by_parent else 0.0,
            "children_per_parent_max": max((len(v) for v in by_parent.values()), default=0),
            "peak_children_per_parent": max((peak(v) for v in by_parent.values()), default=0),
            "peak_active_lineages": peak(records),
            "spans_upper_bound": True}


def store_fanout(home, stores=STORES, max_parents=None, opencode_dbs=None):
    """Store-wide fan-out stats from timestamps only (no prompt text kept). claude: every
    `<proj>/<parent>/subagents/agent-*.jsonl` against its parent file; opencode: session.parent_id."""
    recs = []

    def span_claude(path):
        first = last = None
        n = 0
        for line in open(path, errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            ts = ts_epoch(d.get("timestamp"))
            if ts is not None:
                first = ts if first is None else min(first, ts)
                last = ts if last is None else max(last, ts)
            if d.get("type") == "assistant" and (d.get("message") or {}).get("id"):
                n += 1
        return first, last, n

    if "claude" in stores:
        parents = sorted(glob.glob(home + "/.claude/projects/*/*.jsonl"))[:max_parents]
        for f in parents:
            pid = os.path.basename(f)
            a, b, n = span_claude(f)
            recs.append({"id": "c:" + pid, "parent": None, "start": a, "end": b, "requests": n})
            for k in sorted(glob.glob(os.path.splitext(f)[0] + "/subagents/agent-*.jsonl")):
                a, b, n = span_claude(k)
                recs.append({"id": "c:" + os.path.basename(k), "parent": "c:" + pid, "start": a, "end": b, "requests": n})
    for _src, db in (opencode_ro.resolve_dbs(opencode_dbs) if "opencode" in stores else []):
        if not os.path.exists(db):
            continue
        con = _ro(db)
        try:
            lay = opencode_layouts(con)
            counts = {}
            if "v1" in lay:
                counts.update(con.execute("select session_id, count(*) from message "
                                          "where json_extract(data,'$.role')='assistant' group by session_id"))
            if "v2" in lay:   # a session lives in one layout: V2 rows win, as in opencode_session_layout
                counts.update(con.execute("select session_id, count(*) from session_message "
                                          "where type='assistant' group by session_id"))
            cols = "id,parent_id,time_created,time_updated" if opencode_has_parent_id(con) \
                else "id,null,time_created,time_created"
            for sid, par, tc, tu in con.execute(f"select {cols} from session"):
                recs.append({"id": "o:" + sid, "parent": ("o:" + par) if par else None, "start": ts_epoch(tc),
                             "end": ts_epoch(tu), "requests": counts.get(sid, 0)})
        finally:
            con.close()
    return fanout_stats(recs)


def totals(sessions):
    pc = sum(s["prompt_chars"] for s in sessions)
    ic = sum(s["ideal_chars"] for s in sessions)
    uc = sum(s["ideal_union_chars"] for s in sessions)
    by = {}
    for st in STORES:
        sub = [s for s in sessions if s["store"] == st]
        if sub:
            p, i = sum(s["prompt_chars"] for s in sub), sum(s["ideal_chars"] for s in sub)
            u = sum(s["ideal_union_chars"] for s in sub)
            pv = [s["provider"] for s in sub if s.get("provider")]
            pt = sum(x["prompt_tokens"] for x in pv)
            by[st] = {"sessions": len(sub), "requests": sum(s["requests"] for s in sub),
                      "prompt_tok_est": p // CHARS_PER_TOKEN, "ideal_share": i / p if p else 0.0,
                      "ideal_share_cross_head": u / p if p else 0.0,
                      "provider_cache_share": (sum(x["cache_read_tokens"] for x in pv) / pt) if pt else None,
                      "usage_convention": USAGE_CONVENTION[st]}
    pv = [s["provider"] for s in sessions if s.get("provider")]
    pt = sum(x["prompt_tokens"] for x in pv)
    return {"sessions": len(sessions), "requests": sum(s["requests"] for s in sessions),
            "prompt_tok_est": pc // CHARS_PER_TOKEN, "ideal_tok_est": ic // CHARS_PER_TOKEN,
            "ideal_share": ic / pc if pc else 0.0, "ideal_share_cross_head": uc / pc if pc else 0.0,
            "provider_cache_share": (sum(x["cache_read_tokens"] for x in pv) / pt) if pt else None,
            "by_store": by}


def _flat_sessions(picked):
    """Manifest order: each parent then its children. Yields (id, store, cand, turns, meta)."""
    idx = {s: 0 for s in STORES}
    for s, cand, turns, _st, extra in picked:
        sid = f"{s}-{idx[s]:02d}"
        idx[s] += 1
        yield sid, s, cand, turns, {"parent": None, "children": extra["children"], "link": extra.get("link"),
                                    "replayable": extra.get("replayable", True)}
        for j, kid in enumerate(extra["children"]):
            yield f"{sid}.c{j:02d}", s, kid["cand"], kid["turns"], {
                "parent": sid, "parent_turn": kid["parent_turn"], "async": kid["async"], "link": kid["link"]}


def build_manifest(picked, params):
    """Per-session stats incl. the two ideal bounds. The cross-session-head bound walks sessions in
    manifest order (parents then children) and lets each request match any OTHER session's earlier
    reference prompt: an idealised never-evicted shared cache, not a replay-order claim."""
    head, sessions = HeadIndex(), []
    for sid, s, cand, turns, meta in _flat_sessions(picked):
        prof = ideal_profile(turns)
        union = cross = 0
        for (k, prompt, _prev), (plen, within) in zip(iter_prompts(turns), prof):
            c = min(head.match_and_add(prompt, sid), plen)
            cross += c
            union += max(within, c)
        pc = sum(p for p, _ in prof)
        gaps = [t["gap_s"] for t in turns if t.get("gap_s") is not None]
        row = {"id": sid, "store": s, "source": opencode_ro.source_for(cand[2]) if s == "opencode" else None,
               "source_hash": hashlib.blake2b(str(cand[1]).encode(), digest_size=6).hexdigest(),
               **session_stats(turns), "ideal_cross_only_chars": cross, "ideal_union_chars": union,
               "ideal_share_cross_head": (union / pc) if pc else 0.0, "provider": provider_share(turns),
               "parent": meta["parent"], "gap_s_median": sorted(gaps)[len(gaps) // 2] if gaps else None,
               "gap_s_p90": sorted(gaps)[int(0.9 * (len(gaps) - 1))] if gaps else None,
               "start": _first_ts(turns), "end": _last_ts(turns)}
        if meta["parent"]:
            row.update(parent_turn=meta["parent_turn"], is_async=meta["async"], link=meta["link"])
        else:
            row["children"] = len(meta["children"])
            row["replayable"] = meta["replayable"]
            if meta.get("link"):   # flat mode: this top-level session IS a sub-agent transcript
                row.update(child_of=meta["link"]["parent_hash"], parent_turn=meta["link"]["parent_turn"],
                           is_async=meta["link"]["async"], link=meta["link"]["link"])
        sessions.append(row)
    top = [r for r in sessions if not r["parent"]]
    fan = fanout_stats([{"id": r["id"], "parent": r["parent"], "start": r["start"], "end": r["end"],
                         "requests": r["requests"]} for r in sessions])
    links = {}
    for r in sessions:
        if r.get("link"):
            links[r["link"]] = links.get(r["link"], 0) + 1
    return {"schema": "kpf64.sessions.v2", "params": params, "sessions": sessions,
            "totals": totals(top), "totals_incl_children": totals(sessions), "fanout": fan,
            "child_links": links}


def write_sessions(picked, out_dir, manifest):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {m["id"]: m for m in manifest["sessions"]}
    for sid, s, _cand, turns, info in _flat_sessions(picked):
        rec = {"id": sid, "store": s, "turns": turns, "replayable": info.get("replayable", True)}
        if info["parent"]:
            m = meta[sid]
            rec.update(parent=info["parent"], parent_turn=info["parent_turn"], is_async=info["async"], link=info["link"])
        else:
            rec["children"] = [{"id": f"{sid}.c{j:02d}", "parent_turn": k["parent_turn"], "is_async": k["async"],
                                "link": k["link"]} for j, k in enumerate(info["children"])]
        (out / f"{sid}.json").write_text(json.dumps(rec))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="dir for per-session request files (contain transcript text)")
    ap.add_argument("--home", default=os.environ.get("HOME", "/home/node"),
                    help="where the claude/codex stores live (opencode is NOT read from here)")
    ap.add_argument("--opencode-db", action="append", dest="opencode_db",
                    help="opencode READ-ONLY snapshot (repeatable). Default: both "
                         + ", ".join(OPENCODE_RO_DBS.values()) + ". Live stores are refused.")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--stores", default=None,
                    help="default: all stores in flat mode; families mode drops codex (no parent links, always childless)")
    ap.add_argument("--min-requests", type=int, default=10)
    ap.add_argument("--max-prompt-chars", type=int, default=200_000, help="~50k tokens; must fit one slot's ctx")
    ap.add_argument("--max-requests", type=int, default=200)
    ap.add_argument("--history", choices=("reference", "live"), default="reference",
                    help="recorded in the manifest; the driver regenerates outputs when 'live'")
    ap.add_argument("--allow-empty-store", action="store_true",
                    help="do not fail when a requested store has no sessions (the shortfall is redistributed)")
    ap.add_argument("--mode", choices=("flat", "families"), default="flat",
                    help="flat = the predeclared pool (arms A-D, same sample as before); families = parents with "
                         "linked children for the burst arms")
    ap.add_argument("--min-children", type=int, default=2)
    ap.add_argument("--min-child-requests", type=int, default=3)
    ap.add_argument("--max-children", type=int, default=8, help="families mode: cap children per parent (earliest spawns)")
    ap.add_argument("--fanout-store", action="store_true",
                    help="print STORE-WIDE fan-out stats (timestamps only, no text) and exit; no --out needed")
    a = ap.parse_args(argv)
    stores = default_stores(a.mode) if a.stores is None else tuple(a.stores.split(","))
    a.stores = ",".join(stores)
    if a.fanout_store:
        print(json.dumps(store_fanout(a.home, stores, opencode_dbs=a.opencode_db), indent=1))
        return 0
    if not a.out:
        ap.error("--out is required unless --fanout-store")
    try:
        picked = sample_sessions(a.home, a.n, a.seed, stores, a.min_requests, a.max_prompt_chars,
                                 a.max_requests, a.allow_empty_store, a.mode, a.min_children,
                                 a.min_child_requests, a.max_children, a.opencode_db)
    except ZeroSessionsError as e:
        sys.stderr.write(f"adapter: FAILED (zero sessions): {e}\n")
        return 4
    man = build_manifest(picked, {k: v for k, v in vars(a).items() if k not in ("out", "home", "fanout_store")})
    write_sessions(picked, a.out, man)
    t = man["totals"]
    print(json.dumps({"sessions": t["sessions"], "requests": t["requests"], "prompt_tok_est": t["prompt_tok_est"],
                      "ideal_share": round(t["ideal_share"], 5),
                      "ideal_share_cross_head": round(t["ideal_share_cross_head"], 5),
                      "provider_cache_share": None if t["provider_cache_share"] is None else round(t["provider_cache_share"], 5),
                      "child_sessions": man["fanout"]["child_sessions"],
                      "child_request_share": round(man["fanout"]["child_request_share"], 5),
                      "peak_children_per_parent": man["fanout"]["peak_children_per_parent"],
                      "by_store": {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                                   for k, v in t["by_store"].items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
