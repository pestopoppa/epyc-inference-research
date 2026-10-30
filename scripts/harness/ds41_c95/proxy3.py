#!/usr/bin/env python3
"""DS41-C95 wire recorder, run 3 (proxy3.py). A copy of proxy2.py (which keeps serving the live F12
run untouched); every proxy2 behaviour is unchanged when no run-3 flag is given. It adds:

UFH14-A3 support
  --alias REAL=CANON   Bidirectional path aliasing. Every string in a request body has REAL
                       replaced by CANON before it goes upstream; every string in the reply
                       (streamed SSE deltas with a hold-back buffer, *.done events, plain JSON)
                       has CANON replaced by REAL before it goes back to the harness. The model
                       only ever sees CANON, the harness and its tools only ever see REAL. With
                       REAL = the per-call workspace root and CANON = one fixed string, concurrent
                       calls on the same context send BYTE-IDENTICAL prompts (verified on the F12
                       C2/cxf1 r1 vs r2 first requests: instructions, tools and every input item
                       equal after aliasing), so llama-server's prompt cache can serve them.
  --hold-first         Hold the first tool-bearing model POST: write its final upstream body to
                       <log>/first-body.json, then wait for <log>/RELEASE (run3's group
                       coordinator warms the server with it first). Times out after
                       --hold-timeout-s and forwards anyway. Logged as held_s / hold_released_by.

UFH14-A4 (D2) prefix-preserving compaction
  --keep-prefix-compaction
                       A compaction request is detected (codex: client_metadata
                       x-codex-turn-metadata request_kind == "compaction"; any harness: a known
                       summary marker in the last user message, or a tool-less request that
                       replays the session's last tool-bearing prefix). When it carries the
                       session's FULL history but tools=[] (codex), the session's last tools
                       array and parallel_tool_calls are re-injected with tool_choice "none".
                       Why this is safe (llama.cpp @ ffc1bac82): the chat template receives
                       `tools` whenever the list is non-empty, independent of tool_choice
                       (common/chat.cpp common_chat_template_direct_apply_impl), so the rendered
                       prefix is byte-identical to the previous turn; tool_choice "none" turns off
                       the tool grammar and tool-call parsing (server-common.cpp:1065), so the
                       reply is the plain-text summary codex expects. A separate-summarizer
                       compaction (opencode, Hermes: own system prompt + serialized transcript) is
                       NOT rewritten: re-injecting tools cannot restore a prefix it never shared.
                       Every compaction row records detected_by, prefix_shared, rewritten, and the
                       reply's cache_n / prompt_n (= the re-prefill tokens), plus how often the
                       summary text contains tool-call markup.

D3 task-prompt pinning
  --pin-task           The session's task message (the LAST user message of its first
                       tool-bearing request) is remembered. A later tool-bearing request that
                       lost it is repaired: chat API (opencode, Hermes) -> the message is
                       re-inserted right after the system message(s) if no user message contains
                       it; responses API (codex) -> an input item with the same id whose text was
                       middle-truncated ("...N tokens truncated...") gets its full text back, or
                       the item is re-inserted at its original index. Logged as task_pin.

Usage: proxy3.py --port 0 --log DIR [--upstream URL] [--effort-policy strip|passthrough]
                 [--think-budget N --think-message MSG] [--alias REAL=CANON ...]
                 [--hold-first [--hold-timeout-s S]] [--keep-prefix-compaction] [--pin-task]
Prints `PORT <n>` on stdout once listening. Stdlib only. The request transforms are plain
functions (Session.transform) so run3.py's dry-run replays recorded bodies through them offline.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import http.client
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
}

COMPACTION_MARKERS = (
    ("codex", "You are performing a CONTEXT CHECKPOINT COMPACTION"),
    ("hermes", "You are a summarization agent creating a context checkpoint"),
    ("opencode", "Here is the conversation so far:"),
    ("opencode", "Provide a detailed prompt for continuing our conversation"),
)


def is_model_path(path: str) -> bool:
    return str(path).split("?")[0].rstrip("/").endswith(("chat/completions", "responses"))


# ---------------------------------------------------------------------------------------------
# proxy2 helpers (unchanged)
# ---------------------------------------------------------------------------------------------
def summarize_request(body: dict) -> dict:
    msgs = body.get("messages")
    inp = body.get("input")
    tools = body.get("tools") or []
    names = []
    for t in tools:
        if isinstance(t, dict):
            names.append(t.get("name") or (t.get("function") or {}).get("name") or t.get("type"))
    ctk = body.get("chat_template_kwargs") or {}
    return {
        "model": body.get("model"),
        "n_messages": len(msgs) if isinstance(msgs, list) else None,
        "n_input_items": len(inp)
        if isinstance(inp, list)
        else (1 if isinstance(inp, str) else None),
        "roles": [m.get("role") for m in msgs][-3:] if isinstance(msgs, list) else None,
        "n_tools": len(tools),
        "tool_names": names[:40],
        "max_tokens": body.get(
            "max_tokens", body.get("max_output_tokens", body.get("max_completion_tokens"))
        ),
        "stream": bool(body.get("stream")),
        "temperature": body.get("temperature"),
        "sent_reasoning": {
            k: v
            for k, v in (
                ("reasoning_effort", body.get("reasoning_effort")),
                ("reasoning", body.get("reasoning")),
                ("chat_template_kwargs", ctk or None),
            )
            if v is not None
        },
        "keys": sorted(body.keys()),
    }


def apply_policy(body: dict, policy: str) -> list[str]:
    if policy != "strip":
        return []
    removed = []
    for k in ("reasoning_effort", "reasoning"):
        if k in body:
            body.pop(k)
            removed.append(k)
    ctk = body.get("chat_template_kwargs")
    if isinstance(ctk, dict) and "reasoning_effort" in ctk:
        ctk.pop("reasoning_effort")
        removed.append("chat_template_kwargs.reasoning_effort")
        if not ctk:
            body.pop("chat_template_kwargs")
    return removed


def apply_think_budget(body: dict, path: str, budget: int, message: str) -> dict:
    if budget <= 0 or not is_model_path(path):
        return {}
    inj = {}
    if "thinking_budget_tokens" not in body and "reasoning_budget_tokens" not in body:
        body["thinking_budget_tokens"] = budget
        inj["thinking_budget_tokens"] = budget
    if message and "reasoning_budget_message" not in body:
        body["reasoning_budget_message"] = message
        inj["reasoning_budget_message"] = True
    return inj


def usage_from(obj: dict, acc: dict) -> None:
    if not isinstance(obj, dict):
        return
    resp = obj.get("response") if isinstance(obj.get("response"), dict) else None
    for src in (obj, resp or {}):
        if isinstance(src.get("usage"), dict):
            acc["usage"] = src["usage"]
        if isinstance(src.get("timings"), dict):
            acc["timings"] = src["timings"]
    for ch in obj.get("choices") or []:
        if isinstance(ch, dict) and ch.get("finish_reason"):
            acc["finish_reason"] = ch["finish_reason"]
        d = (ch or {}).get("delta") or (ch or {}).get("message") or {}
        if isinstance(d, dict):
            if d.get("tool_calls"):
                acc["tool_call_chunks"] = acc.get("tool_call_chunks", 0) + 1
            if d.get("content"):
                acc["content_chars"] = acc.get("content_chars", 0) + len(d["content"])
                if "<tool_call" in d["content"]:
                    acc["content_tool_markup"] = acc.get("content_tool_markup", 0) + 1
            rc = d.get("reasoning_content")
            if rc:
                acc["reasoning_chars"] = acc.get("reasoning_chars", 0) + len(rc)
    t = obj.get("type")
    if isinstance(t, str):
        if t == "response.completed" and resp:
            acc["finish_reason"] = resp.get("status")
            out = resp.get("output") or []
            acc["output_types"] = [o.get("type") for o in out if isinstance(o, dict)]
        elif t == "response.output_item.done":
            it = obj.get("item") or {}
            if it.get("type") in ("function_call", "custom_tool_call", "local_shell_call"):
                acc["tool_calls"] = acc.get("tool_calls", 0) + 1
        elif t == "response.output_text.delta" and "<tool_call" in str(obj.get("delta") or ""):
            acc["content_tool_markup"] = acc.get("content_tool_markup", 0) + 1


# ---------------------------------------------------------------------------------------------
# run 3: aliasing
# ---------------------------------------------------------------------------------------------
def replace_all(obj, pairs: list[tuple[str, str]], counter: list[int] | None = None):
    """Recursively replace substrings in every string (keys untouched)."""
    if isinstance(obj, str):
        s = obj
        for a, b in pairs:
            if a in s:
                if counter is not None:
                    counter[0] += s.count(a)
                s = s.replace(a, b)
        return s
    if isinstance(obj, list):
        return [replace_all(x, pairs, counter) for x in obj]
    if isinstance(obj, dict):
        return {k: replace_all(v, pairs, counter) for k, v in obj.items()}
    return obj


class StreamRewriter:
    """CANON->REAL on streamed text fragments. A fragment's tail that could be the start of a
    CANON string is held back until the next fragment (or the field's end) decides it."""

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs
        self.held: dict = {}
        self.count = 0

    def feed(self, key, text: str) -> str:
        s = self.held.pop(key, "") + text
        for c, r in self.pairs:
            if c in s:
                self.count += s.count(c)
                s = s.replace(c, r)
        hold = 0
        for c, _ in self.pairs:
            for n in range(min(len(c) - 1, len(s)), 0, -1):
                if s.endswith(c[:n]):
                    hold = max(hold, n)
                    break
        if hold:
            self.held[key] = s[-hold:]
            s = s[:-hold]
        return s

    def flush(self, pred=lambda k: True) -> dict:
        out = {k: v for k, v in self.held.items() if pred(k) and v}
        for k in list(self.held):
            if pred(k):
                self.held.pop(k)
        return out

    def full(self, obj):
        c = [0]
        o = replace_all(obj, self.pairs, c)
        self.count += c[0]
        return o


def rewrite_chat_event(obj: dict, rw: StreamRewriter) -> list[dict]:
    """One chat.completion.chunk -> the chunk(s) to forward (a synthetic flush chunk may precede
    a chunk that finishes a choice)."""
    pre: list[dict] = []
    for ch in obj.get("choices") or []:
        if not isinstance(ch, dict):
            continue
        idx = ch.get("index", 0)
        d = ch.get("delta")
        if isinstance(d, dict):
            for f in ("content", "reasoning_content"):
                if isinstance(d.get(f), str):
                    d[f] = rw.feed(("c", idx, f), d[f])
            for tc in d.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                    fn["arguments"] = rw.feed(("t", idx, tc.get("index", 0)), fn["arguments"])
        if isinstance(ch.get("message"), dict):
            ch["message"] = rw.full(ch["message"])
        if ch.get("finish_reason"):
            held = rw.flush(lambda k, i=idx: k[1] == i)
            if held:
                delta: dict = {}
                for k, v in held.items():
                    if k[0] == "c":
                        delta[k[2]] = v
                    else:
                        delta.setdefault("tool_calls", []).append(
                            {"index": k[2], "function": {"arguments": v}}
                        )
                pre.append(
                    {k: obj[k] for k in ("id", "object", "created", "model") if k in obj}
                    | {"choices": [{"index": idx, "delta": delta, "finish_reason": None}]}
                )
    return pre + [obj]


_RESP_DELTA_DONE = {
    "response.output_text.delta": ("response.output_text.done", "response.content_part.done"),
    "response.reasoning_text.delta": ("response.reasoning_text.done", "response.content_part.done"),
    "response.function_call_arguments.delta": ("response.function_call_arguments.done",),
}


def rewrite_responses_event(
    evname: str | None, obj: dict, rw: StreamRewriter, last_delta: dict
) -> list[tuple[str | None, dict]]:
    t = obj.get("type") or evname or ""
    if t in _RESP_DELTA_DONE and isinstance(obj.get("delta"), str):
        key = (t, obj.get("item_id"), obj.get("output_index"), obj.get("content_index"))
        obj["delta"] = rw.feed(key, obj["delta"])
        last_delta[key] = dict(obj)
        return [(evname, obj)]
    out: list[tuple[str | None, dict]] = []
    if t.endswith(".done") or t == "response.completed":
        iid, oidx = obj.get("item_id") or (obj.get("item") or {}).get("id"), obj.get("output_index")

        def mine(k) -> bool:
            if t == "response.completed":
                return True
            return (iid is not None and k[1] == iid) or (oidx is not None and k[2] == oidx)

        for k, v in rw.flush(mine).items():
            syn = dict(
                last_delta.get(k)
                or {"type": k[0], "item_id": k[1], "output_index": k[2], "content_index": k[3]}
            )
            syn["delta"] = v
            out.append((k[0], syn))
    out.append((evname, rw.full(obj)))
    return out


# ---------------------------------------------------------------------------------------------
# run 3: per-session state, compaction detection/rewrite, task pinning
# ---------------------------------------------------------------------------------------------
def _strip_ids(x):
    if isinstance(x, dict):
        return {k: _strip_ids(v) for k, v in x.items() if k != "id"}
    if isinstance(x, list):
        return [_strip_ids(v) for v in x]
    return x


def _text_of(item) -> str:
    """Plain text of a chat message or a responses input item."""
    if not isinstance(item, dict):
        return ""
    c = item.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(
            p.get("text", "") for p in c if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


def _turns(body: dict) -> tuple[str, list]:
    if isinstance(body.get("messages"), list):
        return "chat", body["messages"]
    if isinstance(body.get("input"), list):
        return "responses", body["input"]
    return "other", []


def _codex_meta(body: dict) -> dict:
    cm = body.get("client_metadata") or {}
    raw = cm.get("x-codex-turn-metadata") if isinstance(cm, dict) else None
    try:
        return json.loads(raw) if isinstance(raw, str) else {}
    except ValueError:
        return {}


class Session:
    def __init__(self):
        self.tools = None
        self.parallel = None
        self.has_parallel = False
        self.last_tool_body = None  # stripped-id view of the last tool-bearing request
        self.task = None  # {"api", "item", "index", "text", "id"}


class Transformer:
    """Request-side transforms, shared by the live proxy and run3's offline dry-run."""

    def __init__(
        self,
        aliases: list[tuple[str, str]] | None = None,
        keep_prefix: bool = False,
        pin_task: bool = False,
    ):
        self.aliases = aliases or []
        self.keep_prefix = keep_prefix
        self.pin_task = pin_task
        self.sessions: dict[str, Session] = {}

    def session(self, body: dict) -> Session:
        key = str(body.get("prompt_cache_key") or "_")
        return self.sessions.setdefault(key, Session())

    def detect_compaction(self, body: dict, s: Session) -> dict | None:
        api, turns = _turns(body)
        n_tools = len(body.get("tools") or [])
        meta = _codex_meta(body)
        by = None
        if meta.get("request_kind") == "compaction":
            by = "codex-metadata"
        else:
            users = [t for t in turns if isinstance(t, dict) and t.get("role") == "user"]
            last = _text_of(users[-1]) if users else ""
            for harness, marker in COMPACTION_MARKERS:
                if marker in last[:400]:
                    by = f"marker:{harness}"
                    break
        shared, n_prev = 0, 0
        if s.last_tool_body is not None:
            prev = s.last_tool_body
            pa, pt = _turns(prev)
            n_prev = len(pt)
            if pa == api and body.get("instructions") == prev.get("instructions"):
                for a, b in zip(_strip_ids(turns), pt):
                    if a != b:
                        break
                    shared += 1
        # a full-history compaction replays the whole previous tool-bearing request (+ new items)
        full_history = s.last_tool_body is not None and shared >= max(2, n_prev - 1)
        if by is None and n_tools == 0 and s.tools and full_history:
            by = "toolless-shared-prefix"
        if by is None:
            return None
        return {
            "detected_by": by,
            "n_tools_sent": n_tools,
            "prefix_items_shared": shared,
            "prev_items": n_prev,
            "prefix_shared": bool(full_history),
            "codex_compaction": meta.get("compaction"),
        }

    def transform(self, body: dict, path: str) -> tuple[dict, dict]:
        """Returns (new_body, record). The input body is not mutated."""
        rec: dict = {}
        if self.aliases:
            c = [0]
            body = replace_all(body, self.aliases, c)
            if c[0]:
                rec["aliased_req"] = c[0]
        else:
            body = copy.deepcopy(body)
        if not is_model_path(path):
            return body, rec
        s = self.session(body)
        api, turns = _turns(body)
        n_tools = len(body.get("tools") or [])
        if self.pin_task and s.task is not None and api == "responses":
            # restore a middle-truncated task item BEFORE compaction detection, so a compaction that
            # replays the (restored) history still matches the live slot's prefix
            pin = self._pin(body, api, turns, s, insert=False)
            if pin:
                rec["task_pin"] = pin
        comp = self.detect_compaction(body, s)
        if comp is not None:
            rewritten = False
            if self.keep_prefix and n_tools == 0 and s.tools and comp["prefix_shared"]:
                body["tools"] = copy.deepcopy(s.tools)
                body["tool_choice"] = "none"
                if s.has_parallel:
                    body["parallel_tool_calls"] = s.parallel
                rewritten = True
            elif self.keep_prefix and not comp["prefix_shared"]:
                comp["not_rewritten_because"] = (
                    "separate summarizer call: it never shared the live prefix"
                )
            comp["rewritten"] = rewritten
            if rewritten:
                comp["tools_injected"] = len(body["tools"])
            rec["compaction"] = comp
            return body, rec
        if n_tools > 0:
            if self.pin_task and s.task is not None and "task_pin" not in rec:
                pin = self._pin(body, api, turns, s)
                if pin:
                    rec["task_pin"] = pin
            # learn (after pinning, so the remembered prefix is what went upstream)
            s.tools = copy.deepcopy(body.get("tools"))
            s.has_parallel = "parallel_tool_calls" in body
            s.parallel = body.get("parallel_tool_calls")
            s.last_tool_body = _strip_ids(
                {k: body.get(k) for k in ("instructions", "messages", "input") if k in body}
            )
            if s.task is None:
                api, turns = _turns(body)
                idx = max(
                    (
                        i
                        for i, t in enumerate(turns)
                        if isinstance(t, dict) and t.get("role") == "user"
                    ),
                    default=None,
                )
                if idx is not None:
                    s.task = {
                        "api": api,
                        "item": copy.deepcopy(turns[idx]),
                        "index": idx,
                        "text": _text_of(turns[idx]),
                        "id": turns[idx].get("id"),
                    }
        return body, rec

    def _pin(
        self, body: dict, api: str, turns: list, s: Session, insert: bool = True
    ) -> dict | None:
        task = s.task
        if not task["text"] or api != task["api"]:
            return None
        if api == "responses":
            for i, it in enumerate(turns):
                if isinstance(it, dict) and task["id"] and it.get("id") == task["id"]:
                    if _text_of(it) == task["text"]:
                        return None
                    old = len(_text_of(it))
                    turns[i] = copy.deepcopy(task["item"])
                    return {
                        "mode": "restored-truncated",
                        "index": i,
                        "chars_before": old,
                        "chars_after": len(task["text"]),
                    }
            if not insert or any(
                _text_of(it) == task["text"] for it in turns if isinstance(it, dict)
            ):
                return None
            at = min(task["index"], len(turns))
            turns.insert(at, copy.deepcopy(task["item"]))
            return {"mode": "inserted", "index": at, "chars": len(task["text"])}
        if api == "chat":
            for m in turns:
                if isinstance(m, dict) and m.get("role") == "user" and task["text"] in _text_of(m):
                    return None
            at = 0
            while (
                at < len(turns)
                and isinstance(turns[at], dict)
                and turns[at].get("role") in ("system", "developer")
            ):
                at += 1
            turns.insert(at, {"role": "user", "content": task["item"].get("content")})
            return {"mode": "inserted", "index": at, "chars": len(task["text"])}
        return None


# ---------------------------------------------------------------------------------------------
# proxy
# ---------------------------------------------------------------------------------------------
class State:
    def __init__(
        self,
        log_dir: Path,
        upstream: str,
        policy: str,
        think_budget: int = 0,
        think_message: str = "",
        aliases: list[tuple[str, str]] | None = None,
        hold_first: bool = False,
        hold_timeout_s: float = 900.0,
        keep_prefix: bool = False,
        pin_task: bool = False,
    ):
        self.log_dir = log_dir
        self.think_budget = think_budget
        self.think_message = think_message
        u = urlsplit(upstream)
        self.host, self.port = u.hostname, u.port or 80
        self.policy = policy
        self.lock = threading.Lock()
        self.seq = 0
        self.rows = open(log_dir / "wire.jsonl", "a", buffering=1)
        self.bodies = gzip.open(log_dir / "wire-bodies.jsonl.gz", "at")
        self.aliases = aliases or []
        self.back = [(c, r) for r, c in self.aliases]  # reply direction: CANON -> REAL
        self.hold_first = hold_first
        self.hold_timeout_s = hold_timeout_s
        self.held_claimed = False
        self.xf = Transformer(self.aliases, keep_prefix, pin_task)
        self.xf_lock = threading.Lock()


def _atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: State = None  # type: ignore[assignment]

    def log_message(self, *_a):
        pass

    def _hold(self, body: dict, seq: int, row: dict) -> None:
        st = self.state
        _atomic(
            st.log_dir / "first-body.json",
            json.dumps({"seq": seq, "t": time.time(), "path": self.path, "body": body}),
        )
        t_h = time.time()
        by = "timeout"
        while time.time() - t_h < st.hold_timeout_s:
            if (st.log_dir / "RELEASE").exists():
                by = "release"
                break
            time.sleep(0.2)
        row["held_s"] = round(time.time() - t_h, 2)
        row["hold_released_by"] = by

    def _relay(self, method: str):
        st = self.state
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        with st.lock:
            st.seq += 1
            seq = st.seq
        row = {
            "seq": seq,
            "t0": time.time(),
            "method": method,
            "path": self.path,
            "req_bytes": len(raw),
        }
        body = None
        if raw and "json" in (self.headers.get("Content-Type") or "json"):
            try:
                body = json.loads(raw)
            except ValueError:
                body = None
        if isinstance(body, dict):
            row.update(summarize_request(body))
            changed = False
            if st.aliases or st.xf.keep_prefix or st.xf.pin_task:
                with st.xf_lock:
                    body, rec = st.xf.transform(body, self.path)
                row.update(rec)
                changed = True
                if "compaction" in rec and rec["compaction"].get("rewritten"):
                    row["n_tools_upstream"] = len(body.get("tools") or [])
            removed = apply_policy(body, st.policy)
            if removed:
                row["stripped"] = removed
            injected = apply_think_budget(body, self.path, st.think_budget, st.think_message)
            if injected:
                row["injected"] = injected
            if removed or injected or changed:
                raw = json.dumps(body).encode()
            with st.lock:
                st.bodies.write(json.dumps({"seq": seq, "path": self.path, "body": body}) + "\n")
                st.bodies.flush()
            if (
                st.hold_first
                and is_model_path(self.path)
                and (body.get("tools") or [])
                and "compaction" not in row
            ):
                with st.lock:
                    claim = not st.held_claimed
                    st.held_claimed = True
                if claim:
                    self._hold(body, seq, row)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        headers["Content-Length"] = str(len(raw))
        conn = http.client.HTTPConnection(st.host, st.port, timeout=7200)
        acc: dict = {}
        rw = StreamRewriter(st.back) if st.back else None
        try:
            conn.request(method, self.path, body=raw if raw else None, headers=headers)
            resp = conn.getresponse()
            row["status"] = resp.status
            ctype = resp.getheader("Content-Type") or ""
            streaming = "text/event-stream" in ctype
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            if streaming:
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                if rw is None:
                    self._stream_raw(resp, acc)
                else:
                    self._stream_rewrite(resp, acc, rw)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                data = resp.read()
                if rw is not None:
                    try:
                        data = json.dumps(rw.full(json.loads(data))).encode()
                    except ValueError:
                        pass
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                try:
                    usage_from(json.loads(data), acc)
                except ValueError:
                    pass
        except (BrokenPipeError, ConnectionResetError) as exc:
            row["client_gone"] = type(exc).__name__
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            try:
                self.send_error(502, "upstream error")
            except Exception:  # noqa: BLE001
                pass
        finally:
            conn.close()
            row["t1"] = time.time()
            row["wall_s"] = round(row["t1"] - row["t0"], 3)
            row.update(acc)
            if rw is not None and rw.count:
                row["aliased_resp"] = rw.count
            with st.lock:
                st.rows.write(json.dumps(row) + "\n")

    def _chunk(self, b: bytes) -> None:
        if b:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
            self.wfile.flush()

    def _stream_raw(self, resp, acc: dict) -> None:
        """proxy2's byte-exact relay."""
        buf = b""
        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            self._chunk(chunk)
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if line.startswith(b"data:"):
                    payload = line[5:].strip()
                    if payload and payload != b"[DONE]":
                        try:
                            usage_from(json.loads(payload), acc)
                        except ValueError:
                            pass

    def _stream_rewrite(self, resp, acc: dict, rw: StreamRewriter) -> None:
        """Event-wise relay with CANON->REAL rewriting (SSE events are separated by a blank line)."""
        buf = b""
        last_delta: dict = {}
        last_chat: dict = {}

        def emit_event(block: bytes) -> None:
            lines = block.split(b"\n")
            evname = None
            data_lines = []
            other = []
            for ln in lines:
                s = ln.rstrip(b"\r")
                if s.startswith(b"event:"):
                    evname = s[6:].strip().decode()
                elif s.startswith(b"data:"):
                    data_lines.append(s[5:].strip())
                elif s:
                    other.append(s)
            payload = b"\n".join(data_lines)
            if not data_lines:
                self._chunk(block + b"\n\n")
                return
            if payload == b"[DONE]":
                held = rw.flush()
                by_choice: dict = {}
                for k, v in held.items():
                    if k[0] == "c":
                        by_choice.setdefault(k[1], {})[k[2]] = v
                    elif k[0] == "t":
                        by_choice.setdefault(k[1], {}).setdefault("tool_calls", []).append(
                            {"index": k[2], "function": {"arguments": v}}
                        )
                for idx, delta in by_choice.items():
                    syn = {
                        k: last_chat[k]
                        for k in ("id", "object", "created", "model")
                        if k in last_chat
                    }
                    syn["choices"] = [{"index": idx, "delta": delta, "finish_reason": None}]
                    self._chunk(b"data: " + json.dumps(syn).encode() + b"\n\n")
                self._chunk(block + b"\n\n")
                return
            try:
                obj = json.loads(payload)
            except ValueError:
                self._chunk(block + b"\n\n")
                return
            usage_from(obj, acc)
            if isinstance(obj, dict) and ("choices" in obj and "type" not in obj):
                outs = [(evname, o) for o in rewrite_chat_event(obj, rw)]
                last_chat.update(
                    {k: obj[k] for k in ("id", "object", "created", "model") if k in obj}
                )
            elif isinstance(obj, dict):
                outs = rewrite_responses_event(evname, obj, rw, last_delta)
            else:
                outs = [(evname, obj)]
            for name, o in outs:
                head = b"".join(x + b"\n" for x in other)
                if name:
                    head += b"event: " + name.encode() + b"\n"
                self._chunk(head + b"data: " + json.dumps(o).encode() + b"\n\n")

        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            buf += chunk.replace(b"\r\n", b"\n")
            while b"\n\n" in buf:
                block, buf = buf.split(b"\n\n", 1)
                emit_event(block)
        if buf.strip():
            emit_event(buf.strip(b"\n"))

    def do_POST(self):
        self._relay("POST")

    def do_GET(self):
        self._relay("GET")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--log", required=True)
    ap.add_argument("--upstream", default="http://127.0.0.1:8083")
    ap.add_argument("--effort-policy", choices=("strip", "passthrough"), default="strip")
    ap.add_argument("--think-budget", type=int, default=0)
    ap.add_argument("--think-message", default="")
    ap.add_argument("--alias", action="append", default=[], help="REAL=CANON (repeatable)")
    ap.add_argument("--hold-first", action="store_true")
    ap.add_argument("--hold-timeout-s", type=float, default=900.0)
    ap.add_argument("--keep-prefix-compaction", action="store_true")
    ap.add_argument("--pin-task", action="store_true")
    a = ap.parse_args()
    log = Path(a.log)
    log.mkdir(parents=True, exist_ok=True)
    aliases = []
    for spec in a.alias:
        real, _, canon = spec.partition("=")
        if not real or not canon:
            ap.error(f"bad --alias {spec!r}")
        aliases.append((real, canon))
    if a.hold_first:  # a stale capture/release from an earlier attempt of this call must not count
        for f in ("first-body.json", "RELEASE"):
            (log / f).unlink(missing_ok=True)
    Handler.state = State(
        log,
        a.upstream,
        a.effort_policy,
        a.think_budget,
        a.think_message,
        aliases,
        a.hold_first,
        a.hold_timeout_s,
        a.keep_prefix_compaction,
        a.pin_task,
    )
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    print(f"PORT {srv.server_address[1]}", flush=True)
    import signal as _signal

    def _term(*_a):
        raise KeyboardInterrupt

    _signal.signal(_signal.SIGTERM, _term)
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        with Handler.state.lock:
            Handler.state.bodies.close()
            Handler.state.rows.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
