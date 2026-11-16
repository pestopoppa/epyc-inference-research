#!/usr/bin/env python3
"""AGG-1.2 runner: Oolong-real pilot, P-large (planner + typed events + SQL) vs B1 (local REPL agent). PREDECLARED_AGG12.md is authoritative.

Code only: no model, server or process is started here; the keeper jobs run it under their grants.

Phases
  ingest   (CPU row)  extract the ONE selected window from the dataset, store it (orchestrator ``CorpusStore``: one document, line-block chunks),
                      then L4: a typed event extraction per transcript block through the private worker server, deterministic verification
                      against the cited line, and the ``oolong_events`` table (the typed roll / spell schema below). Writes out/store.sqlite,
                      out/events.jsonl, out/result.cpu.json.
  answers  (GPU row)  per question, in order: P-large, B1, retrieval-only comparator (informational); checkpoints (counts only) after 5 and 10
                      questions; then the verdict. Writes out/result.gpu.json.
  score               verdict only (no model), from results.jsonl.

Typed event schema (Jev-style: closed enums, integer slots, no free text except a spell name, a deterministic verifier mirror)
  roll   {line:int, character:enum(characters + Unknown), roll_type:enum(ROLL_TYPES), value:int 0-999, natural:bool}
  spell  {line:int, character:enum(characters + Unknown), spell:[A-Za-z -]{1,40}}
  The model never writes SQL text outside the bounded GBNF either; the SQL mirror (``sql_ok``) rejects anything the grammar would not admit.

P-large here = the orchestrator planner (``planner.plan``, its native path is RECORDED) + the predeclared L4/SQL route over the typed events
(bounded SELECT, one re-ask, answer call). B1 = the Phase 0 SEARCH/READ/FINAL loop (<= 30 calls, BM25) over the same chunks.

Exit codes: 0 complete (or smoke ok), 2 preflight, 3 futility stop (result filed), 4 precondition, 75 partial (resumable).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
for _c in (HERE.parent / "agg_phase0", HERE / "phase0"):
    if (_c / "phase0_arms.py").exists():
        sys.path.insert(0, str(_c))
        break
import phase0_arms as P  # noqa: E402

PREDECLARED_FILE = "PREDECLARED_AGG12.md"
PREDECLARED_SHA = "b370f704a128ce1b62f1cc7a3f3f38277128d33bdbaa5178547d09189caf6356"
DATASET = Path("/mnt/raid0/llm/datasets/oolong-real-3ae6ab58")
CORPUS_ID = "oolong_real_dnd_window"
DOC_ID = "episode"
RC_OK, RC_PREFLIGHT, RC_GATE, RC_PRECOND, RC_PARTIAL = 0, 2, 3, 4, 75

BLOCK_CHARS = 6000            # transcript block (chunk and L4 window) size
CONTEXT_LINES = 5             # previous lines shown as context only
EXTRACT_N_PREDICT = 700
SQL_N_PREDICT = 256
ANSWER_N_PREDICT = 400
WALL_CAP_S = 1800.0           # PREDECLARED: pilot wall cap
M4_CALLS_MAX, M4_PREFILL_MAX = 3, 2510
STOP_Z, STOP_MIN_DISCORDANT, TIE_WITHIN = 3.0, 9, 1
CHECKPOINTS = (5, 10)
ARMS = ("P-large", "B1", "RET")
B1_MAX_CALLS = 30
RET_TOP_K = 8

ROLL_TYPES = ("Acrobatics", "Animal Handling", "Arcana", "Athletics", "Deception", "History", "Insight", "Intimidation",
              "Investigation", "Medicine", "Nature", "Perception", "Performance", "Persuasion", "Religion", "Sleight of Hand",
              "Stealth", "Survival", "Initiative", "Attack", "Damage", "Saving Throw", "Ability Check", "Other")
COUNT_RULE = "End your reply with one line 'FINAL: <integer>'. Write the answer as a whole number (a percentage as the number without the % sign)."


# --------------------------------------------------------------------------- dataset window + transcript
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_sample(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text())
    qs = d["questions"]
    if d.get("n") != len(qs) or len(qs) != 10 or len({q["id"] for q in qs}) != 10:
        raise ValueError("sample is not 10 distinct questions")
    for q in qs:
        if not str(q["gold"]).strip().isdigit():
            raise ValueError(f"non-integer gold in sample: {q['id']}")
    return d


def extract_window(ds: Path, cwid: str) -> str:
    """The context window text of ``cwid`` from dnd/test.jsonl (streamed; the record id is verified after the cheap prefilter)."""
    key = cwid.encode()
    with (ds / "dnd" / "test.jsonl").open("rb") as f:
        for line in f:
            if key not in line[:600]:
                continue
            d = json.loads(line)
            if d.get("context_window_id") == cwid:
                return d["context_window_text"]
    raise LookupError(f"context window {cwid} not found")


def parse_transcript(text: str) -> dict[str, Any]:
    """``{mapping: {player: character}, lines: [(n, text)]}``; n is the 1-based line number inside the episode."""
    s0, e0 = text.rfind("[START OF EPISODE]"), text.rfind("[END OF EPISODE]")   # the instruction header names both markers too: take the LAST pair
    if s0 < 0 or e0 < s0:
        raise ValueError("episode markers not found")
    head, body = text[:s0], text[s0 + len("[START OF EPISODE]"):e0]
    mapping = {}
    for m in re.finditer(r"^(.+?) plays the character (.+?)\.?\s*$", head, re.M):
        mapping[m.group(1).strip()] = m.group(2).strip()
    lines = [ln for ln in body.split("\n") if ln.strip()]
    return {"mapping": mapping, "lines": [(i + 1, ln) for i, ln in enumerate(lines)]}


def clean_name(s: str) -> str:
    return re.sub(r"[^A-Za-z]", "", s)


def characters_of(mapping: dict[str, str]) -> dict[str, str]:
    """sanitised character -> sanitised player (the DM keeps 'DM'); SQL literals can then never contain a quote."""
    return {clean_name(c): clean_name(p) for p, c in mapping.items() if clean_name(c)}


def blocks(lines: list[tuple[int, str]], chars: int = BLOCK_CHARS) -> list[list[tuple[int, str]]]:
    out: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    n = 0
    for ln in lines:
        if cur and n + len(ln[1]) + 1 > chars:
            out.append(cur)
            cur, n = [], 0
        cur.append(ln)
        n += len(ln[1]) + 1
    if cur:
        out.append(cur)
    return out


def block_id(b: list[tuple[int, str]]) -> str:
    return f"b{b[0][0]:05d}-{b[-1][0]:05d}"


# --------------------------------------------------------------------------- typed schema: grammar and verifier
def _alt(items) -> str:
    return " | ".join('"' + str(i).replace("\\", "\\\\").replace('"', '\\"') + '"' for i in items)


def _opt_chain(cls: str, n: int) -> str:
    s = cls
    for _ in range(n - 1):
        s = f"{cls} ({s})?"
    return s


def extract_gbnf(chars: list[str]) -> str:
    ch = list(chars) + ["Unknown"]
    return "\n".join([
        'root ::= "{\\"rolls\\":[" (roll ("," roll)*)? "],\\"spells\\":[" (spell ("," spell)*)? "]}"',
        'roll ::= "{\\"line\\":" int ",\\"character\\":\\"" ch "\\",\\"roll_type\\":\\"" rt "\\",\\"value\\":" val ",\\"natural\\":" bool "}"',
        'spell ::= "{\\"line\\":" int ",\\"character\\":\\"" ch "\\",\\"spell\\":\\"" sp "\\"}"',
        f"int ::= {_opt_chain('[0-9]', 5)}",
        f"val ::= {_opt_chain('[0-9]', 3)}",
        'bool ::= "true" | "false"',
        f"ch ::= {_alt(ch)}",
        f"rt ::= {_alt(ROLL_TYPES)}",
        f"sp ::= [A-Za-z] ({_opt_chain('[A-Za-z -]', 39)})?",
    ])


def line_text(lines: dict[int, str], n: int) -> str:
    return lines.get(n, "")


def verify_event(kind: str, e: dict[str, Any], lines: dict[int, str], allowed: tuple[int, int], chars: set[str]) -> tuple[bool, str]:
    """Deterministic check of one extracted event against the cited line; no model. Returns (ok, reason)."""
    n = e.get("line")
    if not isinstance(n, int) or not (allowed[0] <= n <= allowed[1]) or n not in lines:
        return False, "line_out_of_block"
    if e.get("character") not in chars | {"Unknown"}:
        return False, "character_not_in_enum"
    txt = lines[n]
    if kind == "roll":
        if e.get("roll_type") not in ROLL_TYPES:
            return False, "roll_type_not_in_enum"
        v = e.get("value")
        if not isinstance(v, int) or isinstance(v, bool) or not (0 <= v <= 999):
            return False, "value_not_int"
        if re.search(rf"(?<![0-9]){v}(?![0-9])", txt) is None:
            return False, "value_not_in_line"
        return True, "ok"
    sp = str(e.get("spell") or "").strip()
    if not sp:
        return False, "empty_spell"
    near = " ".join(lines.get(k, "") for k in range(max(n - 3, allowed[0]), n + 1)).lower()
    if re.sub(r"[^a-z]", "", sp.lower()) not in re.sub(r"[^a-z]", "", near):
        return False, "spell_not_near_line"
    return True, "ok"


def extract_prompt(b: list[tuple[int, str]], ctx: list[tuple[int, str]], mapping: dict[str, str]) -> str:
    people = "\n".join(f"{p} plays {c}" for p, c in mapping.items())
    cx = "\n".join(f"[L{n}] {t}" for n, t in ctx) or "(start of episode)"
    tg = "\n".join(f"[L{n}] {t}" for n, t in b)
    return (
        "You read part of a Dungeons & Dragons episode transcript. Lines are numbered [L<n>]. Extract typed events from the TARGET lines "
        "only (the context lines are shown for reference).\n"
        "ROLLS: a dice roll whose numeric result is spoken in a target line (for example 'Liam: 18.' after the DM asked for an investigate "
        "check). roll_type is the skill, save or kind of roll that was asked for; use Other when unclear. value is the number announced. "
        "natural is true only when the speaker calls it a natural roll (nat 20, nat 1, natural 13) or the unmodified die. character is the "
        "character who rolled; Unknown when you cannot tell. Do not extract hit points, damage totals, prices, distances or other numbers.\n"
        "SPELLS: a spell that a character casts in a target line (the spell name appears in that line or just before it); character is the "
        "caster; Unknown when you cannot tell.\n"
        "line is the [L<n>] number of the line where the roll value or the casting is stated. Reply with one JSON object "
        '{"rolls": [...], "spells": [...]} and nothing else; empty lists when there are none.\n'
        f"Players and characters:\n{people}\n\nCONTEXT lines (do not extract):\n{cx}\n\nTARGET lines:\n{tg}\n\nJSON:")


# --------------------------------------------------------------------------- ingest (L4 events)
def extract_block(backend: Any, b: list[tuple[int, str]], ctx: list[tuple[int, str]], mapping: dict[str, str], chars: list[str],
                  all_lines: dict[int, str], *, depth: int = 0) -> dict[str, Any]:
    """One block -> verified events. A reply cut at ``n_predict`` splits the block in two (one level) and merges."""
    gbnf = extract_gbnf(chars)
    r = backend.chat(extract_prompt(b, ctx, mapping), n_predict=EXTRACT_N_PREDICT, grammar=gbnf, id_slot=0)
    if r.get("finish_reason") == "length" and depth < 2 and len(b) > 4:
        h = len(b) // 2
        left = extract_block(backend, b[:h], ctx, mapping, chars, all_lines, depth=depth + 1)
        right = extract_block(backend, b[h:], b[max(h - CONTEXT_LINES, 0):h], mapping, chars, all_lines, depth=depth + 1)
        return {"events": left["events"] + right["events"], "rejected": left["rejected"] + right["rejected"],
                "n_raw": left["n_raw"] + right["n_raw"], "split": True, "status": "ok"}
    try:
        obj = json.loads(r["text"])
        raw = [("roll", e) for e in obj.get("rolls", [])] + [("spell", e) for e in obj.get("spells", [])]
    except (ValueError, AttributeError, TypeError):
        return {"events": [], "rejected": [], "n_raw": 0, "split": False, "status": "error:bad_json", "text": r["text"][-300:]}
    allowed = (b[0][0], b[-1][0])
    ev, rej = [], []
    for kind, e in raw:
        ok, why = verify_event(kind, e, all_lines, allowed, set(chars))
        (ev if ok else rej).append({"kind": kind, **e} if ok else {"kind": kind, "reason": why, **{k: e.get(k) for k in ("line", "value", "spell")}})
    return {"events": ev, "rejected": rej, "n_raw": len(raw), "split": False, "status": "ok"}


def dedupe(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen, out = set(), []
    for e in events:
        k = (e["kind"], e["line"], e["character"], e.get("roll_type"), e.get("value"), str(e.get("spell") or "").lower())
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


EVENT_DDL = ("""CREATE TABLE IF NOT EXISTS oolong_events (
    event_id TEXT PRIMARY KEY, kind TEXT NOT NULL, line INTEGER, character TEXT, player TEXT, roll_type TEXT,
    value INTEGER, nat INTEGER, spell TEXT, block TEXT)""",)


def write_events(conn: sqlite3.Connection, events: list[dict[str, Any]], char_player: dict[str, str]) -> int:
    for s in EVENT_DDL:
        conn.execute(s)
    conn.execute("DELETE FROM oolong_events")
    n = 0
    for i, e in enumerate(events):
        ch = e["character"]
        conn.execute("INSERT INTO oolong_events VALUES (?,?,?,?,?,?,?,?,?,?)",
                     [f"e{i:05d}", e["kind"], e["line"], ch, char_player.get(ch, "Unknown"), e.get("roll_type"),
                      e.get("value"), 1 if e.get("natural") else 0, str(e["spell"]).strip().title().replace("'", "") if e.get("spell") else None,
                      e.get("block")])
        n += 1
    conn.commit()
    return n


def doc_result(blks: list[list[tuple[int, str]]], text_sha: str):
    from src.models.document import DocumentPreprocessResult, Section  # noqa: PLC0415
    secs = [Section(id=f"s{i}", title=f"Lines {b[0][0]}-{b[-1][0]}", level=1, content="\n".join(t for _, t in b), page_start=i + 1, page_end=i + 1)
            for i, b in enumerate(blks)]
    return DocumentPreprocessResult(original_path=f"oolong-real:{text_sha[:16]}", sections=secs, figures=[], total_pages=len(secs))


def run_ingest(a: argparse.Namespace, rec: P.Recorder, sample: dict[str, Any], smoke: bool, guard: Callable[[], None]) -> dict[str, Any]:
    from src.aggregation.store import CorpusStore  # noqa: PLC0415
    from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
    out: Path = a.out
    wp = out / "window.txt"
    if wp.exists():
        text = wp.read_text()
    else:
        text = extract_window(a.dataset, sample["context_window_id"])
        wp.write_text(text)
    if len(text) != sample["context_chars"]:
        raise ValueError(f"window length {len(text)} != sample context_chars {sample['context_chars']}")
    tr = parse_transcript(text)
    blks = blocks(tr["lines"])
    all_lines = dict(tr["lines"])
    cmap = characters_of(tr["mapping"])
    chars = sorted(cmap)
    if "window" not in rec.done:
        rec.emit("window", unit="window", sha256=sha256_text(text), chars=len(text), n_lines=len(tr["lines"]), n_blocks=len(blks),
                 n_characters=len(chars), context_window_id=sample["context_window_id"])
    store = CorpusStore(out / "store.sqlite")
    store.ingest(doc_result(blks, sha256_text(text)), corpus_id=CORPUS_ID, doc_id=DOC_ID, entity=None)
    backend = LlamaServerBackend(a.url, model_id=a.model_id)
    order = list(range(len(blks)))
    if smoke:
        order = [len(blks) // 2]
    for i in order:
        b = blks[i]
        unit = f"l4:{block_id(b)}"
        if unit in rec.done:
            continue
        guard()
        ctx = [] if i == 0 else blks[i - 1][-CONTEXT_LINES:]
        c0, p0, t0 = backend.calls, backend.prefill_tokens, time.time()
        try:
            res = extract_block(backend, b, ctx, tr["mapping"], chars, all_lines)
        except Exception as e:  # noqa: BLE001 - recorded; an out-of-window block is a result, not a crash
            res = {"events": [], "rejected": [], "n_raw": 0, "split": False, "status": f"error:{type(e).__name__}"}
        for e in res["events"]:
            e["block"] = block_id(b)
        rec.emit("t_window", unit=unit, status=res["status"], calls=backend.calls - c0, prefill_tokens=backend.prefill_tokens - p0,
                 n_events=len(res["events"]), n_rejected=len(res["rejected"]), n_raw=res["n_raw"], split=res["split"],
                 events=res["events"], rejected=res["rejected"][:20], wall_s=time.time() - t0, final=True)
        if smoke:
            return {"complete": False, "smoke": True}
    done = all(f"l4:{block_id(b)}" in rec.done for b in blks)
    recs = {r["unit"]: r for r in rec.records if r.get("kind") == "t_window" and r.get("final")}
    events = dedupe([e for b in blks for e in (recs.get(f"l4:{block_id(b)}") or {}).get("events", [])])
    n_ev = write_events(store.conn, events, cmap) if done else 0
    store.close()
    if not done:
        return {"complete": False}
    with (out / "events.jsonl").open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    errs = sum(1 for r in recs.values() if str(r.get("status")).startswith("error"))
    calls = sum(int(r.get("calls") or 0) for r in recs.values())
    res = {"phase": "cpu", "complete": True, "n_events": n_ev, "n_rolls": sum(1 for e in events if e["kind"] == "roll"),
           "n_spells": sum(1 for e in events if e["kind"] == "spell"), "n_blocks": len(blks), "n_block_errors": errs, "calls": calls,
           "n_rejected": sum(int(r.get("n_rejected") or 0) for r in recs.values()), "window_sha256": sha256_text(text),
           "futility_stop": "ingest_empty" if n_ev == 0 else None, "store_sha256": None, "predeclared_sha256": PREDECLARED_SHA,
           "rc": RC_GATE if n_ev == 0 else RC_OK}
    return res


# --------------------------------------------------------------------------- P-large: bounded SQL over the typed events
class EventCatalog:
    def __init__(self, conn: sqlite3.Connection) -> None:
        q = lambda s: [r[0] for r in conn.execute(s).fetchall() if r[0]]  # noqa: E731
        self.characters = tuple(sorted(q("SELECT DISTINCT character FROM oolong_events")))
        self.players = tuple(sorted(q("SELECT DISTINCT player FROM oolong_events")))
        self.spells = tuple(sorted(q("SELECT DISTINCT spell FROM oolong_events")))
        self.roll_types = ROLL_TYPES


COLS = ("character", "player", "roll_type", "spell", "value")


def sql_gbnf(cat: EventCatalog) -> str:
    ne = lambda t: t or ("__none__",)  # noqa: E731
    return "\n".join([
        "root ::= \"SELECT \" sel \" FROM events WHERE kind = '\" kind \"'\" conds group? order? limit?",
        'kind ::= "roll" | "spell"',
        'sel ::= "COUNT(*)" | "COUNT(DISTINCT " col ")" | pct | col ", COUNT(*)"',
        'pct ::= "ROUND(100.0 * SUM(CASE WHEN " cond " THEN 1 ELSE 0 END) / COUNT(*))"',
        f"col ::= {_alt(COLS)}",
        'conds ::= (" AND " cond (" AND " cond (" AND " cond (" AND " cond)?)?)?)?',
        "cond ::= \"roll_type = '\" rt \"'\" | \"character = '\" ch \"'\" | \"player = '\" pl \"'\" | \"spell = '\" sp \"'\" | \"value \" op \" \" num | \"nat = 1\"",
        f"rt ::= {_alt(cat.roll_types)}",
        f"ch ::= {_alt(ne(cat.characters))}",
        f"pl ::= {_alt(ne(cat.players))}",
        f"sp ::= {_alt(ne(cat.spells))}",
        f"op ::= {_alt(('>=', '<=', '!=', '=', '>', '<'))}",
        f"num ::= {_opt_chain('[0-9]', 3)}",
        'group ::= " GROUP BY " col',
        'order ::= " ORDER BY COUNT(*) DESC"',
        'limit ::= " LIMIT " [0-9] [0-9]?',
    ])


def sql_regex(cat: EventCatalog) -> "re.Pattern[str]":
    def alt(t):
        return "(?:" + "|".join(re.escape(x) for x in sorted(t or ("__none__",), key=len, reverse=True)) + ")"
    col = alt(COLS)
    cond = (rf"(?:roll_type = '{alt(cat.roll_types)}'|character = '{alt(cat.characters)}'|player = '{alt(cat.players)}'|spell = '{alt(cat.spells)}'"
            rf"|value (?:>=|<=|!=|=|>|<) [0-9]{{1,3}}|nat = 1)")
    sel = (rf"(?:COUNT\(\*\)|COUNT\(DISTINCT {col}\)|ROUND\(100\.0 \* SUM\(CASE WHEN {cond} THEN 1 ELSE 0 END\) / COUNT\(\*\)\)|{col}, COUNT\(\*\))")
    return re.compile(rf"SELECT {sel} FROM events WHERE kind = '(?:roll|spell)'(?: AND {cond}){{0,4}}(?: GROUP BY {col})?"
                      rf"(?: ORDER BY COUNT\(\*\) DESC)?(?: LIMIT [0-9]{{1,2}})?")


def sql_ok(sql: str, cat: EventCatalog) -> tuple[bool, str]:
    s = sql.strip().rstrip(";")
    return (True, "ok") if sql_regex(cat).fullmatch(s) else (False, "not a statement of the events plan grammar")


def create_events_view(conn: sqlite3.Connection) -> None:
    conn.execute("DROP VIEW IF EXISTS temp.events")
    conn.execute("CREATE TEMP VIEW events AS SELECT kind, line, character, player, roll_type, value, nat, spell FROM oolong_events")


def sql_prompt(question: str, cat: EventCatalog) -> str:
    return ("Write ONE SQLite SELECT over the view events(kind, character, player, roll_type, value, nat, spell). kind is 'roll' or 'spell'. "
            "Every row is ONE typed event extracted from the episode transcript: a roll row has roll_type, value (the number announced) and nat "
            "(1 when it was a natural roll); a spell row has spell. character is the in-game character, player the real player. Always start "
            "with WHERE kind = 'roll' or WHERE kind = 'spell'. 'How many' is COUNT(*) with conditions (a count of distinct characters is "
            "COUNT(DISTINCT character)); a percentage of the rolls that have a property is ROUND(100.0 * SUM(CASE WHEN <condition> THEN 1 ELSE 0 END) "
            "/ COUNT(*)) with no other condition; 'Nat1s' is value = 1 AND nat = 1.\n"
            f"roll_type: {'; '.join(cat.roll_types)}\ncharacter: {'; '.join(cat.characters) or '(none)'}\nplayer: {'; '.join(cat.players) or '(none)'}\n"
            f"spell: {'; '.join(cat.spells) or '(none)'}\nQuestion: {question}\nSQL:")


def sql_answer(backend: Any, conn: sqlite3.Connection, question: str, cat: EventCatalog) -> tuple[str, str, dict[str, Any]]:
    from src.aggregation.llm_backend import OutOfWindow  # noqa: PLC0415
    create_events_view(conn)
    gbnf = sql_gbnf(cat)
    prompt = sql_prompt(question, cat)
    info: dict[str, Any] = {"sql_table": "events", "n_events": conn.execute("SELECT COUNT(*) FROM oolong_events").fetchone()[0], "sql_reask": False}
    sql, why, ok = "", "", False
    try:
        for attempt in (0, 1):
            p = prompt if attempt == 0 else prompt + " " + sql[:300] + f"\nThat statement was rejected ({why}). Write ONE corrected statement.\nSQL:"
            reply = backend.chat(p, n_predict=SQL_N_PREDICT, grammar=gbnf, id_slot=0)
            sql = str(reply["text"]).strip()
            ok, why = (False, "truncated at n_predict") if reply.get("finish_reason") == "length" else sql_ok(sql, cat)
            if attempt == 0:
                info.update(sql_first=sql, sql_first_reason=why)
            if ok:
                break
            info["sql_reask"] = True
    except OutOfWindow:
        return "", "out_of_window", info
    info.update(sql=sql, sql_valid=ok, sql_reason=why)
    rows: list[Any] = []
    if ok:
        try:
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchmany(P.CFG.sql_row_cap)]
        except sqlite3.Error as e:
            info["sql_error"] = str(e)[:200]
    info["n_rows"] = len(rows)
    ctx = f"SQL: {sql}\nRows ({len(rows)}):\n" + "\n".join(json.dumps(r, default=str) for r in rows)
    try:
        ans = backend.chat(f"{ctx}\n\nQuestion: {question}\n{COUNT_RULE}", n_predict=ANSWER_N_PREDICT, id_slot=0)
    except OutOfWindow:
        return "", "out_of_window", info
    info["finish_reason"] = ans.get("finish_reason")
    return ans["text"], "ok", info


def p_large_answer(backend: Any, conn: sqlite3.Connection, q: dict[str, Any], cat: EventCatalog, plan_ctx: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    from src.aggregation import planner  # noqa: PLC0415
    pf = planner.PlannerFacts(corpus_tokens=plan_ctx["corpus_tokens"], n_docs=1, questions_so_far=plan_ctx["qsf"],
                              available_layers=("L0", "L1", "L3", "L4"), sql_fields_present=True)
    dec = planner.plan(q["question"], pf)
    extra = {"planner_native_path": dec.path, "decided_by": dec.decided_by, "receipt_stub": dec.receipt.get("stub", False),
             "planner_corpus_tokens": pf.corpus_tokens, "route": "events_sql (PREDECLARED L4/SQL path; the planner's native path is recorded)"}
    text, st, info = sql_answer(backend, conn, q["question"], cat)
    extra.update(info)
    return text, st, extra


# --------------------------------------------------------------------------- B1 and the retrieval-only comparator
def b1_answer(backend: Any, chunks: list[dict[str, Any]], question: str, *, trace: list[str] | None = None) -> tuple[str, str]:
    """Phase 0 B1 (SEARCH / READ / FINAL, <= 30 calls, BM25) with the corpus description of this episode."""
    from src.aggregation.llm_backend import OutOfWindow  # noqa: PLC0415
    from src.repl_environment.context_search import BM25Index  # noqa: PLC0415
    idx = BM25Index([c["text"] or "" for c in chunks])
    by_id = {c["chunk_id"]: c for c in chunks}
    grammar = 'root ::= ("SEARCH: " | "READ: " | "FINAL: ") [^\\n]+'
    head = ("You answer a question about ONE Dungeons & Dragons episode transcript (split into chunks of numbered lines) using tools. Each turn reply "
            "with exactly one line: 'SEARCH: <keywords>' (lexical search, returns chunk ids), 'READ: <chunk_id>' (returns the chunk text) or "
            f"'FINAL: <answer>'.\n{COUNT_RULE}\nQuestion: {question}\n")
    hist = ""
    for _ in range(B1_MAX_CALLS):
        try:
            r = backend.chat(head + hist + "Next action:", n_predict=120, grammar=grammar, id_slot=0)
        except OutOfWindow:
            return "", "out_of_window"
        act = r["text"].strip()
        if trace is not None:
            trace.append(act[:120])
        hist += f"\n> {act}\n"
        if act.startswith("FINAL:"):
            return act, "ok"
        arg = act.split(":", 1)[1].strip() if ":" in act else ""
        if act.startswith("SEARCH:"):
            hits = idx.scores(arg)[:P.CFG.b1_search_k]
            hist += "\n".join(f"[{chunks[i]['chunk_id']}] {(chunks[i].get('title') or '')[:80]} | "
                              f"{(chunks[i]['text'] or '')[:200].replace(chr(10), ' ')}" for i, _ in hits) or "(no hits)"
        elif act.startswith("READ:"):
            c = by_id.get(arg)
            hist += (c["text"] or "")[:P.CFG.b1_read_chars] if c else "(unknown chunk id)"
    return "", "budget_exhausted"


def retrieval_answer(backend: Any, chunks: list[dict[str, Any]], question: str) -> tuple[str, str]:
    """Informational comparator: BM25 top-k chunks, ONE count prompt, no planner, no REPL."""
    from src.aggregation.llm_backend import OutOfWindow  # noqa: PLC0415
    from src.repl_environment.context_search import BM25Index  # noqa: PLC0415
    idx = BM25Index([c["text"] or "" for c in chunks])
    top = sorted(i for i, _ in idx.scores(question)[:RET_TOP_K])
    ctx = "\n\n".join(chunks[i]["text"] or "" for i in top)
    try:
        r = backend.chat(f"Transcript excerpts:\n{ctx}\n\nQuestion: {question}\n{COUNT_RULE}", n_predict=ANSWER_N_PREDICT, id_slot=0)
    except OutOfWindow:
        return "", "out_of_window"
    return r["text"], "ok"


# --------------------------------------------------------------------------- scoring and the verdict (PREDECLARED_AGG12.md, exactly)
def score_count(gold: str | int, text: str) -> bool | None:
    """Exact integer match after parse (Phase 0 style); None when no number can be read (counted wrong, reported separately)."""
    fl = P.final_line(text or "")
    nums = P.parse_numbers(fl, money_scale=False) if fl else []
    if not nums:
        return None
    v = nums[0][0]
    return bool(float(v).is_integer() and int(v) == int(gold))


def final_answers(records: list[dict[str, Any]], arm: str) -> dict[str, dict[str, Any]]:
    return {r["qid"]: r for r in records if r.get("kind") == "answer" and r.get("arm") == arm and r.get("final")}


def arm_scores(records: list[dict[str, Any]], arm: str, questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-scores every answer from its raw text and the sample gold (the record's own ``correct`` is never trusted)."""
    got = final_answers(records, arm)
    correct: dict[str, bool] = {}
    reason: dict[str, str] = {}
    unparse = cap = oow = 0
    for q in questions:
        r = got.get(q["id"])
        if r is None:
            continue
        if r.get("status") == "budget_exhausted":
            correct[q["id"]], reason[q["id"]] = False, "cap"
            cap += 1
        elif r.get("status") != "ok":
            correct[q["id"]], reason[q["id"]] = False, str(r.get("status"))
            oow += 1
        else:
            s = score_count(q["gold"], r.get("answer", ""))
            correct[q["id"]] = s is True
            if s is None:
                reason[q["id"]] = "unparseable"
                unparse += 1
    return {"correct": correct, "reason": reason, "n_answered": len(correct), "n_correct": sum(correct.values()),
            "n_unparseable": unparse, "n_cap": cap, "n_other_nonok": oow}


def paired(p: dict[str, bool], b: dict[str, bool], ids: list[str]) -> dict[str, Any]:
    both = sum(p.get(i, False) and b.get(i, False) for i in ids)
    only_p = sum(p.get(i, False) and not b.get(i, False) for i in ids)
    only_b = sum(b.get(i, False) and not p.get(i, False) for i in ids)
    neither = len(ids) - both - only_p - only_b
    d = only_p + only_b
    z = (only_p - only_b) / math.sqrt(d) if d else 0.0
    return {"n": len(ids), "both_right": both, "only_p_large": only_p, "only_b1": only_b, "neither": neither, "discordant": d, "z": z,
            "p_large_acc": (both + only_p) / len(ids) if ids else None, "b1_acc": (both + only_b) / len(ids) if ids else None}


def m4_of(records: list[dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    got = final_answers(records, "P-large")
    rs = [got[i] for i in ids if i in got]
    if not rs:
        return {"n": 0, "median_calls": None, "median_prefill": None, "within_phase0_bounds": None}
    mc = statistics.median(int(r.get("calls") or 0) for r in rs)
    mp = statistics.median(int(r.get("prefill_tokens") or 0) for r in rs)
    return {"n": len(rs), "median_calls": mc, "median_prefill": mp, "within_phase0_bounds": bool(mc <= M4_CALLS_MAX and mp <= M4_PREFILL_MAX),
            "bounds": {"median_calls_max": M4_CALLS_MAX, "median_prefill_max": M4_PREFILL_MAX}}


def verdict(records: list[dict[str, Any]], questions: list[dict[str, Any]], *, futility: str | None = None,
            ingest: dict[str, Any] | None = None) -> dict[str, Any]:
    """The mechanical verdict. M2 paired P-large vs B1 counts; STOP-ON-ANSWER after all 10: |z| >= 3 with b + c >= 9, OR the two within 1
    question (tied); otherwise UNRESOLVED (never a tie win). M4 vs the Phase 0 bounds; M1 null with the reason; comparator informational."""
    ids = [q["id"] for q in questions]
    sc = {a: arm_scores(records, a, questions) for a in ARMS}
    out: dict[str, Any] = {
        "schema": "agg12_verdict.v1", "predeclared_sha256": PREDECLARED_SHA, "n_questions": len(ids),
        "m1": None, "m1_reason": "Oolong-real ships no gold-evidence cells (answer only)",
        "arms": {a: {k: v for k, v in s.items() if k != "correct"} | {"correct_by_qid": s["correct"]} for a, s in sc.items()},
        "m4": m4_of(records, ids), "ingest": ingest,
        "retrieval_comparator": {"informational": True, "n_correct": sc["RET"]["n_correct"], "n_answered": sc["RET"]["n_answered"]},
    }
    complete = sc["P-large"]["n_answered"] == len(ids) and sc["B1"]["n_answered"] == len(ids)
    out["complete"] = complete
    if futility:
        out["verdict"], out["answered"] = f"futility:{futility}", False
        return out
    if not complete:
        out["verdict"], out["answered"] = "incomplete", False
        return out
    m2 = paired(sc["P-large"]["correct"], sc["B1"]["correct"], ids)
    out["m2"] = m2
    diff = abs(sc["P-large"]["n_correct"] - sc["B1"]["n_correct"])
    if diff <= TIE_WITHIN:
        out["verdict"], out["answered"] = "answered:tied_within_1", True
    elif m2["discordant"] >= STOP_MIN_DISCORDANT and abs(m2["z"]) >= STOP_Z:
        out["verdict"], out["answered"] = ("answered:p_large_wins" if m2["z"] > 0 else "answered:b1_wins"), True
    else:
        out["verdict"], out["answered"] = "unresolved", False
    out["resolvable_gap_note"] = "at n = 10 only a gap of >= 0.5 is resolvable; a smaller one is UNRESOLVED, never a tie win"
    return out


# --------------------------------------------------------------------------- answers phase
def run_answers(a: argparse.Namespace, rec: P.Recorder, sample: dict[str, Any], smoke: bool, guard: Callable[[], None]) -> dict[str, Any]:
    from src.aggregation.store import CorpusStore  # noqa: PLC0415
    from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
    sp = a.out / "store.sqlite"
    if not sp.exists():
        P.copy_store(a.ingest_dir / "store.sqlite", sp)
        rec.emit("store_copied", src=str(a.ingest_dir / "store.sqlite"), sha256=sha256_file(sp), final=False)
    store = CorpusStore(sp)
    chunks = store.chunks(CORPUS_ID)
    corpus_tokens = store.corpus_tokens(CORPUS_ID)
    cat = EventCatalog(store.conn)
    backend = LlamaServerBackend(a.url, model_id=a.model_id)
    qs = sample["questions"]
    questions = qs[:1] if smoke else qs
    runners: dict[str, Callable[[dict[str, Any], int], tuple[str, str, dict[str, Any]]]] = {
        "P-large": lambda q, i: p_large_answer(backend, store.conn, q, cat, {"corpus_tokens": corpus_tokens, "qsf": i + 1}),
        "B1": lambda q, i: (*b1_answer(backend, chunks, q["question"], trace=(tr := [])), {"loop_actions": tr}),
        "RET": lambda q, i: (*retrieval_answer(backend, chunks, q["question"]), {}),
    }
    wall_before = sum(float(r.get("wall_s") or 0) for r in rec.records if r.get("kind") == "answer" and r.get("final"))
    for i, q in enumerate(questions):
        for arm in ARMS:
            unit = f"answer:{arm}:{q['id']}"
            if unit in rec.done:
                continue
            guard()
            spent = sum(float(r.get("wall_s") or 0) for r in rec.records if r.get("kind") == "answer" and r.get("final"))
            if spent > WALL_CAP_S and not smoke:
                store.close()
                return {"complete": True, "futility": "wall_cap_1800s", "wall_s": spent, "wall_before_resume": wall_before}
            t0, c0, p0 = time.time(), backend.calls, backend.prefill_tokens
            text, st, extra = runners[arm](q, i)
            rec.emit("answer", unit=unit, arm=arm, qid=q["id"], status=st, calls=backend.calls - c0, prefill_tokens=backend.prefill_tokens - p0,
                     answer=text[-600:], final=st in ("ok", "out_of_window", "budget_exhausted"), wall_s=time.time() - t0, **extra)
            if smoke:
                store.close()
                return {"complete": False, "smoke": True}
        n = i + 1
        if n in CHECKPOINTS and f"checkpoint:{n}" not in rec.done:   # COUNTS ONLY: no per-question output is read here
            rec.emit("checkpoint", unit=f"checkpoint:{n}", n_questions=n, final=True,
                     answered={a_: sum(1 for r in rec.records if r.get("kind") == "answer" and r.get("arm") == a_ and r.get("final")) for a_ in ARMS})
    store.close()
    done = all(f"answer:{arm}:{q['id']}" in rec.done for q in qs for arm in ARMS)
    return {"complete": done}


# --------------------------------------------------------------------------- checks and CLI
def orch_setup(a: argparse.Namespace) -> list[str]:
    if a.orch_root is None:
        return ["--orch-root is required"]
    sys.path.insert(0, str(a.orch_root))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    try:
        from src.aggregation import planner, store, llm_backend  # noqa: F401, PLC0415
        from src.models.document import DocumentPreprocessResult, Section  # noqa: F401, PLC0415
        from src.repl_environment.context_search import BM25Index  # noqa: F401, PLC0415
    except Exception as e:  # noqa: BLE001
        return [f"orchestrator import failed: {type(e).__name__}: {e}"]
    return []


def checks(a: argparse.Namespace, sample: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    problems = orch_setup(a)
    info: dict[str, Any] = {}
    pre = a.predeclared or (HERE / "agg12" / PREDECLARED_FILE)
    if pre.exists():
        if sha256_file(pre) != PREDECLARED_SHA:
            problems.append("PREDECLARED_AGG12.md sha256 mismatch")
    else:
        problems.append(f"PREDECLARED file missing: {pre}")
    # grammar/mirror self-test on synthetic events (zero compute)
    cat = EventCatalog(_synthetic_conn())
    ok1, _ = sql_ok("SELECT COUNT(*) FROM events WHERE kind = 'roll' AND roll_type = 'Nature'", cat)
    ok2, _ = sql_ok("SELECT COUNT(*) FROM events WHERE kind = 'roll'; DROP TABLE events", cat)
    if not ok1 or ok2:
        problems.append("SQL mirror self-test failed")
    info["sql_mirror_selftest"] = {"admits_valid": ok1, "rejects_invalid": not ok2}
    info["sample"] = {"context_window_id": sample["context_window_id"], "context_chars": sample["context_chars"], "n": sample["n"]}
    return problems, info


def _synthetic_conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    write_events(c, [{"kind": "roll", "line": 1, "character": "Grog", "roll_type": "Nature", "value": 5, "natural": False, "block": "b"},
                     {"kind": "spell", "line": 2, "character": "Tiberius", "spell": "Stoneskin", "block": "b"}], {"Grog": "Travis", "Tiberius": "Orion"})
    return c


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("ingest", "answers", "score"), required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sample", type=Path, required=True, help="sample.json (sealed by the row)")
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--predeclared", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--orch-root", type=Path, default=None)
    ap.add_argument("--orch-pin", default="")
    ap.add_argument("--research-pin", default="")
    ap.add_argument("--ingest-dir", type=Path, default=None, help="answers/score: the CPU ingest row's out dir")
    ap.add_argument("--url")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--smoke", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        sample = load_sample(a.sample)
    except (OSError, ValueError, KeyError) as e:
        print(json.dumps({"ok": False, "problems": [f"cannot load sample: {type(e).__name__}: {e}"]}))
        return RC_PREFLIGHT
    if a.phase == "score":
        recs = P.Recorder(a.out / "results.jsonl").records if a.out else []
        ing = json.loads((a.ingest_dir / "result.cpu.json").read_text()) if a.ingest_dir and (a.ingest_dir / "result.cpu.json").exists() else None
        print(json.dumps(verdict(recs, sample["questions"], ingest=ing), indent=1, default=str))
        return RC_OK
    problems, info = checks(a, sample)
    if a.phase == "answers":
        ir = a.ingest_dir / "result.cpu.json" if a.ingest_dir else None
        ready = bool(ir and ir.exists() and json.loads(ir.read_text()).get("complete") is True and not json.loads(ir.read_text()).get("futility_stop"))
        info["ingest_ready"] = ready
        if not ready and not a.dry_run:
            problems.append("ingest row not complete / futility stop")
    if a.dry_run:
        plan: dict[str, Any] = {"phase": a.phase, "model_calls": 0, "servers_started": 0, "wrote_files": 0}
        if a.phase == "ingest":
            try:
                text = extract_window(a.dataset, sample["context_window_id"])
                tr = parse_transcript(text)
                bl = blocks(tr["lines"])
                cm = characters_of(tr["mapping"])
                gb = extract_gbnf(sorted(cm))
                plan["window"] = {"chars": len(text), "sha256": sha256_text(text), "matches_sample_chars": len(text) == sample["context_chars"],
                                  "n_lines": len(tr["lines"]), "n_blocks": len(bl), "n_characters": len(cm), "gbnf_chars": len(gb)}
                if len(text) != sample["context_chars"] or not tr["mapping"]:
                    problems.append("window does not match the sample")
            except Exception as e:  # noqa: BLE001
                problems.append(f"window extraction failed: {type(e).__name__}: {e}")
        else:
            plan["arms"] = {"P-large": "planner.plan (native path recorded) + events SQL (bounded grammar, one re-ask) + answer call",
                            "B1": f"SEARCH/READ/FINAL <= {B1_MAX_CALLS} calls, BM25", "RET": f"BM25 top-{RET_TOP_K} chunks, one call (informational)"}
            plan["questions"] = [{"id": q["id"][:8], "type": q["type"]} for q in sample["questions"]]
            plan["checkpoints_after"] = list(CHECKPOINTS)
            plan["wall_cap_s"] = WALL_CAP_S
        print(json.dumps({**plan, **info, "problems": problems, "ok": not problems}, indent=1, default=str))
        return RC_PREFLIGHT if problems else RC_OK
    if a.out is None or not a.url or (a.phase == "answers" and a.ingest_dir is None):
        build_parser().error("--out and --url (and --ingest-dir for answers) are required to run")
    if problems:
        print(json.dumps({"problems": problems}))
        return RC_PRECOND if any("ingest" in p for p in problems) else RC_PREFLIGHT
    a.out.mkdir(parents=True, exist_ok=True)
    rec = P.Recorder(a.out / "results.jsonl")
    smoke = a.smoke or os.environ.get("FILLER_SMOKE") == "1"
    deadline = float(os.environ["FILLER_DEADLINE"]) if os.environ.get("FILLER_DEADLINE") else None
    guard = P.make_guard(deadline, os.environ.get("FILLER_STOPFILE"))
    rec.emit("meta", phase=a.phase, smoke=smoke, predeclared_sha256=PREDECLARED_SHA, orch_pin=a.orch_pin, research_pin=a.research_pin,
             url=a.url, model_id=a.model_id, cfg=dict(vars(P.CFG)), final=False)
    try:
        res = run_ingest(a, rec, sample, smoke, guard) if a.phase == "ingest" else run_answers(a, rec, sample, smoke, guard)
    except P.Stop as e:
        rec.emit("stop", reason=str(e), final=False)
        return RC_PARTIAL
    rc = res.pop("rc", None)
    if a.phase == "answers" and res.get("complete") and not smoke:
        ing = json.loads((a.ingest_dir / "result.cpu.json").read_text())
        res["verdict"] = verdict(rec.records, sample["questions"], futility=res.get("futility"), ingest={k: ing.get(k) for k in ("n_events", "n_rolls", "n_spells", "n_blocks", "calls")})
        rec.emit("verdict", unit="verdict", verdict=res["verdict"]["verdict"], final=True)
        rc = RC_GATE if res.get("futility") else RC_OK
    res.setdefault("phase", "cpu" if a.phase == "ingest" else "gpu")
    if not smoke and res.get("complete") is not None and (a.phase == "answers" or res.get("complete")):
        tmp = a.out / f"result.{res['phase']}.json.tmp"
        tmp.write_text(json.dumps(res, indent=1, default=str) + "\n")
        os.replace(tmp, a.out / f"result.{res['phase']}.json")
    print(json.dumps({k: v for k, v in res.items() if k != "verdict"}, indent=1, default=str))
    if rc is not None:
        return rc
    return RC_OK if (res.get("complete") or smoke) else RC_PARTIAL


if __name__ == "__main__":
    sys.exit(main())
