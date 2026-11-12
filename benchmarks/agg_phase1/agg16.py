#!/usr/bin/env python3
"""AGG-1.6: union-window retrieval and typed multi-field output, an A/B on the AGG-1.11 target cells (library;
``run_agg16.py`` is the CLI).

Code only. Nothing here loads a model, starts a server or signals a process: every model call goes through an injected
backend (``src.aggregation.llm_backend.LlamaServerBackend`` in a keeper job, a fake in tests).

Implements ``PREDECLARED_AGG16.md`` literally (post-hoc and in-sample, as that file says). Arms over the 38 target cells of
``agg111_triage.json`` (the L4-class FORMAT and RETRIEVAL cells, deduplicated):

* **C0**  the saved AGG-1.1 / AGG-1.3 r3 / AGG-1.4 r3 L4 unit of each cell. Never re-run: read from the saved ``results.jsonl``.
* **C1**  the unchanged L4 stage (``facts_llm.LazyFieldBackfill``) over the UNION windows (current top-6 plus a field-specific
          top-3), ``n_predict`` 512.
* **T**   the union windows, one call per distinct window, one typed object with a slot per schema field (at most 2 rows per slot,
          quote at most 100 characters), rows through the unchanged ``facts_llm.extract`` / ``reconcile``.

Phases: ``plan`` (zero model: the window plan and the measured-rate ESTIMATE), ``run`` (CPU: smoke, pilot on 5 cells, the rest,
the metrics, the verdict). Exit codes: 0 complete (or smoke ok), 2 preflight failure, 3 futility stop, 4 precondition, 5 budget stop,
75 partial (resumable).
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

HERE = Path(__file__).resolve().parent
P0DIR = HERE.parent / "agg_phase0"
for _p in (str(P0DIR), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import agg14 as G14  # noqa: E402
import agg_phase0 as A  # noqa: E402
import phase0_arms as P  # noqa: E402
import phase1_arms as X  # noqa: E402
from agg111_triage import FIELD_QUERY  # noqa: E402

# --------------------------------------------------------------------------- pins
PREDECLARED_FILE = "PREDECLARED_AGG16.md"
PREDECLARED_SHA = "c372523a3839834eb6f8a99ce1d67276b8c0b83c6353325fa50a282420c88a66"
TRIAGE_FILE = "agg111_triage.json"
FROZEN = {
    "agg111_triage.json": "f8a2f0ff560ed61d8d97acda193a7a4eb0c607be47e63d435261f05ab4dd4291",
    "agg111_triage.py": "aa9190396e3a7310b9ea48ed69b03e9c2ca4b8dd8fc9308fe96ac49fb5b55ffd",     # FIELD_QUERY, the field-specific retrieval queries
}
FACTS_LLM_SHA = "c64b7c9ed7df88906d38f07e71c5605bcbb4aae66549452ae42487c257dd7451"   # the amended L4 stage (AGG-1.3/1.4 amendment 2)
SHA256 = X.sha256_file

# --------------------------------------------------------------------------- the design (PREDECLARED_AGG16.md)
N_TARGETS, N_L4_FORMAT, N_L4_RETRIEVAL, N_PRIMARY_FORMAT, N_PRIMARY_RETRIEVAL = 38, 21, 17, 14, 13
K_FIELD = 3                                   # field-specific top-3 windows joined to the current top-6
N_PREDICT_C1 = 512
N_PREDICT_T = 1024
T_MAX_ROWS = 2                                # rows per slot
T_QUOTE_CHARS = 100                           # bounded quote length
T_PERIOD_CHARS = 20
T_SLOTS = tuple(A.METRICS) + (G14.FIELD,)     # the 8 schema fields plus long_term_debt
STORE_OF_GROUP = {"a": "agg13", "b": "agg11", "c": "agg11"}
PILOT_CELLS = (                               # exactly 5, fixed in PREDECLARED_AGG16.md
    ("MICROSOFT_2022_10K", "capex", "2022"), ("MICROSOFT_2022_10K", "dividends_paid", "2022"),
    ("COSTCO_2022_10K", "net_income", "2022"), ("COSTCO_2022_10K", "total_assets", "2022"),
    ("COSTCO_2022_10K", "total_current_liabilities", "2022"),
)
PILOT_MIN_FILLED = 2
PILOT_WALL_S = 600
RATE_CALLS_PER_UNIT = 6.0                     # measured (AGG-1.1 / AGG-1.4 receipts): about 6 calls per L4 unit
RATE_S_PER_CALL = 8.2                         # measured (AGG-1.4 r3): about 8.2 s per call
SERVER_START_S = 120.0                        # ESTIMATE: 37.8 GB GGUF from the page cache + warm-up
BUDGET_X = 2.0
TRUNC_MAX = 0.01                              # DONE WHEN: truncated-or-unparseable <= 1 percent
PRIMARY_FORMAT_MIN = 10                       # DONE WHEN: >= 10 of the 14 primary FORMAT cells
RC_OK, RC_PREFLIGHT, RC_GATE, RC_PRECOND, RC_BUDGET, RC_PARTIAL = (X.RC_OK, X.RC_PREFLIGHT, X.RC_GATE, X.RC_PRECOND,
                                                                   X.RC_BUDGET, X.RC_PARTIAL)
FACT_KEEP = ("fact_id", "doc_hash", "doc_id", "entity", "period", "field", "value", "unit", "quote", "page", "chunk_id",
             "confidence", "extractor", "audit_status", "declared_scale", "declared_scale_basis", "unit_override")


# --------------------------------------------------------------------------- inputs and pins
def cell_key(c: dict[str, Any]) -> str:
    return f"{c['doc']}|{c['field']}|{c['fy']}"


def load_targets(set_dir: Path = HERE) -> list[dict[str, Any]]:
    """The 38 target cells: L4 class FORMAT or RETRIEVAL in ``agg111_triage.json``, deduplicated on (document, field,
    fiscal year), ordered by (store, document, field, fiscal year)."""
    tri = json.loads((set_dir / TRIAGE_FILE).read_text())
    seen: dict[str, dict[str, Any]] = {}
    for c in tri["cells"]:
        if c.get("l4_class") not in ("FORMAT", "RETRIEVAL"):
            continue
        t = {"doc": c["doc"], "field": c["field"], "fy": str(c["fy"]), "gold": float(c["gold"]), "group": c["group"],
             "store": STORE_OF_GROUP[c["group"]], "l4_class": c["l4_class"], "primary": c["primary"]}
        seen.setdefault(cell_key(t), t)
    out = sorted(seen.values(), key=lambda t: (t["store"], t["doc"], t["field"], t["fy"]))
    for t in out:
        t["key"] = cell_key(t)
    return out


def targets_problems(targets: list[dict[str, Any]], *, expect: bool = True) -> list[str]:
    out = []
    keys = [t["key"] for t in targets]
    if len(keys) != len(set(keys)):
        out.append("duplicate target cells")
    for t in targets:
        if t["field"] != G14.FIELD and t["field"] not in A.METRICS:
            out.append(f"{t['key']}: no field description")
        if t["fy"] != "2022" and "FY2022" not in (G14.FIELD_DESC if t["field"] == G14.FIELD else A.METRICS[t["field"]]):
            out.append(f"{t['key']}: description cannot be rendered for FY{t['fy']}")
    if expect:
        n = {"n": len(targets), "l4_format": sum(t["l4_class"] == "FORMAT" for t in targets),
             "l4_retrieval": sum(t["l4_class"] == "RETRIEVAL" for t in targets),
             "primary_format": sum(t["primary"] == "FORMAT" for t in targets),
             "primary_retrieval": sum(t["primary"] == "RETRIEVAL" for t in targets)}
        want = {"n": N_TARGETS, "l4_format": N_L4_FORMAT, "l4_retrieval": N_L4_RETRIEVAL, "primary_format": N_PRIMARY_FORMAT,
                "primary_retrieval": N_PRIMARY_RETRIEVAL}
        if n != want:
            out.append(f"target counts {n} != PREDECLARED {want}")
        have = set(keys)
        for k in PILOT_CELLS:
            if "|".join(k) not in have:
                out.append(f"pilot cell {k} is not a target")
        if len(set(PILOT_CELLS)) != 5:
            out.append("the pilot is not 5 distinct cells")
    return out


def pin_problems(set_dir: Path = HERE, *, predeclared_sha: str | None = None, frozen: dict[str, str] | None = None) -> list[str]:
    out = []
    want = PREDECLARED_SHA if predeclared_sha is None else predeclared_sha
    p = set_dir / PREDECLARED_FILE
    got = SHA256(p) if p.exists() else None
    if got != want:
        out.append(f"{PREDECLARED_FILE} sha256 {got} != recorded {want}")
    for name, w in (FROZEN if frozen is None else frozen).items():
        q = set_dir / name
        g = SHA256(q) if q.exists() else None
        if g != w:
            out.append(f"frozen input {name} sha256 {g} != pinned {w}")
    return out


def stage_problems(orch_root: Path) -> list[str]:
    """The L4 stage must be the amended one the PREDECLARED names."""
    f = orch_root / "src" / "aggregation" / "facts_llm.py"
    got = SHA256(f) if f.exists() else None
    return [] if got == FACTS_LLM_SHA else [f"facts_llm.py sha256 {got} != {FACTS_LLM_SHA}"]


def load_inputs(set_dir: Path = HERE) -> dict[str, Any]:
    corpus = json.loads((set_dir / "corpus.json").read_text())
    return {"targets": load_targets(set_dir), "corpus": corpus, "set_dir": set_dir}


def field_desc(field: str, fy: str) -> str:
    base = G14.FIELD_DESC if field == G14.FIELD else A.METRICS[field]
    return X.l4_description(base, fy)


def pilot_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = {t["key"]: t for t in targets}
    return [by["|".join(k)] for k in PILOT_CELLS]


# --------------------------------------------------------------------------- windows and candidates
def mem_store(src: Path, docs: Iterable[str], corpus_id: str = P.CORPUS_ID):
    """A fresh in-memory ``CorpusStore`` holding ONLY the documents and chunks of ``docs`` copied from the read-only seed store
    (``immutable=1``); ``result_json`` is blanked (L4 does not read it). Facts an arm writes never reach the seed."""
    from src.aggregation.store import CorpusStore  # noqa: PLC0415
    st = CorpusStore(":memory:")
    c = sqlite3.connect(f"file:{src}?immutable=1", uri=True)
    try:
        want = sorted(set(docs))
        for row in c.execute(f"SELECT * FROM documents WHERE corpus_id = ? AND doc_id IN ({','.join('?' * len(want))})",
                             [corpus_id] + want).fetchall():
            row = list(row)
            row[-1] = "{}"
            st.conn.execute(f"INSERT INTO documents VALUES ({','.join('?' * len(row))})", row)
            for ch in c.execute("SELECT * FROM chunks WHERE doc_hash = ?", [row[0]]).fetchall():
                st.conn.execute(f"INSERT INTO chunks VALUES ({','.join('?' * len(ch))})", list(ch))
        st.conn.commit()
    finally:
        c.close()
    return st


def doc_windows(store: Any, corpus_id: str = P.CORPUS_ID, window_chars: int = P.CFG.window_chars) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for u in P.windows(store, corpus_id, window_chars):
        out.setdefault(u["doc_id"], []).append(u)
    return out


def current_top(wins: list[dict[str, Any]], field: str, fy: str, k: int = P.CFG.k_per_doc) -> list[dict[str, Any]]:
    return P.bm25_top(wins, f"{field.replace('_', ' ')} {field_desc(field, fy)}", k)


def field_top(wins: list[dict[str, Any]], field: str, k: int = K_FIELD) -> list[dict[str, Any]]:
    return P.bm25_top(wins, FIELD_QUERY[field], k)


def union_candidates(wins: list[dict[str, Any]], field: str, fy: str) -> list[dict[str, Any]]:
    """The current top-6 (sims from the run's query) followed by the field-specific top-3 windows not already present (sims
    normalised to the field query's best window). The same list the AGG-1.11 retrieval probe scored (``cur + fq[:3]``)."""
    cur = current_top(wins, field, fy)
    have = {w["chunk_id"] for w in cur}
    return cur + [w for w in field_top(wins, field) if w["chunk_id"] not in have]


def admission_counts(cands: list[dict[str, Any]], hi: float = P.CFG.hi, lo: float = P.CFG.lo) -> dict[str, int]:
    return {"windows": len(cands), "admit_sim": sum(w["sim"] >= hi for w in cands),
            "gated": sum(lo < w["sim"] < hi for w in cands), "rejected_sim": sum(w["sim"] <= lo for w in cands)}


# --------------------------------------------------------------------------- recording backend
class RecordingBackend:
    """Wraps a ``facts_llm.Backend`` and records one entry per call (stage, finish_reason, tokens). Counters delegate to the
    wrapped backend when it has them (``LlamaServerBackend``) and are kept here otherwise (test fakes)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.log: list[dict[str, Any]] = []

    @property
    def calls(self) -> int:
        return len(self.log)

    @property
    def prefill_tokens(self) -> int:
        return sum(e["prompt_tokens"] for e in self.log)

    def mark(self) -> int:
        return len(self.log)

    def since(self, mark: int) -> list[dict[str, Any]]:
        return self.log[mark:]

    def complete(self, prompt: str, *, grammar: str, n_predict: int, cache_prompt: bool, id_slot: int | None) -> dict[str, Any]:
        t0 = time.time()
        r = self.inner.complete(prompt, grammar=grammar, n_predict=n_predict, cache_prompt=cache_prompt, id_slot=id_slot)
        self.log.append({"stage": "gate" if n_predict == 1 else "extract", "prompt_tokens": int(r.get("prompt_tokens") or 0),
                         "cache_n": int(r.get("cache_n") or 0), "decode_tokens": int(r.get("decode_tokens") or 0),
                         "finish_reason": r.get("finish_reason"), "elapsed_s": float(r.get("elapsed_s") or (time.time() - t0)),
                         "n_predict": n_predict})
        return r


def slim(f: dict[str, Any]) -> dict[str, Any]:
    return {k: f[k] for k in FACT_KEEP if k in f}


# --------------------------------------------------------------------------- scoring (pure)
def cell_hit(facts: Iterable[dict[str, Any]], cell: dict[str, Any], rel_tol: float = P.CFG.rel_tol) -> bool:
    """The AGG-1.1 M1 hit rule for one cell and fiscal year: a fact of the same document and field whose period names the cell's
    fiscal year, value within ``rel_tol`` of the gold after the USD unit scale (millions = 1), no leniency for a 1000x slip."""
    g = cell["gold"]
    for f in facts:
        if f.get("field") != cell["field"] or f.get("doc_id") != cell["doc"] or f.get("value") is None:
            continue
        if cell["fy"] not in str(f.get("period") or ""):
            continue
        v = float(f["value"]) * X.M1R_UNIT_SCALE.get(f.get("unit"), 1.0)
        if abs(abs(v) - abs(g)) <= rel_tol * abs(g) + 1e-9:
            return True
    return False


def calls_stats(entries: list[dict[str, Any]]) -> dict[str, int]:
    ext = [e for e in entries if e["stage"] == "extract"]
    return {"calls": len(entries), "gate_calls": len(entries) - len(ext), "extract_calls": len(ext),
            "prefill_tokens": sum(e["prompt_tokens"] for e in entries), "cache_n": sum(e["cache_n"] for e in entries),
            "decode_tokens": sum(e["decode_tokens"] for e in entries),
            "cut": sum(e.get("finish_reason") == "length" for e in ext)}


def rate(num: int, den: int) -> float | None:
    return (num / den) if den else None


# --------------------------------------------------------------------------- C1: the unchanged L4 stage over the union windows
def run_c1_cell(rec: P.Recorder, st: Any, rb: RecordingBackend, wins: dict[str, list[dict[str, Any]]], cell: dict[str, Any],
                guard: Callable[[], None], cfg: P.Cfg = P.CFG) -> bool:
    from src.aggregation import facts_llm  # noqa: PLC0415
    unit = f"c1:{cell['key']}"
    if unit in rec.done:
        return False
    guard()
    t0 = time.time()
    cands = union_candidates(wins[cell["doc"]], cell["field"], cell["fy"])
    bf = facts_llm.LazyFieldBackfill(st, rb, n_predict=N_PREDICT_C1, id_slot=None, hi=cfg.hi, lo=cfg.lo)
    mark = rb.mark()
    try:
        out = bf.fill(P.CORPUS_ID, cell["field"], field_desc(cell["field"], cell["fy"]), cands)
        status = "ok"
    except Exception as e:  # noqa: BLE001 - recorded per cell, the run continues
        out, status = {"facts": [], "receipts": []}, f"error:{type(e).__name__}:{str(e)[:200]}"
    cs = calls_stats(rb.since(mark))
    notes = [n for r in out.get("receipts") or [] for n in (r.get("notes") or [])]
    unparse = sum(str(n).startswith("unparseable:") for n in notes)
    facts = [slim(f) for f in out["facts"]]
    rec.emit("arm_cell", unit=unit, arm="C1", cell=cell["key"], store=cell["store"], doc=cell["doc"], field=cell["field"],
             fy=cell["fy"], primary=cell["primary"], l4_class=cell["l4_class"], status=status, ok=status == "ok",
             n_windows=len(cands), **admission_counts(cands), **cs, unparseable=unparse,
             bad_replies=_bad(cs["extract_calls"], cs["cut"], unparse), wall_s=time.time() - t0, hit=cell_hit(facts, cell),
             facts=facts, final=status == "ok")
    return True


def _bad(extract_calls: int, cut: int, unparseable: int) -> int:
    """Extraction replies that were truncated or unparseable. A cut reply is normally also unparseable: take the larger, never
    the sum, so one bad reply is counted once."""
    return min(extract_calls, max(cut, unparseable))


# --------------------------------------------------------------------------- T: window-keyed typed multi-field output
def t_grammar(slots: tuple[str, ...] = T_SLOTS) -> str:
    """GBNF for the typed object: one key per slot, each an array of at most ``T_MAX_ROWS`` rows, quote bounded to
    ``T_QUOTE_CHARS`` characters, period to ``T_PERIOD_CHARS``, whitespace between tokens at most one space."""
    row = ('row ::= "{" ws "\\"period\\":" ws pstr "," ws "\\"value\\":" ws num "," ws "\\"unit\\":" ws unit "," ws '
           '"\\"quote\\":" ws qstr "," ws "\\"confidence\\":" ws conf ws "}"')
    rows = 'rows ::= "[" ws ( row ( "," ws row )? )? ws "]"' if T_MAX_ROWS == 2 else None
    if rows is None:
        raise ValueError("t_grammar is written for T_MAX_ROWS == 2")
    keys = ' "," ws '.join(f'"\\"{s}\\":" ws rows' for s in slots)
    lines = [
        f'root ::= "{{" ws {keys} ws "}}"',
        rows,
        row,
        'unit ::= "\\"USD_thousands\\"" | "\\"USD_millions\\"" | "\\"USD_billions\\"" | "\\"USD\\"" | "\\"percent\\"" | "\\"ratio\\"" | "\\"count\\""',
        'conf ::= "\\"high\\"" | "\\"medium\\"" | "\\"low\\""',
        f'pstr ::= "\\"" pchar{{1,{T_PERIOD_CHARS}}} "\\""',
        'pchar ::= [^"\\\\]',
        f'qstr ::= "\\"" qchar{{1,{T_QUOTE_CHARS}}} "\\""',
        'qchar ::= [^"\\\\] | "\\\\" ["\\\\/bfnrt]',
        'num ::= "-"? [0-9]+ ( "." [0-9]+ )?',
        'ws ::= [ ]?',
    ]
    return "\n".join(lines)


def t_prompt(window_text: str, fy: str, slots: tuple[str, ...] = T_SLOTS) -> str:
    fields = "\n".join(f"- {s}: {field_desc(s, fy)}" for s in slots)
    return (
        "You extract financial fields from a filing excerpt.\n"
        f"Fields (return every key; use [] when the excerpt does not state the field):\n{fields}\n"
        f"Return ONE JSON object with one key per field. Each value is an array of at most {T_MAX_ROWS} rows "
        "{period, value, unit, quote, confidence}. `quote` is copied verbatim from the excerpt and is at most "
        f"{T_QUOTE_CHARS} characters (the label and the number).\n"
        "Excerpt:\n" + window_text)


class _Replay:
    """A one-reply backend: lets the unchanged ``facts_llm.extract`` read a slot's rows (parse, quote-in-chunk check, declared
    scale, fact ids) exactly as it reads an L4b reply."""

    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, prompt: str, **_kw: Any) -> dict[str, Any]:
        return {"text": self.text, "prompt_tokens": 0, "cache_n": 0}


def parse_typed(text: str, slots: tuple[str, ...] = T_SLOTS) -> dict[str, list[Any]] | None:
    """The typed object as ``{slot: rows}``, or None when the reply is not a JSON object with a list for every slot."""
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or any(not isinstance(obj.get(s), list) for s in slots):
        return None
    return {s: obj[s] for s in slots}


def typed_window_facts(chunk: dict[str, Any], obj: dict[str, list[Any]], fy: str, bf: Any,
                       slots: tuple[str, ...] = T_SLOTS) -> list[dict[str, Any]]:
    """Facts of one window's typed reply through the unchanged L4b fact construction, one slot at a time, then the unchanged
    parent-chunk declared-scale lookup."""
    from src.aggregation import facts_llm  # noqa: PLC0415
    facts: list[dict[str, Any]] = []
    for s in slots:
        if not obj[s]:
            continue
        r = facts_llm.StageReceipt("L4b_typed")
        facts.extend(facts_llm.extract(chunk, s, field_desc(s, fy), _Replay(json.dumps(obj[s])), n_predict=0, id_slot=None,
                                       receipt=r))
    bf._declared_from_parent_chunk(facts)
    return facts


def t_window_unit(store: str, doc: str, fy: str, chunk_id: str) -> str:
    return f"tw:{store}:{doc}:fy{fy}:{chunk_id}"


def run_t_cell(rec: P.Recorder, st: Any, rb: RecordingBackend, wins: dict[str, list[dict[str, Any]]], cell: dict[str, Any],
               guard: Callable[[], None], cfg: P.Cfg = P.CFG) -> bool:
    from src.aggregation import facts_llm  # noqa: PLC0415
    unit = f"t:{cell['key']}"
    if unit in rec.done:
        return False
    t0 = time.time()
    cands = union_candidates(wins[cell["doc"]], cell["field"], cell["fy"])
    bf = facts_llm.LazyFieldBackfill(st, rb, n_predict=N_PREDICT_T, id_slot=None, hi=cfg.hi, lo=cfg.lo)
    gram = t_grammar()
    by_unit = {r["unit"]: r for r in rec.records if r.get("kind") == "t_window" and r.get("final")}
    fresh: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for w in cands:
        wu = t_window_unit(cell["store"], cell["doc"], cell["fy"], w["chunk_id"])
        r = by_unit.get(wu)
        if r is None:
            guard()
            mark = rb.mark()
            status = "ok"
            try:
                reply = rb.complete(t_prompt(w["text"], cell["fy"]), grammar=gram, n_predict=N_PREDICT_T, cache_prompt=True,
                                    id_slot=None)
            except Exception as e:  # noqa: BLE001
                reply, status = {"text": ""}, f"error:{type(e).__name__}:{str(e)[:200]}"
            entries = rb.since(mark)
            cs = calls_stats(entries)
            obj = parse_typed(reply.get("text") or "") if status == "ok" else None
            facts = [slim(f) for f in typed_window_facts(w, obj, cell["fy"], bf)] if obj is not None else []
            parsed = obj is not None
            cut = cs["cut"] > 0
            r = rec.emit("t_window", unit=wu, store=cell["store"], doc=cell["doc"], fy=cell["fy"], chunk_id=w["chunk_id"],
                         status=status, parsed=parsed, bad=bool(cut or not parsed) and status == "ok",
                         n_rows={s: len(obj[s]) for s in T_SLOTS} if obj else None, **cs, facts=facts,
                         final=status == "ok")
            by_unit[wu] = r
            if status == "ok":
                fresh.append(r)
        raw.extend(r.get("facts") or [])
    kept, _conf = facts_llm.reconcile(raw)
    sums = {k: sum(int(r.get(k) or 0) for r in fresh) for k in ("calls", "prefill_tokens", "cache_n", "decode_tokens")}
    ok = all(by_unit.get(t_window_unit(cell["store"], cell["doc"], cell["fy"], w["chunk_id"]), {}).get("final") for w in cands)
    rec.emit("arm_cell", unit=unit, arm="T", cell=cell["key"], store=cell["store"], doc=cell["doc"], field=cell["field"],
             fy=cell["fy"], primary=cell["primary"], l4_class=cell["l4_class"], status="ok" if ok else "error:window",
             ok=ok, n_windows=len(cands), fresh_windows=len(fresh), **sums, gate_calls=0,
             extract_calls=sums["calls"], bad_replies=sum(1 for r in fresh if r.get("bad")),
             cut=sum(1 for r in fresh if r.get("cut")), unparseable=sum(1 for r in fresh if not r.get("parsed")),
             wall_s=time.time() - t0, hit=cell_hit(kept, cell), facts=[slim(f) for f in kept], final=ok)
    return True


# --------------------------------------------------------------------------- C0: the saved reference units
def c0_unit(results: list[dict[str, Any]], field: str, doc: str) -> dict[str, Any] | None:
    ok = X.l4_ok(results)
    return ok.get(f"l4:{field}:{doc}")


def c0_summary(targets: list[dict[str, Any]], results_by_group: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Calls, prefill, extraction calls and ``unparseable`` receipt notes of each target's saved L4 unit. The target fill is 0
    by construction (the triage class is by the L4 pipeline alone); ``missing`` lists targets with no saved unit."""
    per: dict[str, Any] = {}
    missing = []
    for t in targets:
        r = c0_unit(results_by_group.get(t["group"], []), t["field"], t["doc"])
        if r is None:
            missing.append(t["key"])
            continue
        rc = r.get("receipts") or []
        l4b = next((x for x in rc if x.get("stage") == "L4b_extract"), {})
        l4a = next((x for x in rc if x.get("stage") == "L4a_admit"), {})
        unp = sum(str(n).startswith("unparseable:") for n in (l4b.get("notes") or []))
        per[t["key"]] = {"calls": int(r.get("llm_calls") or 0), "prefill_tokens": int(r.get("prefill_tokens") or 0),
                         "extract_calls": int(l4b.get("calls") or 0), "gate_calls": int(l4a.get("calls") or 0),
                         "unparseable": unp, "reused": bool(r.get("reused")), "n_facts": r.get("n_facts")}
    n = len(per)
    calls = sum(v["calls"] for v in per.values())
    ext = sum(v["extract_calls"] for v in per.values())
    unp = sum(v["unparseable"] for v in per.values())
    return {"arm": "C0", "n_units": n, "missing": missing, "calls": calls, "prefill_tokens": sum(v["prefill_tokens"] for v in per.values()),
            "extract_calls": ext, "unparseable": unp, "bad_rate": rate(unp, ext), "calls_per_unit": rate(calls, n),
            "prefill_per_unit": rate(sum(v["prefill_tokens"] for v in per.values()), n), "fill": 0, "fill_primary_format": 0,
            "fill_primary_retrieval": 0, "fill_by": "construction (agg111_triage l4_class is by the L4 pipeline alone)", "per_cell": per}


# --------------------------------------------------------------------------- arm summaries (pure)
def arm_cells(records: list[dict[str, Any]], arm: str) -> dict[str, dict[str, Any]]:
    return {r["cell"]: r for r in records if r.get("kind") == "arm_cell" and r.get("arm") == arm and r.get("final")}


def arm_summary(records: list[dict[str, Any]], arm: str, targets: list[dict[str, Any]]) -> dict[str, Any]:
    cells = arm_cells(records, arm)
    done = [t for t in targets if t["key"] in cells]
    n = len(done)

    def tot(k: str) -> int:
        return sum(int(cells[t["key"]].get(k) or 0) for t in done)

    def fill(pred: Callable[[dict[str, Any]], bool]) -> dict[str, int]:
        sel = [t for t in done if pred(t)]
        return {"cells": len(sel), "filled": sum(1 for t in sel if cells[t["key"]].get("hit"))}

    ext = tot("extract_calls")
    bad = tot("bad_replies")
    return {"arm": arm, "n_units": n, "calls": tot("calls"), "gate_calls": tot("gate_calls"), "extract_calls": ext,
            "prefill_tokens": tot("prefill_tokens"), "cache_n": tot("cache_n"), "decode_tokens": tot("decode_tokens"),
            "bad_replies": bad, "cut": tot("cut"), "unparseable": tot("unparseable"), "bad_rate": rate(bad, ext),
            "calls_per_unit": rate(tot("calls"), n), "prefill_per_unit": rate(tot("prefill_tokens"), n),
            "wall_s": sum(float(cells[t["key"]].get("wall_s") or 0) for t in done),
            "fill": fill(lambda t: True), "fill_primary_format": fill(lambda t: t["primary"] == "FORMAT"),
            "fill_primary_retrieval": fill(lambda t: t["primary"] == "RETRIEVAL"),
            "fill_primary_l3": fill(lambda t: t["primary"] == "L3"),
            "fill_l4_format": fill(lambda t: t["l4_class"] == "FORMAT"),
            "fill_l4_retrieval": fill(lambda t: t["l4_class"] == "RETRIEVAL"),
            "filled_cells": sorted(t["key"] for t in done if cells[t["key"]].get("hit")),
            "unfilled_cells": sorted(t["key"] for t in done if not cells[t["key"]].get("hit"))}


def verdict(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """DONE WHEN (PREDECLARED_AGG16.md): the winning arm (most primary-FORMAT cells filled; ties: lower bad rate, then fewer calls
    per unit) has a truncated-or-unparseable rate <= 1 percent AND >= 10 of the 14 primary FORMAT cells filled."""
    arms = [a for a in ("C1", "T") if a in summaries and summaries[a]["n_units"] > 0]
    if not arms:
        return {"met": False, "winner": None, "why": "no arm ran"}

    def key(a: str) -> tuple:
        s = summaries[a]
        return (-s["fill_primary_format"]["filled"], s["bad_rate"] if s["bad_rate"] is not None else 1.0,
                s["calls_per_unit"] if s["calls_per_unit"] is not None else 1e9)

    win = min(arms, key=key)
    s = summaries[win]
    ok_rate = s["bad_rate"] is not None and s["bad_rate"] <= TRUNC_MAX
    ok_fill = s["fill_primary_format"]["filled"] >= PRIMARY_FORMAT_MIN
    return {"met": bool(ok_rate and ok_fill), "winner": win, "bad_rate": s["bad_rate"], "bad_rate_ok": ok_rate,
            "bad_rate_max": TRUNC_MAX, "primary_format_filled": s["fill_primary_format"]["filled"],
            "primary_format_cells": s["fill_primary_format"]["cells"], "primary_format_ok": ok_fill,
            "primary_format_min": PRIMARY_FORMAT_MIN, "complete": s["n_units"] == N_TARGETS,
            "arms_compared": {a: {"primary_format_filled": summaries[a]["fill_primary_format"]["filled"],
                                  "bad_rate": summaries[a]["bad_rate"], "calls_per_unit": summaries[a]["calls_per_unit"]}
                              for a in arms}}


# --------------------------------------------------------------------------- non-target cells touched (canonical.v1)
def _close(a: float, b: float, tol: float = P.CFG.rel_tol) -> bool:
    return abs(abs(a) - abs(b)) <= tol * abs(b) + 1e-9


def canon_ok_fy(canon: list[dict[str, Any]], cells: list[dict[str, Any]]) -> dict[str, bool]:
    by = {(r["doc_id"], r["field"], str(r["period"])): r["value"] for r in canon}
    return {c["key"]: (by.get((c["doc"], c["field"], c["fy"])) is not None and _close(by[(c["doc"], c["field"], c["fy"])], c["gold"]))
            for c in cells}


def touched_effects(saved_rows: list[dict[str, Any]], arm_facts: list[dict[str, Any]], universe: list[dict[str, Any]],
                    target_keys: set[str], corpus_id: str = P.CORPUS_ID) -> dict[str, Any]:
    """Fixes and breaks on the non-target ``universe`` cells the arm touched (it produced a fact for the document, field and fiscal
    year), under ``canonical.v1``: the saved facts alone versus the saved facts plus the arm's facts."""
    from src.aggregation.store import select_canonical  # noqa: PLC0415
    non = [c for c in universe if c["key"] not in target_keys]
    af = [dict(f, corpus_id=corpus_id) for f in arm_facts if f.get("value") is not None]
    touched = [c for c in non if any(f.get("doc_id") == c["doc"] and f.get("field") == c["field"] and c["fy"] in str(f.get("period") or "")
                                     for f in af)]
    before = canon_ok_fy(select_canonical(saved_rows)[0], touched)
    after = canon_ok_fy(select_canonical(saved_rows + af)[0], touched)
    fixed = sorted(k for k in after if after[k] and not before[k])
    broken = sorted(k for k in before if before[k] and not after[k])
    return {"non_target_cells": len(non), "touched": len(touched), "touched_cells": sorted(c["key"] for c in touched),
            "fixed": fixed, "broken": broken, "n_fixed": len(fixed), "n_broken": len(broken),
            "ok_before": sum(before.values()), "ok_after": sum(after.values())}


# --------------------------------------------------------------------------- the plan (zero model)
def plan_cells(targets: list[dict[str, Any]], wins_by_store: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """The offline window plan: per target the union size and admission split, and the distinct windows T would call."""
    per: dict[str, Any] = {}
    distinct: set[tuple[str, str, str, str]] = set()
    c1_windows = 0
    c1_extract_upper = 0
    for t in targets:
        cands = union_candidates(wins_by_store[t["store"]][t["doc"]], t["field"], t["fy"])
        ac = admission_counts(cands)
        per[t["key"]] = ac
        c1_windows += ac["windows"]
        c1_extract_upper += ac["admit_sim"] + ac["gated"]
        for w in cands:
            distinct.add((t["store"], t["doc"], t["fy"], w["chunk_id"]))
    return {"per_cell": per, "c1_union_windows": c1_windows, "c1_extract_calls_upper_bound": c1_extract_upper,
            "c1_gate_calls": sum(v["gated"] for v in per.values()), "t_distinct_windows": len(distinct),
            "t_calls": len(distinct)}


def estimates(plan: dict[str, Any], n_targets: int = N_TARGETS) -> dict[str, Any]:
    """Wall estimates at the MEASURED rates (about 6 calls per unit, about 8.2 s per call). ESTIMATES, not measurements: the
    run replaces them. T's calls are the offline distinct-window count; its per-call cost is taken at the same rate although the
    typed reply decodes longer."""
    c1_calls = n_targets * RATE_CALLS_PER_UNIT
    c1_s = c1_calls * RATE_S_PER_CALL
    t_s = plan["t_calls"] * RATE_S_PER_CALL
    return {"label": "ESTIMATE", "rates": {"calls_per_unit": RATE_CALLS_PER_UNIT, "s_per_call": RATE_S_PER_CALL},
            "c1_calls": c1_calls, "c1_s": round(c1_s, 1), "t_calls": plan["t_calls"], "t_s": round(t_s, 1),
            "server_start_s": SERVER_START_S, "total_s": round(c1_s + t_s + SERVER_START_S, 1)}


# --------------------------------------------------------------------------- the run
def _result(out: Path, **res: Any) -> dict[str, Any]:
    res.setdefault("phase", "cpu")
    res["predeclared_sha256"] = PREDECLARED_SHA
    X._atomic_write(out / "result.cpu.json", json.dumps(res, indent=1, default=str) + "\n")
    return res


def pilot_eval(records: list[dict[str, Any]], pilot: list[dict[str, Any]], *, booked_s: float) -> dict[str, Any]:
    """Pure arithmetic over the pilot cells' records: fill per arm, continuation, wall, budget."""
    keys = {t["key"] for t in pilot}
    out: dict[str, Any] = {"cells": sorted(keys), "arms": {}, "budget_stops": []}
    wall = 0.0
    for arm in ("C1", "T"):
        cs = {k: r for k, r in arm_cells(records, arm).items() if k in keys}
        filled = sum(1 for r in cs.values() if r.get("hit"))
        w = sum(float(r.get("wall_s") or 0) for r in cs.values())
        wall += w
        out["arms"][arm] = {"done": len(cs), "filled": filled, "wall_s": round(w, 1), "continue": filled >= PILOT_MIN_FILLED,
                            "calls": sum(int(r.get("calls") or 0) for r in cs.values())}
    out["wall_s"] = round(wall, 1)
    out["projected_wall_s"] = round(wall * N_TARGETS / len(keys), 1)
    if wall > PILOT_WALL_S:
        out["budget_stops"].append("pilot_wall_over_600s")
    if out["projected_wall_s"] > BUDGET_X * booked_s:
        out["budget_stops"].append("projected_wall_over_2x_booked")
    out["futility"] = not any(a["continue"] for a in out["arms"].values())
    return out


def run_ab(rec: P.Recorder, out: Path, inp: dict[str, Any], backend: Any, *, seeds: dict[str, Path],
           c0_results: dict[str, Path], guard: Callable[[], None], booked_s: float, smoke: bool = False,
           cfg: P.Cfg = P.CFG) -> dict[str, Any]:
    targets = inp["targets"]
    rb = RecordingBackend(backend)
    stores: dict[tuple[str, str], Any] = {}
    wins: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for arm in ("C1", "T"):
        for name, path in seeds.items():
            docs = {t["doc"] for t in targets if t["store"] == name}
            if docs:
                stores[(arm, name)] = mem_store(path, docs)
                wins[(arm, name)] = doc_windows(stores[(arm, name)], P.CORPUS_ID, cfg.window_chars)

    def c1(cell: dict[str, Any]) -> None:
        run_c1_cell(rec, stores[("C1", cell["store"])], rb, wins[("C1", cell["store"])], cell, guard, cfg)

    def tt(cell: dict[str, Any]) -> None:
        run_t_cell(rec, stores[("T", cell["store"])], rb, wins[("T", cell["store"])], cell, guard, cfg)

    pilot = pilot_targets(targets)
    if smoke:
        c1(pilot[0])
        return {"complete": False, "smoke": True, "rc": RC_OK}
    for cell in pilot:
        c1(cell)
    for cell in pilot:
        tt(cell)
    if "pilot_done" not in rec.done:
        ev = pilot_eval(rec.records, pilot, booked_s=booked_s)
        rec.emit("pilot_done", unit="pilot_done", **ev)
        if ev["futility"]:
            rec.emit("stop", reason="futility_pilot", final=False, pilot=ev)
            return _result(out, complete=True, futility_stop="pilot", verdict="phase1_gate_tripped", pilot=ev, rc=RC_GATE)
        if ev["budget_stops"]:
            rec.emit("stop", reason="budget_stop:" + "+".join(ev["budget_stops"]), final=False, pilot=ev)
            return {"complete": False, "budget_stop": "+".join(ev["budget_stops"]), "rc": RC_BUDGET, **ev}
    pd = next(r for r in rec.records if r.get("unit") == "pilot_done")
    cont = {a: bool(pd["arms"][a]["continue"]) for a in ("C1", "T")}
    if not any(cont.values()):                      # a re-grant after a futility stop files the same verdict again
        return _result(out, complete=True, futility_stop="pilot", verdict="phase1_gate_tripped", pilot=pd, rc=RC_GATE)
    rest = [t for t in targets if t["key"] not in {p["key"] for p in pilot}]
    for cell in rest:
        if cont["C1"]:
            c1(cell)
    for cell in rest:
        if cont["T"]:
            tt(cell)
    # an arm that stopped at the pilot reports its pilot cells only
    for arm, fn in (("C1", c1), ("T", tt)):
        if cont[arm]:
            missing = [t for t in targets if t["key"] not in arm_cells(rec.records, arm)]
            if missing:
                return {"complete": False, "reason": f"{arm}: {len(missing)} cells errored; re-grant retries them", "rc": RC_PARTIAL}
    return finish(rec, out, inp, seeds=seeds, c0_results=c0_results, cont=cont, pilot_ev=pd)


def read_results(paths: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    return {g: X.read_jsonl(p) for g, p in paths.items() if p.exists()}


def saved_rows(seed: Path, corpus_id: str = P.CORPUS_ID) -> list[dict[str, Any]]:
    from src.aggregation.store import FACT_COLUMNS  # noqa: PLC0415
    c = sqlite3.connect(f"file:{seed}?immutable=1", uri=True)
    try:
        cur = c.execute(f"SELECT d.corpus_id, {', '.join('f.' + k for k in FACT_COLUMNS)} FROM facts f JOIN documents d "
                        "ON f.doc_hash = d.doc_hash WHERE d.corpus_id = ?", [corpus_id])
        return [dict(zip(("corpus_id",) + FACT_COLUMNS, r)) for r in cur.fetchall()]
    finally:
        c.close()


def universes(set_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Verified gold cells per seed store: agg11 = the 211 FY2022 cells; agg13 = the filled FY2021 cells of the three replaced
    documents (``agg13_cells.csv``)."""
    import csv  # noqa: PLC0415
    import agg19 as G19  # noqa: PLC0415
    inp19 = G19.load_inputs(set_dir)
    u11 = [{"doc": d, "field": f, "fy": "2022", "gold": g, "key": f"{d}|{f}|2022"} for d, f, g in inp19["cells211"]]
    u13 = []
    with (set_dir / "agg13_cells.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["status"] == "filled" and r["value_usd_m"].strip():
                u13.append({"doc": r["doc_name"], "field": r["metric"], "fy": str(r["fiscal_year"]),
                            "gold": float(r["value_usd_m"].replace(",", "")), "key": f"{r['doc_name']}|{r['metric']}|{r['fiscal_year']}"})
    return {"agg11": u11, "agg13": u13}


def arm_facts_by_store(records: list[dict[str, Any]], arm: str) -> dict[str, list[dict[str, Any]]]:
    """C1: the facts of its cell records. T: every window's raw facts (deduplicated by window) reconciled per document."""
    from src.aggregation import facts_llm  # noqa: PLC0415
    out: dict[str, list[dict[str, Any]]] = {}
    if arm == "C1":
        for r in arm_cells(records, "C1").values():
            out.setdefault(r["store"], []).extend(r.get("facts") or [])
        return out
    by_doc: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in records:
        if r.get("kind") == "t_window" and r.get("final"):
            by_doc.setdefault((r["store"], r["doc"], r["fy"]), []).extend(r.get("facts") or [])
    for (store, _doc, _fy), raw in by_doc.items():
        kept, _c = facts_llm.reconcile(raw)
        out.setdefault(store, []).extend(slim(f) for f in kept)
    return out


def finish(rec: P.Recorder, out: Path, inp: dict[str, Any], *, seeds: dict[str, Path], c0_results: dict[str, Path],
           cont: dict[str, bool], pilot_ev: dict[str, Any]) -> dict[str, Any]:
    targets = inp["targets"]
    summaries = {"C0": c0_summary(targets, read_results(c0_results)),
                 "C1": arm_summary(rec.records, "C1", targets), "T": arm_summary(rec.records, "T", targets)}
    uni = universes(inp["set_dir"])
    tkeys = {t["key"] for t in targets}
    effects = {}
    for arm in ("C1", "T"):
        per_store = arm_facts_by_store(rec.records, arm)
        effects[arm] = {name: touched_effects(saved_rows(seeds[name]), per_store.get(name, []), uni[name], tkeys)
                        for name in seeds}
        effects[arm]["n_fixed"] = sum(v["n_fixed"] for v in effects[arm].values() if isinstance(v, dict))
        effects[arm]["n_broken"] = sum(v["n_broken"] for v in effects[arm].values() if isinstance(v, dict))
    v = verdict(summaries)
    rec.emit("summary", unit="summary", summaries={k: {kk: vv for kk, vv in s.items() if kk != "per_cell"} for k, s in summaries.items()},
             effects=effects, verdict=v)
    return _result(out, complete=True, verdict="agg_1_6_met" if v["met"] else "agg_1_6_not_met", done_when=v, summaries=summaries,
                   non_target_effects=effects, pilot=pilot_ev, continued=cont, post_hoc=True, in_sample=True, rc=RC_OK)
