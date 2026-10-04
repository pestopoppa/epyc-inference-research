"""The paired coherence gate: one verdict per item, the deciding tier recorded.

    tier 0  paired byte-identity. Candidate vs base (anchor) on the same prompt, greedy decoding
            (asserted by the caller). Identical -> PASS. Identity = equal sha256 of the full output
            text AND equal token ids when both arms carry them. A row's own `sha256` is verified
            against its text, never trusted in place of it.
    tier 1  deterministic, for items that diverged:
            (a) ground-truth answer checks (answers.py), (b) degeneracy.v2 (degeneracy.py) on real
            token ids (tokens.py refuses synthetic ones).
            PAIRED: a REGRESSION is the candidate doing worse than base on the same item: a higher
            degeneracy severity, or correct -> wrong/unanswered. Items bad in both arms are BOTH_BAD:
            reported, not failed. A decisive answer check that is correct (or improved) is PASS.
    tier 2  `judge_fn(item_pair) -> verdict`, called ONLY for items that diverged at tier 0, did not
            regress at tier 1 and have no decisive ground truth. Injected by the caller; with no judge
            such items are NEEDS_REVIEW ("unjudged").

Gate: FAIL if any REGRESSION; else INCOMPLETE if any NEEDS_REVIEW (or n == 0); else PASS.
Every item record carries the full texts (or their paths + sha256): no verdict without the material.
"""
from __future__ import annotations

import collections
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from . import answers as A
from . import degeneracy as D
from . import tokens as T

SCHEMA_ID = "epyc.coherence_gate.v1"
LIBRARY_VERSION = "coherence_gate 1.0.0"
VERDICTS = ("PASS", "REGRESSION", "BOTH_BAD", "NEEDS_REVIEW")

JudgeFn = Callable[[dict], Any]


class GateInputError(ValueError):
    """Rows are unusable: no text, sha256 mismatch, duplicate ids, missing id."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Output:
    id: str
    text: str                    # canonical full output (what identity and degeneracy read)
    answer_text: str             # what graders read (content without reasoning when given separately)
    sha256: str
    ref: dict                    # how the record stores the text: {"text": ...} or {"path": ...}
    token_ids: list | None
    finish: str | None
    http_ok: bool
    prompt: str | None = None
    extra: dict = field(default_factory=dict)


def normalize_row(row: dict, base_dir: Path | None = None) -> Output:
    """Accepted shapes: {text} | {content, reasoning?} | {text_path}; optional token_ids|tokens,
    finish|finish_reason|stop_type, http_ok|http|error, sha256|output_sha256, answer_text, prompt.
    Canonical text = `text`, else `reasoning + "\\n" + content` (content alone without reasoning)."""
    if not isinstance(row, dict) or not isinstance(row.get("id"), (str, int)):
        raise GateInputError(f"row without an id: {str(row)[:120]}")
    rid = str(row["id"])
    ref: dict
    if "text" in row and row["text"] is not None:
        text = str(row["text"])
        answer_text = row.get("answer_text", text)
        ref = {"text": text}
    elif "content" in row or "reasoning" in row:
        content, reasoning = row.get("content") or "", row.get("reasoning") or ""
        text = f"{reasoning}\n{content}" if reasoning else content
        answer_text = row.get("answer_text", content)
        ref = {"text": text}
    elif row.get("text_path"):
        p = Path(row["text_path"])
        if not p.is_absolute() and base_dir is not None:
            p = base_dir / p
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as e:
            raise GateInputError(f"{rid}: text_path unreadable: {e}") from e
        answer_text = row.get("answer_text", text)
        ref = {"path": str(p)}
    else:
        raise GateInputError(f"{rid}: no output text (text / content / text_path). A sha256 alone is "
                             f"not enough: the gate never decides without the material it decides on")
    digest = sha256_text(text)
    claimed = row.get("sha256") or row.get("output_sha256")
    if claimed and claimed != digest:
        raise GateInputError(f"{rid}: row sha256 {claimed[:16]}... does not match its text {digest[:16]}...")
    ids = row.get("token_ids", row.get("tokens"))
    finish = row.get("finish", row.get("finish_reason", row.get("stop_type")))
    http = row.get("http")
    http_ok = bool(row.get("http_ok", True)) and not row.get("error") and http in (None, 200)
    return Output(id=rid, text=text, answer_text=str(answer_text), sha256=digest, ref=ref, token_ids=ids,
                  finish=finish, http_ok=http_ok, prompt=row.get("prompt"))


def _index(rows: Iterable[dict] | dict, what: str, base_dir: Path | None = None) -> dict[str, Output]:
    if isinstance(rows, dict):
        rows = [{"id": k, **v} if "id" not in v else v for k, v in rows.items()]
    out: dict[str, Output] = {}
    for r in rows:
        o = normalize_row(r, base_dir)
        if o.id in out:
            raise GateInputError(f"duplicate id {o.id!r} in {what}")
        out[o.id] = o
    return out


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    rows = []
    # split on "\n" ONLY: str.splitlines() also breaks on U+2028/U+0085, which are legal inside JSON
    # strings (benchmarks/prompts/question_pool.jsonl carries them)
    for i, line in enumerate(p.read_text(encoding="utf-8").split("\n"), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                raise GateInputError(f"{p}:{i}: not JSON: {e}") from e
    for r in rows:  # resolve relative text_path against the rows file, once, at load
        if isinstance(r, dict) and r.get("text_path") and not Path(r["text_path"]).is_absolute():
            r["text_path"] = str((p.parent / r["text_path"]).resolve())
    return rows


# ------------------------------------------------------------------------------ per-pair pieces
def _identity(b: Output, c: Output) -> dict:
    basis = "sha256(text)"
    same = b.sha256 == c.sha256
    if b.token_ids is not None and c.token_ids is not None:
        basis = "sha256(text)+token_ids"
        same = same and list(b.token_ids) == list(c.token_ids)
    return {"identical": same, "basis": basis, "base_sha256": b.sha256, "cand_sha256": c.sha256}


def _degeneracy(b: Output, c: Output, tokenizer: T.Tokenizer | None, allow_text_only: bool) -> tuple[dict, dict]:
    """degeneracy.v2 on both arms, with a SYMMETRIC token source: real ids where carried, the injected
    tokenizer for an arm without them, else (declared) text-only for BOTH arms."""
    have = [o.token_ids is not None for o in (b, c) if o.http_ok]
    force = None
    if not all(have) and tokenizer is None and allow_text_only:
        force = T.MODE_TEXT_ONLY   # symmetric: never compare BPE stats against surrogate stats
    res, prov = {}, {}
    for arm, o in (("base", b), ("cand", c)):
        if not o.http_ok:
            res[arm], prov[arm] = D.classify("", [], o.finish, http_ok=False), {"mode": "n/a (http error)"}
            continue
        ids, pv = T.resolve(o.text, o.token_ids, tokenizer, allow_text_only, force_mode=force)
        res[arm], prov[arm] = D.classify(o.text, ids, o.finish), pv
    return res, prov


def _text_ref(o: Output) -> dict:
    """Full text inline, or the path it was read from; always with its sha256."""
    return {**o.ref, "sha256": o.sha256, "n_chars": len(o.text),
            **({"answer_text": o.answer_text} if o.answer_text != o.text else {})}


def _judge(judge_fn: JudgeFn, pair: dict) -> tuple[str, list[str], Any]:
    try:
        out = judge_fn(pair)
    except Exception as e:  # noqa: BLE001 - a judge failure is an unreviewed item, never a pass
        return "NEEDS_REVIEW", [f"judge error: {type(e).__name__}: {e}"], None
    verdict, reasons = (out, []) if isinstance(out, str) else (out.get("verdict"), list(out.get("reasons") or []))
    if verdict not in VERDICTS:
        return "NEEDS_REVIEW", [f"judge returned invalid verdict {verdict!r}"], out
    return verdict, reasons, out


def evaluate_pair(b: Output | None, c: Output | None, truth: dict | None = None, *,
                  judge_fn: JudgeFn | None = None, tokenizer: T.Tokenizer | None = None,
                  allow_text_only: bool = False, graders: dict | None = None) -> dict:
    pid = (b or c).id  # type: ignore[union-attr]
    if b is None or c is None:
        missing = "base" if b is None else "candidate"
        return {"id": pid, "tier": None, "verdict": "NEEDS_REVIEW", "reasons": [f"unpaired: {missing} output missing"],
                "identity": None, "degeneracy": None, "answer_check": None, "judge": None,
                "texts": {("cand" if b is None else "base"): _text_ref(b or c)}}  # type: ignore[arg-type]
    rec: dict = {"id": pid, "tier": None, "verdict": None, "reasons": [], "identity": None,
                 "degeneracy": None, "answer_check": None, "judge": None, "texts": {"base": _text_ref(b), "cand": _text_ref(c)}}
    ident = _identity(b, c) if (b.http_ok and c.http_ok) else {"identical": False, "basis": "http error",
                                                                "base_sha256": b.sha256, "cand_sha256": c.sha256}
    rec["identity"] = ident
    deg, prov = _degeneracy(b, c, tokenizer, allow_text_only)
    sb, sc = D.severity(deg["base"]), D.severity(deg["cand"])
    rec["degeneracy"] = {"base": deg["base"], "cand": deg["cand"], "severity": {"base": sb, "cand": sc},
                         "token_provenance": prov}
    ac = None
    if truth is not None and A.grader_for(truth) is not None:
        ac = {"base": A.grade(b.answer_text, truth, graders) if b.http_ok else None,
              "cand": A.grade(c.answer_text, truth, graders) if c.http_ok else None}
    rec["answer_check"] = ac

    # ---- tier 0
    if ident["identical"]:
        rec["tier"], rec["verdict"] = 0, "PASS"
        rec["reasons"].append(f"byte-identical ({ident['basis']})")
        if sb > 0:
            rec["reasons"].append(f"shared defect: {deg['cand']['cls']} {deg['cand'].get('reasons', [])} in both arms")
        if ac and ac["cand"] and ac["cand"]["status"] != "correct":
            rec["reasons"].append(f"shared answer: {ac['cand']['status']} in both arms")
        return rec

    # ---- tier 1
    rec["tier"] = 1
    if not b.http_ok:
        rec["verdict"] = "BOTH_BAD" if not c.http_ok else "NEEDS_REVIEW"
        rec["reasons"].append("HTTP-ERROR in both arms" if not c.http_ok else
                              "base unavailable (HTTP-ERROR): no anchor to compare against; re-run the base arm")
        return rec
    reg, notes = [], []
    cand_review_only = deg["cand"]["cls"] == "DEGENERATE" and deg["cand"].get("review")
    if sc > sb:
        msg = f"degeneracy {deg['base']['cls']} -> {deg['cand']['cls']} {deg['cand'].get('reasons', [])}"
        (notes if cand_review_only else reg).append(msg + (" (ascii-only: review)" if cand_review_only else ""))
    rb = rc = None
    if ac and ac["base"] and ac["cand"]:
        rb, rc = ac["base"]["status"], ac["cand"]["status"]
        if A.RANK[rc] > A.RANK[rb]:
            reg.append(f"answer {rb} -> {rc} (expected {ac['cand']['expected']!r}, got {ac['cand']['extracted']!r})")
    if reg:
        rec["verdict"], rec["reasons"] = "REGRESSION", reg + notes
        return rec
    if sb > sc:
        notes.append(f"candidate less degenerate than base ({deg['base']['cls']} -> {deg['cand']['cls']})")
    deg_both = sb > 0 and sc > 0
    # an abstention is a decisive non-answer (model behaviour); a bare "unanswered" (no extractable
    # answer, e.g. truncation) in both arms decides nothing and goes to tier 2
    abst = bool(ac and ac["base"] and ac["cand"] and ac["base"].get("abstained") and ac["cand"].get("abstained"))
    ans_both = (rb is not None and rb != "correct" and rc != "correct"
                and (not (rb == rc == "unanswered") or abst))
    if deg_both or ans_both:
        rec["verdict"] = "BOTH_BAD"
        if deg_both:
            notes.append(f"degeneracy {deg['base']['cls']} (base) / {deg['cand']['cls']} (cand)")
        if ans_both:
            notes.append(("abstained in both arms (model behaviour)" if abst else f"answer {rb} (base) / {rc} (cand)")
                         + f", expected {ac['cand']['expected']!r}")  # type: ignore[index]
        rec["reasons"] = notes
        return rec
    if rc == "correct":
        rec["verdict"] = "PASS"
        rec["reasons"] = [f"diverged; answer {rb} -> correct" + (" (improvement)" if rb != "correct" else "")] + notes
        return rec

    # ---- tier 2 (diverged, no regression, no decisive ground truth)
    why = ("ground truth undecided: unanswered in both arms" if rb == "unanswered" else
           "no ground truth for this item" if ac is None else "ground truth not decisive")
    notes.insert(0, f"diverged at tier 0; {why}")
    if judge_fn is None:
        rec["verdict"], rec["reasons"] = "NEEDS_REVIEW", notes + ["unjudged: no judge_fn injected"]
        return rec
    pair = {"id": pid, "prompt": b.prompt or c.prompt,
            "base": {"text": b.text, "answer_text": b.answer_text, "sha256": b.sha256},
            "cand": {"text": c.text, "answer_text": c.answer_text, "sha256": c.sha256},
            "degeneracy": {"base": deg["base"]["cls"], "cand": deg["cand"]["cls"]}, "notes": list(notes)}
    v, jr, raw = _judge(judge_fn, pair)
    rec["tier"], rec["verdict"], rec["reasons"] = 2, v, notes + jr
    rec["judge"] = {"id": getattr(judge_fn, "judge_id", getattr(judge_fn, "__name__", repr(judge_fn))), "raw": raw}
    return rec


# ------------------------------------------------------------------------------ whole run
def aggregate(items: list[dict], judge_given: bool) -> dict:
    v = collections.Counter(i["verdict"] for i in items)
    tiers = collections.Counter(str(i["tier"]) for i in items)
    modes = collections.Counter(
        f"{arm}:{(i['degeneracy'] or {}).get('token_provenance', {}).get(arm, {}).get('mode')}"
        for i in items if i.get("degeneracy") for arm in ("base", "cand"))
    n = len(items)
    agg = {"n": n, "identical": sum(1 for i in items if (i.get("identity") or {}).get("identical")),
           "pass": v["PASS"], "regressions": v["REGRESSION"], "both_bad": v["BOTH_BAD"],
           "needs_review": v["NEEDS_REVIEW"], "by_tier": dict(sorted(tiers.items())),
           "identical_with_shared_defect": sum(1 for i in items if i["tier"] == 0 and len(i["reasons"]) > 1),
           "token_modes": dict(sorted(modes.items())), "judge_given": judge_given}
    if agg["regressions"]:
        agg["gate"] = "FAIL"
        agg["gate_reasons"] = [f"{i['id']}: {'; '.join(i['reasons'])}" for i in items if i["verdict"] == "REGRESSION"]
    elif agg["needs_review"] or n == 0:
        agg["gate"] = "INCOMPLETE"
        agg["gate_reasons"] = (["no items"] if n == 0 else
                               [f"{agg['needs_review']} item(s) NEEDS_REVIEW"
                                + ("" if judge_given else " and no judge was given")])
    else:
        agg["gate"] = "PASS"
        agg["gate_reasons"] = []
    return agg


def evaluate(base_rows: Iterable[dict] | dict, cand_rows: Iterable[dict] | dict,
             truth: Iterable[dict] | dict | None = None, *, judge_fn: JudgeFn | None = None,
             tokenizer: T.Tokenizer | None = None, allow_text_only: bool = False,
             graders: dict | None = None, anchor: dict | None = None) -> dict:
    """Paired gate over two row sets keyed by `id`. Raises TokenProvenanceError on synthetic ids or
    when no token source is available, and GateInputError on unusable rows: both fail closed.

    `anchor` names what the base arm IS (e.g. {source_commit, binary_sha256, linkage_sha256} for a
    kernel, or {"arm": "speculative.n_max=0", "binary_version": "10303"} for a same-binary reference).
    It is recorded, never interpreted. kernel-research.md clause 4: a coherence label without a named
    anchor is not a verdict, so a codified kernel gate must pass one; `anchor_named` says whether it did."""
    if anchor is not None and (not isinstance(anchor, dict) or not anchor
                               or not all(v not in (None, "") for v in anchor.values())):
        raise GateInputError("anchor must be a non-empty dict with non-empty values")
    base = _index(base_rows, "base")
    cand = _index(cand_rows, "candidate")
    skipped_truth = 0
    if truth is None:
        tmap: dict = {}
    elif isinstance(truth, dict):
        tmap = {str(k): v for k, v in truth.items()}
    else:
        tmap = {}
        for t in truth:
            if isinstance(t, dict) and t.get("id") is not None:
                tmap.setdefault(str(t["id"]), t)
            else:
                skipped_truth += 1  # an id-less truth row can grade nothing (question_pool has one)
    order = list(base) + [k for k in cand if k not in base]
    items = [evaluate_pair(base.get(k), cand.get(k), tmap.get(k), judge_fn=judge_fn, tokenizer=tokenizer,
                           allow_text_only=allow_text_only, graders=graders)
             for k in order]
    return {"schema": SCHEMA_ID, "library": LIBRARY_VERSION, "degeneracy_classifier": D.V2_ID,
            "degeneracy_lineage": D.V2_LINEAGE, "graders": A.GRADERS_VERSION,
            "decoding": "greedy (asserted by the caller; tier 0 identity is only decisive under greedy)",
            "tokenizer": getattr(tokenizer, "tokenizer_id", None), "text_only_declared": allow_text_only,
            "judge": (getattr(judge_fn, "judge_id", getattr(judge_fn, "__name__", repr(judge_fn)))
                      if judge_fn else None),
            "anchor": anchor, "anchor_named": anchor is not None,
            "truth_rows_without_id": skipped_truth,
            "items": items, "aggregate": aggregate(items, judge_fn is not None)}
