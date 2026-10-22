"""lexicon.py — CS-8.LEX1: lexicon builder + keyword scorer (deterministic, CPU-only).

Builder: CS domain term list -> deduplicated per-language phrase files with variants
{as written, lowercase, Title Case, plural}. Scorer: keyword P/R/F per the Contextual
Earnings-22 definition, lexicon-term insertions/hour on lexicon-free audio, and
non-list-word (term-masked) WER.

Reference read (not run): OpenBench @e1310807 src/openbench/metric/keyword_boosting_metrics/
boosting_metrics.py (found on disk under /mnt/raid0/llm/tmp/dive-contextual-earnings22/repo).
Differences from it, all deliberate:
  * alignment is align.align (min-edit, ties -> max exact matches), not texterrors;
  * normalisation is the caller's job (stt_norm); tokens here are whitespace-split strings;
  * a hypothesis term span is FP unless it is the aligned image of an identical reference
    occurrence (OpenBench only flags n-gram FPs where the aligned ref phrase differs, and
    1-gram FPs only when the aligned ref word is not a keyword);
  * precision/recall are None when the denominator is 0 (OpenBench returns 0 via +1e-8);
  * case-folded AND case-sensitive variants; OpenBench folds via the Whisper normaliser only.
OpenBench has NO non-list-word WER; masking here is our own definition (see non_list_wer).
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .align import MATCH, align, edit_distance

MIN_TERM_CHARS = 3  # entries of two characters or fewer are dropped
HIGHER_BETTER = ("term_recall", "term_precision", "term_f")
LOWER_BETTER = ("insertions_per_hour", "non_list_wer")


# --------------------------------------------------------------------------- builder
def load_term_list(path: str) -> dict:
    """Load `{lang: [{term, kind, allow_single_subword?, plural?}]}` from JSON or YAML."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        import yaml  # optional dependency, only for YAML inputs
        return yaml.safe_load(text)
    return json.loads(text)


def title_case(phrase: str) -> str:
    return " ".join(w[:1].upper() + w[1:].lower() for w in phrase.split())


def english_plural(phrase: str) -> str:
    """Explicit English rule on the LAST word: +es after s/x/z/ch/sh, else +s (mechanical:
    'MI210' -> 'MI210s'; use `plural: false` on an entry to suppress)."""
    words = phrase.split()
    last = words[-1]
    low = last.lower()
    words[-1] = last + ("es" if low.endswith(("s", "x", "z", "ch", "sh")) else "s")
    return " ".join(words)


def _norm_entry(e) -> dict:
    return {"term": e} if isinstance(e, str) else dict(e)


def trie_nodes(phrases: Iterable[str], tokenize: Optional[Callable[[str], Sequence]] = None) -> int:
    """Node count (excluding root) of the prefix tree over phrases. `tokenize` defaults to
    whitespace words; LEX2 passes the runner's subword tokenizer so the count matches the
    boosting tree the runner actually builds."""
    tokenize = tokenize or str.split
    root: dict = {}
    n = 0
    for p in phrases:
        node = root
        for t in tokenize(p):
            if t not in node:
                node[t] = {}
                n += 1
            node = node[t]
    return n


def build_lexicon(term_list: Dict[str, list],
                  is_single_subword: Optional[Callable[[str, str], bool]] = None,
                  tokenize: Optional[Callable[[str], Sequence]] = None) -> dict:
    """Build per-language phrase sets.

    is_single_subword(term, lang) -> bool: supplied by the runner's tokenizer in LEX2 (we cannot
      know subword counts without it). None => no term is flagged single-subword (nothing dropped
      on that ground). A flagged term is kept only if its entry has allow_single_subword: true.
    Plural: lang 'en' => english_plural(as-written); any other language => NO plural unless the
      entry gives `plural: "<form>"`. `plural: false` suppresses it for en. Never guessed.
    Variants are generated from the as-written form; dedup is exact-string per language, so
    'EPYC' and 'epyc' variants collapse only where the strings are identical.
    Returns {phrases:{lang:[sorted]}, lexicon_sha256, report}.
    """
    report = {"n_terms_in": 0, "n_dropped_short": 0, "n_dropped_subword": 0,
              "n_phrases_before_expansion": 0, "n_phrases_after_expansion": 0,
              "n_trie_nodes_before_expansion": 0, "n_trie_nodes_after_expansion": 0,
              "per_language": {}}
    phrases: Dict[str, List[str]] = {}
    for lang in sorted(term_list):
        base: Set[str] = set()
        full: Set[str] = set()
        lrep = {"n_terms_in": 0, "n_dropped_short": 0, "n_dropped_subword": 0}
        for raw in term_list[lang]:
            e = _norm_entry(raw)
            term = " ".join(str(e["term"]).split())
            lrep["n_terms_in"] += 1
            if len(term) < MIN_TERM_CHARS:
                lrep["n_dropped_short"] += 1
                continue
            if is_single_subword is not None and is_single_subword(term, lang) \
                    and not e.get("allow_single_subword", False):
                lrep["n_dropped_subword"] += 1
                continue
            base.add(term)
            variants = {term, term.lower(), title_case(term)}
            pl = e.get("plural")
            if isinstance(pl, str) and pl.strip():
                variants.add(" ".join(pl.split()))
            elif lang == "en" and pl is not False:
                variants.add(english_plural(term))
            full |= variants
        phrases[lang] = sorted(full)
        lrep.update(n_phrases_before_expansion=len(base), n_phrases_after_expansion=len(full),
                    n_trie_nodes_before_expansion=trie_nodes(base, tokenize),
                    n_trie_nodes_after_expansion=trie_nodes(full, tokenize))
        report["per_language"][lang] = lrep
        for k, v in lrep.items():
            report[k] += v
    canon = json.dumps(phrases, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {"phrases": phrases,
            "lexicon_sha256": hashlib.sha256(canon.encode("utf-8")).hexdigest(),
            "report": report}


def write_phrase_files(build: dict, out_dir: str) -> List[str]:
    """Write `<lang>.txt` (one phrase per line, sorted) and return the paths."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for lang, ps in sorted(build["phrases"].items()):
        p = os.path.join(out_dir, f"{lang}.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("".join(x + "\n" for x in ps))
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- scorer
def _key(case_sensitive: bool) -> Callable[[str], str]:
    return (lambda t: t) if case_sensitive else (lambda t: t.casefold())


def _term_set(terms: Iterable[str], key) -> Set[Tuple[str, ...]]:
    return {tuple(key(w) for w in t.split()) for t in terms if t.split()}


def term_spans(tokens: Sequence[str], termset: Set[Tuple[str, ...]], key) -> List[Tuple[int, int]]:
    """Non-overlapping contiguous term spans [start, end), longest match first, left to right."""
    if not termset:
        return []
    maxn = max(len(t) for t in termset)
    ktoks = [key(t) for t in tokens]
    spans, i = [], 0
    while i < len(ktoks):
        for n in range(min(maxn, len(ktoks) - i), 0, -1):
            if tuple(ktoks[i:i + n]) in termset:
                spans.append((i, i + n))
                i += n
                break
        else:
            i += 1
    return spans


def keyword_counts(ref_text: str, hyp_text: str, terms: Sequence[str], case_sensitive: bool) -> dict:
    """tp/fn/fp for one clip. TP: every token of a reference term occurrence is a MATCH pair and
    the hypothesis indices are consecutive. FN: other reference occurrences. FP: a hypothesis term
    span not at the aligned image of a TP (misaligned correct text is FP)."""
    key = _key(case_sensitive)
    ts = _term_set(terms, key)
    ref, hyp = ref_text.split(), hyp_text.split()
    al = align([key(t) for t in ref], [key(t) for t in hyp])
    pair_by_ref = {p.ref_idx: p for p in al.pairs if p.ref_idx is not None}
    tp = fn = 0
    tp_hyp_starts: Set[int] = set()
    for s, e in term_spans(ref, ts, key):
        ps = [pair_by_ref[i] for i in range(s, e)]
        ok = all(p.op == MATCH for p in ps) and all(
            ps[k + 1].hyp_idx == ps[k].hyp_idx + 1 for k in range(len(ps) - 1))
        if ok:
            tp += 1
            tp_hyp_starts.add(ps[0].hyp_idx)
        else:
            fn += 1
    fp = sum(1 for s, _ in term_spans(hyp, ts, key) if s not in tp_hyp_starts)
    return {"tp": tp, "fn": fn, "fp": fp}


def _prf(tp: int, fn: int, fp: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = None if p is None or r is None else (0.0 if p + r == 0 else 2 * p * r / (p + r))
    return {"term_precision": p, "term_recall": r, "term_f": f}


def keyword_prf(clips: Sequence[Tuple[str, str]], terms: Sequence[str], case_sensitive: bool) -> dict:
    """Pooled (micro) counts over (ref, hyp) clips. Undefined ratios are None, never 0."""
    tot = {"tp": 0, "fn": 0, "fp": 0}
    for ref, hyp in clips:
        for k, v in keyword_counts(ref, hyp, terms, case_sensitive).items():
            tot[k] += v
    return {**tot, **_prf(**tot)}


def insertions_per_hour(lexicon_free: Sequence[Tuple[str, float]], terms: Sequence[str],
                        case_sensitive: bool) -> float:
    """Term-span occurrences in hypotheses of lexicon-FREE clips (every one is an insertion) per
    hour of audio. `lexicon_free` = [(hyp_text, duration_seconds)], duration from decoded sample
    counts. Refuses (ValueError) on zero/negative total duration."""
    secs = sum(d for _, d in lexicon_free)
    if any(d < 0 for _, d in lexicon_free) or secs <= 0:
        raise ValueError("insertions_per_hour: total duration must be > 0 seconds")
    key = _key(case_sensitive)
    ts = _term_set(terms, key)
    n = sum(len(term_spans(h.split(), ts, key)) for h, _ in lexicon_free)
    return n / (secs / 3600.0)


def non_list_wer(clips: Sequence[Tuple[str, str]], terms: Sequence[str], case_sensitive: bool) -> Optional[float]:
    """Pooled WER over non-list words. Masking choice: term spans are DELETED from both reference
    and hypothesis (greedy longest-match, per `case_sensitive` key) before pooled edit distance
    with align.edit_distance; the comparison itself is case-sensitive on the remaining tokens
    when case_sensitive else case-folded. A wrongly produced term in the hypothesis is masked too,
    so term errors never leak into this number (they are scored by P/R/F and insertions/h).
    None when the masked reference has no tokens."""
    key = _key(case_sensitive)
    ts = _term_set(terms, key)
    errs = nref = 0
    for ref_text, hyp_text in clips:
        out = []
        for text in (ref_text, hyp_text):
            toks = text.split()
            drop = {i for s, e in term_spans(toks, ts, key) for i in range(s, e)}
            out.append([key(t) for i, t in enumerate(toks) if i not in drop])
        errs += edit_distance(out[0], out[1])
        nref += len(out[0])
    return errs / nref if nref else None


def score_record(*, lexicon_sha256: str, list_scope: str, case_sensitive: bool,
                 terms: Sequence[str], clips: Sequence[Tuple[str, str]],
                 lexicon_free: Sequence[Tuple[str, float]]) -> dict:
    """One scoring record (feeds CS-8.LEX2 / CS-2.STT1 candidates)."""
    prf = keyword_prf(clips, terms, case_sensitive)
    return {
        "lexicon_sha256": lexicon_sha256,
        "list_scope": list_scope,
        "case_policy": "case_sensitive" if case_sensitive else "case_folded",
        **prf,
        "insertions_per_hour": insertions_per_hour(lexicon_free, terms, case_sensitive),
        "non_list_wer": non_list_wer(clips, terms, case_sensitive),
        "metric_directions": {**{m: "higher_better" for m in HIGHER_BETTER},
                              **{m: "lower_better" for m in LOWER_BETTER}},
    }
