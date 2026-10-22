"""YARN-AGG-1 window runner: the ITEM CONTRACT shared by every cell builder, plus scorer dispatch.

Every cell builder (``oolong_local``, ``ruler_local``, ``helmet_local``, ``longbio_local``) yields plain dicts
with the keys below. The arm runner (``arm_runner``) never looks inside a suite: it sends ``messages``,
records the server's own token counts, and calls ``score(item, output, model=...)``.

Item keys (all JSON-serialisable):

  item_id       str   stable across arms and runs; unique within a plan (e.g. ``ruler_niah_mk_2@32k#0003``)
  cell          str   the cell the item belongs to (e.g. ``ruler_niah_mk_2@32k``, ``oolong_synth@131072``)
  suite         str   ``oolong`` | ``ruler`` | ``helmet`` | ``longbio``
  task          str   suite-level task name (``niah_multikey_2``, ``counting``, ``json_kv``, ``standard`` ...)
  context_id    str   items sharing a context_id share a byte-identical message prefix (one cached prefill);
                      the runner keeps them adjacent. Also the cluster key for clustered CIs.
  messages      list  OpenAI chat messages; context first, question LAST (cache-friendly)
  max_tokens    int   generation cap for this item
  thinking      bool  ``chat_template_kwargs.enable_thinking`` (False for every gate cell)
  scorer        str   a key of ``SCORERS``
  gold          any   what the scorer compares against (list of strings for substring scorers)
  datapoint     dict  optional: the original row minus its context text (the Oolong scorer needs it)
  nominal_len   int   the generator's nominal length (NEVER used for a zone label)
  real_n_offline int|None  context tokens counted offline with the served model's tokenizer.json
  n_source      str   ``model_tokenizer_json`` when real_n_offline is set
  pins          dict  provenance (repo sha, dataset revision, generator seed, tokenizer sha256 ...)

``prompt_sha256(item)`` hashes the messages; the runner stores it with every result so a resumed or
cross-arm pair is refused if the two arms did not see byte-identical prompts.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping

REQUIRED_KEYS = ("item_id", "cell", "suite", "task", "context_id", "messages", "max_tokens", "thinking",
                 "scorer", "gold", "nominal_len")

#: name -> fn(item, output_text, *, model) -> {"score": float in [0, 1], ...extra fields}
SCORERS: dict[str, Callable[..., dict]] = {}

DEFAULT_TOKENIZER_JSON = "/mnt/raid0/llm/models/turboderp/Qwen3.8-27B-exl3-4.00bpw/tokenizer.json"
DEFAULT_TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"


def register(name: str):
    def deco(fn):
        if name in SCORERS and SCORERS[name] is not fn:
            raise ValueError(f"scorer {name!r} registered twice")
        SCORERS[name] = fn
        return fn
    return deco


class ItemContractError(ValueError):
    pass


def validate_item(item: Mapping[str, Any]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in item]
    if missing:
        raise ItemContractError(f"item {item.get('item_id')!r} lacks {missing}")
    msgs = item["messages"]
    if not (isinstance(msgs, list) and msgs and all(isinstance(m, dict) and "role" in m and "content" in m
                                                   for m in msgs)):
        raise ItemContractError(f"item {item['item_id']!r}: messages malformed")
    if item["scorer"] not in SCORERS:
        _import_scorers()
    if item["scorer"] not in SCORERS:
        raise ItemContractError(f"item {item['item_id']!r}: unknown scorer {item['scorer']!r}")


def prompt_sha256(item: Mapping[str, Any]) -> str:
    blob = json.dumps({"messages": item["messages"], "max_tokens": item["max_tokens"],
                       "thinking": bool(item["thinking"])}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _import_scorers() -> None:
    """Import every builder module so its scorers register. A builder that fails to import (missing optional
    dependency) simply contributes no scorer; validate_item then names the unknown scorer."""
    import importlib
    for mod in ("oolong_local", "ruler_local", "helmet_local", "longbio_local"):
        try:
            importlib.import_module(f"scripts.benchmark.yarn_gate.{mod}")
        except ImportError:
            pass


def score(item: Mapping[str, Any], output: str, *, model: str) -> dict:
    if item["scorer"] not in SCORERS:
        _import_scorers()
    res = SCORERS[item["scorer"]](item, output, model=model)
    s = float(res["score"])
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"scorer {item['scorer']} returned {s} for {item['item_id']}")
    return {**res, "score": s}


# ── generic scorers usable by any builder ────────────────────────────────────────────────────────────
@register("string_match_all")
def string_match_all(item: Mapping[str, Any], output: str, *, model: str) -> dict:
    """RULER ``string_match_all``: fraction of gold strings found (case-insensitive) in the output."""
    refs = [str(g) for g in (item["gold"] if isinstance(item["gold"], list) else [item["gold"]])]
    low = (output or "").lower()
    hits = sum(1 for r in refs if r.lower() in low)
    return {"score": hits / len(refs) if refs else 0.0, "hits": hits, "n_refs": len(refs)}


@register("exact_match")
def exact_match_scorer(item: Mapping[str, Any], output: str, *, model: str) -> dict:
    from scripts.benchmark.yarn_gate.agg_scoring import exact_match
    golds = item["gold"] if isinstance(item["gold"], list) else [item["gold"]]
    return {"score": 1.0 if any(exact_match(output or "", g) for g in golds) else 0.0}
