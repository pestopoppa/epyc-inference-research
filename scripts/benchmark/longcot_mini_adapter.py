#!/usr/bin/env python3
"""K-LCM-1: LongCoT-Mini benchmark adapter + deterministic structural scorer.

Suite registration: "longcot_mini"  (intake-386 / audit RE-4)

Purpose
-------
A ~500-row, *easy*-difficulty, long-horizon reasoning suite for local models,
packaged for a single clean-window run.  It exists because EV-9 found the
production suites saturated; LongCoT-Mini is deliberately *non-saturated* and
tests the reasoning-compression premise (unbounded chain-of-thought is not
always helpful) on our stack.

Data source (already staged on this host, no download step)
-----------------------------------------------------------
    /mnt/raid0/llm/epyc-inference-research/data/longcot-mini/
        data-00000-of-00001.arrow   (HuggingFace `datasets` Arrow table)
        dataset_info.json
        state.json

Loaded with ``datasets.load_from_disk`` — the on-disk object is a single
``Dataset`` (NOT a ``DatasetDict``).  All 507 rows are ``difficulty == "easy"``;
that single "easy" partition IS the split K-LCM-1 asks for.

Schema (per row): question_id, domain, difficulty, template, prompt, answer,
canary.

Answer shapes (ground-truthed 2026-07-17)
-----------------------------------------
Every ``answer`` is a JSON value.  By domain:
  - chemistry (100): JSON string — a SMILES, e.g. '"C1(CCCC1)NC1=CC=..."'
  - chess      (100): JSON string — a move-count number-as-string ("391365")
                      or a FEN board ("8/r7/kn3p2/... w - - 122 349")
  - cs         (100): JSON array of ints ([15, 392, 2790]) or JSON object
                      ({"Q1": "...", "Q2": 3159991384, ...})
  - math       (102): JSON array of expression-strings
                      (["16", "13", "54", "89"], ["2013^{4025}", "2692", "26"])
  - logic      (105): answer == "null" (JSON null) — NO stored gold.

Every prompt instructs the model to emit its final answer as::

    solution = <value>

so the scorer extracts the text after the last ``solution =`` in the model
response and compares it structurally to the gold ``answer``.

Deterministic scoring (MEASUREMENT.md hard requirement: NO LLM-judge)
---------------------------------------------------------------------
``scoring_method = "structural_exact_match"`` — a custom, fully deterministic
method handled by this module (mirrors how ``tulving_episodic_adapter`` owns its
``f1_list`` method).  It is deterministic because:
  1. the model answer is extracted by a fixed regex (last ``solution = ...``),
  2. gold and prediction are parsed as JSON (with a Python-literal fallback),
  3. both sides are recursively *canonicalized* — dict keys sorted, whitespace
     collapsed, numeric scalars normalized (so ``391365`` == ``"391365"``),
     string case PRESERVED (SMILES/FEN are case-sensitive),
  4. equality is a pure structural compare.
No sampling, no network, no model-in-the-loop — identical inputs always yield
identical scores.

Unscorable "logic" rows
-----------------------
The 105 ``logic`` rows carry ``answer == "null"``: the dataset ships no gold for
them (their solutions require a per-puzzle simulator/checker — Sokoban,
BlocksWorld, Sudoku, Hanoi, ... — which is neither a stored-gold match nor an
LLM-judge, and is out of scope for this adapter).  They are therefore EXCLUDED
by default so they cannot pollute a deterministic accuracy number.  Set
``LONGCOT_MINI_INCLUDE_UNSCORABLE=1`` (or ``include_unscorable=True``) to load
all 507 rows; the logic rows are then emitted with
``scoring_config["is_scorable"] = False`` and MUST be reported separately (never
counted as correct).

Default suite size: 402 scorable rows (chemistry 100 + chess 100 + cs 100 +
math 102).

Canary / contamination signal
------------------------------
Each row carries a ``canary`` UUID (a benchmark-contamination guard string that
should NEVER legitimately appear in a solving model's output).  It is carried
through into ``scoring_config["canary"]`` and ``metadata["canary"]``.  A run
should record, per question, whether the canary UUID appears in the model
response (use ``detect_canary_leak``): any hit is a strong signal the model was
trained on this dataset, which invalidates the accuracy reading for that row.
Canary presence is a SEPARATE signal — it is not folded into pass/fail.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Optional

# Allow standalone import outside the benchmarks package (mirrors siblings).
try:
    from dataset_adapters import BaseAdapter
except ImportError:  # pragma: no cover - path shim
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset_adapters import BaseAdapter

# ── Default local path ────────────────────────────────────────────────────────

_DEFAULT_DATA_DIR = Path(
    "/mnt/raid0/llm/epyc-inference-research/data/longcot-mini"
)

# The sentinel the dataset uses for "no stored gold" (JSON null, serialized).
_NULL_GOLD = "null"

SCORING_METHOD = "structural_exact_match"
SOLUTION_FORMAT_INSTRUCTION = (
    "\n\nReturn the final answer on its own final line exactly as:\n"
    "solution = <value>"
)

# Domains that ship a real stored gold answer (logic ships none).
SCORABLE_DOMAINS = ("chemistry", "chess", "cs", "math")


# ── Deterministic structural scorer ───────────────────────────────────────────

# Matches "solution = <rest of line>" (case-insensitive). We take the LAST
# occurrence in the model response, then parse a value from the tail.
_SOLUTION_RE = re.compile(r"solution\s*=\s*", re.IGNORECASE)


def _extract_solution_text(response: str) -> Optional[str]:
    """Return the raw text after the LAST ``solution =`` marker, else None.

    Deterministic: last-occurrence anchor (models often echo the format
    instruction earlier; the final answer is last).
    """
    if not response:
        return None
    matches = list(_SOLUTION_RE.finditer(response))
    if not matches:
        return None
    tail = response[matches[-1].end():]
    return tail.strip() or None


def _parse_leading_value(text: str) -> Any:
    """Parse a JSON / Python-literal value from the start of ``text``.

    Handles single-line scalars and multi-line ``{...}`` / ``[...]`` blocks by
    scanning to the matching balanced bracket.  Falls back to the first line as
    a raw string.  Never raises.
    """
    if text is None:
        return None
    s = text.strip()
    if not s:
        return ""

    # Strip a trailing period/qualifier that some models add after the value on
    # the same line only for the scalar path (handled below).
    if s[0] in "[{\"'":
        # Balanced-bracket scan for containers; quoted-string scan for strings.
        opener = s[0]
        if opener in "[{":
            closer = "]" if opener == "[" else "}"
            depth = 0
            in_str = False
            esc = False
            end = None
            for i, ch in enumerate(s):
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            candidate = s[:end] if end is not None else s
        else:  # quoted string
            candidate = s.split("\n", 1)[0]
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate)
            except Exception:
                continue
        # Unparseable container/quoted token → strip quotes if present.
        return candidate.strip().strip("\"'")

    # Scalar path: take the first line, try JSON/literal, else raw string.
    first_line = s.split("\n", 1)[0].strip()
    # Drop a single trailing sentence period that is clearly punctuation.
    stripped = first_line[:-1] if first_line.endswith(".") and not first_line[:-1].endswith(".") else first_line
    for candidate in (first_line, stripped):
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate)
            except Exception:
                continue
    return first_line


def _norm_scalar(v: Any) -> Any:
    """Canonicalize a scalar. Numbers (and numeric strings) collapse to a single
    canonical numeric form so ``391365`` == ``"391365"`` == ``391365.0``.
    Non-numeric strings keep their CASE (SMILES/FEN are case-sensitive) but have
    surrounding whitespace stripped and internal whitespace runs collapsed.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if math.isfinite(v) and v.is_integer() else v
    if isinstance(v, str):
        t = " ".join(v.strip().split())
        # numeric string → canonical number
        if re.fullmatch(r"[+-]?\d+", t):
            return int(t)
        try:
            f = float(t)
            if not math.isfinite(f):
                return t
            return int(f) if f.is_integer() else f
        except (ValueError, TypeError):
            return t
    return v


def _canonicalize(v: Any) -> Any:
    """Recursively canonicalize a parsed value for structural equality."""
    if isinstance(v, dict):
        return {str(k): _canonicalize(val) for k, val in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [_canonicalize(x) for x in v]
    return _norm_scalar(v)


def _coerce_gold(gold: Any) -> Any:
    """Accept gold as a parsed value OR as its JSON string form and canonicalize."""
    if isinstance(gold, str):
        try:
            gold = json.loads(gold)
        except Exception:
            pass
    return _canonicalize(gold)


def score_structural(model_response: str, gold: Any) -> dict:
    """Deterministically score one model response against a gold answer.

    Args:
        model_response: raw model output text.
        gold: the gold answer — either a parsed value or its JSON string.

    Returns:
        dict(correct: bool, extracted, predicted, gold, reason).
    """
    gold_canon = _coerce_gold(gold)
    extracted = _extract_solution_text(model_response)
    if extracted is None:
        return {
            "correct": False,
            "extracted": None,
            "predicted": None,
            "gold": gold_canon,
            "reason": "no_solution_marker",
        }
    predicted = _canonicalize(_parse_leading_value(extracted))
    correct = predicted == gold_canon
    return {
        "correct": correct,
        "extracted": extracted,
        "predicted": predicted,
        "gold": gold_canon,
        "reason": "match" if correct else "mismatch",
    }


def detect_canary_leak(model_response: str, canary: str) -> bool:
    """True if the canary UUID appears in the model response (contamination)."""
    if not canary or not model_response:
        return False
    return canary in model_response


# ── Adapter ───────────────────────────────────────────────────────────────────


class LongCoTMiniAdapter(BaseAdapter):
    """LongCoT-Mini easy-split reasoning suite (intake-386 / RE-4).

    Loads the pre-staged HuggingFace Arrow dataset and exposes each row as a
    benchmark question in the shape ``run_benchmark.py`` / the downstream scorer
    expect.  Default: the 402 rows that carry a real stored gold answer.

    Args:
        data_dir: root of the ``longcot-mini`` dataset dir (Arrow).
        include_unscorable: if True (or env LONGCOT_MINI_INCLUDE_UNSCORABLE
            truthy), also load the 105 ``logic`` rows that have no stored gold;
            they are flagged ``is_scorable = False`` and must be reported apart.
    """

    suite_name = "longcot_mini"
    has_real_tiers = False  # single "easy" difficulty; domain lives in metadata

    def __init__(
        self,
        data_dir: Optional[Path | str] = None,
        include_unscorable: Optional[bool] = None,
    ):
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        if include_unscorable is None:
            include_unscorable = os.environ.get(
                "LONGCOT_MINI_INCLUDE_UNSCORABLE", ""
            ).strip().lower() in ("1", "true", "yes", "on")
        self._include_unscorable = bool(include_unscorable)

    # ── loading ───────────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        if not self._data_dir.exists():
            print(
                f"  [adapter] LongCoTMini: data not found at {self._data_dir}\n"
                "  Expected a HuggingFace Arrow dataset (load_from_disk) with "
                "columns: question_id, domain, difficulty, template, prompt, "
                "answer, canary."
            )
            self._dataset = []
            return
        try:
            from datasets import load_from_disk
        except ImportError:
            print("  [adapter] LongCoTMini: `datasets` not importable")
            self._dataset = []
            return

        try:
            ds = load_from_disk(str(self._data_dir))
        except Exception as e:  # pragma: no cover - defensive
            print(f"  [adapter] LongCoTMini load failed: {e}")
            self._dataset = []
            return

        rows = []
        for row in ds:
            gold_raw = row.get("answer", _NULL_GOLD)
            is_scorable = (gold_raw is not None) and (str(gold_raw) != _NULL_GOLD)
            if not is_scorable and not self._include_unscorable:
                continue
            rows.append(dict(row))
        # Stable order: by question_id so runs are reproducible.
        rows.sort(key=lambda r: str(r.get("question_id", "")))
        self._dataset = rows

    # ── question shaping ──────────────────────────────────────────────────────

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question_id = str(row.get("question_id", f"lcm_{idx:04d}"))
        domain = str(row.get("domain", ""))
        difficulty = str(row.get("difficulty", "easy"))
        template = str(row.get("template", ""))
        prompt = str(row.get("prompt", ""))
        if "solution =" not in prompt.lower():
            prompt = prompt.rstrip() + SOLUTION_FORMAT_INSTRUCTION
        canary = str(row.get("canary", ""))
        gold_raw = row.get("answer", _NULL_GOLD)

        is_scorable = (gold_raw is not None) and (str(gold_raw) != _NULL_GOLD)

        gold_value = None
        expected_str = ""
        if is_scorable:
            try:
                gold_value = json.loads(gold_raw)
            except Exception:
                gold_value = gold_raw  # non-JSON gold → keep raw string
            expected_str = json.dumps(_canonicalize(gold_value), ensure_ascii=False)

        return {
            "id": f"longcot_mini_{question_id}",
            "suite": "longcot_mini",
            "prompt": prompt,
            "context": "",
            "expected": expected_str,  # canonical JSON of the gold (or "")
            "scoring": [],
            "image_path": "",
            "tier": 1,
            "scoring_method": SCORING_METHOD,
            "scoring_config": {
                "is_scorable": is_scorable,
                "domain": domain,
                "template": template,
                "question_id": question_id,
                "canary": canary,          # contamination guard — carry through
                "extract_pattern": r"solution\s*=\s*(.+)",
            },
            "metadata": {
                "question_id": question_id,
                "domain": domain,
                "difficulty": difficulty,
                "template": template,
                "canary": canary,
                "is_scorable": is_scorable,
                "gold_raw": gold_raw,
                "gold_value": gold_value,
            },
        }

    # ── scoring convenience method (mirrors TulvingEpisodicAdapter) ────────────

    @staticmethod
    def compute_score_for_result(model_response: str, prompt_dict: dict) -> dict:
        """Deterministically score one model response for a prompt dict.

        Returns a dict with: correct (bool | None), predicted, gold, reason,
        is_scorable, canary_leak, domain, template, question_id.
        ``correct`` is None for unscorable (logic / null-gold) rows.
        """
        meta = prompt_dict.get("metadata", {})
        canary = meta.get("canary", "")
        canary_leak = detect_canary_leak(model_response, canary)
        base = {
            "is_scorable": bool(meta.get("is_scorable", False)),
            "canary_leak": canary_leak,
            "domain": meta.get("domain", ""),
            "template": meta.get("template", ""),
            "question_id": meta.get("question_id", ""),
        }
        if not meta.get("is_scorable", False):
            base.update(
                {"correct": None, "predicted": None, "gold": None,
                 "reason": "unscorable_null_gold",
                 "extracted": _extract_solution_text(model_response)}
            )
            return base
        result = score_structural(model_response, meta.get("gold_value"))
        base.update(result)
        return base


if __name__ == "__main__":
    adapter = LongCoTMiniAdapter()
    adapter._ensure_loaded()
    total = adapter.total_available
    print(f"LongCoTMiniAdapter: {total} scorable rows loaded (default)")
    if total:
        from collections import Counter
        dom = Counter(r.get("domain") for r in adapter._dataset)
        print("  domains:", dict(dom))
        s = adapter.sample(n=1, seed=42)[0]
        print(f"  sample id={s['id']} method={s['scoring_method']} "
              f"scorable={s['scoring_config']['is_scorable']} "
              f"canary={s['scoring_config']['canary'][:8]}...")
        # tiny scorer smoke
        gold = s["metadata"]["gold_value"]
        good = f"...reasoning...\nsolution = {json.dumps(gold, ensure_ascii=False)}"
        print("  self-score (correct answer):",
              LongCoTMiniAdapter.compute_score_for_result(good, s)["correct"])
