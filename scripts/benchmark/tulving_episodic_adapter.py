#!/usr/bin/env python3
"""P3b: Tulving Episodic Memory benchmark adapter + deterministic F1 scorer.

Source: "Episodic Memories Generation and Evaluation Benchmark for LLMs"
  arXiv 2501.13121, ICLR 2025.  GitHub: ahstat/episodic-memory-benchmark.
  Data:  https://doi.org/10.6084/m9.figshare.28244480

Dataset structure (after Figshare download + extraction):
  data/
    Udefault_Sdefault_seed0/          ← default benchmark (20-ch short, 200-ch long)
    UdefaultOrdered_Sdefault_seed0/   ← ordered ablation
    Unews_Snews_seed1/                ← world-news style
    Uscifi_Sscifi_seed2/              ← sci-fi style

  Inside each variant:
    books/<model_dir>/                ← the generated book narrative
    df_qa_<nchs>chapters_*.parquet   ← QA pairs (pandas DataFrame)
       Columns: question, correct_answer, retrieval_type, get, cue,
                cue_completed, chapter, date, location, entity, content, ...

Suite registration: "tulving_episodic"

Manual download step (≈150 MB zip):
    python -c "
    import requests, zipfile, io
    url = 'https://ndownloader.figshare.com/files/51825077'
    r = requests.get(url, stream=True)
    dest = '/mnt/raid0/llm/data/eval/tulving_episodic/'
    import pathlib; pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(dest)
    print('Extracted to', dest)
    "
    # Or via wget:
    # wget -P /mnt/raid0/llm/data/eval/tulving_episodic/ \\
    #      https://ndownloader.figshare.com/files/51825077

Metrics (per benchmark paper):
  Simple Recall Score:
    Group questions by number of matching events (0, 1, 2, 3–5, 6+).
    Average F1 within each group, then average across groups.
    Group 0 checks hallucination (expected empty answer).

  Chronological Awareness Score:
    Average of: Latest State score (F1 on single-latest-value questions)
                + Chronological Order score (Kendall τ on ordered-list questions).
"""

from __future__ import annotations

import ast
import json
import re
import string
import unicodedata
from pathlib import Path
from typing import Optional

# Allow standalone import outside the benchmarks package
try:
    from dataset_adapters import BaseAdapter
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset_adapters import BaseAdapter

# ── Default local path ───────────────────────────────────────────────────────

_DEFAULT_DATA_DIR = Path("/mnt/raid0/llm/data/eval/tulving_episodic")

# Preferred variant: 20-chapter short default (10K tokens, ~456 QA pairs)
_DEFAULT_VARIANT = "Udefault_Sdefault_seed0"

# ── Deterministic F1 scorer ──────────────────────────────────────────────────


def _normalise_token(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    # Unicode NFC normalisation
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Remove punctuation (keep digits, letters, space)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def _tokenise(text: str) -> list[str]:
    """Split normalised text into tokens."""
    return _normalise_token(text).split()


def _token_f1(prediction: str, ground_truth: str) -> float:
    """Standard token-level F1 score (matches SQuAD / SimpleQA convention).

    Covers ~95% of Tulving benchmark answer types:
    dates ("Sep 22, 2026"), locations ("New York City"), entity names,
    event content phrases.  Exact numeric matches are caught automatically
    because digit tokens are preserved after punctuation stripping.
    """
    pred_tokens = _tokenise(prediction)
    gt_tokens = _tokenise(ground_truth)

    if not gt_tokens and not pred_tokens:
        return 1.0
    if not gt_tokens or not pred_tokens:
        return 0.0

    # Token-level intersection (bag of words, capped to min count)
    from collections import Counter
    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)
    common = sum((pred_counts & gt_counts).values())

    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)

    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _best_item_f1(pred_item: str, gt_list: list[str]) -> float:
    """Return the maximum token-F1 between pred_item and any item in gt_list."""
    if not gt_list:
        return 0.0
    return max(_token_f1(pred_item, gt) for gt in gt_list)


def score_f1_list(
    predicted_items: list[str],
    ground_truth_items: list[str],
    *,
    threshold: float = 0.5,
) -> dict:
    """Compute set-level F1 for episodic-memory list answers.

    Deterministic implementation covering:
    - Dates:      "Sep 22, 2026", "March 14, 2024"
    - Locations:  "New York City", "Paris"
    - Entities:   "Jackson Ramos", "Emilia Hooks"
    - Contents:   short event-description phrases

    Returns a dict with keys:
      precision, recall, f1, matched_gt_items, nb_pred, nb_gt

    Args:
        predicted_items: Model-extracted list of answer items.
        ground_truth_items: Oracle list (from QA dataset).
        threshold: Token-F1 threshold to count a predicted item as a match.
    """
    nb_gt = len(ground_truth_items)
    nb_pred = len(predicted_items)

    if nb_gt == 0 and nb_pred == 0:
        return {
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "matched_gt_items": [], "nb_pred": 0, "nb_gt": 0,
        }
    if nb_gt == 0:
        # Predicted something when ground truth is empty → hallucination
        return {
            "precision": 0.0, "recall": 1.0, "f1": 0.0,
            "matched_gt_items": [], "nb_pred": nb_pred, "nb_gt": 0,
        }
    if nb_pred == 0:
        return {
            "precision": 1.0, "recall": 0.0, "f1": 0.0,
            "matched_gt_items": [], "nb_pred": 0, "nb_gt": nb_gt,
        }

    # Greedy matching: for each GT item, find the best-matching predicted item
    # (capped to prevent over-counting the same prediction twice)
    gt_matched_scores: list[float] = []
    remaining_preds = list(predicted_items)
    matched_gt: list[str] = []

    for gt_item in ground_truth_items:
        best_score = 0.0
        best_idx = -1
        for i, pred in enumerate(remaining_preds):
            s = _token_f1(pred, gt_item)
            if s > best_score:
                best_score = s
                best_idx = i
        gt_matched_scores.append(best_score)
        if best_score >= threshold and best_idx >= 0:
            matched_gt.append(gt_item)
            remaining_preds.pop(best_idx)

    sum_scores = sum(gt_matched_scores)

    # Lenient: cap nb_pred at nb_gt (matches paper's lenient policy)
    nb_pred_lenient = min(nb_pred, nb_gt)

    precision = sum_scores / nb_pred_lenient if nb_pred_lenient > 0 else 0.0
    recall = sum_scores / nb_gt if nb_gt > 0 else 0.0

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_gt_items": matched_gt,
        "nb_pred": nb_pred,
        "nb_gt": nb_gt,
    }


def _extract_list_from_response(response: str) -> list[str]:
    """Parse a model response into a list of answer items.

    Handles:
    - Bullet lists:  "• Item\n- Item\n* Item"
    - Numbered lists: "1. Item\n2. Item"
    - Comma-separated: "A, B, C"
    - One item per line (fallback)

    Returns an empty list for explicit abstentions like "I don't know".
    """
    # Abstention patterns (also "None" which is the prompt-instructed abstention token)
    abstention_re = re.compile(
        r"(?is)(none|n/a|i don'?t know\.?|i'?m not sure\.?|"
        r"i cannot (answer|determine).*|no information( available)?\.?|"
        r"not mentioned\.?|not available\.?)"
    )
    if abstention_re.fullmatch(response.strip()):
        return []

    # Try bullet / numbered list first
    bullet_re = re.compile(r"^[\s]*[-•*]\s*(.+)$", re.MULTILINE)
    numbered_re = re.compile(r"^[\s]*\d+[.)]\s*(.+)$", re.MULTILINE)

    bullets = bullet_re.findall(response)
    numbered = numbered_re.findall(response)

    if bullets:
        items = [b.strip() for b in bullets if b.strip()]
    elif numbered:
        items = [n.strip() for n in numbered if n.strip()]
    else:
        # Try comma-separated (if the line has commas and ≤200 chars)
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        candidates = []
        for line in lines:
            if "," in line and len(line) < 200:
                candidates.extend(p.strip() for p in line.split(",") if p.strip())
        if candidates:
            items = candidates
        else:
            # Last resort: one item per line
            items = lines

    return items


def _llm_judge_fallback_hook(
    predicted_items: list[str],
    ground_truth_items: list[str],
    retrieval_type: str,
) -> Optional[float]:
    """Optional LLM-judge fallback.  NOT CALLED in the deterministic scorer.

    Override this function or pass a callable to TulvingEpisodicAdapter
    if you want semantic matching beyond token-F1 (e.g., paraphrase matching
    for "Event contents" retrieval type).

    Returns None to indicate the deterministic scorer should be used.
    """
    return None


# ── Composite score computation ──────────────────────────────────────────────


def compute_simple_recall_score(per_question_results: list[dict]) -> float:
    """Compute Simple Recall Score per paper methodology.

    Groups questions by the number of matching events in the ground truth
    (0, 1, 2, 3–5, 6+), averages F1 within each group, then averages groups.

    Args:
        per_question_results: List of dicts with keys:
          - f1: float
          - nb_gt: int  (number of ground truth items)

    Returns:
        Simple Recall Score ∈ [0, 1].
    """
    # Group by nb_gt bucket
    groups: dict[str, list[float]] = {
        "0": [], "1": [], "2": [], "3-5": [], "6+": [],
    }

    def _bucket(nb_gt: int) -> str:
        if nb_gt == 0:
            return "0"
        if nb_gt == 1:
            return "1"
        if nb_gt == 2:
            return "2"
        if nb_gt <= 5:
            return "3-5"
        return "6+"

    for r in per_question_results:
        b = _bucket(r.get("nb_gt", 0))
        groups[b].append(r.get("f1", 0.0))

    group_avgs = [
        sum(vs) / len(vs)
        for vs in groups.values()
        if vs
    ]
    if not group_avgs:
        return 0.0
    return sum(group_avgs) / len(group_avgs)


def compute_chronological_awareness_score(
    latest_results: list[dict],
    chronological_results: list[dict],
) -> float:
    """Compute Chronological Awareness Score per paper methodology.

    Average of:
      Latest State score:        mean F1 over 'latest' questions.
      Chronological Order score: mean Kendall τ over 'chronological' questions
                                 (requires the full ordered lists to be present).

    Args:
        latest_results:       List of dicts with key 'f1'.
        chronological_results: List of dicts with key 'kendall_tau' (float).

    Returns:
        Chronological Awareness Score ∈ [-1, 1] (typically ≥ 0 for a useful model).
    """
    latest_score = 0.0
    if latest_results:
        latest_score = sum(r.get("f1", 0.0) for r in latest_results) / len(latest_results)

    chrono_score = 0.0
    if chronological_results:
        chrono_score = sum(r.get("kendall_tau", 0.0) for r in chronological_results) / len(chronological_results)

    if not latest_results and not chronological_results:
        return 0.0
    if not latest_results:
        return chrono_score
    if not chronological_results:
        return latest_score

    return (latest_score + chrono_score) / 2.0


# ── Dataset adapter ──────────────────────────────────────────────────────────


class TulvingEpisodicAdapter(BaseAdapter):
    """Episodic memory QA pairs from the Tulving Benchmark (arXiv 2501.13121).

    Loads the pre-generated QA pairs from the Figshare dataset (after manual
    extraction).  Each row is one question from the benchmark.

    The default variant uses the 20-chapter short book (≈456 QA pairs, 10K
    tokens) — appropriate for verifying adapter correctness without very large
    context windows.  The 200-chapter variant (686 QA pairs, 100K tokens) is
    the canonical long-context evaluation target.

    Args:
        data_dir: Root directory of the extracted Figshare data.
        variant: Dataset variant subfolder (default: Udefault_Sdefault_seed0).
        chapters: Target chapter count for choosing the QA parquet file
                  (20 = short, 200 = long).  Falls back to whatever is found.
        llm_judge: Optional callable(predicted_items, gt_items, retrieval_type)
                   → Optional[float].  If it returns a non-None float, that
                   overrides the deterministic token-F1 for the given question.
    """

    suite_name = "tulving_episodic"
    has_real_tiers = True

    def __init__(
        self,
        data_dir: Optional[Path | str] = None,
        variant: str = _DEFAULT_VARIANT,
        chapters: int = 20,
        llm_judge=None,
    ):
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self._variant = variant
        self._target_chapters = chapters
        self._llm_judge = llm_judge or _llm_judge_fallback_hook
        self._book_text: Optional[str] = None

    # ── loading ─────────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._dataset is not None:
            return

        variant_dir = self._data_dir / self._variant
        if not variant_dir.exists():
            print(
                f"  [adapter] TulvingEpisodic: data not found at {variant_dir}\n"
                "  Manual download command:\n"
                "    python -c \"\n"
                "    import requests, zipfile, io, pathlib\n"
                "    url = 'https://ndownloader.figshare.com/files/51825077'\n"
                "    r = requests.get(url, stream=True)\n"
                f"    dest = '{self._data_dir}'\n"
                "    pathlib.Path(dest).mkdir(parents=True, exist_ok=True)\n"
                "    with zipfile.ZipFile(io.BytesIO(r.content)) as z:\n"
                "        z.extractall(dest)\n"
                "    \""
            )
            self._dataset = []
            return

        rows = self._load_qa_from_variant(variant_dir)
        if not rows:
            # Try loading from JSON fallback
            rows = self._load_from_json(variant_dir)

        self._dataset = rows

    def _load_qa_from_variant(self, variant_dir: Path) -> list[dict]:
        """Load QA pairs from parquet files in the variant directory."""
        # The Figshare extract stores chapter count in the parent directory
        # (for example ``..._nbchapters_19_.../df_qa.parquet``), not in the
        # parquet filename. Select the intended QA table only; debug/book
        # parquet files must not expand the 20ch run into 100K/1M variants.
        parquet_files = sorted(variant_dir.rglob("df_qa.parquet"))
        target_files = self._select_target_qa_files(parquet_files)
        if not target_files:
            target_files = parquet_files  # Fallback: any parquet file

        if not target_files:
            return []
        self._book_text = self._load_book_text(target_files[0].parent)

        try:
            import pandas as pd
            dfs = []
            for pf in target_files:
                try:
                    dfs.append(pd.read_parquet(pf))
                except Exception:
                    pass
            if not dfs:
                return []
            df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
            return df.to_dict(orient="records")
        except ImportError:
            print("  [adapter] pandas not available; falling back to JSON")
            return []

    def _select_target_qa_files(self, parquet_files: list[Path]) -> list[Path]:
        """Pick QA parquet files for the configured chapter target."""
        if not parquet_files:
            return []

        def chapter_count(path: Path) -> int | None:
            match = re.search(r"nbchapters_(\d+)", str(path))
            if not match:
                return None
            return int(match.group(1))

        by_chapter: dict[int, list[Path]] = {}
        for path in parquet_files:
            chapters = chapter_count(path)
            if chapters is not None:
                by_chapter.setdefault(chapters, []).append(path)
        if not by_chapter:
            return []

        claude_files = [p for p in parquet_files if "model_claude" in str(p)]
        if claude_files:
            claude_by_chapter: dict[int, list[Path]] = {}
            for path in claude_files:
                chapters = chapter_count(path)
                if chapters is not None:
                    claude_by_chapter.setdefault(chapters, []).append(path)
            if claude_by_chapter:
                by_chapter = claude_by_chapter

        # The nominal 20ch dataset is represented as 19 chapters for the
        # Claude-generated default book and 20 for the GPT-4o variant. Prefer
        # the closest lower-or-equal count to preserve the documented 456-QA
        # default, then fall back to the closest absolute match.
        lower_or_equal = [c for c in by_chapter if c <= self._target_chapters]
        if lower_or_equal:
            selected = max(lower_or_equal)
        else:
            selected = min(by_chapter, key=lambda c: abs(c - self._target_chapters))
        return by_chapter[selected]

    def _load_from_json(self, variant_dir: Path) -> list[dict]:
        """Fallback: load QA from JSON files (exported via epbench io.export_list)."""
        json_files = sorted(variant_dir.rglob("df_qa*.json"))
        rows = []
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    rows.extend(data)
                elif isinstance(data, dict):
                    rows.append(data)
            except Exception:
                pass
        return rows

    def _load_book_text(self, variant_dir: Path) -> Optional[str]:
        """Load the full book narrative text (for context injection into prompts)."""
        json_book = variant_dir / "book.json"
        if json_book.exists():
            try:
                data = json.loads(json_book.read_text(encoding="utf-8"))
                if isinstance(data, str):
                    return data
            except Exception:
                pass
        book_files = sorted(variant_dir.rglob("book*.txt")) + sorted(
            variant_dir.rglob("*.txt")
        )
        if book_files:
            return book_files[0].read_text(encoding="utf-8")
        return None

    # ── tier assignment ──────────────────────────────────────────────────────

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        # Use number of ground truth items as difficulty proxy
        nb_gt = self._nb_gt(row)
        # T3: chronological ordering OR many items (≥6)
        if row.get("get") == "chronological" or nb_gt >= 6:
            return 3
        # T1: zero-answer hallucination checks OR single-item latest-state
        if nb_gt == 0 or (row.get("get") == "latest" and nb_gt <= 1):
            return 1
        return 2

    @staticmethod
    def _nb_gt(row: dict) -> int:
        """Number of ground truth items for the row."""
        return len(
            TulvingEpisodicAdapter._parse_correct_answer(row.get("correct_answer", []))
        )

    @staticmethod
    def _parse_correct_answer(raw) -> list[str]:
        """Coerce correct_answer to a list of strings."""
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if isinstance(raw, list):
            return [str(x) for x in raw if x is not None]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if x is not None]
                return [str(parsed)]
            except json.JSONDecodeError:
                pass
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if x is not None]
                return [str(parsed)]
            except (SyntaxError, ValueError):
                return [raw] if raw else []
        return []

    # ── prompt construction ──────────────────────────────────────────────────

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question = str(row.get("question", "")).strip()
        correct_answer_raw = row.get("correct_answer", [])
        ground_truth = self._parse_correct_answer(correct_answer_raw)
        retrieval_type = str(row.get("retrieval_type", ""))
        get_style = str(row.get("get", "all"))
        cue = str(row.get("cue", ""))
        chapter = int(row.get("chapter", -1)) if row.get("chapter") is not None else -1
        nb_gt = len(ground_truth)

        tier = self._get_tier_for_index(idx)

        # Instruction suffix depends on retrieval type
        if retrieval_type in ("Times",):
            instruction = (
                "List all dates/times. Format each as a separate line starting with '- '."
                " If none, say 'None'."
            )
        elif retrieval_type in ("Spaces",):
            instruction = (
                "List all locations. Format each as a separate line starting with '- '."
                " If none, say 'None'."
            )
        elif retrieval_type in ("Entities",):
            instruction = (
                "List all entity/person names. Format each as a separate line starting with '- '."
                " If none, say 'None'."
            )
        elif retrieval_type in ("Event contents",):
            instruction = (
                "List all event descriptions. Format each as a separate line starting with '- '."
                " If none, say 'None'."
            )
        else:
            instruction = (
                "Answer the question precisely. List items one per line starting with '- '."
                " If none, say 'None'."
            )

        prompt = f"{question}\n\n{instruction}"
        if self._book_text:
            prompt = (
                "Book narrative:\n"
                f"{self._book_text.strip()}\n\n"
                "---\n\n"
                f"{prompt}"
            )

        # Serialise expected as JSON for storage (we keep it as list in metadata)
        expected_str = json.dumps(ground_truth)

        return {
            "id": f"tulving_{self._variant}_ch{chapter:04d}_q{idx:04d}",
            "suite": "tulving_episodic",
            "prompt": prompt,
            "context": self._book_text or "",
            "expected": expected_str,  # JSON-encoded list
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "f1_list",   # custom — handled by compute_f1_for_result()
            "scoring_config": {
                "normalize": True,
                "threshold": 0.5,
                "retrieval_type": retrieval_type,
                "get_style": get_style,
                "nb_gt": nb_gt,
                "llm_judge_fallback": False,  # deterministic only
            },
            "metadata": {
                "cue": cue,
                "retrieval_type": retrieval_type,
                "get_style": get_style,
                "chapter": chapter,
                "nb_gt": nb_gt,
                "ground_truth_items": ground_truth,
            },
        }

    # ── scoring convenience method ───────────────────────────────────────────

    @staticmethod
    def compute_f1_for_result(
        model_response: str,
        prompt_dict: dict,
        *,
        llm_judge=None,
    ) -> dict:
        """Compute deterministic F1 for a single model response.

        Args:
            model_response: Raw text response from the model.
            prompt_dict: The prompt dict returned by _row_to_prompt.
            llm_judge: Optional callable(pred_items, gt_items, retrieval_type)
                       → Optional[float].  If non-None return, overrides F1.

        Returns:
            dict with keys: precision, recall, f1, nb_pred, nb_gt,
                            matched_gt_items, get_style, retrieval_type.
        """
        meta = prompt_dict.get("metadata", {})
        ground_truth = meta.get("ground_truth_items", [])
        retrieval_type = meta.get("retrieval_type", "")
        get_style = meta.get("get_style", "all")

        predicted = _extract_list_from_response(model_response)

        # Try LLM-judge fallback first (if provided)
        if llm_judge is not None:
            judge_score = llm_judge(predicted, ground_truth, retrieval_type)
            if judge_score is not None:
                nb_gt = len(ground_truth)
                return {
                    "precision": judge_score,
                    "recall": judge_score,
                    "f1": judge_score,
                    "nb_pred": len(predicted),
                    "nb_gt": nb_gt,
                    "matched_gt_items": [],
                    "get_style": get_style,
                    "retrieval_type": retrieval_type,
                    "source": "llm_judge",
                }

        result = score_f1_list(predicted, ground_truth)
        result["get_style"] = get_style
        result["retrieval_type"] = retrieval_type
        result["source"] = "deterministic"
        return result


if __name__ == "__main__":
    adapter = TulvingEpisodicAdapter()
    adapter._ensure_loaded()
    total = adapter.total_available
    print(f"TulvingEpisodicAdapter: {total} QA pairs loaded")
    if total > 0:
        samples = adapter.sample(n=3, seed=42)
        for s in samples:
            print(f"  id={s['id']}  tier={s['tier']}")
            meta = s.get("metadata", {})
            print(f"    cue={meta.get('cue')}  type={meta.get('retrieval_type')}")
            print(f"    gt={meta.get('ground_truth_items')[:3]}")
