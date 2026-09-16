#!/usr/bin/env python3
"""Long-context evaluation dataset adapters.

Provides adapters for 5 long-context benchmark datasets, integrating
with the existing BaseAdapter framework in dataset_adapters.py.

Suites:
  - longbench:             LongBench (THUDM, 21 tasks, 5K-30K context)
  - zeroscrolls:           ZeroSCROLLS (tau, 10 tasks, 10K-100K+ context)
  - leval:                 L-Eval (L4NLP, 20 tasks, 3K-60K context)
  - ruler:                 RULER (NVIDIA, synthetic, configurable 4K-128K+)
  - needle_parameterized:  Needle-in-a-Haystack (parameterized depth/length)
  - beam:                  BEAM conversational memory (arXiv 2510.27246, 100K split)

All adapters produce standard prompt dicts compatible with
compare_orchestrator_direct.py and the seeding harness.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from dataset_adapters import BaseAdapter

try:
    from beam_memory_retrievers import (
        CHUNKING, BM25PairChunkRetriever, TracePairChunkRetriever, pair_chunks, render_message,
    )
except ImportError:  # imported as a package module
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).parent))
    from beam_memory_retrievers import (  # noqa: E402
        CHUNKING, BM25PairChunkRetriever, TracePairChunkRetriever, pair_chunks, render_message,
    )

EVAL_DIR = Path("/mnt/raid0/llm/data/eval")


# ── LongBench ───────────────────────────────────────────────────────────────


class LongBenchAdapter(BaseAdapter):
    """LongBench v2: 503 multiple-choice long-context questions (THUDM).

    Uses v2 (parquet-native) since v1 uses deprecated HF loading scripts.
    Fields: _id, domain, sub_domain, difficulty, length, question,
            choice_A/B/C/D, answer, context.

    Tiers based on difficulty field: easy=1, medium=2, hard=3.
    """

    suite_name = "longbench"
    has_real_tiers = True

    _DIFFICULTY_TIER = {"easy": 1, "medium": 2, "hard": 3}

    def _ensure_loaded(self):
        if self._dataset is not None:
            return

        # Try local JSONL first (from download script)
        jsonl_path = EVAL_DIR / "longbench" / "longbench_v2.jsonl"
        if jsonl_path.exists():
            import json
            rows = []
            for line in jsonl_path.read_text().strip().split("\n"):
                if line.strip():
                    rows.append(json.loads(line))
            self._dataset = rows
            return

        # Fallback: load from HF directly
        try:
            import datasets as hf
            ds = hf.load_dataset("THUDM/LongBench-v2", split="train",
                                 cache_dir=str(EVAL_DIR / "longbench"))
            self._dataset = [row for row in ds]
        except Exception as e:
            print(f"  [longbench] Load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        return self._DIFFICULTY_TIER.get(row.get("difficulty", "medium"), 2)

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        context = row.get("context", "")
        question = row.get("question", "")
        choices = "\n".join(
            f"{label}) {row.get(f'choice_{label}', '')}"
            for label in ["A", "B", "C", "D"]
            if row.get(f"choice_{label}")
        )
        answer = row.get("answer", "")

        prompt_text = f"Context:\n{context}\n\nQuestion: {question}\n\n{choices}"

        return {
            "id": f"longbench_{row.get('_id', idx)}",
            "suite": "longbench",
            "prompt": prompt_text,
            "expected": answer,
            "scoring_method": "exact_match",
            "tier": self._DIFFICULTY_TIER.get(row.get("difficulty", "medium"), 2),
            "metadata": {
                "domain": row.get("domain", ""),
                "sub_domain": row.get("sub_domain", ""),
                "difficulty": row.get("difficulty", ""),
                "length": row.get("length", ""),
                "context_length_chars": len(context),
            },
        }


# ── ZeroSCROLLS ─────────────────────────────────────────────────────────────


class ZeroSCROLLSAdapter(BaseAdapter):
    """ZeroSCROLLS: 10 zero-shot long-context tasks (tau).

    Uses validation split (test has no labels — leaderboard only).
    """

    suite_name = "zeroscrolls"
    has_real_tiers = True

    _TASKS = [
        "gov_report", "summ_screen_fd", "qmsum", "squality",
        "qasper", "narrative_qa", "quality", "musique",
        "space_digest", "book_sum_sort",
    ]

    _SUMMARIZATION = {"gov_report", "summ_screen_fd", "qmsum", "squality"}
    _QA = {"qasper", "narrative_qa", "quality", "musique"}
    _AGGREGATION = {"space_digest", "book_sum_sort"}

    def _ensure_loaded(self):
        if self._dataset is not None:
            return

        import json
        all_rows = []
        base = EVAL_DIR / "zeroscrolls"

        for task in self._TASKS:
            task_dir = base / task
            if not task_dir.exists():
                continue
            # Find validation JSONL files
            candidates = list(task_dir.rglob("*val*.jsonl")) + list(task_dir.rglob("*validation*.jsonl"))
            if not candidates:
                candidates = list(task_dir.rglob("*.jsonl"))
            for jsonl_file in candidates:
                try:
                    for line in jsonl_file.read_text().strip().split("\n"):
                        if line.strip():
                            row = json.loads(line)
                            row["_task"] = task
                            all_rows.append(row)
                except Exception as e:
                    print(f"  [zeroscrolls] {jsonl_file.name} parse failed: {e}")

        self._dataset = all_rows
        if not all_rows:
            print("  [zeroscrolls] No data loaded — run download_long_context_datasets.py first")

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        input_text = row.get("input", "")
        chars = len(input_text)
        if chars < 20_000:
            return 1
        elif chars < 50_000:
            return 2
        return 3

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        task = row.get("_task", "unknown")
        input_text = row.get("input", "")
        target = row.get("output", "")

        if task in self._SUMMARIZATION:
            prompt = f"Summarize the following document:\n\n{input_text}"
            scoring = "llm_judge"
        elif task in self._QA:
            prompt = input_text
            scoring = "substring"
        else:
            prompt = input_text
            scoring = "llm_judge"

        return {
            "id": f"zeroscrolls_{task}_{idx}",
            "suite": "zeroscrolls",
            "prompt": prompt,
            "expected": target if isinstance(target, str) else str(target),
            "scoring_method": scoring,
            "tier": self._get_tier_for_index(idx),
            "metadata": {
                "task": task,
                "context_length_chars": len(input_text),
            },
        }


# ── L-Eval ──────────────────────────────────────────────────────────────────


class LEvalAdapter(BaseAdapter):
    """L-Eval: 20 tasks spanning exam, writing, summarization, math (L4NLP).

    Configs split into closed-ended (exact answer) and open-ended (generation).
    """

    suite_name = "leval"
    has_real_tiers = True

    _CONFIGS = [
        "coursera", "gsm100", "quality", "topic_retrieval_longchat",
        "tpo", "codeU", "sci_fi", "gov_report_summ",
        "meeting_summ", "news_summ", "paper_assistant",
        "patent_summ", "review_summ", "tv_show_summ",
        "financial_qa", "legal_contract_qa", "multidoc_qa",
        "natural_question", "scientific_qa",
    ]

    _CLOSED_ENDED = {
        "coursera", "gsm100", "quality", "topic_retrieval_longchat",
        "tpo", "codeU",
    }

    def _ensure_loaded(self):
        if self._dataset is not None:
            return

        import json
        all_rows = []
        base = EVAL_DIR / "leval"

        # L-Eval files may be named like "coursera.jsonl" or in subdirectories
        for cfg in self._CONFIGS:
            candidates = list(base.rglob(f"*{cfg}*.jsonl"))
            for jsonl_file in candidates:
                try:
                    for line in jsonl_file.read_text().strip().split("\n"):
                        if line.strip():
                            row = json.loads(line)
                            row["_config"] = cfg
                            all_rows.append(row)
                except Exception as e:
                    print(f"  [leval] {jsonl_file.name} parse failed: {e}")

        self._dataset = all_rows
        if not all_rows:
            print("  [leval] No data loaded — run download_long_context_datasets.py first")

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        input_text = row.get("input", "")
        instructions = row.get("instructions", "")
        # instructions may be a list of questions
        if isinstance(instructions, list):
            instr_text = "\n".join(instructions)
        else:
            instr_text = str(instructions)
        total_chars = len(instr_text) + len(input_text)
        if total_chars < 10_000:
            return 1
        elif total_chars < 30_000:
            return 2
        return 3

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        cfg = row.get("_config", "unknown")
        instructions = row.get("instructions", "")
        input_text = row.get("input", "")
        # L-Eval uses "outputs" (plural) — may be a list of acceptable answers
        outputs = row.get("outputs", row.get("output", ""))

        if isinstance(instructions, list):
            instr_text = "\n".join(instructions)
        else:
            instr_text = str(instructions)

        if isinstance(outputs, list):
            expected = outputs[0] if outputs else ""
        else:
            expected = str(outputs)

        prompt = f"{input_text}\n\n{instr_text}" if input_text else instr_text
        scoring = "exact_match" if cfg in self._CLOSED_ENDED else "llm_judge"

        return {
            "id": f"leval_{cfg}_{idx}",
            "suite": "leval",
            "prompt": prompt.strip(),
            "expected": expected,
            "scoring_method": scoring,
            "tier": self._get_tier_for_index(idx),
            "metadata": {
                "config": cfg,
                "type": "closed" if cfg in self._CLOSED_ENDED else "open",
                "context_length_chars": len(instr_text) + len(input_text),
            },
        }


# ── RULER ───────────────────────────────────────────────────────────────────


class RULERAdapter(BaseAdapter):
    """RULER: Synthetic long-context tasks at configurable context lengths.

    Generates tasks on-demand using RULER's synthetic generation scripts.
    Does NOT use a static HF dataset.

    Task types: NIAH (needle), variable tracking, common words, QA.
    """

    suite_name = "ruler"
    has_real_tiers = True

    _RULER_REPO = EVAL_DIR / "ruler" / "repo"

    def __init__(self, context_length: int = 4096, num_examples: int = 50):
        self._context_length = context_length
        self._num_examples = num_examples

    def _ensure_loaded(self):
        if self._dataset is not None:
            return

        self._dataset = []
        rng = random.Random(42)

        # Generate NIAH (needle-in-a-haystack) tasks
        for i in range(self._num_examples):
            needle_key = f"key_{rng.randint(1000, 9999)}"
            needle_value = f"value_{rng.randint(100000, 999999)}"
            depth = rng.random()

            # Build haystack from noise text
            filler = " ".join(f"word{rng.randint(0, 10000)}" for _ in range(self._context_length // 5))
            words = filler.split()
            insert_pos = int(len(words) * depth)
            needle_sentence = f"The special {needle_key} is {needle_value}."
            words.insert(insert_pos, needle_sentence)
            haystack = " ".join(words)

            self._dataset.append({
                "task": "niah",
                "input": f"{haystack}\n\nQuestion: What is the value of {needle_key}?",
                "expected": needle_value,
                "depth": depth,
                "context_length": self._context_length,
                "idx": i,
            })

    def _get_tier_for_index(self, idx: int) -> int:
        if self._context_length <= 8192:
            return 1
        elif self._context_length <= 32768:
            return 2
        return 3

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        return {
            "id": f"ruler_niah_{self._context_length}_{idx}",
            "suite": "ruler",
            "prompt": row["input"],
            "expected": row["expected"],
            "scoring_method": "exact_match",
            "tier": self._get_tier_for_index(idx),
            "metadata": {
                "task": row["task"],
                "depth": row.get("depth", 0),
                "context_length_target": self._context_length,
                "context_length_chars": len(row["input"]),
            },
        }


# ── Needle-in-a-Haystack (Parameterized) ───────────────────────────────────


class NeedleAdapter(BaseAdapter):
    """Parameterized needle-in-a-haystack using Paul Graham essays.

    Generates a matrix of test cases across:
      - context_lengths: [4096, 8192, 16384, 32768, 65536]
      - needle_positions: [0.1, 0.25, 0.5, 0.75, 0.9]
      - num_needles: [1]

    Uses real essay text (not synthetic filler) from the reference repo.
    """

    suite_name = "needle_parameterized"
    has_real_tiers = True

    _ESSAYS_DIR = EVAL_DIR / "needle" / "repo" / "needlehaystack" / "PaulGrahamEssays"
    _NEEDLE_TEMPLATE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."

    def __init__(
        self,
        context_lengths: list[int] | None = None,
        needle_positions: list[float] | None = None,
    ):
        self._context_lengths = context_lengths or [4096, 8192, 16384, 32768, 65536]
        self._needle_positions = needle_positions or [0.1, 0.25, 0.5, 0.75, 0.9]

    def _load_haystack(self) -> str:
        """Load all Paul Graham essays as a single haystack string."""
        essays_dir = self._ESSAYS_DIR
        if not essays_dir.exists():
            return ""
        texts = []
        for f in sorted(essays_dir.glob("*.txt")):
            texts.append(f.read_text(errors="replace"))
        return "\n\n".join(texts)

    def _ensure_loaded(self):
        if self._dataset is not None:
            return

        haystack_full = self._load_haystack()
        if not haystack_full:
            print("  [needle] Paul Graham essays not found — using synthetic haystack")
            rng = random.Random(42)
            haystack_full = " ".join(f"word{rng.randint(0, 50000)}" for _ in range(200000))

        self._dataset = []
        for ctx_len in self._context_lengths:
            # Truncate haystack to target char length (rough: 4 chars/token)
            target_chars = ctx_len * 4
            haystack = haystack_full[:target_chars]

            for depth in self._needle_positions:
                insert_pos = int(len(haystack) * depth)
                text_with_needle = (
                    haystack[:insert_pos]
                    + f"\n{self._NEEDLE_TEMPLATE}\n"
                    + haystack[insert_pos:]
                )

                self._dataset.append({
                    "context_length": ctx_len,
                    "depth": depth,
                    "haystack_chars": len(text_with_needle),
                    "input": text_with_needle,
                    "question": "What is the best thing to do in San Francisco?",
                    "expected": "eat a sandwich and sit in Dolores Park on a sunny day",
                })

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        ctx = row["context_length"]
        if ctx <= 8192:
            return 1
        elif ctx <= 32768:
            return 2
        return 3

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        return {
            "id": f"needle_{row['context_length']}_{row['depth']:.2f}",
            "suite": "needle_parameterized",
            "prompt": f"{row['input']}\n\nQuestion: {row['question']}",
            "expected": row["expected"],
            "scoring_method": "substring",
            "tier": self._get_tier_for_index(idx),
            "metadata": {
                "context_length_target": row["context_length"],
                "context_length_chars": row["haystack_chars"],
                "needle_depth": row["depth"],
            },
        }


# ── BEAM (conversational long-term memory) ─────────────────────────────────


#: HF ``Mohammadta/BEAM`` split names (the paper's "128K" scale is labelled 100K
#: in the released parquet). 10M ships only in the GitHub repo tree.
BEAM_SPLITS = ("100K", "500K", "1M", "10M")
BEAM_HF_DATASET = "Mohammadta/BEAM"
#: Size of the default split's parquet on HF (dataset sha 3205395e), for the
#: missing-data message; 500K is 33,956,263 B and 1M is 66,156,374 B.
BEAM_100K_PARQUET_BYTES = 5_429_768

#: Reference-answer field per ability; the dataset uses a different key for
#: each question family.
_BEAM_REFERENCE_FIELDS = (
    "answer", "ideal_answer", "ideal_response", "ideal_summary", "expected_compliance",
)
_BEAM_DIFFICULTY_TIER = {"easy": 1, "clear": 1, "medium": 2, "hard": 3}


# ── M-12b arms (B2) ─────────────────────────────────────────────────────────
#
# The arm is chosen like the Tulving CME-4 arm: a constructor argument, else an env var,
# because ``get_adapter("beam")`` constructs with no arguments. Both memory arms share
# one prompt header (M-12c(7): prompt-matched, so the control and the arm under test
# differ ONLY in which excerpts they show); the arm itself is recorded in each prompt's
# ``provenance`` and the scorer refuses a ``--arm`` that disagrees with it.

BEAM_CONTEXT_FULL = "full"      # BEAM's Vanilla column: the whole history (memory-off)
BEAM_CONTEXT_RAG = "rag"        # naive-memory control: pair_chunk x BM25
BEAM_CONTEXT_TRACE = "trace"    # arm under test: pair_chunk through the trace store
BEAM_CONTEXT_MODES = (BEAM_CONTEXT_FULL, BEAM_CONTEXT_RAG, BEAM_CONTEXT_TRACE)
BEAM_CONTEXT_MODE_ENV = "BEAM_CONTEXT_MODE"
BEAM_RETRIEVAL_TOP_K_ENV = "BEAM_RETRIEVAL_TOP_K"
#: ~10 x the median pair chunk (~2.9K chars) ~= 6K Qwen tokens: the same order as the
#: Tulving retrieved arm's top-5 chapters, and far inside the 32K the paper gave RAG.
BEAM_DEFAULT_RETRIEVAL_TOP_K = 10
BEAM_FULL_HEADER = "The following is the complete history of your conversation with the user.\n\n"
BEAM_RETRIEVED_HEADER = (
    "The following are excerpts retrieved from the history of your conversation with "
    "the user.\n\n")
BEAM_NO_EXCERPTS = "(no excerpts retrieved)"


def beam_context_kind_of_prompt(prompt: str) -> str:
    """``"full"`` or ``"retrieved"``, read from a stored prompt's header, else ``"unknown"``."""
    if prompt.startswith(BEAM_FULL_HEADER):
        return "full"
    if prompt.startswith(BEAM_RETRIEVED_HEADER):
        return "retrieved"
    return "unknown"


class BEAMLoadError(RuntimeError):
    """BEAM data that is present but cannot be loaded faithfully.

    Raised rather than returning an empty dataset: a missing ``pyarrow`` or a
    ``probing_questions`` payload that no longer parses must never degrade into
    "zero questions" or "every question missing" (the Tulving-adapter defect).
    """


def _beam_flatten_chat(chat) -> list[dict]:
    """Flatten either BEAM chat shape into an ordered list of message dicts.

    HF parquet: ``list<list<message>>``. GitHub ``chat.json``: a list of
    ``{"batch_number", "turns": list<list<message>>}``.
    """
    out: list[dict] = []

    def walk(node) -> None:
        if hasattr(node, "tolist"):
            node = node.tolist()
        if isinstance(node, dict):
            if "turns" in node:
                walk(node["turns"])
            elif "role" in node and "content" in node:
                out.append(node)
            else:
                raise BEAMLoadError(f"unrecognised BEAM chat node with keys {sorted(node)}")
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)
        elif node is not None:
            raise BEAMLoadError(f"unrecognised BEAM chat node of type {type(node).__name__}")

    walk(chat)
    return out


def _beam_parse_probing_questions(raw, conversation_id: str) -> dict:
    """``probing_questions`` is a STRING in the parquet (dataset card: ``ast.literal_eval``)."""
    from beam_scoring import BEAM_ABILITIES

    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise BEAMLoadError(
                    f"conversation {conversation_id}: probing_questions parses as neither a "
                    f"Python literal nor JSON ({exc})") from None
    if not isinstance(parsed, dict):
        raise BEAMLoadError(f"conversation {conversation_id}: probing_questions is not a mapping")
    unknown = sorted(set(parsed) - set(BEAM_ABILITIES))
    missing = sorted(set(BEAM_ABILITIES) - set(parsed))
    if unknown or missing:
        raise BEAMLoadError(
            f"conversation {conversation_id}: ability keys drifted "
            f"(unknown={unknown}, missing={missing})")
    for ability, questions in parsed.items():
        if not isinstance(questions, list) or not questions:
            raise BEAMLoadError(f"conversation {conversation_id}: {ability} has no questions")
        for i, q in enumerate(questions):
            rubric = q.get("rubric") if isinstance(q, dict) else None
            if (not isinstance(q, dict) or not str(q.get("question", "")).strip()
                    or not isinstance(rubric, list) or not rubric
                    or not all(isinstance(item, str) and item.strip() for item in rubric)):
                raise BEAMLoadError(
                    f"conversation {conversation_id}: {ability}[{i}] needs a question and a "
                    "non-empty rubric of strings")
    return parsed


class BEAMAdapter(BaseAdapter):
    """BEAM: conversational long-term memory, nugget-judged (arXiv 2510.27246).

    One prompt per probing question: the whole conversation history rendered as
    a transcript, then the probing question as the next user turn (the paper's
    Section 2.1 formalism, the vanilla long-context arm). 100K split = 20
    conversations x 10 abilities x 2 questions = 400 prompts.

    Scoring is ``llm_judge`` against the served judge (``judge_port`` 8082), and
    the NUGGET LIST rides in ``scoring_config`` so the judge scores per nugget
    (0 / 0.5 / 1), not per answer. The fold that turns verdicts into a BEAM
    number is ``beam_scoring.fold_beam`` (CME-2). The probing question is
    carried too, so the judge prompt can apply its responsiveness check (BEAM's
    own harness discards it — CME-3).

    Sources, in order: HF parquet (``<data_dir>/**/<split>-*.parquet``), then the
    GitHub repo tree (``<data_dir>/**/chats/<split>/<n>/``). Loading parquet
    needs ``pyarrow`` and FAILS LOUDLY without it. No data at all is recorded as
    a degraded source, never as an empty benchmark in disguise.

    Data licence: CC BY-SA 4.0 (HF card); code MIT.

    M-12b arms (B2): ``context_mode`` is ``"full"`` (default; the history above),
    ``"rag"`` (``pair_chunk`` x BM25, the naive-memory control) or ``"trace"``
    (the same chunks through the orchestrator trace store). Defaults to
    ``$BEAM_CONTEXT_MODE``. Both memory arms use ``retrieval_top_k`` chunks
    (``$BEAM_RETRIEVAL_TOP_K``, else 10) and one prompt header. ``retriever``
    injects a callable ``(question, *, conversation_id, top_k) -> list[str]``.
    """

    suite_name = "beam"
    has_real_tiers = True
    #: All questions of a conversation share its history; keep them contiguous so the
    #: server's prompt cache is hit (M-12 B5). Adapter order is conversation order.
    preserve_order = True
    #: M-12 B3: explicit generation parameters (see docs/m12-long-context-gpu-recipe.md §5).
    #: 2048 tokens ~= 5x the longest reference answer (1,584 chars, summarization); thinking
    #: off so the budget is the answer (M-12c(3)); temperature 0 for a comparable A/B.
    inference_params = {
        "temperature": 0.0,
        "max_tokens": 2048,
        "enable_thinking": False,
        "cache_prompt": True,
        "timeout": 1800,
    }

    def __init__(self, data_dir: Path | str | None = None, split: str = "100K",
                 context_mode: str | None = None, retriever=None,
                 retrieval_top_k: int | None = None):
        super().__init__()
        if split not in BEAM_SPLITS:
            raise ValueError(f"BEAM split must be one of {BEAM_SPLITS}, got {split!r}")
        self._data_dir = Path(data_dir) if data_dir else EVAL_DIR / "beam"
        self._split = split
        self._transcripts: dict[str, str] = {}
        self._chunks: dict[str, list[str]] = {}
        self.source_kind: str | None = None
        mode = context_mode or os.environ.get(BEAM_CONTEXT_MODE_ENV) or BEAM_CONTEXT_FULL
        if mode not in BEAM_CONTEXT_MODES:
            raise ValueError(f"BEAM context_mode must be one of {BEAM_CONTEXT_MODES}, got {mode!r}")
        if retriever is not None and not callable(retriever):
            raise TypeError("retriever must be callable")
        top_k = retrieval_top_k
        if top_k is None:
            top_k = int(os.environ.get(BEAM_RETRIEVAL_TOP_K_ENV) or BEAM_DEFAULT_RETRIEVAL_TOP_K)
        if top_k < 1:
            raise ValueError("retrieval_top_k must be >= 1")
        self.context_mode = mode
        self._retriever = retriever
        self._retrieval_top_k = top_k

    # ── loading ──────────────────────────────────────────────────────────

    def _parquet_files(self) -> list[Path]:
        if not self._data_dir.exists():
            return []
        return sorted(self._data_dir.rglob(f"{self._split}-*.parquet"))

    def _repo_conversation_dirs(self) -> list[Path]:
        if not self._data_dir.exists():
            return []
        dirs = []
        for split_dir in sorted(self._data_dir.rglob(f"chats/{self._split}")):
            for conv in split_dir.iterdir():
                if (conv / "chat.json").is_file() and (
                        conv / "probing_questions" / "probing_questions.json").is_file():
                    dirs.append(conv)
        return sorted(dirs, key=lambda p: (int(p.name) if p.name.isdigit() else 10**9, p.name))

    def _load_parquet_rows(self, files: list[Path]) -> list[dict]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise BEAMLoadError(
                f"BEAM parquet present at {files[0]} but pyarrow is not importable ({exc}). "
                "Use an interpreter with pyarrow (e.g. /mnt/raid0/llm/delta-Mem/.venv/bin/python); "
                "refusing to report zero questions.") from exc
        rows: list[dict] = []
        for path in files:
            rows.extend(pq.read_table(path).to_pylist())
        return rows

    def _load_repo_rows(self, dirs: list[Path]) -> list[dict]:
        rows = []
        for conv in dirs:
            rows.append({
                "conversation_id": conv.name,
                "chat": json.loads((conv / "chat.json").read_text(encoding="utf-8")),
                "probing_questions": json.loads(
                    (conv / "probing_questions" / "probing_questions.json").read_text(
                        encoding="utf-8")),
            })
        return rows

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        self._ensure_accounting()

        parquet = self._parquet_files()
        if parquet:
            conversations = self._load_parquet_rows(parquet)
            self.source_kind = "hf_parquet"
        else:
            repo_dirs = self._repo_conversation_dirs()
            conversations = self._load_repo_rows(repo_dirs) if repo_dirs else []
            self.source_kind = "repo_tree" if repo_dirs else None

        if not conversations:
            self.record_degraded_source(
                "beam",
                f"no BEAM {self._split} data under {self._data_dir}; stage "
                f"data/{self._split}-00000-of-00001.parquet from HF {BEAM_HF_DATASET} "
                f"({BEAM_100K_PARQUET_BYTES:,} B for 100K)",
            )
            self._dataset = []
            return

        from beam_scoring import BEAM_ABILITIES

        entries: list[dict] = []
        seen: set[str] = set()
        for conv_index, conv in enumerate(conversations):
            conversation_id = str(conv.get("conversation_id") or conv_index)
            if conversation_id in seen:
                raise BEAMLoadError(f"duplicate BEAM conversation_id {conversation_id!r}")
            seen.add(conversation_id)
            messages = _beam_flatten_chat(conv.get("chat"))
            if not messages:
                raise BEAMLoadError(f"conversation {conversation_id}: empty chat")
            self._transcripts[conversation_id] = self._render_transcript(messages)
            self._chunks[conversation_id] = pair_chunks(messages)
            probing = _beam_parse_probing_questions(conv.get("probing_questions"),
                                                    conversation_id)
            for ability in BEAM_ABILITIES:
                for q_index, question in enumerate(probing[ability]):
                    entries.append({
                        "conversation_id": conversation_id,
                        "ability": ability,
                        "question_index": q_index,
                        "question": question,
                    })
        self._dataset = entries
        if entries and self.context_mode != BEAM_CONTEXT_FULL and self._retriever is None:
            if self.context_mode == BEAM_CONTEXT_RAG:
                self._retriever = BM25PairChunkRetriever(self._chunks)
            else:
                # Raises TraceRetrieverUnavailable; the arm never degrades to another arm.
                self._retriever = TracePairChunkRetriever(
                    self._chunks, store_id=f"{self._split}-{self.source_kind}")

    @staticmethod
    def _render_transcript(messages: list[dict]) -> str:
        return "\n\n".join(render_message(msg) for msg in messages)

    def retriever_name(self) -> str | None:
        if self.context_mode == BEAM_CONTEXT_FULL:
            return None
        return getattr(self._retriever, "name", None) or "injected"

    def provenance(self) -> dict:
        record = {
            "suite": self.suite_name,
            "split": self._split,
            "beam_source": self.source_kind,
            "context_mode": self.context_mode,
        }
        if self.context_mode != BEAM_CONTEXT_FULL:
            record.update({"chunking": CHUNKING, "retrieval_top_k": self._retrieval_top_k,
                           "retriever": self.retriever_name()})
        return record

    # ── prompts ──────────────────────────────────────────────────────────

    def _get_tier_for_index(self, idx: int) -> int:
        difficulty = str(self._dataset[idx]["question"].get("difficulty", "medium")).lower()
        return _BEAM_DIFFICULTY_TIER.get(difficulty, 2)

    @staticmethod
    def _reference(question: dict) -> tuple[str, str]:
        for field in _BEAM_REFERENCE_FIELDS:
            value = question.get(field)
            if isinstance(value, str) and value.strip():
                return value, field
        return "", ""

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        from beam_scoring import (
            FOLD_NAME, FOLD_VERSION, JUDGE_PROMPT_VERSION, NUGGET_VERDICTS,
        )

        q = row["question"]
        conversation_id = row["conversation_id"]
        ability = row["ability"]
        question_text = str(q["question"]).strip()
        nuggets = [str(item) for item in q["rubric"]]
        transcript = self._transcripts[conversation_id]
        reference, reference_field = self._reference(q)

        retrieved_chunks = None
        if self.context_mode == BEAM_CONTEXT_FULL:
            prompt = BEAM_FULL_HEADER + f"{transcript}\n\n---\n\nUser: {question_text}"
        else:
            if self._retriever is None:
                raise RuntimeError(
                    f"context_mode={self.context_mode!r} has no retriever; refusing to build a "
                    "prompt that would silently be another arm")
            excerpts = [
                str(x).strip() for x in self._retriever(
                    question_text, conversation_id=conversation_id,
                    top_k=self._retrieval_top_k)
                if str(x).strip()
            ]
            retrieved_chunks = len(excerpts)
            body = "\n\n".join(excerpts) if excerpts else BEAM_NO_EXCERPTS
            prompt = BEAM_RETRIEVED_HEADER + f"{body}\n\n---\n\nUser: {question_text}"
        scoring_config = {
            "judge_port": 8082,
            "per_nugget": True,
            "nuggets": nuggets,
            "nugget_verdict_scale": list(NUGGET_VERDICTS),
            "probing_question": question_text,
            "ability": ability,
            "fold": FOLD_NAME,
            "fold_version": FOLD_VERSION,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
        }
        if ability == "event_ordering":
            # BEAM's reference fold scores ordering by tau_norm against the rubric list.
            scoring_config["tau_reference"] = "nuggets"

        return {
            "id": f"beam_{self._split}_{conversation_id}_{ability}_{row['question_index']}",
            "suite": "beam",
            "prompt": prompt,
            "context": "",
            "expected": reference,
            "scoring": [],
            "image_path": "",
            "tier": self._get_tier_for_index(idx),
            "scoring_method": "llm_judge",
            "scoring_config": scoring_config,
            "metadata": {
                "split": self._split,
                "conversation_id": conversation_id,
                "ability": ability,
                "question_index": row["question_index"],
                "difficulty": q.get("difficulty", ""),
                "n_nuggets": len(nuggets),
                "reference_field": reference_field,
                "context_length_chars": len(transcript),
                "beam_source": self.source_kind,
                "context_mode": self.context_mode,
                "retrieved_chunks": retrieved_chunks,
                "prompt_chars": len(prompt),
            },
            "provenance": self.provenance(),
        }
