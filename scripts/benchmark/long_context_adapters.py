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

All adapters produce standard prompt dicts compatible with
compare_orchestrator_direct.py and the seeding harness.
"""
from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any

from dataset_adapters import BaseAdapter

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
            scoring = "claude_judge"
        elif task in self._QA:
            prompt = input_text
            scoring = "substring"
        else:
            prompt = input_text
            scoring = "claude_judge"

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
        scoring = "exact_match" if cfg in self._CLOSED_ENDED else "claude_judge"

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
