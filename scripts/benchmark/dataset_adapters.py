#!/usr/bin/env python3
from __future__ import annotations

"""On-the-fly benchmark question sampling from HuggingFace datasets.

Provides adapters for all debug suite categories, loading real benchmark
questions directly from HuggingFace cached datasets. Each adapter knows
how to extract (question, expected_answer, scoring_method, scoring_config)
from its source dataset's schema.

Supported suites and their data sources:
  - general:              MMLU (cais/mmlu, 14,042 questions)
  - math:                 GSM8K (gsm8k, 1,319) + MATH-500 (HuggingFaceH4/MATH-500, 500)
  - coder:                HumanEval (openai_humaneval, 164) + MBPP (mbpp, 500)
  - thinking:             ARC-Challenge (allenai/ai2_arc, 1,172) + HellaSwag (Rowan/hellaswag, 10,042)
  - instruction_precision: IFEval (google/IFEval, 541)
  - vl:                   OCRBench + ChartQA (via extract_vl_debug_suite.py, 3,500)
  - physics:              PHYBench (Eureka-Lab/PHYBench, 100 with answers)
  - physreason:           PhysReason (zhibei1204/PhysReason, 1,200 problems, ~3,100 sub-questions)
  - aime:                 AIME 2024 (Maxwell-Jia/AIME_2024, 30) + AIME 2025 (opencompass/AIME2025, 30)
  - aime25:               AIME 2025 only (opencompass/AIME2025, 30) — no 2024 contamination
  - gpqa_diamond:         GPQA Diamond hard subset (198) — MC from ankner/gpqa, membership from hendrydong mirror
  - olympiadbench:        OlympiadBench (math-ai/olympiadbench, ~674 text-only math competition)
  - agentic:              No public dataset (stays YAML-based)
  - long_context:         Synthetic (stays YAML-based)

Usage:
    from dataset_adapters import get_adapter, ADAPTER_SUITES

    adapter = get_adapter("math")
    questions = adapter.sample(n=10, seed=42)
    # Returns list of prompt dicts compatible with compare_orchestrator_direct.py
"""

import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)

# Suites that have dataset adapters (vs YAML-only)
ADAPTER_SUITES = {
    "general", "math", "coder", "thinking", "instruction_precision", "vl",
    "gaia", "cruxeval", "bigcodebench",
    # Phase 1 hard benchmarks (mode-advantage signal)
    "gpqa", "mmlu_pro", "simpleqa", "hotpotqa", "livecodebench",
    # Phase 2 hard benchmarks
    "debugbench", "usaco",
    # Phase 3: physics reasoning
    "physics",
    # Phase 3: multimodal physics reasoning
    "physreason",
    # Phase 4: competition math
    "aime", "olympiadbench",
    # Architect bench: uncontaminated / hard-subset variants of the above
    "aime25", "gpqa_diamond", "gpqa_diamond_cot",
    # Phase 5: long-context evaluation datasets
    "longbench", "zeroscrolls", "leval", "ruler", "needle_parameterized",
    # Phase 6: knowledge reliability / hallucination detection
    "omniscience",
    # Phase 6: long-context multi-document reasoning
    "aa_lcr",
    # Phase 7: document extraction quality
    "document_extraction",
    # EV-3: verifier benchmarks (NVIDIA Scoring-Verifiers / HE-R+)
    "scoring_verifiers",
    # P3b: episodic memory (Tulving Benchmark, arXiv 2501.13121)
    "tulving_episodic",
    # K-LCM-1: LongCoT-Mini easy-split deterministic reasoning (intake-386/RE-4)
    "longcot_mini",
}

# Suites that stay YAML-based (no public dataset or intentionally synthetic)
YAML_ONLY_SUITES = {
    "agentic", "long_context", "mode_advantage", "mode_advantage_hard", "real_suite_v1",
    "skill_transfer", "web_research",
}


def _get_long_context_adapter(class_name: str):
    """Lazy-import a long-context adapter class to avoid loading HF datasets at import time."""
    try:
        from long_context_adapters import (
            LongBenchAdapter, ZeroSCROLLSAdapter, LEvalAdapter,
            RULERAdapter, NeedleAdapter,
        )
        return {
            "LongBenchAdapter": LongBenchAdapter,
            "ZeroSCROLLSAdapter": ZeroSCROLLSAdapter,
            "LEvalAdapter": LEvalAdapter,
            "RULERAdapter": RULERAdapter,
            "NeedleAdapter": NeedleAdapter,
        }[class_name]
    except ImportError:
        return None


def _get_scoring_verifiers_adapter():
    """Lazy-import ScoringVerifiersAdapter (EV-3)."""
    try:
        from scoring_verifiers_adapter import ScoringVerifiersAdapter
        return ScoringVerifiersAdapter
    except ImportError:
        return None


def _get_tulving_episodic_adapter():
    """Lazy-import TulvingEpisodicAdapter (P3b)."""
    try:
        from tulving_episodic_adapter import TulvingEpisodicAdapter
        return TulvingEpisodicAdapter
    except ImportError:
        return None


def _get_document_extraction_adapter():
    """Lazy-import OpenDataLoader-bench document extraction adapter."""
    try:
        from document_extraction_adapter import DocumentExtractionDatasetAdapter
        return DocumentExtractionDatasetAdapter
    except ImportError:
        return None


def _get_longcot_mini_adapter():
    """Lazy-import LongCoT-Mini adapter (K-LCM-1 / intake-386 / RE-4)."""
    try:
        from longcot_mini_adapter import LongCoTMiniAdapter
        return LongCoTMiniAdapter
    except ImportError:
        return None


def get_adapter(suite: str) -> Optional["BaseAdapter"]:
    """Get the dataset adapter for a suite, or None if YAML-only."""
    adapters = {
        "general": MMLUAdapter,
        "math": MathAdapter,
        "coder": CoderAdapter,
        "thinking": ThinkingAdapter,
        "instruction_precision": IFEvalAdapter,
        "vl": VLAdapter,
        "gaia": GaiaAdapter,
        "cruxeval": CRUXEvalAdapter,
        "bigcodebench": BigCodeBenchAdapter,
        # Phase 1 hard benchmarks
        "gpqa": GPQAAdapter,
        "mmlu_pro": MMLUProAdapter,
        "simpleqa": SimpleQAAdapter,
        "hotpotqa": HotpotQAAdapter,
        "livecodebench": LiveCodeBenchAdapter,
        # Phase 2 hard benchmarks
        "debugbench": DebugBenchAdapter,
        "usaco": USACOAdapter,
        # Phase 3: physics reasoning
        "physics": PHYBenchAdapter,
        "physreason": PhysReasonAdapter,
        # Phase 4: competition math
        "aime": AIMEAdapter,
        "olympiadbench": OlympiadBenchAdapter,
        # Architect bench: AIME 2025 only (no 2024 contamination) + GPQA Diamond
        "aime25": AIME25Adapter,
        "gpqa_diamond": GPQADiamondAdapter,
        "gpqa_diamond_cot": GPQADiamondCoTAdapter,
        # Phase 6: knowledge reliability / hallucination detection
        "omniscience": AAOmniscienceAdapter,
        # Phase 6: long-context multi-document reasoning
        "aa_lcr": AALCRAdapter,
        # Phase 7: document extraction quality
        "document_extraction": _get_document_extraction_adapter(),
        # Phase 5: long-context evaluation datasets
        "longbench": _get_long_context_adapter("LongBenchAdapter"),
        "zeroscrolls": _get_long_context_adapter("ZeroSCROLLSAdapter"),
        "leval": _get_long_context_adapter("LEvalAdapter"),
        "ruler": _get_long_context_adapter("RULERAdapter"),
        "needle_parameterized": _get_long_context_adapter("NeedleAdapter"),
        # EV-3: verifier benchmarks (NVIDIA Scoring-Verifiers / HE-R+)
        "scoring_verifiers": _get_scoring_verifiers_adapter(),
        # P3b: episodic memory (Tulving Benchmark, arXiv 2501.13121)
        "tulving_episodic": _get_tulving_episodic_adapter(),
        # K-LCM-1: LongCoT-Mini easy-split deterministic reasoning
        "longcot_mini": _get_longcot_mini_adapter(),
    }
    cls = adapters.get(suite)
    if cls is None:
        return None
    return cls()


class BaseAdapter:
    """Base class for dataset adapters."""

    suite_name: str = ""
    _dataset = None

    # Adapters with real difficulty data should set this True
    has_real_tiers: bool = False

    def __init__(self):
        self._reset_accounting()

    def _reset_accounting(self) -> None:
        self.dropped_rows: int = 0
        self.dropped_by_reason: dict[str, int] = {}
        self.degraded_sources: list[dict[str, str]] = []

    def _ensure_accounting(self) -> None:
        if not hasattr(self, "dropped_rows"):
            self._reset_accounting()

    def drop_row(self, reason: str, *, source: str | None = None) -> None:
        """Record one source row that did not become a benchmark prompt."""
        self._ensure_accounting()
        reason_key = f"{source}:{reason}" if source else reason
        self.dropped_rows += 1
        self.dropped_by_reason[reason_key] = self.dropped_by_reason.get(reason_key, 0) + 1

    def record_degraded_source(self, source: str, error: BaseException | str) -> None:
        """Record a failed optional sub-source without hiding the degradation."""
        self._ensure_accounting()
        message = str(error)
        self.degraded_sources.append({"source": source, "error": message})
        logger.warning(
            "[adapter:%s] degraded source %s: %s",
            self.suite_name or self.__class__.__name__,
            source,
            message,
        )

    def accounting_summary(self) -> dict:
        """Return loss/degradation metadata for pool headers and tests."""
        self._ensure_accounting()
        summary = {
            "dropped_rows": self.dropped_rows,
            "dropped_by_reason": dict(sorted(self.dropped_by_reason.items())),
            "degraded_sources": list(self.degraded_sources),
        }
        if hasattr(self, "source_counts"):
            summary["source_counts"] = dict(getattr(self, "source_counts"))
        return summary

    def _ensure_loaded(self):
        raise NotImplementedError

    @property
    def total_available(self) -> int:
        self._ensure_loaded()
        return len(self._dataset) if self._dataset is not None else 0

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        raise NotImplementedError

    def sample(self, n: int = 10, seed: int = 42, stratify: bool = False) -> list[dict]:
        """Sample n questions. If stratify=True AND adapter has real tiers,
        draw equal counts per tier for balanced difficulty distribution."""
        self._ensure_loaded()
        if not self._dataset:
            return []
        if stratify and self.has_real_tiers:
            return self._stratified_sample(n, seed)
        rng = random.Random(seed)
        indices = rng.sample(range(len(self._dataset)), min(n, len(self._dataset)))
        return [self._row_to_prompt(i, self._dataset[i]) for i in indices]

    def _stratified_sample(self, n: int, seed: int) -> list[dict]:
        """Draw equal questions per tier. Requires _get_tier_for_index()."""
        rng = random.Random(seed)
        # Bucket indices by tier
        tier_buckets: dict[int, list[int]] = {}
        for i in range(len(self._dataset)):
            t = self._get_tier_for_index(i)
            tier_buckets.setdefault(t, []).append(i)

        tiers = sorted(tier_buckets.keys())
        if not tiers:
            return []

        # Equal share per tier, remainder distributed round-robin
        per_tier = n // len(tiers)
        remainder = n % len(tiers)

        results = []
        for i, t in enumerate(tiers):
            bucket = tier_buckets[t]
            count = per_tier + (1 if i < remainder else 0)
            count = min(count, len(bucket))
            indices = rng.sample(bucket, count)
            results.extend(self._row_to_prompt(idx, self._dataset[idx]) for idx in indices)

        rng.shuffle(results)
        return results

    def extract_all(self) -> list[dict]:
        """Extract ALL questions from this adapter as prompt dicts.

        Calls _ensure_loaded() then iterates the full dataset through
        _row_to_prompt(). Used by question_pool.py to pre-extract the
        complete question corpus into a JSONL file.
        """
        self._ensure_accounting()
        self._ensure_loaded()
        if not self._dataset:
            return []
        results = []
        for i in range(len(self._dataset)):
            try:
                row = self._dataset[i] if not isinstance(self._dataset[i], int) else {}
                prompt = self._row_to_prompt(i, row)
                if prompt:
                    results.append(prompt)
                else:
                    self.drop_row("empty_prompt")
            except Exception:
                self.drop_row("row_to_prompt_exception")
                continue
        if self.dropped_rows:
            logger.warning(
                "[adapter:%s] dropped %d source row(s): %s",
                self.suite_name or self.__class__.__name__,
                self.dropped_rows,
                self.dropped_by_reason,
            )
        return results

    def _get_tier_for_index(self, idx: int) -> int:
        """Return tier for a given dataset index. Override in adapters with real tiers."""
        return 1


# ── MMLU (General Knowledge) ─────────────────────────────────────────────


class MMLUAdapter(BaseAdapter):
    """MMLU: 14,042 multiple-choice questions across 57 subjects."""

    suite_name = "general"
    has_real_tiers = True  # Subject-based difficulty mapping
    CHOICE_LABELS = ["A", "B", "C", "D"]

    HARD_SUBJECTS = {
        "abstract_algebra", "college_mathematics", "formal_logic",
        "college_physics", "electrical_engineering", "machine_learning",
        "conceptual_physics", "college_chemistry", "anatomy",
    }
    EASY_SUBJECTS = {
        "high_school_geography", "high_school_us_history",
        "miscellaneous", "us_foreign_policy",
    }

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._dataset = hf.load_dataset("cais/mmlu", "all", split="test")
        except Exception as e:
            print(f"  [adapter] MMLU load failed: {e}")
            self._dataset = []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question = row["question"]
        choices = row["choices"]
        answer_idx = row["answer"]
        subject = row.get("subject", "general")

        # Build multiple-choice prompt
        prompt_lines = [question, ""]
        for i, choice in enumerate(choices):
            prompt_lines.append(f"{self.CHOICE_LABELS[i]}) {choice}")
        prompt_lines.append("")
        prompt_lines.append("Answer with the letter only (A, B, C, or D).")

        expected = self.CHOICE_LABELS[answer_idx]

        # Tier based on subject difficulty
        if subject in self.HARD_SUBJECTS:
            tier = 3
        elif subject in self.EASY_SUBJECTS:
            tier = 1
        else:
            tier = 2

        return {
            "id": f"mmlu_{subject}_{idx:05d}",
            "suite": "general",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": expected,
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "multiple_choice",
            "scoring_config": {},
        }

    def _get_tier_for_index(self, idx: int) -> int:
        subject = self._dataset[idx].get("subject", "general")
        if subject in self.HARD_SUBJECTS:
            return 3
        elif subject in self.EASY_SUBJECTS:
            return 1
        return 2


# ── GSM8K + MATH-500 (Math) ──────────────────────────────────────────────


class MathAdapter(BaseAdapter):
    """GSM8K (1,319) + MATH-500 (500) = 1,819 math problems."""

    suite_name = "math"
    has_real_tiers = True  # GSM8K=T1, MATH-500 level 1-3=T2, level 4-5=T3
    _gsm8k = None
    _math500 = None
    source_counts: dict[str, int] = {}

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        self._ensure_accounting()
        self.source_counts = {"gsm8k": 0, "math500": 0}
        try:
            import datasets as hf  # type: ignore
        except Exception as e:
            self.record_degraded_source("datasets_import", e)
            print(f"  [adapter] Math datasets load failed: {e}")
            self._gsm8k = []
            self._math500 = []
            self._dataset = []
            return

        try:
            self._gsm8k = hf.load_dataset("gsm8k", "main", split="test")
        except Exception as e:
            self.record_degraded_source("gsm8k", e)
            print(f"  [adapter] GSM8K load failed: {e}")
            self._gsm8k = []

        try:
            self._math500 = hf.load_dataset("HuggingFaceH4/MATH-500", split="test")
        except Exception as e:
            self.record_degraded_source("math500", e)
            print(f"  [adapter] MATH-500 load failed: {e}")
            self._math500 = []

        n_gsm8k = len(self._gsm8k) if self._gsm8k else 0
        n_math500 = len(self._math500) if self._math500 else 0
        self.source_counts = {"gsm8k": n_gsm8k, "math500": n_math500}
        self._dataset = list(range(n_gsm8k + n_math500))

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        # idx is from unified list; row is ignored, we index directly
        gsm8k_len = len(self._gsm8k) if self._gsm8k else 0

        if idx < gsm8k_len:
            return self._gsm8k_prompt(idx, self._gsm8k[idx])
        else:
            math_idx = idx - gsm8k_len
            return self._math500_prompt(math_idx, self._math500[math_idx])

    def _get_tier_for_index(self, idx: int) -> int:
        gsm8k_len = len(self._gsm8k) if self._gsm8k else 0
        if idx < gsm8k_len:
            return 1  # GSM8K = grade-school
        math_idx = idx - gsm8k_len
        if self._math500 and math_idx < len(self._math500):
            level = self._math500[math_idx].get("level", 3)
            return 2 if level <= 3 else 3
        return 1

    def sample(self, n: int = 10, seed: int = 42, stratify: bool = False) -> list[dict]:
        self._ensure_loaded()
        if not self._dataset:
            return []
        if stratify:
            return self._stratified_sample(n, seed)
        rng = random.Random(seed)
        # Split: ~60% GSM8K, ~40% MATH-500
        gsm8k_len = len(self._gsm8k) if self._gsm8k else 0
        math_len = len(self._math500) if self._math500 else 0

        n_gsm = min(int(n * 0.6), gsm8k_len)
        n_math = min(n - n_gsm, math_len)
        if n_math < n - n_gsm:
            n_gsm = min(n - n_math, gsm8k_len)

        results = []
        if n_gsm > 0:
            gsm_indices = rng.sample(range(gsm8k_len), n_gsm)
            results.extend(self._gsm8k_prompt(i, self._gsm8k[i]) for i in gsm_indices)
        if n_math > 0:
            math_indices = rng.sample(range(math_len), n_math)
            results.extend(self._math500_prompt(i, self._math500[i]) for i in math_indices)

        rng.shuffle(results)
        return results

    @staticmethod
    def _extract_gsm8k_answer(answer_text: str) -> str:
        """Extract numeric answer from GSM8K solution (<answer> tags or ####)."""
        # Try <answer> tag format first, fall back to legacy ####
        match = re.search(r"<answer>(.*?)</answer>", answer_text, re.DOTALL)
        if not match:
            match = re.search(r"####\s*(.+)", answer_text)
        if match:
            return match.group(1).strip().replace(",", "")
        return answer_text.strip()

    def _gsm8k_prompt(self, idx: int, row: dict) -> dict:
        question = row["question"]
        answer_text = row["answer"]
        expected = self._extract_gsm8k_answer(answer_text)

        return {
            "id": f"gsm8k_{idx:05d}",
            "suite": "math",
            "prompt": question + "\n\nSolve step by step. Put your final numeric answer inside <answer></answer> tags.",
            "context": "",
            "expected": expected,
            "scoring": [],
            "image_path": "",
            "tier": 1,  # GSM8K is grade-school level
            "scoring_method": "exact_match",
            "scoring_config": {"extract_pattern": r"<answer>(.*?)</answer>"},
        }

    def _math500_prompt(self, idx: int, row: dict) -> dict:
        problem = row["problem"]
        answer = row.get("answer", "")
        level = row.get("level", 3)
        subject = row.get("subject", "")

        # Map MATH difficulty level to tier
        tier = 2 if level <= 3 else 3

        return {
            "id": f"math500_{subject}_{idx:05d}",
            "suite": "math",
            "prompt": problem + "\n\nPut your final answer in \\boxed{}.",
            "context": "",
            "expected": answer,
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "substring",
            "scoring_config": {"case_sensitive": False},
        }


# ── HumanEval + MBPP (Coder) ─────────────────────────────────────────────


class CoderAdapter(BaseAdapter):
    """HumanEval (164) + MBPP (500) = 664 coding problems."""

    suite_name = "coder"
    _humaneval = None
    _mbpp = None

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._humaneval = hf.load_dataset("openai_humaneval", split="test")
            self._mbpp = hf.load_dataset("mbpp", split="test")
            self._dataset = list(range(len(self._humaneval) + len(self._mbpp)))
        except Exception as e:
            print(f"  [adapter] Coder datasets load failed: {e}")
            self._dataset = []

    def sample(self, n: int = 10, seed: int = 42, stratify: bool = False) -> list[dict]:
        self._ensure_loaded()
        if not self._dataset:
            return []
        rng = random.Random(seed)
        he_len = len(self._humaneval) if self._humaneval else 0
        mbpp_len = len(self._mbpp) if self._mbpp else 0

        n_he = min(int(n * 0.4), he_len)
        n_mbpp = min(n - n_he, mbpp_len)
        if n_mbpp < n - n_he:
            n_he = min(n - n_mbpp, he_len)

        results = []
        if n_he > 0:
            he_indices = rng.sample(range(he_len), n_he)
            results.extend(self._humaneval_prompt(i) for i in he_indices)
        if n_mbpp > 0:
            mbpp_indices = rng.sample(range(mbpp_len), n_mbpp)
            results.extend(self._mbpp_prompt(i) for i in mbpp_indices)

        rng.shuffle(results)
        return results

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        he_len = len(self._humaneval) if self._humaneval else 0
        if idx < he_len:
            return self._humaneval_prompt(idx)
        return self._mbpp_prompt(idx - he_len)

    def _humaneval_prompt(self, idx: int) -> dict:
        row = self._humaneval[idx]
        prompt_text = row["prompt"]
        canonical = row["canonical_solution"]
        test_code = row["test"]
        entry_point = row["entry_point"]
        task_id = row["task_id"]

        # Build a prompt that asks to complete the function
        full_prompt = (
            f"Complete the following Python function:\n\n"
            f"```python\n{prompt_text}```\n\n"
            f"Write only the function body (the part after the signature)."
        )

        return {
            "id": f"humaneval_{task_id.replace('/', '_')}",
            "suite": "coder",
            "prompt": full_prompt,
            "context": "",
            "expected": entry_point,
            "scoring": [],
            "image_path": "",
            "tier": 2,
            "scoring_method": "substring",
            "scoring_config": {"case_sensitive": True, "substring": entry_point},
        }

    def _mbpp_prompt(self, idx: int) -> dict:
        row = self._mbpp[idx]
        task_id = row["task_id"]
        text = row["text"]
        test_list = row.get("test_list", [])

        # Include test cases as hints
        test_hint = ""
        if test_list:
            test_hint = "\n\nTest cases:\n" + "\n".join(f"  {t}" for t in test_list[:3])

        full_prompt = (
            f"{text}{test_hint}\n\n"
            f"Write a Python function to solve this."
        )

        # Extract expected function name from test cases
        func_name = ""
        if test_list:
            match = re.search(r"assert\s+(\w+)\(", test_list[0])
            if match:
                func_name = match.group(1)

        return {
            "id": f"mbpp_{task_id:04d}",
            "suite": "coder",
            "prompt": full_prompt,
            "context": "",
            "expected": func_name or "def",
            "scoring": [],
            "image_path": "",
            "tier": 1,
            "scoring_method": "substring",
            "scoring_config": {"case_sensitive": True, "substring": func_name or "def"},
        }


# ── ARC-Challenge + HellaSwag (Thinking) ──────────────────────────────────


class ThinkingAdapter(BaseAdapter):
    """ARC-Challenge (1,172) + HellaSwag (10,042) = 11,214 reasoning questions."""

    suite_name = "thinking"
    _arc = None
    _hellaswag = None

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._arc = hf.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            self._hellaswag = hf.load_dataset("Rowan/hellaswag", split="validation")
            self._dataset = list(range(len(self._arc) + len(self._hellaswag)))
        except Exception as e:
            print(f"  [adapter] Thinking datasets load failed: {e}")
            self._dataset = []

    def sample(self, n: int = 10, seed: int = 42, stratify: bool = False) -> list[dict]:
        self._ensure_loaded()
        if not self._dataset:
            return []
        rng = random.Random(seed)
        arc_len = len(self._arc) if self._arc else 0
        hs_len = len(self._hellaswag) if self._hellaswag else 0

        n_arc = min(int(n * 0.5), arc_len)
        n_hs = min(n - n_arc, hs_len)
        if n_hs < n - n_arc:
            n_arc = min(n - n_hs, arc_len)

        results = []
        if n_arc > 0:
            arc_indices = rng.sample(range(arc_len), n_arc)
            results.extend(self._arc_prompt(i) for i in arc_indices)
        if n_hs > 0:
            hs_indices = rng.sample(range(hs_len), n_hs)
            results.extend(self._hellaswag_prompt(i) for i in hs_indices)

        rng.shuffle(results)
        return results

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        arc_len = len(self._arc) if self._arc else 0
        if idx < arc_len:
            return self._arc_prompt(idx)
        return self._hellaswag_prompt(idx - arc_len)

    CHOICE_LABELS = ["A", "B", "C", "D", "E"]

    def _arc_prompt(self, idx: int) -> dict:
        row = self._arc[idx]
        question = row["question"]
        choices_data = row["choices"]
        answer_key = row["answerKey"]
        qid = row["id"]

        # ARC choices format: {"text": [...], "label": [...]}
        labels = choices_data["label"]
        texts = choices_data["text"]

        prompt_lines = [question, ""]
        for label, text in zip(labels, texts):
            prompt_lines.append(f"{label}) {text}")
        prompt_lines.append("")
        prompt_lines.append("Answer with the letter only.")

        return {
            "id": f"arc_{qid}",
            "suite": "thinking",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": answer_key,
            "scoring": [],
            "image_path": "",
            "tier": 2,
            "scoring_method": "multiple_choice",
            "scoring_config": {},
        }

    def _hellaswag_prompt(self, idx: int) -> dict:
        row = self._hellaswag[idx]
        context = row["ctx"]
        endings = row["endings"]
        label = row["label"]
        ind = row["ind"]

        prompt_lines = [
            "Choose the most plausible continuation:",
            "",
            f"Context: {context}",
            "",
        ]
        for i, ending in enumerate(endings):
            prompt_lines.append(f"{self.CHOICE_LABELS[i]}) {ending}")
        prompt_lines.append("")
        prompt_lines.append("Answer with the letter only (A, B, C, or D).")

        expected = self.CHOICE_LABELS[int(label)] if isinstance(label, (int, str)) else "A"

        return {
            "id": f"hellaswag_{ind:05d}",
            "suite": "thinking",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": expected,
            "scoring": [],
            "image_path": "",
            "tier": 1,
            "scoring_method": "multiple_choice",
            "scoring_config": {},
        }


# ── IFEval (Instruction Precision) ───────────────────────────────────────


class IFEvalAdapter(BaseAdapter):
    """IFEval: 541 instruction-following prompts with verifiable constraints."""

    suite_name = "instruction_precision"
    has_real_tiers = True  # Tier from constraint count: 1→T1, 2-3→T2, 4+→T3

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._dataset = hf.load_dataset("google/IFEval", split="train")
        except Exception as e:
            print(f"  [adapter] IFEval load failed: {e}")
            self._dataset = []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        prompt = row["prompt"]
        key = row["key"]
        instruction_ids = row.get("instruction_id_list", [])
        kwargs_list = row.get("kwargs", [])

        # IFEval doesn't have simple expected answers — it has constraint verifiers.
        # We extract the first instruction as the primary constraint to check.
        primary_constraint = instruction_ids[0] if instruction_ids else "unknown"

        # Build scoring config from IFEval's constraint types
        scoring_method, scoring_config = self._constraint_to_scoring(
            primary_constraint, kwargs_list[0] if kwargs_list else {}
        )

        # Determine tier from constraint complexity
        n_constraints = len(instruction_ids)
        tier = 1 if n_constraints <= 1 else (2 if n_constraints <= 3 else 3)

        return {
            "id": f"ifeval_{key}",
            "suite": "instruction_precision",
            "prompt": prompt,
            "context": "",
            "expected": "",  # IFEval uses programmatic verification
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": scoring_method,
            "scoring_config": scoring_config,
            "ifeval_instructions": instruction_ids,
            "ifeval_kwargs": kwargs_list,
        }

    @staticmethod
    def _constraint_to_scoring(constraint_id: str, kwargs: dict) -> tuple[str, dict]:
        """Map IFEval constraint type to our scoring system."""
        # IFEval constraints: https://github.com/google-research/google-research/tree/master/instruction_following_eval
        if "no_comma" in constraint_id:
            return "programmatic", {"verifier": "no_comma"}
        elif "number_highlighted_sections" in constraint_id:
            n = kwargs.get("num_highlights", 1)
            return "programmatic", {"verifier": "highlighted_sections", "count": n}
        elif "number_paragraphs" in constraint_id:
            n = kwargs.get("num_paragraphs", 1)
            return "programmatic", {"verifier": "paragraph_count", "count": n}
        elif "number_words" in constraint_id or "length" in constraint_id:
            n = kwargs.get("num_words")
            rel = kwargs.get("relation", "at_least")
            return "programmatic", {"verifier": "word_count", "count": n, "relation": rel}
        elif "number_sentences" in constraint_id:
            n = kwargs.get("num_sentences")
            rel = kwargs.get("relation", "at_least")
            return "programmatic", {"verifier": "sentence_count", "count": n, "relation": rel}
        elif "postscript" in constraint_id:
            return "substring", {"case_sensitive": False, "substring": "P.S."}
        elif "title" in constraint_id:
            return "programmatic", {"verifier": "has_title"}
        elif "json_format" in constraint_id or "json" in constraint_id:
            return "programmatic", {"verifier": "json_valid"}
        elif "number_placeholders" in constraint_id:
            n = kwargs.get("num_placeholders", 1)
            return "programmatic", {"verifier": "placeholder_count", "count": n}
        elif "bullet_list" in constraint_id or "number_bullet" in constraint_id:
            n = kwargs.get("num_bullets")
            return "programmatic", {"verifier": "bullet_count", "count": n}
        elif "keywords" in constraint_id:
            kw = kwargs.get("keywords", [])
            return "programmatic", {"verifier": "contains_keywords", "keywords": kw}
        elif "forbidden" in constraint_id:
            fw = kwargs.get("forbidden_words", [])
            return "programmatic", {"verifier": "no_forbidden_words", "forbidden": fw}
        elif "language" in constraint_id:
            lang = kwargs.get("language", "en")
            return "programmatic", {"verifier": "language", "language": lang}
        else:
            # Generic fallback — just check response is non-empty
            return "programmatic", {"verifier": "non_empty", "constraint": constraint_id}

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        n_constraints = len(row.get("instruction_id_list", []))
        return 1 if n_constraints <= 1 else (2 if n_constraints <= 3 else 3)


# ── VL (Vision-Language) ──────────────────────────────────────────────────


class VLAdapter(BaseAdapter):
    """VL: delegates to extract_vl_debug_suite.VLDatasetAdapter (3,500 questions)."""

    suite_name = "vl"
    _vl_adapter = None

    def _ensure_loaded(self):
        if self._vl_adapter is not None:
            return
        try:
            from extract_vl_debug_suite import VLDatasetAdapter
            self._vl_adapter = VLDatasetAdapter()
            self._dataset = list(range(self._vl_adapter.total_available))
        except ImportError:
            print("  [adapter] VL adapter not available (extract_vl_debug_suite.py)")
            self._dataset = []

    @property
    def total_available(self) -> int:
        self._ensure_loaded()
        return self._vl_adapter.total_available if self._vl_adapter else 0

    def sample(self, n: int = 10, seed: int = 42, stratify: bool = False) -> list[dict]:
        self._ensure_loaded()
        if self._vl_adapter:
            return self._vl_adapter.sample(n=n, seed=seed, extract_images=True)
        return []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        return {}  # Not used — sample() delegates directly

    def extract_all(self) -> list[dict]:
        """VL adapter delegates to VLDatasetAdapter — sample everything."""
        self._ensure_loaded()
        if self._vl_adapter:
            try:
                return self._vl_adapter.sample(
                    n=self._vl_adapter.total_available, seed=0, extract_images=True,
                )
            except Exception:
                return []
        return []


# ── GAIA (Multi-step tool use) ───────────────────────────────────────────


class GaiaAdapter(BaseAdapter):
    """GAIA: 165 dev questions requiring multi-step reasoning and tool use.

    Source: gaia-benchmark/GAIA on HuggingFace (CC-BY-4.0).
    Questions have exact-match answers (number, name, or short string).
    Levels 1-3 map to tiers T1-T3.

    File attachments are staged to /mnt/raid0/llm/tmp/gaia/{question_id}/
    so REPL mode can access them.
    """

    suite_name = "gaia"
    has_real_tiers = True
    _STAGING_DIR = Path("/mnt/raid0/llm/tmp/gaia")
    # Skip questions requiring audio/video processing
    _SKIP_EXTENSIONS = {".mp3", ".wav", ".mp4", ".avi", ".mov", ".flac", ".ogg"}

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            ds = hf.load_dataset(
                "gaia-benchmark/GAIA", "2023_all", split="validation",
            )
            # Filter out questions with unsupported file types
            filtered = []
            for i, row in enumerate(ds):
                file_name = row.get("file_name", "") or ""
                if file_name:
                    ext = Path(file_name).suffix.lower()
                    if ext in self._SKIP_EXTENSIONS:
                        continue
                filtered.append(row)
            self._dataset = filtered
        except Exception as e:
            print(f"  [adapter] GAIA load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        level = self._dataset[idx].get("Level", 1)
        return min(max(int(level), 1), 3)

    def _stage_file(self, question_id: str, row: dict) -> str:
        """Stage attached file to temp dir. Returns path hint or empty string."""
        file_name = row.get("file_name", "") or ""
        file_bytes = row.get("file_path", "") or ""
        if not file_name:
            return ""

        staging = self._STAGING_DIR / question_id
        staging.mkdir(parents=True, exist_ok=True)
        dest = staging / file_name

        if not dest.exists():
            # file_path in GAIA dataset is the actual path to the file
            # In HF datasets, this may be a local cache path
            try:
                if isinstance(file_bytes, (str, Path)) and Path(file_bytes).exists():
                    import shutil
                    shutil.copy2(file_bytes, dest)
                elif isinstance(file_bytes, bytes):
                    dest.write_bytes(file_bytes)
            except Exception as e:
                return ""

        return f"\nThe file is available at: {dest}"

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question = row.get("Question", "")
        answer = row.get("Final answer", "") or row.get("answer", "")
        level = row.get("Level", 1)
        task_id = row.get("task_id", f"gaia_{idx:04d}")

        # Clean question ID for filesystem
        clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(task_id))

        # Stage any attached files
        file_hint = self._stage_file(clean_id, row)

        prompt = question.strip()
        if file_hint:
            prompt += file_hint

        prompt += (
            "\n\nGive a short, precise answer. "
            "If the answer is a number, give just the number. "
            "Put your final answer inside <answer></answer> tags."
        )

        return {
            "id": f"gaia_{clean_id}",
            "suite": "gaia",
            "prompt": prompt,
            "context": "",
            "expected": str(answer).strip(),
            "scoring": [],
            "image_path": "",
            "tier": min(max(int(level), 1), 3),
            "scoring_method": "exact_match",
            "scoring_config": {
                "extract_pattern": r"<answer>(.*?)</answer>",
                "normalize": True,
            },
        }


# ── CRUXEval (Code output/input prediction) ─────────────────────────────


class CRUXEvalAdapter(BaseAdapter):
    """CRUXEval: 800 functions × 2 tasks (output + input prediction).

    Source: cruxeval-org/cruxeval on HuggingFace.
    Output prediction is the pure REPL-advantage case: "just run the code."
    Input prediction tests reasoning: "what input gives this output?"

    Scoring: code_execution (assertion-based).
    """

    suite_name = "cruxeval"
    _raw_dataset = None

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._raw_dataset = hf.load_dataset(
                "cruxeval-org/cruxeval", split="test",
            )
            # Each row becomes 2 questions (output pred + input pred)
            self._dataset = list(range(len(self._raw_dataset) * 2))
        except Exception as e:
            print(f"  [adapter] CRUXEval load failed: {e}")
            self._dataset = []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        # idx 0..N-1 = output prediction, N..2N-1 = input prediction
        raw_len = len(self._raw_dataset) if self._raw_dataset else 0
        if idx < raw_len:
            return self._output_prompt(idx)
        return self._input_prompt(idx - raw_len)

    def _output_prompt(self, idx: int) -> dict:
        row = self._raw_dataset[idx]
        code = row.get("code", "")
        input_val = row.get("input", "")
        output_val = row.get("output", "")

        prompt = (
            f"What does the following Python code print when called with "
            f"the given input?\n\n"
            f"```python\n{code}\n```\n\n"
            f"Input: `{input_val}`\n\n"
            f"Give the exact output inside <answer></answer> tags."
        )

        return {
            "id": f"cruxeval_output_{idx:04d}",
            "suite": "cruxeval",
            "prompt": prompt,
            "context": "",
            "expected": str(output_val).strip(),
            "scoring": [],
            "image_path": "",
            "tier": 1,  # Output prediction = just run it
            "scoring_method": "exact_match",
            "scoring_config": {
                "extract_pattern": r"<answer>(.*?)</answer>",
                "normalize": True,
            },
        }

    def _input_prompt(self, idx: int) -> dict:
        row = self._raw_dataset[idx]
        code = row.get("code", "")
        input_val = row.get("input", "")
        output_val = row.get("output", "")

        prompt = (
            f"Given the following Python code and its output, determine what "
            f"input was provided.\n\n"
            f"```python\n{code}\n```\n\n"
            f"Output: `{output_val}`\n\n"
            f"Give the exact input value inside <answer></answer> tags."
        )

        return {
            "id": f"cruxeval_input_{idx:04d}",
            "suite": "cruxeval",
            "prompt": prompt,
            "context": "",
            "expected": str(input_val).strip(),
            "scoring": [],
            "image_path": "",
            "tier": 2,  # Input prediction = harder reasoning
            "scoring_method": "exact_match",
            "scoring_config": {
                "extract_pattern": r"<answer>(.*?)</answer>",
                "normalize": True,
            },
        }


# ── BigCodeBench (Multi-library coding) ──────────────────────────────────


class BigCodeBenchAdapter(BaseAdapter):
    """BigCodeBench: 1,140 coding tasks requiring 139 Python libraries.

    Source: bigcode/bigcodebench on HuggingFace (Apache 2.0).
    Scoring: code_execution (5.6 test cases per task, 99% branch coverage).

    Multi-library composition (pandas + matplotlib + scipy in one task) is
    where REPL + specialized coder >> direct frontdoor.
    """

    suite_name = "bigcodebench"

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._dataset = hf.load_dataset(
                "bigcode/bigcodebench", split="v0.1.2",
            )
        except Exception as e:
            try:
                import datasets as hf
                # Fallback to default split
                self._dataset = hf.load_dataset(
                    "bigcode/bigcodebench", split="default",
                )
            except Exception as e:
                print(f"  [adapter] BigCodeBench load failed: {e}")
                self._dataset = []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        task_id = row.get("task_id", f"bcb_{idx:04d}")
        instruct_prompt = row.get("instruct_prompt", "")
        complete_prompt = row.get("complete_prompt", "")
        test_code = row.get("test", "")
        canonical = row.get("canonical_solution", "")
        entry_point = row.get("entry_point", "")
        libs = row.get("libs", [])

        # Use instruct prompt if available, else complete_prompt
        prompt_text = instruct_prompt or complete_prompt
        if not prompt_text:
            prompt_text = f"Implement the function `{entry_point}`."

        # Determine tier based on library complexity
        lib_count = len(libs) if isinstance(libs, list) else 0
        if lib_count >= 3:
            tier = 3  # Multi-library = hard
        elif lib_count >= 2:
            tier = 2
        else:
            tier = 1

        # Build test assertions from test field
        scoring_config: dict = {
            "language": "python",
            "timeout": 30,  # BigCodeBench tasks can be complex
        }
        if test_code:
            scoring_config["test_code"] = test_code
        elif entry_point:
            scoring_config["entry_point"] = entry_point

        return {
            "id": f"bcb_{task_id}",
            "suite": "bigcodebench",
            "prompt": prompt_text.strip(),
            "context": "",
            "expected": entry_point,
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "code_execution",
            "scoring_config": scoring_config,
        }


# ── GPQA (Graduate-level Science) ─────────────────────────────────────────


class GPQAAdapter(BaseAdapter):
    """GPQA Diamond: 448 graduate-level science questions.

    Source: Idavidrein/gpqa on HuggingFace.
    Questions designed to be "Google-proof" — experts score 65%, GPT-4 = 39%.
    Perfect for mode-advantage: frontdoor fails, tools/specialists help.

    Scoring: multiple_choice (A/B/C/D).
    Tiers: Based on subdomain difficulty.
    """

    suite_name = "gpqa"
    has_real_tiers = True
    CHOICE_LABELS = ["A", "B", "C", "D"]

    # Subdomains with higher difficulty (based on benchmark papers)
    HARD_SUBDOMAINS = {"physics", "chemistry"}
    EASY_SUBDOMAINS = {"biology"}

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            # Use ankner/gpqa (ungated mirror of the original)
            # Original Idavidrein/gpqa requires access approval
            self._dataset = hf.load_dataset(
                "ankner/gpqa", split="train",
            )
        except Exception as e:
            print(f"  [adapter] GPQA load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        subdomain = row.get("Subdomain", "").lower()
        # Map subdomain to tier
        if any(hard in subdomain for hard in self.HARD_SUBDOMAINS):
            return 3
        elif any(easy in subdomain for easy in self.EASY_SUBDOMAINS):
            return 1
        return 2

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question = row.get("Question", "")
        # GPQA has Correct Answer and Incorrect Answer 1-3 fields
        correct_answer = row.get("Correct Answer", "")
        incorrect_1 = row.get("Incorrect Answer 1", "")
        incorrect_2 = row.get("Incorrect Answer 2", "")
        incorrect_3 = row.get("Incorrect Answer 3", "")

        # Collect all non-empty choices
        choices = [correct_answer, incorrect_1, incorrect_2, incorrect_3]
        choices = [c for c in choices if c]

        # Randomize choice order deterministically based on question hash
        import hashlib
        seed = int(hashlib.sha256(question.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        rng.shuffle(choices)

        # Find correct answer index after shuffle
        correct_idx = choices.index(correct_answer) if correct_answer in choices else 0
        expected_letter = self.CHOICE_LABELS[correct_idx]

        # Build prompt
        prompt_lines = [question, ""]
        for i, choice in enumerate(choices[:4]):  # Max 4 choices
            prompt_lines.append(f"{self.CHOICE_LABELS[i]}) {choice}")
        prompt_lines.append("")
        prompt_lines.append("Answer with the letter only (A, B, C, or D).")

        subdomain = row.get("Subdomain", "general")
        tier = self._get_tier_for_index(idx)

        return {
            "id": f"gpqa_{subdomain}_{idx:04d}",
            "suite": "gpqa",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": expected_letter,
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "multiple_choice",
            "scoring_config": {},
        }


# ── MMLU-Pro (Extended Multiple-Choice) ─────────────────────────────────────

class MMLUProAdapter(BaseAdapter):
    """MMLU-Pro: 12,032 extended multiple-choice questions (10 options, A-J).

    Source: TIGER-Lab/MMLU-Pro on HuggingFace.
    Harder variant of MMLU with 10 distractor options per question.
    Used as a quality gate for v7+ kernel promotion.

    Scoring: multiple_choice (A-J).
    Tiers: Based on category difficulty.
    """

    suite_name = "mmlu_pro"
    has_real_tiers = True
    CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    # Tier 3: hard STEM + professional
    HARD_CATEGORIES = {
        "math", "physics", "chemistry", "computer science",
        "engineering", "biology", "health",
    }
    # Tier 2: social sciences + humanities
    MEDIUM_CATEGORIES = {
        "economics", "psychology", "philosophy", "history", "law",
    }
    # Tier 1: business + other
    EASY_CATEGORIES = {"business", "other"}

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._dataset = hf.load_dataset(
                "TIGER-Lab/MMLU-Pro", split="test",
            )
        except Exception as e:
            print(f"  [adapter] MMLU-Pro load failed: {e}")
            self._dataset = []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question = row["question"]
        options = row["options"]
        answer = row["answer"]
        category = row.get("category", "other")

        prompt_lines = [question, ""]
        for i, opt in enumerate(options):
            prompt_lines.append(f"{self.CHOICE_LABELS[i]}) {opt}")
        prompt_lines.append("")
        prompt_lines.append("Answer with the letter only (A through J).")

        return {
            "id": f"mmlu_pro_{category}_{idx:05d}",
            "suite": "mmlu_pro",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": answer,
            "scoring": [],
            "image_path": "",
            "tier": self._get_tier_for_index(idx),
            "scoring_method": "multiple_choice",
            "scoring_config": {},
        }

    def _get_tier_for_index(self, idx: int) -> int:
        category = self._dataset[idx].get("category", "other")
        if category in self.HARD_CATEGORIES:
            return 3
        elif category in self.MEDIUM_CATEGORIES:
            return 2
        return 1


# ── SimpleQA (Factual Accuracy) ───────────────────────────────────────────


class SimpleQAAdapter(BaseAdapter):
    """SimpleQA: 4,326 short factual questions.

    Source: MAISAAI/openai_simple_qa_test_set on HuggingFace.
    Questions have unambiguous, short factual answers.
    GPT-4 scores <40% — ideal for mode-advantage with search tools.

    Scoring: exact_match (normalized).
    Tiers: Based on question complexity heuristics.
    """

    suite_name = "simpleqa"
    has_real_tiers = True

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._dataset = hf.load_dataset(
                "MAISAAI/openai_simple_qa_test_set", split="train",
            )
        except Exception as e:
            print(f"  [adapter] SimpleQA load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        question = row.get("problem", "")
        answer = row.get("answer", "")

        # Tier heuristics:
        # T3: Long answers (likely multi-part)
        # T2: Medium answers or questions with dates/numbers
        # T1: Short answers
        answer_words = len(answer.split())
        if answer_words > 10:
            return 3
        elif answer_words > 3 or re.search(r"\d{4}", question):
            return 2
        return 1

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question = row.get("problem", "")
        answer = row.get("answer", "")
        metadata = row.get("metadata", {}) or {}
        topic = metadata.get("topic", "general") if isinstance(metadata, dict) else "general"

        prompt = (
            f"{question}\n\n"
            "Give a short, precise answer. "
            "Put your final answer inside <answer></answer> tags."
        )

        tier = self._get_tier_for_index(idx)
        clean_topic = re.sub(r"[^a-zA-Z0-9_]", "_", str(topic))[:20]

        return {
            "id": f"simpleqa_{clean_topic}_{idx:05d}",
            "suite": "simpleqa",
            "prompt": prompt,
            "context": "",
            "expected": answer.strip(),
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "f1",
            "scoring_config": {
                "extract_pattern": r"<answer>(.*?)</answer>",
                "threshold": 0.8,
                "normalize": True,
            },
        }


# ── HotpotQA (Multi-hop Reasoning) ────────────────────────────────────────


class HotpotQAAdapter(BaseAdapter):
    """HotpotQA: Multi-hop reasoning questions requiring 2+ facts.

    Source: hotpotqa/hotpot_qa on HuggingFace.
    Questions require combining information from multiple documents.
    30B fails at ~40%, search tools can push to ~80%.

    Scoring: f1 (token-level F1 score).
    Tiers: Based on question type (bridge vs comparison).
    """

    suite_name = "hotpotqa"
    has_real_tiers = True

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            # Load the distractor setting (harder than fullwiki)
            ds = hf.load_dataset(
                "hotpotqa/hotpot_qa", "distractor", split="validation",
            )
            # Filter to "hard" questions only
            self._dataset = ds.filter(lambda x: x.get("level", "") == "hard")
        except Exception as e:
            print(f"  [adapter] HotpotQA load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        q_type = row.get("type", "")
        # Comparison questions are generally harder than bridge questions
        if q_type == "comparison":
            return 3
        return 2  # All "hard" level questions are at least T2

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question = row.get("question", "")
        answer = row.get("answer", "")
        q_id = row.get("id", f"hotpot_{idx:05d}")
        q_type = row.get("type", "bridge")
        supporting_facts = row.get("supporting_facts", {})
        context = row.get("context", {})

        # Build context from supporting paragraphs
        context_text = ""
        if context:
            titles = context.get("title", [])
            sentences_list = context.get("sentences", [])
            for title, sentences in zip(titles, sentences_list):
                context_text += f"### {title}\n"
                context_text += " ".join(sentences) + "\n\n"

        prompt = question
        if context_text:
            prompt = f"Context:\n{context_text.strip()}\n\nQuestion: {question}"

        prompt += (
            "\n\nGive a short, precise answer based on the context. "
            "Put your final answer inside <answer></answer> tags."
        )

        tier = self._get_tier_for_index(idx)

        return {
            "id": f"hotpot_{q_type}_{q_id}",
            "suite": "hotpotqa",
            "prompt": prompt,
            "context": context_text,
            "expected": answer.strip(),
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "f1",
            "scoring_config": {
                "extract_pattern": r"<answer>(.*?)</answer>",
                "threshold": 0.5,  # Minimum F1 to count as correct
            },
        }


# ── LiveCodeBench (Competition Programming) ───────────────────────────────


class LiveCodeBenchAdapter(BaseAdapter):
    """LiveCodeBench: Competition programming problems from LeetCode.

    Source: greengerong/leetcode on HuggingFace (2,360 problems).
    Alternative to livecodebench/code_generation (deprecated loading script).

    Problems include difficulty tags and reference solutions.
    Ideal for REPL mode-advantage: iterative testing beats direct inference.

    Scoring: code_execution (against extracted test cases) or substring.
    Tiers: Based on LeetCode difficulty (Easy/Medium/Hard).
    """

    suite_name = "livecodebench"
    has_real_tiers = True

    # LeetCode difficulty mapping
    DIFFICULTY_MAP = {
        "easy": 1,
        "medium": 2,
        "hard": 3,
    }

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._dataset = hf.load_dataset(
                "greengerong/leetcode", split="train",
            )
        except Exception as e:
            print(f"  [adapter] LiveCodeBench (LeetCode) load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        difficulty = row.get("difficulty", "")
        if difficulty:
            difficulty_lower = str(difficulty).lower()
            return self.DIFFICULTY_MAP.get(difficulty_lower, 2)
        # Fallback: estimate from problem ID (later problems tend to be harder)
        problem_id = row.get("id", idx)
        if isinstance(problem_id, int) and problem_id > 1500:
            return 3
        elif isinstance(problem_id, int) and problem_id > 800:
            return 2
        return 1

    def _extract_test_cases(self, content: str) -> list[tuple[str, str]]:
        """Extract example test cases from problem content."""
        tests = []
        # Pattern: Input: X Output: Y (or similar variations)
        pattern = re.compile(
            r"(?:Input|Example[^:]*Input)[:\s]*`?([^`\n]+)`?\s*"
            r"(?:Output)[:\s]*`?([^`\n]+)`?",
            re.IGNORECASE | re.MULTILINE
        )
        for match in pattern.finditer(content):
            inp = match.group(1).strip()
            out = match.group(2).strip()
            if inp and out:
                tests.append((inp, out))
        return tests[:3]  # Limit to 3 examples

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        title = row.get("title", f"Problem {idx}")
        content = row.get("content", "")
        difficulty = row.get("difficulty", "Medium")
        slug = row.get("slug", f"problem-{idx}")
        python_solution = row.get("python", "")

        # Clean HTML from content
        content_clean = re.sub(r"<[^>]+>", " ", content)
        content_clean = re.sub(r"\s+", " ", content_clean).strip()

        # Extract test cases
        test_cases = self._extract_test_cases(content)

        # Build prompt
        prompt_lines = [
            f"# {title}",
            "",
            content_clean,
            "",
        ]

        if test_cases:
            prompt_lines.append("### Examples:")
            for i, (inp, out) in enumerate(test_cases, 1):
                prompt_lines.append(f"Example {i}:")
                prompt_lines.append(f"  Input: {inp}")
                prompt_lines.append(f"  Output: {out}")
                prompt_lines.append("")

        prompt_lines.append(
            "Write a Python function to solve this problem. "
            "Include proper type hints and handle edge cases."
        )

        tier = self._get_tier_for_index(idx)

        # Build test code from extracted cases
        test_code = ""
        if test_cases and python_solution:
            # Try to extract function name from solution
            fn_match = re.search(r"def\s+(\w+)\s*\(", python_solution)
            if fn_match:
                fn_name = fn_match.group(1)
                test_code = f"# Test cases for {fn_name}\n"
                for inp, out in test_cases:
                    test_code += f"# assert {fn_name}({inp}) == {out}\n"

        # Determine scoring method based on content
        scoring_method = "code_execution" if test_code else "substring"
        scoring_config = {
            "language": "python",
            "timeout": 30,
        }
        if test_code:
            scoring_config["test_code"] = test_code
        else:
            # Fallback: check for function definition
            scoring_config["case_sensitive"] = True
            scoring_config["substring"] = "def "

        return {
            "id": f"leetcode_{slug}",
            "suite": "livecodebench",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": "def ",  # At minimum, expect a function
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": scoring_method,
            "scoring_config": scoring_config,
        }

    def sample(
        self, n: int = 10, seed: int = 42, stratify: bool = False,
        filter_difficulty: str | None = None,
    ) -> list[dict]:
        """Sample with optional difficulty filter.

        Args:
            n: Number of samples.
            seed: Random seed.
            stratify: Whether to stratify by tier.
            filter_difficulty: "easy", "medium", "hard", or None for all.
        """
        self._ensure_loaded()
        if not self._dataset:
            return []

        if filter_difficulty:
            # Filter by difficulty string
            target_tier = self.DIFFICULTY_MAP.get(filter_difficulty.lower(), 2)
            filtered_indices = [
                i for i in range(len(self._dataset))
                if self._get_tier_for_index(i) == target_tier
            ]
            rng = random.Random(seed)
            indices = rng.sample(filtered_indices, min(n, len(filtered_indices)))
            return [self._row_to_prompt(i, self._dataset[i]) for i in indices]

        # Default sampling
        if stratify and self.has_real_tiers:
            return self._stratified_sample(n, seed)

        rng = random.Random(seed)
        indices = rng.sample(range(len(self._dataset)), min(n, len(self._dataset)))
        return [self._row_to_prompt(i, self._dataset[i]) for i in indices]


# ── DebugBench (Bug Finding/Fixing) ───────────────────────────────────────


class DebugBenchAdapter(BaseAdapter):
    """DebugBench: 4,253 buggy code instances across 3 languages.

    Source: Rtian/DebugBench on HuggingFace.
    Contains buggy code with explanations, solutions, and bug categories.
    Perfect for REPL mode-advantage: iterative debugging >> direct inference.

    Scoring: code_execution (run fixed code against test cases).
    Tiers: easy=T1, medium=T2, hard=T3 (from LeetCode difficulty).
    """

    suite_name = "debugbench"
    has_real_tiers = True

    LEVEL_MAP = {"easy": 1, "medium": 2, "hard": 3}

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._dataset = hf.load_dataset("Rtian/DebugBench", split="test")
        except Exception as e:
            print(f"  [adapter] DebugBench load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        level = row.get("level", "medium").lower()
        return self.LEVEL_MAP.get(level, 2)

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question = row.get("question", "")
        buggy_code = row.get("buggy_code", "")
        solution = row.get("solution", "")
        bug_explanation = row.get("bug_explanation", "")
        examples = row.get("examples", [])
        constraints = row.get("constraints", "")
        language = row.get("language", "python3")
        level = row.get("level", "medium")
        slug = row.get("slug", f"debug_{idx:04d}")
        category = row.get("category", "")

        # Map language to standard names
        lang_map = {"python3": "python", "cpp": "cpp", "java": "java"}
        lang = lang_map.get(language, language)

        # Build prompt
        prompt_lines = [
            f"# Bug Fixing Task ({lang.upper()})",
            "",
            f"## Problem Description",
            question[:500] if len(question) > 500 else question,
            "",
        ]

        if examples:
            prompt_lines.append("## Examples")
            for i, ex in enumerate(examples[:2]):
                prompt_lines.append(f"```")
                prompt_lines.append(str(ex)[:200])
                prompt_lines.append(f"```")
            prompt_lines.append("")

        if constraints:
            prompt_lines.append(f"## Constraints")
            prompt_lines.append(constraints[:200])
            prompt_lines.append("")

        prompt_lines.extend([
            f"## Buggy Code",
            f"```{lang}",
            buggy_code[:1000] if len(buggy_code) > 1000 else buggy_code,
            "```",
            "",
            "Find and fix the bug(s) in the code above. "
            "Provide the corrected code. "
            "Fix ONLY the bug — do NOT rewrite, rename variables, "
            "change data structures, or optimize. Keep the original code structure.",
        ])

        tier = self._get_tier_for_index(idx)

        # For Python, we can do code_execution scoring
        scoring_method = "code_execution" if lang == "python" else "substring"
        scoring_config = {"language": lang, "timeout": 30}

        if lang != "python":
            # For non-Python, check that key parts of solution appear
            scoring_config = {"case_sensitive": True}

        return {
            "id": f"debugbench_{slug}_{lang}",
            "suite": "debugbench",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": solution[:100] if solution else "def ",
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": scoring_method,
            "scoring_config": scoring_config,
            "metadata": {
                "language": lang,
                "category": category,
                "bug_explanation": bug_explanation[:200],
            },
        }

    def sample(
        self,
        n: int = 10,
        seed: int = 42,
        stratify: bool = False,
        filter_language: str | None = None,
        filter_category: str | None = None,
    ) -> list[dict]:
        """Sample with optional language/category filter.

        Args:
            n: Number of samples.
            seed: Random seed.
            stratify: Whether to stratify by tier.
            filter_language: "python3", "cpp", "java", or None for all.
            filter_category: Bug category filter or None for all.
        """
        self._ensure_loaded()
        if not self._dataset:
            return []

        # Apply filters
        filtered_indices = list(range(len(self._dataset)))

        if filter_language:
            filtered_indices = [
                i for i in filtered_indices
                if self._dataset[i].get("language", "") == filter_language
            ]

        if filter_category:
            filtered_indices = [
                i for i in filtered_indices
                if filter_category.lower() in self._dataset[i].get("category", "").lower()
            ]

        if not filtered_indices:
            return []

        rng = random.Random(seed)
        indices = rng.sample(filtered_indices, min(n, len(filtered_indices)))
        return [self._row_to_prompt(i, self._dataset[i]) for i in indices]


# ── USACO (Olympiad Programming) ──────────────────────────────────────────


class USACOAdapter(BaseAdapter):
    """USACO: Olympiad-level competitive programming problems.

    Source: codegenning/usacobench_formatted on HuggingFace.
    307 problems across Bronze/Silver/Gold/Platinum divisions.
    GPT-4 scores 8.7% zero-shot — ideal for REPL + specialist escalation.

    Scoring: code_execution (against test cases).
    Tiers: Bronze=T1, Silver=T2, Gold/Platinum=T3.
    """

    suite_name = "usaco"
    has_real_tiers = True

    DIVISION_MAP = {
        "bronze": 1,
        "silver": 2,
        "gold": 3,
        "platinum": 3,
    }

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            # Use streaming to avoid timeout on large dataset
            self._dataset = hf.load_dataset(
                "codegenning/usacobench_formatted",
                split="test",
                streaming=False,
            )
        except Exception as e:
            print(f"  [adapter] USACO load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        division = row.get("division", row.get("difficulty", row.get("level", "silver"))).lower()
        return self.DIVISION_MAP.get(division, 2)

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        # Schema varies — try multiple field names
        problem = row.get("problem", row.get("question", row.get("prompt", "")))
        problem_id = row.get("problem_id", row.get("id", f"usaco_{idx:04d}"))
        division = row.get("division", row.get("difficulty", row.get("level", "silver")))
        solution = row.get("solution", row.get("code", ""))
        # Parse test cases — dataset uses "input_output" JSON string with
        # {"inputs": [...], "outputs": [...]} format (codegenning/usacobench_formatted)
        test_cases = row.get("test_cases", row.get("tests", []))
        input_output_raw = row.get("input_output", "")

        # Build prompt
        prompt_lines = [
            f"# USACO Problem ({division.title()} Division)",
            "",
            problem[:2000] if len(problem) > 2000 else problem,
            "",
            "Write a Python solution that reads from stdin and writes to stdout.",
            "Your solution should handle all test cases within time limits.",
        ]

        tier = self._get_tier_for_index(idx)

        # Build test code from input_output JSON (primary) or test_cases (fallback)
        test_code = ""
        if input_output_raw and isinstance(input_output_raw, str):
            try:
                import json
                io = json.loads(input_output_raw)
                inputs = io.get("inputs", [])
                outputs = io.get("outputs", [])
                if inputs and outputs:
                    pairs = []
                    for inp, out in zip(inputs[:5], outputs[:5]):
                        pairs.append(f"({repr(inp)}, {repr(out)})")
                    test_code = f"TEST_CASES = [{', '.join(pairs)}]"
            except (json.JSONDecodeError, TypeError):
                pass
        if not test_code and test_cases and isinstance(test_cases, list):
            test_cases_str = []
            for tc in test_cases[:5]:
                if isinstance(tc, dict):
                    inp = repr(tc.get("input", ""))
                    out = repr(tc.get("output", tc.get("expected", "")))
                    test_cases_str.append(f"({inp}, {out})")
            if test_cases_str:
                test_code = f"TEST_CASES = [{', '.join(test_cases_str)}]"

        return {
            "id": f"usaco_{division}_{problem_id}",
            "suite": "usaco",
            "prompt": "\n".join(prompt_lines),
            "context": "",
            "expected": "",  # Code execution determines correctness
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "code_execution",
            "scoring_config": {
                "language": "python",
                "timeout": 120,  # USACO problems need more time
                "test_code": test_code,
            },
        }

    def sample(
        self,
        n: int = 10,
        seed: int = 42,
        stratify: bool = False,
        filter_division: str | None = None,
    ) -> list[dict]:
        """Sample with optional division filter.

        Args:
            n: Number of samples.
            seed: Random seed.
            stratify: Whether to stratify by tier.
            filter_division: "bronze", "silver", "gold", "platinum", or None.
        """
        self._ensure_loaded()
        if not self._dataset:
            return []

        if filter_division:
            filtered_indices = [
                i for i in range(len(self._dataset))
                if self._dataset[i].get("division", "").lower() == filter_division.lower()
            ]
            if not filtered_indices:
                return []
            rng = random.Random(seed)
            indices = rng.sample(filtered_indices, min(n, len(filtered_indices)))
            return [self._row_to_prompt(i, self._dataset[i]) for i in indices]

        if stratify and self.has_real_tiers:
            return self._stratified_sample(n, seed)

        rng = random.Random(seed)
        indices = rng.sample(range(len(self._dataset)), min(n, len(self._dataset)))
        return [self._row_to_prompt(i, self._dataset[i]) for i in indices]


# ── PHYBench (Physics Reasoning) ─────────────────────────────────────────


class PHYBenchAdapter(BaseAdapter):
    """PHYBench: 500 physics problems, 100 with symbolic LaTeX ground truth.

    Source: Eureka-Lab/PHYBench on HuggingFace.
    Covers mechanics, electricity, thermodynamics, optics, modern, advanced physics.
    Best-performing LLM (Gemini 2.5 Pro) scores 36.9% vs human 61.9%.

    Scoring: substring match on LaTeX expressions (model outputs \\boxed{}).
    Tiers: mapped from physics domain difficulty.
    Only rows with non-empty answers are included (100 unique problems).
    """

    suite_name = "physics"
    has_real_tiers = True

    # Map PHYBench tags to difficulty tiers
    TAG_TIER_MAP = {
        "MECHANICS": 1,        # Most common, classical — tier 1
        "THERMODYNAMICS": 2,   # Intermediate
        "ELECTRICITY": 2,      # Intermediate
        "OPTICS": 2,           # Intermediate
        "MODERN": 3,           # Relativity, QM — hard
        "ADVANCED": 3,         # Research-level — hard
    }

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            raw = hf.load_dataset("Eureka-Lab/PHYBench", split="train")
            # Filter: only keep rows with non-empty answers, deduplicate by id
            seen_ids = set()
            filtered = []
            for row in raw:
                answer = row.get("answer", "").strip()
                row_id = row.get("id")
                if answer and row_id not in seen_ids:
                    seen_ids.add(row_id)
                    filtered.append(dict(row))
            self._dataset = filtered
        except Exception as e:
            print(f"  [adapter] PHYBench load failed: {e}")
            self._dataset = []

    @staticmethod
    def _clean_latex(answer: str) -> str:
        """Strip LaTeX display delimiters from answer, keep inner expression."""
        a = answer.strip()
        # Remove \[...\] or $$...$$ wrappers
        for start, end in [("\\[", "\\]"), ("$$", "$$")]:
            if a.startswith(start) and a.endswith(end):
                a = a[len(start):-len(end)].strip()
                break
        return a

    def _get_tier_for_index(self, idx: int) -> int:
        tag = self._dataset[idx].get("tag", "MECHANICS")
        return self.TAG_TIER_MAP.get(tag, 2)

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        content = row.get("content", "")
        answer_raw = row.get("answer", "")
        tag = row.get("tag", "MECHANICS")
        row_id = row.get("id", idx)
        tier = self.TAG_TIER_MAP.get(tag, 2)
        expected = self._clean_latex(answer_raw)

        prompt = (
            f"{content}\n\n"
            "Solve this physics problem step by step. "
            "Put your final answer in \\boxed{}."
        )

        return {
            "id": f"phybench_{tag.lower()}_{row_id}",
            "suite": "physics",
            "prompt": prompt,
            "context": "",
            "expected": expected,
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "llm_judge",
            "scoring_config": {},  # semantic equivalence — handles LaTeX format variants
        }


# ── AIME 2024+2025 (Competition Math) ────────────────────────────────


class AIMEAdapter(BaseAdapter):
    """AIME 2024 (30) + AIME 2025 (30) = 60 competition math problems.

    Sources:
      - Maxwell-Jia/AIME_2024 (30 problems, MIT)
      - opencompass/AIME2025 (30 problems, MIT, two configs)

    All AIME answers are integers 000-999.
    Scoring: exact_match with numeric extraction.
    Tiers: All T3 (olympiad-level).
    """

    suite_name = "aime"
    has_real_tiers = False
    _aime2024 = None
    _aime2025 = None

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            # AIME 2024: Maxwell-Jia/AIME_2024 — fields: ID, Problem, Solution, Answer
            self._aime2024 = hf.load_dataset(
                "Maxwell-Jia/AIME_2024", split="train",
            )
        except Exception as e:
            print(f"  [adapter] AIME 2024 load failed: {e}")
            self._aime2024 = []

        try:
            import datasets as hf
            # AIME 2025: opencompass/AIME2025 — two configs, fields: question, answer
            ds_i = hf.load_dataset(
                "opencompass/AIME2025", "AIME2025-I", split="test",
            )
            ds_ii = hf.load_dataset(
                "opencompass/AIME2025", "AIME2025-II", split="test",
            )
            # Combine into a list of dicts with uniform schema
            combined = []
            for i, row in enumerate(ds_i):
                combined.append({
                    "id": f"2025-I-{i + 1}",
                    "problem": row["question"],
                    "answer": str(row["answer"]),
                })
            for i, row in enumerate(ds_ii):
                combined.append({
                    "id": f"2025-II-{i + 1}",
                    "problem": row["question"],
                    "answer": str(row["answer"]),
                })
            self._aime2025 = combined
        except Exception as e:
            print(f"  [adapter] AIME 2025 load failed: {e}")
            self._aime2025 = []

        # Unified index list
        n24 = len(self._aime2024) if self._aime2024 else 0
        n25 = len(self._aime2025) if self._aime2025 else 0
        self._dataset = list(range(n24 + n25))

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        n24 = len(self._aime2024) if self._aime2024 else 0

        if idx < n24:
            r = self._aime2024[idx]
            problem_id = r.get("ID", f"2024-{idx + 1}")
            problem = r["Problem"]
            answer = str(r["Answer"])
        else:
            r = self._aime2025[idx - n24]
            problem_id = r["id"]
            problem = r["problem"]
            answer = r["answer"]

        return {
            "id": f"aime_{problem_id}",
            "suite": "aime",
            "prompt": (
                f"{problem}\n\n"
                "This is an AIME problem. The answer is an integer from 000 to 999. "
                "Show your work step by step, then state your final answer as a single integer."
            ),
            "context": "",
            "expected": answer,
            "scoring": [],
            "image_path": "",
            "tier": 3,  # All AIME = olympiad-level
            "scoring_method": "exact_match",
            "scoring_config": {"extract_pattern": r"(\d+)\s*$", "normalize_numeric": True},
        }


# ── OlympiadBench (Olympiad-level Math) ──────────────────────────────


class OlympiadBenchAdapter(BaseAdapter):
    """OlympiadBench: 674 olympiad-level math problems (text-only, English).

    Source: math-ai/olympiadbench on HuggingFace.
    Open-ended math competition problems across Algebra, Combinatorics,
    Geometry, and Number Theory. Answers are numerical, expressions, or tuples.

    Scoring: substring (answers may be LaTeX expressions).
    Tiers: All T3 (olympiad-level competition problems).
    """

    suite_name = "olympiadbench"
    has_real_tiers = True

    SUBFIELD_TIER_MAP = {
        "Algebra": 2,
        "Combinatorics": 3,
        "Geometry": 3,
        "Number Theory": 3,
    }

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            raw = hf.load_dataset("math-ai/olympiadbench", split="test")
            # Filter to text-only problems with answers
            filtered = []
            for row in raw:
                answers = row.get("final_answer", [])
                if not answers or not any(a.strip() for a in answers):
                    continue
                filtered.append(dict(row))
            self._dataset = filtered
        except Exception as e:
            print(f"  [adapter] OlympiadBench load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        subfield = self._dataset[idx].get("subfield", "")
        return self.SUBFIELD_TIER_MAP.get(subfield, 3)

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        question = row.get("question", "")
        answers = row.get("final_answer", [])
        # Use first answer; some problems have multiple valid answers
        expected = answers[0].strip() if answers else ""
        is_multi = row.get("is_multiple_answer", False)
        answer_type = row.get("answer_type", "Numerical")
        subfield = row.get("subfield", "Math")
        row_id = row.get("id", idx)
        unit = row.get("unit", "")
        tier = self.SUBFIELD_TIER_MAP.get(subfield, 3)

        # If multiple answers, join them for expected
        if is_multi and len(answers) > 1:
            expected = ", ".join(a.strip() for a in answers if a.strip())

        suffix = "\n\nSolve step by step. Put your final answer in \\boxed{}."
        if unit:
            suffix = f"\n\nSolve step by step. Express your answer in {unit}. Put your final answer in \\boxed{{}}."

        return {
            "id": f"olympiadbench_{subfield.lower().replace(' ', '_')}_{row_id}",
            "suite": "olympiadbench",
            "prompt": question + suffix,
            "context": "",
            "expected": expected,
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "substring",
            "scoring_config": {"case_sensitive": False},
        }


# ── PhysReason (Multimodal Physics Reasoning) ────────────────────────


class PhysReasonAdapter(BaseAdapter):
    """PhysReason: 1,200 physics problems with ~3,100 sub-questions.

    Source: zhibei1204/PhysReason on HuggingFace (zip download).
    1,200 multi-step physics problems across 4 difficulty levels.
    81% include diagrams (JPG/PNG). Flattened to one entry per sub-question.

    Scoring: llm_judge (semantic equivalence via local worker model).
    Tiers: knowledge/easy=T1, medium=T2, difficult=T3.
    """

    suite_name = "physreason"
    has_real_tiers = True

    DIFFICULTY_TIER_MAP = {
        "knowledge": 1,
        "easy": 1,
        "medium": 2,
        "difficult": 3,
    }

    # Cached extracted data directory
    _extract_dir: str | None = None

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            self._dataset = self._load_from_zip()
        except Exception as e:
            print(f"  [adapter] PhysReason load failed: {e}")
            self._dataset = []

    def _load_from_zip(self) -> list[dict]:
        """Download zip from HF, extract, flatten to sub-questions."""
        import json as _json
        import os
        import zipfile

        from huggingface_hub import hf_hub_download

        # Try full dataset first, fall back to mini
        for filename, dirname in [
            ("PhysReason-full.zip", "PhysReason_full"),
            ("PhysReason-mini.zip", "PhysReason-mini"),
        ]:
            try:
                zip_path = hf_hub_download(
                    "zhibei1204/PhysReason", filename, repo_type="dataset"
                )
                break
            except Exception:
                continue
        else:
            print("  [adapter] PhysReason: could not download dataset")
            return []

        extract_base = "/mnt/raid0/llm/tmp/physreason"
        data_dir = os.path.join(extract_base, dirname)

        # Extract only if not already present
        if not os.path.isdir(data_dir):
            os.makedirs(extract_base, exist_ok=True)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(extract_base)

        self._extract_dir = data_dir

        # Parse all problem.json files and flatten to sub-questions
        flattened = []
        for problem_dir in sorted(os.listdir(data_dir)):
            pjson = os.path.join(data_dir, problem_dir, "problem.json")
            if not os.path.isfile(pjson):
                continue

            with open(pjson, "rb") as f:
                raw = f.read()
            # Handle BOM encodings
            if raw[:2] == b"\xff\xfe":
                text = raw.decode("utf-16-le")
            elif raw[:3] == b"\xef\xbb\xbf":
                text = raw[3:].decode("utf-8")
            else:
                text = raw.decode("utf-8")

            try:
                data = _json.loads(text)
            except _json.JSONDecodeError:
                continue

            difficulty = data.get("difficulty", "medium")
            qs = data.get("question_structure", {})
            context = qs.get("context", "")
            answers = data.get("answer", [])
            image_list = data.get("question_image_list", [])
            image_caption = data.get("image_captions", "")

            # Resolve first image path (if present)
            image_path = ""
            if image_list:
                img_rel = image_list[0]
                img_abs = os.path.join(data_dir, problem_dir, img_rel)
                if os.path.isfile(img_abs):
                    image_path = img_abs

            # Collect sub-questions (keys like sub_question_1, sub_question_2, ...)
            sub_qs = []
            for key in sorted(qs.keys()):
                if key.startswith("sub_question_") and isinstance(qs[key], str):
                    sub_qs.append((key, qs[key]))

            # Flatten: one entry per sub-question
            for sq_idx, (sq_key, sq_text) in enumerate(sub_qs):
                if sq_idx >= len(answers):
                    continue  # No answer for this sub-question

                flattened.append({
                    "problem_dir": problem_dir,
                    "difficulty": difficulty,
                    "context": context,
                    "sub_question": sq_text,
                    "sq_key": sq_key,
                    "sq_idx": sq_idx,
                    "expected": answers[sq_idx],
                    "image_path": image_path,
                    "image_caption": image_caption,
                })

        return flattened

    def _get_tier_for_index(self, idx: int) -> int:
        difficulty = self._dataset[idx].get("difficulty", "medium")
        return self.DIFFICULTY_TIER_MAP.get(difficulty, 2)

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        context = row.get("context", "")
        sub_q = row.get("sub_question", "")
        expected = row.get("expected", "")
        difficulty = row.get("difficulty", "medium")
        problem_dir = row.get("problem_dir", f"unknown_{idx}")
        sq_idx = row.get("sq_idx", 0)
        image_path = row.get("image_path", "")
        image_caption = row.get("image_caption", "")
        tier = self.DIFFICULTY_TIER_MAP.get(difficulty, 2)

        # Build prompt with context + sub-question
        parts = [context]
        if image_caption and image_path:
            parts.append(f"\n[Diagram description: {image_caption}]")
        parts.append(f"\nQuestion: {sub_q}")
        parts.append(
            "\nSolve step by step. Put your final answer in \\boxed{}."
        )
        prompt = "\n".join(parts)

        return {
            "id": f"physreason_{problem_dir}_sq{sq_idx}",
            "suite": "physreason",
            "prompt": prompt,
            "context": "",
            "expected": expected,
            "scoring": [],
            "image_path": image_path,
            "tier": tier,
            "scoring_method": "llm_judge",
            "scoring_config": {"judge_port": 8082},
        }


# ── AA-Omniscience (Knowledge Reliability / Hallucination Detection) ─────


class AAOmniscienceAdapter(BaseAdapter):
    """AA-Omniscience: 600 factual questions testing knowledge reliability.

    Source: ArtificialAnalysis/AA-Omniscience-Public on HuggingFace.
    Tests whether models answer correctly OR correctly abstain when uncertain.
    Scoring penalizes confident hallucination (wrong answers scored worse
    than explicit abstention).

    Domains: Finance (accounting, IFRS/GAAP, tax, M&A, banking, investments).
    Scoring: f1 with abstention-aware penalty.
    Tiers: Based on answer complexity and domain difficulty.
    """

    suite_name = "omniscience"
    has_real_tiers = True

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            self._dataset = hf.load_dataset(
                "ArtificialAnalysis/AA-Omniscience-Public", split="train",
            )
        except Exception as e:
            print(f"  [adapter] AA-Omniscience load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        question = row.get("question", "")
        answer = row.get("answer", "")

        # Tier by answer complexity and question length
        answer_len = len(answer)
        q_words = len(question.split())

        # T3: long answers or very long questions (multi-step lookup)
        if answer_len > 25 or q_words > 60:
            return 3
        # T1: short answers with short questions (direct recall)
        if answer_len <= 10 and q_words <= 25:
            return 1
        return 2

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        domain = row.get("domain", "Finance")
        topic = row.get("topic", "general")
        question = row.get("question", "")
        answer = row.get("answer", "")
        qid = row.get("question_id", idx)

        prompt = (
            f"You are answering questions about {domain}, "
            f"and in particular {topic}.\n\n"
            f"{question}\n\n"
            "Answer with JUST the answer (no explanation). "
            "If you do not know the answer, or you need more context "
            "or tools to answer the question, say \"I don't know\" — "
            "it is better to say this than to get the wrong answer.\n\n"
            "Put your final answer inside <answer></answer> tags."
        )

        tier = self._get_tier_for_index(idx)
        clean_topic = re.sub(r"[^a-zA-Z0-9_]", "_", str(topic))[:20]

        return {
            "id": f"omniscience_{clean_topic}_{qid:04d}",
            "suite": "omniscience",
            "prompt": prompt,
            "context": "",
            "expected": answer.strip(),
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "f1",
            "scoring_config": {
                "extract_pattern": r"<answer>(.*?)</answer>",
                "threshold": 0.8,
                "normalize": True,
                "abstention_patterns": [
                    r"(?i)i don'?t know",
                    r"(?i)i'?m not sure",
                    r"(?i)i cannot",
                    r"(?i)insufficient",
                    r"(?i)not enough (context|information)",
                ],
            },
        }


# ── AA-LCR (Long Context Reasoning) ─────────────────────────────────────


class AALCRAdapter(BaseAdapter):
    """AA-LCR: 100 long-context multi-document reasoning questions.

    Source: ArtificialAnalysis/AA-LCR on HuggingFace.
    Each question requires reasoning across multiple documents (~100K tokens).
    Documents must be pre-downloaded via download_aa_lcr.py.

    Scoring: f1 (free-form answers vary from short facts to ranked lists).
    Tiers: Based on expected input token count.
    """

    suite_name = "aa_lcr"
    has_real_tiers = True

    JSONL_PATH = Path("/mnt/raid0/llm/data/eval/aa_lcr/aa_lcr.jsonl")

    def _ensure_loaded(self):
        if self._dataset is not None:
            return

        if not self.JSONL_PATH.exists():
            print(
                "  [adapter] AA-LCR JSONL not found. Run:\n"
                "    python scripts/benchmark/download_aa_lcr.py"
            )
            self._dataset = []
            return

        try:
            rows = []
            for line in self.JSONL_PATH.read_text().strip().split("\n"):
                if line.strip():
                    rows.append(json.loads(line))
            self._dataset = rows
        except Exception as e:
            print(f"  [adapter] AA-LCR load failed: {e}")
            self._dataset = []

    def _get_tier_for_index(self, idx: int) -> int:
        row = self._dataset[idx]
        tokens = row.get("context_tokens_expected", 80000)

        # T1: shorter contexts (<80K tokens)
        if tokens < 80000:
            return 1
        # T3: very long contexts (>100K tokens)
        if tokens > 100000:
            return 3
        return 2

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        context = row.get("context", "")
        question = row.get("question", "")
        answer = row.get("answer", "")
        entry_id = row.get("id", f"aa_lcr_{idx:03d}")
        category = row.get("document_category", "unknown")

        if not context:
            return None

        prompt = (
            "You are given a collection of documents. Read them carefully "
            "and answer the question that follows.\n\n"
            "--- DOCUMENTS ---\n\n"
            f"{context}\n\n"
            "--- QUESTION ---\n\n"
            f"{question}\n\n"
            "Answer precisely based on the documents. "
            "Put your final answer inside <answer></answer> tags."
        )

        tier = self._get_tier_for_index(idx)

        return {
            "id": entry_id,
            "suite": "aa_lcr",
            "prompt": prompt,
            "context": "",
            "expected": answer.strip(),
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "f1",
            "scoring_config": {
                "extract_pattern": r"<answer>(.*?)</answer>",
                "threshold": 0.5,
                "normalize": True,
            },
        }


# ── AIME 2025 only (architect-bench: uncontaminated competition math) ────────


class AIME25Adapter(BaseAdapter):
    """AIME 2025 ONLY (30 problems) — the architect-bench reasoning core.

    Deliberately excludes AIME 2024 (which the combined `aime` suite mixes in):
    2024 predates the training cutoff of every arm under test, so a 2024 score
    conflates recall with reasoning. 2025 is the uncontaminated half.

    Source: opencompass/AIME2025 (configs AIME2025-I / AIME2025-II, MIT).
    All answers are integers 0-999. Scoring: exact_match, numeric.
    Tiers: all T3 (olympiad-level).
    """

    suite_name = "aime25"
    has_real_tiers = False

    @staticmethod
    def _clean_answer(raw: str) -> str:
        """AIME answers are integers 0-999, but the upstream set carries units.

        e.g. 2025-II-5 ships as '336^\\circ'. Left as-is it can never be
        matched by an integer-emitting model, silently scoring 0 for every arm.
        """
        m = re.search(r"\d{1,3}", str(raw))
        return str(int(m.group(0))) if m else str(raw).strip()

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        combined = []
        try:
            import datasets as hf
            for cfg, label in (("AIME2025-I", "I"), ("AIME2025-II", "II")):
                ds = hf.load_dataset("opencompass/AIME2025", cfg, split="test")
                for i, row in enumerate(ds):
                    combined.append({
                        "id": f"2025-{label}-{i + 1}",
                        "problem": row["question"],
                        "answer": self._clean_answer(row["answer"]),
                    })
        except Exception as e:
            print(f"  [adapter] AIME 2025 load failed: {e}")
            combined = []
        self._dataset = combined

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        return {
            "id": f"aime25_{row['id']}",
            "suite": "aime25",
            "prompt": (
                f"{row['problem']}\n\n"
                "This is an AIME problem. The answer is an integer between 0 and 999. "
                "Reason step by step, then give your final answer on its own last line "
                "in the form: ANSWER: <integer>"
            ),
            "context": "",
            "expected": row["answer"],
            "scoring": [],
            "image_path": "",
            "tier": 3,
            "scoring_method": "exact_match",
            "scoring_config": {
                # Tried in order; first pattern that matches wins (last occurrence).
                # Ordered most-explicit -> most-permissive so a stray digit in the
                # working-out never outranks a stated final answer.
                "extract_patterns": [
                    r"ANSWER:\s*\**\s*(\d{1,3})",
                    r"\\boxed\{\s*(\d{1,3})\s*\}",
                    r"(?:final\s+answer|answer)\s*(?:is)?\s*[:=]?\s*\**\s*(\d{1,3})",
                    r"(\d{1,3})\s*\**\s*\.?\s*$",
                ],
                "normalize_numeric": True,
            },
        }


# ── GPQA-Diamond (architect-bench: the 198-question hard subset) ─────────────


class GPQADiamondAdapter(GPQAAdapter):
    """GPQA-Diamond: the 198-question expert-validated hard subset.

    The `gpqa` suite above loads ankner/gpqa, which is GPQA **main** (448) —
    NOT Diamond. Diamond is the subset both expert validators answered
    correctly and most non-experts got wrong; it is what published
    GPQA-Diamond numbers refer to, so the architect bench needs it.

    Membership comes from the hendrydong/gpqa_diamond mirror (198 rows, verified
    to match all 198 into ankner/gpqa by normalized question text); the
    multiple-choice framing + real distractors come from ankner/gpqa, since the
    hendrydong mirror is free-response only.

    Scoring: multiple_choice (A/B/C/D). Tiers inherited from subdomain.
    """

    suite_name = "gpqa_diamond"
    has_real_tiers = True

    _DIAMOND_GLOB = (
        "/mnt/raid0/llm/cache/huggingface/hub/datasets--hendrydong--gpqa_diamond/"
        "snapshots/*/data/*.parquet"
    )

    @staticmethod
    def _norm_q(s: str) -> str:
        import unicodedata
        s = unicodedata.normalize("NFKC", s or "")
        return re.sub(r"\s+", " ", s).strip().lower()

    def _diamond_keys(self) -> set:
        """Normalized question text of the 198 Diamond items."""
        import glob as _glob
        import datasets as hf
        rows = None
        matches = sorted(_glob.glob(self._DIAMOND_GLOB))
        if matches:
            rows = hf.load_dataset("parquet", data_files=matches[0], split="train")
        else:
            rows = hf.load_dataset("hendrydong/gpqa_diamond", split="test")
        return {self._norm_q(r["problem"]) for r in rows}

    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            import datasets as hf
            main = hf.load_dataset("ankner/gpqa", split="train")
            keys = self._diamond_keys()
            selected = [dict(r) for r in main if self._norm_q(r.get("Question", "")) in keys]
            if len(selected) != len(keys):
                print(f"  [adapter] gpqa_diamond: matched {len(selected)}/{len(keys)} "
                      f"Diamond questions into ankner/gpqa")
            self._dataset = selected
        except Exception as e:
            print(f"  [adapter] GPQA-Diamond load failed: {e}")
            self._dataset = []

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        q = super()._row_to_prompt(idx, row)
        # Positional ids are not stable across dataset revisions; the architect
        # bench pairs arms across sessions, so key on the question itself.
        import hashlib
        qhash = hashlib.sha256(self._norm_q(row.get("Question", "")).encode()).hexdigest()[:12]
        q["suite"] = "gpqa_diamond"
        q["id"] = f"gpqa_diamond_{qhash}"
        return q


class GPQADiamondCoTAdapter(GPQADiamondAdapter):
    """GPQA-Diamond, but the model is allowed to reason before answering.

    The plain `gpqa_diamond` prompt says "Answer with the letter only", which
    at enable_thinking=false yields ~2 completion tokens — a pure
    prior/intuition probe with no chain of thought. That is a fine knowledge
    measurement but it cannot speak to *reasoning depth*, which is the whole
    question the architect bench exists to settle. This variant asks for
    step-by-step work and a tagged final answer.
    """

    suite_name = "gpqa_diamond_cot"

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        q = super()._row_to_prompt(idx, row)
        base = q["prompt"].replace(
            "Answer with the letter only (A, B, C, or D).",
            "Reason step by step, then give your final answer on its own last "
            "line in the form: ANSWER: <letter>",
        )
        q["prompt"] = base
        q["suite"] = "gpqa_diamond_cot"
        q["id"] = q["id"].replace("gpqa_diamond_", "gpqa_diamond_cot_", 1)
        return q
