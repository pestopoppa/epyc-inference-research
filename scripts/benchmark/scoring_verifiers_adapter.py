#!/usr/bin/env python3
"""EV-3: NVIDIA Scoring Verifiers benchmark adapter.

Source: nvidia/Scoring-Verifiers on HuggingFace.
  - HE-R+ (HumanEval-Reasoning+): code-correctness verification tasks.
  - Additional verifier tasks from the NVIDIA verifier benchmark suite.

Each item presents a code snippet (or solution) and asks the model to judge
correctness — binary (correct / incorrect) or scored (0–1).  The ground-truth
label is the oracle verifier decision.

Scoring:
  - scoring_method: "multiple_choice" (binary correct/incorrect)
  - scoring_method_continuous: "exact_match" on normalised label

Suite registration: "scoring_verifiers"

Dataset download (manual step if HF is unreachable):
    python -c "
    from huggingface_hub import snapshot_download
    snapshot_download('nvidia/Scoring-Verifiers',
                      repo_type='dataset',
                      local_dir='/mnt/raid0/llm/data/eval/scoring_verifiers')
    "

Usage:
    from scoring_verifiers_adapter import ScoringVerifiersAdapter
    adapter = ScoringVerifiersAdapter()
    questions = adapter.sample(n=10, seed=42)

    # Or load from a local path:
    adapter = ScoringVerifiersAdapter(
        local_path='/mnt/raid0/llm/data/eval/scoring_verifiers'
    )
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Optional

# Allow standalone import outside the benchmarks package
try:
    from dataset_adapters import BaseAdapter
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset_adapters import BaseAdapter

# Default local cache path — adapter falls back to HF streaming if absent
_DEFAULT_LOCAL_PATH = Path("/mnt/raid0/llm/data/eval/scoring_verifiers")

# HuggingFace dataset identifier
_HF_DATASET_ID = "nvidia/Scoring-Verifiers"


class ScoringVerifiersAdapter(BaseAdapter):
    """NVIDIA Scoring-Verifiers: verifier benchmark items (HE-R+ and companions).

    Each item contains a problem, a candidate solution, and an oracle label
    indicating whether the solution is correct.  The model acts as a verifier.

    Tiers:
      T1 — simple / syntactically-trivial solutions (easy to judge)
      T2 — functionally correct but stylistically complex
      T3 — subtle bugs or edge-case failures (hard to detect)

    Args:
        local_path: Optional local directory downloaded from HF.  If None,
            the adapter tries the default path then HF streaming.
        subset: HF config name (default "all").  Pass e.g. "he_r_plus" for the
            HumanEval-Reasoning+ subset if the dataset uses named configs.
    """

    suite_name = "scoring_verifiers"
    has_real_tiers = True

    def __init__(
        self,
        local_path: Optional[Path | str] = None,
        subset: str = "all",
    ):
        self._local_path: Path | None = (
            Path(local_path) if local_path else _DEFAULT_LOCAL_PATH
        )
        self._subset = subset
        self._dataset: list[dict] | None = None

    # ── loading ─────────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._dataset is not None:
            return

        # 1. Try local JSONL (snapshot_download path)
        if self._local_path and self._local_path.exists():
            rows = self._load_from_local(self._local_path)
            if rows:
                self._dataset = self._expand_solution_rows(rows)
                return

        # 2. Fallback: HF datasets streaming
        try:
            import datasets as hf
            ds = hf.load_dataset(_HF_DATASET_ID, split="test")
            self._dataset = self._expand_solution_rows([dict(row) for row in ds])
        except Exception:
            try:
                import datasets as hf
                ds = hf.load_dataset(_HF_DATASET_ID, split="train")
                self._dataset = self._expand_solution_rows([dict(row) for row in ds])
            except Exception as e:
                print(
                    f"  [adapter] ScoringVerifiers load failed: {e}\n"
                    "  Manual download:\n"
                    "    python -c \"from huggingface_hub import snapshot_download; "
                    f"snapshot_download('{_HF_DATASET_ID}', repo_type='dataset', "
                    f"local_dir='{_DEFAULT_LOCAL_PATH}')\""
                )
                self._dataset = []

    @staticmethod
    def _load_from_local(base: Path) -> list[dict]:
        """Load rows from a local snapshot directory (JSONL or parquet files)."""
        rows: list[dict] = []

        # Look for JSONL files first
        jsonl_files = sorted(base.rglob("*.jsonl"))
        if jsonl_files:
            for jf in jsonl_files:
                subset = jf.stem
                for line in jf.read_text(encoding="utf-8").strip().split("\n"):
                    line = line.strip()
                    if line:
                        try:
                            row = json.loads(line)
                            row.setdefault("subset", subset)
                            rows.append(row)
                        except json.JSONDecodeError:
                            pass
            if rows:
                return rows

        # Try parquet via pandas
        parquet_files = sorted(base.rglob("*.parquet"))
        if parquet_files:
            try:
                import pandas as pd
                dfs = [pd.read_parquet(f) for f in parquet_files]
                import pandas
                combined = pandas.concat(dfs, ignore_index=True)
                return combined.to_dict(orient="records")
            except ImportError:
                pass

        return rows

    @staticmethod
    def _expand_solution_rows(rows: list[dict]) -> list[dict]:
        """Expand benchmark problems into per-candidate verifier examples.

        The Scoring-Verifiers JSONL files store one programming problem per row
        and put candidate solutions plus oracle scores in ``all_solutions``.
        EV-4 needs calibration labels per candidate, not one unlabeled problem
        row, so each solution becomes its own verifier item.
        """
        expanded: list[dict] = []
        for row_idx, row in enumerate(rows):
            solutions = row.get("all_solutions")
            if not isinstance(solutions, list) or not solutions:
                expanded.append(row)
                continue

            for sol_idx, solution in enumerate(solutions):
                if not isinstance(solution, dict):
                    continue
                item = {
                    "id": f"{row.get('task_id', row.get('id', row_idx))}::sol{sol_idx}",
                    "subset": row.get("subset", row.get("split", "unknown")),
                    "task_id": row.get("task_id"),
                    "problem": row.get("prompt", row.get("problem", "")),
                    "solution": solution.get("solution", ""),
                    "label": solution.get("average_test_score"),
                    "average_test_score": solution.get("average_test_score"),
                    "rank": solution.get("rank"),
                    "entry_point": row.get("entry_point"),
                    "canonical_solution": row.get("canonical_solution"),
                    "test": row.get("test", row.get("assertion", row.get("test_list"))),
                    "source_row": row_idx,
                    "source_solution_index": sol_idx,
                }
                expanded.append(item)

        return expanded

    # ── prompt construction ──────────────────────────────────────────────────

    @staticmethod
    def _normalise_label(raw_label) -> str:
        """Coerce oracle label to 'correct' or 'incorrect'."""
        if raw_label is None:
            return "incorrect"
        s = str(raw_label).strip().lower()
        if s in {"1", "true", "yes", "correct", "pass", "passed"}:
            return "correct"
        if s in {"0", "false", "no", "incorrect", "fail", "failed"}:
            return "incorrect"
        # Numeric score: ≥0.5 → correct
        try:
            return "correct" if float(s) >= 0.5 else "incorrect"
        except ValueError:
            return "incorrect"

    @staticmethod
    def _estimate_tier(row: dict) -> int:
        """Heuristic tier from solution length and problem complexity."""
        solution = str(row.get("solution", row.get("completion", row.get("code", ""))))
        problem = str(row.get("problem", row.get("prompt", row.get("task", ""))))
        sol_lines = solution.count("\n")
        # T3: long solution or complex problem (may hide subtle bugs)
        if sol_lines > 20 or len(problem) > 400:
            return 3
        # T1: very short solution (trivial to judge)
        if sol_lines <= 4 and len(problem) <= 150:
            return 1
        return 2

    def _get_tier_for_index(self, idx: int) -> int:
        return self._estimate_tier(self._dataset[idx])

    def _row_to_prompt(self, idx: int, row: dict) -> dict:
        # Field names vary across NVIDIA verifier subsets — try common aliases
        problem = (
            row.get("problem")
            or row.get("prompt")
            or row.get("task")
            or row.get("question")
            or ""
        )
        solution = (
            row.get("solution")
            or row.get("completion")
            or row.get("code")
            or row.get("response")
            or ""
        )
        raw_label = (
            row.get("label")
            or row.get("is_correct")
            or row.get("score")
            or row.get("verdict")
            or row.get("ground_truth")
        )
        subset_tag = str(row.get("subset", row.get("split", row.get("source", "unknown"))))
        item_id = str(row.get("id", row.get("idx", idx)))

        expected = self._normalise_label(raw_label)
        tier = self._estimate_tier(row)

        # Build verifier prompt
        prompt_parts = [
            "You are an expert code verifier.",
            "",
            "## Problem",
            problem.strip() if problem else "(no problem statement provided)",
            "",
            "## Candidate Solution",
            "```python" if "python" in str(row.get("language", "python")).lower() else "```",
            solution.strip() if solution else "(empty solution)",
            "```",
            "",
            "Does this solution correctly solve the problem?",
            "",
            "Answer with exactly one word: **correct** or **incorrect**.",
        ]

        return {
            "id": f"sv_{subset_tag}_{item_id}",
            "suite": "scoring_verifiers",
            "prompt": "\n".join(prompt_parts),
            "context": "",
            "expected": expected,
            "scoring": [],
            "image_path": "",
            "tier": tier,
            "scoring_method": "multiple_choice",
            "scoring_config": {
                "choices": ["correct", "incorrect"],
                "normalize": True,
            },
            "metadata": {
                "subset": subset_tag,
                "raw_label": str(raw_label),
                "average_test_score": row.get("average_test_score"),
                "rank": row.get("rank"),
                "task_id": row.get("task_id"),
            },
        }


if __name__ == "__main__":
    adapter = ScoringVerifiersAdapter()
    adapter._ensure_loaded()
    total = adapter.total_available
    print(f"ScoringVerifiersAdapter: {total} items loaded")
    if total > 0:
        samples = adapter.sample(n=3, seed=42)
        for s in samples:
            print(f"  id={s['id']}  expected={s['expected']}  tier={s['tier']}")
