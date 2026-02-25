#!/usr/bin/env python3
from __future__ import annotations

"""
Benchmark Results Storage

Manages benchmark results:
- One JSON file per model-config combination
- Append-only JSONL index for all runs
- Skip logic for existing results

Directory structure:
  /mnt/raid0/llm/claude/benchmarks/results/
  ├── runs/
  │   └── YYYYMMDD_HHMMSS/           # Run directory
  │       ├── manifest.json           # What was tested
  │       └── {model}_{config}.json   # Results for model-config
  └── index.jsonl                     # Append-only index
"""

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


# Default paths
RESULTS_DIR = "/mnt/raid0/llm/claude/benchmarks/results"
INDEX_FILE = "index.jsonl"


@dataclass
class QuestionResult:
    """Result for a single question."""

    question_id: str
    prompt: str
    response: str
    tokens_per_second: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_time_ms: Optional[float] = None
    algorithmic_score: Optional[int] = None
    score_reason: Optional[str] = None
    acceptance_rate: Optional[float] = None


@dataclass
class ModelConfigResult:
    """Results for a model-config combination."""

    model_role: str
    model_path: str
    config_name: str
    run_id: str
    timestamp: str
    results: dict[str, dict[str, QuestionResult]] = field(default_factory=dict)  # suite -> question_id -> result
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        results_dict = {}
        for suite, questions in self.results.items():
            results_dict[suite] = {}
            for qid, result in questions.items():
                results_dict[suite][qid] = asdict(result)

        return {
            "model_role": self.model_role,
            "model_path": self.model_path,
            "config_name": self.config_name,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "results": results_dict,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfigResult":
        """Create from JSON dict."""
        results = {}
        for suite, questions in data.get("results", {}).items():
            results[suite] = {}
            for qid, qdata in questions.items():
                results[suite][qid] = QuestionResult(**qdata)

        return cls(
            model_role=data["model_role"],
            model_path=data["model_path"],
            config_name=data["config_name"],
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            results=results,
            summary=data.get("summary", {}),
        )


class ResultsManager:
    """Manages benchmark results storage."""

    def __init__(self, results_dir: str = RESULTS_DIR):
        self.results_dir = Path(results_dir)
        self.runs_dir = self.results_dir / "runs"
        self.index_file = self.results_dir / INDEX_FILE

        # Ensure directories exist
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def generate_run_id(self) -> str:
        """Generate a new run ID based on timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_run_dir(self, run_id: str) -> Path:
        """Get the directory for a specific run."""
        return self.runs_dir / run_id

    def get_result_file(self, run_id: str, model_role: str, config_name: str) -> Path:
        """Get the path to a result file."""
        run_dir = self.get_run_dir(run_id)
        filename = f"{model_role}_{config_name}.json"
        return run_dir / filename

    def result_exists(
        self,
        run_id: str,
        model_role: str,
        config_name: str,
        suite: Optional[str] = None,
        question_id: Optional[str] = None,
    ) -> bool:
        """Check if a result already exists.

        Args:
            run_id: The run identifier.
            model_role: The model role name.
            config_name: The configuration name.
            suite: Optional suite to check.
            question_id: Optional question to check.

        Returns:
            True if result exists.
        """
        result_file = self.get_result_file(run_id, model_role, config_name)
        if not result_file.exists():
            return False

        if suite is None and question_id is None:
            return True

        # Check for specific suite/question
        with open(result_file) as f:
            data = json.load(f)

        results = data.get("results", {})

        if suite and question_id:
            return suite in results and question_id in results[suite]
        elif suite:
            return suite in results
        else:
            # question_id only - check all suites
            for s_results in results.values():
                if question_id in s_results:
                    return True
            return False

    def load_result(
        self, run_id: str, model_role: str, config_name: str
    ) -> Optional[ModelConfigResult]:
        """Load an existing result file.

        Returns:
            ModelConfigResult or None if not found.
        """
        result_file = self.get_result_file(run_id, model_role, config_name)
        if not result_file.exists():
            return None

        with open(result_file) as f:
            data = json.load(f)

        return ModelConfigResult.from_dict(data)

    def save_result(self, result: ModelConfigResult) -> Path:
        """Save a result to disk.

        Creates or updates the result file and appends to index.

        Returns:
            Path to the saved file.
        """
        run_dir = self.get_run_dir(result.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        result_file = self.get_result_file(
            result.run_id, result.model_role, result.config_name
        )

        # Update summary
        result.summary = self._compute_summary(result)

        # Save result file
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Append to index
        self._append_to_index(result)

        return result_file

    def add_question_result(
        self,
        run_id: str,
        model_role: str,
        config_name: str,
        model_path: str,
        suite: str,
        question_result: QuestionResult,
    ) -> ModelConfigResult:
        """Add or update a single question result.

        Loads existing results if present, adds the question, and saves.

        Returns:
            Updated ModelConfigResult.
        """
        # Load existing or create new
        existing = self.load_result(run_id, model_role, config_name)

        if existing:
            result = existing
        else:
            result = ModelConfigResult(
                model_role=model_role,
                model_path=model_path,
                config_name=config_name,
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
            )

        # Add question result
        if suite not in result.results:
            result.results[suite] = {}
        result.results[suite][question_result.question_id] = question_result

        # Save
        self.save_result(result)

        return result

    def _compute_summary(self, result: ModelConfigResult) -> dict[str, Any]:
        """Compute summary statistics for a result."""
        total_tps = []
        total_scores = []
        questions_tested = 0
        questions_passed = 0

        for suite_results in result.results.values():
            for qresult in suite_results.values():
                questions_tested += 1

                if qresult.tokens_per_second is not None:
                    total_tps.append(qresult.tokens_per_second)

                if qresult.algorithmic_score is not None:
                    total_scores.append(qresult.algorithmic_score)
                    if qresult.algorithmic_score >= 2:
                        questions_passed += 1

        return {
            "avg_tokens_per_second": sum(total_tps) / len(total_tps) if total_tps else None,
            "avg_algorithmic_score": sum(total_scores) / len(total_scores) if total_scores else None,
            "questions_tested": questions_tested,
            "questions_passed": questions_passed,
        }

    def _append_to_index(self, result: ModelConfigResult) -> None:
        """Append a summary entry to the JSONL index."""
        entry = {
            "run_id": result.run_id,
            "model_role": result.model_role,
            "config_name": result.config_name,
            "timestamp": result.timestamp,
            "summary": result.summary,
        }

        with open(self.index_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def add_speed_result(
        self,
        run_id: str,
        model_role: str,
        config_name: str,
        model_path: str,
        tokens_per_second: float,
        inherits_quality_from: str,
        acceptance_rate: Optional[float] = None,
    ) -> ModelConfigResult:
        """Add a speed-test-only result (inherits quality from baseline).

        For speculative decoding configs where quality is identical to baseline,
        we only measure speed and reference the baseline for quality scores.

        Returns:
            ModelConfigResult with speed data and inheritance reference.
        """
        result = ModelConfigResult(
            model_role=model_role,
            model_path=model_path,
            config_name=config_name,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
        )

        # Store speed test in a special "speed_test" suite
        result.results["speed_test"] = {
            "speed_measurement": QuestionResult(
                question_id="speed_measurement",
                prompt="[speed test prompt]",
                response="[speed test only - quality inherited]",
                tokens_per_second=tokens_per_second,
                acceptance_rate=acceptance_rate,
            )
        }

        # Mark as inheriting quality from baseline
        result.summary = {
            "avg_tokens_per_second": tokens_per_second,
            "acceptance_rate": acceptance_rate,
            "speed_test_only": True,
            "inherits_quality_from": inherits_quality_from,
        }

        # Save
        self.save_result(result)

        return result

    def get_all_runs(self) -> list[str]:
        """Get all run IDs (excludes test_ runs)."""
        if not self.runs_dir.exists():
            return []
        return sorted([
            d.name for d in self.runs_dir.iterdir()
            if d.is_dir() and not d.name.startswith("test_")
        ])

    def get_latest_run(self) -> Optional[str]:
        """Get the most recent run ID by modification time."""
        if not self.runs_dir.exists():
            return None
        runs = [
            d for d in self.runs_dir.iterdir()
            if d.is_dir() and not d.name.startswith("test_")
        ]
        if not runs:
            return None
        # Sort by modification time, newest first
        latest = max(runs, key=lambda d: d.stat().st_mtime)
        return latest.name


def save_result(
    run_id: str,
    model_role: str,
    config_name: str,
    model_path: str,
    suite: str,
    question_result: QuestionResult,
) -> Path:
    """Convenience function to save a single question result."""
    manager = ResultsManager()
    result = manager.add_question_result(
        run_id, model_role, config_name, model_path, suite, question_result
    )
    return manager.get_result_file(run_id, model_role, config_name)


def result_exists(
    run_id: str,
    model_role: str,
    config_name: str,
    suite: Optional[str] = None,
    question_id: Optional[str] = None,
) -> bool:
    """Convenience function to check if result exists."""
    manager = ResultsManager()
    return manager.result_exists(run_id, model_role, config_name, suite, question_id)


# Model registry path for model-path deduplication
MODEL_REGISTRY_PATH = "/mnt/raid0/llm/claude/orchestration/model_registry.yaml"


def get_roles_by_model_path(model_path: str) -> list[str]:
    """Get all roles that use the same model path.

    Reads model_registry.yaml and returns all role names that point to the
    same underlying model file. This enables skip logic based on model path
    rather than role name.

    Args:
        model_path: Absolute path to the model file.

    Returns:
        List of role names using this model path.
    """
    if not os.path.exists(MODEL_REGISTRY_PATH):
        return []

    with open(MODEL_REGISTRY_PATH) as f:
        registry = yaml.safe_load(f)

    roles_with_path = []
    model_basename = os.path.basename(model_path)

    for role_name, role_config in registry.get("roles", {}).items():
        # Model path can be in role_config["model"]["path"] or role_config["model_path"]
        model_info = role_config.get("model", {})
        if isinstance(model_info, dict):
            role_model_path = model_info.get("path", "")
        else:
            role_model_path = role_config.get("model_path", "")

        # Match on full path or basename (for flexibility)
        if role_model_path == model_path or os.path.basename(role_model_path) == model_basename:
            roles_with_path.append(role_name)

    return roles_with_path


def result_exists_for_model(
    run_id: str,
    model_path: str,
    config_name: str,
    suite: str,
    question_id: str,
) -> Optional[str]:
    """Check if ANY role using the same model path has results for this question.

    This enables model-path-aware skip logic: if frontdoor has already tested
    a question, coder_escalation (which uses the same model) can skip that test
    and copy the result.

    Args:
        run_id: The run identifier.
        model_path: Path to the model file being tested.
        config_name: The configuration name (e.g., 'baseline', 'moe4').
        suite: The test suite name.
        question_id: The question identifier.

    Returns:
        The role name that has the result, or None if no result exists.
    """
    roles_with_same_model = get_roles_by_model_path(model_path)
    manager = ResultsManager()

    for other_role in roles_with_same_model:
        if manager.result_exists(run_id, other_role, config_name, suite, question_id):
            return other_role

    return None


def copy_result_from_role(
    run_id: str,
    from_role: str,
    to_role: str,
    config_name: str,
    suite: str,
    question_id: str,
    model_path: str,
) -> bool:
    """Copy a specific question result from one role to another.

    Used when the same model is tested under different role names. Instead of
    re-running the test, copy the existing result.

    Args:
        run_id: The run identifier.
        from_role: Source role that has the result.
        to_role: Destination role to copy to.
        config_name: The configuration name.
        suite: The test suite name.
        question_id: The question identifier.
        model_path: Path to the model (for the new result's metadata).

    Returns:
        True if copy succeeded, False otherwise.
    """
    manager = ResultsManager()

    # Load source result
    source_result = manager.load_result(run_id, from_role, config_name)
    if not source_result:
        return False

    # Get the question result from source
    if suite not in source_result.results:
        return False
    if question_id not in source_result.results[suite]:
        return False

    source_question = source_result.results[suite][question_id]

    # Add to destination role
    manager.add_question_result(
        run_id=run_id,
        model_role=to_role,
        config_name=config_name,
        model_path=model_path,
        suite=suite,
        question_result=source_question,
    )

    return True


if __name__ == "__main__":
    print("=== Results Manager Test ===\n")

    manager = ResultsManager()

    # Show existing runs
    runs = manager.get_all_runs()
    print(f"Existing runs: {runs}")
    print(f"Latest run: {manager.get_latest_run()}")

    # Create a test result
    run_id = "test_" + manager.generate_run_id()
    print(f"\nCreating test run: {run_id}")

    qresult = QuestionResult(
        question_id="t1_q1_test",
        prompt="Test prompt",
        response="Test response",
        tokens_per_second=42.5,
        prompt_tokens=10,
        completion_tokens=50,
        total_time_ms=1176.5,
        algorithmic_score=3,
        score_reason="Test passed",
    )

    path = save_result(
        run_id=run_id,
        model_role="test_model",
        config_name="baseline",
        model_path="/mnt/raid0/llm/models/test.gguf",
        suite="thinking",
        question_result=qresult,
    )
    print(f"Saved to: {path}")

    # Check if exists
    exists = result_exists(run_id, "test_model", "baseline", "thinking", "t1_q1_test")
    print(f"Result exists: {exists}")

    # Clean up test file
    if path.exists():
        path.unlink()
        print("Cleaned up test file")
