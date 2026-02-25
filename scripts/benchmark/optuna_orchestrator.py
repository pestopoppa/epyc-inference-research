#!/usr/bin/env python3
from __future__ import annotations

"""
Optuna-based Runtime Parameter Optimization for Orchestrator

Optimizes RUNTIME-TUNABLE parameters only (no server restart required):
1. Routing: confidence_threshold, q_weight, min_q_value
2. Escalation: max_retries, max_escalations

NOTE: Model-specific params (temperature, speculative_k, entropy_threshold)
are excluded - they require server restarts and have been benchmarked separately.
See orchestration/model_registry.yaml for those configurations.

Usage:
    python scripts/benchmark/optuna_orchestrator.py --layer routing --trials 30
    python scripts/benchmark/optuna_orchestrator.py --layer escalation --trials 25
    python scripts/benchmark/optuna_orchestrator.py --analyze --select-robust --checkpoint checkpoint.yaml
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import optuna
    import yaml
    import httpx
    from sklearn.cluster import KMeans
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install optuna pyyaml httpx scikit-learn numpy")
    sys.exit(1)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


def _read_registry_timeout(category: str, key: str, fallback: int) -> int:
    """Read timeout from model_registry.yaml."""
    registry_path = PROJECT_ROOT / "orchestration" / "model_registry.yaml"
    try:
        with registry_path.open() as f:
            data = yaml.safe_load(f)
        timeouts = data.get("runtime_defaults", {}).get("timeouts", {})
        cat_data = timeouts.get(category, {})
        return cat_data.get(key, timeouts.get("default", fallback))
    except Exception as e:
        return fallback


# Constants
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = _read_registry_timeout("backends", "inference_default", 120)
CHECKPOINT_PATH = PROJECT_ROOT / "orchestration" / "optimization_checkpoint.yaml"
STUDY_DB = PROJECT_ROOT / "orchestration" / "optuna_study.db"


@dataclass
class LayerConfig:
    """Configuration for a layer's optimization."""
    name: str
    params: dict  # param_name -> (min, max, type)
    test_suite: str
    metric_weights: dict


# Layer configurations - RUNTIME-TUNABLE PARAMS ONLY
# Model-specific params (temperature, speculative_k) require server restart
# and are benchmarked separately in orchestration/model_registry.yaml
LAYER_CONFIGS = {
    "routing": LayerConfig(
        name="routing",
        params={
            # MemRL routing parameters (all runtime-tunable)
            "confidence_threshold": (0.4, 0.9, "float"),  # When to trust learned routing
            "q_weight": (0.5, 0.9, "float"),  # Balance: learned vs semantic similarity
            "min_q_value": (0.2, 0.5, "float"),  # Minimum Q-value to consider memory
            "min_similarity": (0.2, 0.5, "float"),  # Minimum semantic similarity
        },
        test_suite="t1_routing",  # Routing accuracy tests
        metric_weights={
            "parse_success_rate": 0.3,
            "task_completion_rate": 0.4,
            "avg_turns_inverse": 0.3,
        }
    ),
    "escalation": LayerConfig(
        name="escalation",
        params={
            # Escalation behavior (runtime-tunable)
            "max_retries": (1, 4, "int"),  # Retries before escalation
            "max_escalations": (1, 3, "int"),  # Max escalation depth
        },
        test_suite="t3_escalation",  # Escalation accuracy tests
        metric_weights={
            "task_completion_rate": 0.4,
            "escalation_accuracy": 0.4,
            "avg_turns_inverse": 0.2,
        }
    ),
    "learning": LayerConfig(
        name="learning",
        params={
            # Q-learning parameters (runtime-tunable, affect future behavior)
            "learning_rate": (0.05, 0.2, "float"),  # Q-value update speed
            "success_reward": (0.7, 1.0, "float"),  # Reward for success
            "failure_reward": (-0.7, -0.3, "float"),  # Penalty for failure
        },
        test_suite="t2_delegation",  # General task completion
        metric_weights={
            "task_completion_rate": 0.5,
            "execution_success_rate": 0.3,
            "avg_turns_inverse": 0.2,
        }
    ),
}


def load_checkpoint(path: Path) -> dict:
    """Load optimization checkpoint."""
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {
        "schema_version": 1,
        "meta": {
            "started": None,
            "last_updated": None,
        },
        "layers": {
            name: {
                "status": "pending",
                "optimal_params": None,
                "metrics": None,
                "trials_run": 0,
            }
            for name in LAYER_CONFIGS.keys()
        }
    }


def save_checkpoint(checkpoint: dict, path: Path):
    """Save optimization checkpoint."""
    checkpoint["meta"]["last_updated"] = datetime.now().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(checkpoint, f, default_flow_style=False, sort_keys=False)


def get_frozen_params(checkpoint: dict) -> dict:
    """Get parameters from completed (frozen) layers."""
    frozen = {}
    for layer_name, layer_data in checkpoint.get("layers", {}).items():
        if layer_data.get("status") == "complete" and layer_data.get("optimal_params"):
            frozen[layer_name] = layer_data["optimal_params"]
    return frozen


def run_test_suite(
    suite: str,
    params: dict,
    frozen_params: dict,
    api_url: str,
    timeout: int
) -> dict:
    """Run a test suite and return metrics."""

    # Merge frozen params with current trial params
    config = {}
    for layer_name, layer_params in frozen_params.items():
        config.update(layer_params)
    config.update(params)

    # Import bench_orchestrator functions
    try:
        from scripts.benchmark.bench_orchestrator import run_suite
    except ImportError:
        # Fallback to direct execution
        pass

    results = {
        "parse_success_rate": 0.0,
        "task_completion_rate": 0.0,
        "avg_turns": 0.0,
        "latency_ms": 0.0,
        "schema_validation_rate": 0.0,
        "execution_success_rate": 0.0,
        "escalation_accuracy": 0.0,
    }

    try:
        # Call the orchestrator API with test prompts
        prompt_dir = PROJECT_ROOT / "benchmarks" / "prompts" / "v1" / "orchestrator" / suite

        if not prompt_dir.exists():
            print(f"Warning: Suite directory not found: {prompt_dir}")
            return results

        total_tests = 0
        parse_success = 0
        task_complete = 0
        schema_valid = 0
        execution_success = 0
        escalation_correct = 0
        total_turns = 0
        total_latency = 0

        for prompt_file in prompt_dir.glob("*.txt"):
            total_tests += 1
            prompt_text = prompt_file.read_text().split("---")[0].strip()

            # Call API
            try:
                with httpx.Client(timeout=timeout) as client:
                    start = time.perf_counter()
                    response = client.post(
                        f"{api_url}/chat",
                        json={
                            "prompt": prompt_text,
                            "real_mode": True,
                            **config
                        }
                    )
                    latency = (time.perf_counter() - start) * 1000

                    if response.status_code == 200:
                        data = response.json()

                        # Parse success: did we get valid output?
                        if data.get("answer"):
                            parse_success += 1

                        # Task completion: was FINAL() called?
                        if data.get("final_called"):
                            task_complete += 1

                        # Schema validation (for structured output tests)
                        if "json" in prompt_file.name.lower():
                            try:
                                json.loads(data.get("answer", ""))
                                schema_valid += 1
                            except Exception as e:
                                pass

                        # Execution success (code ran without error)
                        if not data.get("execution_error"):
                            execution_success += 1

                        # Escalation accuracy
                        if data.get("escalation_triggered"):
                            # Check if escalation was appropriate
                            if "complex" in prompt_file.name or "architecture" in prompt_file.name:
                                escalation_correct += 1

                        total_turns += data.get("turns", 0)
                        total_latency += latency

            except Exception as e:
                print(f"  Error on {prompt_file.name}: {e}")
                continue

        if total_tests > 0:
            results["parse_success_rate"] = parse_success / total_tests
            results["task_completion_rate"] = task_complete / total_tests
            results["avg_turns"] = total_turns / total_tests if total_tests > 0 else 0
            results["latency_ms"] = total_latency / total_tests if total_tests > 0 else 0
            results["schema_validation_rate"] = schema_valid / total_tests
            results["execution_success_rate"] = execution_success / total_tests
            results["escalation_accuracy"] = escalation_correct / max(1, sum(1 for f in prompt_dir.glob("*.txt") if "complex" in f.name or "architecture" in f.name))

    except Exception as e:
        print(f"Error running test suite: {e}")

    return results


def compute_score(metrics: dict, weights: dict) -> float:
    """Compute weighted score from metrics."""
    score = 0.0

    for metric_name, weight in weights.items():
        if metric_name == "avg_turns_inverse":
            # Lower turns is better
            value = 1.0 / max(1.0, metrics.get("avg_turns", 1.0))
        elif metric_name == "latency_inverse":
            # Lower latency is better (normalize to 0-1 range)
            latency = metrics.get("latency_ms", 1000)
            value = 1.0 / max(1.0, latency / 1000)  # Convert to seconds
        else:
            value = metrics.get(metric_name, 0.0)

        score += weight * value

    return score


def create_objective(
    layer_config: LayerConfig,
    frozen_params: dict,
    api_url: str,
    timeout: int
):
    """Create Optuna objective function for a layer."""

    def objective(trial: optuna.Trial) -> float:
        # Suggest parameters
        params = {}
        for param_name, (min_val, max_val, param_type) in layer_config.params.items():
            if param_type == "float":
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            elif param_type == "int":
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)

        try:
            # Run tests
            metrics = run_test_suite(
                layer_config.test_suite,
                params,
                frozen_params,
                api_url,
                timeout
            )

            # Compute score
            score = compute_score(metrics, layer_config.metric_weights)

            # Store metrics for later analysis
            for key, value in metrics.items():
                trial.set_user_attr(key, value)

            return score

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0.0  # Return worst score on failure

    return objective


def cluster_select_robust(study: optuna.Study, top_percent: float = 0.2) -> dict:
    """Select robust config using cluster-based selection."""

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if len(trials) < 5:
        # Not enough trials for clustering, just return best
        return study.best_params

    # Get top performers
    trials_sorted = sorted(trials, key=lambda t: t.value or 0, reverse=True)
    n_top = max(3, int(len(trials) * top_percent))
    top_trials = trials_sorted[:n_top]

    # Extract parameter vectors
    param_names = list(study.best_params.keys())
    X = np.array([[t.params.get(p, 0) for p in param_names] for t in top_trials])

    # Normalize
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1  # Avoid division by zero
    X_norm = (X - X_min) / X_range

    # Cluster
    n_clusters = min(3, len(top_trials))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_norm)

    # Find largest cluster
    cluster_sizes = [sum(labels == i) for i in range(n_clusters)]
    largest_cluster = np.argmax(cluster_sizes)

    # Get centroid of largest cluster
    centroid = kmeans.cluster_centers_[largest_cluster]
    centroid_unnorm = centroid * X_range + X_min

    # Find trial closest to centroid
    cluster_mask = labels == largest_cluster
    cluster_X = X_norm[cluster_mask]
    cluster_trials = [t for t, m in zip(top_trials, cluster_mask) if m]

    distances = np.linalg.norm(cluster_X - centroid, axis=1)
    closest_idx = np.argmin(distances)
    robust_trial = cluster_trials[closest_idx]

    print(f"\nCluster analysis:")
    print(f"  Top {n_top} trials analyzed")
    print(f"  {n_clusters} clusters found")
    print(f"  Largest cluster: {cluster_sizes[largest_cluster]} trials")
    print(f"  Selected trial: #{robust_trial.number} (score: {robust_trial.value:.4f})")
    print(f"  vs Best trial: #{study.best_trial.number} (score: {study.best_value:.4f})")

    return robust_trial.params


def optimize_layer(
    layer_name: str,
    n_trials: int,
    checkpoint: dict,
    api_url: str,
    timeout: int,
    dry_run: bool = False
) -> dict:
    """Optimize a single layer."""

    layer_config = LAYER_CONFIGS.get(layer_name)
    if not layer_config:
        print(f"Error: Unknown layer: {layer_name}")
        return {}

    frozen_params = get_frozen_params(checkpoint)

    # Check if previous layers are complete
    layer_order = ["routing", "escalation", "learning"]
    layer_idx = layer_order.index(layer_name)
    for prev_layer in layer_order[:layer_idx]:
        if checkpoint["layers"][prev_layer]["status"] != "complete":
            print(f"Warning: Previous layer '{prev_layer}' not complete. Consider running it first.")

    print(f"\n{'='*60}")
    print(f"Optimizing Layer: {layer_name}")
    print(f"{'='*60}")
    print(f"Parameters: {list(layer_config.params.keys())}")
    print(f"Test suite: {layer_config.test_suite}")
    print(f"Trials: {n_trials}")
    print(f"Frozen params: {frozen_params}")
    print(f"{'='*60}\n")

    # Mark layer as in progress
    checkpoint["layers"][layer_name]["status"] = "in_progress"
    if not checkpoint["meta"]["started"]:
        checkpoint["meta"]["started"] = datetime.now().isoformat()

    if dry_run:
        print("DRY RUN - would optimize with these settings")
        return {}

    # Create or load study
    study_name = f"orchestrator_{layer_name}"
    storage = f"sqlite:///{STUDY_DB}"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Resuming existing study with {len(study.trials)} trials")
    except Exception as e:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        print("Created new study")

    # Create objective
    objective = create_objective(layer_config, frozen_params, api_url, timeout)

    # Run optimization
    remaining_trials = n_trials - len(study.trials)
    if remaining_trials > 0:
        print(f"\nRunning {remaining_trials} trials...")
        study.optimize(
            objective,
            n_trials=remaining_trials,
            show_progress_bar=True,
            callbacks=[
                lambda study, trial: save_checkpoint(checkpoint, CHECKPOINT_PATH)
            ]
        )

    # Select robust config
    print("\n\nSelecting robust configuration...")
    robust_params = cluster_select_robust(study)

    # Get metrics for robust config
    best_trial = None
    for t in study.trials:
        if t.params == robust_params:
            best_trial = t
            break

    if best_trial is None:
        best_trial = study.best_trial

    metrics = {attr: best_trial.user_attrs.get(attr, 0) for attr in [
        "parse_success_rate", "task_completion_rate", "avg_turns",
        "latency_ms", "schema_validation_rate", "execution_success_rate",
        "escalation_accuracy"
    ]}

    # Update checkpoint
    checkpoint["layers"][layer_name].update({
        "status": "complete",
        "optimal_params": robust_params,
        "metrics": metrics,
        "trials_run": len(study.trials),
        "best_score": best_trial.value,
    })

    save_checkpoint(checkpoint, CHECKPOINT_PATH)

    print(f"\n{'='*60}")
    print(f"Layer {layer_name} COMPLETE")
    print(f"{'='*60}")
    print(f"Optimal params: {robust_params}")
    print(f"Metrics: {metrics}")
    print(f"Score: {best_trial.value:.4f}")

    return robust_params


def analyze_checkpoint(checkpoint: dict):
    """Analyze and display checkpoint status."""

    print(f"\n{'='*60}")
    print("Optimization Checkpoint Analysis")
    print(f"{'='*60}")

    meta = checkpoint.get("meta", {})
    print(f"\nStarted: {meta.get('started', 'N/A')}")
    print(f"Last updated: {meta.get('last_updated', 'N/A')}")

    print(f"\n{'Layer':<15} {'Status':<12} {'Trials':<8} {'Score':<10}")
    print("-" * 50)

    for layer_name, layer_data in checkpoint.get("layers", {}).items():
        status = layer_data.get("status", "pending")
        trials = layer_data.get("trials_run", 0)
        score = layer_data.get("best_score", "-")
        if isinstance(score, float):
            score = f"{score:.4f}"
        print(f"{layer_name:<15} {status:<12} {trials:<8} {score:<10}")

        if layer_data.get("optimal_params"):
            print(f"  Params: {layer_data['optimal_params']}")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Optuna Orchestrator Optimizer")
    parser.add_argument(
        "--layer",
        choices=["routing", "escalation", "learning"],
        help="Layer to optimize (runtime-tunable params only)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=25,
        help="Number of trials to run"
    )
    parser.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT_PATH),
        help="Checkpoint file path"
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Orchestrator API URL"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Timeout per test"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze checkpoint status"
    )
    parser.add_argument(
        "--select-robust",
        action="store_true",
        help="Re-run robust selection on completed studies"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without running"
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)

    if args.analyze:
        analyze_checkpoint(checkpoint)
        return

    if args.select_robust:
        print("Re-running robust selection for all completed layers...")
        for layer_name in ["routing", "escalation", "learning"]:
            if checkpoint["layers"][layer_name]["status"] == "complete":
                study_name = f"orchestrator_{layer_name}"
                try:
                    study = optuna.load_study(
                        study_name=study_name,
                        storage=f"sqlite:///{STUDY_DB}"
                    )
                    robust_params = cluster_select_robust(study)
                    checkpoint["layers"][layer_name]["optimal_params"] = robust_params
                    print(f"{layer_name}: {robust_params}")
                except Exception as e:
                    print(f"Could not analyze {layer_name}: {e}")
        save_checkpoint(checkpoint, checkpoint_path)
        return

    if not args.layer:
        print("Error: --layer is required for optimization")
        print("Use --analyze to view checkpoint status")
        parser.print_help()
        sys.exit(1)

    optimize_layer(
        args.layer,
        args.trials,
        checkpoint,
        args.api_url,
        args.timeout,
        args.dry_run
    )


if __name__ == "__main__":
    main()
