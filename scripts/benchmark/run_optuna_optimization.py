#!/usr/bin/env python3
from __future__ import annotations

"""
Standalone Optuna Optimization for Orchestrator

A fire-and-forget script that manages the full optimization lifecycle:
1. Manages llama-server lifecycle (start/stop)
2. Manages orchestrator API lifecycle (start/stop uvicorn)
3. Auto-starts optuna-dashboard on 0.0.0.0:8050 for LAN monitoring
4. Runs layered optimization with per-trial checkpointing
5. Generates final report and updates model_registry.yaml

Usage:
    # Single layer optimization
    ./run_optuna_optimization.py --layer frontdoor --trials 30

    # All layers sequentially (overnight run)
    ./run_optuna_optimization.py --all-layers --trials-per-layer 25

    # With live dashboard (auto-starts on port 8050)
    ./run_optuna_optimization.py --layer frontdoor --dashboard

    # Resume from checkpoint
    ./run_optuna_optimization.py --resume

    # Show current status
    ./run_optuna_optimization.py --status

    # Dry run (show what would happen)
    ./run_optuna_optimization.py --layer frontdoor --dry-run
"""

import argparse
import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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
    print("Run: pip install optuna optuna-dashboard pyyaml httpx scikit-learn numpy")
    sys.exit(1)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Constants and Paths
# =============================================================================

LLAMA_CPP_PATH = Path("/mnt/raid0/llm/llama.cpp/build/bin")
LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_PATH = PROJECT_ROOT / "orchestration" / "optimization_checkpoint.yaml"
STUDY_DB = PROJECT_ROOT / "orchestration" / "optuna_study.db"
REGISTRY_PATH = PROJECT_ROOT / "orchestration" / "model_registry.yaml"
REPORT_PATH = PROJECT_ROOT / "orchestration" / "optimization_report.md"

# Server ports
FRONTDOOR_PORT = 8080
WORKER_PORT = 8082
API_PORT = 8000
DASHBOARD_PORT = 8050

# Default model (dev mode)
DEV_MODEL = Path("/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf")

# Timeouts - read from registry (single source of truth)
def _read_registry_timeout(category: str, key: str, fallback: int) -> int:
    """Read timeout from model_registry.yaml."""
    try:
        with REGISTRY_PATH.open() as f:
            data = yaml.safe_load(f)
        timeouts = data.get("runtime_defaults", {}).get("timeouts", {})
        cat_data = timeouts.get(category, {})
        return cat_data.get(key, timeouts.get("default", fallback))
    except Exception as e:
        return fallback

SERVER_HEALTH_TIMEOUT = 60  # seconds (local health check, not registry)
API_HEALTH_TIMEOUT = 30  # seconds (local health check)
TRIAL_TIMEOUT = _read_registry_timeout("benchmark", "optuna_trial", 300)  # seconds

# Global mock mode flag (set by --mock argument)
USE_MOCK_MODE = False


# =============================================================================
# Layer Configuration
# =============================================================================

@dataclass
class LayerConfig:
    """Configuration for a layer's optimization."""
    name: str
    params: dict  # param_name -> (min, max, type)
    test_suite: str
    metric_weights: dict
    default_trials: int = 25


LAYER_CONFIGS = {
    "frontdoor": LayerConfig(
        name="frontdoor",
        params={
            "temperature": (0.0, 0.5, "float"),
            "speculative_k": (8, 32, "int"),
            "entropy_threshold": (4.0, 6.0, "float"),
        },
        test_suite="t1_routing",
        metric_weights={
            "parse_success_rate": 0.4,
            "avg_turns_inverse": 0.3,
            "latency_inverse": 0.3,
        },
        default_trials=30
    ),
    "formalizer": LayerConfig(
        name="formalizer",
        params={
            "temperature": (0.0, 0.2, "float"),
        },
        test_suite="t2_delegation",
        metric_weights={
            "schema_validation_rate": 0.5,
            "latency_inverse": 0.5,
        },
        default_trials=20
    ),
    "specialists": LayerConfig(
        name="specialists",
        params={
            "temperature": (0.1, 0.5, "float"),
            "speculative_k": (8, 24, "int"),
            "early_abort_tokens": (50, 150, "int"),
        },
        test_suite="t3_escalation",
        metric_weights={
            "task_completion_rate": 0.4,
            "escalation_accuracy": 0.3,
            "latency_inverse": 0.3,
        },
        default_trials=30
    ),
    "workers": LayerConfig(
        name="workers",
        params={
            "temperature": (0.2, 0.6, "float"),
            "repetition_threshold": (0.15, 0.35, "float"),
        },
        test_suite="t2_delegation",
        metric_weights={
            "execution_success_rate": 0.5,
            "latency_inverse": 0.5,
        },
        default_trials=25
    ),
}

LAYER_ORDER = ["frontdoor", "formalizer", "specialists", "workers"]


# =============================================================================
# Process Management
# =============================================================================

class ProcessManager:
    """Manages subprocesses with cleanup on exit."""

    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, cleaning up...")
        self.cleanup()
        sys.exit(1)

    def start(self, name: str, cmd: list[str], log_file: Optional[Path] = None) -> bool:
        """Start a process and track it."""
        if name in self.processes:
            print(f"Process {name} already running")
            return True

        try:
            if log_file:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_fh = open(log_file, "w")
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid
                )
            else:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid
                )

            self.processes[name] = proc
            print(f"Started {name} (PID: {proc.pid})")
            return True

        except Exception as e:
            print(f"Failed to start {name}: {e}")
            return False

    def stop(self, name: str, timeout: int = 5):
        """Stop a process gracefully."""
        if name not in self.processes:
            return

        proc = self.processes[name]
        if proc.poll() is not None:
            del self.processes[name]
            return

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            pass

        del self.processes[name]
        print(f"Stopped {name}")

    def is_running(self, name: str) -> bool:
        """Check if a process is still running."""
        if name not in self.processes:
            return False
        return self.processes[name].poll() is None

    def cleanup(self):
        """Stop all managed processes."""
        for name in list(self.processes.keys()):
            self.stop(name)


def kill_port(port: int):
    """Kill any process using the specified port."""
    try:
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(1)
    except Exception as e:
        pass


def wait_for_health(url: str, timeout: int = _read_registry_timeout("external", "mcp_client", 30)) -> bool:
    """Wait for a health endpoint to respond."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{url}/health")
                if resp.status_code == 200:
                    return True
        except Exception as e:
            pass
        time.sleep(1)
    return False


def get_local_ip() -> str:
    """Get the local IP address for LAN access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        return "localhost"


# =============================================================================
# Server Lifecycle
# =============================================================================

def start_llama_servers(pm: ProcessManager, model_path: Path = DEV_MODEL) -> bool:
    """Start the llama-server instances for frontdoor and worker."""

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return False

    server_bin = LLAMA_CPP_PATH / "llama-server"
    if not server_bin.exists():
        print(f"Error: llama-server not found: {server_bin}")
        return False

    # Kill any existing processes on our ports
    for port in [FRONTDOOR_PORT, WORKER_PORT]:
        kill_port(port)

    # Start frontdoor server
    frontdoor_cmd = [
        "numactl", "--interleave=all",
        str(server_bin),
        "-m", str(model_path),
        "--host", "0.0.0.0",
        "--port", str(FRONTDOOR_PORT),
        "-np", "4",
        "-c", "4096",
        "-t", "16"
    ]
    if not pm.start("frontdoor", frontdoor_cmd, LOG_DIR / "llama-server-frontdoor.log"):
        return False

    # Start worker server
    worker_cmd = [
        "numactl", "--interleave=all",
        str(server_bin),
        "-m", str(model_path),
        "--host", "0.0.0.0",
        "--port", str(WORKER_PORT),
        "-np", "4",
        "-c", "4096",
        "-t", "16"
    ]
    if not pm.start("worker", worker_cmd, LOG_DIR / "llama-server-worker.log"):
        pm.stop("frontdoor")
        return False

    # Wait for health
    print("Waiting for servers to become healthy...")
    if not wait_for_health(f"http://localhost:{FRONTDOOR_PORT}", SERVER_HEALTH_TIMEOUT):
        print(f"Error: Frontdoor server failed to start. Check {LOG_DIR / 'llama-server-frontdoor.log'}")
        pm.stop("frontdoor")
        pm.stop("worker")
        return False

    if not wait_for_health(f"http://localhost:{WORKER_PORT}", SERVER_HEALTH_TIMEOUT):
        print(f"Error: Worker server failed to start. Check {LOG_DIR / 'llama-server-worker.log'}")
        pm.stop("frontdoor")
        pm.stop("worker")
        return False

    print(f"Llama servers healthy on ports {FRONTDOOR_PORT} and {WORKER_PORT}")
    return True


def start_orchestrator_api(pm: ProcessManager) -> bool:
    """Start the orchestrator API server."""

    kill_port(API_PORT)

    api_cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api:app",
        "--host", "0.0.0.0",
        "--port", str(API_PORT),
        "--log-level", "warning"
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    if not pm.start("api", api_cmd, LOG_DIR / "orchestrator-api.log"):
        return False

    print("Waiting for orchestrator API to become healthy...")
    if not wait_for_health(f"http://localhost:{API_PORT}", API_HEALTH_TIMEOUT):
        print(f"Error: Orchestrator API failed to start. Check {LOG_DIR / 'orchestrator-api.log'}")
        pm.stop("api")
        return False

    print(f"Orchestrator API healthy on port {API_PORT}")
    return True


def start_dashboard(pm: ProcessManager) -> bool:
    """Start the Optuna dashboard for LAN monitoring."""

    kill_port(DASHBOARD_PORT)

    dashboard_cmd = [
        sys.executable, "-m", "optuna_dashboard.run",
        f"sqlite:///{STUDY_DB}",
        "--host", "0.0.0.0",
        "--port", str(DASHBOARD_PORT)
    ]

    if pm.start("dashboard", dashboard_cmd, LOG_DIR / "optuna-dashboard.log"):
        local_ip = get_local_ip()
        print(f"\nOptuna Dashboard available at:")
        print(f"  Local:     http://localhost:{DASHBOARD_PORT}")
        print(f"  LAN:       http://{local_ip}:{DASHBOARD_PORT}")
        print()
        return True

    return False


def stop_all_servers(pm: ProcessManager):
    """Stop all servers in reverse order."""
    for name in ["dashboard", "api", "worker", "frontdoor"]:
        pm.stop(name)


# =============================================================================
# Checkpoint Management
# =============================================================================

def load_checkpoint() -> dict:
    """Load optimization checkpoint."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return yaml.safe_load(f) or {}
    return create_empty_checkpoint()


def create_empty_checkpoint() -> dict:
    """Create an empty checkpoint structure."""
    return {
        "schema_version": 1,
        "meta": {
            "started": None,
            "last_updated": None,
            "description": "Layered optimization checkpoint for orchestrator parameters"
        },
        "layers": {
            name: {
                "status": "pending",
                "optimal_params": None,
                "metrics": None,
                "trials_run": 0,
                "best_score": None,
            }
            for name in LAYER_CONFIGS.keys()
        },
        "history": []
    }


def save_checkpoint(checkpoint: dict):
    """Save optimization checkpoint."""
    checkpoint["meta"]["last_updated"] = datetime.now().isoformat()
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        yaml.dump(checkpoint, f, default_flow_style=False, sort_keys=False)


def get_frozen_params(checkpoint: dict) -> dict:
    """Get parameters from completed (frozen) layers."""
    frozen = {}
    for layer_name, layer_data in checkpoint.get("layers", {}).items():
        if layer_data.get("status") == "complete" and layer_data.get("optimal_params"):
            frozen[layer_name] = layer_data["optimal_params"]
    return frozen


def get_next_layer(checkpoint: dict) -> Optional[str]:
    """Get the next layer that needs optimization."""
    for layer_name in LAYER_ORDER:
        status = checkpoint["layers"][layer_name].get("status", "pending")
        if status in ["pending", "in_progress"]:
            return layer_name
    return None


# =============================================================================
# Test Suite Execution
# =============================================================================

def run_test_suite(
    suite: str,
    params: dict,
    frozen_params: dict,
    api_url: str = f"http://localhost:{API_PORT}",
    timeout: int = TRIAL_TIMEOUT
) -> dict:
    """Run a test suite and return functional metrics."""

    # Merge frozen params with current trial params
    config = {}
    for layer_name, layer_params in frozen_params.items():
        config.update(layer_params)
    config.update(params)

    results = {
        "parse_success_rate": 0.0,
        "task_completion_rate": 0.0,
        "avg_turns": 0.0,
        "latency_ms": 0.0,
        "schema_validation_rate": 0.0,
        "execution_success_rate": 0.0,
        "escalation_accuracy": 0.0,
    }

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
    escalation_tests = 0

    for prompt_file in prompt_dir.glob("*.txt"):
        total_tests += 1
        prompt_text = prompt_file.read_text().split("---")[0].strip()

        try:
            with httpx.Client(timeout=timeout) as client:
                start = time.perf_counter()
                response = client.post(
                    f"{api_url}/chat",
                    json={
                        "prompt": prompt_text,
                        "real_mode": not USE_MOCK_MODE,
                        "mock_mode": USE_MOCK_MODE,
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
                    if "json" in prompt_file.name.lower() or "structured" in prompt_file.name.lower():
                        try:
                            json.loads(data.get("answer", ""))
                            schema_valid += 1
                        except Exception as e:
                            pass

                    # Execution success (code ran without error)
                    if not data.get("execution_error"):
                        execution_success += 1

                    # Escalation accuracy
                    if "complex" in prompt_file.name or "architecture" in prompt_file.name:
                        escalation_tests += 1
                        if data.get("escalation_triggered"):
                            escalation_correct += 1

                    total_turns += data.get("turns", 1)
                    total_latency += latency

        except Exception as e:
            print(f"  Error on {prompt_file.name}: {e}")
            continue

    if total_tests > 0:
        results["parse_success_rate"] = parse_success / total_tests
        results["task_completion_rate"] = task_complete / total_tests
        results["avg_turns"] = total_turns / total_tests
        results["latency_ms"] = total_latency / total_tests
        results["schema_validation_rate"] = schema_valid / total_tests
        results["execution_success_rate"] = execution_success / total_tests
        results["escalation_accuracy"] = escalation_correct / max(1, escalation_tests)

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
            value = 1.0 / max(1.0, latency / 1000)
        else:
            value = metrics.get(metric_name, 0.0)

        score += weight * value

    return score


# =============================================================================
# Optuna Optimization
# =============================================================================

def create_objective(
    layer_config: LayerConfig,
    frozen_params: dict,
    checkpoint: dict
):
    """Create Optuna objective function for a layer."""

    consecutive_failures = 0
    max_consecutive_failures = 5

    def objective(trial: optuna.Trial) -> float:
        nonlocal consecutive_failures

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
                frozen_params
            )

            # Compute score
            score = compute_score(metrics, layer_config.metric_weights)

            # Store metrics for later analysis
            for key, value in metrics.items():
                trial.set_user_attr(key, value)

            # Reset failure counter on success
            if score > 0:
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            # Abort if too many consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(
                    f"Aborting: {max_consecutive_failures} consecutive trials scored 0. "
                    "The orchestrator may be broken."
                )

            return score

        except RuntimeError:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(
                    f"Aborting: {max_consecutive_failures} consecutive failures"
                )
            return 0.0

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
    X_range[X_range == 0] = 1
    X_norm = (X - X_min) / X_range

    # Cluster
    n_clusters = min(3, len(top_trials))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_norm)

    # Find largest cluster
    cluster_sizes = [sum(labels == i) for i in range(n_clusters)]
    largest_cluster = int(np.argmax(cluster_sizes))

    # Get centroid of largest cluster
    centroid = kmeans.cluster_centers_[largest_cluster]

    # Find trial closest to centroid
    cluster_mask = labels == largest_cluster
    cluster_X = X_norm[cluster_mask]
    cluster_trials = [t for t, m in zip(top_trials, cluster_mask) if m]

    distances = np.linalg.norm(cluster_X - centroid, axis=1)
    closest_idx = int(np.argmin(distances))
    robust_trial = cluster_trials[closest_idx]

    print(f"\nCluster analysis:")
    print(f"  Top {n_top} trials analyzed")
    print(f"  {n_clusters} clusters found, sizes: {cluster_sizes}")
    print(f"  Largest cluster: {cluster_sizes[largest_cluster]} trials")
    print(f"  Selected trial: #{robust_trial.number} (score: {robust_trial.value:.4f})")
    print(f"  Best trial: #{study.best_trial.number} (score: {study.best_value:.4f})")

    return robust_trial.params


def optimize_layer(
    layer_name: str,
    n_trials: int,
    checkpoint: dict,
    dry_run: bool = False
) -> Optional[dict]:
    """Optimize a single layer."""

    layer_config = LAYER_CONFIGS.get(layer_name)
    if not layer_config:
        print(f"Error: Unknown layer: {layer_name}")
        return None

    frozen_params = get_frozen_params(checkpoint)

    # Check if previous layers are complete
    layer_idx = LAYER_ORDER.index(layer_name)
    for prev_layer in LAYER_ORDER[:layer_idx]:
        if checkpoint["layers"][prev_layer]["status"] != "complete":
            print(f"Warning: Previous layer '{prev_layer}' not complete. Results may be suboptimal.")

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
    save_checkpoint(checkpoint)

    if dry_run:
        print("DRY RUN - would optimize with these settings")
        return None

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
    objective = create_objective(layer_config, frozen_params, checkpoint)

    # Checkpoint callback
    def checkpoint_callback(study, trial):
        checkpoint["layers"][layer_name]["trials_run"] = len(study.trials)
        if study.best_value is not None:
            checkpoint["layers"][layer_name]["best_score"] = study.best_value
        save_checkpoint(checkpoint)

    # Run optimization
    remaining_trials = n_trials - len(study.trials)
    if remaining_trials > 0:
        print(f"\nRunning {remaining_trials} trials...")
        try:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                show_progress_bar=True,
                callbacks=[checkpoint_callback]
            )
        except RuntimeError as e:
            print(f"\nOptimization aborted: {e}")
            return None

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

    # Add to history
    checkpoint["history"].append({
        "layer": layer_name,
        "completed": datetime.now().isoformat(),
        "trials": len(study.trials),
        "score": best_trial.value,
        "params": robust_params
    })

    save_checkpoint(checkpoint)

    print(f"\n{'='*60}")
    print(f"Layer {layer_name} COMPLETE")
    print(f"{'='*60}")
    print(f"Optimal params: {robust_params}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    print(f"Score: {best_trial.value:.4f}")

    return robust_params


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(checkpoint: dict):
    """Generate optimization report."""

    report = []
    report.append("# Orchestrator Optimization Report")
    report.append("")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("")

    meta = checkpoint.get("meta", {})
    report.append("## Summary")
    report.append("")
    report.append(f"- Started: {meta.get('started', 'N/A')}")
    report.append(f"- Completed: {meta.get('last_updated', 'N/A')}")
    report.append("")

    # Layer results
    report.append("## Layer Results")
    report.append("")
    report.append("| Layer | Status | Trials | Score | Parameters |")
    report.append("|-------|--------|--------|-------|------------|")

    for layer_name in LAYER_ORDER:
        layer = checkpoint["layers"].get(layer_name, {})
        status = layer.get("status", "pending")
        trials = layer.get("trials_run", 0)
        score = layer.get("best_score")
        params = layer.get("optimal_params", {})

        score_str = f"{score:.4f}" if score else "-"
        params_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in params.items()) if params else "-"

        report.append(f"| {layer_name} | {status} | {trials} | {score_str} | {params_str} |")

    report.append("")

    # Optimal configuration
    report.append("## Optimal Configuration")
    report.append("")
    report.append("```yaml")
    report.append("optimized_params:")
    for layer_name in LAYER_ORDER:
        layer = checkpoint["layers"].get(layer_name, {})
        params = layer.get("optimal_params", {})
        if params:
            report.append(f"  {layer_name}:")
            for k, v in params.items():
                if isinstance(v, float):
                    report.append(f"    {k}: {v:.4f}")
                else:
                    report.append(f"    {k}: {v}")
    report.append("```")
    report.append("")

    # Metrics
    report.append("## Metrics by Layer")
    report.append("")
    for layer_name in LAYER_ORDER:
        layer = checkpoint["layers"].get(layer_name, {})
        metrics = layer.get("metrics", {})
        if metrics:
            report.append(f"### {layer_name}")
            report.append("")
            for k, v in metrics.items():
                if isinstance(v, float):
                    report.append(f"- {k}: {v:.4f}")
                else:
                    report.append(f"- {k}: {v}")
            report.append("")

    # History
    if checkpoint.get("history"):
        report.append("## Optimization History")
        report.append("")
        for entry in checkpoint["history"]:
            report.append(f"- **{entry['layer']}** completed {entry['completed']}")
            report.append(f"  - Trials: {entry['trials']}, Score: {entry['score']:.4f}")
        report.append("")

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report))
    print(f"Report saved to: {REPORT_PATH}")


def update_registry(checkpoint: dict):
    """Update model_registry.yaml with optimal parameters."""

    if not REGISTRY_PATH.exists():
        print(f"Warning: Registry not found: {REGISTRY_PATH}")
        return

    with open(REGISTRY_PATH) as f:
        registry = yaml.safe_load(f)

    # Add optimized_params section
    registry["optimized_params"] = {}
    for layer_name in LAYER_ORDER:
        layer = checkpoint["layers"].get(layer_name, {})
        params = layer.get("optimal_params")
        if params:
            registry["optimized_params"][layer_name] = params

    registry["optimized_params"]["_meta"] = {
        "optimized": checkpoint["meta"].get("last_updated"),
        "schema_version": 1
    }

    with open(REGISTRY_PATH, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

    print(f"Registry updated: {REGISTRY_PATH}")


# =============================================================================
# Main Entry Points
# =============================================================================

def run_preflight(mock_mode: bool = False) -> bool:
    """Run preflight checks."""
    print("\n=== Preflight Checks ===\n")

    if mock_mode:
        print("[MOCK MODE] Skipping model and server checks\n")

    checks_passed = True

    # Check model exists (skip in mock mode)
    if not mock_mode:
        if DEV_MODEL.exists():
            print(f"[OK] Dev model exists: {DEV_MODEL.name}")
        else:
            print(f"[FAIL] Dev model not found: {DEV_MODEL}")
            checks_passed = False

        # Check llama-server
        server_bin = LLAMA_CPP_PATH / "llama-server"
        if server_bin.exists():
            print(f"[OK] llama-server exists")
        else:
            print(f"[FAIL] llama-server not found: {server_bin}")
            checks_passed = False

    # Check test suite exists
    prompt_dir = PROJECT_ROOT / "benchmarks" / "prompts" / "v1" / "orchestrator"
    if prompt_dir.exists():
        suites = list(prompt_dir.glob("t*"))
        print(f"[OK] Test suites found: {len(suites)}")
    else:
        print(f"[FAIL] Test suites not found: {prompt_dir}")
        checks_passed = False

    # Check ports
    for port, name in [(FRONTDOOR_PORT, "frontdoor"), (WORKER_PORT, "worker"),
                        (API_PORT, "api"), (DASHBOARD_PORT, "dashboard")]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        if result == 0:
            print(f"[WARN] Port {port} ({name}) in use - will kill existing process")
        else:
            print(f"[OK] Port {port} ({name}) available")

    print()
    return checks_passed


def show_status():
    """Show current optimization status."""
    checkpoint = load_checkpoint()

    print(f"\n{'='*60}")
    print("Optimization Status")
    print(f"{'='*60}")

    meta = checkpoint.get("meta", {})
    print(f"\nStarted: {meta.get('started', 'Not started')}")
    print(f"Last updated: {meta.get('last_updated', 'N/A')}")

    print(f"\n{'Layer':<15} {'Status':<12} {'Trials':<8} {'Score':<10}")
    print("-" * 50)

    for layer_name in LAYER_ORDER:
        layer = checkpoint.get("layers", {}).get(layer_name, {})
        status = layer.get("status", "pending")
        trials = layer.get("trials_run", 0)
        score = layer.get("best_score")
        score_str = f"{score:.4f}" if score else "-"

        print(f"{layer_name:<15} {status:<12} {trials:<8} {score_str:<10}")

    next_layer = get_next_layer(checkpoint)
    if next_layer:
        print(f"\nNext layer to optimize: {next_layer}")
    else:
        print("\nAll layers complete!")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Optuna Optimization for Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single layer optimization
  %(prog)s --layer frontdoor --trials 30

  # All layers sequentially (overnight run)
  %(prog)s --all-layers

  # Resume from checkpoint
  %(prog)s --resume

  # Show status
  %(prog)s --status
        """
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--layer",
        choices=LAYER_ORDER,
        help="Single layer to optimize"
    )
    group.add_argument(
        "--all-layers",
        action="store_true",
        help="Optimize all layers sequentially"
    )
    group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Show current optimization status"
    )

    parser.add_argument(
        "--trials",
        type=int,
        help="Number of trials (overrides default per layer)"
    )
    parser.add_argument(
        "--trials-per-layer",
        type=int,
        help="Number of trials per layer (for --all-layers)"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start Optuna dashboard for monitoring"
    )
    parser.add_argument(
        "--no-servers",
        action="store_true",
        help="Don't start llama-servers (assume already running)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without running"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock mode (no real inference, for testing the optimization flow)"
    )

    args = parser.parse_args()

    # Set global mock mode
    global USE_MOCK_MODE
    if args.mock:
        USE_MOCK_MODE = True
        args.no_servers = True  # Mock mode doesn't need llama-servers

    # Handle status
    if args.status:
        show_status()
        return

    # Validate arguments
    if not args.layer and not args.all_layers and not args.resume:
        parser.print_help()
        print("\nError: Specify --layer, --all-layers, or --resume")
        sys.exit(1)

    # Load checkpoint
    checkpoint = load_checkpoint()

    # Determine layers to optimize
    if args.layer:
        layers_to_optimize = [args.layer]
    elif args.all_layers:
        layers_to_optimize = LAYER_ORDER
    elif args.resume:
        next_layer = get_next_layer(checkpoint)
        if next_layer:
            layers_to_optimize = LAYER_ORDER[LAYER_ORDER.index(next_layer):]
        else:
            print("All layers already complete!")
            return

    # Run preflight
    if not run_preflight(mock_mode=USE_MOCK_MODE):
        print("Preflight checks failed. Fix issues and retry.")
        sys.exit(1)

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Mock mode: {USE_MOCK_MODE}")
        print(f"Would optimize layers: {layers_to_optimize}")
        for layer in layers_to_optimize:
            n_trials = args.trials or args.trials_per_layer or LAYER_CONFIGS[layer].default_trials
            print(f"  {layer}: {n_trials} trials")
        return

    # Initialize process manager
    pm = ProcessManager()

    try:
        # Start servers
        if not args.no_servers:
            print("\n=== Starting Servers ===\n")
            if not start_llama_servers(pm):
                print("Failed to start llama-servers")
                sys.exit(1)

            if not start_orchestrator_api(pm):
                print("Failed to start orchestrator API")
                sys.exit(1)

        # Start dashboard
        if args.dashboard or args.all_layers:
            print("\n=== Starting Dashboard ===\n")
            start_dashboard(pm)

        # Optimize layers
        print("\n=== Starting Optimization ===\n")

        for layer_name in layers_to_optimize:
            n_trials = args.trials or args.trials_per_layer or LAYER_CONFIGS[layer_name].default_trials

            result = optimize_layer(layer_name, n_trials, checkpoint)

            if result is None:
                print(f"\nOptimization of {layer_name} failed or aborted")
                break

            # Reload checkpoint for next layer
            checkpoint = load_checkpoint()

        # Generate report
        print("\n=== Generating Report ===\n")
        generate_report(checkpoint)
        update_registry(checkpoint)

        print("\n=== Optimization Complete ===\n")
        show_status()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        # Cleanup
        print("\n=== Cleaning Up ===\n")
        stop_all_servers(pm)


if __name__ == "__main__":
    main()
