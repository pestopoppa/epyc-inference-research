"""Species 1 — NumericSwarm: Optuna multi-objective optimization with hot-swap.

Defines parameter surfaces from config/models.py, uses NSGA-II for
multi-objective search, and cluster-based robust selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

log = logging.getLogger("autopilot.numeric_swarm")

ORCH_ROOT = Path(__file__).resolve().parents[3]
OPTUNA_DB = ORCH_ROOT / "orchestration" / "optuna_study.db"


@dataclass
class ParamSpec:
    name: str
    low: float
    high: float
    param_type: str = "float"  # "float" or "int"
    log_scale: bool = False


# ── Parameter surfaces ───────────────────────────────────────────

SURFACES: dict[str, list[ParamSpec]] = {
    "memrl_retrieval": [
        ParamSpec("memrl_retrieval.q_weight", 0.3, 0.95),
        ParamSpec("memrl_retrieval.min_similarity", 0.1, 0.6),
        ParamSpec("memrl_retrieval.min_q_value", 0.1, 0.6),
        ParamSpec("memrl_retrieval.confidence_threshold", 0.3, 0.9),
        ParamSpec("memrl_retrieval.semantic_k", 5, 50, "int"),
        ParamSpec("memrl_retrieval.prior_strength", 0.05, 0.5),
    ],
    "think_harder": [
        ParamSpec("think_harder.min_expected_roi", 0.005, 0.1),
        ParamSpec("think_harder.token_budget_min", 512, 4096, "int"),
        ParamSpec("think_harder.token_budget_max", 2048, 8192, "int"),
        ParamSpec("think_harder.cot_roi_threshold", 0.1, 0.7),
    ],
    "chat_pipeline": [
        ParamSpec("chat.try_cheap_first_q_threshold", 0.3, 0.9),
    ],
    "chat_long_context": [
        ParamSpec("chat.long_context_threshold_chars", 12000, 80000, "int"),
    ],
    "chat_summarization": [
        ParamSpec("chat.summarization_threshold_tokens", 3000, 30000, "int"),
    ],
    "chat_review_skip": [
        ParamSpec("chat.review_skip_q_threshold", 0.45, 0.80),
    ],
    "monitor": [
        ParamSpec("monitor.entropy_threshold", 2.0, 6.0),
        ParamSpec("monitor.repetition_threshold", 0.1, 0.6),
        ParamSpec("monitor.entropy_spike_threshold", 1.0, 4.0),
    ],
    "escalation": [
        ParamSpec("escalation.max_retries", 1, 5, "int"),
        ParamSpec("escalation.max_escalations", 1, 5, "int"),
    ],
    "repl_executor": [
        # Sweep the REPL per-turn token cap (Fast-RLM truncation eval, originally
        # deferred in 01-fast-rlm-budget-controls.md §4). Range covers our default
        # (768) and reaches up to 4096 to test whether more headroom helps complex
        # REPL turns, and down to 256 to test the fast-rlm "aggressive truncation"
        # hypothesis. Note: when difficulty_signal mode == "enforce" the band-
        # adaptive budgets dominate; this knob sweeps the flat-cap path that runs
        # outside enforce-mode.
        ParamSpec("repl.turn_token_cap", 256, 4096, "int"),
        # Sibling: frontdoor REPL turn cap when tool_required=False. Default 768
        # (env: ORCHESTRATOR_FRONTDOOR_REPL_NON_TOOL_N_TOKENS). Same band-adaptive
        # bypass caveat applies.
        ParamSpec("repl.frontdoor_non_tool_token_cap", 256, 4096, "int"),
    ],
    "repl_budget": [
        # Fast-RLM REPL execution count cap. Default 30. Feature-flag-gated by
        # `worker_call_budget`; sweep is meaningful only when that flag is on.
        ParamSpec("repl.worker_call_budget_cap", 5, 100, "int"),
        # Fast-RLM aggregate completion-token cap per task. Default 200,000.
        # Feature-flag-gated by `task_token_budget`. Range probes both
        # aggressive (50K — forces tight reasoning) and conservative (500K).
        ParamSpec("repl.task_token_budget_cap", 50000, 500000, "int"),
    ],
    "kv_compaction": [
        # Expected-Attention KV cache compression knobs (autopilot program.md
        # Tier 4.5). Applied via kv_compress.compress_slot() — NOT env-restart.
        # keep_ratio: fraction of KV entries to KEEP per compaction. Safe range
        # [0.50, 0.90]; below 0.25 format degrades. Lower bound capped at 0.25
        # so autopilot does not sample the catastrophic cliff zone.
        ParamSpec("kv.keep_ratio", 0.25, 0.90, "float"),
        # Sink tokens never evicted. Default 4. Protects system prompt.
        ParamSpec("kv.keep_first", 2, 16, "int"),
        # Future positions for RoPE averaging in EA scoring. Default 128.
        # Higher = more stable scores, slower.
        ParamSpec("kv.n_future", 64, 1024, "int"),
    ],
    "gpu_placement_policy": [
        # AXA-3 class-3 GPU placement/teleport policy knobs. These remain
        # default-off policy parameters: NumericSwarm can tune the thresholds and
        # lease weights, but it cannot flip placement_policy.teleport_enabled.
        ParamSpec("placement_policy.long_running_trigger_tokens", 0, 4096, "int"),
        ParamSpec("placement_policy.rate_window_tokens", 16, 1024, "int"),
        ParamSpec("placement_policy.min_resident_remaining_tokens", 64, 1024, "int"),
        ParamSpec("placement_policy.min_speedup", 1.0, 4.0, "float"),
        ParamSpec("placement_policy.lease_interactive_weight", 0.0, 4.0, "float"),
        ParamSpec("placement_policy.lease_batch_weight", 0.0, 4.0, "float"),
        ParamSpec("placement_policy.lease_eval_weight", 0.0, 4.0, "float"),
    ],
    "spec_decode_role_restart": [
        # AP-3 class-2 launch knobs. These are expensive role-restart params:
        # apply_params classifies them as role_restart and live application is
        # gated separately. Categorical spec-type and KV-profile choices are not
        # NumericSwarm floats; they stay explicit quality-gated action params.
        ParamSpec("role_restart.frontdoor_draft_max", 1, 8, "int"),
        ParamSpec("role_restart.frontdoor_draft_min", 0, 8, "int"),
        ParamSpec("role_restart.frontdoor_draft_p_min", 0.0, 0.5, "float"),
        ParamSpec("role_restart.frontdoor_draft_p_split", 0.0, 1.0, "float"),
        ParamSpec("role_restart.frontdoor_ngram_mod_n_min", 0, 1024, "int"),
        ParamSpec("role_restart.frontdoor_ngram_mod_n_max", 0, 1024, "int"),
        ParamSpec("role_restart.frontdoor_ngram_mod_n_match", 1, 1024, "int"),
        ParamSpec("role_restart.worker_draft_max", 1, 8, "int"),
        ParamSpec("role_restart.worker_draft_min", 0, 8, "int"),
        ParamSpec("role_restart.worker_draft_p_min", 0.0, 0.5, "float"),
        ParamSpec("role_restart.worker_draft_p_split", 0.0, 1.0, "float"),
        ParamSpec("role_restart.worker_threads_draft", 1, 96, "int"),
        ParamSpec("role_restart.worker_ngram_mod_n_min", 0, 1024, "int"),
        ParamSpec("role_restart.worker_ngram_mod_n_max", 0, 1024, "int"),
        ParamSpec("role_restart.worker_ngram_mod_n_match", 1, 1024, "int"),
        ParamSpec("role_restart.architect_draft_max", 1, 8, "int"),
        ParamSpec("role_restart.architect_draft_min", 0, 8, "int"),
        ParamSpec("role_restart.architect_draft_p_min", 0.0, 0.5, "float"),
        ParamSpec("role_restart.architect_draft_p_split", 0.0, 1.0, "float"),
        ParamSpec("role_restart.architect_ngram_mod_n_min", 0, 1024, "int"),
        ParamSpec("role_restart.architect_ngram_mod_n_max", 0, 1024, "int"),
        ParamSpec("role_restart.architect_ngram_mod_n_match", 1, 1024, "int"),
    ],
}


class NumericSwarm:
    """Species 1: Optuna-based multi-objective parameter optimization."""

    def __init__(
        self,
        db_path: Path | None = None,
        seed: int = 42,
        epoch_label: str | None = None,
    ):
        self.db_path = db_path or OPTUNA_DB
        self.seed = seed
        self.epoch_label = _study_name_token(epoch_label)
        self._studies: dict[str, Any] = {}
        self._import_optuna()

    def _import_optuna(self) -> None:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self._optuna = optuna
        except ImportError:
            log.error("Optuna not installed. Run: pip install optuna")
            raise

    def _get_study(self, surface: str) -> Any:
        """Get or create an Optuna study for a surface."""
        if surface in self._studies:
            return self._studies[surface]

        optuna = self._optuna
        storage = f"sqlite:///{self.db_path}"
        study_name = self._study_name(surface)

        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage,
            )
            log.info("Loaded existing study '%s' with %d trials", study_name, len(study.trials))
        except KeyError:
            sampler = optuna.samplers.NSGAIISampler(seed=self.seed)
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler,
                directions=["maximize", "maximize", "maximize", "maximize"],
                # quality, speed, -cost, reliability
            )
            log.info("Created new study '%s'", study_name)

        self._studies[surface] = study
        return study

    def suggest_trial(self, surface: str) -> dict[str, Any]:
        """Suggest a new trial for a parameter surface.

        Returns dict of param_name → value.
        """
        if surface not in SURFACES:
            raise ValueError(f"Unknown surface: {surface}. Available: {list(SURFACES.keys())}")

        study = self._get_study(surface)
        trial = study.ask()

        params = {}
        for spec in SURFACES[surface]:
            if spec.param_type == "int":
                val = trial.suggest_int(spec.name, int(spec.low), int(spec.high))
            elif spec.log_scale:
                val = trial.suggest_float(spec.name, spec.low, spec.high, log=True)
            else:
                val = trial.suggest_float(spec.name, spec.low, spec.high)
            params[spec.name] = val

        log.info("Trial %d for surface '%s': %s", trial.number, surface, params)
        return {"trial_number": trial.number, "params": params, "surface": surface}

    def report_result(
        self,
        surface: str,
        trial_number: int,
        objectives: tuple[float, float, float, float],
    ) -> None:
        """Report trial result (quality, speed, -cost, reliability)."""
        study = self._get_study(surface)
        study.tell(trial_number, list(objectives))
        log.info(
            "Reported trial %d on '%s': %s",
            trial_number, surface, objectives,
        )

    def mark_failed(self, surface: str, trial_number: int, reason: str = "") -> None:
        """Mark an Optuna trial failed when params could not be applied."""
        from optuna.trial import TrialState

        study = self._get_study(surface)
        try:
            study.tell(trial_number, state=TrialState.FAIL)
            log.warning(
                "Marked trial %d on '%s' failed: %s",
                trial_number, surface, reason,
            )
        except ValueError as e:
            log.warning(
                "Could not mark trial %d on '%s' failed: %s",
                trial_number, surface, e,
            )

    def best_params(self, surface: str, method: str = "cluster") -> dict[str, Any]:
        """Get best parameters using cluster-based robust selection.

        method: "cluster" for robust centroid, "best_quality" for highest quality.
        """
        study = self._get_study(surface)

        if len(study.trials) < 3:
            log.warning("Not enough trials for robust selection (have %d)", len(study.trials))
            return {}

        if method == "cluster":
            return self._cluster_select(study, surface)
        elif method == "best_quality":
            # Best trial by first objective (quality)
            best = max(study.best_trials, key=lambda t: t.values[0])
            return dict(best.params)
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def _cluster_select(
        self, study: Any, surface: str, top_percent: float = 0.2
    ) -> dict[str, Any]:
        """Cluster-based robust selection (reuses optuna_orchestrator pattern).

        Take top 20% trials, cluster, return centroid of largest cluster.
        """
        import numpy as np

        trials = [t for t in study.trials if t.state.name == "COMPLETE"]
        if len(trials) < 5:
            # Not enough for clustering, return best
            best = max(trials, key=lambda t: t.values[0])
            return dict(best.params)

        # Sort by quality (first objective) and take top fraction
        trials.sort(key=lambda t: t.values[0], reverse=True)
        n_top = max(3, int(len(trials) * top_percent))
        top_trials = trials[:n_top]

        # Build parameter matrix
        param_names = [spec.name for spec in SURFACES[surface]]
        matrix = np.array([
            [t.params.get(name, 0.0) for name in param_names]
            for t in top_trials
        ])

        # Normalize columns
        col_range = matrix.max(axis=0) - matrix.min(axis=0)
        col_range[col_range == 0] = 1.0
        normalized = (matrix - matrix.min(axis=0)) / col_range

        # K-means with k=3
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=min(3, len(top_trials)), random_state=self.seed, n_init=10)
            labels = km.fit_predict(normalized)
        except ImportError:
            # Fallback: return centroid of all top trials
            centroid = matrix.mean(axis=0)
            # Find closest trial to centroid
            dists = np.linalg.norm(matrix - centroid, axis=1)
            best_idx = np.argmin(dists)
            return dict(top_trials[best_idx].params)

        # Find largest cluster
        from collections import Counter
        counts = Counter(labels)
        largest_label = counts.most_common(1)[0][0]
        cluster_indices = [i for i, label in enumerate(labels) if label == largest_label]

        # Centroid of largest cluster (in original space)
        cluster_matrix = matrix[cluster_indices]
        centroid = cluster_matrix.mean(axis=0)

        # Find closest trial to centroid
        dists = np.linalg.norm(cluster_matrix - centroid, axis=1)
        best_local_idx = np.argmin(dists)
        best_idx = cluster_indices[best_local_idx]

        return dict(top_trials[best_idx].params)

    def importance(self, surface: str) -> dict[str, float]:
        """Parameter importance via fANOVA (if available)."""
        try:
            from optuna.importance import FanovaImportanceEvaluator
            study = self._get_study(surface)
            if len(study.trials) < 10:
                return {}
            evaluator = FanovaImportanceEvaluator(seed=self.seed)
            # Importance for first objective (quality)
            return self._optuna.importance.get_param_importances(
                study, evaluator=evaluator, target=lambda t: t.values[0]
            )
        except Exception as e:
            log.debug("fANOVA importance not available: %s", e)
            return {}

    def mark_epoch(self, reason: str) -> None:
        """Invalidate existing studies by starting a new epoch.

        After regime changes (StructuralLab flag changes, PromptForge prompt
        changes), old Optuna trials reflect a different optimization landscape.
        This method increments an epoch counter and creates fresh studies,
        so new trials start from the new regime's baseline.

        The old studies remain in the SQLite DB for historical reference
        but are no longer used for suggestions.
        """
        # Determine next epoch number by checking existing study names
        optuna = self._optuna
        storage = f"sqlite:///{self.db_path}"
        try:
            existing = optuna.study.get_all_study_summaries(storage=storage)
            epoch_numbers = []
            for s in existing:
                parts = s.study_name.split("_epoch")
                if len(parts) == 2:
                    try:
                        epoch_numbers.append(int(parts[1]))
                    except ValueError:
                        pass
            next_epoch = max(epoch_numbers, default=0) + 1
        except Exception:
            next_epoch = 1

        log.info(
            "Marking epoch %d (reason: %s). Old studies invalidated.",
            next_epoch, reason,
        )

        # Clear cached studies — they'll be recreated with new epoch suffix
        self._studies.clear()

        # Override study naming to include epoch
        self._epoch = next_epoch

    def _study_name(self, surface: str) -> str:
        """Get study name with optional epoch suffix."""
        base = f"autopilot_{surface}"
        if self.epoch_label:
            base = f"{base}_era_{self.epoch_label}"
        epoch = getattr(self, "_epoch", 0)
        if epoch > 0:
            return f"{base}_epoch{epoch}"
        return base

    def summary(self) -> dict[str, Any]:
        """Summary for controller consumption."""
        result = {}
        for surface in SURFACES:
            try:
                study = self._get_study(surface)
                trials = [t for t in study.trials if t.state.name == "COMPLETE"]
                result[surface] = {
                    "n_trials": len(trials),
                    "best_quality": max((t.values[0] for t in trials), default=0),
                }
            except Exception:
                result[surface] = {"n_trials": 0}
        return result


def _study_name_token(value: str | None) -> str:
    """Return a stable Optuna study-name-safe token for an instrument era."""
    if not value:
        return ""
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return token[:80]
