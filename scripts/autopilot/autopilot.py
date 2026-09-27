#!/usr/bin/env python3
"""AutoPilot: Continuous recursive optimization for the EPYC orchestration stack.

Main controller loop: observe → reason → act → evaluate → record → meta-learn.

Usage:
    python autopilot.py start [--dry-run] [--max-trials N]
    python autopilot.py status
    python autopilot.py pause
    python autopilot.py resume
    python autopilot.py report
    python autopilot.py plot
    python autopilot.py checkpoint [--production-best]
    python autopilot.py restore [--checkpoint PATH]
    python autopilot.py digest [--no-state-update]
    python autopilot.py peaf [--min-n N]   # PEAF cheap-kill report (intake-571 spike)

Environment flags:
    EPYC_AUTOPILOT_PEAF=0   Disable PEAF (Prediction-Error-As-Feature). Default ON: the
                            controller is asked to forecast each trial's objectives and
                            surprise is logged alongside actuals (logging-only, never
                            feeds back into scoring). Disable for baseline A/B or if
                            controller drift is suspected. See `python autopilot.py peaf`.
"""

from __future__ import annotations

import argparse
import atexit
from collections import deque
import copy
import dataclasses
from dataclasses import asdict
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TYPE_CHECKING

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

import yaml

from src.registry.stack_priors import live_stack_slot_query_ports
from src.registry.capability_registry import (
    build_action_availability_section,
    load_capability_registry,
)
from src.autopilot_core.journal_reconstruction import (
    objectives_from_journal_row,
    parse_journal_ts,
    reconstruct_archive_from_journal_rows,
)
from src.autopilot_core.journal_snapshot_replay import archive_payload_from_verified_snapshot
from src.autopilot_core.infra_fingerprint import (
    NON_COMPARABLE as INFRA_NON_COMPARABLE,
    collect_infra_fingerprint,
    compare_infra_fingerprints,
    fingerprint_digest,
    regime_digest,
)
from src.autopilot_core import ap55_promotion_gate
from src.autopilot_core.rlvr_tiers import rlvr_reward_from_result
from src.autopilot_core.live_reproductions import live_reproductions
from src.autopilot_core.measurement_guards import (
    is_quality_admissible as _is_quality_admissible,
)
from experiment_journal import (
    ExperimentJournal,
    JournalEntry,
    build_baseline_pin,
    scrub_legacy_scale_text,
    select_lineage_parent,
)
from planner_roster import PlannerRosterError, validate_active_environment
from pareto_archive import (
    ParetoArchive,
    ParetoArchive as _ConcreteParetoArchive,
    ParetoEntry,
    pareto_archive_from_journal_rows,
)
from safety_gate import Baseline, DEFAULT_BASELINE_PATH, EvalResult, SafetyGate, _atomic_write_text
from safety_gate import PromotionGuardView, configure_promotion_guard_archive
from eval_tower import EVAL_EXECUTION_INSTRUMENT_ID, EvalTower
from config_applicator import apply_params  # noqa: F401 - re-export for actions.py tests
from config_applicator import health_check
from meta_optimizer import MetaOptimizer, SpeciesBudget
from progress_plots import PLOTS_DIR, generate_all_plots
import peaf
from species import Seeder, NumericSwarm, PromptForge, StructuralLab, EvolutionManager
from species.prompt_forge import CODE_MUTATION_ALLOWLIST, new_file_mutation_root_labels
import species.prompt_forge as _prompt_forge_module
from eval_leakage_monitor import EvalLeakageMonitor
from digest import generate_digest, should_generate_today
from short_term_memory import ShortTermMemory
from self_criticism import SelfCriticism, generate_self_criticism
from phase_status import (
    AsyncTaskRunner,
    HealthEscalationTracker,
    PhaseTracker,
    DEFAULT_HEALTH_ESCALATE_AFTER,
    DEFAULT_JOURNAL_DIR as PHASE_DEFAULT_JOURNAL_DIR,
    DEFAULT_OUTCOME_RECENT_WINDOW_TRIALS,
    DEFAULT_OUTCOME_STALL_FRONTIER_TRIALS,
    DEFAULT_OUTCOME_STALL_PROMOTION_TRIALS,
    _build_outcome_progress_report as _phase_outcome_progress_report,
)

# 2026-05-22 Tranche-5 refactor — extracted modules. Public names re-imported below.
import controller_io
from controller_io import PLANNER_ARCHIVE_PATH, invoke_controller as _invoke_controller_impl
from run_manifest import build_run_manifest, manifest_drift_reasons
import rejected_mutation_ledger
import reproposal_rate
from planner_coordinator import plan_with_providers, uncritiqued_dispatch_block_reason
from state_store import (
    OBSERVATIONAL_ACTION_BLACKLIST_DENYLIST,
    _auto_blacklist_reason_class,
    _blacklist_expires_at,
    _entry_is_non_expiring_blacklist,
    append_blacklist as _append_blacklist_impl,
    check_blacklist,
    format_model_signatures,
    load_blacklist as _load_blacklist_impl,
    ModelSignaturesUnavailableError,
    load_model_signatures as _load_model_signatures_impl,
    load_state as _load_state_impl,
    save_state as _save_state_impl,
)
from blacklist_purge_plan import purge_scoped_target, retryable_reexploration_target
from operator_hypotheses import (
    build_planner_block as _build_operator_hypotheses_block,
    record_resolution_from_rationale as _record_operator_hypothesis_resolution,
)
from vidya_planner_bridge import build_settled_ground_block as _build_vidya_settled_ground_block
from vidya_planner_bridge import build_kvq_context as _build_vidya_kvq_context
from state_lock import (
    OPERATOR_PAUSE_OWNER,
    PAUSE_COLLISION_FIELD,
    PAUSE_LEASE_FIELDS,
    PAUSE_OWNER_FIELD,
    clear_pause_lease,
    state_write_lock,
    supersede_pause_lease,
)
from state_ownership import clear_halt_latch, halt_latch_message
from actions import dispatch_action, SkipOutcome, _structural_noop_reason
from actions import SERVED_CONTENT_STATE_KEY, _public_pending_copy, served_content_record
from paired_stats import QuestionOutcome, mcnemar_from_vectors, verdict_from_result
from src.autopilot_core.action_identity import (
    EPHEMERAL_ACTION_KEYS,
    action_signature,
    CONTENT_IDENTIFIED_ACTION_TYPES,
    INFRA_REGIME_DIGEST_KEY,
    SERVED_CONTENT_KEY,
    action_config_identity,
    canonical_action,
    config_fingerprint,
    row_config_identity,
    served_content_files,
)
from src.autopilot_core.learning_exclusions import (
    FRONTIER_ADMISSION_KEY,
    FRONTIER_ADMISSION_REPRESENTATIVE,
    BENIGN_LEARNING_EXCLUSIONS,
    NON_CORRUPT_LEARNING_EXCLUSIONS,
    classify_learning_exclusion,
    row_is_representative_member,
)
from src.autopilot_core.multitier_decision import (
    DEFAULT_MAX_ATTEMPTS_PER_TIER,
    MULTITIER_POLICY_VERSION,
    REQUIRED_VALIDATION_TIERS,
    build_tier_candidate_evidence,
    evaluate_tier_validation,
)
from src.autopilot_core.planner_evidence import (
    DEFAULT_EVIDENCE_CORE_ID,
    format_planner_evidence_section,
)
from src.autopilot_core.sequential_verdict import (
    DEFAULT_POLICY as SEQ_DEFAULT_POLICY,
    baseline_profile_from_trials,
    rate_noninferiority_z,
)
from src.autopilot_core.authority_consent import (
    SEQ_P0_2_BRIDGE_MODE,
    seq_p0_2_bridge_status,
)
from src.autopilot_core.tier_specs import (
    DEFAULT_FRONTIER_TIER,
    LEGACY_OBJECTIVE_POLICY,
    MIN_FRONTIER_EVAL_TIER,
    RATE_4D_OBJECTIVE_POLICY,
    UnmeasuredObjectiveError,
    goodput_qph_from,
    legacy_objectives_from,
    objectives_from,
    objectives_measurable,
    seq_task_rate_qph_from,
    seq_task_rate_qph_from_row,
    spec_for,
    task_rate_objectives_from,
    task_rate_qph_from,
)
from src.autopilot_core.baseline_ledger import (
    apply_baseline_ledger_authority,
    format_baseline_ledger_summary,
    reconcile_baseline_ledger,
)
from src.autopilot_core.instrument_era_guard import (
    E7_EVAL_INSTRUMENT_BOUNDARY,
    E7_EVAL_INSTRUMENT_ERA_ID,
    last_known_eval_quality_era,
    EVAL_QUALITY_SCOPE,
    active_eval_quality_era,
)

# Preflight diagnostics from seeding infra
sys.path.insert(0, str(SCRIPT_DIR.parent / "benchmark"))
try:
    from seeding_infra import get_preflight_diagnostics
except ImportError:
    get_preflight_diagnostics = None  # type: ignore[assignment]

# Strategy store for species memory (B1)
from orchestration.repl_memory.strategy_store import StrategyStore

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Durable earlyoom control-plane protection (mirrors orchestrator_stack). Guarded so a
# resolution hiccup never blocks autopilot import/startup — it is strictly best-effort.
try:
    from scripts.server.stack_processes import set_oom_score_adj as _set_oom_score_adj
except Exception:  # pragma: no cover - import-path fallback

    def _set_oom_score_adj(pids: Any, adj: int = -1000) -> int:
        return 0


log = logging.getLogger("autopilot")

ARCHIVE_SOURCE_STATE = "state"
ARCHIVE_SOURCE_JOURNAL_CURRENT_RUN = "journal-current-run"
ARCHIVE_SOURCE_JOURNAL_ALL = "journal-all"
ARCHIVE_SOURCE_CHOICES = (
    ARCHIVE_SOURCE_STATE,
    ARCHIVE_SOURCE_JOURNAL_CURRENT_RUN,
    ARCHIVE_SOURCE_JOURNAL_ALL,
)

if TYPE_CHECKING:
    from autopilot_tui import AutoPilotTUI

_EPHEMERAL_ACTION_KEYS = EPHEMERAL_ACTION_KEYS
_action_signature = action_signature
_canonical_action = canonical_action
_config_fingerprint = config_fingerprint

STATE_PATH = ORCH_ROOT / "orchestration" / "autopilot_state.json"
LOCK_PATH = ORCH_ROOT / "orchestration" / ".autopilot.lock"
BLACKLIST_PATH = SCRIPT_DIR / "failure_blacklist.yaml"
OPERATOR_OUTBOX_PATH = ORCH_ROOT / "orchestration" / "autopilot_operator_outbox.jsonl"
EXIT_BREADCRUMB_PATH = ORCH_ROOT / "logs" / "autopilot_exit_breadcrumb.jsonl"
# Prompt-budget cap (2026-06-10): only the most-recent N blacklist entries are
# RENDERED into the planner prompt (the full list is always enforced at dispatch
# by check_blacklist()). Keeps the unbounded-growth blacklist from dominating the
# ~80KB prompt. Operator-tunable.
BLACKLIST_RENDER_CAP = int(os.environ.get("AUTOPILOT_BLACKLIST_RENDER_CAP", "18"))
MULTITIER_PROMOTION_ENABLED = os.environ.get(
    "AUTOPILOT_MULTITIER_PROMOTION", "0"
).strip().lower() in {"1", "true", "yes", "on"}
MULTITIER_MAX_ATTEMPTS_PER_TIER = int(
    os.environ.get(
        "AUTOPILOT_MULTITIER_MAX_ATTEMPTS_PER_TIER",
        str(DEFAULT_MAX_ATTEMPTS_PER_TIER),
    )
)
def _multitier_attempt_cap() -> int:
    """Per-tier attempt cap, never below the reproduction bar a final_t1 must clear.

    Review fix 5: final_t1 promotion needs >= N reproductions (N = the frontier rule's
    ``BASELINE_PROMOTION_REPRO_MIN`` or rule (b)'s ``AUTOPILOT_EMPTY_FRONTIER_MIN_REPRO``).
    Each final_t1 attempt adds one; if the source row does not count, the attempts alone
    must reach N. A lower configured cap would reject every candidate before it could
    reproduce enough — a silent freeze — so the cap is raised to N.
    """
    from safety_gate import BASELINE_PROMOTION_REPRO_MIN, empty_frontier_min_repro

    return max(
        1,
        MULTITIER_MAX_ATTEMPTS_PER_TIER,
        int(BASELINE_PROMOTION_REPRO_MIN),
        int(empty_frontier_min_repro()),
    )


MULTITIER_BASELINE_STATE_KEY = "multitier_baseline_bundle"
MULTITIER_PENDING_STATE_KEY = "multitier_pending_validation"


class ExitBreadcrumb:
    """Append durable process-lifecycle facts for post-mortem recovery.

    The regular AutoPilot log can be lost when a process is killed abruptly.
    This compact JSONL trail uses a single append+fsync write, so a received
    SIGINT/SIGTERM and all cooperative exits survive independently of the
    logging pipeline.  SIGKILL and OOM still cannot run Python cleanup; their
    diagnostic signal is the absence of a terminal breadcrumb.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or EXIT_BREADCRUMB_PATH
        self._context: dict[str, Any] = {}
        self._terminal_written = False

    def set_context(self, **context: Any) -> None:
        self._context.update({key: value for key, value in context.items() if value is not None})

    def write(self, reason: str, **details: Any) -> bool:
        """Best-effort append with fsync; never disrupt the optimizer itself."""
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "reason": reason,
            **self._context,
            **details,
        }
        encoded = (json.dumps(payload, sort_keys=True, default=str) + "\n").encode("utf-8")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, encoded)
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            return False
        return True

    def mark_terminal(self, reason: str, **details: Any) -> bool:
        """Write at most one cooperative terminal record per process."""
        if self._terminal_written:
            return True
        written = self.write(reason, terminal=True, **details)
        if written:
            self._terminal_written = True
        return written

    def register_atexit(self) -> None:
        """Classify any otherwise-unclassified interpreter teardown."""
        atexit.register(self.mark_terminal, "interpreter_exit")


OPERATOR_OUTBOX_RENDER_CAP = int(os.environ.get("AUTOPILOT_OPERATOR_OUTBOX_RENDER_CAP", "5"))
PRIOR_DECISION_DIGEST_CAP = int(os.environ.get("AUTOPILOT_PRIOR_DECISION_DIGEST_CAP", "4"))
PLANNER_JOURNAL_SUMMARY_LIMIT = int(os.environ.get("AUTOPILOT_PLANNER_JOURNAL_SUMMARY_LIMIT", "12"))
PLANNER_STRUCTURED_INSIGHTS_LIMIT = int(
    os.environ.get("AUTOPILOT_PLANNER_STRUCTURED_INSIGHTS_LIMIT", "18")
)
BSV3_CONFLICT_POLICY_ENV = "AUTOPILOT_BSV3_CONFLICT_POLICY"
ORCHESTRATOR_URL = "http://localhost:8000"
SEQ_BASELINE_PROFILE_LIMIT = int(os.environ.get("AUTOPILOT_SEQ_BASELINE_PROFILE_LIMIT", "120"))
# B4 / SEQ-1: minimum number of incumbent-representative trials the sequential null
# profile must be built from. Below this the profile is treated as UNAVAILABLE
# (empty baseline_profile) rather than run against a contaminated/thin mixture,
# which anti-conservatively depresses the null and accrues wealth too easily. 3
# mirrors the other reproduction/probe minimums (BASELINE_PROMOTION_REPRO_MIN,
# HIGHER_TIER_PROBE_MIN_TRIALS_PER_TIER, MAD_MIN_SAMPLES).
SEQ_BASELINE_PROFILE_MIN_TRIALS = int(
    os.environ.get("AUTOPILOT_SEQ_BASELINE_PROFILE_MIN_TRIALS", "3")
)
SEQ_PRIOR_OBS_LIMIT = int(os.environ.get("AUTOPILOT_SEQ_PRIOR_OBS_LIMIT", "120"))
# SEQ-B: minimum number of incumbent-representative trials that must carry a VALID paired
# rate before the rate axis has a comparator at all. Mirrors SEQ_BASELINE_PROFILE_MIN_TRIALS
# on the quality axis. Journal evidence: trial 836 ran the rate axis against a comparator
# built from a SINGLE prior row. Below this the rate comparator is UNAVAILABLE, which omits
# the axis (conservative: no rate evidence => no baseline ratchet), never substitutes a guess.
SEQ_BASELINE_RATE_MIN_TRIALS = int(os.environ.get("AUTOPILOT_SEQ_BASELINE_RATE_MIN_TRIALS", "3"))
SEQ_BASELINE_REFRESH_CADENCE = int(os.environ.get("AUTOPILOT_SEQ_BASELINE_REFRESH_CADENCE", "10"))
# AP-55 (b): a baseline-reference draw (the incumbent's seed re-run) measured in a
# different infra regime cannot anchor a comparison, so a regime change makes a
# re-run due. The minimum spacing keeps a churning fingerprint (e.g. a tracked file
# edited under a running daemon) from turning every other trial into a re-run.
AP55_REGIME_REFRESH_MIN_TRIALS = int(
    os.environ.get("AUTOPILOT_AP55_REGIME_REFRESH_MIN_TRIALS", "3")
)
# Forced re-runs that never land as a reference (invalid / skipped / blocked eval)
# are bounded per reference: spaced by AP55_REGIME_REFRESH_MIN_TRIALS trial ids and
# capped at this many attempts, after which forcing stops (logged) until a new
# reference draw lands.
AP55_SEED_RERUN_MAX_ATTEMPTS = int(os.environ.get("AUTOPILOT_AP55_SEED_RERUN_MAX_ATTEMPTS", "3"))
AP55_SEED_RERUN_ATTEMPTS_STATE_KEY = "ap55_seed_rerun_attempts"
SEQ_BASELINE_BLOCK_RETRY_CADENCE = int(
    os.environ.get("AUTOPILOT_SEQ_BASELINE_BLOCK_RETRY_CADENCE", "5")
)
SEQ_BASELINE_REFERENCE_STALE_AFTER_S = float(
    os.environ.get("AUTOPILOT_SEQ_BASELINE_REFERENCE_STALE_AFTER_S", str(48 * 3600))
)
SEQ_PROMOTION_FINAL_CONFIRM_E = float(
    os.environ.get("AUTOPILOT_SEQ_PROMOTION_FINAL_CONFIRM_E", "100")
)
SEQ_PROMOTION_DELTA_CI_ALPHA = float(
    os.environ.get("AUTOPILOT_SEQ_PROMOTION_DELTA_CI_ALPHA", "0.05")
)
SEQ_ALPHA_WEALTH_BUDGET = float(os.environ.get("AUTOPILOT_SEQ_ALPHA_WEALTH_BUDGET", "1.0"))
SEQ_PROMOTION_FRESH_EVAL_TIER = int(os.environ.get("AUTOPILOT_SEQ_PROMOTION_FRESH_EVAL_TIER", "2"))
SEQ_CANDIDATE_REPLAY_ENABLED = os.environ.get(
    "AUTOPILOT_SEQ_CANDIDATE_REPLAY", "1"
).strip().lower() not in {"0", "false", "no", "off"}
SEQ_CANDIDATE_REPLAY_MIN_COMBINED_E = float(
    os.environ.get("AUTOPILOT_SEQ_CANDIDATE_REPLAY_MIN_COMBINED_E", "0.9")
)
SEQ_CANDIDATE_REPLAY_MIN_QUALITY_E = float(
    os.environ.get("AUTOPILOT_SEQ_CANDIDATE_REPLAY_MIN_QUALITY_E", "1.0")
)
# 2026-08-04 — GRACE PERIOD. Below this k, a candidate is replayed REGARDLESS of its
# accumulated E. Without it the E filters above are applied to a candidate's FIRST
# sample, and a single noisy observation decides whether it is ever measured again.
#
# Measured over the whole journal (141 candidates, 1,362 seq rows): 89 candidates
# (63%) were stranded at k=1, and their E_quality distribution is
#
#     stranded  min 0.923   MEDIAN 0.999   max 1.019      (filter: 1.0)
#     continued min 0.975   median 1.025   max 11.551
#
# The two populations overlap almost completely at the bottom; the median stranded
# candidate missed by 0.001. That is a coin flip at the third decimal deciding a
# candidate's entire life, and it is the direct cause of `confirmed = 0 in 396`:
# an e-process accumulates evidence multiplicatively across trials, so a candidate
# held at k=1 cannot clear ANY bar, however good it is.
#
# This is a COMPUTE-ALLOCATION heuristic and touches no statistical guarantee — the
# same conclusion the 2026-07-28 re-adjudication reached about `budget=8`
# ("no bearing on anytime-validity; confirm_e=20.0, the Ville bound for alpha=0.05,
# untouched"). Ville holds for any stopping rule; giving a candidate MORE samples
# cannot inflate the false-positive rate, it only reduces false negatives.
SEQ_CANDIDATE_REPLAY_MIN_K = int(os.environ.get("AUTOPILOT_SEQ_CANDIDATE_REPLAY_MIN_K", "6"))
# Raised 12 -> 60. The cap must exceed the k at which a real candidate can reach
# confirm_e, or replay abandons winners just short of the bar: the leading candidate
# (70902e4b665474e7) reached E_quality 11.55 at k=40 and needed ~9 more trials at its
# observed growth of 1.0631x/trial. A cap of 12 made confirmation unreachable by
# construction for every candidate that needed sustained evidence.
SEQ_CANDIDATE_REPLAY_MAX_K = int(os.environ.get("AUTOPILOT_SEQ_CANDIDATE_REPLAY_MAX_K", "60"))


def _required_gate_env() -> dict[str, str]:
    """The authority contract, DERIVED from the launcher that establishes it.

    2026-08-04: this used to be a second hand-maintained copy of the same contract.
    `start_authority_daemon.AUTHORITY_ENV` sets the environment; this dict
    validated it; and the two drifted the moment an operator changed one. Flipping
    AUTOPILOT_SEQ_VERDICT to "0" in the launcher (SEQ-B was unreachable) left this
    copy at "1", so every daemon start was refused —

        ERROR: AutoPilot authority gate env mismatch; refusing direct start.
        Missing or mismatched env: {"AUTOPILOT_SEQ_VERDICT": {"actual":"0","expected":"1"}}

    — and the supervisor burned its three restarts against a contradiction between two
    files that are supposed to say the same thing. The 2026-08-04 architecture review
    had already flagged this pair by name ("one contract, two hand-maintained copies").

    Now there is one source. The launcher declares the contract; this validates the
    subset the daemon must actually have. A key the launcher stops setting fails loudly
    here, which is the check's real purpose — it exists to catch a bare `autopilot.py
    start` that would silently drop sequential verdicts or tool sentinels, NOT to
    second-guess an operator's deliberate switch.
    """
    keys = (
        "AUTOPILOT_SEQ_VERDICT",
        "AUTOPILOT_SEQ_P0_2_BRIDGE",
        "AUTOPILOT_W6_AUDIT_BLOCK",
        "AUTOPILOT_PLANNER_HINTS",
        "AUTOPILOT_TOOL_SENTINELS",
        "AUTOPILOT_STEPPING_STONES",
        "AUTOPILOT_PLANNER_SPEND_BREAKER",
        "AUTOPILOT_PLANNER_ROSTER_POLICY_ACTIVE",
    )
    try:
        import importlib.util as _ilu

        _spec = _ilu.spec_from_file_location(
            "_authority_contract",
            str(Path(__file__).resolve().parent / "start_authority_daemon.py"),
        )
        if _spec is None or _spec.loader is None:
            raise ImportError("launcher spec unavailable")
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        declared = dict(_mod.AUTHORITY_ENV)
    except Exception:  # noqa: BLE001 — never make the gate unreachable
        # Fail SAFE, not open: if the launcher cannot be read we cannot know the
        # contract, so validate nothing rather than enforce a stale guess.
        return {}
    return {k: declared[k] for k in keys if k in declared}


AUTOPILOT_REQUIRED_GATE_ENV = _required_gate_env()
AUTOPILOT_AUTHORITY_LAUNCHER = "scripts/autopilot/start_authority_daemon.py"
SAFE_FALLBACK_SEED_N = 14
FALLBACK_SEED_CANDIDATES = (14, 16, 18, 20, 24, 30, 40, 50, 10)

# 2026-05-23 constrained-creativity planner knobs (gated on stagnation).
# Lean prompt is the default; the rich rubric+synthesis fragment activates
# only when one of the stagnation signals fires, to avoid spending prompt
# budget when autopilot is mid-exploit on a working lead.
CREATIVITY_N = 3  # candidates the rich prompt asks the controller to generate
TAIL_WINDOW = 30  # lookback for action_distribution "under-used" classification
TAIL_SEED_COUNT = 3  # seeds (not candidates) passed to LLM as inspiration
STAGNATION_HV_EPS = 1e-3  # hv_slope_10 strictly below this triggers rich prompt
STAGNATION_STREAK = 3  # N consecutive same-action_type trials triggers rich prompt
PLOT_INTERVAL = 10  # Generate plots every N trials
# Plots also refresh on a wall-clock timer so a long mid-decade gap (e.g. a
# multi-hour tier-2 eval) doesn't leave the dashboard stale, and a blocking
# lifecycle render is time-capped so shutdown can never hang on it.
PLOT_MAX_AGE_S = 600.0
PLOT_SYNC_TIMEOUT_S = 180.0
# A "trial" is an experiment that COLLECTS METRICS (runs an eval). These meta /
# housekeeping actions intentionally return no EvalResult — they mutate bookkeeping
# state (knowledge distillation, memory reset) without measuring anything — so they must
# NOT consume a trial number. They still execute and are logged; they simply do not count.
# MAX_CONSECUTIVE_META halts the loop if the planner gets stuck emitting only meta actions
# (the 2026-05-31 gate-lock symptom: ~80 distill_knowledge in a row, counter ticking with
# zero trials run), turning a silent no-op spin into a loud, fast stop for operator review.
META_NOOP_ACTIONS = {"distill_knowledge", "reset_memories"}
MAX_CONSECUTIVE_META = 5

# 2026-06-04 — non-executing-action (invalid/skipped) handling. The dispatcher
# now returns a SkipOutcome (not bare None) for actions that fail validation or
# are dropped by a guard. These are first-class outcomes: journaled, counted,
# fingerprinted, blacklisted, and circuit-broken — closing the blind spot that
# let the planner re-sample an impossible action 119× (graph_router deadlock).
MAX_CONSECUTIVE_SKIP = int(os.environ.get("AUTOPILOT_MAX_CONSECUTIVE_SKIP", "4"))
# A repeated *invalid* signature (stable validator reason, e.g. an unmet flag
# dependency) is auto-blacklisted at this many occurrences. Only "invalid"
# outcomes are blacklisted — generic "skipped" ones use too coarse a pattern.
INVALID_SIGNATURE_BLACKLIST_THRESHOLD = int(
    os.environ.get("AUTOPILOT_INVALID_BLACKLIST_THRESHOLD", "2")
)

# 2026-06-04 — draft_critique authority. When the binding critic rejects/revises
# a draft, the SUBSTITUTED action (safe fallback / revised_action) is what runs —
# but the rejected draft must still feed the invalid-action feedback + blacklist,
# or the planner could re-draft the same rejected action forever while the critic
# silently swaps in seed_batch. A run of consecutive rejected drafts is a stuck
# planner and halts loudly (mirrors MAX_CONSECUTIVE_META / MAX_CONSECUTIVE_SKIP).
MAX_CONSECUTIVE_REJECTED_DRAFTS = int(
    os.environ.get("AUTOPILOT_MAX_CONSECUTIVE_REJECTED_DRAFTS", "4")
)

# 2026-08-04 — the planner's deterministic pre-dispatch guard ("revise/reject must
# produce a materially different action") used to HALT the daemon on its FIRST hit,
# unlike every sibling breaker above, which substitutes a safe action and only halts
# after a RUN. That is a category error: a critic returning 'revise' with an action
# equal to the draft is a routine planner-quality event, not an unrecoverable one like
# `planners_offline_no_deterministic_fallback` next to it. Evidence: the 15:23:54 halt
# at trial 1472 ended the run on a single `revise` whose substituted action matched the
# draft — one ordinary critic disagreement stopped the whole loop, which is a large part
# of why AutoPilot was not ratcheting. Give it the same run-based discipline; a genuine
# stuck planner still halts, just not on a single event.
MAX_CONSECUTIVE_PLANNER_DETERMINISTIC_BLOCKS = int(
    os.environ.get("AUTOPILOT_MAX_CONSECUTIVE_PLANNER_DETERMINISTIC_BLOCKS", "4")
)

PLANNER_DETERMINISTIC_BLOCK_STATE_KEY = "consecutive_planner_deterministic_blocks"


def _planner_deterministic_block_decision(
    state: dict[str, Any],
    planner_decision: Any,
    max_blocks: int = MAX_CONSECUTIVE_PLANNER_DETERMINISTIC_BLOCKS,
) -> tuple[str, int]:
    """Record one deterministic pre-dispatch block; decide substitute-vs-halt.

    Returns ``("substitute" | "halt", consecutive_count)``. Extracted from the trial loop
    so the breaker's semantics are testable on their own — the behaviour it replaced was a
    bare ``break`` buried a thousand lines into ``_run_loop_inner``, which is precisely why
    a single ordinary critic disagreement could end a run unnoticed.

    The rejected draft is always recorded as invalid-action feedback, so the planner learns
    from a substituted trial exactly as it does from a critic-rejected one.
    """
    state["last_invalid_action"] = getattr(planner_decision, "draft_action", None)
    state["last_invalid_reason"] = getattr(planner_decision, "deterministic_block_reason", "")
    state["last_invalid_status"] = "planner_deterministic_guard"
    # autopilot_state.json is operator-editable JSON and survives restarts, so this counter
    # can arrive as junk or as a negative. Both used to DISABLE the breaker silently: a
    # stored -3 counts up through -2, -1, 0 ... and never reaches the limit, and a
    # non-numeric value raised straight out of the trial loop. Clamp to a sane floor and
    # treat unparseable as "no prior blocks" — a guard that fails open is not a guard.
    try:
        prior = int(state.get(PLANNER_DETERMINISTIC_BLOCK_STATE_KEY, 0) or 0)
    except (TypeError, ValueError):
        log.warning(
            "Ignoring non-numeric %s in autopilot state (%r); treating as 0.",
            PLANNER_DETERMINISTIC_BLOCK_STATE_KEY,
            state.get(PLANNER_DETERMINISTIC_BLOCK_STATE_KEY),
        )
        prior = 0
    count = max(0, prior) + 1
    state[PLANNER_DETERMINISTIC_BLOCK_STATE_KEY] = count
    return ("halt" if count >= max_blocks else "substitute", count)


# 2026-06-04 — experiment-quota policy (separate memory maintenance from the
# optimization budget). Once the memory store is already large, an unbounded run
# of passive seed/distill actions is the planner rationalizing no-op work; cap
# consecutive passive actions and force a frontier-moving experiment instead.
PASSIVE_ACTIONS = {"seed_batch", "distill_knowledge", "distill_skillbank"}
OUTCOME_PROGRESS_ACTIONS = {
    "code_mutation",
    "gepa_optimize",
    "numeric_trial",
    "prompt_mutation",
    "structural_experiment",
    "train_routing_models",
}
SEQ_PROMOTION_DEPENDENT_ACTIONS = {
    "code_mutation",
    "gepa_optimize",
    "numeric_trial",
    "prompt_mutation",
    "structural_experiment",
}
SEQ_GATE_PREFLIGHT_ENABLED = os.environ.get(
    "AUTOPILOT_SEQ_GATE_PREFLIGHT", "1"
).strip().lower() not in {"0", "false", "no", "off"}
SEQ_GATE_PREFLIGHT_MIN_SEQ_ROWS = int(
    os.environ.get("AUTOPILOT_SEQ_GATE_PREFLIGHT_MIN_SEQ_ROWS", "20")
)
SEQ_GATE_PREFLIGHT_RECENT_WINDOW = int(
    os.environ.get("AUTOPILOT_SEQ_GATE_PREFLIGHT_RECENT_WINDOW", "40")
)
SEQ_GATE_PREFLIGHT_MAX_RATE_E = float(
    os.environ.get("AUTOPILOT_SEQ_GATE_PREFLIGHT_MAX_RATE_E", "2.0")
)
QUOTA_MEMORY_THRESHOLD = int(os.environ.get("AUTOPILOT_QUOTA_MEMORY_THRESHOLD", "2000"))
MAX_CONSECUTIVE_PASSIVE = int(os.environ.get("AUTOPILOT_MAX_CONSECUTIVE_PASSIVE", "3"))
HIGHER_TIER_PROBE_GUARD = os.environ.get(
    "AUTOPILOT_HIGHER_TIER_PROBE_GUARD", "1"
).strip().lower() not in {"0", "false", "no", "off"}
HIGHER_TIER_PROBE_TIERS = tuple(
    int(part)
    for part in os.environ.get("AUTOPILOT_HIGHER_TIER_PROBE_TIERS", "2,3").split(",")
    if part.strip().isdigit() and int(part) > DEFAULT_FRONTIER_TIER
) or (2, 3)
HIGHER_TIER_PROBE_MIN_GAP_TRIALS = int(
    os.environ.get("AUTOPILOT_HIGHER_TIER_PROBE_MIN_GAP_TRIALS", "12")
)
HIGHER_TIER_PROBE_STALE_TRIALS = int(
    os.environ.get("AUTOPILOT_HIGHER_TIER_PROBE_STALE_TRIALS", "24")
)
HIGHER_TIER_PROBE_MIN_TRIALS_PER_TIER = int(
    os.environ.get("AUTOPILOT_HIGHER_TIER_PROBE_MIN_TRIALS_PER_TIER", "3")
)
# Internal numeric_trial fallbacks may omit params so Optuna suggests values.
# Model-authored planner actions must carry one explicit param; the coordinator
# blocks empty-param planner output before dispatch.
_FALLBACK_NUMERIC_SURFACES = ("think_harder", "escalation", "monitor", "memrl_retrieval")
_PLANNER_HINTS_ENABLED = os.environ.get("AUTOPILOT_PLANNER_HINTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_PLANNER_SUPPRESSED_NUMERIC_SURFACES: set[str] = set()
_PLANNER_DENYLISTED_FEATURE_FLAGS: set[str] = set()


def _operator_suppressed_numeric_surfaces() -> set[str]:
    """Return launch-scoped numeric surfaces the operator has explicitly withheld."""
    return {
        surface.strip()
        for surface in os.environ.get("AUTOPILOT_SUPPRESSED_NUMERIC_SURFACES", "").split(",")
        if surface.strip()
    }


_PLANNER_STRATEGY_HINT_QUERIES: tuple[tuple[str, str], ...] = (
    (
        "structural_lab",
        "tool use sentinel lane native tools repl react_mode activation latency",
    ),
    (
        "seeder",
        "seed_batch tool_use repl native tools code math retrieval tool_helpfulness",
    ),
    (
        "prompt_forge",
        "prompt mutation repl tool helpfulness verbosity CALL native tools",
    ),
    (
        "numeric_swarm",
        "tool output compression repl tool budget latency tool_helpfulness",
    ),
)


def _configured_numeric_surfaces() -> tuple[str, ...]:
    """Return the NumericSwarm surfaces available to planner/forced trials."""
    try:
        from species.numeric_swarm import SURFACES as _NS_SURFACES
    except Exception:
        surfaces = _FALLBACK_NUMERIC_SURFACES
    else:
        surfaces = (
            tuple(
                surface for surface in _NS_SURFACES if isinstance(surface, str) and surface.strip()
            )
            or _FALLBACK_NUMERIC_SURFACES
        )

    suppressed = _PLANNER_SUPPRESSED_NUMERIC_SURFACES
    if not suppressed:
        return surfaces
    return tuple(surface for surface in surfaces if surface not in suppressed)


def _planner_convention_bindings(
    strategy_store: StrategyStore | None,
    journal: ExperimentJournal | None,
    *,
    species: str,
) -> set[str]:
    """Return live convention bindings for planner visibility."""
    if not _PLANNER_HINTS_ENABLED or strategy_store is None:
        return set()
    if not hasattr(strategy_store, "retrieve_conventions"):
        log.warning(
            "Skipping %s planner convention bindings: StrategyStore lacks retrieve_conventions()",
            species,
        )
        return set()
    try:
        conventions = strategy_store.retrieve_conventions(
            species=species,
            journal=journal,
        )
    except TypeError:
        log.warning(
            "Skipping %s planner convention bindings: incompatible "
            "retrieve_conventions() signature",
            species,
        )
        return set()
    except Exception as exc:
        log.warning("Skipping %s planner convention bindings: %s", species, exc)
        return set()

    bindings: set[str] = set()
    for entry in conventions:
        metadata = getattr(entry, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            continue
        if str(metadata.get("bind_status", "")).strip().lower() != "live":
            continue
        identifiers = metadata.get("bind_identifiers", [])
        if not isinstance(identifiers, list):
            continue
        bindings.update(
            str(identifier).strip() for identifier in identifiers if str(identifier).strip()
        )
    return bindings


def _refresh_planner_convention_bindings(
    strategy_store: StrategyStore | None,
    journal: ExperimentJournal | None,
    *,
    reason: str = "planner_turn",
) -> None:
    """Refresh default-off convention bindings for prompts and validation."""
    previous_flags = set(_PLANNER_DENYLISTED_FEATURE_FLAGS)
    previous_surfaces = set(_PLANNER_SUPPRESSED_NUMERIC_SURFACES)
    _PLANNER_DENYLISTED_FEATURE_FLAGS.clear()
    _PLANNER_SUPPRESSED_NUMERIC_SURFACES.clear()
    _PLANNER_SUPPRESSED_NUMERIC_SURFACES.update(_operator_suppressed_numeric_surfaces())
    if not _PLANNER_HINTS_ENABLED:
        controller_io.set_suppressed_numeric_surfaces(_PLANNER_SUPPRESSED_NUMERIC_SURFACES)
        return

    _PLANNER_DENYLISTED_FEATURE_FLAGS.update(
        _planner_convention_bindings(
            strategy_store,
            journal,
            species="structural_lab",
        )
    )
    _PLANNER_SUPPRESSED_NUMERIC_SURFACES.update(
        _planner_convention_bindings(
            strategy_store,
            journal,
            species="numeric_swarm",
        )
    )
    controller_io.set_suppressed_numeric_surfaces(_PLANNER_SUPPRESSED_NUMERIC_SURFACES)
    current_flags = set(_PLANNER_DENYLISTED_FEATURE_FLAGS)
    current_surfaces = set(_PLANNER_SUPPRESSED_NUMERIC_SURFACES)
    if (
        reason == "startup"
        or current_flags != previous_flags
        or current_surfaces != previous_surfaces
    ):
        log.info(
            "Planner convention bindings refreshed (%s): denylisted_flags=%s "
            "suppressed_numeric_surfaces=%s",
            reason,
            sorted(current_flags),
            sorted(current_surfaces),
        )


def _install_planner_convention_bindings(
    strategy_store: StrategyStore | None,
    journal: ExperimentJournal | None,
) -> None:
    """Install planner convention bindings at startup."""
    _refresh_planner_convention_bindings(
        strategy_store,
        journal,
        reason="startup",
    )


def _planner_strategy_entry_line(entry: Any) -> str:
    if isinstance(entry, dict):
        metadata = entry.get("metadata", {}) or {}
        species = str(entry.get("species", "") or "unknown")
        entry_type = str(entry.get("entry_type", "") or "raw")
        title = (
            str(entry.get("title", "") or "").strip()
            or str(entry.get("description", "") or "").strip()
            or "strategy"
        )
        content = (
            str(entry.get("generalized_content", "") or "").strip()
            or str(entry.get("insight", "") or "").strip()
        )
    else:
        metadata = getattr(entry, "metadata", {}) or {}
        species = str(getattr(entry, "species", "") or "unknown")
        entry_type = str(getattr(entry, "entry_type", "") or "raw")
        title = (
            str(getattr(entry, "title", "") or "").strip()
            or str(getattr(entry, "description", "") or "").strip()
            or "strategy"
        )
        content = (
            str(getattr(entry, "generalized_content", "") or "").strip()
            or str(getattr(entry, "insight", "") or "").strip()
        )
    if not isinstance(metadata, dict):
        metadata = {}
    if len(content) > 260:
        content = content[:257].rstrip() + "..."

    tags = [f"{species}/{entry_type}"]
    bind_status = str(metadata.get("bind_status", "") or "").strip()
    if bind_status:
        tags.append(f"bind={bind_status}")
    identifiers = metadata.get("bind_identifiers", [])
    if isinstance(identifiers, list) and identifiers:
        joined = ",".join(str(item) for item in identifiers[:4])
        tags.append(f"orchestrator_ids={joined}")
        if any(
            str(item).strip().lower() in {"tool", "tools", "tool_use", "repl", "react_mode", "call"}
            for item in identifiers
        ):
            tags.append("scope=orchestrator_eval_tools_not_planner_tools")
    source_handoff = str(metadata.get("source_handoff", "") or "").strip()
    if source_handoff:
        tags.append(f"handoff={source_handoff}")

    return f"- [{' | '.join(tags)}] {title}: {content}".strip()


def _planner_entry_id(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("id", "") or "")
    return str(getattr(entry, "id", "") or "")


def _operator_seed_planner_entries(
    strategy_store: StrategyStore | None,
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    """Return explicit operator-seeded hints without relying on vector search."""
    if max_rows <= 0 or strategy_store is None:
        return []
    conn = getattr(strategy_store, "_conn", None)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT id, species, entry_type, description, insight, "
            "metadata_json, created_at "
            "FROM strategies "
            "WHERE entry_type IN ('pattern', 'convention') "
            "ORDER BY created_at DESC "
            "LIMIT 256"
        ).fetchall()
    except Exception as exc:
        log.warning("Skipping operator StrategyStore planner hints: %s", exc)
        return []

    entries: list[dict[str, Any]] = []
    for row in rows:
        try:
            metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}

        seed_campaign = str(metadata.get("seed_campaign", "") or "").lower()
        entry_id = str(row["id"] or "")
        if not (
            metadata.get("planner_visible") is True
            or str(metadata.get("seeded_by", "") or "").lower() == "operator"
            or seed_campaign.startswith("operator")
            or entry_id.startswith("opseed-")
        ):
            continue

        insight_format = metadata.get("insight_format", {})
        if not isinstance(insight_format, dict):
            insight_format = {}
        entries.append(
            {
                "id": entry_id,
                "species": str(row["species"] or "all"),
                "entry_type": str(row["entry_type"] or "pattern"),
                "description": str(row["description"] or ""),
                "insight": str(row["insight"] or ""),
                "title": (
                    str(insight_format.get("title", "") or "").strip()
                    or str(row["description"] or "").strip()
                ),
                "generalized_content": (
                    str(insight_format.get("generalized_content", "") or "").strip()
                    or str(row["insight"] or "").strip()
                ),
                "metadata": metadata,
                "created_at": str(row["created_at"] or ""),
            }
        )

    bind_rank = {"live": 0, "future": 1, "context": 2}
    tranche_rank = {"green": 0, "guardrail": 1, "frozen": 2}
    entries.sort(
        key=lambda entry: (
            bind_rank.get(
                str(entry["metadata"].get("bind_status", "") or "").lower(),
                3,
            ),
            tranche_rank.get(
                str(entry["metadata"].get("tranche", "") or "").lower(),
                3,
            ),
        )
    )
    return entries[:max_rows]


def _build_planner_strategy_hints(
    strategy_store: StrategyStore | None,
    journal: ExperimentJournal | None,
    *,
    max_rows: int = 10,
) -> str:
    """Render bounded StrategyStore rows into the planner prompt each turn."""
    if not _PLANNER_HINTS_ENABLED:
        return "(disabled; set AUTOPILOT_PLANNER_HINTS=1 to include StrategyStore rows)"
    if strategy_store is None:
        return "(unavailable: StrategyStore did not load)"

    rows: list[Any] = []
    seen: set[str] = set()

    def add_entries(entries: Any) -> None:
        if not entries:
            return
        for entry in entries:
            entry_id = _planner_entry_id(entry)
            if entry_id and entry_id in seen:
                continue
            if entry_id:
                seen.add(entry_id)
            rows.append(entry)
            if len(rows) >= max_rows:
                return

    try:
        health_fn = getattr(strategy_store, "search_index_health", None)
        if callable(health_fn):
            health = health_fn()
            if not health.get("healthy", True):
                add_entries(
                    [
                        {
                            "id": "strategy-search-index-degraded",
                            "species": "system",
                            "entry_type": "warning",
                            "title": "StrategyStore search index degraded",
                            "generalized_content": (
                                f"{health.get('summary', 'search mirror degraded')}; "
                                f"repair={health.get('repair_hint', 'rebuild search indexes')}"
                            ),
                        }
                    ]
                )
    except Exception as exc:
        rows.append(
            {
                "id": "strategy-search-index-health-error",
                "species": "system",
                "entry_type": "warning",
                "title": "StrategyStore search index health unavailable",
                "generalized_content": str(exc),
            }
        )

    add_entries(
        _operator_seed_planner_entries(
            strategy_store,
            max_rows=min(max_rows, max(4, max_rows // 2)),
        )
    )

    try:
        if hasattr(strategy_store, "retrieve_for_journal"):
            for species, query in _PLANNER_STRATEGY_HINT_QUERIES:
                add_entries(
                    strategy_store.retrieve_for_journal(
                        query,
                        journal=journal,
                        k=3,
                        species=species,
                    )
                )
                if len(rows) >= max_rows:
                    break
    except Exception as exc:
        rows.append(
            {
                "id": "strategy-retrieval-error",
                "species": "system",
                "entry_type": "error",
                "title": "StrategyStore retrieval unavailable",
                "generalized_content": str(exc),
            }
        )

    try:
        if len(rows) < max_rows and hasattr(strategy_store, "retrieve_conventions"):
            for species in ("structural_lab", "numeric_swarm", "seeder", "prompt_forge"):
                add_entries(
                    strategy_store.retrieve_conventions(
                        species=species,
                        journal=journal,
                        limit=3,
                    )
                )
                if len(rows) >= max_rows:
                    break
    except Exception as exc:
        rows.append(
            {
                "id": "strategy-conventions-error",
                "species": "system",
                "entry_type": "error",
                "title": "StrategyStore conventions unavailable",
                "generalized_content": str(exc),
            }
        )

    if not rows:
        return "(no StrategyStore rows matched the current planner hint queries)"

    return "\n".join(_planner_strategy_entry_line(entry) for entry in rows[:max_rows])


def _build_higher_tier_planner_pressure(
    archive: ParetoArchive | None,
    gate: SafetyGate | None,
    *,
    tiers: tuple[int, ...] = (2, 3),
    w8_candidate_generation_active: bool = False,
) -> str:
    """Summarize non-default eval tiers for planner pressure without cross-tier scoring."""
    if archive is None:
        return "(unavailable: Pareto archive did not load)"

    lines = [
        "Use higher-tier evidence as optimization pressure, not as cross-tier scores:",
        "- T2/T3 are broader and harder slices of the real task pool; T3 includes expert/hard workflow tasks.",
        "- Preserve T1 as the deployment safety lane, but expect durable wins to generalize: T1 gains that never lift T2/T3 are overfit risk.",
        "- Prefer actions likely to improve T2/T3 same-tier frontier quality, then validate with deep_eval tier 2 or 3.",
        "- If T1/T2 hypervolume is plateauing, use the current kernel era for T3 hard-workflow exploration instead of repeated local T1 exploitation.",
        "- Never compare raw quality across tiers; compare each tier only to its own baseline/frontier.",
    ]
    if w8_candidate_generation_active:
        lines.append(
            "- W8 candidate-generation override: do not emit seed_batch, "
            "deep_eval, or structural_prune for higher-tier coverage this turn. "
            "Preserve T2/T3 pressure through an available replayable candidate "
            "action instead: numeric_trial with journaled applied params, or a "
            "one-flag structural_experiment with a same-tier T2/T3 falsifier."
        )
    else:
        lines.extend(
            [
                "- T3 hard-workflow probes should favor technical tool-use, REPL, and multi-turn agentic hypotheses when W8 is not requesting promotion evidence.",
                "- When W8 replay evidence is not asking for a specific promotion eval, prefer deep_eval tier 3 if T3 coverage/frontier is thin.",
            ]
        )
    baseline = getattr(gate, "baseline", None) if gate is not None else None
    plateau_parts: list[str] = []
    for tier in (DEFAULT_FRONTIER_TIER, 2):
        try:
            summary = archive.summary(tier=tier)
        except Exception:
            continue
        try:
            frontier_size = int(summary.get("frontier_size") or 0)
            hv_slope = float(summary.get("hv_slope_50"))
        except (TypeError, ValueError):
            continue
        if frontier_size > 0 and abs(hv_slope) < STAGNATION_HV_EPS:
            plateau_parts.append(f"T{tier} hv_slope_50={hv_slope:+.6f}")
    if plateau_parts:
        lines.append(
            "- Plateau signal: "
            + ", ".join(plateau_parts)
            + "; prioritize hard-workflow generalization pressure (especially T3) "
            "until the next instrument/kernel era resets frontier-speed evidence."
        )
    for tier in tiers:
        if tier <= DEFAULT_FRONTIER_TIER:
            continue
        try:
            summary = archive.summary(tier=tier)
        except Exception as exc:
            lines.append(f"- T{tier}: unavailable ({exc})")
            continue
        frontier_size = int(summary.get("frontier_size") or 0)
        best_quality = float(summary.get("best_quality") or 0.0)
        try:
            baseline_quality = (
                float(baseline.quality_for_tier(tier)) if baseline is not None else None
            )
        except Exception:
            baseline_quality = None
        if frontier_size <= 0:
            baseline_text = (
                f"; baseline_q={baseline_quality:.3f}" if baseline_quality is not None else ""
            )
            if w8_candidate_generation_active:
                lines.append(
                    f"- T{tier}: empty frontier{baseline_text}; defer the "
                    f"deep_eval tier {tier} coverage probe until W8 candidate "
                    "generation clears."
                )
            else:
                lines.append(
                    f"- T{tier}: empty frontier{baseline_text}; schedule deep_eval tier {tier} "
                    "when evidence budget allows."
                )
            continue
        delta_text = ""
        if baseline_quality is not None:
            delta_text = f", delta_vs_baseline={best_quality - baseline_quality:+.3f}"
        lines.append(
            f"- T{tier}: frontier={frontier_size}, best_q={best_quality:.3f}{delta_text}, "
            f"best_speed={float(summary.get('best_speed') or 0.0):.1f} t/s"
        )
    return "\n".join(lines)


_POOL_QUESTION_COUNT_CACHE: dict[str, Any] | None = None


def _pool_question_count_summary() -> tuple[int | None, dict[int, int]]:
    """Return total + per-tier pool counts, cached by question-pool file metadata."""
    candidates = (
        ORCH_ROOT / "benchmarks" / "prompts" / "question_pool.jsonl",
        Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl"),
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            stat = path.stat()
            cache_key = (str(path), int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            continue
        global _POOL_QUESTION_COUNT_CACHE
        if _POOL_QUESTION_COUNT_CACHE and _POOL_QUESTION_COUNT_CACHE.get("cache_key") == cache_key:
            return (
                _POOL_QUESTION_COUNT_CACHE.get("total"),
                dict(_POOL_QUESTION_COUNT_CACHE.get("tier_counts") or {}),
            )
        total: int | None = None
        tier_counts: dict[int, int] = {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    text = raw.strip()
                    if not text:
                        continue
                    row = json.loads(text)
                    if not isinstance(row, dict):
                        continue
                    if row.get("__pool_metadata__"):
                        try:
                            metadata_total = int(row.get("total_questions") or 0)
                            total = metadata_total if metadata_total > 0 else total
                        except (TypeError, ValueError):
                            pass
                        continue
                    try:
                        tier = int(row.get("tier"))
                    except (TypeError, ValueError):
                        continue
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1
        except Exception:
            continue
        if total is None and tier_counts:
            total = sum(tier_counts.values())
        _POOL_QUESTION_COUNT_CACHE = {
            "cache_key": cache_key,
            "total": total,
            "tier_counts": dict(tier_counts),
        }
        return total, tier_counts
    return None, {}


def _pool_total_question_count() -> int | None:
    """Return the question-pool size from metadata or cached per-tier scan."""
    total, _tier_counts = _pool_question_count_summary()
    return total


def _eval_question_results(entry: JournalEntry) -> list[dict[str, Any]]:
    details = entry.eval_details if isinstance(entry.eval_details, dict) else {}
    nested = details.get("details") if isinstance(details.get("details"), dict) else {}
    for container in (details, nested):
        for key in ("question_results", "per_question_results", "per_question"):
            raw = container.get(key)
            if isinstance(raw, list):
                return [item for item in raw if isinstance(item, dict)]
    return []


def _build_eval_coverage_pressure(
    journal: ExperimentJournal | None,
    *,
    pool_total_questions: int | None = None,
    pool_tier_questions: Mapping[int | str, int] | None = None,
    w8_candidate_generation_active: bool = False,
) -> str:
    """Summarize eval-task coverage so planner search does not overfit a narrow slice."""
    if journal is None:
        return "(unavailable: ExperimentJournal did not load)"

    entries = journal.entries_with_supersessions()
    question_rows = 0
    distinct: set[tuple[str, str]] = set()
    tier_trials: dict[int, int] = {}
    tier_question_rows: dict[int, int] = {}
    tier_distinct: dict[int, set[tuple[str, str]]] = {}
    suite_distinct: dict[str, set[str]] = {}
    for entry in entries:
        results = _eval_question_results(entry)
        if not results:
            continue
        tier = int(entry.tier)
        tier_trials[tier] = tier_trials.get(tier, 0) + 1
        for result in results:
            qid = str(
                result.get("qid") or result.get("question_id") or result.get("id") or ""
            ).strip()
            if not qid:
                continue
            suite = str(result.get("suite") or "unknown").strip() or "unknown"
            question_rows += 1
            key = (suite, qid)
            distinct.add(key)
            tier_question_rows[tier] = tier_question_rows.get(tier, 0) + 1
            tier_distinct.setdefault(tier, set()).add(key)
            if not suite.startswith("sentinel_"):
                suite_distinct.setdefault(suite, set()).add(qid)

    if question_rows <= 0:
        if w8_candidate_generation_active:
            return (
                "No scored question-result rows are available yet. W8 candidate "
                "generation is the active strict blocker, so do not use "
                "seed_batch or deep_eval merely to collect coverage; choose an "
                "available replayable candidate action."
            )
        return (
            "No scored question-result rows are available yet; prefer seed_batch "
            "or deep_eval before drawing planner-learning conclusions."
        )

    distinct_count = len(distinct)
    repeat_factor = question_rows / max(1, distinct_count)
    tier_pool_counts: dict[int, int] = {}
    if pool_tier_questions is not None:
        for tier, count in pool_tier_questions.items():
            try:
                tier_pool_counts[int(tier)] = int(count)
            except (TypeError, ValueError):
                continue
    if pool_total_questions is None and pool_tier_questions is None:
        total, tier_pool_counts = _pool_question_count_summary()
    else:
        total = pool_total_questions
    coverage_text = ""
    if total:
        coverage_text = f", pool_coverage<={distinct_count * 100.0 / total:.2f}% of {total}"
    tier_text = (
        ", ".join(f"T{tier}={count}" for tier, count in sorted(tier_trials.items())) or "none"
    )
    detail_tiers = sorted(
        set(tier_trials) | set(tier_question_rows) | set(tier_distinct) | set(tier_pool_counts)
    )
    tier_detail_parts: list[str] = []
    for tier in detail_tiers:
        distinct_for_tier = len(tier_distinct.get(tier, set()))
        part = (
            f"T{tier}:trials={tier_trials.get(tier, 0)},"
            f"rows={tier_question_rows.get(tier, 0)},"
            f"distinct={distinct_for_tier}"
        )
        pool_count = tier_pool_counts.get(tier)
        if pool_count:
            part += f",pool={pool_count},coverage<={distinct_for_tier * 100.0 / pool_count:.2f}%"
        tier_detail_parts.append(part)
    tier_detail_text = ", ".join(tier_detail_parts) or "none"
    under_sampled_higher_tiers = [tier for tier in (2, 3) if tier_trials.get(tier, 0) < 5]
    under_sampled_text = ""
    if under_sampled_higher_tiers:
        tier_list = ", ".join(
            f"T{tier}={tier_trials.get(tier, 0)} trial(s)" for tier in under_sampled_higher_tiers
        )
        if w8_candidate_generation_active:
            under_sampled_text = (
                f" Higher-tier coverage is thin ({tier_list}), but W8 candidate "
                "generation is the active strict blocker; do not propose "
                "seed_batch or deep_eval for coverage this turn. Preserve the "
                "hard-workflow/tool-use/REPL hypothesis in the falsifier of an "
                "available replayable candidate action."
            )
        else:
            under_sampled_text = (
                f" Higher-tier coverage is thin ({tier_list}); when the Fable gate "
                "does not require a specific W8/promotion action, prefer deep_eval on "
                "the thinnest tier or seed_batch coverage probes that broaden task families. "
                "If T3 is thin, treat hard workflow, tool-use, REPL, and multi-turn task "
                "coverage as high-value exploration."
            )
    least_covered_suites = sorted(
        ((suite, len(qids)) for suite, qids in suite_distinct.items()),
        key=lambda item: item[1],
    )[:5]
    suite_text = ""
    if least_covered_suites:
        suite_text = (
            " Least-covered non-sentinel suites: "
            + ", ".join(f"{suite}={count}" for suite, count in least_covered_suites)
            + "."
        )
    if w8_candidate_generation_active:
        guidance = (
            "If repeat_factor is high or coverage is low, keep the coverage "
            "pressure visible but do not emit seed_batch or deep_eval while W8 "
            "candidate generation is strict. Prefer an available replayable "
            "numeric_trial or one-flag structural_experiment whose falsifier "
            "names expected same-tier T2/T3 movement. Keep fixed authority-core "
            "evidence separate from planner-learning coverage."
        )
    else:
        guidance = (
            "If repeat_factor is high or coverage is low, prefer actions that add "
            "decision-grade diversity: seed_batch on under-covered suites, deep_eval tier 2/3, "
            "or tool-use/REPL/agentic coverage probes. Healthy optimization should eventually "
            "lift same-tier T2/T3 frontier quality; T1-only gains are overfit risk. Keep fixed "
            "authority-core evidence separate from planner-learning coverage."
        )
    lines = [
        (
            f"Eval coverage: {distinct_count} distinct qids / {question_rows} scored rows "
            f"(repeat_factor={repeat_factor:.2f}x{coverage_text}); eval trials by tier: {tier_text}."
        ),
        f"Tier detail: {tier_detail_text}.{under_sampled_text}{suite_text}",
        guidance,
    ]
    return "\n".join(lines)


def _format_outcome_rate(metric: Mapping[str, Any]) -> str:
    count = metric.get("count", 0)
    total = metric.get("total", 0)
    rate = metric.get("rate")
    rate_text = "n/a" if rate is None else f"{float(rate):.1%}"
    return f"{count}/{total} ({rate_text})"


def _format_outcome_per_100(metric: Mapping[str, Any]) -> str:
    count = metric.get("count", 0)
    total = metric.get("total", 0)
    per_100 = metric.get("per_100")
    per_100_text = "n/a" if per_100 is None else f"{float(per_100):.1f}/100"
    return f"{count}/{total} ({per_100_text})"


def _build_outcome_progress_pressure(
    journal_dir: Path | None = None,
    *,
    max_trials_since_frontier: int = DEFAULT_OUTCOME_STALL_FRONTIER_TRIALS,
    max_trials_since_promotion: int = DEFAULT_OUTCOME_STALL_PROMOTION_TRIALS,
    recent_window_trials: int = DEFAULT_OUTCOME_RECENT_WINDOW_TRIALS,
) -> str:
    """Render outcome-yield pressure for the controller prompt.

    This is advisory planner context only. It reuses the phase-health outcome
    progress fold but does not change health, safety, archive, or promotion
    decisions.
    """
    try:
        report = _phase_outcome_progress_report(
            journal_dir=journal_dir or PHASE_DEFAULT_JOURNAL_DIR,
            max_trials_since_frontier=max(0, int(max_trials_since_frontier)),
            max_trials_since_promotion=max(0, int(max_trials_since_promotion)),
            recent_window_trials=max(0, int(recent_window_trials)),
        )
    except Exception as exc:  # noqa: BLE001 - prompt advisory must not block planning
        return f"(outcome progress pressure unavailable: {exc})"

    status = str(report.get("status") or "unknown")
    rates = report.get("rates") if isinstance(report.get("rates"), Mapping) else {}
    keepable = _format_outcome_rate(rates.get("keepable_rate") or {})
    wasted = _format_outcome_rate(rates.get("wasted_eval_rate") or {})
    excluded = _format_outcome_rate(rates.get("learning_excluded_rate") or {})
    regressions = _format_outcome_rate(rates.get("regression_per_active_trial") or {})
    promotions_per_100 = _format_outcome_per_100(
        rates.get("promotions_per_100_active_trials") or {}
    )
    blockers = [str(item) for item in report.get("blockers") or [] if str(item)]
    lines = [
        (
            "Outcome progress: "
            f"status={status}, latest_trial={report.get('latest_trial_id')}, "
            f"frontier_admissions={report.get('frontier_admissions')}, "
            f"latest_frontier={report.get('latest_frontier_trial_id')}, "
            f"trials_since_frontier={report.get('trials_since_frontier')}/"
            f"{report.get('max_trials_since_frontier')}, "
            f"baseline_promotions={report.get('baseline_promotions')}, "
            f"latest_promotion={report.get('latest_promotion_trial_id')}, "
            f"trials_since_promotion={report.get('trials_since_promotion')}/"
            f"{report.get('max_trials_since_promotion')}."
        ),
        (
            f"Recent outcome rates over {recent_window_trials} trials: "
            f"keepable={keepable}, wasted_eval={wasted}, "
            f"learning_excluded={excluded}, "
            f"regression_per_active_trial={regressions}, "
            f"promotions_per_100_active_trials={promotions_per_100}."
        ),
    ]
    if blockers:
        lines.append("Outcome blockers: " + "; ".join(blockers))
        lines.append(
            "Planner pressure: choose actions with a credible path to keepable "
            "frontier or promotion evidence; avoid no-op, already-refuted, or "
            "seed-only churn unless another explicit evidence lane requires it."
        )
    else:
        lines.append(
            "Planner pressure: outcome flow is not currently stalled; continue "
            "highest-information actions while preserving W8/W6 evidence gates."
        )
    return "\n".join(lines)


def _outcome_progress_frontier_stalled(outcome_progress_pressure_text: str) -> bool:
    text = outcome_progress_pressure_text.lower()
    return "outcome blockers:" in text and "frontier admission stale" in text


def _action_can_move_outcome_frontier(action: Mapping[str, Any]) -> bool:
    return str(action.get("type") or "") in OUTCOME_PROGRESS_ACTIONS


_QUOTA_NUMERIC_SURFACES = _configured_numeric_surfaces()


def _blacklisted_action_skip(action: dict[str, Any], blocked_reason: str) -> SkipOutcome:
    """Convert a pre-dispatch blacklist hit into journalable planner feedback."""
    action_type = str(action.get("type") or "unknown")
    return SkipOutcome("invalid", f"action blacklisted: {blocked_reason}", action_type)


def _question_outcome_map(question_results: Any) -> dict[str, bool]:
    """Normalize compact question result rows into a sequential verdict map.

    CJ-8: rows whose ``disposition`` carries no quality signal (``infra_failed``,
    ``scoring_failed``) are EXCLUDED. They used to arrive as ``correct=False``
    and poison ``baseline_profile_from_trials``, which is the reference every
    later e-process observation is centred against — so one backend blip moved
    the baseline itself, not just one trial.
    """
    if not isinstance(question_results, list):
        return {}
    outcomes: dict[str, bool] = {}
    for item in question_results:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("qid") or item.get("question_id") or "").strip()
        if not qid:
            continue
        if not _is_quality_admissible(item.get("disposition")):
            continue
        outcomes[qid] = bool(item.get("correct"))
    return outcomes


def _question_outcome_excluded(question_results: Any) -> int:
    """How many rows :func:`_question_outcome_map` dropped. Reported, never
    silently absorbed: a row that vanishes from a denominator is otherwise
    indistinguishable from one that was never asserted."""
    if not isinstance(question_results, list):
        return 0
    return sum(
        1 for item in question_results
        if isinstance(item, dict)
        and not _is_quality_admissible(item.get("disposition"))
    )


# D6 / FIELD-1: the documented EvalResult metric families that feed the journal's
# eval_details payload. Each name below is a field on safety_gate.EvalResult whose
# value was previously DROPPED from the journal (only the METRIC grep-lines and,
# for a few, JournalEntry top-level columns carried them). The AP-4 axes comment on
# EvalResult says these persist for H-LB; they must actually land in eval_details.
#
# The list is explicit (not reflected from the dataclass) BY DESIGN: a NEW
# diversity_/rubric_/reviewer_ field added to EvalResult must be wired here too, and
# the completeness guard in tests/unit/test_eval_details_field_parity.py fails until
# it is — so the journal schema can never silently drop a freshly-added family axis.
_EVAL_DETAILS_FLOAT_FIELDS: tuple[str, ...] = (
    # EV-8 diversity (5) — NaN means "unavailable this trial".
    "diversity_entropy",
    "diversity_distinct2",
    "diversity_self_bleu",
    "diversity_ttr",
    "diversity_semantic_embedding_agreement",
    # EV-9 / MindDR rubric (4).
    "rubric_reasoning_trajectory",
    "rubric_tool_calls",
    "rubric_outline",
    "rubric_content_stage",
    # AP-4 reviewer-calibration axes (4; review_decision_latency_ms is grouped here
    # even though it lacks the reviewer_ prefix — it is the 4th AP-4 axis).
    "reviewer_fa_rate",
    "reviewer_fr_rate",
    "reviewer_fa_fr_ratio",
    "review_decision_latency_ms",
    # intake-378 branching density + AP-16 instruction-token budget + AM compaction.
    "branching_density",
    "instruction_token_ratio",
    "avg_prompt_tokens",
)
_EVAL_DETAILS_INT_FIELDS: tuple[str, ...] = (
    "instruction_token_count",
    "compaction_events",
)


def _finite_or_none(value: Any) -> float | None:
    """Coerce to float, mapping non-finite / NaN / unparseable to None (null-gated)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _eval_details_from_result(result: Any) -> dict[str, Any]:
    """D6 / FIELD-1 journal leg: map the documented EvalResult metric families into the
    eval_details payload, null-gated.

    Null-gating rule (documented once): a float family value that is NaN / non-finite
    (the dataclass default for "unavailable this trial") becomes ``None``; an int family
    value is emitted as an int (0 default is a real value, not "unavailable"). Every
    family key is ALWAYS present in the returned dict (present-with-None rather than
    omitted) so downstream consumers and the parity guard can rely on a stable schema.
    The journal serializer additionally runs strict-JSON ``json_sanitize`` (NaN -> null),
    so None here is belt-and-suspenders + explicit intent.
    """
    payload: dict[str, Any] = {}
    for name in _EVAL_DETAILS_FLOAT_FIELDS:
        payload[name] = _finite_or_none(getattr(result, name, None))
    for name in _EVAL_DETAILS_INT_FIELDS:
        raw = getattr(result, name, None)
        try:
            payload[name] = int(raw)
        except (TypeError, ValueError):
            payload[name] = None
    # ETR-2 (2026-07-20 audit): carry the measured/placeholder distinction on EVERY
    # journaled trial, not only on the aggregate path that happens to stash it in
    # `details`. A plain bool, never null-gated: False means the row's `quality` is a
    # placeholder (nothing was scored), so a later reader can never mistake it for a
    # measured 0.0.
    payload["quality_measured"] = bool(getattr(result, "quality_measured", True))
    payload["quality_unmeasured_reason"] = str(
        getattr(result, "quality_unmeasured_reason", "") or ""
    )
    return payload


# B4 / SEQ-2: the refusal reason the gate returns when a promotion is blocked because
# the sequential null profile / verdict was unavailable (seq_confirmed=None). Kept as a
# named constant so the autopilot-side surfacing is not a bare string literal.
SEQ_INPUTS_UNAVAILABLE_REASON = "seq_inputs_unavailable"


def _log_baseline_update_result(trial_counter: int, baseline_update: Any) -> None:
    """Surface a baseline update / refusal outcome cleanly (B4 / SEQ-2).

    Preserves the existing 'auto-raised' and generic 'skipped' log lines, and adds a
    DISTINCT line when the gate refuses because the sequential inputs were unavailable
    (seq_confirmed=None -> reason ``seq_inputs_unavailable``). That refusal is expected
    (no trustworthy null this trial => no ratchet), so it must be visible and not lost
    among unrelated 'baseline update skipped' reasons.
    """
    rule = getattr(baseline_update, "promotion_rule", "") or "n/a"
    if baseline_update.updated:
        log.info(
            "Trial %d: T%d baseline auto-raised %.3f → %.3f (promotion_rule=%s)",
            trial_counter,
            baseline_update.tier,
            baseline_update.previous_quality or 0.0,
            baseline_update.new_quality,
            rule,
        )
        return
    log.info("Trial %d: baseline promotion not applied (promotion_rule=%s)", trial_counter, rule)
    reason = getattr(baseline_update, "reason", "") or ""
    # The gate carries the machine token on BaselineUpdateResult.seq_refused_reason
    # (the human `reason` is a longer sentence); check the token first, then fall back
    # to a substring match in case a path stashes it in `reason`.
    seq_refused = getattr(baseline_update, "seq_refused_reason", "") or ""
    if seq_refused == SEQ_INPUTS_UNAVAILABLE_REASON or SEQ_INPUTS_UNAVAILABLE_REASON in reason:
        log.info(
            "Trial %d: baseline promotion REFUSED — sequential inputs unavailable "
            "(%s); seq_confirmed=None so no baseline ratchet this trial (B4/SEQ-2): %s",
            trial_counter,
            SEQ_INPUTS_UNAVAILABLE_REASON,
            reason,
        )
        return
    # RTG-02 (2026-09-14): a refusal that nonetheless re-anchored the independent SPEED axis
    # is a state WRITE, not a no-op, and must not read as "skipped". The throughput fence just
    # closed on its own instrument while quality stayed refused.
    if getattr(baseline_update, "speed_reseeded", False):
        log.warning(
            "Trial %d: quality promotion refused BUT the speed axis was re-anchored "
            "(frontdoor_speed + autopilot_speed_era stamped from this in-era measurement; "
            "RTG-02): %s",
            trial_counter,
            reason,
        )
        return
    log.info(
        "Trial %d: baseline update skipped (%s)",
        trial_counter,
        reason,
    )


def _question_outcome_vector(
    question_results: Any,
    *,
    trial_id: int | None,
) -> dict[str, QuestionOutcome]:
    """Normalize compact question result rows for paired diagnostics.

    CJ-8. THIS is the live McNemar leak: the vector built here feeds
    ``mcnemar_from_vectors`` in ``_seq_paired_baseline_diagnostics`` with no
    disposition filtering on either side, so an ``infra_failed`` row landed in a
    discordant cell as a substantive wrong answer. Excluded rows leave the
    paired universe entirely — an unmatched pair is what
    ``require_matched_comparison`` exists to surface, and it is strictly better
    than a fabricated one.
    """
    if not isinstance(question_results, list):
        return {}
    vector: dict[str, QuestionOutcome] = {}
    try:
        tid = int(trial_id) if trial_id is not None else -1
    except (TypeError, ValueError):
        tid = -1
    for item in question_results:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("qid") or item.get("question_id") or "").strip()
        if not qid:
            continue
        if not _is_quality_admissible(item.get("disposition")):
            continue
        vector[qid] = QuestionOutcome(
            qid=qid,
            suite=str(item.get("suite") or "").strip(),
            correct=bool(item.get("correct")),
            trial_id=tid,
        )
    return vector


def _latest_seq_baseline_reference_vector(
    journal: ExperimentJournal,
    *,
    tier: int,
    candidate_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the latest trusted marked baseline-reference vector for a tier.

    AP-55 (b): with ``candidate_fingerprint``, a reference draw from a different
    infra regime (NON_COMPARABLE) is skipped; the chosen draw's regime verdict is
    returned as ``regime`` (UNVERIFIED for an unfingerprinted legacy draw).
    """
    for entry in reversed(journal.entries_with_supersessions()):
        if getattr(entry, "bug_corrupted_by", ""):
            continue
        if getattr(entry, "outcome_status", "ok") in {"invalid", "skipped"}:
            continue
        try:
            if int(getattr(entry, "tier", -1)) != int(tier):
                continue
        except (TypeError, ValueError):
            continue
        eval_details = getattr(entry, "eval_details", {}) or {}
        if not isinstance(eval_details, dict):
            continue
        if not eval_details.get("seq_baseline_reference_draw"):
            continue
        regime = ""
        if candidate_fingerprint is not None:
            regime = compare_infra_fingerprints(
                candidate_fingerprint, getattr(entry, "infra_fingerprint", None)
            )["status"]
            if regime == INFRA_NON_COMPARABLE:
                continue
        vector = _question_outcome_vector(
            eval_details.get("question_results"),
            trial_id=getattr(entry, "trial_id", None),
        )
        if vector:
            return {
                "trial_id": getattr(entry, "trial_id", None),
                "timestamp": getattr(entry, "timestamp", ""),
                "reason": eval_details.get("seq_baseline_reference_reason", ""),
                "vector": vector,
                "regime": regime,
            }
    return None


def _seq_paired_baseline_diagnostics(
    *,
    journal: ExperimentJournal,
    tier: int,
    candidate: str,
    candidate_trial_id: int,
    question_results: Any,
    candidate_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Observation-only same-qid baseline/candidate paired evidence."""
    candidate_vector = _question_outcome_vector(
        question_results,
        trial_id=candidate_trial_id,
    )
    if not candidate_vector:
        return {
            "status": "no_candidate_vector",
            "candidate": candidate,
            "candidate_trial_id": candidate_trial_id,
            "used_for_gating": False,
        }
    baseline = _latest_seq_baseline_reference_vector(
        journal, tier=tier, candidate_fingerprint=candidate_fingerprint
    )
    if baseline is None:
        return {
            "status": "no_baseline_reference_vector",
            "candidate": candidate,
            "candidate_trial_id": candidate_trial_id,
            "candidate_vector_qids": len(candidate_vector),
            "used_for_gating": False,
        }
    result = mcnemar_from_vectors(
        baseline["vector"],
        candidate_vector,
        label_a=f"baseline_reference:{baseline['trial_id']}",
        label_b=f"candidate:{candidate_trial_id}",
    )
    payload = asdict(result)
    payload["mcnemar_verdict"] = verdict_from_result(result)
    payload.update(
        {
            "status": "ok" if result.shared_qids > 0 else "no_shared_qids",
            "candidate": candidate,
            "candidate_trial_id": candidate_trial_id,
            "candidate_vector_qids": len(candidate_vector),
            # CJ-9. Rows excluded for a non-quality disposition never entered the
            # pairing. They are reported here so a shrinking `shared_qids` is
            # readable as an exclusion rather than as a smaller run.
            "candidate_excluded_non_quality": _question_outcome_excluded(
                question_results),
            "baseline_reference_trial_id": baseline.get("trial_id"),
            "baseline_reference_timestamp": baseline.get("timestamp", ""),
            "baseline_reference_reason": baseline.get("reason", ""),
            "baseline_reference_regime": baseline.get("regime", ""),
            "baseline_reference_vector_qids": len(baseline["vector"]),
            "comparison": "latest_seq_baseline_reference_vs_candidate",
            "method": "exact_mcnemar_sign_test",
            "used_for_gating": False,
        }
    )
    return payload


def _parse_journal_timestamp(timestamp: Any) -> float | None:
    if not timestamp:
        return None
    try:
        text = str(timestamp)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        return None


def _seq_baseline_reference_state(
    journal: ExperimentJournal,
    *,
    tier: int,
    now_ts: float | None = None,
    current_fingerprint: Mapping[str, Any] | None = None,
    regime_forcing: bool = False,
) -> dict[str, Any]:
    """Return freshness/cadence state for the seq baseline reference profile.

    AP-55 (b): with ``current_fingerprint``, the latest reference draw is also
    checked against the current infra regime. ``regime_due`` is set when that draw
    is NON_COMPARABLE or carries no fingerprint (a pre-AP-55 row, read as-is), and
    at least ``AP55_REGIME_REFRESH_MIN_TRIALS`` trials have run since it, or when
    no reference draw exists at all (bootstrap). UNVERIFIED (e.g. a declared-only
    kernel) does not make a re-run due: another draw could not verify it either.

    ``regime_due`` (and its effect on ``due``) is only ever set when
    ``regime_forcing`` is true — the caller passes the AUTOPILOT_AP55_SEED_RERUN
    knob, so an unset knob never schedules a regime re-run, seq on or off.
    ``regime_status`` is reported either way.
    """
    now = time.time() if now_ts is None else float(now_ts)
    trusted_profile_trials = 0
    trials_since_reference = 0
    latest_profile_trial_id = None
    latest_reference_trial_id = None
    latest_reference_ts = None
    latest_reference_fp: Any = None

    for entry in reversed(journal.entries_with_supersessions()):
        if getattr(entry, "bug_corrupted_by", ""):
            continue
        if getattr(entry, "outcome_status", "ok") in {"invalid", "skipped"}:
            continue
        try:
            if int(getattr(entry, "tier", -1)) != int(tier):
                continue
        except (TypeError, ValueError):
            continue
        eval_details = getattr(entry, "eval_details", {}) or {}
        if not isinstance(eval_details, dict):
            continue
        if not _question_outcome_map(eval_details.get("question_results")):
            continue

        trusted_profile_trials += 1
        entry_ts = _parse_journal_timestamp(getattr(entry, "timestamp", ""))
        if latest_profile_trial_id is None:
            latest_profile_trial_id = getattr(entry, "trial_id", None)

        if latest_reference_trial_id is None and eval_details.get("seq_baseline_reference_draw"):
            latest_reference_trial_id = getattr(entry, "trial_id", None)
            latest_reference_ts = entry_ts
            latest_reference_fp = getattr(entry, "infra_fingerprint", None)
        elif latest_reference_trial_id is None:
            trials_since_reference += 1

    age_s = max(0.0, now - latest_reference_ts) if latest_reference_ts is not None else None
    stale_reference = age_s is not None and age_s > SEQ_BASELINE_REFERENCE_STALE_AFTER_S
    due = SEQ_BASELINE_REFRESH_CADENCE > 0 and (
        latest_reference_trial_id is None
        or trials_since_reference >= SEQ_BASELINE_REFRESH_CADENCE
        or stale_reference
    )
    reason = ""
    if latest_reference_trial_id is None:
        reason = "no marked seq baseline-reference draw"
    elif stale_reference:
        reason = f"baseline reference age {age_s:.0f}s exceeds stale threshold"
    elif trials_since_reference >= SEQ_BASELINE_REFRESH_CADENCE:
        reason = f"{trials_since_reference} trusted profile trials since baseline reference draw"

    regime: dict[str, Any] | None = None
    regime_due = False
    has_current = current_fingerprint is not None and bool(fingerprint_digest(current_fingerprint))
    if has_current and latest_reference_trial_id is None:
        regime = {"status": "NO_REFERENCE", "differing_components": []}
        regime_due = bool(regime_forcing)
        if regime_due:
            due = True
            reason = "no baseline reference draw for this tier (AP-55 bootstrap)"
    elif has_current:
        if not fingerprint_digest(latest_reference_fp):
            regime = {"status": "UNFINGERPRINTED", "differing_components": []}
        else:
            regime = compare_infra_fingerprints(
                current_fingerprint, latest_reference_fp,
                reference_label=f"seq_baseline_reference_{latest_reference_trial_id}",
            )
        regime_due = (
            bool(regime_forcing)
            and regime["status"] in (INFRA_NON_COMPARABLE, "UNFINGERPRINTED")
            and trials_since_reference >= max(0, AP55_REGIME_REFRESH_MIN_TRIALS)
        )
        if regime_due:
            due = True
            moved = ",".join(regime.get("differing_components") or []) or "no fingerprint"
            reason = f"infra regime changed since baseline reference draw ({moved})"

    return {
        "tier": int(tier),
        "trusted_profile_trials": trusted_profile_trials,
        "latest_profile_trial_id": latest_profile_trial_id,
        "latest_reference_trial_id": latest_reference_trial_id,
        "latest_reference_age_s": age_s,
        "latest_reference_ts": latest_reference_ts,
        "trials_since_reference": trials_since_reference,
        "stale_reference": bool(stale_reference),
        "due": bool(due),
        "reason": reason,
        "regime_status": regime["status"] if regime else "",
        "regime_due": bool(regime_due),
    }


def _seq_baseline_draw_action() -> dict[str, Any]:
    raw = os.environ.get("AUTOPILOT_SEQ_BASELINE_DRAW_ACTION", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("type"):
                return parsed
        except json.JSONDecodeError:
            log.warning("Invalid AUTOPILOT_SEQ_BASELINE_DRAW_ACTION JSON; using default")
    # n=14 avoids known blacklisted seed_batch sizes while still collecting a
    # small reference draw through the standard metric-collecting path.
    return {"type": "seed_batch", "n_questions": 14}


def _seed_action_candidates(
    preferred: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def add(candidate: dict[str, Any]) -> None:
        if candidate not in candidates:
            candidates.append(candidate)

    if isinstance(preferred, dict) and preferred.get("type") == "seed_batch":
        add(dict(preferred))
        if "suites" in preferred:
            no_suite = dict(preferred)
            no_suite.pop("suites", None)
            add(no_suite)

    for n_questions in FALLBACK_SEED_CANDIDATES:
        add({"type": "seed_batch", "n_questions": int(n_questions)})
        add(
            {
                "type": "seed_batch",
                "n_questions": int(n_questions),
                "suites": ["coder", "math"],
            }
        )
    return candidates


def _first_unblacklisted_seed_action(
    blacklist: list[dict[str, Any]],
    *,
    preferred: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    last_blocked = ""
    for candidate in _seed_action_candidates(preferred):
        blocked = check_blacklist(candidate, blacklist)
        if not blocked:
            return candidate, ""
        last_blocked = blocked
    return None, last_blocked or "all measured seed fallbacks are blacklisted"


def _first_dispatchable_seed_action(
    blacklist: list[dict[str, Any]],
    *,
    preferred: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str, dict[str, Any] | None]:
    """Return an unblocked seed action, or an audit-scoped retryable seed action."""
    last_blocked = ""
    for candidate in _seed_action_candidates(preferred):
        blocked = check_blacklist(candidate, blacklist)
        if not blocked:
            return candidate, "", None
        retry_meta = _p0_3_retryable_blacklist_match(candidate, blacklist)
        if retry_meta is not None:
            return candidate, blocked, retry_meta
        last_blocked = blocked
    return None, last_blocked or "all measured seed fallbacks are blacklisted", None


def _seed_fallback_exhaustion_reason(blacklist: list[dict[str, Any]]) -> str | None:
    """Return a planner-facing reason when every measured seed fallback is blocked."""
    candidate, reason = _first_unblacklisted_seed_action(blacklist)
    if candidate is not None:
        return None
    return reason or "all measured seed fallbacks are blacklisted"


def _critic_fallback_seed_skip(
    action: dict[str, Any],
    blacklist: list[dict[str, Any]],
) -> SkipOutcome | None:
    """Skip a critic seed fallback when the measured seed ladder is exhausted."""
    # 2026-08-04: `action` can be None here. The planner's deterministic guard sets it
    # to None to mean "revise/reject produced no materially different dispatch action"
    # (logged as "Planner fallback/degraded mode"), and the whole critic-repair chain
    # below assumed a dict. Trial 1470 died on `'NoneType' object has no attribute
    # 'get'`, the supervisor burned its three restarts, and AutoPilot stayed down.
    #
    # The surrounding loop already knew this: line ~7719 reads
    # `action.get("type") if isinstance(action, dict) else action`. The repair chain
    # never got the same treatment. There is nothing to repair when there is no
    # action, so pass it through untouched rather than inventing one.
    if not isinstance(action, dict):
        return None
    if action.get("type") != "seed_batch":
        return None
    reason = _seed_fallback_exhaustion_reason(blacklist)
    if not reason:
        return None
    return SkipOutcome(
        "skipped",
        f"critic fallback seed_batch unavailable: {reason}",
        "planner_coordinator",
    )


def _replace_exhausted_critic_seed_fallback(
    action: dict[str, Any],
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None = None,
    *,
    trial_counter: int = 0,
) -> tuple[dict[str, Any], dict[str, Any] | None, SkipOutcome | None]:
    """Replace an exhausted critic seed fallback with a metric-bearing action."""
    if not isinstance(action, dict):
        return action, rationale, None
    seed_skip = _critic_fallback_seed_skip(action, blacklist)
    if seed_skip is None:
        return action, rationale, None

    replacement, numeric_reason = _first_unblacklisted_numeric_trial_action(
        blacklist,
        trial_counter=trial_counter,
    )
    if replacement is None:
        return (
            action,
            rationale,
            SkipOutcome(
                seed_skip.status,
                f"{seed_skip.reason}; numeric fallback unavailable: {numeric_reason}",
                seed_skip.action_type,
            ),
        )

    next_rationale = {
        **(rationale or {}),
        "critic_seed_fallback_replaced": True,
        "critic_seed_fallback_unavailable_reason": seed_skip.reason,
        "critic_seed_fallback_replacement": dict(replacement),
    }
    log.warning(
        "Critic seed fallback is unavailable (%s); using metric-bearing numeric_trial fallback %s.",
        seed_skip.reason,
        json.dumps(replacement, default=str),
    )
    return replacement, next_rationale, None


def _first_unblacklisted_numeric_trial_action(
    blacklist: list[dict[str, Any]],
    *,
    trial_counter: int = 0,
) -> tuple[dict[str, Any] | None, str]:
    last_blocked = ""
    surfaces = _configured_numeric_surfaces()
    if not surfaces:
        return None, ("all quota numeric_trial surfaces are suppressed by planner conventions")
    for offset in range(len(surfaces)):
        surface = surfaces[(trial_counter + offset) % len(surfaces)]
        candidate = {"type": "numeric_trial", "surface": surface, "params": {}}
        blocked = check_blacklist(candidate, blacklist)
        if not blocked:
            return candidate, ""
        last_blocked = blocked
    return None, last_blocked or "all quota numeric_trial surfaces are blacklisted"


def _replace_blacklisted_seed_fallback(
    action: dict[str, Any],
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None = None,
    *,
    reason_label: str = "blacklisted fallback",
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if action.get("type") != "seed_batch":
        return action, rationale
    blocked = check_blacklist(action, blacklist)
    if not blocked:
        return action, rationale
    retry_meta = _p0_3_retryable_blacklist_match(action, blacklist)
    if retry_meta is not None:
        return action, _record_p0_3_reexploration_rationale(rationale, retry_meta)
    replacement, _ = _first_unblacklisted_seed_action(
        blacklist,
        preferred=action,
    )
    if replacement is None:
        log.warning(
            "%s seed action %s is blacklisted (%s), and no measured seed "
            "fallback remains unblocked.",
            reason_label,
            json.dumps(action, default=str),
            blocked,
        )
        return action, rationale
    next_rationale = {
        **(rationale or {}),
        "fallback_seed_reselected": True,
        "fallback_seed_reselected_reason": blocked,
        "fallback_seed_reselected_from": dict(action),
        "fallback_seed_reselected_context": reason_label,
    }
    log.warning(
        "%s seed action %s is blacklisted (%s); using measured fallback %s.",
        reason_label,
        json.dumps(action, default=str),
        blocked,
        json.dumps(replacement, default=str),
    )
    return replacement, next_rationale


def _replace_blacklisted_w8_candidate_action(
    action: dict[str, Any],
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None = None,
    *,
    trial_counter: int,
    w8_replay_pressure_text: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Keep W8 candidate generation from burning a trial on a known bad action."""
    if not _w8_replay_pressure_active(w8_replay_pressure_text):
        return action, rationale
    blocked = check_blacklist(action, blacklist)
    if not blocked:
        return action, rationale
    retry_meta = _p0_3_retryable_blacklist_match(action, blacklist)
    if retry_meta is not None:
        next_rationale = _record_p0_3_reexploration_rationale(rationale, retry_meta)
        log.warning(
            "W8 candidate action %s matches P0.3 retryable blacklist target %s; "
            "dispatching for audit-scoped re-exploration.",
            json.dumps(action, default=str),
            retry_meta.get("target_key", "unknown"),
        )
        return action, next_rationale

    replacement, fallback_reason = _first_unblacklisted_numeric_trial_action(
        blacklist,
        trial_counter=trial_counter,
    )
    if replacement is None:
        log.warning(
            "W8 candidate action %s is blacklisted (%s), but no replayable "
            "numeric fallback remains unblocked: %s",
            json.dumps(action, default=str),
            blocked,
            fallback_reason,
        )
        return action, rationale

    next_rationale = {
        **(rationale or {}),
        "w8_blacklisted_candidate_replaced": True,
        "w8_blacklisted_candidate_reason": blocked,
        "w8_blacklisted_candidate_original": dict(action),
        "w8_blacklisted_candidate_replacement": dict(replacement),
        "falsifier": (
            (rationale or {}).get("falsifier")
            or "W8 blacklisted-candidate replacement fails to produce replayable seq evidence"
        ),
    }
    log.warning(
        "W8 candidate action %s is blacklisted (%s); using replayable numeric_trial fallback %s.",
        json.dumps(action, default=str),
        blocked,
        json.dumps(replacement, default=str),
    )
    return replacement, next_rationale


def _replace_blacklisted_autonomous_action(
    action: dict[str, Any],
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Keep no-controller mode from burning trials on known-blocked/meta actions."""
    blocked = check_blacklist(action, blacklist)
    is_meta_noop = action.get("type") in META_NOOP_ACTIONS
    if blocked and not is_meta_noop:
        retry_meta = _p0_3_retryable_blacklist_match(action, blacklist)
        if retry_meta is not None:
            return action, _record_p0_3_reexploration_rationale(rationale, retry_meta)
    if not blocked and not is_meta_noop:
        return action, rationale
    reason = blocked or "autonomous meta action does not collect metrics"
    replacement, fallback_reason = _first_unblacklisted_seed_action(blacklist)
    if replacement is None:
        log.warning(
            "Autonomous action %s is not dispatchable (%s), and no measured seed "
            "fallback remains unblocked: %s",
            json.dumps(action, default=str),
            reason,
            fallback_reason,
        )
        return action, rationale
    next_rationale = {
        **(rationale or {}),
        "autonomous_blacklisted_replaced": True,
        "autonomous_blacklisted_reason": reason,
        "autonomous_blacklisted_from": dict(action),
    }
    log.warning(
        "Autonomous action %s is not dispatchable (%s); using measured seed fallback %s.",
        json.dumps(action, default=str),
        reason,
        json.dumps(replacement, default=str),
    )
    return replacement, next_rationale


def _seq_baseline_reference_block_key(reference: dict[str, Any]) -> str:
    ref_id = reference.get("latest_reference_trial_id")
    ref_token = "none" if ref_id is None else str(ref_id)
    return f"tier={reference.get('tier')}:reference={ref_token}"


def _seq_baseline_block_retry_due(
    blocked_state: dict[str, Any],
    *,
    trial_counter: int,
) -> bool:
    if SEQ_BASELINE_BLOCK_RETRY_CADENCE <= 0:
        return True
    try:
        blocked_trial = int(blocked_state.get("trial_id"))
    except (TypeError, ValueError):
        return True
    return trial_counter - blocked_trial >= SEQ_BASELINE_BLOCK_RETRY_CADENCE


def _maybe_force_seq_baseline_draw(
    action: dict[str, Any],
    *,
    state: dict[str, Any],
    journal: ExperimentJournal,
    tier: int,
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None,
    trial_counter: int,
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Force the 01c baseline-reference cadence when seq shadowing is enabled.

    AP-55 (b): a regime change since the last reference draw also makes a draw due.
    With seq disabled, ``AUTOPILOT_AP55_SEED_RERUN=1`` forces ONLY those
    regime-triggered re-runs (the cadence/staleness reasons stay seq-only).
    """
    ap55_force = _ap55_seed_rerun_forcing_enabled()
    if not enabled and not ap55_force:
        return action, rationale, None
    reference = _seq_baseline_reference_state(
        journal,
        tier=tier,
        # Only fingerprint when the knob can act on it: with the knob unset the
        # seq cadence is exactly what it was before AP-55.
        current_fingerprint=_infra_fingerprint_for_trial() if ap55_force else None,
        regime_forcing=ap55_force,
    )
    if not (reference["due"] if enabled else reference.get("regime_due")):
        return action, rationale, None
    if reference.get("regime_due") and not _ap55_seed_rerun_attempt_allowed(
        state, reference, trial_counter=trial_counter
    ):
        if not enabled:
            return action, rationale, None
        # Seq cadence reasons still apply on their own terms; drop only the AP-55 one.
        base = _seq_baseline_reference_state(journal, tier=tier)
        if not base["due"]:
            return action, rationale, None
        reference = base

    forced = _seq_baseline_draw_action()
    reference_key = _seq_baseline_reference_block_key(reference)
    blocked_state = state.get("seq_baseline_draw_blocked")
    if (
        isinstance(blocked_state, dict)
        and blocked_state.get("reference_key") == reference_key
        and blocked_state.get("action") == forced
    ):
        if not _seq_baseline_block_retry_due(
            blocked_state,
            trial_counter=trial_counter,
        ):
            return action, rationale, None
    blocked_reason = check_blacklist(forced, blacklist)
    retry_meta: dict[str, Any] | None = None
    if blocked_reason:
        fallback, fallback_reason, retry_meta = _first_dispatchable_seed_action(
            blacklist,
            preferred=forced,
        )
        if fallback is not None:
            forced = fallback
            blocked_reason = ""
        else:
            state["seq_baseline_draw_blocked"] = {
                "trial_id": trial_counter,
                "action": forced,
                "reason": fallback_reason or blocked_reason,
                "reference": reference,
                "reference_key": reference_key,
            }
            log.warning(
                "Seq baseline-reference draw due but forced action is blacklisted: %s",
                fallback_reason or blocked_reason,
            )
            return action, rationale, None
    next_rationale = dict(rationale or {})
    next_rationale["seq_baseline_reference_draw"] = True
    next_rationale["seq_baseline_reference_reason"] = reference["reason"]
    if retry_meta is not None:
        next_rationale = _record_p0_3_reexploration_rationale(next_rationale, retry_meta)
        next_rationale["seq_baseline_reference_retryable_blacklist"] = True
    state["seq_baseline_draw_blocked"] = None
    if reference.get("regime_due"):
        _ap55_note_seed_rerun_attempt(state, reference, trial_counter=trial_counter)
    state["seq_baseline_draw_forced"] = {
        "trial_id": trial_counter,
        "action": forced,
        "reference": reference,
    }
    log.info(
        "Forcing seq baseline-reference draw at trial %d: %s",
        trial_counter,
        reference["reason"],
    )
    return forced, next_rationale, reference


def _seq_rate_axis_is_advisory(seq: Mapping[str, Any] | None) -> bool:
    if not isinstance(seq, Mapping):
        return False
    return str(seq.get("rate_axis_mode") or "") == SEQ_P0_2_BRIDGE_MODE


def _seq_combined_e(seq: dict[str, Any] | None) -> float | None:
    if not isinstance(seq, dict):
        return None
    try:
        e_quality = float(seq["E_quality"])
    except (KeyError, TypeError, ValueError):
        return None
    if _seq_rate_axis_is_advisory(seq):
        return e_quality
    try:
        e_rate = float(seq["E_rate_noninf"])
    except (KeyError, TypeError, ValueError):
        return None
    return min(e_quality, e_rate)


def _seq_alpha_confirmation_blocked(alpha_wealth: Mapping[str, Any] | None) -> bool:
    if not isinstance(alpha_wealth, Mapping):
        return False
    charge_is_new = alpha_wealth.get(
        "candidate_confirmation_charge_is_new",
        alpha_wealth.get("candidate_is_new"),
    )
    return bool(charge_is_new) and (
        alpha_wealth.get("new_fingerprint_confirmations_allowed") is False
    )


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _seq_gate_reachability_report(
    journal: ExperimentJournal | None,
    *,
    tier: int,
    min_seq_rows: int | None = None,
    recent_window: int | None = None,
    max_rate_e: float | None = None,
) -> dict[str, Any]:
    """Model whether new seq candidates can currently reach the promotion gate.

    This is a loop-side burn guard only. It does not alter SafetyGate semantics
    or promotion thresholds; it prevents new promotion-dependent candidates from
    consuming eval wall when recent rate evidence says the rate axis is not
    growing toward confirmation.
    """
    min_rows = SEQ_GATE_PREFLIGHT_MIN_SEQ_ROWS if min_seq_rows is None else min_seq_rows
    window = SEQ_GATE_PREFLIGHT_RECENT_WINDOW if recent_window is None else recent_window
    max_rate = SEQ_GATE_PREFLIGHT_MAX_RATE_E if max_rate_e is None else max_rate_e
    bridge = seq_p0_2_bridge_status()
    rate_axis_advisory = bool(bridge["enabled"])

    entries = [
        entry
        for entry in _iter_journal_entries(journal)
        if not getattr(entry, "bug_corrupted_by", "")
        and getattr(entry, "outcome_status", "ok") not in {"invalid", "skipped"}
        and _entry_action_tier(entry) == tier
    ]
    seq_rows: list[dict[str, Any]] = []
    latest_by_candidate: dict[str, dict[str, Any]] = {}
    task_rates: list[tuple[int, float]] = []
    for entry in entries:
        eval_details = getattr(entry, "eval_details", {}) or {}
        if isinstance(eval_details, dict):
            # SEQ-B: the reachability report MUST measure the rate the same way the gate
            # does, or it certifies a comparator the gate never uses. This report existed
            # while the gate was unreachable and did not surface it.
            row = {
                "quality": getattr(entry, "quality", 0.0),
                "eval_details": eval_details,
            }
            task_rate = seq_task_rate_qph_from_row(row)
            if task_rate is not None and task_rate > 0.0:
                try:
                    task_rates.append((int(getattr(entry, "trial_id", 0) or 0), task_rate))
                except (TypeError, ValueError):
                    task_rates.append((0, task_rate))

        seq = getattr(entry, "seq", {}) or {}
        if not isinstance(seq, dict):
            continue
        if str(seq.get("core_id") or DEFAULT_EVIDENCE_CORE_ID) != DEFAULT_EVIDENCE_CORE_ID:
            continue
        candidate = str(seq.get("candidate") or "")
        if not candidate:
            continue
        seq_rows.append(seq)
        try:
            trial_id = int(getattr(entry, "trial_id", 0) or 0)
        except (TypeError, ValueError):
            trial_id = 0
        previous = latest_by_candidate.get(candidate)
        if previous is None or trial_id >= int(previous.get("trial_id", -1)):
            latest_by_candidate[candidate] = {"trial_id": trial_id, "seq": seq}

    e_rates: list[float] = []
    e_qualities: list[float] = []
    combined_values: list[float] = []
    confirmed_candidates = 0
    for row in latest_by_candidate.values():
        seq = row["seq"]
        if seq.get("confirmed") is True or str(seq.get("state") or "") == "confirmed":
            confirmed_candidates += 1
        try:
            e_rates.append(float(seq.get("E_rate_noninf")))
        except (TypeError, ValueError):
            pass
        try:
            e_qualities.append(float(seq.get("E_quality")))
        except (TypeError, ValueError):
            pass
        combined = _seq_combined_e(seq)
        if combined is not None:
            combined_values.append(float(combined))

    task_rates.sort(key=lambda item: item[0])
    recent_rates = [rate for _, rate in task_rates[-max(1, window) :]]
    baseline_rates = [rate for _, rate in task_rates[-SEQ_BASELINE_PROFILE_LIMIT:]]
    # SEQ-B: median on BOTH sides. This compared a MEDIAN of recent rates against a MEAN
    # of baseline rates — two different estimators of the same heavy-tailed quantity — so
    # the report's own `recent_rate_z` inherited the outlier inflation it was supposed to
    # detect, and `rate_axis_flat` read as an honest "candidates are slower".
    baseline_task_rate = _median(baseline_rates)
    recent_median_rate = _median(recent_rates)
    recent_rate_z: float | None = None
    if baseline_task_rate and recent_median_rate is not None:
        try:
            recent_rate_z = rate_noninferiority_z(
                recent_median_rate,
                baseline_task_rate,
                margin=SEQ_DEFAULT_POLICY.rate_noninferiority_margin,
            )
        except ValueError:
            recent_rate_z = None

    max_e_rate = max(e_rates) if e_rates else None
    max_e_quality = max(e_qualities) if e_qualities else None
    max_combined_e = max(combined_values) if combined_values else None
    insufficient = len(seq_rows) < max(0, min_rows) or recent_rate_z is None
    rate_axis_flat = (
        not insufficient
        and confirmed_candidates == 0
        and max_e_rate is not None
        and max_e_rate < min(SEQ_DEFAULT_POLICY.confirm_e, max_rate)
        and recent_rate_z <= 0.0
    )
    if insufficient:
        status = "insufficient_evidence"
    elif rate_axis_flat and not rate_axis_advisory:
        status = "rate_axis_unreachable"
    elif rate_axis_flat and rate_axis_advisory:
        status = "rate_axis_advisory_bridge"
    else:
        status = "reachable"
    return {
        "status": status,
        "ok_to_dispatch_candidates": status != "rate_axis_unreachable",
        "tier": tier,
        "policy_version": SEQ_DEFAULT_POLICY.version,
        "rate_axis_mode": SEQ_P0_2_BRIDGE_MODE if rate_axis_advisory else "binding_joint",
        "rate_axis_binding": not rate_axis_advisory,
        "p0_2_bridge": bridge,
        "confirm_e": SEQ_DEFAULT_POLICY.confirm_e,
        "promotion_required_e": SEQ_PROMOTION_FINAL_CONFIRM_E,
        "seq_rows": len(seq_rows),
        "candidate_count": len(latest_by_candidate),
        "confirmed_candidates": confirmed_candidates,
        "max_E_rate_noninf": None if max_e_rate is None else round(max_e_rate, 6),
        "max_E_quality": None if max_e_quality is None else round(max_e_quality, 6),
        "max_combined_E": None if max_combined_e is None else round(max_combined_e, 6),
        "baseline_task_rate": (
            None if baseline_task_rate is None else round(baseline_task_rate, 6)
        ),
        "recent_median_task_rate": (
            None if recent_median_rate is None else round(recent_median_rate, 6)
        ),
        "recent_rate_z": None if recent_rate_z is None else round(recent_rate_z, 6),
        "recent_window": window,
        "min_seq_rows": min_rows,
        "max_rate_e_for_block": max_rate,
    }


def _is_seq_promotion_dependent_action(action: Mapping[str, Any] | None) -> bool:
    if not isinstance(action, Mapping):
        return False
    return str(action.get("type") or "") in SEQ_PROMOTION_DEPENDENT_ACTIONS


def _maybe_defer_seq_unreachable_candidate_action(
    action: dict[str, Any],
    *,
    state: dict[str, Any],
    journal: ExperimentJournal,
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None,
    trial_counter: int,
    tier: int,
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Replace promotion-dependent work when seq preflight says it cannot promote."""
    if not enabled or not SEQ_GATE_PREFLIGHT_ENABLED:
        return action, rationale, None
    if not _is_seq_promotion_dependent_action(action):
        return action, rationale, None

    candidate_inputs = _seq_inputs_for_trial(
        journal=journal,
        action=action,
        tier=tier,
        quality_exclude_before_ts=_quality_exclude_before_ts_from_state(state),
    )
    alpha_wealth = candidate_inputs.get("alpha_wealth") or {}
    alpha_dispatch_allowed = alpha_wealth.get(
        "new_fingerprint_dispatch_allowed",
        alpha_wealth.get("new_fingerprint_confirmations_allowed"),
    )
    alpha_blocked = bool(alpha_wealth.get("candidate_is_new")) and alpha_dispatch_allowed is False
    reachability = _seq_gate_reachability_report(journal, tier=tier)
    should_defer = alpha_blocked or not bool(reachability.get("ok_to_dispatch_candidates", True))
    if not should_defer:
        state["seq_gate_reachability_preflight"] = {
            "trial_id": trial_counter,
            "status": "passed",
            "action": dict(action),
            "reachability": reachability,
            "alpha_wealth": alpha_wealth,
        }
        return action, rationale, None

    reason = "alpha_wealth_exhausted" if alpha_blocked else str(reachability["status"])
    payload = {
        "trial_id": trial_counter,
        "status": "blocked_unreachable",
        "reason": reason,
        "original_action": dict(action),
        "replacement_action": None,
        "fallback_reason": (
            "seq preflight blocked promotion-dependent action; seed fallback disabled "
            "because it cannot repair exhausted alpha wealth or a rate-axis-unreachable gate"
        ),
        "retryable_blacklist_target": None,
        "reachability": reachability,
        "alpha_wealth": alpha_wealth,
    }
    state["seq_gate_reachability_preflight"] = payload
    log.warning(
        "Seq gate preflight blocking promotion-dependent action at trial %d: %s (%s); "
        "not substituting seed fallback",
        trial_counter,
        json.dumps(action, default=str),
        reason,
    )
    return action, rationale, payload


def _seq_gate_preflight_dispatch_block_reason(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    if str(payload.get("status") or "") not in {
        "blocked_unreachable",
        "blocked_no_fallback",
    }:
        return ""
    reason = str(payload.get("reason") or "unknown")
    return f"seq_gate_preflight_{reason}"


def _seq_promotion_delta_ci(seq: dict[str, Any] | None) -> dict[str, Any]:
    """Return the Phase-2.4 one-sided non-regression CI for a promotion eval."""
    if not isinstance(seq, dict):
        return {
            "status": "missing",
            "excludes_regression": False,
            "reason": "missing seq block",
        }
    try:
        mean_delta = float(seq["z"])
        n_eff = int(seq["r_eff"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "missing",
            "excludes_regression": False,
            "reason": "missing z/r_eff promotion-delta evidence",
        }
    if n_eff <= 0:
        return {
            "status": "insufficient",
            "n_eff": n_eff,
            "mean_delta": round(mean_delta, 6),
            "excludes_regression": False,
            "reason": "no effective paired questions in promotion eval",
        }

    alpha = max(1e-12, min(0.5, SEQ_PROMOTION_DELTA_CI_ALPHA))
    half_width = math.sqrt(2.0 * math.log(1.0 / alpha) / n_eff)
    lower_bound = mean_delta - half_width
    return {
        "status": "ok",
        "confidence": round(1.0 - alpha, 6),
        "alpha": alpha,
        "n_eff": n_eff,
        "mean_delta": round(mean_delta, 6),
        "half_width": round(half_width, 6),
        "lower_bound": round(lower_bound, 6),
        "excludes_regression": lower_bound >= 0.0,
    }


def _annotate_seq_promotion_finalization(
    seq: dict[str, Any] | None,
    *,
    baseline_reference: dict[str, Any] | None,
    is_fresh_eval: bool,
    fresh_eval_context: dict[str, Any] | None = None,
) -> bool | None:
    """Add baseline-promotion finalization metadata to a seq journal block."""
    if not isinstance(seq, dict):
        return None
    reference = baseline_reference or {}
    combined_e = _seq_combined_e(seq)
    delta_ci = _seq_promotion_delta_ci(seq) if is_fresh_eval else None
    stale_reference = bool(reference.get("stale_reference"))
    seq["baseline_reference"] = {
        "tier": reference.get("tier"),
        "latest_reference_trial_id": reference.get("latest_reference_trial_id"),
        "latest_reference_age_s": reference.get("latest_reference_age_s"),
        "trials_since_reference": reference.get("trials_since_reference"),
        "stale_reference": stale_reference,
    }
    seq["baseline_reference_state"] = "stale-reference" if stale_reference else "fresh"
    seq["baseline_promotion_required_E"] = SEQ_PROMOTION_FINAL_CONFIRM_E
    seq["baseline_promotion_fresh_eval"] = bool(is_fresh_eval)
    seq["baseline_promotion_rate_axis_mode"] = seq.get("rate_axis_mode") or "binding_joint"
    seq["baseline_promotion_combined_E_mode"] = (
        "quality_only_rate_advisory"
        if _seq_rate_axis_is_advisory(seq)
        else "joint_min_quality_rate"
    )
    if combined_e is not None:
        seq["baseline_promotion_combined_E"] = round(combined_e, 6)
    if delta_ci is not None:
        seq["baseline_promotion_delta_ci"] = delta_ci
    if fresh_eval_context:
        seq["baseline_promotion_fresh_eval_for"] = {
            "candidate": fresh_eval_context.get("candidate"),
            "source_trial_id": fresh_eval_context.get("source_trial_id"),
        }
    finalized = (
        bool(seq.get("confirmed"))
        and bool(is_fresh_eval)
        and not stale_reference
        and combined_e is not None
        and combined_e >= SEQ_PROMOTION_FINAL_CONFIRM_E
        and delta_ci is not None
        and bool(delta_ci.get("excludes_regression"))
    )
    seq["baseline_promotion_finalized"] = finalized
    return finalized


def _maybe_force_seq_promotion_fresh_eval(
    action: dict[str, Any],
    *,
    state: dict[str, Any],
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None,
    trial_counter: int,
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Force one large fresh eval for a pending seq-confirmed promotion."""
    pending = state.get("seq_pending_promotion_fresh_eval")
    if not enabled or not isinstance(pending, dict):
        return action, rationale, None
    attempts = int(pending.get("attempts", 0) or 0)
    if attempts >= 1:
        return action, rationale, None
    tier = max(MIN_FRONTIER_EVAL_TIER, int(pending.get("tier") or SEQ_PROMOTION_FRESH_EVAL_TIER))
    candidate_action = pending.get("action")
    replay_blocker = _seq_promotion_replay_blocker(candidate_action)
    if replay_blocker:
        state.pop("seq_pending_promotion_fresh_eval", None)
        state.pop("_seq_promotion_candidate_replay", None)
        state["seq_last_promotion_blocked"] = {
            "trial_id": trial_counter,
            "candidate": pending.get("candidate"),
            "source_trial_id": pending.get("source_trial_id"),
            "reason": replay_blocker,
        }
        log.warning(
            "Seq promotion fresh eval blocked for candidate %s: %s",
            pending.get("candidate"),
            replay_blocker,
        )
        return action, rationale, None
    forced = {"type": "deep_eval", "tier": tier}
    blocked_reason = check_blacklist(forced, blacklist)
    if blocked_reason:
        pending["attempts"] = attempts + 1
        pending["blocked_reason"] = blocked_reason
        pending["blocked_at_trial"] = trial_counter
        pending["blocked_action"] = forced
        state["seq_pending_promotion_fresh_eval"] = pending
        log.warning(
            "Seq promotion fresh eval due but forced action is blacklisted: %s",
            blocked_reason,
        )
        return action, rationale, None

    pending["attempts"] = attempts + 1
    pending["forced_trial_id"] = trial_counter
    state["seq_pending_promotion_fresh_eval"] = pending
    state["_seq_promotion_candidate_replay"] = {
        "trial_id": trial_counter,
        "candidate": pending.get("candidate"),
        "source_trial_id": pending.get("source_trial_id"),
        "action": candidate_action,
    }
    next_rationale = dict(rationale or {})
    next_rationale["seq_promotion_fresh_eval"] = True
    next_rationale["seq_promotion_candidate"] = pending.get("candidate")
    log.info(
        "Forcing seq promotion fresh eval at trial %d for candidate %s",
        trial_counter,
        pending.get("candidate"),
    )
    return forced, next_rationale, dict(pending)


def _seq_promotion_replay_blocker(action: Any, served_content: Any = None) -> str:
    """Return why a seq-promotion candidate cannot be replayed for fresh eval.

    AP-9 still guards new planner-proposed numeric_trial actions before dispatch.
    W8 replay is different: a materialized NumericSwarm trial may contain several
    applied params, but it is a single recorded candidate being re-measured.

    Operator decision 2026-09-16: a prompt / code / GEPA mutation is replayable only with a
    served-file identity (``served_content``, the sha of what the candidate trial served).
    A replay re-measures that same content; callers that pass no identity keep mutations
    blocked (the seq fresh-eval and replay-selection paths are unchanged).
    """
    if not isinstance(action, dict):
        return "candidate action is missing or not an object"
    action_type = str(action.get("type") or "")
    if action_type in CONTENT_IDENTIFIED_ACTION_TYPES:
        if served_content_files(served_content) is None:
            return (
                f"candidate {action_type} has no served-file identity (served_content sha); "
                "its content cannot be re-served for validation"
            )
        return ""
    if action_type == "numeric_trial":
        params = action.get("params")
        if not isinstance(params, dict) or not params:
            return "candidate numeric_trial lacks replayable applied params"
        return ""
    elif action_type == "structural_experiment":
        flags = action.get("flags")
        if not isinstance(flags, dict) or not flags:
            return "candidate structural_experiment lacks replayable flags"
        if len(flags) > 1:
            return (
                f"candidate structural_experiment changes {len(flags)} flags at once "
                f"({list(flags.keys())}); limit to 1 for clean attribution"
            )
        for key, value in flags.items():
            if not isinstance(key, str) or not isinstance(value, bool):
                return "candidate structural_experiment flags must map string names to booleans"
    else:
        return f"candidate action type is not replayable: {action_type or 'unknown'}"

    return ""


def _multitier_validation_served_content(
    state: dict[str, Any], context: Mapping[str, Any], trial_counter: int
) -> dict[str, Any] | None:
    """A mutation candidate's validation stage: the served files, hashed right after eval.

    The due-action check verified the content before the forced eval; this re-hash proves
    it was still served when the eval finished. A change in between rejects the candidate
    (and the stage row carries no identity, so it can never count as a reproduction).
    """
    expected = context.get("candidate_served_content")
    if not expected or context.get("stage") in {None, "rollback"}:
        return None
    pending = state.get(MULTITIER_PENDING_STATE_KEY)
    mismatch = _multitier_served_content_mismatch({"candidate_served_content": expected})
    if mismatch:
        if isinstance(pending, dict):
            _reject_multitier_candidate(
                state, pending, reason=f"during validation: {mismatch}", trial_counter=trial_counter
            )
        log.error("Trial %d: multitier mutation candidate rejected — %s", trial_counter, mismatch)
        return None
    return _multitier_served_content_now(expected)


def _multitier_served_content_now(expected: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Hash the files a mutation candidate served, as they are on disk now (eval time)."""
    files = served_content_files(expected) if isinstance(expected, Mapping) else None
    if files is None:
        return None
    return served_content_record([ORCH_ROOT / path for path in files], root=ORCH_ROOT)


def _multitier_served_content_mismatch(pending: Mapping[str, Any]) -> str:
    """Why the currently served files are not the candidate's content ("" when they are)."""
    expected = pending.get("candidate_served_content")
    if not expected:
        return ""
    want = served_content_files(expected)
    now = _multitier_served_content_now(expected)
    if want is None or now is None:
        return (
            "mutation candidate content cannot be re-served: a served file is missing or "
            "unreadable (reverted or moved since staging)"
        )
    if now["files"] != want:
        changed = sorted(p for p in want if now["files"].get(p) != want[p])
        return (
            "mutation candidate content is no longer served: file sha changed since staging "
            f"({', '.join(changed)}); it was reverted, re-mutated or auto-committed away"
        )
    return ""


def _mutation_restore_is_complete(restore: Any, served_content: Any) -> bool:
    files = served_content_files(served_content)
    if not isinstance(restore, Mapping) or files is None:
        return False
    contents = restore.get("preimage")
    return (
        restore.get("kind") in {"prompt", "code"}
        and bool(restore.get("file"))
        and isinstance(contents, str)
        and bool(restore.get("preimage_sha256"))
    )


def _multitier_baseline_for(state: Mapping[str, Any], tier: int) -> Mapping[str, Any] | None:
    bundle = state.get(MULTITIER_BASELINE_STATE_KEY)
    if not isinstance(bundle, Mapping):
        return None
    if str(bundle.get("policy_version") or "") != MULTITIER_POLICY_VERSION:
        return None
    tiers = bundle.get("tiers")
    if not isinstance(tiers, Mapping):
        return None
    baseline = tiers.get(str(int(tier)), tiers.get(int(tier)))
    return baseline if isinstance(baseline, Mapping) else None


def _strict_baseline_quality(baseline: Any, tier: int) -> float | None:
    """Read a same-tier baseline without permitting legacy T1 fallback.

    The small compatibility fallback keeps older test doubles working while
    preserving strict behavior for every non-canonical tier.
    """
    if baseline is None:
        return None
    try:
        return baseline.quality_for_tier(tier, strict=True)
    except TypeError:
        if int(tier) != DEFAULT_FRONTIER_TIER:
            return None
        return baseline.quality_for_tier(tier)


def _multitier_required_baselines_ready(state: Mapping[str, Any]) -> tuple[bool, str]:
    missing = [
        tier for tier in REQUIRED_VALIDATION_TIERS if _multitier_baseline_for(state, tier) is None
    ]
    if missing:
        return False, "missing matched incumbent baseline(s): " + ", ".join(
            f"T{tier}" for tier in missing
        )
    return True, "ready"


def _multitier_effective_action(
    action: dict[str, Any], context: Mapping[str, Any] | None
) -> dict[str, Any]:
    if isinstance(context, Mapping) and context.get("stage") == "rollback":
        return action
    candidate_action = context.get("candidate_action") if isinstance(context, Mapping) else None
    return dict(candidate_action) if isinstance(candidate_action, Mapping) else action


def _multitier_candidate_is_eligible(
    *,
    state: dict[str, Any],
    gate: SafetyGate,
    action: dict[str, Any],
    eval_result: EvalResult,
    verdict: Any,
    pareto_status: str,
    served_content: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    if not MULTITIER_PROMOTION_ENABLED:
        return False, "policy disabled"
    if isinstance(state.get(MULTITIER_PENDING_STATE_KEY), dict):
        return False, "another candidate is already pending"
    if state.get("multitier_rollback_stalled"):
        return False, "a multitier rollback is stalled; operator action required"
    if int(getattr(eval_result, "tier", 0) or 0) != DEFAULT_FRONTIER_TIER:
        return False, "not a T1 screening result"
    ready, reason = _multitier_required_baselines_ready(state)
    if not ready:
        return False, reason
    blocker = _seq_promotion_replay_blocker(action, served_content)
    if blocker:
        return False, blocker
    action_type = str(action.get("type") or "")
    if action_type in CONTENT_IDENTIFIED_ACTION_TYPES:
        restore = (served_content or {}).get("restore") if isinstance(served_content, Mapping) else None
        if not _mutation_restore_is_complete(restore, served_content):
            return False, f"candidate {action_type} lacks an exact restore preimage"
    elif action_type == "numeric_trial":
        preimage = action.get("_multitier_restore_preimage")
        if (
            not isinstance(preimage, dict)
            or preimage.get("status") != "ok"
            or preimage.get("reversible") is not True
        ):
            return False, "candidate numeric_trial lacks an exact reversible runtime preimage"
    elif action_type == "structural_experiment":
        restore_flags = action.get("_multitier_restore_flags")
        if not isinstance(restore_flags, dict) or not restore_flags:
            prior_flags = (getattr(eval_result, "details", {}) or {}).get(
                "flag_prior_values"
            )
            if isinstance(prior_flags, dict) and prior_flags:
                action["_multitier_restore_flags"] = dict(prior_flags)
                restore_flags = action["_multitier_restore_flags"]
        if not isinstance(restore_flags, dict) or not restore_flags:
            return False, "candidate structural_experiment lacks exact restore flags"
    if not bool(verdict):
        return False, "T1 SafetyGate failed"
    if pareto_status != "frontier":
        return False, "candidate is not on the T1 frontier"
    baseline_quality = _strict_baseline_quality(gate.baseline, DEFAULT_FRONTIER_TIER)
    if baseline_quality is None or eval_result.quality <= baseline_quality:
        return False, "candidate does not improve the strict T1 incumbent"
    if gate.use_sequential:
        seq = getattr(verdict, "seq", None)
        if not isinstance(seq, dict) or not bool(seq.get("confirmed")):
            return False, "T1 sequential confirmation is still accumulating"
    return True, "ready"


def _start_multitier_validation(
    state: dict[str, Any],
    *,
    action: dict[str, Any],
    eval_result: EvalResult,
    verdict: Any,
    trial_counter: int,
    served_content: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    seq = getattr(verdict, "seq", None)
    candidate_action = dict(action)
    if str(candidate_action.get("type") or "") == "structural_experiment":
        prior_flags = (getattr(eval_result, "details", {}) or {}).get("flag_prior_values")
        if isinstance(prior_flags, dict) and prior_flags:
            candidate_action["_multitier_restore_flags"] = dict(prior_flags)
    pending = {
        "policy_version": MULTITIER_POLICY_VERSION,
        "status": "pending",
        "candidate": _config_fingerprint(action),
        "candidate_action": candidate_action,
        "source_trial_id": int(trial_counter),
        "source_t1_quality": float(eval_result.quality),
        "next_tier": int(REQUIRED_VALIDATION_TIERS[0]),
        "required_tiers": list(REQUIRED_VALIDATION_TIERS),
        "attempts_by_tier": {str(tier): [] for tier in REQUIRED_VALIDATION_TIERS},
        "verdicts_by_tier": {},
        "higher_tier_improvements": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seq_context": dict(seq) if isinstance(seq, dict) else None,
        "final_t1_attempts": 0,
    }
    files = served_content_files(served_content) if isinstance(served_content, Mapping) else None
    if files is not None:
        # Mutation candidates: the exact content every validation stage must re-serve, and
        # the preimage a rejection restores. Kept on the pending record, never on the
        # action, so action fingerprints and journal rows are unchanged.
        pending["candidate_served_content"] = {"files": files}
        restore = served_content.get("restore")
        if isinstance(restore, Mapping):
            pending["candidate_restore"] = dict(restore)
    state[MULTITIER_PENDING_STATE_KEY] = pending
    state["multitier_last_event"] = {
        "event": "candidate_staged",
        "trial_id": trial_counter,
        "candidate": pending["candidate"],
        "next_tier": pending["next_tier"],
    }
    return pending


def _maybe_force_multitier_due_action(
    *,
    state: dict[str, Any],
    blacklist: list[dict[str, Any]],
    trial_counter: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    if not MULTITIER_PROMOTION_ENABLED:
        return None, None, None
    pending = state.get(MULTITIER_PENDING_STATE_KEY)
    if state.get("multitier_rollback_stalled"):
        # Review fix 6: a rollback that failed MULTITIER_ROLLBACK_MAX_FAILURES times is not
        # re-forced every trial (a stall); it waits for an operator (alarm already raised).
        return None, None, None
    if state.get("multitier_rollback_pending") and isinstance(pending, dict):
        return (
            {"type": "rollback", "to_checkpoint": "production_best"},
            {
                "multitier_rollback": True,
                "reason": pending.get("blocked_reason") or "candidate rejected",
            },
            {"stage": "rollback", **_public_pending_copy(pending)},
        )
    if not isinstance(pending, dict) or pending.get("status") != "pending":
        return None, None, None
    candidate_action = pending.get("candidate_action")
    blocker = _seq_promotion_replay_blocker(
        candidate_action, pending.get("candidate_served_content")
    )
    if not blocker:
        blocker = _multitier_served_content_mismatch(pending)
    if blocker:
        pending["status"] = "blocked"
        pending["blocked_reason"] = blocker
        state["multitier_last_rejected"] = _public_pending_copy(pending)
        state["multitier_rollback_pending"] = True
        return (
            {"type": "rollback", "to_checkpoint": "production_best"},
            {
                "multitier_rollback": True,
                "reason": blocker,
            },
            {"stage": "rollback", **_public_pending_copy(pending)},
        )

    next_tier = pending.get("next_tier")
    if next_tier == "final_t1":
        tier = DEFAULT_FRONTIER_TIER
    else:
        try:
            tier = int(next_tier)
        except (TypeError, ValueError):
            pending["status"] = "blocked"
            pending["blocked_reason"] = f"invalid next_tier={next_tier!r}"
            state["multitier_last_rejected"] = _public_pending_copy(pending)
            state["multitier_rollback_pending"] = True
            return (
                {"type": "rollback", "to_checkpoint": "production_best"},
                {
                    "multitier_rollback": True,
                    "reason": pending["blocked_reason"],
                },
                {"stage": "rollback", **_public_pending_copy(pending)},
            )

    forced = {"type": "deep_eval", "tier": tier}
    blocked_reason = check_blacklist(forced, blacklist)
    if blocked_reason:
        pending["status"] = "blocked"
        pending["blocked_reason"] = f"multitier validation action blacklisted: {blocked_reason}"
        state["multitier_last_rejected"] = _public_pending_copy(pending)
        state["multitier_rollback_pending"] = True
        return (
            {"type": "rollback", "to_checkpoint": "production_best"},
            {
                "multitier_rollback": True,
                "reason": pending["blocked_reason"],
            },
            {"stage": "rollback", **_public_pending_copy(pending)},
        )

    context = {
        "policy_version": MULTITIER_POLICY_VERSION,
        "stage": "final_t1" if next_tier == "final_t1" else f"tier_{tier}",
        "tier": tier,
        "candidate": pending.get("candidate"),
        "candidate_action": dict(candidate_action),
        "source_trial_id": pending.get("source_trial_id"),
        "trial_id": trial_counter,
        "seq_context": pending.get("seq_context"),
    }
    if pending.get("candidate_served_content"):
        context["candidate_served_content"] = dict(pending["candidate_served_content"])
    state["_multitier_candidate_validation"] = dict(context)
    return (
        forced,
        {
            "multitier_validation": True,
            "multitier_stage": context["stage"],
            "multitier_candidate": context["candidate"],
        },
        context,
    )


def _reject_multitier_candidate(
    state: dict[str, Any], pending: dict[str, Any], *, reason: str, trial_counter: int
) -> None:
    pending["status"] = "rejected"
    pending["blocked_reason"] = reason
    pending["terminal_trial_id"] = trial_counter
    state["multitier_last_rejected"] = _public_pending_copy(pending)
    state["multitier_rollback_pending"] = True
    state["multitier_last_event"] = {
        "event": "candidate_rejected",
        "trial_id": trial_counter,
        "candidate": pending.get("candidate"),
        "reason": reason,
    }


def _record_multitier_validation_result(
    state: dict[str, Any],
    *,
    context: Mapping[str, Any] | None,
    eval_result: EvalResult,
    verdict: Any,
    trial_counter: int,
) -> dict[str, Any] | None:
    if not isinstance(context, Mapping) or context.get("stage") in {None, "rollback"}:
        return None
    pending = state.get(MULTITIER_PENDING_STATE_KEY)
    if not isinstance(pending, dict):
        return None
    if pending.get("status") not in {None, "pending"}:
        return None  # already rejected this trial (e.g. its served content changed)
    if pending.get("candidate") != context.get("candidate"):
        _reject_multitier_candidate(
            state,
            pending,
            reason="multitier candidate identity changed during validation",
            trial_counter=trial_counter,
        )
        return None
    if context.get("stage") == "final_t1":
        pending["final_t1_attempts"] = int(pending.get("final_t1_attempts") or 0) + 1
        if not bool(verdict):
            _reject_multitier_candidate(
                state,
                pending,
                reason="final T1 promotion evaluation failed SafetyGate",
                trial_counter=trial_counter,
            )
        return {"status": "final_t1", "attempts": pending["final_t1_attempts"]}

    tier = int(context.get("tier") or 0)
    attempts_by_tier = pending.setdefault("attempts_by_tier", {})
    attempts = attempts_by_tier.setdefault(str(tier), [])
    attempts.append(build_tier_candidate_evidence(eval_result))
    baseline = _multitier_baseline_for(state, tier)
    tier_verdict = evaluate_tier_validation(attempts, baseline, tier=tier)
    verdict_payload = tier_verdict.to_dict()
    if not bool(verdict):
        verdict_payload["status"] = "regression"
        verdict_payload["reason"] = "same-tier SafetyGate failed: " + "; ".join(
            getattr(verdict, "violations", [])[:3]
        )
    pending.setdefault("verdicts_by_tier", {})[str(tier)] = verdict_payload
    state["multitier_last_event"] = {
        "event": "tier_validation",
        "trial_id": trial_counter,
        "candidate": pending.get("candidate"),
        "tier": tier,
        "status": verdict_payload["status"],
    }

    if verdict_payload["status"] == "pass":
        if verdict_payload.get("improvement"):
            pending.setdefault("higher_tier_improvements", []).append(tier)
        remaining = [required for required in REQUIRED_VALIDATION_TIERS if required > tier]
        pending["next_tier"] = remaining[0] if remaining else "final_t1"
    elif verdict_payload["status"] == "inconclusive" and len(attempts) < max(
        1, _multitier_attempt_cap()
    ):
        pending["next_tier"] = tier
    else:
        _reject_multitier_candidate(
            state,
            pending,
            reason=verdict_payload["reason"],
            trial_counter=trial_counter,
        )
    return verdict_payload


def _finish_multitier_promotion(
    state: dict[str, Any],
    *,
    baseline_update: Any | None,
    trial_counter: int,
) -> None:
    pending = state.get(MULTITIER_PENDING_STATE_KEY)
    if not isinstance(pending, dict) or pending.get("next_tier") != "final_t1":
        return
    if baseline_update is not None and bool(getattr(baseline_update, "updated", False)):
        pending["status"] = "accepted"
        pending["terminal_trial_id"] = trial_counter
        state["multitier_last_accepted"] = _public_pending_copy(pending)
        state["multitier_last_event"] = {
            "event": "candidate_accepted",
            "trial_id": trial_counter,
            "candidate": pending.get("candidate"),
            "higher_tier_improvements": list(pending.get("higher_tier_improvements") or []),
        }
        state["multitier_production_checkpoint_due"] = {
            "trial_id": trial_counter,
            "candidate": pending.get("candidate"),
            "candidate_action": dict(pending.get("candidate_action") or {}),
            "policy_version": MULTITIER_POLICY_VERSION,
        }
        state.pop(MULTITIER_PENDING_STATE_KEY, None)
        return
    attempts = int(pending.get("final_t1_attempts") or 0)
    if attempts >= _multitier_attempt_cap():
        _reject_multitier_candidate(
            state,
            pending,
            reason=(
                "final T1 baseline promotion did not clear existing promotion gates: "
                + str(getattr(baseline_update, "reason", "no baseline update"))
            ),
            trial_counter=trial_counter,
        )


def _seq_replay_terminal_keep_revert_decision(entry: Any) -> bool:
    """Return whether AP-24 made this latest seq row terminal for replay."""
    decision = str(getattr(entry, "keep_revert_decision", "") or "").strip()
    if decision == "revert":
        return True
    if decision != "excluded":
        return False
    seq = getattr(entry, "seq", {}) or {}
    if not isinstance(seq, dict) or str(seq.get("state") or "") != "accumulating":
        return True
    return bool(str(getattr(entry, "failure_analysis", "") or "").strip())


def _seq_candidate_replay_payload(
    journal: ExperimentJournal,
    *,
    tier: int,
    min_combined_e: float = SEQ_CANDIDATE_REPLAY_MIN_COMBINED_E,
    min_quality_e: float = SEQ_CANDIDATE_REPLAY_MIN_QUALITY_E,
    max_k: int = SEQ_CANDIDATE_REPLAY_MAX_K,
    min_k: int = SEQ_CANDIDATE_REPLAY_MIN_K,
    core_id: str = DEFAULT_EVIDENCE_CORE_ID,
) -> dict[str, Any] | None:
    """Pick an accumulating replayable candidate that needs more W8 power."""
    best: dict[str, Any] | None = None
    try:
        entries = journal.entries_with_supersessions()
    except Exception:  # noqa: BLE001 - diagnostics should not halt AutoPilot
        return None
    latest_by_candidate: dict[str, Any] = {}
    for entry in entries:
        if getattr(entry, "bug_corrupted_by", ""):
            continue
        if getattr(entry, "outcome_status", "ok") in {"invalid", "skipped"}:
            continue
        try:
            if int(getattr(entry, "tier", -1)) != int(tier):
                continue
        except (TypeError, ValueError):
            continue
        seq = getattr(entry, "seq", {}) or {}
        if not isinstance(seq, dict):
            continue
        if str(seq.get("core_id") or core_id) != core_id:
            continue
        candidate = str(seq.get("candidate") or "")
        if not candidate:
            continue
        previous = latest_by_candidate.get(candidate)
        previous_trial = int(getattr(previous, "trial_id", -1) or -1) if previous else -1
        trial_id = int(getattr(entry, "trial_id", -1) or -1)
        if trial_id >= previous_trial:
            latest_by_candidate[candidate] = entry

    for entry in latest_by_candidate.values():
        if _seq_replay_terminal_keep_revert_decision(entry):
            # AP-24 verdict-failed rows keep outcome_status="ok"; replay must
            # honor the explicit keep/revert decision or W8 can force
            # already-failed configs back into the measurement loop. Benign
            # learning exclusions remain replayable while still accumulating.
            continue
        seq = getattr(entry, "seq", {}) or {}
        if seq.get("confirmed") is True or str(seq.get("state") or "") != "accumulating":
            continue
        candidate = str(seq.get("candidate") or "")
        try:
            k = int(seq.get("k") or 0)
        except (TypeError, ValueError):
            k = 0
        if max_k > 0 and k >= max_k:
            continue
        action = getattr(entry, "config_snapshot", {}) or {}
        if _seq_promotion_replay_blocker(action):
            continue
        combined = _seq_combined_e(seq)
        if combined is None:
            combined = seq.get("baseline_promotion_combined_E")
        try:
            combined_f = float(combined)
        except (TypeError, ValueError):
            continue
        try:
            e_quality = float(seq.get("E_quality") or 0.0)
        except (TypeError, ValueError):
            e_quality = 0.0
        # GRACE PERIOD (2026-08-04): under min_k, accumulated E is not yet evidence of
        # anything, so it must not be used to abandon the candidate. Applying these
        # thresholds at k=1 stranded 89 of 141 candidates on a single noisy sample whose
        # median missed the cut by 0.001 (0.999 vs 1.0). Above min_k the filters apply
        # unchanged, so a genuinely dead candidate is still dropped — just not before it
        # has been measured enough times for "dead" to mean something.
        within_grace = min_k > 0 and k < min_k
        if not within_grace:
            if combined_f < min_combined_e:
                continue
            if e_quality < min_quality_e:
                continue
        try:
            e_rate = float(seq.get("E_rate_noninf") or 0.0)
        except (TypeError, ValueError):
            e_rate = 0.0
        payload = {
            "candidate": candidate,
            "source_trial_id": int(getattr(entry, "trial_id", 0) or 0),
            "action": dict(action),
            "k": k,
            "combined_E": round(combined_f, 6),
            "E_quality": round(e_quality, 6),
            "E_rate_noninf": round(e_rate, 6),
        }
        key = (-k, combined_f, e_quality, e_rate, payload["source_trial_id"])
        best_key = (
            (
                -int(best.get("k", 0)),
                float(best.get("combined_E", 0.0)),
                float(best.get("E_quality", 0.0)),
                float(best.get("E_rate_noninf", 0.0)),
                int(best.get("source_trial_id", 0)),
            )
            if best
            else None
        )
        if best is None or key > best_key:
            best = payload
    return best


def _maybe_force_seq_candidate_replay(
    action: dict[str, Any],
    *,
    state: dict[str, Any],
    journal: ExperimentJournal,
    tier: int,
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None,
    trial_counter: int,
    enabled: bool,
    lab: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Replay an accumulating W8 candidate before it is abandoned underpowered."""
    if not enabled or not SEQ_CANDIDATE_REPLAY_ENABLED:
        return action, rationale, None
    if isinstance(state.get("seq_pending_promotion_fresh_eval"), dict):
        return action, rationale, None
    payload = _seq_candidate_replay_payload(journal, tier=tier)
    if payload is None:
        return action, rationale, None
    forced = dict(payload["action"])
    blocked_reason = check_blacklist(forced, blacklist)
    if not blocked_reason and forced.get("type") == "structural_experiment":
        blocked_reason = _structural_noop_reason(forced.get("flags", {}), lab)
    if blocked_reason:
        state["seq_candidate_replay_blocked"] = {
            "trial_id": trial_counter,
            "candidate": payload["candidate"],
            "source_trial_id": payload["source_trial_id"],
            "reason": blocked_reason,
            "action": forced,
            "combined_E": payload["combined_E"],
        }
        return action, rationale, None
    state["seq_candidate_replay_forced"] = {
        "trial_id": trial_counter,
        "candidate": payload["candidate"],
        "source_trial_id": payload["source_trial_id"],
        "action": forced,
        "k": payload["k"],
        "combined_E": payload["combined_E"],
        "E_quality": payload["E_quality"],
        "E_rate_noninf": payload["E_rate_noninf"],
    }
    state.pop("seq_candidate_replay_blocked", None)
    next_rationale = dict(rationale or {})
    next_rationale["seq_candidate_replay"] = True
    next_rationale["seq_candidate"] = payload["candidate"]
    next_rationale["seq_candidate_source_trial_id"] = payload["source_trial_id"]
    log.info(
        "Forcing seq candidate replay at trial %d for candidate %s "
        "(source trial %s, k=%s, combined_E=%s)",
        trial_counter,
        payload["candidate"],
        payload["source_trial_id"],
        payload["k"],
        payload["combined_E"],
    )
    return forced, next_rationale, dict(payload)


def _maybe_force_seq_due_action(
    *,
    state: dict[str, Any],
    journal: ExperimentJournal,
    tier: int,
    blacklist: list[dict[str, Any]],
    trial_counter: int,
    enabled: bool,
    lab: Any | None = None,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Return a forced sequential action before spending a planner turn.

    These due-checks consume only journal/state/blacklist data. Running them
    after the controller meant AutoPilot could pay for a full draft+critique and
    then discard it for an obligatory seq replay/fresh-eval. Keep the same
    priority order as the historical post-planner override.
    """
    placeholder_action = {"type": "seed_batch", "n_questions": SAFE_FALLBACK_SEED_N}
    placeholder_rationale: dict[str, Any] = {}

    action, rationale, fresh_eval_context = _maybe_force_seq_promotion_fresh_eval(
        placeholder_action,
        state=state,
        blacklist=blacklist,
        rationale=placeholder_rationale,
        trial_counter=trial_counter,
        enabled=enabled,
    )
    if fresh_eval_context is not None:
        return action, rationale, fresh_eval_context, None, None

    action, rationale, baseline_reference = _maybe_force_seq_baseline_draw(
        placeholder_action,
        state=state,
        journal=journal,
        tier=tier,
        blacklist=blacklist,
        rationale=placeholder_rationale,
        trial_counter=trial_counter,
        enabled=enabled,
    )
    if baseline_reference is not None:
        return action, rationale, None, baseline_reference, None

    action, rationale, replay_context = _maybe_force_seq_candidate_replay(
        placeholder_action,
        state=state,
        journal=journal,
        tier=tier,
        blacklist=blacklist,
        rationale=placeholder_rationale,
        trial_counter=trial_counter,
        enabled=enabled,
        lab=lab,
    )
    if replay_context is not None:
        return action, rationale, None, None, replay_context

    return None, None, None, None, None


def _update_seq_promotion_fresh_eval_state(
    state: dict[str, Any],
    *,
    seq: dict[str, Any] | None,
    action: dict[str, Any],
    eval_result: EvalResult,
    trial_counter: int,
    is_fresh_eval: bool,
    finalized: bool | None,
    baseline_update: Any | None = None,
    seq_alpha_wealth: dict[str, Any] | None = None,
) -> None:
    """Maintain the bounded pending fresh-eval state for seq baseline promotion."""
    if not isinstance(seq, dict):
        return
    baseline_update_reason = getattr(baseline_update, "reason", None)
    delta_ci = seq.get("baseline_promotion_delta_ci")
    if finalized:
        state.pop("_seq_promotion_candidate_replay", None)
        if baseline_update is not None and not bool(getattr(baseline_update, "updated", False)):
            state.pop("seq_pending_promotion_fresh_eval", None)
            blocked = {
                "trial_id": trial_counter,
                "candidate": seq.get("candidate"),
                "reason": "baseline-update-refused",
                "baseline_update_reason": baseline_update_reason,
                "combined_E": seq.get("baseline_promotion_combined_E"),
            }
            if delta_ci is not None:
                blocked["delta_ci"] = delta_ci
            state["seq_last_promotion_blocked"] = blocked
            return
        state.pop("seq_pending_promotion_fresh_eval", None)
        finalized_state = {
            "trial_id": trial_counter,
            "candidate": seq.get("candidate"),
            "combined_E": seq.get("baseline_promotion_combined_E"),
            "baseline_update_reason": baseline_update_reason,
        }
        if delta_ci is not None:
            finalized_state["delta_ci"] = delta_ci
        state["seq_last_promotion_finalized"] = finalized_state
        return
    if seq.get("baseline_reference_state") == "stale-reference":
        state.pop("_seq_promotion_candidate_replay", None)
        state.pop("seq_pending_promotion_fresh_eval", None)
        state["seq_last_promotion_blocked"] = {
            "trial_id": trial_counter,
            "candidate": seq.get("candidate"),
            "reason": "stale-reference",
        }
        return
    if is_fresh_eval:
        state.pop("_seq_promotion_candidate_replay", None)
        state.pop("seq_pending_promotion_fresh_eval", None)
        blocked = {
            "trial_id": trial_counter,
            "candidate": seq.get("candidate"),
            "reason": (
                "fresh-eval did not confirm"
                if not seq.get("confirmed")
                else "fresh-eval did not reach finalization threshold"
            ),
            "combined_E": seq.get("baseline_promotion_combined_E"),
        }
        if delta_ci is not None:
            blocked["delta_ci"] = delta_ci
        state["seq_last_promotion_blocked"] = blocked
        return
    if not seq.get("confirmed"):
        return
    alpha_wealth = seq_alpha_wealth or {}
    if _seq_alpha_confirmation_blocked(alpha_wealth):
        state.pop("_seq_promotion_candidate_replay", None)
        state.pop("seq_pending_promotion_fresh_eval", None)
        state["seq_last_promotion_blocked"] = {
            "trial_id": trial_counter,
            "candidate": seq.get("candidate"),
            "reason": "alpha-wealth-budget-exhausted",
            "alpha_wealth": dict(alpha_wealth),
        }
        return
    if MULTITIER_PROMOTION_ENABLED and isinstance(state.get(MULTITIER_PENDING_STATE_KEY), dict):
        # Binding T2/T3 validation owns the next due actions. The final T1
        # fresh evaluation is queued only after both higher tiers pass.
        return
    state["seq_pending_promotion_fresh_eval"] = {
        "candidate": seq.get("candidate"),
        "core_id": seq.get("core_id") or DEFAULT_EVIDENCE_CORE_ID,
        "source_trial_id": trial_counter,
        "tier": int(getattr(eval_result, "tier", DEFAULT_FRONTIER_TIER) or DEFAULT_FRONTIER_TIER),
        "action": dict(action),
        "combined_E": seq.get("baseline_promotion_combined_E"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attempts": 0,
    }


def _seq_alpha_wealth_state(
    candidates: Any,
    *,
    candidate: str,
    confirmed_candidates: Any = None,
    budget: float = SEQ_ALPHA_WEALTH_BUDGET,
    alpha: float = SEQ_DEFAULT_POLICY.alpha,
) -> dict[str, Any]:
    """Bound the free multiple-testing channel across seq candidate fingerprints."""
    existing = {str(item) for item in (candidates or []) if str(item or "").strip()}
    confirmed_existing = {
        str(item) for item in (confirmed_candidates or []) if str(item or "").strip()
    }
    candidate = str(candidate or "").strip()
    including_candidate = set(existing)
    if candidate:
        including_candidate.add(candidate)
    candidate_is_new = bool(candidate and candidate not in existing)
    alpha = max(0.0, float(alpha))
    budget = max(0.0, float(budget))
    bridge = seq_p0_2_bridge_status()
    bridge_enabled = bool(bridge["enabled"])
    charged_existing = set(confirmed_existing if bridge_enabled else existing)
    charged_including_candidate = set(charged_existing)
    if candidate:
        charged_including_candidate.add(candidate)
    candidate_confirmation_charge_is_new = bool(candidate and candidate not in charged_existing)
    alpha_spent = len(charged_existing) * alpha
    alpha_spent_including_candidate = len(charged_including_candidate) * alpha
    legacy_alpha_spent = len(existing) * alpha
    legacy_alpha_spent_including_candidate = len(including_candidate) * alpha
    confirmations_allowed = (
        not candidate_confirmation_charge_is_new or alpha_spent_including_candidate <= budget
    )
    return {
        "policy_version": SEQ_DEFAULT_POLICY.version,
        "alpha_wealth_mode": (
            "confirmed_fresh_eval_stage" if bridge_enabled else "tested_fingerprint"
        ),
        "p0_2_bridge": bridge,
        "alpha": round(alpha, 6),
        "budget": round(budget, 6),
        "fingerprints_tested": len(existing),
        "fingerprints_tested_including_candidate": len(including_candidate),
        "fingerprints_confirmed": len(confirmed_existing),
        "fingerprints_charged": len(charged_existing),
        "fingerprints_charged_including_candidate": len(charged_including_candidate),
        "candidate": candidate,
        "candidate_is_new": candidate_is_new,
        "candidate_confirmation_charge_is_new": candidate_confirmation_charge_is_new,
        "alpha_spent": round(alpha_spent, 6),
        "alpha_spent_including_candidate": round(alpha_spent_including_candidate, 6),
        "legacy_tested_fingerprint_alpha_spent": round(legacy_alpha_spent, 6),
        "legacy_tested_fingerprint_alpha_spent_including_candidate": round(
            legacy_alpha_spent_including_candidate, 6
        ),
        "expected_false_confirms": round(alpha_spent, 6),
        "expected_false_confirms_including_candidate": round(alpha_spent_including_candidate, 6),
        "budget_remaining": round(max(0.0, budget - alpha_spent), 6),
        "budget_remaining_including_candidate": round(
            max(0.0, budget - alpha_spent_including_candidate), 6
        ),
        "budget_exhausted": alpha_spent >= budget if budget > 0.0 else bool(existing),
        "new_fingerprint_confirmations_allowed": confirmations_allowed,
        "new_fingerprint_dispatch_allowed": confirmations_allowed,
    }


def _seq_trial_is_incumbent_representative(entry: Any) -> bool:
    """B4 / SEQ-1 provenance gate for the sequential null (incumbent) profile.

    Keep a same-tier trial's per-question outcomes in the null profile ONLY when the
    trial is plausibly incumbent-representative, using the strongest provenance signal
    the journal rows actually carry:
      * ``pareto_status == "frontier"`` — the trial sat on the live Pareto frontier
        (the best-known / incumbent set) when recorded; OR
      * ``keep_revert_decision == "keep"`` — the trial's config was accepted (kept),
        i.e. promoted-like.

    ``config_diff`` is deliberately NOT used as a signal: it is computed relative to
    the immediate same-species PARENT, not the incumbent baseline, so a chain of
    unpromoted experiments can each show an empty diff yet none represent the
    incumbent — exactly the depressed off-incumbent rows the audit (SEQ-1) flags as
    the source of anti-conservative wealth accrual from a mixed profile.
    """
    if str(getattr(entry, "pareto_status", "") or "") == "frontier":
        return True
    if str(getattr(entry, "keep_revert_decision", "") or "") == "keep":
        return True
    return False


def _seq_inputs_for_trial(
    *,
    journal: ExperimentJournal,
    action: dict[str, Any],
    tier: int,
    candidate_override: str | None = None,
    quality_exclude_before_ts: float | None = None,
) -> dict[str, Any]:
    """Build default-off W4 shadow inputs from trusted prior journal evidence.

    ``quality_exclude_before_ts`` is the active eval-quality instrument-era boundary epoch
    (the analogue of the speed axis's ``pareto_exclude_before_ts``). When set, journal rows
    whose ``timestamp`` predates the boundary are dropped from EVERY fold here — the null
    (baseline) profile, prior quality/rate observations, and the seq/alpha-wealth candidate
    sets — so pre-boundary PRIORS never pool with post-boundary evidence in the anytime-valid
    e-process wealth. When ``None`` (default) the fence is inert and all rows fold, so the
    unit contract is unchanged.
    """
    candidate = candidate_override or _config_fingerprint(action)
    prior_quality_obs: list[tuple[int | None, float]] = []
    prior_rate_obs: list[tuple[int | None, float]] = []
    baseline_trials: list[dict[str, bool]] = []
    baseline_task_rates: list[float] = []
    seq_candidates: set[str] = set()
    seq_confirmed_candidates: set[str] = set()

    for entry in reversed(journal.entries_with_supersessions()):
        if getattr(entry, "bug_corrupted_by", ""):
            continue
        if getattr(entry, "outcome_status", "ok") in {"invalid", "skipped"}:
            continue
        # Defect #1/#2: eval-instrument era fence. A pre-boundary row is a PRIOR — exclude it
        # from the null profile, prior obs, and seq/alpha candidate sets so the e-process
        # wealth never mixes the pre-/post-boundary instrument. An unparseable timestamp is
        # treated as pre-boundary (fail-closed): it cannot be proven to belong to this era.
        if quality_exclude_before_ts is not None:
            entry_ts = _parse_journal_timestamp(getattr(entry, "timestamp", ""))
            if entry_ts is None or entry_ts < quality_exclude_before_ts:
                continue
        try:
            if int(getattr(entry, "tier", -1)) != int(tier):
                continue
        except (TypeError, ValueError):
            continue
        eval_details = getattr(entry, "eval_details", {}) or {}
        if not isinstance(eval_details, dict):
            continue

        # B4 / SEQ-1: only incumbent-representative trials feed the null profile +
        # its rate baseline. A dominated/unpromoted experiment (a config change that
        # was NOT kept and is not on the frontier) that scored poorly would depress
        # the mixture and accrue e-process wealth anti-conservatively. The candidate's
        # OWN prior obs (below) are NOT gated — those rebuild the candidate's evidence.
        incumbent_representative = _seq_trial_is_incumbent_representative(entry)
        outcome_map = _question_outcome_map(eval_details.get("question_results"))
        if (
            incumbent_representative
            and outcome_map
            and len(baseline_trials) < SEQ_BASELINE_PROFILE_LIMIT
        ):
            baseline_trials.append(outcome_map)
        if incumbent_representative and len(baseline_task_rates) < SEQ_BASELINE_PROFILE_LIMIT:
            # SEQ-B: the incumbent comparator MUST be the same measurement as the
            # candidate's (`seq_task_rate_qph_from`). `n_questions` is deliberately NOT
            # passed: `seq_task_rate_qph` derives the numerator from question_results —
            # the questions the wall clock actually covers — on both sides. Passing
            # `len(outcome_map)` here while the candidate side divided by
            # `EvalResult.n_questions` (decision partition only) is exactly what scored an
            # unchanged config 15% slower on every trial.
            row = {
                "quality": getattr(entry, "quality", 0.0),
                "eval_details": eval_details,
            }
            task_rate = seq_task_rate_qph_from_row(row)
            if task_rate is not None and task_rate > 0.0:
                baseline_task_rates.append(task_rate)

        seq = getattr(entry, "seq", {}) or {}
        if not isinstance(seq, dict):
            continue
        if str(seq.get("core_id") or DEFAULT_EVIDENCE_CORE_ID) != DEFAULT_EVIDENCE_CORE_ID:
            continue
        seq_candidate = str(seq.get("candidate") or "")
        if seq_candidate:
            seq_candidates.add(seq_candidate)
            if seq.get("confirmed") is True or str(seq.get("state") or "") == "confirmed":
                seq_confirmed_candidates.add(seq_candidate)
        if seq_candidate != candidate:
            continue
        if len(prior_quality_obs) < SEQ_PRIOR_OBS_LIMIT and "z" in seq:
            try:
                prior_quality_obs.append((entry.trial_id, float(seq["z"])))
            except (TypeError, ValueError):
                pass
        if len(prior_rate_obs) < SEQ_PRIOR_OBS_LIMIT and "z_rate" in seq:
            try:
                prior_rate_obs.append((entry.trial_id, float(seq["z_rate"])))
            except (TypeError, ValueError):
                pass

    # B4 / SEQ-1: if too few incumbent-representative trials survived the provenance
    # filter, the null profile would be thin/contaminated. Treat the sequential inputs
    # as UNAVAILABLE (empty profile) so gate.check() skips the sequential path
    # (it requires `bool(baseline_profile)`), leaving verdict.seq is None and
    # seq_confirmed=None downstream — the conservative outcome, not a silent mixture.
    if len(baseline_trials) < SEQ_BASELINE_PROFILE_MIN_TRIALS:
        baseline_profile: dict[str, float] = {}
        baseline_task_rate: float | None = None
    else:
        baseline_profile = baseline_profile_from_trials(reversed(baseline_trials))
        # SEQ-B: MEDIAN, not mean. Task rate (questions per eval-wall-hour) is a
        # heavy-right-tailed statistic — an aborted batch divides its question count by a
        # near-zero wall clock. The arithmetic mean is not a robust estimator of it: a
        # single such row inside the 120-row pool moved the comparator by ~36,000 qph and
        # pinned every subsequent candidate at the clip floor. The median is unaffected by
        # a minority of corrupt rows regardless of where the validity floor sits, and it is
        # already the house estimator for this quantity (`recent_median_task_rate` in
        # `_seq_readiness_snapshot`). This changes the ESTIMATOR of the null, not the null:
        # the comparator is still built only from PRIOR incumbent-representative trials, so
        # it stays predictable (F_{t-1}-measurable) and the e-process stays anytime-valid.
        baseline_task_rate = (
            _median(baseline_task_rates)
            if len(baseline_task_rates) >= SEQ_BASELINE_RATE_MIN_TRIALS
            else None
        )
    return {
        "candidate": candidate,
        "core_id": DEFAULT_EVIDENCE_CORE_ID,
        "baseline_profile": baseline_profile,
        "baseline_task_rate": baseline_task_rate,
        "prior_quality_obs": list(reversed(prior_quality_obs)),
        "prior_rate_obs": list(reversed(prior_rate_obs)),
        "baseline_reference": _seq_baseline_reference_state(journal, tier=tier),
        "alpha_wealth": _seq_alpha_wealth_state(
            seq_candidates,
            candidate=candidate,
            confirmed_candidates=seq_confirmed_candidates,
        ),
    }


def _format_blacklist_pattern(pattern: Any) -> str:
    if not isinstance(pattern, dict) or not pattern:
        return "{}"
    return json.dumps(pattern, sort_keys=True, separators=(",", ":"), default=str)


def _format_blacklist_entry_status(entry: dict[str, Any]) -> str:
    """Return planner-facing freshness metadata for an enforced blacklist row."""
    parts: list[str] = []
    purge_target = purge_scoped_target(entry)
    if purge_target is not None:
        parts.append(f"purge-scoped={purge_target.get('target_key', 'unknown')}")
        if entry.get("source_trial", 0) == -1:
            parts.append("manual-purge-approval-required")
    reason_class = _auto_blacklist_reason_class(entry)
    if reason_class:
        parts.append(f"class={reason_class}")
    expires_at = _blacklist_expires_at(entry)
    if expires_at is not None:
        parts.append(f"expires={expires_at.date().isoformat()}")
    elif _entry_is_non_expiring_blacklist(entry):
        parts.append("non-expiring")
    else:
        parts.append("no-expiry-metadata")
    return "; ".join(parts)


def _format_blacklist_for_prompt(blacklist: list[dict[str, Any]]) -> str:
    """Render blacklist prompt context without hiding older enforced patterns."""
    if not blacklist:
        return "  (none)"

    retryable = [
        (entry, retry_meta)
        for entry in blacklist
        if (retry_meta := _p0_3_retryable_blacklist_entry(entry)) is not None
    ]
    enforced = [entry for entry in blacklist if _p0_3_retryable_blacklist_entry(entry) is None]
    cap = max(0, BLACKLIST_RENDER_CAP)
    shown = enforced[-cap:] if cap else []
    older = enforced[:-cap] if cap else list(enforced)
    lines: list[str] = []

    if retryable:
        lines.append(
            "  Retryable blacklist re-exploration entries "
            f"({len(retryable)} automated era/infra-contaminated patterns; manual "
            "purge still approval-token gated):"
        )
        for entry, retry_meta in retryable:
            reason = str(entry.get("reason", ""))
            if len(reason) > 80:
                reason = reason[:79] + "..."
            lines.append(
                "  - {pattern} (target={target}; source_trial={trial}; {status}) -- {reason}".format(
                    pattern=_format_blacklist_pattern(entry.get("pattern", {})),
                    target=retry_meta.get("target_key", "unknown"),
                    trial=retry_meta.get("source_trial", "unknown"),
                    status=_format_blacklist_entry_status(entry),
                    reason=reason,
                )
            )

    if shown:
        lines.append(
            f"  Recent enforced entries ({len(shown)} newest; all {len(enforced)} enforced; "
            "expired/observational/broad numeric rows are loader-filtered):"
        )
        for entry in shown:
            reason = str(entry.get("reason", ""))
            if len(reason) > 80:
                reason = reason[:79] + "..."
            lines.append(
                "  - {pattern} ({status}) -- {reason}".format(
                    pattern=_format_blacklist_pattern(entry.get("pattern", {})),
                    status=_format_blacklist_entry_status(entry),
                    reason=reason,
                )
            )

    if older:
        lines.append(f"  Older enforced patterns ({len(older)}; reasons omitted, still blocked):")
        for entry in older:
            suffix_parts = [_format_blacklist_entry_status(entry)]
            if entry.get("source_trial") is not None:
                suffix_parts.append(f"source_trial={entry['source_trial']}")
            suffix = "; ".join(part for part in suffix_parts if part)
            lines.append(f"    - {_format_blacklist_pattern(entry.get('pattern', {}))} ({suffix})")

    if not shown and not older:
        lines.append("  Enforced entries: (none)")

    return "\n".join(lines)


def _p0_3_retryable_blacklist_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return retryable_reexploration_target(entry)
    except Exception:
        log.debug("P0.3 retryable blacklist classification failed", exc_info=True)
        return None


def _p0_3_retryable_blacklist_match(
    action: dict[str, Any],
    blacklist: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    for entry in reversed(blacklist):
        retry_meta = _p0_3_retryable_blacklist_entry(entry)
        if retry_meta is None:
            continue
        pattern = entry.get("pattern", {})
        if not isinstance(pattern, dict):
            continue
        if pattern and all(action.get(k) == v for k, v in pattern.items()):
            return {
                "reason": entry.get("reason", "blacklisted"),
                **retry_meta,
            }
    return None


def _record_p0_3_reexploration_rationale(
    rationale: dict[str, Any] | None,
    retry_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        **(rationale or {}),
        "p0_3_blacklist_reexploration": True,
        "p0_3_blacklist_reexploration_target": retry_meta.get("target_key"),
        "p0_3_blacklist_reexploration_source_trial": retry_meta.get("source_trial"),
        "p0_3_blacklist_reexploration_reason": retry_meta.get("reason", ""),
        "p0_3_blacklist_reexploration_scope": retry_meta.get("retry_scope"),
        "falsifier": (
            (rationale or {}).get("falsifier")
            or "P0.3 instrument-era blacklist re-exploration fails the current safety gate"
        ),
    }


def _is_observational_feedback_action(action: Any) -> bool:
    return bool(
        isinstance(action, dict) and action.get("type") in OBSERVATIONAL_ACTION_BLACKLIST_DENYLIST
    )


def _is_observational_feedback_signature(signature: str) -> bool:
    try:
        action = json.loads(signature)
    except (TypeError, json.JSONDecodeError):
        return False
    return _is_observational_feedback_action(action)


def _enforce_experiment_quota(
    action: dict[str, Any],
    state: dict[str, Any],
    memory_count: int,
    rationale: dict[str, Any] | None = None,
    trial_counter: int = 0,
    blacklist: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Cap consecutive passive (seed/distill) actions once memory is large.

    Below QUOTA_MEMORY_THRESHOLD the system is still legitimately seeding, so
    passive actions pass through (the counter is still tracked). At or above the
    threshold, a run of more than MAX_CONSECUTIVE_PASSIVE passive actions is
    replaced with a frontier-moving numeric_trial so the planner cannot
    rationalize passive work forever. Non-passive actions reset the counter.
    """
    atype = action.get("type", "")
    if atype not in PASSIVE_ACTIONS:
        state["consecutive_passive_actions"] = 0
        return action, rationale

    streak = int(state.get("consecutive_passive_actions", 0))
    if memory_count >= QUOTA_MEMORY_THRESHOLD and streak >= MAX_CONSECUTIVE_PASSIVE:
        forced, blocked_reason = _first_unblacklisted_numeric_trial_action(
            blacklist or [],
            trial_counter=trial_counter,
        )
        if forced is None:
            log.warning(
                "Experiment quota due after %d consecutive passive actions, but "
                "all numeric_trial quota surfaces are blacklisted: %s",
                streak,
                blocked_reason,
            )
            state["experiment_quota_blocked"] = {
                "trial_id": trial_counter,
                "reason": blocked_reason,
                "action": action,
            }
            state["consecutive_passive_actions"] = streak + 1
            return action, rationale
        surface = forced["surface"]
        log.warning(
            "Experiment quota: %d consecutive passive actions with memory_count=%d "
            ">= %d threshold; forcing frontier-moving numeric_trial(surface=%s) "
            "instead of another '%s'.",
            streak,
            memory_count,
            QUOTA_MEMORY_THRESHOLD,
            surface,
            atype,
        )
        state["consecutive_passive_actions"] = 0
        state["experiment_quota_blocked"] = None
        return (
            forced,
            {
                **(rationale or {}),
                "experiment_quota_forced": True,
                "experiment_quota_reason": (
                    f"{streak} consecutive passive actions at memory={memory_count}"
                ),
            },
        )

    state["consecutive_passive_actions"] = streak + 1
    return action, rationale


def _entry_action_tier(entry: Any) -> int | None:
    try:
        return int(getattr(entry, "tier"))
    except (TypeError, ValueError):
        return None


def _iter_journal_entries(journal: ExperimentJournal | None) -> list[Any]:
    if journal is None:
        return []
    try:
        if hasattr(journal, "entries_with_supersessions"):
            return list(journal.entries_with_supersessions())
        return list(journal.all_entries())
    except Exception:
        return []


def _higher_tier_trial_stats(
    journal: ExperimentJournal | None,
    *,
    tiers: tuple[int, ...] = HIGHER_TIER_PROBE_TIERS,
) -> dict[int, dict[str, int | None]]:
    stats: dict[int, dict[str, int | None]] = {
        tier: {"count": 0, "last_trial_id": None} for tier in tiers
    }
    for entry in _iter_journal_entries(journal):
        if getattr(entry, "bug_corrupted_by", ""):
            continue
        tier = _entry_action_tier(entry)
        if tier not in stats:
            continue
        current = stats[tier]
        current["count"] = int(current.get("count") or 0) + 1
        try:
            trial_id = int(getattr(entry, "trial_id"))
        except (TypeError, ValueError):
            continue
        last = current.get("last_trial_id")
        if last is None or trial_id > int(last):
            current["last_trial_id"] = trial_id
    return stats


def _archive_frontier_size(archive: ParetoArchive | None, tier: int) -> int | None:
    if archive is None:
        return None
    try:
        summary = archive.summary(tier=tier)
    except Exception:
        return None
    try:
        return int(summary.get("frontier_size") or 0)
    except (TypeError, ValueError):
        return None


def _maybe_force_higher_tier_probe(
    action: dict[str, Any],
    state: dict[str, Any],
    *,
    journal: ExperimentJournal | None = None,
    archive: ParetoArchive | None = None,
    blacklist: list[dict[str, Any]] | None = None,
    rationale: dict[str, Any] | None = None,
    trial_counter: int = 0,
    w8_replay_pressure_text: str = "",
    outcome_progress_pressure_text: str = "",
    tiers: tuple[int, ...] = HIGHER_TIER_PROBE_TIERS,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Bound T1-only exploitation by forcing occasional T2/T3 eval probes.

    The default frontier and promotion logic remain T1-scoped. This guard only
    prevents higher-tier evidence from going stale while the planner repeatedly
    exploits T1; each tier is still compared to its own baseline/frontier.
    """
    if not HIGHER_TIER_PROBE_GUARD:
        return action, rationale
    if _w8_replay_pressure_active(w8_replay_pressure_text):
        return action, rationale
    if isinstance(rationale, dict) and (
        rationale.get("critic_reject_numeric_fallback")
        or rationale.get("critic_reject_safe_fallback")
    ):
        # A binding critic already rejected the planner's preferred action and
        # routed to a fallback. Do not let a quota/probe guard
        # immediately resurrect that rejected shape in the same trial.
        return action, rationale

    try:
        selected_tier = int(action.get("tier", DEFAULT_FRONTIER_TIER))
    except (TypeError, ValueError):
        selected_tier = DEFAULT_FRONTIER_TIER
    if action.get("type") == "deep_eval" and selected_tier in tiers:
        return (
            action,
            {
                **(rationale or {}),
                "higher_tier_probe_satisfied_by_selected_action": True,
                "higher_tier_probe_tier": selected_tier,
            },
        )
    if _outcome_progress_frontier_stalled(outcome_progress_pressure_text):
        return (
            action,
            {
                **(rationale or {}),
                "higher_tier_probe_skipped_outcome_stalled": True,
            },
        )

    guard_state = state.get("higher_tier_probe_guard")
    if isinstance(guard_state, dict):
        try:
            last_forced = int(guard_state.get("last_forced_trial_id"))
        except (TypeError, ValueError):
            last_forced = None
        if (
            last_forced is not None
            and trial_counter - last_forced < HIGHER_TIER_PROBE_MIN_GAP_TRIALS
        ):
            return action, rationale

    stats = _higher_tier_trial_stats(journal, tiers=tiers)
    candidates: list[tuple[tuple[int, int, int, int, int], int, str]] = []
    for tier in tiers:
        count = int(stats.get(tier, {}).get("count") or 0)
        last_trial_id = stats.get(tier, {}).get("last_trial_id")
        gap = (
            HIGHER_TIER_PROBE_STALE_TRIALS + 1
            if last_trial_id is None
            else max(0, trial_counter - int(last_trial_id))
        )
        frontier_size = _archive_frontier_size(archive, tier)
        no_frontier = frontier_size == 0
        deficit = max(0, HIGHER_TIER_PROBE_MIN_TRIALS_PER_TIER - count)
        stale = last_trial_id is None or gap >= HIGHER_TIER_PROBE_STALE_TRIALS
        if not (deficit or stale or no_frontier):
            continue
        forced = {"type": "deep_eval", "tier": tier}
        blocked = check_blacklist(forced, blacklist or [])
        if blocked:
            continue
        reason = (
            f"T{tier} higher-tier probe due: count={count}, "
            f"last_trial={last_trial_id if last_trial_id is not None else 'never'}, "
            f"gap={gap}, frontier={frontier_size if frontier_size is not None else 'unknown'}"
        )
        score = (
            1 if last_trial_id is None else 0,
            1 if no_frontier else 0,
            deficit,
            gap,
            -tier,
        )
        candidates.append((score, tier, reason))

    if not candidates:
        return action, rationale

    _score, tier, reason = max(candidates, key=lambda item: item[0])
    forced = {"type": "deep_eval", "tier": tier}
    state["higher_tier_probe_guard"] = {
        "last_forced_trial_id": trial_counter,
        "last_forced_tier": tier,
        "last_forced_reason": reason,
        "original_action": dict(action),
        "forced_action": dict(forced),
    }
    log.warning(
        "Higher-tier probe guard forcing deep_eval(tier=%d) instead of %s: %s",
        tier,
        json.dumps(action, default=str),
        reason,
    )
    return (
        forced,
        {
            **(rationale or {}),
            "higher_tier_probe_forced": True,
            "higher_tier_probe_tier": tier,
            "higher_tier_probe_reason": reason,
            "higher_tier_probe_original": dict(action),
        },
    )


def _maybe_force_outcome_progress_action(
    action: dict[str, Any],
    state: dict[str, Any],
    *,
    blacklist: list[dict[str, Any]] | None = None,
    rationale: dict[str, Any] | None = None,
    trial_counter: int = 0,
    outcome_progress_pressure_text: str = "",
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Escalate stale outcome flow from prompt advice into a bounded fallback.

    Outcome pressure is not a safety gate and does not alter scoring. It only
    prevents stale frontier flow from being satisfied by passive seeding,
    validation-only evals, or housekeeping when a metric-bearing numeric trial
    remains available.
    """
    if not _outcome_progress_frontier_stalled(outcome_progress_pressure_text):
        return action, rationale
    if _action_can_move_outcome_frontier(action):
        state["outcome_progress_forced"] = None
        return (
            action,
            {
                **(rationale or {}),
                "outcome_progress_satisfied_by_selected_action": True,
            },
        )

    forced, blocked_reason = _first_unblacklisted_numeric_trial_action(
        blacklist or [],
        trial_counter=trial_counter,
    )
    if forced is None:
        state["outcome_progress_blocked"] = {
            "trial_id": trial_counter,
            "reason": blocked_reason,
            "action": action,
        }
        log.warning(
            "Outcome progress is frontier-stalled, but no numeric_trial "
            "fallback remains available: %s",
            blocked_reason,
        )
        return action, rationale

    log.warning(
        "Outcome progress is frontier-stalled; forcing numeric_trial(surface=%s) instead of '%s'.",
        forced["surface"],
        action.get("type", "unknown"),
    )
    state["outcome_progress_forced"] = {
        "trial_id": trial_counter,
        "original_action": action,
        "forced_action": forced,
    }
    state.pop("outcome_progress_blocked", None)
    return (
        forced,
        {
            **(rationale or {}),
            "outcome_progress_forced": True,
            "outcome_progress_original": dict(action),
            "falsifier": (
                (rationale or {}).get("falsifier")
                or "outcome-progress numeric fallback fails to produce keepable frontier evidence"
            ),
        },
    )


def _maybe_force_frontier_rerun_action(
    action: dict[str, Any],
    state: dict[str, Any],
    *,
    journal: ExperimentJournal | None = None,
    archive: ParetoArchive | None = None,
    blacklist: list[dict[str, Any]] | None = None,
    rationale: dict[str, Any] | None = None,
    trial_counter: int = 0,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Turn an active frontier-rerun marker into a concrete numeric trial."""
    marker = state.get("frontier_rerun_required")
    if not (isinstance(marker, dict) and marker.get("required")):
        return action, rationale
    min_trials = _frontier_rerun_min_trials(marker)
    completed_trials = _frontier_rerun_completed_numeric_trials(marker, journal)
    # E8-PANELS-a: keep the marker's own counters CURRENT while it is open.
    #
    # `completed_numeric_trials` used to be written only by
    # `_clear_frontier_rerun_marker`, i.e. once the gate had already been
    # satisfied. Every reader of the OPEN marker therefore saw the value it was
    # created with — 0 — while the live count was recomputed here each cycle and
    # only ever reached the rationale. That is the 0/16-vs-15/16 split: the
    # banner (`gen_system_card`) and the operator brief
    # (`optimization_brief.py:188`, rendering "pending <era> numeric rerun
    # (0/16)") were reading a field nothing updated mid-run.
    #
    # Reporting only — nothing gates on this number. The decision path uses the
    # freshly computed `completed_trials` local, and `speed_hold` keys off
    # `required`, not the count. Skipped when `journal is None`, because the
    # counter returns 0 in that case and would overwrite a real value with a
    # placeholder.
    if journal is not None:
        marker["completed_numeric_trials"] = completed_trials
        marker["min_numeric_trials"] = min_trials
    pending = state.get("frontier_rerun_pending_clear")
    if isinstance(pending, dict) and pending.get("trial_id") is not None:
        if journal is not None and completed_trials < min_trials:
            action, forced_rationale = _force_frontier_rerun_numeric_action(
                action,
                state,
                marker=marker,
                blacklist=blacklist or [],
                rationale=rationale,
                trial_counter=trial_counter,
                completed_trials=completed_trials,
                min_trials=min_trials,
            )
            return action, forced_rationale
        if journal is not None and completed_trials >= min_trials:
            _clear_frontier_rerun_marker(
                state,
                marker=marker,
                pending=pending,
                completed_trials=completed_trials,
                min_trials=min_trials,
                archive=archive,
            )
            return action, {
                **(rationale or {}),
                "frontier_rerun_cleared": True,
                "frontier_rerun_completed_numeric_trials": completed_trials,
                "frontier_rerun_min_numeric_trials": min_trials,
                "frontier_rerun_archive_snapshot": state["frontier_rerun_required"].get(
                    "archive_snapshot"
                ),
            }
        return action, {
            **(rationale or {}),
            "frontier_rerun_pending_clear": True,
            "frontier_rerun_pending_trial_id": pending.get("trial_id"),
            "frontier_rerun_completed_numeric_trials": completed_trials,
            "frontier_rerun_min_numeric_trials": min_trials,
        }
    if action.get("type") == "numeric_trial":
        state["frontier_rerun_forced"] = None
        state["frontier_rerun_pending_clear"] = {
            "trial_id": trial_counter,
            "action": action,
            "reason": str(marker.get("reason") or "frontier rerun required"),
        }
        return action, {
            **(rationale or {}),
            "frontier_rerun_satisfied_by_selected_action": True,
        }

    return _force_frontier_rerun_numeric_action(
        action,
        state,
        marker=marker,
        blacklist=blacklist or [],
        rationale=rationale,
        trial_counter=trial_counter,
        completed_trials=completed_trials,
        min_trials=min_trials,
    )


def _clear_frontier_rerun_marker(
    state: dict[str, Any],
    *,
    marker: dict[str, Any],
    pending: dict[str, Any],
    completed_trials: int,
    min_trials: int,
    archive: ParetoArchive | None = None,
) -> None:
    """Persist that the era-scoped frontier rerun has satisfied its gate."""
    reason = str(marker.get("reason") or "frontier rerun required")
    archive_snapshot = _frontier_rerun_archive_snapshot(archive)
    state["frontier_rerun_required"] = {
        **marker,
        "required": False,
        "cleared_at": datetime.now(timezone.utc).isoformat(),
        "cleared_after_trial_id": pending.get("trial_id"),
        "completed_numeric_trials": completed_trials,
        "min_numeric_trials": min_trials,
        "archive_snapshot": archive_snapshot,
        "reason": (
            f"frontier rerun satisfied: {completed_trials}/{min_trials} "
            f"current-era numeric trials complete; {reason}"
        ),
    }
    state["frontier_rerun_forced"] = None
    state.pop("frontier_rerun_pending_clear", None)
    state.pop("frontier_rerun_blocked", None)


def _frontier_rerun_archive_snapshot(archive: ParetoArchive | None) -> dict[str, Any]:
    """Compact evidence that the current-era archive was rebuilt at rerun clear."""
    if archive is None:
        return {"status": "unavailable", "reason": "archive not supplied"}
    try:
        summary = dict(archive.summary(tier=DEFAULT_FRONTIER_TIER))
        frontier = archive.frontier(tier=DEFAULT_FRONTIER_TIER)
    except Exception as exc:  # pragma: no cover - defensive operator evidence only
        return {"status": "error", "reason": str(exc)}

    return {
        "status": "ok",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "tier": DEFAULT_FRONTIER_TIER,
        "frontier_size": int(summary.get("frontier_size") or 0),
        "total_entries": int(summary.get("total_entries") or 0),
        "hypervolume": summary.get("hypervolume"),
        "best_quality": summary.get("best_quality"),
        "best_speed": summary.get("best_speed"),
        "trial_ids": [entry.trial_id for entry in frontier],
    }


def _frontier_rerun_min_trials(marker: dict[str, Any]) -> int:
    try:
        return max(1, int(marker.get("min_numeric_trials") or 1))
    except (TypeError, ValueError):
        return 1


def _frontier_rerun_opened_ts(marker: dict[str, Any]) -> float | None:
    for key in ("rerun_started_at", "opened_at"):
        ts = parse_journal_ts(marker.get(key))
        if ts is not None:
            return ts
    return None


def _frontier_rerun_completed_numeric_trials(
    marker: dict[str, Any],
    journal: ExperimentJournal | None,
) -> int:
    if journal is None:
        return 0
    opened_ts = _frontier_rerun_opened_ts(marker)
    entries = (
        journal.entries_with_supersessions()
        if hasattr(journal, "entries_with_supersessions")
        else journal.all_entries()
    )
    completed = 0
    for entry in entries:
        if entry.bug_corrupted_by:
            continue
        if entry.action_type != "numeric_trial":
            continue
        if entry.tier < MIN_FRONTIER_EVAL_TIER:
            continue
        ts = parse_journal_ts(entry.timestamp)
        if opened_ts is not None and (ts is None or ts < opened_ts):
            continue
        completed += 1
    return completed


def _force_frontier_rerun_numeric_action(
    action: dict[str, Any],
    state: dict[str, Any],
    *,
    marker: dict[str, Any],
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None,
    trial_counter: int,
    completed_trials: int,
    min_trials: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    forced, blocked_reason = _first_unblacklisted_numeric_trial_action(
        blacklist,
        trial_counter=trial_counter,
    )
    if forced is None:
        log.warning(
            "Frontier rerun required, but all numeric_trial surfaces are blacklisted: %s",
            blocked_reason,
        )
        state["frontier_rerun_blocked"] = {
            "trial_id": trial_counter,
            "reason": blocked_reason,
            "action": action,
            "completed_numeric_trials": completed_trials,
            "min_numeric_trials": min_trials,
        }
        return action, rationale

    reason = str(marker.get("reason") or "frontier rerun required")
    log.warning(
        "Frontier rerun required (%s); forcing numeric_trial(surface=%s) "
        "instead of '%s' (%d/%d completed numeric trials).",
        reason,
        forced["surface"],
        action.get("type", "unknown"),
        completed_trials,
        min_trials,
    )
    state["frontier_rerun_forced"] = {
        "trial_id": trial_counter,
        "reason": reason,
        "original_action": action,
        "forced_action": forced,
        "completed_numeric_trials": completed_trials,
        "min_numeric_trials": min_trials,
    }
    state["frontier_rerun_pending_clear"] = {
        "trial_id": trial_counter,
        "action": forced,
        "reason": reason,
        "completed_numeric_trials": completed_trials,
        "min_numeric_trials": min_trials,
    }
    state.pop("frontier_rerun_blocked", None)
    return (
        forced,
        {
            **(rationale or {}),
            "frontier_rerun_forced": True,
            "frontier_rerun_reason": reason,
            "frontier_rerun_completed_numeric_trials": completed_trials,
            "frontier_rerun_min_numeric_trials": min_trials,
        },
    )


def _build_feature_flags_block(
    lab: Any,
    *,
    denylisted_flags: set[str] | None = None,
) -> str:
    """Render live feature-flag state + dependency rules for the planner prompt.

    The planner previously had no view of which flags exist, what they depend on,
    or which are currently on — so it proposed graph_router (off) without
    specialist_routing (off) 119× in a row. This block gives it the same
    dependency rules the validator enforces, plus live state, so it can enable a
    missing dependency first instead of re-sampling an impossible flag.
    """
    try:
        current = lab.current_flags()
    except Exception:
        current = {}
    try:
        schema = lab.flag_schema()
    except Exception:
        schema = []
    if not schema:
        return "  (feature flag registry unavailable)"

    def state_str(name: str) -> str:
        if not current:
            return "?"
        return "ON" if current.get(name) else "OFF"

    on = sorted(n for n, v in current.items() if v) if current else []
    lines: list[str] = []
    if current:
        lines.append(f"Currently ON ({len(on)}): " + (", ".join(on) or "(none)"))
    else:
        lines.append("Currently ON: (unknown — orchestrator /config not reachable)")
    denylisted_flags = set(denylisted_flags or set())
    if denylisted_flags:
        lines.append(
            "Convention-denylisted flags: "
            + ", ".join(sorted(denylisted_flags))
            + " (do not propose structural_experiment for these flags)"
        )
    lines.append("")
    lines.append(
        "Flags WITH dependencies (single-variable rule, AP-9: enable a missing "
        "dependency in its OWN trial first, then the dependent flag NEXT trial):"
    )
    any_dep = False
    for spec in schema:
        deps = spec.get("dependencies") or []
        if not deps:
            continue
        any_dep = True
        dep_states = ", ".join(f"{d}={state_str(d)}" for d in deps)
        deps_met = bool(current) and all(current.get(d) for d in deps)
        verdict = "DEPS MET — eligible" if deps_met else "DEPS MISSING — do NOT propose yet"
        lines.append(
            f"  - {spec['name']} (currently {state_str(spec['name'])}) "
            f"requires [{dep_states}] → {verdict}"
        )
    if not any_dep:
        lines.append("  (no dependency-bearing flags in registry)")
    lines.append("")
    lines.append(
        "RULE: never propose structural_experiment for a flag whose dependencies "
        "are not all currently ON. Setting a flag that is already ON is a no-op "
        "and burns a trial."
    )
    if denylisted_flags:
        lines.append(
            "RULE: never propose structural_experiment for convention-denylisted "
            "flags; the dispatcher will reject them before evaluation."
        )
    return "\n".join(lines)


def _build_last_invalid_feedback(state: dict[str, Any]) -> str:
    """Render the last non-executing action + its reason for the planner prompt.

    This is the residue that was previously discarded. Showing it (with the
    repeat count and blacklist threshold) is what stops the planner re-emitting
    an action the validator already rejected.
    """
    counts = state.get("invalid_signature_counts", {}) or {}
    repeated = sorted(
        (
            (c, s)
            for s, c in counts.items()
            if c >= 2 and not _is_observational_feedback_signature(s)
        ),
        reverse=True,
    )[:5]

    def _repeated_block(lines: list[str]) -> str:
        # Persistent across the run (NOT cleared when a trial succeeds), so a
        # repeatedly-rejected/invalid signature still surfaces even after the
        # single-turn last_invalid_action has been cleared by a good trial.
        if repeated:
            lines.append(
                "  Repeatedly non-executing signatures this run "
                f"(auto-blacklisted at {INVALID_SIGNATURE_BLACKLIST_THRESHOLD}×):"
            )
            for c, s in repeated:
                lines.append(f"    {c}×  {s[:160]}")
        return "\n".join(lines)

    act = state.get("last_invalid_action")
    if _is_observational_feedback_action(act):
        lines = [
            "  (last non-executing action was observational and remains schedulable; "
            "it is not treated as a repeat blocker)"
        ]
        if repeated:
            return _repeated_block(lines)
        return "\n".join(lines)
    if not act:
        if repeated:
            return _repeated_block(["  (last action executed; but these signatures keep failing:)"])
        return "  (none — the last action executed and produced metrics)"
    reason = state.get("last_invalid_reason", "")
    status = state.get("last_invalid_status", "skipped")
    sig = _action_signature(act)
    n = int(counts.get(sig, 0))
    lines = [
        f"⚠ Your LAST action did NOT execute (status={status}) — it collected no metrics.",
        f"  action: {json.dumps(act, default=str)}",
        f"  reason: {reason}",
    ]
    if status in ("invalid", "critic_rejected"):
        lines.append(
            f"  this exact action has failed {n}× and will be AUTO-BLACKLISTED at "
            f"{INVALID_SIGNATURE_BLACKLIST_THRESHOLD}×."
        )
    lines.append(
        "  DO NOT repeat it. Fix the stated reason (e.g. enable a missing "
        "dependency flag in its own trial first) or choose a different action."
    )
    return _repeated_block(lines)


_OPERATOR_DOMAIN_CRITIQUE_MARKERS = (
    "operator-domain",
    "operator domain",
    "operator-facing",
    "operator owned",
    "operator-owned",
    "operator approval",
    "human-owned",
    "human owned",
    "measurement trust boundary",
    "trust boundary",
    "outside the autopilot action space",
    "widening safety-gate",
    "safety-gate threshold",
    "baseline refresh",
    "instrument era",
    "instrument-era",
    "era row",
)


def _critique_issues(critique: Any) -> list[str]:
    try:
        return [str(issue) for issue in (getattr(critique, "issues", []) or [])]
    except Exception:
        return []


def _is_operator_domain_critique(critique: Any) -> bool:
    """Return true when the critic rejected work that belongs in operator review.

    The controller should not keep redrafting measurement-trust-boundary or
    operator-consent actions as ordinary AutoPilot actions. Persisting them in
    a separate outbox preserves the useful hypothesis without converting it into
    an eval-wasting invalid-action loop.
    """
    text = " ".join(
        [
            str(getattr(critique, "decision", "")),
            *_critique_issues(critique),
        ]
    ).lower()
    return any(marker in text for marker in _OPERATOR_DOMAIN_CRITIQUE_MARKERS)


def _append_operator_outbox_item(
    draft_action: dict[str, Any],
    critique: Any,
    trial_id: int,
    *,
    path: Path = OPERATOR_OUTBOX_PATH,
) -> bool:
    """Append one open operator-review item, deduped by action signature."""
    signature = _action_signature(draft_action)
    if path.exists():
        try:
            with open(path) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        row.get("status", "open") == "open"
                        and row.get("action_signature") == signature
                    ):
                        return False
        except OSError as exc:
            log.warning("Could not read operator outbox %s: %s", path, exc)

    issues = _critique_issues(critique)
    row = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "open",
        "kind": "critic_rejected_operator_domain",
        "source_trial": trial_id,
        "action": draft_action,
        "action_signature": signature,
        "critic_decision": str(getattr(critique, "decision", "reject")),
        "critic_issues": issues,
        "operator_prompt": (
            "Review whether this operator-domain hypothesis warrants a "
            "measurement-policy amendment, era row, or explicit runbook action. "
            "AutoPilot must not re-propose it as an autonomous action."
        ),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return True
    except OSError as exc:
        log.warning("Could not append operator outbox %s: %s", path, exc)
        return False


def _build_operator_outbox_feedback(
    path: Path = OPERATOR_OUTBOX_PATH,
    *,
    limit: int = OPERATOR_OUTBOX_RENDER_CAP,
) -> str:
    """Render open operator-domain items into the planner prompt."""
    if limit <= 0:
        return "  (disabled by AUTOPILOT_OPERATOR_OUTBOX_RENDER_CAP)"
    if not path.exists():
        return "  (none)"

    rows: deque[dict[str, Any]] = deque(maxlen=limit)
    try:
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("status", "open") == "open":
                    rows.append(row)
    except OSError as exc:
        return f"  (operator outbox unavailable: {exc})"

    if not rows:
        return "  (none)"

    lines = [
        "  Open operator-domain items from critic-rejected drafts "
        "(planner context only; NOT an action gate):"
    ]
    for row in rows:
        issue = "; ".join(str(x) for x in (row.get("critic_issues") or [])[:2])
        action_text = json.dumps(row.get("action", {}), sort_keys=True, default=str)
        lines.append(f"  - trial {row.get('source_trial', '?')}: {action_text[:180]}")
        if issue:
            lines.append(f"    critic: {issue[:220]}")
    lines.append(
        "  Do NOT re-propose these as autonomous actions; choose a schedulable "
        "AutoPilot action unless an operator has closed the outbox item."
    )
    return "\n".join(lines)


def _build_prior_planner_decision_digest(
    archive_path: Path = PLANNER_ARCHIVE_PATH,
    *,
    limit: int = PRIOR_DECISION_DIGEST_CAP,
) -> str:
    """Render bounded planner continuity from archive records, not chat resume."""
    if limit <= 0:
        return "  (disabled by AUTOPILOT_PRIOR_DECISION_DIGEST_CAP)"
    if not archive_path.exists():
        return "  (none yet — planner archive has no prior decisions)"

    try:
        tail: deque[str] = deque(maxlen=max(limit * 6, limit))
        with open(archive_path) as fh:
            for line in fh:
                if line.strip():
                    tail.append(line)
    except Exception as exc:
        return f"  (prior decision digest unavailable: {exc})"

    rows: list[str] = []
    for raw in reversed(tail):
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if record.get("type") == "planner_coordinator":
            critique = record.get("critique_decision") or "none"
            degraded = "degraded" if record.get("degraded") else "clean"
            fallback = str(record.get("fallback_reason") or "").strip()
            issues = record.get("critique_issues") or []
            issue_text = ""
            if isinstance(issues, list) and issues:
                issue_text = " issues=" + "; ".join(str(i)[:80] for i in issues[:2])
            line = (
                f"  - action={record.get('action_type') or '?'} "
                f"draft={record.get('draft_provider') or '?'} "
                f"critic={record.get('critic_provider') or 'none'} "
                f"verdict={critique} status={degraded}"
            )
            if fallback:
                line += f" fallback={fallback[:120]}"
            rows.append(line + issue_text)
        elif record.get("type") == "planner_provider_call" and not record.get("ok", True):
            line = (
                f"  - provider={record.get('provider') or '?'} "
                f"role={record.get('role') or '?'} "
                f"status={record.get('status') or 'failed'}"
            )
            error = str(record.get("error") or "").strip()
            if error:
                line += f" error={error[:120]}"
            rows.append(line)

        if len(rows) >= limit:
            break

    if not rows:
        return "  (none yet — no coordinator decisions in planner archive tail)"
    rows.reverse()
    return "\n".join(rows)


def _build_repo_readiness_advisory(
    pickup_path: Path | None = None,
    *,
    limit: int = 5,
) -> str:
    """Render a passive repo-readiness pickup artifact for planner context."""
    raw_path = str(pickup_path or os.environ.get(REPO_READINESS_PICKUP_ENV, "")).strip()
    if not raw_path:
        return f"  (disabled; set {REPO_READINESS_PICKUP_ENV}=<advisory pickup json> to include)"

    path = Path(raw_path).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"  (repo-readiness pickup unavailable: {exc})"

    mode = str(payload.get("mode") or "")
    if mode != "advisory_only" or bool(payload.get("authority_gate")):
        return (
            "  (repo-readiness pickup ignored: expected mode=advisory_only "
            "and authority_gate=false)"
        )

    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return "  (repo-readiness pickup has no candidate items)"

    lines = [
        "  Planner context only. This is NOT an acceptance gate and MUST NOT "
        "override owning handoffs, GitNexus impact, or measurement gates.",
    ]
    source_count = payload.get("source_item_count")
    item_count = payload.get("item_count")
    generated_at = payload.get("generated_at")
    meta: list[str] = []
    if generated_at:
        meta.append(f"generated_at={generated_at}")
    if source_count is not None:
        meta.append(f"source_items={source_count}")
    if item_count is not None:
        meta.append(f"shown_items={item_count}")
    if meta:
        lines.append("  " + ", ".join(meta))

    for item in items[: max(0, limit)]:
        if not isinstance(item, dict):
            continue
        priority = str(item.get("priority") or "?")
        repo = str(item.get("repo") or "?")
        criterion = str(item.get("criterion_id") or "?")
        objective = str(item.get("objective") or item.get("acceptance") or "").strip()
        item_id = str(item.get("id") or f"{repo}:{criterion}")
        objective_text = objective[:140] if objective else "(no objective)"
        lines.append(
            f"  - {priority} {item_id}: repo={repo} criterion={criterion}; {objective_text}"
        )

    rules = payload.get("pickup_rules")
    if isinstance(rules, list) and rules:
        lines.append("  Required before acting: " + "; ".join(str(r) for r in rules[:4]))
    else:
        lines.append(
            "  Required before acting: review owning handoff; run GitNexus impact; "
            "rerun the scorer after any patch."
        )
    return "\n".join(lines)


def _latest_model_gate_report_path(reports_dir: Path | None = None) -> Path | None:
    """Return the newest generated model gate report artifact, if present."""
    base = reports_dir or ORCH_ROOT / "orchestration" / "reports"
    candidates: list[Path] = []
    for pattern in MODEL_GATE_REPORT_GLOBS:
        candidates.extend(path for path in base.glob(pattern) if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _model_gate_evidence_brief(action: dict[str, Any]) -> str:
    evidence = action.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        return ""

    wanted_keys = (
        "latest_seq_trial_id",
        "latest_combined_E",
        "latest_required_E",
        "latest_fresh_eval",
        "latest_seq_state",
        "open_requirements",
        "telemetry_collection_reason",
        "telemetry_collection_blocker",
        "canary_role_sample_deficit",
        "canary_arm_volume_deficit",
        "canary_arm_balance_deficits",
    )
    parts: list[str] = []
    for key in wanted_keys:
        if key not in evidence:
            continue
        value = evidence.get(key)
        if isinstance(value, float):
            value_text = f"{value:.6g}"
        else:
            value_text = str(value)
        parts.append(f"{key}={value_text[:120]}")
    return "; ".join(parts[:6])


def _build_model_gate_advisory(
    report_path: Path | None = None,
    *,
    reports_dir: Path | None = None,
    limit: int = 4,
) -> str:
    """Render latest generated Fable gate next-actions for planner context.

    This intentionally consumes an existing report artifact instead of running
    model_gate_report.py in the planner loop. The report generator performs
    live process and evidence checks that are too expensive for every planner
    turn; this block only makes the latest durable snapshot visible.
    """
    path = report_path or _latest_model_gate_report_path(reports_dir)
    if path is None:
        return (
            "  (no model gate report artifact found; run "
            "uv run python scripts/autopilot/model_gate_report.py --json --strict)"
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stat = path.stat()
    except Exception as exc:
        return f"  (model gate advisory unavailable: {exc})"

    summary = payload.get("summary") if isinstance(payload, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    actions = payload.get("next_actions") if isinstance(payload, dict) else []
    if not isinstance(actions, list):
        actions = []

    now_s = datetime.now(timezone.utc).timestamp()
    age_s = max(0, int(now_s - stat.st_mtime))
    freshness = "fresh" if age_s <= MODEL_GATE_ADVISORY_MAX_AGE_S else "stale"
    lines = [
        "  Planner context only. This is NOT an acceptance gate and MUST NOT "
        "override owning handoffs, GitNexus impact, or measurement gates.",
        (
            f"  latest_artifact={path.name} age_s={age_s} freshness={freshness} "
            f"ready={bool(payload.get('ready'))}"
        ),
    ]

    active = summary.get("active_next_action_keys")
    blocked = summary.get("blocked_next_action_keys")
    if active:
        lines.append(f"  active_next_actions={active}")
    if blocked:
        lines.append(f"  blocked_next_actions={blocked}")

    if not actions:
        lines.append("  (report contains no next_actions)")
        return "\n".join(lines)

    status_rank = {"active": 0, "ready": 1, "blocked": 2}
    sorted_actions = sorted(
        (action for action in actions if isinstance(action, dict)),
        key=lambda action: (
            status_rank.get(str(action.get("status") or ""), 9),
            str(action.get("priority") or "P9"),
        ),
    )
    for action in sorted_actions[: max(0, limit)]:
        key = str(action.get("key") or "?")
        status = str(action.get("status") or "?")
        priority = str(action.get("priority") or "?")
        reason = str(action.get("reason") or "").replace("\n", " ").strip()
        line = f"  - {priority} {status} {key}"
        if reason:
            line += f": {reason[:180]}"
        evidence = _model_gate_evidence_brief(action)
        if evidence:
            line += f" [{evidence}]"
        elif status == "blocked":
            blocked_by = action.get("blocked_by")
            if isinstance(blocked_by, list) and blocked_by:
                line += f" [blocked_by={str(blocked_by[0])[:140]}]"
        lines.append(line)

    return "\n".join(lines)


def _baseline_pin_for_trial(gate: Any, eval_result: Any) -> dict[str, Any]:
    """EV-14e: capture the structured baseline pin a trial is about to be judged against.

    Read-only against the gate: it uses ``Baseline.pin_tier(register=False)`` (the
    EV-14c pin accessor), so it can neither register a measurement window nor move
    the reference. The rebaseline hold is mirrored rather than re-derived — when the
    gate is holding, the gate itself forces ``baseline_q = None``, so the pin records
    ``baseline_quality=None`` with ``suppressed_by="quality_rebaseline_hold"`` instead
    of a cross-era number the comparison never used.

    Never raises: a trial must not be lost because its provenance annotation failed.
    """
    try:
        baseline = getattr(gate, "baseline", None)
        if baseline is None or not hasattr(baseline, "pin_tier"):
            return {}
        tier = int(getattr(eval_result, "tier", 0) or 0)
        pin = baseline.pin_tier(tier, register=False)
        hold = bool(getattr(gate, "quality_rebaseline_required", False))
        if hold:
            suppressed = "quality_rebaseline_hold"
        elif pin.quality is None:
            suppressed = "no_same_tier_baseline"
        else:
            suppressed = ""
        return build_baseline_pin(
            tier=tier,
            baseline_quality=None if hold else pin.quality,
            candidate_quality=getattr(eval_result, "quality", None),
            baseline_revision=pin.revision,
            eval_quality_era=pin.eval_quality_era,
            autopilot_speed_era=getattr(baseline, "autopilot_speed_era", ""),
            per_suite_quality=dict(pin.per_suite_quality or {}),
            per_suite_counts=dict(pin.per_suite_counts or {}),
            baseline_path=str(getattr(baseline, "source_path", "") or ""),
            suppressed_by=suppressed,
        )
    except Exception as exc:  # noqa: BLE001 - provenance must never fail a trial
        log.warning("EV-14e baseline pin capture failed: %s: %s", type(exc).__name__, exc)
        return {"schema_version": 1, "source": "capture_error", "error": str(exc)[:200]}


def _record_reproposal_rate_windows(journal: Any) -> None:
    """VB-AP53-RATE: append closed-window re-proposal rate rows (fail-open, never alters a trial)."""
    try:
        reproposal_rate.record_closed_windows(journal)
    except Exception as exc:  # noqa: BLE001 - record_closed_windows is already fail-open
        log.debug("VB-AP53-RATE hook failed: %s", exc)


def _build_rejected_config_feedback(journal: Any) -> str:
    """AP-53: still-standing hard rejections of concrete configs, from the folded journal."""
    try:
        entries = (
            journal.entries_with_supersessions()
            if hasattr(journal, "entries_with_supersessions")
            else journal.all_entries()
        )
        block = rejected_mutation_ledger.render_rejected_configs_for_planner(list(entries))
    except Exception as exc:  # noqa: BLE001 - prompt context must never block planning
        log.debug("AP-53 rejected-config feedback failed: %s", exc)
        return ""
    return f"\n\n{block}" if block else ""


BASELINE_INFRA_FINGERPRINTS_STATE_KEY = "baseline_infra_fingerprints"


def _ap55_seed_rerun_forcing_enabled() -> bool:
    return os.environ.get("AUTOPILOT_AP55_SEED_RERUN", "").strip().lower() in {"1", "true", "yes", "on"}


def _ap55_attempt_key(reference: Mapping[str, Any]) -> str:
    return f"tier={reference.get('tier')}:reference={reference.get('latest_reference_trial_id')}"


def _ap55_seed_rerun_attempt_allowed(
    state: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    trial_counter: int,
) -> bool:
    """Bound regime-triggered re-runs that never land as a reference draw.

    A forced re-run whose eval comes back invalid/skipped is neither a reference nor
    a trusted trial, so the reference state alone would re-force it forever. Attempts
    are counted per reference (a landed draw changes the key and resets the count),
    spaced by AP55_REGIME_REFRESH_MIN_TRIALS trial ids, and capped.
    """
    rec = state.get(AP55_SEED_RERUN_ATTEMPTS_STATE_KEY)
    if not isinstance(rec, Mapping) or rec.get("key") != _ap55_attempt_key(reference):
        return True
    count = int(rec.get("count") or 0)
    if count >= max(1, AP55_SEED_RERUN_MAX_ATTEMPTS):
        log.warning(
            "AP-55 seed re-run NOT forced: %d attempts since reference %s never landed "
            "(cap AUTOPILOT_AP55_SEED_RERUN_MAX_ATTEMPTS=%d); promotion gate leg stays %s",
            count, reference.get("latest_reference_trial_id"),
            AP55_SEED_RERUN_MAX_ATTEMPTS, reference.get("regime_status"),
        )
        return False
    try:
        last = int(rec.get("last_trial_id"))
    except (TypeError, ValueError):
        return True
    if trial_counter - last < max(1, AP55_REGIME_REFRESH_MIN_TRIALS):
        log.info(
            "AP-55 seed re-run deferred: attempt %d at trial %d did not land a reference; "
            "next attempt after %d trials",
            count, last, AP55_REGIME_REFRESH_MIN_TRIALS,
        )
        return False
    return True


def _ap55_note_seed_rerun_attempt(
    state: dict[str, Any],
    reference: Mapping[str, Any],
    *,
    trial_counter: int,
) -> None:
    key = _ap55_attempt_key(reference)
    rec = state.get(AP55_SEED_RERUN_ATTEMPTS_STATE_KEY)
    count = int(rec.get("count") or 0) if isinstance(rec, Mapping) and rec.get("key") == key else 0
    state[AP55_SEED_RERUN_ATTEMPTS_STATE_KEY] = {
        "key": key,
        "count": count + 1,
        "last_trial_id": int(trial_counter),
        "reason": reference.get("reason", ""),
    }


def _infra_fingerprint_for_trial() -> dict[str, Any]:
    """AP-55: capture the infra regime (never raises, zero inference)."""
    try:
        return collect_infra_fingerprint()
    except Exception as exc:  # noqa: BLE001 - provenance must never fail a trial
        return {"schema_version": 1, "status": "capture_error", "error": str(exc)[:200]}


def _infra_comparability_for_trial(
    state: Mapping[str, Any],
    tier: int,
    trial_fingerprint: Mapping[str, Any],
    *,
    dispatch_fingerprint: Mapping[str, Any] | None = None,
    baseline_revision: Any = None,
) -> dict[str, Any]:
    """AP-55: compare this trial's regime with its tier baseline reference's regime.

    The reference fingerprint is recorded when a baseline is promoted
    (``state[BASELINE_INFRA_FINGERPRINTS_STATE_KEY][tier]``). A baseline seeded
    before AP-55 has none, so the verdict is UNVERIFIED — never COMPARABLE by
    default. A regime change between dispatch and eval completion is marked
    NON_COMPARABLE regardless of the baseline: the trial itself straddled regimes.
    """
    try:
        refs = state.get(BASELINE_INFRA_FINGERPRINTS_STATE_KEY)
        reference = refs.get(str(int(tier))) if isinstance(refs, Mapping) else None
        label = f"baseline_tier_{int(tier)}"
        if baseline_revision is not None:
            label += f"_rev{baseline_revision}"
        verdict = compare_infra_fingerprints(
            trial_fingerprint,
            reference if isinstance(reference, Mapping) else None,
            reference_label=label,
        )
        if dispatch_fingerprint is not None:
            drift = compare_infra_fingerprints(
                trial_fingerprint, dispatch_fingerprint, reference_label="trial_dispatch"
            )
            verdict["mid_trial"] = {
                "status": drift["status"],
                "differing_components": drift["differing_components"],
            }
            if drift["status"] == "NON_COMPARABLE":
                verdict["status"] = "NON_COMPARABLE"
                verdict["reason"] = (
                    "infra changed during trial: " + ",".join(drift["differing_components"])
                )
        return verdict
    except Exception as exc:  # noqa: BLE001
        return {"status": "UNVERIFIED", "reason": f"comparability error: {exc}"[:200]}


def _question_outcome_counts(question_results: Any) -> tuple[int, int]:
    """(correct, scored) over quality-admissible outcomes (CJ-8 exclusions apply)."""
    outcomes = _question_outcome_map(question_results)
    return sum(1 for ok in outcomes.values() if ok), len(outcomes)


def _ap55_promotion_gate_for_trial(
    journal: Any,
    *,
    tier: int,
    baseline_pin: Mapping[str, Any] | None,
    candidate_action: Mapping[str, Any],
    question_results: Any,
    trial_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    """AP-55 (b)+(c): same-regime seed re-run and batch homogeneity, before a winner.

    The batch is every trusted journal row of this tier judged against the same
    baseline revision (EV-14e pin) in this trial's regime, pooled by config
    fingerprint, plus this candidate. Rows without a pin or a fingerprint are
    excluded as they are — never back-filled. Mode: AUTOPILOT_AP55_PROMOTION_GATE
    (shadow by default: recorded, never binding). Never raises.
    """
    try:
        entries = (
            journal.entries_with_supersessions()
            if hasattr(journal, "entries_with_supersessions")
            else journal.all_entries()
        )
        revision = baseline_pin.get("baseline_revision") if isinstance(baseline_pin, Mapping) else None

        def _key(row: Any) -> str:
            snap = getattr(row, "config_snapshot", None)
            return _config_fingerprint(snap) if isinstance(snap, dict) and snap else ""

        return ap55_promotion_gate.promotion_gate(
            entries,
            tier=int(tier),
            baseline_revision=revision,
            candidate_key=_config_fingerprint(dict(candidate_action)),
            candidate_counts=_question_outcome_counts(question_results),
            candidate_fingerprint=trial_fingerprint,
            counts_fn=lambda row: _question_outcome_counts(
                (getattr(row, "eval_details", {}) or {}).get("question_results")
            ),
            key_fn=_key,
        )
    except Exception as exc:  # noqa: BLE001 - provenance must never fail a trial
        mode = ap55_promotion_gate.gate_mode()
        hold = mode != "shadow"
        return {"mode": mode, "status": "error", "error": str(exc)[:200],
                "hold": hold, "hold_reasons": ["gate_error"] if hold else [],
                "counterfactual": {m: {"hold": True, "hold_reasons": ["gate_error"]}
                                   for m in ("enforce", "strict")}}


def _record_baseline_infra_fingerprint(
    state: dict[str, Any],
    baseline_update: Any,
    trial_fingerprint: Mapping[str, Any],
) -> None:
    """AP-55: a promoted baseline carries the regime it was measured in."""
    if baseline_update is None or not getattr(baseline_update, "updated", False):
        return
    if not fingerprint_digest(trial_fingerprint):
        return
    refs = state.get(BASELINE_INFRA_FINGERPRINTS_STATE_KEY)
    if not isinstance(refs, dict):
        refs = {}
    refs[str(int(baseline_update.tier))] = dict(trial_fingerprint)
    state[BASELINE_INFRA_FINGERPRINTS_STATE_KEY] = refs


def _record_skip_trial(
    journal: Any,
    trial_id: int,
    action: dict[str, Any],
    species: str,
    status: str,
    reason: str,
    memory_count: int,
    *,
    bug_corrupted_by: str = "",
    bug_corrupted_reason: str = "",
    run_manifest: dict[str, Any] | None = None,
) -> None:
    """Journal a non-executing trial so it leaves durable residue (audit + planner).

    Uses pareto_status="skipped" and outcome_status to keep these out of the
    frontier/quality math while still recording that the trial number was
    consumed and why.
    """
    from experiment_journal import DeficiencyCategory

    deficiency = (
        DeficiencyCategory.INVALID_ACTION.value
        if status == "invalid"
        else DeficiencyCategory.DISPATCH_SKIPPED.value
    )
    entry = JournalEntry(
        trial_id=trial_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        species=species or "dispatch",
        action_type=action.get("type", ""),
        tier=0,
        quality=0.0,
        speed=0.0,
        cost=0.0,
        reliability=0.0,
        pareto_status="skipped",
        config_snapshot=dict(action),
        reasoning=json.dumps(action, default=str),
        memory_count=memory_count,
        failure_analysis=reason,
        deficiency_category=deficiency,
        outcome_status=status,
        bug_corrupted_by=bug_corrupted_by,
        bug_corrupted_reason=bug_corrupted_reason,
        run_manifest=dict(run_manifest or {}),  # AP-63(a)
    )
    journal.record(entry)
    _record_reproposal_rate_windows(journal)


def _record_rejected_draft(
    state: dict[str, Any],
    draft_action: dict[str, Any],
    critique: Any,
    trial_id: int,
) -> bool:
    """Feed a critic-rejected/revised draft into the invalid-action machinery.

    In draft_critique mode the SUBSTITUTED action (safe fallback / revised) is
    what dispatches, so the rejected draft would otherwise vanish — letting the
    planner re-draft it forever while the critic silently swaps in seed_batch.
    Record it like a non-executing action: fingerprint + count (persistent),
    surface it in the next prompt, auto-blacklist on repeat, and bump the
    consecutive-rejected-draft streak. Returns True if it was blacklisted.

    Note: the draft is NOT journaled as a trial (no trial number is consumed —
    the substituted action runs and is journaled instead); this only records the
    feedback residue so the rejected draft cannot bypass the loop.
    """
    issues = _critique_issues(critique)
    reason = "critic rejected: " + (
        "; ".join(issues) if issues else getattr(critique, "decision", "rejected")
    )
    sig = _action_signature(draft_action)
    if _is_observational_feedback_action(draft_action):
        rejected_observational = state.setdefault("critic_rejected_observational_signatures", {})
        rejected_observational[sig] = {
            "trial_id": trial_id,
            "action": draft_action,
            "reason": reason,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        log.warning(
            "Critic %s observational draft %s; recorded as advisory feedback "
            "without invalid-signature poisoning",
            getattr(critique, "decision", "rejected"),
            json.dumps(draft_action, default=str),
        )
        if _is_operator_domain_critique(critique):
            _append_operator_outbox_item(draft_action, critique, trial_id)
        return False
    sig_counts = state.setdefault("invalid_signature_counts", {})
    sig_counts[sig] = int(sig_counts.get(sig, 0)) + 1
    rejected_signatures = state.setdefault("critic_rejected_signatures", {})
    rejected_signatures[sig] = {
        "trial_id": trial_id,
        "action": draft_action,
        "reason": reason,
        "count": sig_counts[sig],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    state["last_invalid_action"] = draft_action
    state["last_invalid_reason"] = reason
    state["last_invalid_status"] = "critic_rejected"
    state["consecutive_rejected_drafts"] = int(state.get("consecutive_rejected_drafts", 0)) + 1

    blacklisted = False
    if sig_counts[sig] >= INVALID_SIGNATURE_BLACKLIST_THRESHOLD:
        append_blacklist(
            draft_action,
            trial_id,
            f"Auto-blacklisted: {sig_counts[sig]}× critic-rejected — {reason[:80]}",
            reason_class="critic_rejected",
        )
        blacklisted = True
    outboxed = False
    if _is_operator_domain_critique(critique):
        outboxed = _append_operator_outbox_item(draft_action, critique, trial_id)
    log.warning(
        "Critic %s the draft %s (substituted); recorded as feedback "
        "[signature seen %d×, consecutive rejected drafts=%d]%s%s",
        getattr(critique, "decision", "rejected"),
        json.dumps(draft_action, default=str),
        sig_counts[sig],
        state["consecutive_rejected_drafts"],
        " — BLACKLISTED" if blacklisted else "",
        " — OPERATOR_OUTBOXED" if outboxed else "",
    )
    return blacklisted


def _critic_rejected_signature_skip(
    action: dict[str, Any],
    state: dict[str, Any],
) -> SkipOutcome | None:
    """Reject exact repeats of a prior critic-rejected planner draft.

    Prompt feedback is advisory; this is the dispatch-side shield. It is exact
    signature keyed so a materially changed retry remains available, but the
    planner cannot burn another evaluation by emitting the same already-rejected
    action after a substituted fallback trial cleared last_invalid_action.
    """
    if not isinstance(action, dict):
        return None
    if _is_observational_feedback_action(action):
        return None
    if action.get("type") == "numeric_trial" and not (action.get("params") or {}):
        # Empty numeric params are an Optuna request, not the replay artifact.
        # NumericSwarm samples concrete values at dispatch and actions.py mutates
        # the action before journaling, so the apparent signature is not an exact
        # repeat in the material sense that this guard is meant to block.
        return None
    rejected = state.get("critic_rejected_signatures", {}) or {}
    if not isinstance(rejected, dict):
        return None
    signature = _action_signature(action)
    record = rejected.get(signature)
    if not isinstance(record, dict):
        return None
    reason = str(record.get("reason", "critic rejected this exact action"))
    trial = record.get("trial_id", "?")
    return SkipOutcome(
        status="invalid",
        reason=(
            "exact action signature was previously critic-rejected at "
            f"trial {trial}: {reason[:220]}; change a material field or choose "
            "a different action"
        ),
        action_type=str(action.get("type", "unknown")),
    )


def _force_metric_action_after_meta(
    action: dict[str, Any],
    state: dict[str, Any],
    rationale: dict[str, Any] | None = None,
    blacklist: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Replace repeated meta no-ops with a measured action."""
    if (
        action.get("type") in META_NOOP_ACTIONS
        and int(state.get("consecutive_meta_actions", 0)) > 0
    ):
        log.warning(
            "Planner proposed metric-free meta action '%s' after %d "
            "consecutive meta action(s); forcing seed_batch to restore "
            "measurement.",
            action.get("type"),
            int(state.get("consecutive_meta_actions", 0)),
        )
        forced, forced_rationale = _replace_blacklisted_seed_fallback(
            {"type": "seed_batch", "n_questions": SAFE_FALLBACK_SEED_N},
            blacklist or [],
            {
                **(rationale or {}),
                "meta_action_forced_metric_trial": True,
            },
            reason_label="meta-action replacement",
        )
        return forced, forced_rationale
    return action, rationale


def _env_float(name: str, default: float, *, minimum: float = 0.1) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


PAUSE_POLL_S = _env_float("AUTOPILOT_PAUSE_POLL_S", 1.0)
HEALTH_BACKOFF_S = _env_float("AUTOPILOT_HEALTH_BACKOFF_S", 10.0)
# Escalate a health-check outage after this many CONSECUTIVE failures. The count
# is only meaningful next to the period: one iteration costs HEALTH_BACKOFF_S
# (10s) plus the health_check(retries=2) probe itself, which spends ~2s on a
# refused connection (two attempts, each followed by its own 1s sleep) — so ~12s
# per retry, and 30 retries is ~6 minutes of continuous failure. That clears the
# longest restart wait this codebase sanctions (300s, scripts/server/
# stack_commands.py) with margin, so an ordinary slow API restart rides through
# without crying wolf, while a genuinely dead API is impossible to miss within
# ~6 minutes instead of never.
HEALTH_ESCALATE_AFTER = _env_int(
    "AUTOPILOT_HEALTH_ESCALATE_AFTER", DEFAULT_HEALTH_ESCALATE_AFTER
)


def _stream_points_to_path(stream: Any, path: Path) -> bool:
    try:
        stream_stat = os.fstat(stream.fileno())
        path_stat = path.stat()
        return stream_stat.st_dev == path_stat.st_dev and stream_stat.st_ino == path_stat.st_ino
    except OSError:
        return False


def _autopilot_logging_handlers(
    log_path: Path,
    stream: Any | None = None,
) -> list[logging.Handler]:
    stream = stream or sys.stderr
    handlers: list[logging.Handler] = [logging.StreamHandler(stream)]
    if not _stream_points_to_path(stream, log_path):
        handlers.append(logging.FileHandler(log_path, mode="a"))
    return handlers


# ── Controller Prompt Template ───────────────────────────────────

CONSTITUTION_PATH = SCRIPT_DIR / "constitution.md"
SYSTEM_CARD_PATH = SCRIPT_DIR / "system_card.md"
STACK_PRIORS_PATH = ORCH_ROOT / "orchestration" / "derived" / "stack_priors.yaml"
REPO_READINESS_PICKUP_ENV = "AUTOPILOT_REPO_READINESS_PICKUP"
STARTUP_ATTESTATION_PATHS = (
    ORCH_ROOT / "orchestration" / "model_registry.yaml",
    ORCH_ROOT / "orchestration" / "tool_registry.yaml",
    STACK_PRIORS_PATH,
    BLACKLIST_PATH,
)
MODEL_GATE_REPORT_GLOBS = (
    # New name FIRST — the generator is `scripts/autopilot/model_gate_report.py` as of the
    # 2026-08-04 de-FABLE rename, so reports it writes from now on carry the new prefix.
    # Matching only the old spelling would have made every future report invisible here
    # while the freshness check kept passing on the last pre-rename artifact: a stale-
    # evidence failure that looks exactly like "no new reports", not like a bug.
    "model_gate_report_*.json",
    "model_gate_*.json",
    # Old spelling still READ so the existing artifacts under orchestration/reports/
    # keep resolving. Same doctrine as the env-var alias immediately below.
    "fable5_gate_report_*.json",
    "fable5_gate_*.json",
)
MODEL_GATE_ADVISORY_MAX_AGE_S = int(
    # New name preferred; the old spelling is still READ so an operator or script
    # that exports it keeps working. Nothing sets it today (checked), but silently
    # ignoring a knob someone relies on is worse than carrying one alias.
    os.environ.get("AUTOPILOT_MODEL_GATE_ADVISORY_MAX_AGE_S")
    or os.environ.get("AUTOPILOT_FABLE_GATE_ADVISORY_MAX_AGE_S", "14400")
)


def _file_digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _startup_attestation_payload() -> dict[str, Any]:
    """Return the effective gate env and runtime config hash at process boot."""
    file_hashes: dict[str, str | None] = {
        str(path): _file_digest(path) for path in STARTUP_ATTESTATION_PATHS
    }
    digest = hashlib.sha256()
    for path, value in sorted(file_hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update((value or "missing").encode("utf-8"))
        digest.update(b"\0")
    gate_env = {
        key: os.environ.get(key, "")
        for key in sorted(
            set(AUTOPILOT_REQUIRED_GATE_ENV)
            | {
                "AUTOPILOT_PLANNER_PRIMARY",
                "AUTOPILOT_PLANNER_CRITIC",
                "AUTOPILOT_PLANNER_MODE",
                "AUTOPILOT_PLANNER_CRITIQUE_POLICY",
                "AUTOPILOT_PLANNER_SPEND_BREAKER",
                "AUTOPILOT_STEPPING_STONES",
                "AUTOPILOT_PLANNER_ROSTER_POLICY_ACTIVE",
                "AUTOPILOT_PLANNER_ROSTER_PATH",
                "AUTOPILOT_PLANNER_ROSTER_SHA256",
                "AUTOPILOT_CODEX_MODEL",
                "AUTOPILOT_CODEX_EFFORT",
                "AUTOPILOT_CODEX_CRITIC_MODEL",
                "AUTOPILOT_CODEX_CRITIC_EFFORT",
            }
        )
    }
    missing_or_mismatch = {
        key: {"expected": expected, "actual": os.environ.get(key, "")}
        for key, expected in AUTOPILOT_REQUIRED_GATE_ENV.items()
        if os.environ.get(key) != expected
    }
    try:
        validate_active_environment(os.environ)
    except PlannerRosterError as exc:
        missing_or_mismatch["AUTOPILOT_PLANNER_ROSTER"] = {
            "expected": "valid sealed 2-Codex/0-Claude roster",
            "actual": str(exc),
        }
    return {
        "schema_version": 1,
        "pid": os.getpid(),
        "gate_env": gate_env,
        "missing_or_mismatch": missing_or_mismatch,
        "p0_2_bridge": seq_p0_2_bridge_status(),
        "config_hash": digest.hexdigest(),
        "file_hashes": file_hashes,
    }


def _startup_gate_error(payload: Mapping[str, Any]) -> str:
    mismatches = payload.get("missing_or_mismatch") or {}
    rendered = json.dumps(mismatches, sort_keys=True)
    return (
        "ERROR: AutoPilot authority gate env mismatch; refusing direct start.\n"
        f"Use `{AUTOPILOT_AUTHORITY_LAUNCHER}` so the daemon starts with the "
        "supervisor and required authority environment.\n"
        f"Missing or mismatched env: {rendered}"
    )


def _enforce_startup_gate_env() -> None:
    """Fail closed before starting the daemon with incomplete authority env."""
    startup_attestation = _startup_attestation_payload()
    if not startup_attestation["missing_or_mismatch"]:
        return
    message = _startup_gate_error(startup_attestation)
    print(message, file=sys.stderr)
    raise SystemExit(2)


EPISODIC_INTEGRITY_CHECK = (
    Path(__file__).resolve().parents[2] / "scripts/maintenance/check_episodic_integrity.py"
)


EPISODIC_GATE_WAIT_S = float(os.environ.get("AUTOPILOT_EPISODIC_GATE_WAIT_S", "180"))
EPISODIC_GATE_POLL_S = 15.0


def _run_episodic_check(semantic: bool) -> dict | None:
    """Run the integrity checker; None if it could not run at all."""
    argv = [sys.executable, str(EPISODIC_INTEGRITY_CHECK), "--json"]
    if semantic:
        argv.append("--semantic")
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=600)
        return json.loads(proc.stdout)
    except Exception as exc:
        log.warning("episodic integrity check could not run (%s)", exc)
        return None


def _semantic_was_skipped(report: dict) -> bool:
    return any(
        c.get("check") == "semantic_self_match" and c.get("skipped")
        for c in report.get("checks", [])
    )


def _enforce_episodic_integrity_gate() -> None:
    """Fail closed if the episodic store cannot be SHOWN to be sound.

    WHY THIS GATE EXISTS
    --------------------
    From 2026-07-05 to 2026-07-27 the store's vector resolution was silently
    wrong and AutoPilot ran on it the whole time: every trial that consulted
    episodic memory received semantically random neighbours, and every component
    reported healthy because each was internally consistent. Twenty-two days of
    trials ran through a broken instrument without one alarm.

    WHY IT WAITS FOR THE EMBEDDERS RATHER THAN SHRUGGING
    ----------------------------------------------------
    An embedder outage is not a neutral "cannot verify" state — it is the
    condition that *causes* the corruption. ``use_fallback`` defaults to True in
    EmbeddingConfig and EmbedderPoolConfig, and every live site builds a bare
    ``TaskEmbedder()``, so with BGE down a write does not fail: it silently
    stores a SHA-256 pseudo-vector (measured 89.0% all-zero, 2.8% NaN, 8.1%
    well-formed-but-meaningless). Worse, until the ``degenerate_vectors`` check
    existed the *only* detector for the well-formed ones was semantic
    self-match, which needs the very embedders that were down.

    So: wait out a boot window, then refuse. A trial run on an unverifiable
    store produces evidence that looks valid — which is exactly what the last 22
    days of trials were. The write path now also raises
    ``DegenerateEmbeddingError`` at the chokepoint, so with BGE down AutoPilot
    could not record memories anyway.

    Overrides (both logged loudly):
      ``AUTOPILOT_SKIP_EPISODIC_GATE=1``     skip entirely
      ``AUTOPILOT_EPISODIC_GATE_WAIT_S=N``   change the embedder boot window
    """
    if os.environ.get("AUTOPILOT_SKIP_EPISODIC_GATE") == "1":
        log.warning(
            "EPISODIC INTEGRITY GATE BYPASSED via AUTOPILOT_SKIP_EPISODIC_GATE=1 — "
            "memory-derived results from this run are NOT trustworthy"
        )
        return
    if not EPISODIC_INTEGRITY_CHECK.exists():
        log.warning(
            "episodic integrity checker missing at %s; store is UNVERIFIED",
            EPISODIC_INTEGRITY_CHECK,
        )
        return

    report = _run_episodic_check(semantic=True)
    if report is None:
        log.warning("episodic integrity check unavailable; store is UNVERIFIED")
        return

    # Metadata failures are structural — retrying cannot help, so fail now
    # rather than burning the embedder boot window first.
    hard_failures = [
        c
        for c in report.get("checks", [])
        if c.get("pass") is False and c.get("check") != "semantic_self_match"
    ]
    if hard_failures:
        _episodic_gate_fail(hard_failures)

    # Only the semantic check can be transiently unavailable. Give the
    # embedders a boot window before treating it as terminal.
    deadline = time.monotonic() + EPISODIC_GATE_WAIT_S
    while _semantic_was_skipped(report) and time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        log.warning(
            "episodic gate: embedders unreachable, the DECISIVE check cannot run; "
            "retrying in %.0fs (%.0fs of the boot window left)",
            min(EPISODIC_GATE_POLL_S, remaining),
            remaining,
        )
        time.sleep(min(EPISODIC_GATE_POLL_S, max(remaining, 0)))
        retried = _run_episodic_check(semantic=True)
        if retried is not None:
            report = retried

    for c in report.get("checks", []):
        if c.get("pass") is False:
            log.error("episodic %s FAILED: %s", c["check"], c["detail"])

    if _semantic_was_skipped(report):
        print(
            "ERROR: the episodic embedders stayed unreachable for "
            f"{EPISODIC_GATE_WAIT_S:.0f}s; refusing to start AutoPilot.\n\n"
            "This is not merely 'cannot verify'. With the embedders down, every\n"
            "episodic write falls back to a SHA-256 pseudo-vector (89% all-zero,\n"
            "2.8% NaN), so running now would actively corrupt the store — and the\n"
            "write path will raise DegenerateEmbeddingError anyway.\n\n"
            "Start the embedders (ports 8090-8095), then retry.\n"
            "  orchestrator_stack.py status\n"
            "Wait longer: AUTOPILOT_EPISODIC_GATE_WAIT_S=<seconds>\n"
            "Run anyway on an unverified store: AUTOPILOT_SKIP_EPISODIC_GATE=1",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if not report.get("ok"):
        _episodic_gate_fail([c for c in report.get("checks", []) if c.get("pass") is False])

    log.info(
        "episodic integrity gate: PASS (%d checks, semantic verified)",
        len(report.get("checks", [])),
    )


def _episodic_gate_fail(failed: list) -> None:
    print(
        "ERROR: episodic memory integrity check FAILED; refusing to start AutoPilot.\n"
        + "\n".join(f"  [FAIL] {c['check']}: {c['detail']}" for c in failed)
        + "\n\nA trial run on a broken store produces evidence that looks valid.\n"
        "Repair first: scripts/maintenance/repair_faiss_id_map.py (desync) or\n"
        "scripts/maintenance/reseed_episodic_store.py (mapping/vector defects).\n"
        "See handoffs/active/episodic-memory-integrity.md.\n"
        "To run anyway on a known-degraded store: AUTOPILOT_SKIP_EPISODIC_GATE=1",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _format_deep_eval_tier_options() -> str:
    tiers = [str(tier) for tier in sorted(controller_io.DEEP_EVAL_TIERS)]
    if not tiers:
        return "(none)"
    if len(tiers) == 1:
        return tiers[0]
    if len(tiers) == 2:
        return " or ".join(tiers)
    return f"{', '.join(tiers[:-1])}, or {tiers[-1]}"


CONTROLLER_PROMPT_TEMPLATE = """\
You are the AutoPilot meta-reasoning controller for an LLM orchestration stack.
Your job: analyze current system state and propose the SINGLE best next action.
This is a non-interactive JSON controller call, not a Claude Code Plan Mode
session. Do not create files, do not write a plan under ~/.claude/plans, and do
not use planner-side mutation tools. If the best move is operator-facing, say so
briefly in reasoning and still emit the closest valid AutoPilot action block.

## Controller Constitution (durable human-authored policy)

{constitution}

## Generated System Card (live facts, regenerated from repository state)

{system_card}

## Current State

### Pareto Archive
{pareto_summary}

### Pareto Frontier Geometry
{pareto_geometry}

### Evidence Power and Sequential Candidate Status
{planner_evidence}

### Journal Trustworthiness (bug_corrupted filtering)
{journal_trustworthiness}

### Hypotheses Under Test (last 3 trustworthy trials)
{hypotheses_under_test}

### Operator Hypotheses (operator-stated priors, still open)
{operator_hypotheses_block}

### Settled Experimental Ground (Vidya read-only; advisory for proposal selection)
{vidya_settled_ground_block}

### KV Quant Benchmark Evidence (Vidya read-only; bench observations, not serving or promotion warrant)
{vidya_kvq_evidence_block}

If the chosen action uses a KV-quant claim above, list its exact claim ID in
`vidya_claim_ids` in `autopilot_rationale`. Use an empty list when the evidence
did not influence the action. A claim shown in this prompt is not automatically
support for your action.

### Experiment Journal (bounded recent entries)
{journal_summary}

### Seeder Status
{seeder_status}

### Recent Batch Telemetry (autopilot adapts n_questions to fit budget)
{batch_telemetry}

### Species Effectiveness
{species_effectiveness}

### System Health
- Orchestrator: {health_status}
- Memory count: {memory_count}
- Q-value converged: {converged}

### Slot Memory (KV cache usage — consider slot_compact if tokens > 4000)
{slot_memory}

### Action Availability
{action_availability}

### Fable 5 Gate Advisory (latest generated report, non-authority)
{model_gate_advisory}

### Higher-Tier Objective Pressure (same-tier, non-authority)
{higher_tier_pressure}

### Eval Coverage Pressure (planner-learning, non-authority)
{eval_coverage_pressure}

### Outcome Progress Pressure (planner-learning, non-authority)
{outcome_progress_pressure}

### StrategyStore Planner Hints (refreshed each planner turn)
Planner tool boundary: StrategyStore rows mentioning tools, REPL, CALL, or
tool-use refer to orchestrator/model execution inside AutoPilot actions and
evals. They are not permission for this controller process to call planner-side
tools. The planner process is read-only: while drafting the JSON action, use
only Read/Grep/Glob for repository inspection when needed. Never use Bash, Edit,
MultiEdit, Write, NotebookEdit, apply_patch, or any other planner-side
mutation/execution tool to satisfy a hint. If a mutation is warranted, return an
AutoPilot action such as code_mutation, prompt_mutation, structural_experiment,
or numeric_trial and let the orchestrator dispatch it.
{planner_strategy_hints}

### Repo-Readiness Advisory Pickup (default-off, non-authority)
{repo_readiness_advisory}

### Species Budget
{budget}

### Suite Quality Trends (last 10 evals)
{suite_quality_trends}

### Recent Insights (cross-species, structured per action_type, bug-corrupted excluded)
{insights_structured}

### Exploration mode
Stagnation signal: {stagnation_signal}

{exploration_block}

### Short-Term Memory (accumulated learnings this session)
{short_term_memory}

### Prior Planner Decisions (bounded archive digest — replaces chat-session resume)
{prior_planner_decisions}

### Self-Criticism from Last Trial
{last_criticism}

### Active Model Performance Signatures
{model_signatures}

### Blacklisted Configurations
{blacklist_text}

### Operator Outbox (critic-rejected operator-domain hypotheses)
{operator_outbox_feedback}

### Feature Flags (live state + dependency rules — read before any structural_experiment)
{feature_flags_block}

### Last Non-Executing Action (validator/dispatch feedback — MUST address before re-proposing)
{last_invalid_feedback}

### Plot Paths (reference for trend analysis)
{plot_paths}

## Action Guidelines

1. If memories < 500: ALWAYS prioritize seeding (seed_batch)
2. If Q-values converged and models not trained: trigger train_routing_models
3. If models trained and not enabled: try structural_experiment with routing
   features — but ONLY for a flag whose dependencies are ALL currently ON (see
   "Feature Flags" above). One flag per trial (AP-9): e.g. graph_router requires
   specialist_routing, so if specialist_routing is OFF, enable IT first (its own
   trial), then graph_router next trial. Never re-propose a flag the validator
   just rejected (see "Last Non-Executing Action").
4. If stagnating (hv_slope < 0.001): try prompt_mutation or widen numeric search
5. If quality regression after changes: rollback to last good checkpoint
6. Consider the species budget allocation when choosing actions
7. If any slot shows >4000 tokens cached, consider slot_compact to free KV memory
   (use keep_ratio=0.3, target a live port from Slot Memory or the generated system card)

## Available Actions

The schemas below are already filtered by the "Action Availability" section for
this turn. Do not emit an action type unless its schema appears in this section.

Respond with EXACTLY ONE action in a ```json:autopilot_actions block:

{available_action_schemas}

Include brief reasoning before the action block.

After the action block, ALSO emit a second fenced block tagged
`autopilot_rationale` carrying the chosen action's falsifier and its self-scored
rubric. This sidecar is observability-only — a missing or malformed block will
not abort the trial, but populating it lets future planner passes grade new
candidates against still-open hypotheses. `operator_hypothesis` is OPTIONAL and
omitted unless this trial's own outcome resolves an operator prior. Do not supply
trial ids — the trial that just ran is attached as the evidence. A refuted
operator hypothesis is a first-class result; record it:

```json:autopilot_rationale
{{"falsifier": "<one-line predicted outcome whose absence invalidates this hypothesis>",
 "vidya_claim_ids": [],
 "rubric_scores": {{"info_gain": <1-5>, "coherence": <1-5>, "usefulness": <1-5>,
   "synthesis_note": "<optional one-line on fusion / cleaner model>"}},
 "operator_hypothesis": {{"id": "<id from Operator Hypotheses, ONLY if resolved>",
   "status": "confirmed|refuted|inconclusive",
   "note": "<what the outcome showed>"}}}}
```
"""


def _format_available_action_schemas(action_types: list[str]) -> str:
    """Render only currently selectable action schemas for the controller prompt."""
    ordered = [action_type for action_type in action_types if action_type]
    numeric_surface_options = "|".join(_configured_numeric_surfaces()) or "(none)"
    numeric_param_options = _format_numeric_surface_param_options()
    code_targets = ", ".join(CODE_MUTATION_ALLOWLIST)
    new_file_roots = ", ".join(new_file_mutation_root_labels())
    deep_eval_tier_options = _format_deep_eval_tier_options()
    schemas = {
        "seed_batch": (
            '- Seed: {{"type": "seed_batch", "n_questions": 10-50, "suites": ["coder","math",...]}}'
        ),
        "numeric_trial": (
            '- Numeric: {{"type": "numeric_trial", "surface": "'
            f'{numeric_surface_options}", "params": {{"surface.param": value}}}}\n'
            "  (Model-authored planner actions must include exactly one explicit "
            "param. Empty params are reserved for internal deterministic fallbacks.\n"
            f"  Valid params by surface: {numeric_param_options}\n"
            "  There is no numeric_trial tool_activation_threshold knob; do not "
            "invent one.)"
        ),
        "prompt_mutation": (
            '- Prompt: {{"type": "prompt_mutation", "file": "frontdoor.md", '
            '"mutation": "targeted_fix|compress|few_shot_evolution", '
            '"description": "..."}}'
        ),
        "gepa_optimize": (
            '- GEPA: {{"type": "gepa_optimize", "file": "frontdoor.md", '
            '"max_evals": 50, "description": "..."}}\n'
            "  (AP-19: Evolutionary prompt optimization via GEPA — runs ~50 "
            "evals internally, returns best candidate)"
        ),
        "code_mutation": (
            '- Code: {{"type": "code_mutation", "file": "src/escalation.py", '
            '"mutation": "targeted_fix", "description": "..."}}\n'
            f"  (Existing-file mutations: ONLY files in allowlist: {code_targets}. "
            'For scaffold/schema evolution, use mutation "new_file" only under '
            f"these roots: {new_file_roots}; memory schema-evolution files must "
            "be default-inert — a static AST screen rejects imports, classes, "
            "underscore-prefixed names and raise/with/while/lambda in that lane, "
            "so propose plain constants and pure functions only.)"
        ),
        "structural_experiment": (
            '- Structural: {{"type": "structural_experiment", '
            '"flags": {{"feature_name": true/false}}}}'
        ),
        "consult_gate_probe": (
            '- Consult gate probe: {{"type": "consult_gate_probe", '
            '"task_suite": "targeted", "turns": 10, "tier": 3}}\n'
            "  (Runs baseline vs blanket consult vs targeted-gate edit-transaction "
            "turns; records consult calls/skips/reruns/quality/latency. Use this "
            "to optimize review_before_commit gating on hard workflow tiers.)"
        ),
        "structural_prune": (
            '- Prune: {{"type": "structural_prune", "file": "frontdoor.md", '
            '"block": "## Section Name", "description": "..."}}\n'
            "  (Delete an instruction block from a .md prompt file — accepted "
            "only if quality >= baseline AND instruction_token_ratio decreases)"
        ),
        "slot_compact": (
            '- Compact: {{"type": "slot_compact", "port": 0, "slot_id": 0, '
            '"keep_ratio": 0.3, "scorer": "expected_attention", '
            '"keep_first": 5, "n_future": 128}}\n'
            "  (AM KV compaction — replace port=0 with a live port from Slot "
            "Memory or the generated system card before emitting the action. "
            "Use after long-context queries to free memory. Evaluates quality "
            "post-compact.)"
        ),
        "train_routing_models": (
            '- Train: {{"type": "train_routing_models", "min_memories": 500}}\n'
            "  (min_memories: integer 1-100000; default 500. Do NOT set it to "
            "the current memory_count — the validator REJECTS any value above "
            "100000.)"
        ),
        "distill_skillbank": (
            '- Distill skillbank: {{"type": "distill_skillbank", '
            '"teacher": "claude", "categories": ["routing"]}}'
        ),
        "reset_memories": (
            '- Reset: {{"type": "reset_memories", "keep_seen": true, "keep_skills": true}}'
        ),
        "deep_eval": (
            '- Deep eval: {{"type": "deep_eval", "tier": 3}}\n'
            "  (Choose tier 3 when expert/hard workflow coverage or frontier "
            "evidence is thin; choose tier 2 for comprehensive validation or "
            f"W8 promotion-eval evidence. Supported tiers: {deep_eval_tier_options}. "
            "Do NOT include target_trial, suites, baseline_recheck, or "
            "instrumentation fields.)"
        ),
        "rollback": ('- Rollback: {{"type": "rollback", "to_checkpoint": "production_best"}}'),
        "distill_knowledge": (
            '- Distill knowledge: {{"type": "distill_knowledge", "last_n": 10}}\n'
            "  (Run every ~5 trials to extract insights from recent outcomes "
            "into strategy memory)"
        ),
    }
    lines = [schemas[action_type] for action_type in ordered if action_type in schemas]
    if lines:
        return "\n".join(lines).replace("{{", "{").replace("}}", "}")
    return "- Pause: no currently selectable autonomous action schema is available."


def _format_numeric_surface_param_options() -> str:
    try:
        from species.numeric_swarm import SURFACES as _NS_SURFACES
    except Exception:
        return "unavailable (use only params already visible in prior valid numeric_trial rows)"

    parts: list[str] = []
    for surface in _configured_numeric_surfaces():
        specs = _NS_SURFACES.get(surface, [])
        labels: list[str] = []
        for spec in specs:
            name = str(getattr(spec, "name", "") or "")
            if not name:
                continue
            ptype = str(getattr(spec, "param_type", "float") or "float")
            low = getattr(spec, "low", "?")
            high = getattr(spec, "high", "?")
            labels.append(f"{name} {ptype}[{low},{high}]")
        parts.append(f"{surface}: {', '.join(labels) if labels else '(no params discovered)'}")
    return "; ".join(parts) if parts else "(none)"


def _read_guidance_file(path: Path, missing_label: str) -> str:
    try:
        return path.read_text()
    except OSError:
        return f"({missing_label} not found)"


def _render_system_card(state: dict[str, Any] | None = None) -> str:
    """Render live controller facts, failing closed if generation breaks."""
    try:
        from gen_system_card import generate_system_card

        return generate_system_card(ORCH_ROOT, state_override=state)
    except Exception as exc:
        return (
            "# AutoPilot Generated System Card\n\n"
            "SYSTEM CARD GENERATION FAILED.\n\n"
            "## Degraded Stack Guidance\n\n"
            f"- generator_error: {type(exc).__name__}: {exc}\n"
            "- Live role, port, tier, throughput, baseline, and trust-boundary facts "
            "are unavailable.\n"
            "- Do not use checked-in `system_card.md`, historical program text, "
            "handoffs, memories, or old logs as authoritative stack truth.\n"
            "- Choose an observational or no-stack-change action, or pause for "
            "operator repair, until `scripts/autopilot/gen_system_card.py --check` "
            "passes again.\n"
        )


# ── Exploration block (stagnation-gated creative-prompt fragment) ─

_EXPLORATION_LEAN = """\
Before emitting your single action, briefly enumerate up to 3 alternatives you
considered. For each, give a one-line reason for rejection OR pick it as
your action.
"""


_EXPLORATION_RICH_TEMPLATE = """\
The system is STAGNATING. Run the constrained-creativity protocol:

1. **Generate {n} candidate actions.** Each must be:
   - Non-obvious relative to the recent default (last 3 trials' action_types).
   - Consistent with the Pareto geometry, blacklist, model signatures, and
     the last 30 trials' evidence above — do NOT optimize for weirdness.
   - Inspired (not constrained) by the under-used action types below — these
     are directions explored ≤1 time in the last {window} trials:

{tail_seeds}

2. **For each candidate, write one line each:**
   - why-low-typicality (what makes this non-default given current evidence)
   - falsifier (the concrete observation that would invalidate it)

3. **Score each candidate 1–5 on three axes:**
   - **info_gain** — novelty + falsifiability + explanatory compression
     (does resolving this trial reduce posterior uncertainty more than the
     obvious next step?)
   - **coherence** — consistency with facts, signatures, blacklist, the
     last 30 trials, and the still-open hypotheses listed below
   - **usefulness** — expected Pareto improvement per unit compute, minus
     risk of being decorative nonsense

4. **Synthesize.** If your top-2 candidates can be FUSED into one action
   that dominates both (e.g. a numeric_trial whose params encode a
   structural_experiment's hypothesis), prefer the fusion. The best
   creative idea reduces uncertainty, not adds complexity.

5. **Quote, don't regenerate.** The chosen action's rubric_scores in the
   rationale sidecar must be copied verbatim from your candidate table —
   no re-grading after the fact.

### Still-open hypotheses (carry an explicit falsifier, not yet resolved)
{unfalsified}

"""


_RECOVERY_ONLY_ACTIONS = {
    "distill_knowledge",
    "reset_memories",
    "rollback",
}


def _slot_compaction_viable(slot_memory_text: str) -> bool:
    """True when any production slot currently has cached tokens."""
    return any(
        int(m.group(1)) > 0 for m in re.finditer(r"(\d+)\s+tokens cached", slot_memory_text or "")
    )


def _type_only_blacklisted_actions(blacklist: list[dict[str, Any]]) -> dict[str, str]:
    """Map action_type -> reason for exact type-only blacklist entries."""
    blocked: dict[str, str] = {}
    for entry in blacklist:
        pattern = entry.get("pattern", {})
        if not isinstance(pattern, dict):
            continue
        if set(pattern) == {"type"} and isinstance(pattern.get("type"), str):
            blocked[pattern["type"]] = entry.get("reason", "blacklisted")
    return blocked


def _build_action_availability(
    *,
    journal: ExperimentJournal,
    known_actions: list[str],
    memory_count: int,
    converged: bool,
    slot_memory_text: str,
    blacklist: list[dict[str, Any]],
    suppressed_numeric_surfaces: set[str] | None = None,
    w8_replay_pressure_text: str = "",
) -> tuple[str, list[str], list[str]]:
    """Return prompt text plus viable exploration and selectable action types."""
    blocked: dict[str, str] = {}
    cautions: dict[str, str] = {}
    priority: list[str] = []

    w8_candidate_generation_active = _w8_candidate_generation_pressure(w8_replay_pressure_text)
    w8_replay_pressure_active = _w8_replay_pressure_active(w8_replay_pressure_text)
    if w8_candidate_generation_active:
        priority.append(
            "W8 candidate generation is the active strict blocker. For this turn, "
            "candidate generation actions are ONLY an explicit or Optuna-suggested "
            "numeric_trial that journals applied params, or a one-flag "
            "structural_experiment. Do not emit seed_batch, deep_eval, or "
            "structural_prune; they are deferrals unless a seq due action is forcing "
            "them."
        )
        blocked["seed_batch"] = (
            "W8 candidate generation is active; seed_batch cannot create replayable "
            "W8 candidate evidence"
        )
        blocked["deep_eval"] = (
            "W8 candidate generation is active; deep_eval validates candidates but "
            "does not create a replayable candidate"
        )
        blocked["structural_prune"] = (
            "W8 candidate generation is active; structural_prune is not replayable "
            "W8 candidate evidence"
        )
    elif w8_replay_pressure_active:
        priority.append(
            "W8 replay/confirmation evidence is active. For this turn, avoid "
            "seed_batch, deep_eval, and structural_prune unless a seq due action "
            "has already forced a specific replay or fresh-eval. Prefer a "
            "replayable numeric_trial or one-flag structural_experiment with a "
            "W8 falsifier."
        )
        blocked["seed_batch"] = (
            "W8 replay pressure is active; seed_batch cannot replay or strengthen "
            "accumulating W8 candidate evidence"
        )
        blocked["deep_eval"] = (
            "W8 replay pressure is active; generic deep_eval is validation-only "
            "and should wait for the seq-specific fresh-eval path"
        )
        blocked["structural_prune"] = (
            "W8 replay pressure is active; structural_prune is not replayable W8 candidate evidence"
        )

    blocked.update(_type_only_blacklisted_actions(blacklist))

    if memory_count < 500:
        blocked["train_routing_models"] = (
            f"needs >=500 routing memories; current memory_count={memory_count}"
        )
    elif not converged:
        cautions["train_routing_models"] = (
            "routing memories exist, but the current seeder session is not "
            "converged yet; avoid training as a stagnation escape hatch"
        )
    # Validator caps train_routing_models.min_memories at 100000 (controller_io.py).
    # When the live corpus already exceeds the cap, say so explicitly so the planner
    # does NOT infer "use the current memory_count" and cross the hidden schema limit
    # (root cause of the #776 invalid-action pause, 2026-06-11).
    if memory_count > 100_000 and "train_routing_models" not in blocked:
        _cap_note = (
            f"min_memories is capped at 100000 — do NOT set it to the live "
            f"memory_count ({memory_count}); valid range 1-100000 (default 500)"
        )
        _existing = cautions.get("train_routing_models")
        cautions["train_routing_models"] = f"{_existing}; {_cap_note}" if _existing else _cap_note

    if not _slot_compaction_viable(slot_memory_text):
        blocked["slot_compact"] = (
            "no queried slot currently has cached tokens to compact; this is not "
            "evidence that the eval instrument or host is contaminated"
        )

    seed_exhaustion = _seed_fallback_exhaustion_reason(blacklist)
    if seed_exhaustion and "seed_batch" not in blocked:
        blocked["seed_batch"] = (
            "all configured measured seed fallback candidates are blacklisted; "
            f"last reason: {seed_exhaustion}"
        )

    suppressed_numeric_surfaces = set(suppressed_numeric_surfaces or set())
    if suppressed_numeric_surfaces:
        available_numeric_surfaces = _configured_numeric_surfaces()
        if available_numeric_surfaces:
            cautions["numeric_trial"] = (
                "convention-suppressed numeric surfaces are unavailable: "
                + ", ".join(sorted(suppressed_numeric_surfaces))
                + "; use only the numeric surfaces shown in the action schema"
            )
        else:
            blocked["numeric_trial"] = (
                "all numeric surfaces are suppressed by planner conventions: "
                + ", ".join(sorted(suppressed_numeric_surfaces))
            )

    cautions["reset_memories"] = "destructive recovery action; do not use for ordinary stagnation"
    cautions["rollback"] = (
        "recovery-only action; use only after a concrete regression from a known-good point"
    )
    cautions["distill_knowledge"] = (
        "metric-free meta action; do not use to break a measurement stall"
    )

    try:
        insights = journal.insights_structured(n=120, exclude_bug_corrupted=True)
    except Exception:
        insights = {}

    for action_type in ("prompt_mutation", "train_routing_models"):
        info = insights.get(action_type)
        if not info:
            continue
        if info.get("successes", 0) == 0 and info.get("failures", 0) >= 1:
            cautions[action_type] = (
                f"recent evidence is negative: trials {info.get('trials_supporting', [])} "
                f"produced only failures ({info.get('observation', '')})"
            )

    lines = []
    if priority:
        lines.append("Priority pressure:")
        for item in priority:
            lines.append(f"- {item}")
    if blocked:
        lines.append("Currently unavailable for active constraints:")
        for action_type, reason in sorted(blocked.items()):
            lines.append(f"- `{action_type}`: {reason}")
    if cautions:
        lines.append("Use only with a concrete new falsifier:")
        for action_type, reason in sorted(cautions.items()):
            lines.append(f"- `{action_type}`: {reason}")
    if not lines:
        lines.append("(no action-specific availability constraints detected)")

    try:
        lines.append(build_action_availability_section(load_capability_registry()))
    except Exception as exc:
        lines.append(f"Capability registry levers (generated): unavailable ({exc})")

    viable_tail_actions = [
        action_type
        for action_type in known_actions
        if action_type not in blocked
        and action_type not in _RECOVERY_ONLY_ACTIONS
        and not (action_type == "train_routing_models" and not converged)
    ]
    selectable_actions = [
        action_type for action_type in known_actions if action_type not in blocked
    ]
    return "\n".join(lines), viable_tail_actions, selectable_actions


def _dispatch_allowed_action_types(
    selectable_action_types: list[str] | None,
    *,
    seq_due_bypassed_planner: bool,
    seq_fresh_eval_context: dict[str, Any] | None,
    seq_baseline_draw_reference: dict[str, Any] | None,
    seq_candidate_replay_context: dict[str, Any] | None,
    multitier_validation_context: dict[str, Any] | None = None,
    seq_gate_preflight: dict[str, Any] | None = None,
) -> list[str] | None:
    """Return the final dispatch allowlist for ordinary planner-selected actions.

    The planner prompt uses ``selectable_action_types`` as the active/shadow
    boundary. Final dispatch must enforce the same boundary, but existing
    sequential due actions, forced replays/fresh evals, and seq-gate deferral
    replacements are internal policy actions governed by their own gates rather
    than planner availability text.
    """
    if (
        seq_due_bypassed_planner
        or seq_fresh_eval_context is not None
        or seq_baseline_draw_reference is not None
        or seq_candidate_replay_context is not None
        or multitier_validation_context is not None
        or (
            isinstance(seq_gate_preflight, dict)
            and seq_gate_preflight.get("status") == "deferred"
            and isinstance(seq_gate_preflight.get("replacement_action"), dict)
        )
    ):
        return None
    if selectable_action_types is None:
        return None
    return list(selectable_action_types)


def _w8_candidate_generation_pressure(text: str) -> bool:
    """Return True when W8 has no replayable accumulating candidate."""
    normalized = str(text or "").lower()
    if "w8 replay pressure:" not in normalized:
        return False
    return "no accumulating candidate exists" in normalized or (
        "accumulating candidate" in normalized
        and "0/" in normalized
        and "are replayable" in normalized
    )


def _w8_replay_pressure_active(text: str) -> bool:
    """Return True when the planner evidence asks this turn to serve W8."""
    normalized = str(text or "").lower()
    if "w8 replay pressure:" not in normalized:
        return False
    return any(
        marker in normalized
        for marker in (
            "confirmed candidate(s) await fresh",
            "no accumulating candidate exists",
            "accumulating candidate(s) are replayable",
        )
    )


def _w8_candidate_generation_deferral_reason(action: dict[str, Any]) -> str | None:
    """Return why an action cannot create replayable W8 candidate evidence."""
    action_type = str(action.get("type") or "")
    if action_type == "numeric_trial":
        # Empty params are acceptable at dispatch time: NumericSwarm fills Optuna
        # params and actions.py journals the applied params before W8 recording.
        return None
    if action_type == "structural_experiment":
        flags = action.get("flags")
        if isinstance(flags, dict) and flags:
            return None
        return "structural_experiment_missing_flags"
    return f"unreplayable_action={action_type or 'unknown'}"


def _replace_w8_candidate_generation_deferral(
    action: dict[str, Any],
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None,
    *,
    trial_counter: int,
    w8_replay_pressure_text: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Convert W8-blocked deferral actions into replayable candidate generation."""
    if not _w8_replay_pressure_active(w8_replay_pressure_text):
        return action, rationale
    reason = _w8_candidate_generation_deferral_reason(action)
    if reason is None:
        return action, rationale
    replacement, fallback_reason = _first_unblacklisted_numeric_trial_action(
        blacklist,
        trial_counter=trial_counter,
    )
    if replacement is None:
        log.warning(
            "W8 candidate generation pressure is active but %s cannot be replaced: %s",
            reason,
            fallback_reason,
        )
        return action, rationale
    next_rationale = {
        **(rationale or {}),
        "w8_candidate_generation_replaced": True,
        "w8_candidate_generation_original": dict(action),
        "w8_candidate_generation_reason": reason,
        "w8_candidate_generation_replacement": dict(replacement),
        "falsifier": (
            (rationale or {}).get("falsifier")
            or "W8 candidate generation replacement fails to produce replayable seq evidence"
        ),
    }
    log.warning(
        "W8 candidate generation pressure replaced %s action %s with numeric_trial fallback %s.",
        reason,
        json.dumps(action, default=str),
        json.dumps(replacement, default=str),
    )
    return replacement, next_rationale


def _repair_critic_reject_fallback_for_w8(
    action: dict[str, Any],
    blacklist: list[dict[str, Any]],
    rationale: dict[str, Any] | None,
    *,
    trial_counter: int,
    w8_replay_pressure_text: str,
) -> tuple[dict[str, Any], dict[str, Any] | None, SkipOutcome | None, bool]:
    """Keep critic fallback paths aligned with the active W8 replayability gate."""
    if not isinstance(action, dict):
        # No dispatchable action to repair — the planner guard already degraded it.
        return action, rationale, None, False
    action, rationale, seed_skip = _replace_exhausted_critic_seed_fallback(
        action,
        blacklist,
        rationale,
        trial_counter=trial_counter,
    )
    if seed_skip is not None:
        return action, rationale, seed_skip, False

    action, rationale = _replace_w8_candidate_generation_deferral(
        action,
        blacklist,
        rationale,
        trial_counter=trial_counter,
        w8_replay_pressure_text=w8_replay_pressure_text,
    )
    repaired = (
        _w8_replay_pressure_active(w8_replay_pressure_text)
        and _w8_candidate_generation_deferral_reason(action) is None
    )
    if repaired:
        rationale = {
            **(rationale or {}),
            "critic_reject_loop_repaired_by_w8_candidate": True,
        }
    return action, rationale, None, repaired


def _build_exploration_block(
    journal: ExperimentJournal,
    archive: ParetoArchive,
    known_actions: list[str],
) -> tuple[str, str]:
    """Return (exploration_block_text, stagnation_signal_text).

    Selects between the lean fragment (default) and the rich constrained-
    creativity fragment when at least one stagnation signal fires:
      - hv_slope_10 strictly below STAGNATION_HV_EPS
      - trustworthy trial count < 5 (low-signal regime)
      - last STAGNATION_STREAK trials share an action_type
    """
    reasons: list[str] = []

    # Pareto-hypervolume slope (already computed by archive.geometry()).
    # The threshold auto-calibrates from recent hv_slope noise where possible;
    # falls back to STAGNATION_HV_EPS (the hand-tuned constant) early on.
    hv_slope_10 = None
    try:
        geom = archive.geometry()
        hv_slope_10 = geom.get("hv_slope_10") if isinstance(geom, dict) else None
    except Exception:
        pass
    eps = STAGNATION_HV_EPS
    try:
        if hasattr(archive, "hv_slope_noise_floor"):
            eps = archive.hv_slope_noise_floor(floor_default=STAGNATION_HV_EPS)
    except Exception:
        eps = STAGNATION_HV_EPS
    if hv_slope_10 is not None and hv_slope_10 < eps:
        reasons.append(f"hv_slope_10={hv_slope_10:+.5f} < eps={eps:.5f}")

    # Trustworthiness low-signal flag.
    try:
        trust = journal.trustworthiness_score()
        if trust.get("low_signal"):
            reasons.append(f"trustworthy={trust.get('trustworthy', 0)} < 5")
    except Exception:
        pass

    # Action-type streak over the last STAGNATION_STREAK trials.
    try:
        recent = journal.recent(STAGNATION_STREAK)
        if len(recent) == STAGNATION_STREAK:
            types = {e.action_type for e in recent}
            if len(types) == 1:
                reasons.append(
                    f"last {STAGNATION_STREAK} trials all action_type={next(iter(types))}"
                )
    except Exception:
        pass

    if not reasons:
        return _EXPLORATION_LEAN, "none (lean prompt)"

    # Rich fragment: tail seeds + unfalsified hypotheses.
    try:
        tail = journal.tail_action_candidates(
            known_action_types=known_actions,
            last_n=TAIL_WINDOW,
            n_sample=TAIL_SEED_COUNT,
        )
    except Exception:
        tail = []
    if tail:
        tail_text = "\n".join(f"  - {t}" for t in tail)
    else:
        tail_text = "  (no under-used action types — every action_type seen recently)"

    try:
        unfalsified = journal.unfalsified_hypotheses(n=5)
    except Exception:
        unfalsified = []
    if unfalsified:
        unfalsified_text = "\n".join(
            f"  #{tid}: {hyp[:160]}\n     falsifier: {fal[:160]}" for tid, hyp, fal in unfalsified
        )
    else:
        unfalsified_text = "  (no recent trials with explicit falsifiers yet)"
    # Operator-stated priors participate in the creativity protocol's coherence
    # comparison, but remain clearly labelled as priors with no measured evidence.
    try:
        from operator_hypotheses import still_open as _still_open_operator_hypotheses

        operator_open = _still_open_operator_hypotheses()
    except Exception:  # noqa: BLE001 - the always-rendered block carries the loud alarm
        operator_open = []
    if operator_open:
        unfalsified_text += "\n" + "\n".join(
            f"  [operator prior: {item.id} | evidence_trial_ids=[]]: "
            f"{item.hypothesis[:160]}\n     falsifier: {item.falsifier[:160]}"
            for item in operator_open[:5]
        )

    block = _EXPLORATION_RICH_TEMPLATE.format(
        n=CREATIVITY_N,
        window=TAIL_WINDOW,
        tail_seeds=tail_text,
        unfalsified=unfalsified_text,
    )
    return block, "; ".join(reasons)


# ── State Management ─────────────────────────────────────────────


# State / blacklist / signatures helpers moved to state_store.py (2026-05-22 refactor).
# Wrappers below preserve the original autopilot.py API by supplying STATE_PATH,
# BLACKLIST_PATH, and the model-quality-signatures path.

_MODEL_SIGNATURES_PATH = ORCH_ROOT / "orchestration" / "model_quality_signatures.yaml"
_MODEL_DESCRIPTORS_PATH = ORCH_ROOT / "orchestration" / "model_descriptors.yaml"


def _maybe_reimport_pareto_from_journal(
    archive: "ParetoArchive",
    journal: "ExperimentJournal",
    trial_id: int,
) -> bool:
    """Re-add a single journal entry to the Pareto archive if missing.

    Per handoffs/active/autopilot-exogenous-restart-resilience.md Section 5.7.
    Handles the historical corruption window where the journal advanced but
    the derived state-cache archive was not persisted → on restart the journal
    has the entry but the on-disk Pareto archive does not.

    Returns True if the entry was re-imported, False otherwise.

    Edge cases:
      - SKIP if no JournalEntry matches trial_id.
      - SKIP if entry.bug_corrupted_by is non-empty (placeholders + tagged
        trials must never enter the archive).
      - SKIP if the archive already has an entry with this trial_id.
      - For valid entries: construct ParetoEntry from journal fields and
        call archive.update() — let the dominance check re-classify;
        do NOT preserve the JournalEntry's stale pareto_status.
    """
    entries = (
        journal.entries_with_supersessions()
        if hasattr(journal, "entries_with_supersessions")
        else journal.all_entries()
    )
    entry = next((e for e in entries if e.trial_id == trial_id), None)
    if entry is None:
        log.info("Pareto re-import: no journal entry for trial %d", trial_id)
        return False
    if entry.bug_corrupted_by:
        log.info(
            "Pareto re-import: trial %d is bug_corrupted_by=%s, skipping",
            trial_id,
            entry.bug_corrupted_by,
        )
        return False
    # Trusted within-noise rows (mad_noise / reproduction_confirmed) are managed as
    # robust-median REPRESENTATIVES, not raw per-trial points. Re-importing one as a single
    # noisy sample would contradict that policy, so skip — the representative is rebuilt from
    # the persisted reproduction cluster on the next reproduction.
    # Decision (c): clean rows stamped as frontier representatives are cluster members too.
    excl_by = (entry.eval_details or {}).get("learning_exclusion", {}).get("by", "")
    if excl_by in BENIGN_LEARNING_EXCLUSIONS or row_is_representative_member(asdict(entry)):
        log.info(
            "Pareto re-import: trial %d is a representative-cluster member (%s) — "
            "representative-managed, not raw-re-imported.",
            trial_id,
            excl_by or FRONTIER_ADMISSION_REPRESENTATIVE,
        )
        return False
    # Already in archive? (use the private _all_entries list — it's the
    # only enumeration of every observed trial; the public surface exposes
    # only frontier-related views, and re-importing a duplicate would
    # double-count in archive statistics.)
    existing_ids = {e.trial_id for e in getattr(archive, "_all_entries", [])}
    if trial_id in existing_ids:
        log.info("Pareto re-import: trial %d already in archive", trial_id)
        return False
    p_entry = ParetoEntry(
        trial_id=entry.trial_id,
        objectives=spec_for(entry.tier).objectives_from_row(
            {
                "quality": entry.quality,
                "speed": entry.speed,
                "cost": entry.cost,
                "reliability": entry.reliability,
            }
        )
        or (entry.quality, entry.speed, -entry.cost, entry.reliability),
        config_snapshot=entry.config_snapshot,
        git_tag=entry.git_tag,
        eval_tier=entry.tier,
        reasoning=(entry.reasoning or "")[:200],
        parent_trial=entry.parent_trial,
        memory_count=entry.memory_count,
        active_flags=list(entry.active_flags or []),
        species=entry.species,
        timestamp=entry.timestamp,
    )
    new_status = archive.update(p_entry)
    log.info("Pareto re-import: trial %d added (status=%s)", trial_id, new_status)
    # The startup journal-authority sync persists the reconstructed archive
    # after recovery finishes. Do not write a one-off archive snapshot here;
    # it can diverge from the folded append-only journal view.
    return True


def _recover_from_in_flight_trial(
    state: dict[str, Any],
    journal: "ExperimentJournal",
    archive: "ParetoArchive",
    trial_counter: int,
) -> int:
    """Apply the Phase 6b in_flight_trial recovery sequence.

    Returns the new trial_counter. Mutates `state` in place but does NOT
    save_state() — caller owns persistence so save can be paired with
    other startup-time state mutations.

    No-op (returns trial_counter unchanged) if state["in_flight_trial"]
    is None.

    Two recovery cases (handoff Section 5.7):
      (a) journal_max >= prior_in_flight.trial_id → trial was journaled
          before the crash. Bump trial_counter past it and attempt to
          re-import a missing Pareto entry from the journal.
      (b) journal_max < prior_in_flight.trial_id → trial died BEFORE
          journal.record. Write an AUTOPILOT_KILLED placeholder so the
          planner sees the gap, and bump trial_counter past it.

    On exit, state["in_flight_trial"] is None.
    """
    prior_in_flight = state.get("in_flight_trial")
    if prior_in_flight is None:
        return trial_counter

    journal_max = journal.next_trial_id() - 1
    prior_tid = int(prior_in_flight.get("trial_id", -1))
    log.warning(
        "Autopilot recovery: detected in_flight_trial marker "
        "(trial_id=%d, host_pid=%s, host_started_at=%s). "
        "Journal max trial_id=%d.",
        prior_tid,
        prior_in_flight.get("host_pid"),
        prior_in_flight.get("host_started_at"),
        journal_max,
    )
    if journal_max >= prior_tid:
        new_counter = max(trial_counter, journal_max + 1)
        log.info(
            "Recovery: trial %d was journaled before crash; bumping trial_counter %d → %d",
            prior_tid,
            trial_counter,
            new_counter,
        )
        trial_counter = new_counter
        state["trial_counter"] = trial_counter
        try:
            _maybe_reimport_pareto_from_journal(archive, journal, prior_tid)
        except Exception as exc:
            log.warning("Pareto re-import for trial %d failed: %s", prior_tid, exc)
    else:
        log.warning(
            "Recovery: trial %d died BEFORE journal.record. Writing AUTOPILOT_KILLED placeholder.",
            prior_tid,
        )
        try:
            placeholder = JournalEntry(
                trial_id=prior_tid,
                timestamp=datetime.now(timezone.utc).isoformat(),
                species="(killed)",
                action_type=(prior_in_flight.get("action") or {}).get("type", "unknown"),
                tier=0,
                quality=0.0,
                speed=0.0,
                cost=0.0,
                reliability=0.0,
                pareto_status="dominated",
                failure_analysis=(
                    f"Autopilot process killed before journal.record() "
                    f"(prior host_pid={prior_in_flight.get('host_pid')}, "
                    f"prior host_started_at={prior_in_flight.get('host_started_at')}, "
                    f"died at trial_id={prior_tid})."
                ),
                bug_corrupted_by="autopilot_killed_mid_trial",
                bug_corrupted_reason="incomplete trial; no eval evidence available",
                deficiency_category="autopilot_killed_mid_trial",
                # AP-63(a): the manifest the killed dispatch was bound to (drift was
                # already refused above by _reject_in_flight_manifest_drift).
                run_manifest=_in_flight_run_manifest(state, prior_tid),
            )
            journal.record(placeholder)
            trial_counter = max(trial_counter, prior_tid + 1)
            state["trial_counter"] = trial_counter
        except Exception as exc:
            log.error(
                "Failed to write AUTOPILOT_KILLED placeholder for trial %d: %s",
                prior_tid,
                exc,
            )
            # Fail closed: the in-flight marker is the only durable evidence
            # that this trial still requires recovery.  Clearing it after a
            # failed append would silently create an unaccounted journal gap.
            raise RuntimeError(
                f"Could not append interrupted-trial placeholder for trial {prior_tid}"
            ) from exc
    state["in_flight_trial"] = None
    return trial_counter


def _run_manifest_source_paths() -> dict[str, Path]:
    """Return the source set that defines an AutoPilot trial's control path."""
    return {
        "autopilot": Path(__file__),
        "controller_io": Path(controller_io.__file__),
        "eval_tower": SCRIPT_DIR / "eval_tower.py",
    }


def _run_manifest_evaluator() -> dict[str, str]:
    return {"class": "EvalTower", "url": ORCHESTRATOR_URL}


def _in_flight_run_manifest(state: dict[str, Any], trial_id: int) -> dict[str, Any]:
    """Return the dispatch-bound run manifest for ``trial_id`` (AP-63(a)).

    Only the manifest the in-flight WAL marker bound BEFORE dispatch is stamped onto
    the journal row. A marker for another trial, a legacy marker without a manifest,
    or no marker at all (the trial never dispatched) yields ``{}`` — a manifest built
    at write time would attest to sources the dispatch never ran.
    """
    in_flight = state.get("in_flight_trial")
    if not isinstance(in_flight, dict):
        return {}
    try:
        if int(in_flight.get("trial_id", -1)) != int(trial_id):
            return {}
    except (TypeError, ValueError):
        return {}
    manifest = in_flight.get("run_manifest")
    return copy.deepcopy(manifest) if isinstance(manifest, dict) else {}


def _reject_in_flight_manifest_drift(state: dict[str, Any]) -> None:
    """Fail closed before recovering a trial from a changed run environment."""
    in_flight = state.get("in_flight_trial")
    if not isinstance(in_flight, dict):
        return
    manifest = in_flight.get("run_manifest")
    if manifest is None:  # Legacy marker: preserve the existing recovery path.
        return
    if not isinstance(manifest, dict):
        reasons = ["malformed-manifest"]
    else:
        reasons = manifest_drift_reasons(
            manifest,
            source_paths=_run_manifest_source_paths(),
            evaluator=_run_manifest_evaluator(),
        )
    if not reasons:
        return
    detail = ",".join(reasons)
    state["paused"] = True
    state["_dispatch_deficiency"] = f"run_manifest_drift:{detail}"
    save_state(state)
    raise RuntimeError("Refusing in-flight trial recovery after run-manifest drift: " + detail)


def _default_state() -> dict[str, Any]:
    return {
        "trial_counter": 0,
        "session_id": None,
        "paused": False,
        "species_budget": SpeciesBudget().as_dict(),
        "td_errors": [],
        "seeder_state": {},
        # 2026-05-23 Phase 6b — autopilot self-crash recovery markers.
        # in_flight_trial is set immediately BEFORE dispatch_action and
        # cleared only AFTER the final atomic save_state. On startup,
        # _recover_in_flight_trial inspects it: if non-None, the prior
        # autopilot instance died mid-trial. See cmd_start's recovery
        # block for the exact taxonomy of cases handled.
        # autopilot_fleet_started_at is bumped on every autopilot start
        # so operators (and future watchdogs) can detect whether the
        # currently-running instance is the same one that recorded a
        # given trial.
        "in_flight_trial": None,
        "autopilot_fleet_started_at": None,
    }


def load_state() -> dict[str, Any]:
    return _load_state_impl(STATE_PATH, _default_state)


def _normalize_state_before_save(state: dict[str, Any]) -> None:
    if not state.get("paused"):
        state.pop("pause_reason", None)
    # 2026-08-01 AP-37 removal migration. load_state() returns the on-disk JSON
    # verbatim (state_store.load_state does NOT merge _default_state), so a key
    # dropped from _default_state would otherwise live forever in an existing
    # autopilot_state.json. The detector is gone; drop its state blob at the next
    # save so the file stops carrying an all-null "guardrail" that reads as
    # evidence of health. Idempotent, and a no-op on fresh state.
    state.pop("diversity_stall_state", None)


def _contrastive_trace_outcome(
    *,
    pareto_status: str,
    verdict: Any,
    bug_corrupted_by: str = "",
    outcome_status: str = "ok",
) -> str:
    """Return the MH-7 contrastive label for a completed trial, or empty."""
    if bug_corrupted_by or outcome_status != "ok":
        return ""
    passed = bool(getattr(verdict, "passed", verdict))
    if not passed:
        return "failure"
    if pareto_status == "frontier":
        return "success"
    return ""


def _bsv3_conflict_policy() -> str:
    """Default-off policy for BSV-3 ledger/incumbent state promotion."""
    raw = os.environ.get(BSV3_CONFLICT_POLICY_ENV, "").strip().lower()
    if raw in {"", "0", "false", "no", "off", "none"}:
        return "off"
    if raw in {"observe", "block", "review"}:
        return raw
    log.warning("Invalid %s=%r; using off", BSV3_CONFLICT_POLICY_ENV, raw)
    return "off"


def _bsv3_conflict_policy_decision(
    conflict_report: Mapping[str, Any] | None,
    *,
    policy: str | None = None,
) -> dict[str, Any]:
    """Decide whether BSV-3 may promote its ledger/incumbent state.

    This deliberately governs only BSV diagnostic state. It does not alter the
    SafetyGate verdict, ParetoArchive admission, blacklists, baseline updates,
    or action dispatch.
    """
    normalized_policy = (policy or _bsv3_conflict_policy()).strip().lower()
    if normalized_policy not in {"off", "observe", "block", "review"}:
        normalized_policy = "off"
    severity = str((conflict_report or {}).get("severity") or "none")
    conflict_count = int((conflict_report or {}).get("conflict_count") or 0)
    withhold = False
    if normalized_policy == "block":
        withhold = severity == "blocking"
    elif normalized_policy == "review":
        withhold = severity in {"watch", "blocking"}
    return {
        "version": "bsv-3-conflict-policy-v1",
        "enabled": normalized_policy != "off",
        "policy": normalized_policy,
        "severity": severity,
        "conflict_count": conflict_count,
        "ledger_update_allowed": not withhold,
        "incumbent_update_allowed": not withhold,
        "scope": "bsv_ledger_incumbent_state_only",
        "reason": (
            f"policy={normalized_policy} withheld BSV state promotion for severity={severity}"
            if withhold
            else f"policy={normalized_policy} allows BSV state promotion for severity={severity}"
        ),
    }


def _update_contrastive_trace_state(
    state: dict[str, Any],
    tower: Any,
    *,
    trace_text: str,
    trial_id: int,
    species: str,
    action_type: str,
    pareto_status: str,
    verdict: Any,
    bug_corrupted_by: str = "",
    failure_analysis: str = "",
    eval_result: Any = None,
    outcome_status: str = "ok",
) -> None:
    """Store labeled trace examples for the next PromptForge mutation context."""
    outcome = _contrastive_trace_outcome(
        pareto_status=pareto_status,
        verdict=verdict,
        bug_corrupted_by=bug_corrupted_by,
        outcome_status=outcome_status,
    )
    if not outcome or not str(trace_text or "").strip():
        return
    updater = getattr(tower, "update_contrastive_trace_bank", None)
    if not callable(updater):
        return
    if outcome == "failure":
        reason = failure_analysis
    else:
        tier = getattr(eval_result, "tier", "?")
        quality = getattr(eval_result, "quality", 0.0)
        speed = getattr(eval_result, "speed", 0.0)
        reason = f"T{tier} frontier q={quality:.3f} s={speed:.1f}"
    try:
        bank = updater(
            state.get("contrastive_trace_bank"),
            trace_text=trace_text,
            outcome=outcome,
            trial_id=trial_id,
            species=species,
            action_type=action_type,
            reason=reason,
        )
        state["contrastive_trace_bank"] = bank
        formatter = getattr(tower, "capture_contrastive_traces", None)
        if callable(formatter):
            state["contrastive_traces"] = formatter(
                k_success=2,
                k_failure=2,
                trace_bank=bank,
            )
        ir_builder = getattr(tower, "build_critic_trace_ir", None)
        if callable(ir_builder):
            trace_ir = ir_builder(
                trace_bank=bank,
                trial_id=trial_id,
                failure_summary=failure_analysis if outcome == "failure" else "",
                k_success=2,
                k_failure=2,
            )
            if trace_ir.get("trace_examples"):
                state["critic_trace_ir"] = trace_ir
                ir_formatter = getattr(tower, "format_critic_trace_ir", None)
                if callable(ir_formatter):
                    state["critic_trace_ir_prompt"] = ir_formatter(trace_ir)
    except Exception as exc:  # trace feedback must never disrupt trial completion
        log.debug("Contrastive trace update skipped for trial %d: %s", trial_id, exc)


def save_state(
    state: dict[str, Any],
    *,
    merge_control: bool = False,
    _lock: bool = True,
) -> None:
    """Persist autopilot state under the cross-process H4 write lock.

    Single-writer discipline: ``autopilot_state.json`` is a whole-file JSON
    rewritten by 5+ processes; atomic ``os.replace`` stops torn reads but NOT
    lost updates. Every write therefore serializes on ``state_write_lock``.

    ``merge_control=True`` (daemon periodic / trial-end / lifecycle save): while
    holding the lock, RE-READ the on-disk out-of-band CONTROL fields (paused /
    pause_reason / _in_cache_flush) and merge them into ``state`` BEFORE writing,
    so a dashboard, host_health, config_applicator, or operator pause set
    out-of-band while a trial was in flight survives the daemon's whole-file
    save. The daemon keeps ownership of trial/frontier/pareto fields — only the
    clearly out-of-band control flags in ``_EXTERNAL_CONTROL_FIELDS`` merge.

    ``_lock=False``: the caller already holds ``state_write_lock(STATE_PATH)``
    (e.g. ``cmd_pause``, which wraps its own load->modify->save so the whole
    read-modify-write is atomic). flock is NOT reentrant across two fds in one
    process, so nested acquisition would fail open after the timeout — pass
    False to skip re-acquiring the lock the caller already holds.
    """

    def _write() -> None:
        if merge_control:
            merged = _merge_external_control_fields(state)
            if merged:
                log.info(
                    "Merged out-of-band control fields under lock before save: %s",
                    ", ".join(merged),
                )
        _normalize_state_before_save(state)
        _save_state_impl(STATE_PATH, state)

    if _lock:
        with state_write_lock(STATE_PATH):
            _write()
    else:
        _write()


# Control-plane fields owned OUT-OF-BAND of the autopilot daemon: a dashboard
# pause click, the host_health / config_applicator cache-flush pause, and the
# operator `autopilot.py pause` all write these directly to disk while the
# daemon holds a long-lived in-memory state dict across a whole trial. When the
# daemon re-reads under the write lock before a save it merges ONLY these — never
# counters / frontier / in-flight metadata (those stay daemon-owned). Note:
# ``pause_reason`` is written solely by the dashboard (never by the daemon,
# which only pops it), so it is safe to treat as out-of-band-owned.
#
# ``paused`` alone cannot say WHOSE pause is on disk, so the pause-lease fields
# (``pause_owner`` / ``pause_token`` / ``pause_collision``) travel with it. They
# MUST be listed here: ``_save_state_impl`` writes the whole file from the
# daemon's in-memory dict, so a lease key the daemon never learned about would be
# dropped at the next trial-end save — and a dropped token reads as "superseded"
# to the applicator that is holding it. They are always written as an explicit
# ``None`` rather than popped, so a release is a VALUE change the merge below
# (which only looks at keys present on disk) can actually see.
_EXTERNAL_CONTROL_FIELDS: tuple[str, ...] = (
    "paused",
    "pause_reason",
    "_in_cache_flush",
) + PAUSE_LEASE_FIELDS


def _merge_external_control_fields(
    state: dict[str, Any],
    disk_state: dict[str, Any] | None = None,
) -> list[str]:
    """Preserve operator control fields changed while a trial was in-flight.

    The loop keeps a long-lived in-memory state dict during dispatch/eval. An
    external ``autopilot.py pause`` writes ``paused=True`` to disk, but the
    trial-end save path can otherwise overwrite it with stale ``paused=False``.
    Merge only control-plane fields, never counters or in-flight metadata.

    Callers that persist afterward must do so under ``state_write_lock`` (the
    daemon reaches this via ``save_state(..., merge_control=True)``) so the
    re-read and the write are one atomic critical section.
    """
    if disk_state is None:
        try:
            disk_state = load_state()
        except Exception as exc:
            log.warning("Failed to reload control fields before state save: %s", exc)
            return []

    changed: list[str] = []
    for key in _EXTERNAL_CONTROL_FIELDS:
        if key in disk_state and disk_state.get(key) != state.get(key):
            state[key] = disk_state[key]
            changed.append(key)
    return changed


def clear_halt_latch_on_resume_edge(state: dict[str, Any]) -> dict[str, Any]:
    """Clear the skip/meta halt latch when the loop observes a resume. Daemon-only.

    Called from the iteration top on the ``paused`` True→False edge. The latch is
    ``consecutive_skip_actions`` / ``last_invalid_*`` (skip breaker),
    ``consecutive_meta_actions`` (meta breaker) and the ``_dispatch_deficiency`` /
    ``_meta_halt_reason`` markers. Leaving it set means the very next non-executing
    action re-trips ``skip_streak >= MAX_CONSECUTIVE_SKIP`` and the operator's resume
    looks like it never happened.

    WHY IT MUST HAPPEN HERE. Every latched field is daemon-owned: this loop holds it
    in one in-memory dict for the whole run and rewrites the file wholesale at every
    save, merging back only ``_EXTERNAL_CONTROL_FIELDS``. The dashboard button and
    ``cmd_resume`` used to clear the skip latch from their own processes and report
    success; that write was destroyed at this loop's next save and the halt came back
    (the meta latch got this treatment on 2026-05-31; the skip latch did not until
    2026-08-12). Out-of-band writers may still clear it when the daemon is DOWN —
    :func:`state_ownership.clear_halt_latch` decides which case it is and reports the
    other honestly rather than faking a clear.
    """
    latch = clear_halt_latch(STATE_PATH, state)
    if latch["outcome"] == "cleared":
        log.info(
            "Resume after pause: cleared the %s halt latch (fields: %s)",
            latch.get("deficiency") or "unattributed",
            ", ".join(latch["fields"]),
        )
        save_state(state)
    return latch


def _archive_entry_count(payload: object) -> int:
    if not isinstance(payload, dict):
        return 0
    entries = payload.get("all_entries")
    return len(entries) if isinstance(entries, list) else 0


def _snapshot_scope_matches(
    payload: Mapping[str, Any],
    *,
    exclude_before_ts: float | None,
    deinflate_before_ts: float | None,
    deinflate_factor: float,
) -> bool:
    """A snapshot is authority only for the era scope it was folded under (W3).

    Snapshots record ``exclusions.exclude_before_ts``; a snapshot folded without
    the live epoch exclusion (e.g. an automatic segment snapshot) would otherwise
    hand pre-epoch entries to the live frontier. Deinflation is not recorded at
    archive level, so any active deinflation forces the full replay.
    """
    if deinflate_before_ts is not None and deinflate_factor != 1.0:
        return False
    exclusions = payload.get("exclusions")
    recorded = exclusions.get("exclude_before_ts") if isinstance(exclusions, Mapping) else None
    if recorded is None or exclude_before_ts is None:
        return recorded is None and exclude_before_ts is None
    try:
        return float(recorded) == float(exclude_before_ts)
    except (TypeError, ValueError):
        return False


def _sync_segment_snapshot_scope(journal: Any, state: Mapping[str, Any]) -> None:
    """W3: fold automatic segment snapshots under the live archive-authority scope.

    Active deinflation is not recorded in an archive payload, so it disables
    segment snapshots rather than writing one the authority path must refuse.
    """
    if not hasattr(journal, "segment_snapshot_scope"):
        return
    try:
        deinflate_before_ts, deinflate_factor, exclude_before_ts = (
            _archive_epoch_params_from_state(state)
        )
        if deinflate_before_ts is not None and deinflate_factor != 1.0:
            journal.segment_snapshots = False
            return
        journal.segment_snapshot_scope = {
            "objective_policy": _live_objective_policy_from_state(state),
            "exclude_before_ts": exclude_before_ts,
        }
    except Exception as exc:  # noqa: BLE001 - never block a trial write
        log.warning("W3 segment snapshot scope sync failed; disabling: %s", exc)
        journal.segment_snapshots = False


def _journal_archive_payload_for_authority(
    journal: ExperimentJournal,
    *,
    deinflate_before_ts: float | None = None,
    deinflate_factor: float = 1.0,
    exclude_before_ts: float | None = None,
    objective_policy: str = LEGACY_OBJECTIVE_POLICY,
    extra_rows: Iterable[dict[str, Any]] = (),
) -> dict[str, Any] | None:
    rows = _journal_rows_for_archive(journal) + [dict(r) for r in extra_rows]
    if hasattr(journal, "ledger_events"):
        snapshot_payload = archive_payload_from_verified_snapshot(
            rows,
            journal.ledger_events(),
        )
        if (
            snapshot_payload is not None
            and str(snapshot_payload.get("objective_policy") or LEGACY_OBJECTIVE_POLICY)
            == objective_policy
            and _snapshot_scope_matches(
                snapshot_payload,
                exclude_before_ts=exclude_before_ts,
                deinflate_before_ts=deinflate_before_ts,
                deinflate_factor=deinflate_factor,
            )
        ):
            return snapshot_payload
    return reconstruct_archive_from_journal_rows(
        rows,
        None,
        current_run_only=False,
        deinflate_before_ts=deinflate_before_ts,
        deinflate_factor=deinflate_factor,
        exclude_before_ts=exclude_before_ts,
        objective_policy=objective_policy,
    )


def _bug_tag_for_learning_exclusion(excluded_by: str, reason: str) -> tuple[str, str]:
    """The (bug_corrupted_by, reason) pair a trial row carries for its learning exclusion."""
    if (
        excluded_by
        and excluded_by not in BENIGN_LEARNING_EXCLUSIONS
        and excluded_by not in NON_CORRUPT_LEARNING_EXCLUSIONS
    ):
        return excluded_by, reason
    return "", ""


def _served_content_for_row(record: Any) -> dict[str, Any] | None:
    """The row form of a handler's served-content record (validated), or None."""
    files = served_content_files(record) if isinstance(record, dict) else None
    if files is None:
        return None
    out: dict[str, Any] = {"files": files}
    if isinstance(record, dict) and record.get("captured_at"):
        out["captured_at"] = str(record["captured_at"])
    return out


def _provisional_trial_row(
    *,
    trial_id: int,
    timestamp: str,
    eval_result: Any,
    action: dict[str, Any],
    learning_excluded_by: str,
    learning_excluded_reason: str,
    frontier_admission: str,
    comparability: dict[str, Any] | None,
    infra_digest: str = "",
    regime_digest: str = "",
    served_content: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decision (c): the objective-bearing projection of the row this trial WILL journal.

    Built from the same values the JournalEntry below is built from, so the promotion guard
    sees the candidate exactly as replay will. ``_provisional_row_mismatch`` checks the
    recorded row against it after ``journal.record``.
    """
    bug, _reason = _bug_tag_for_learning_exclusion(learning_excluded_by, learning_excluded_reason)
    eval_details: dict[str, Any] = {
        "question_results": list(getattr(eval_result, "question_results", []) or []),
        "details": getattr(eval_result, "details", {}) or {},
        "eval_wall_s": getattr(eval_result, "eval_wall_s", 0.0),
        "objective_policy_live": RATE_4D_OBJECTIVE_POLICY,
        "infra_comparability": str((comparability or {}).get("status", "") or ""),
        "infra_fingerprint_digest": infra_digest,
        INFRA_REGIME_DIGEST_KEY: regime_digest,
    }
    if served_content:
        eval_details[SERVED_CONTENT_KEY] = served_content
    if learning_excluded_by:
        eval_details["learning_exclusion"] = {
            "by": learning_excluded_by,
            "reason": learning_excluded_reason,
        }
    if frontier_admission:
        eval_details[FRONTIER_ADMISSION_KEY] = frontier_admission
    return {
        "trial_id": int(trial_id),
        "action_type": str(action.get("type", "")) if isinstance(action, dict) else "",
        "timestamp": timestamp,
        "tier": int(eval_result.tier),
        "quality": eval_result.quality,
        "speed": eval_result.speed,
        "cost": eval_result.cost,
        "reliability": eval_result.reliability,
        "bug_corrupted_by": bug,
        "config_snapshot": action,
        "reasoning": json.dumps(action),
        "eval_details": eval_details,
        "comparability": dict(comparability or {}),
    }


def _reconcile_promotion_with_journal(
    gate: Any,
    baseline_update: Any,
    baseline_before: Any,
    provisional: dict[str, Any],
    recorded: dict[str, Any],
    trial_counter: int,
) -> Any:
    """Re-review B2: roll back a promotion decided on evidence the journal does not hold.

    Runs after ``journal.record`` and before anything else persists. On a mismatch the
    in-memory baseline is restored to its pre-decision copy and the update is returned as
    refused, so neither the baseline_promotion event nor the promoted baseline_state is
    written. Returns the (possibly replaced) update.
    """
    diffs = _provisional_row_mismatch(provisional, recorded)
    if not diffs:
        return baseline_update
    promoted = baseline_update is not None and bool(getattr(baseline_update, "updated", False))
    log.error(
        "Trial %d: journaled row differs from the row the promotion guard evaluated on %s; "
        "the promotion decision (%s) did not see the recorded evidence.%s",
        trial_counter,
        diffs,
        getattr(baseline_update, "promotion_rule", "") if baseline_update else "none",
        " PROMOTION ROLLED BACK (baseline restored, no ledger event)." if promoted else "",
    )
    if not promoted:
        if baseline_update is not None and getattr(baseline_update, "speed_reseeded", False):
            # AP-57: a speed-axis reseed decided on the same unrecorded evidence is rolled
            # back exactly like a promotion, so no reseed receipt or reseeded state lands.
            log.error(
                "Trial %d: SPEED-AXIS RESEED ROLLED BACK (baseline restored, no ledger event).",
                trial_counter,
            )
            gate.baseline = baseline_before
            return dataclasses.replace(baseline_update, speed_reseeded=False, speed_reseed={})
        return baseline_update
    gate.baseline = baseline_before
    return dataclasses.replace(
        baseline_update,
        updated=False,
        reason=f"rolled back: journaled row differs from the evaluated row on {diffs}",
    )


def _provisional_row_mismatch(provisional: dict[str, Any], recorded: dict[str, Any]) -> list[str]:
    """Fields where the journaled row disagrees with the row the promotion decision saw."""
    from src.autopilot_core.live_reproductions import row_comparability_status

    diffs: list[str] = []
    for key in ("trial_id", "timestamp", "tier", "bug_corrupted_by"):
        if provisional.get(key) != recorded.get(key):
            diffs.append(key)
    if _config_fingerprint(provisional.get("config_snapshot")) != _config_fingerprint(
        recorded.get("config_snapshot")
    ):
        diffs.append("config_fingerprint")
    if row_config_identity(provisional) != row_config_identity(recorded):
        diffs.append("served_config_identity")
    if row_is_representative_member(provisional) != row_is_representative_member(recorded):
        diffs.append("representative_membership")
    if row_comparability_status(provisional) != row_comparability_status(recorded):
        diffs.append("comparability")
    for policy in (LEGACY_OBJECTIVE_POLICY, RATE_4D_OBJECTIVE_POLICY):
        if objectives_from_journal_row(provisional, objective_policy=policy) != (
            objectives_from_journal_row(recorded, objective_policy=policy)
        ):
            diffs.append(f"objectives[{policy}]")
    return diffs


CLEAN_TRIAL_REPRESENTATIVES_ENV = "AUTOPILOT_CLEAN_TRIAL_REPRESENTATIVES"


def _clean_trial_representatives_enabled() -> bool:
    """Decision (c) is ON unless explicitly disabled (``=0``)."""
    return os.environ.get(CLEAN_TRIAL_REPRESENTATIVES_ENV, "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _install_promotion_guard_scope(journal: Any, state: Mapping[str, Any]) -> None:
    """Gate-frontier (2026-09-16): promotion evidence reads the LIVE frontier.

    The safety gate's promotion checks (archive-max refusal, frontier-representative check,
    reproduced fields) used to replay the whole journal under the LEGACY tokens/second policy,
    pre-flip rows included. They now rebuild through the same authority path that seeds the
    live archive: live ``pareto_objective_policy``, live ``pareto_exclude_before_ts`` epoch
    fence, verified segment snapshot when its scope matches. ``state`` is the loop's live
    dict, read on every call, so a rebase applied in-process is honoured.
    """

    def _provider(pending_rows: tuple[dict[str, Any], ...] = ()) -> PromotionGuardView:
        policy = _live_objective_policy_from_state(state)
        deinflate_before_ts, deinflate_factor, exclude_before_ts = (
            _archive_epoch_params_from_state(dict(state))
        )
        pending = [dict(r) for r in pending_rows]
        payload = _journal_archive_payload_for_authority(
            journal,
            deinflate_before_ts=deinflate_before_ts,
            deinflate_factor=deinflate_factor,
            exclude_before_ts=exclude_before_ts,
            objective_policy=policy,
            extra_rows=pending,
        )
        resolved_policy = str((payload or {}).get("objective_policy") or policy)

        def _reproductions(tier: int, identity: str) -> list[dict[str, Any]]:
            return live_reproductions(
                _journal_rows_for_archive(journal) + pending,
                tier=tier,
                identity=identity,
                objective_policy=resolved_policy,
                exclude_before_ts=exclude_before_ts,
            )

        return PromotionGuardView(
            archive=_ConcreteParetoArchive.from_archive_payload(payload, read_only=True),
            objective_policy=resolved_policy,
            scope="live",
            exclude_before_ts=exclude_before_ts,
            reproductions=_reproductions,
        )

    configure_promotion_guard_archive(_provider)


def _live_objective_policy_from_state(state: Mapping[str, Any]) -> str:
    return str(state.get("pareto_objective_policy") or LEGACY_OBJECTIVE_POLICY).strip()


def _assert_eval_execution_instrument_state(state: Mapping[str, Any]) -> None:
    """Fail closed until the human-owned measurement boundary is applied."""
    policy = _live_objective_policy_from_state(state)
    execution_id = str(state.get("eval_execution_instrument_id") or "").strip()
    errors: list[str] = []
    if policy != RATE_4D_OBJECTIVE_POLICY:
        errors.append(
            f"pareto_objective_policy={policy or '<missing>'} (required {RATE_4D_OBJECTIVE_POLICY})"
        )
    if execution_id != EVAL_EXECUTION_INSTRUMENT_ID:
        errors.append(
            f"eval_execution_instrument_id={execution_id or '<missing>'} "
            f"(required {EVAL_EXECUTION_INSTRUMENT_ID})"
        )
    if errors:
        raise RuntimeError(
            "AutoPilot eval instrument is not ratified for this code: "
            + "; ".join(errors)
            + ". Apply the consolidated resource-lanes instrument ratifier before restart."
        )


def _archive_epoch_params_from_state(
    state: dict[str, Any],
) -> tuple[float | None, float, float | None]:
    """Extract Pareto replay epoch params from autopilot state."""
    strict_epoch = _numeric_swarm_epoch_label_from_state(state) is not None
    deinflate_before_ts: float | None = None
    try:
        deinflate_before_ts = float(state.get("pareto_epoch_ts") or 0.0) or None
    except (TypeError, ValueError) as exc:
        if strict_epoch:
            raise ValueError("invalid pareto_epoch_ts for active speed era") from exc
    deinflate_factor: float = 1.0
    try:
        deinflate_factor = float(state.get("pareto_pre_epoch_speed_factor", 1.0))
    except (TypeError, ValueError) as exc:
        if strict_epoch:
            raise ValueError("invalid pareto_pre_epoch_speed_factor for active speed era") from exc
    exclude_before_ts: float | None = None
    try:
        exclude_before_ts = float(state.get("pareto_exclude_before_ts") or 0.0) or None
    except (TypeError, ValueError) as exc:
        if strict_epoch:
            raise ValueError("invalid pareto_exclude_before_ts for active speed era") from exc
    if strict_epoch and (deinflate_before_ts is None or exclude_before_ts is None):
        raise ValueError("active speed era requires pareto_epoch_ts and pareto_exclude_before_ts")
    return deinflate_before_ts, deinflate_factor, exclude_before_ts


def _numeric_swarm_epoch_label_from_state(state: dict[str, Any]) -> str | None:
    """Return the persistent NumericSwarm study era for the live speed instrument."""
    eras = state.get("active_instrument_eras")
    if isinstance(eras, dict):
        value = eras.get("autopilot_speed")
        if isinstance(value, str) and value.strip():
            return value.strip()
    marker = state.get("frontier_rerun_required")
    if isinstance(marker, dict) and marker.get("required"):
        raise ValueError("frontier rerun requires active_instrument_eras.autopilot_speed")
    return None


# Quality-axis instrument-era fence (defect #1/#3/#4). The QUALITY analogue of the speed
# axis's active_instrument_eras.autopilot_speed + pareto_epoch_ts / pareto_exclude_before_ts.
EVAL_QUALITY_ERA_STATE_KEY = EVAL_QUALITY_SCOPE  # "eval_quality" key under active_instrument_eras
QUALITY_EPOCH_TS_KEY = "quality_epoch_ts"
QUALITY_EXCLUDE_BEFORE_TS_KEY = "quality_exclude_before_ts"


def _active_eval_quality_era_from_state(state: dict[str, Any]) -> str | None:
    eras = state.get("active_instrument_eras")
    if isinstance(eras, dict):
        value = eras.get(EVAL_QUALITY_ERA_STATE_KEY)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _quality_epoch_params_from_state(state: dict[str, Any]) -> tuple[str | None, float | None]:
    """Return (active eval_quality era id, quality_exclude_before_ts) — strict/fail-closed.

    Mirrors :func:`_archive_epoch_params_from_state` on the QUALITY axis. When state declares
    an active ``eval_quality`` era but the exclude timestamp is missing/invalid, RAISE — the
    fence must not silently degrade to unfenced (the exact speed-axis ``strict_epoch``
    contract). When no era is declared, returns ``(None, None)`` and the quality axis runs
    unfenced (single-era world / tests), so every pre-existing path is unaffected.
    """
    era = _active_eval_quality_era_from_state(state)
    if era is None:
        return None, None
    try:
        exclude_before_ts = float(state.get(QUALITY_EXCLUDE_BEFORE_TS_KEY) or 0.0) or None
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid quality_exclude_before_ts for active eval_quality era") from exc
    if exclude_before_ts is None:
        raise ValueError("active eval_quality era requires quality_exclude_before_ts")
    return era, exclude_before_ts


def _quality_exclude_before_ts_from_state(state: dict[str, Any]) -> float | None:
    """Active eval-quality boundary epoch for evidence fences (None => unfenced)."""
    return _quality_epoch_params_from_state(state)[1]


def _migrate_eval_quality_era(state: dict[str, Any]) -> bool:
    """Startup migration guard: seed the eval_quality instrument-era fence when absent.

    Idempotent — only acts when ``active_instrument_eras.eval_quality`` is missing. Resolves
    the active eval_quality era from the human-owned registry via
    ``instrument_era_guard.active_eval_quality_era``; on a registry read failure it falls
    forward to the ``E7-eval-instrument`` code constant (still fences) but ONLY once the clock
    is at/after that boundary. Returns True when it mutated state. A no-op (False) before any
    boundary opens, so the pre-boundary single-era world stays unfenced.

    This is the code-path state migration the operator constraint requires: the era registry
    itself (human-amendment-only) is never written here — only autopilot_state.json's derived
    fence keys are seeded, on next startup, behind this guard.
    """
    if _active_eval_quality_era_from_state(state) is not None:
        return False  # already migrated / operator-set

    guard = active_eval_quality_era()
    boundary_epoch: float | None = None
    era_id = ""
    source = ""
    if guard.get("ok"):
        era_id = str(guard.get("era_id") or "").strip() or E7_EVAL_INSTRUMENT_ERA_ID
        boundary_epoch = guard.get("boundary_epoch")
        source = f"registry:{guard.get('path')}"
    elif guard.get("status") == "no_active_era":
        # Registry read fine; no eval_quality era active yet (clock before the boundary).
        # Correct to leave the quality axis unfenced.
        return False
    else:
        # Registry unreadable/malformed. Fall back to the last era the LIVE SYSTEM
        # recorded, not to a code constant.
        #
        # This branch used to fence at E7_EVAL_INSTRUMENT_BOUNDARY (2026-07-21) and stamp
        # E7_EVAL_INSTRUMENT_ERA_ID. That constant stopped tracking the live era at the
        # v8 eval boundary on 2026-07-25 and eval_quality is now at E16, so an unreadable
        # registry silently UNDER-fenced by three weeks and several instrument changes —
        # every cross-instrument observation in that span read as in-era and was admitted.
        # Fail-open, in the guard whose job is to fail closed. Bumping the constant would
        # only move the staleness to the next cutover.
        last_known = last_known_eval_quality_era()
        fallback_epoch = last_known.get("boundary_epoch")
        if last_known.get("ok") and fallback_epoch is not None:
            # Keep the original rule: never over-fence a clock that predates the
            # boundary being fenced at. What changed is WHICH boundary — the last
            # one the live system actually recorded, instead of a literal that
            # stopped tracking reality three weeks ago.
            if time.time() < float(fallback_epoch):
                log.warning(
                    "eval_quality era migration deferred — registry unresolved (%s) and "
                    "clock is before the last recorded boundary for %s; quality axis "
                    "remains unfenced.",
                    guard.get("status"),
                    last_known.get("era_id"),
                )
                return False
            era_id = str(last_known["era_id"])
            boundary_epoch = float(fallback_epoch)
            source = f"last-known-era fallback (registry {guard.get('status')})"
            log.warning(
                "eval_quality era registry unresolved (%s); fencing on the last recorded "
                "era %s from %s.",
                guard.get("status"),
                era_id,
                last_known.get("path"),
            )
        else:
            # No registry AND no witness. Refuse rather than invent a boundary: an
            # unfenced quality axis is visibly wrong, a wrongly-fenced one is not.
            log.error(
                "eval_quality era migration aborted — registry unresolved (%s) and no "
                "last-known era available (%s); quality axis remains unfenced and NO "
                "stale boundary was substituted.",
                guard.get("status"),
                last_known.get("status"),
            )
            return False

    if boundary_epoch is None:
        boundary_epoch = _parse_journal_timestamp(E7_EVAL_INSTRUMENT_BOUNDARY)
    if boundary_epoch is None:
        log.error(
            "eval_quality era migration aborted — could not resolve a boundary epoch; "
            "quality axis remains unfenced."
        )
        return False

    eras = state.get("active_instrument_eras")
    eras = dict(eras) if isinstance(eras, dict) else {}
    eras[EVAL_QUALITY_ERA_STATE_KEY] = era_id
    state["active_instrument_eras"] = eras
    state[QUALITY_EPOCH_TS_KEY] = boundary_epoch
    state[QUALITY_EXCLUDE_BEFORE_TS_KEY] = boundary_epoch
    log.warning(
        "MIGRATION: seeded eval_quality instrument-era fence — era=%s boundary_epoch=%.1f "
        "(exclude/epoch) source=%s. Pre-boundary quality journal rows and pre-boundary "
        "baseline are now PRIORS; the next quality promote/revert is era-fenced.",
        era_id,
        boundary_epoch,
        source,
    )
    return True


def _apply_journal_archive_authority(
    state: dict[str, Any],
    journal: ExperimentJournal,
    archive: ParetoArchive,
) -> bool | None:
    """Apply the append-only journal archive fold to memory.

    Returns True when cached state changed, False when it already matched, and
    None when the journal has no archive-bearing rows.
    """
    deinflate_before_ts, deinflate_factor, exclude_before_ts = _archive_epoch_params_from_state(
        state
    )
    archive_payload = _journal_archive_payload_for_authority(
        journal,
        deinflate_before_ts=deinflate_before_ts,
        deinflate_factor=deinflate_factor,
        exclude_before_ts=exclude_before_ts,
        objective_policy=_live_objective_policy_from_state(state),
    )
    if archive_payload is None:
        return None
    changed = "pareto_archive" in state
    state.pop("pareto_archive", None)
    archive._replace_from_archive_payload(archive_payload)
    return changed


def _sync_startup_archive_from_journal_authority(
    state: dict[str, Any],
    journal: ExperimentJournal,
    archive: ParetoArchive,
) -> bool:
    """Remove stale startup archive cache before writing lifecycle state.

    Preflight repairs archive authority from the append-only journal, but a
    direct ``autopilot.py start`` must uphold the same invariant. Otherwise the
    first startup save (fleet timestamp, pause-loop bookkeeping, etc.) can
    preserve a stale cached ``state["pareto_archive"]`` before any trial runs.
    """
    if state.get("_allow_empty_frontier_rebase"):
        return False
    before_count = _archive_entry_count(state.get("pareto_archive"))
    deinflate_before_ts, deinflate_factor, exclude_before_ts = _archive_epoch_params_from_state(
        state
    )
    journal_count = _archive_entry_count(
        _journal_archive_payload_for_authority(
            journal,
            deinflate_before_ts=deinflate_before_ts,
            deinflate_factor=deinflate_factor,
            exclude_before_ts=exclude_before_ts,
            objective_policy=_live_objective_policy_from_state(state),
        )
    )
    changed = _apply_journal_archive_authority(state, journal, archive)
    if changed is not True:
        return False
    log.warning(
        "Startup archive cache removed; journal fold is authoritative "
        "(cached_state_entries %d, journal_entries %d)",
        before_count,
        journal_count,
    )
    return True


def _baseline_state_for_startup_gate(
    state: dict[str, Any],
    journal: ExperimentJournal,
) -> dict[str, Any]:
    """Return the startup baseline authority payload for SafetyGate.

    ``baseline_state`` stays authoritative while present. After W4 cutover
    removes that cache, the append-only promotion ledger is allowed to seed the
    gate only when the same reconciliation rules say the fold is cutover-ready.
    """
    state_baseline = state.get("baseline_state")
    if isinstance(state_baseline, dict) and state_baseline:
        return state_baseline
    reconciliation = reconcile_baseline_ledger(
        journal.baseline_ledger_events(),  # AP-57: promotions + reseeds
        None,
    )
    if reconciliation.cutover_ready and isinstance(reconciliation.folded_state, dict):
        log.info("Using baseline promotion ledger fold for SafetyGate startup baseline")
        return reconciliation.folded_state
    return {}


def _save_state_with_journal_archive_authority(
    state: dict[str, Any],
    journal: ExperimentJournal,
    archive: ParetoArchive,
    *,
    context: str,
    merge_control: bool = False,
) -> bool:
    """Persist state with append-only journal folds as cache authority.

    ``merge_control`` is forwarded to ``save_state`` so the daemon's trial-end /
    lifecycle saves re-read and merge out-of-band control fields (paused /
    pause_reason / _in_cache_flush) under the write lock before writing (H4). The
    forward is done via an explicit branch so the default (non-merging) path still
    calls ``save_state(state)`` with a single positional arg, matching test doubles.
    """
    archive_changed = _apply_journal_archive_authority(state, journal, archive)
    baseline_changed = apply_baseline_ledger_authority(
        state,
        journal.baseline_ledger_events(),  # AP-57: promotions + reseeds
    )
    if archive_changed is None:
        log.warning(
            "Journal archive authority unavailable during %s; saving lifecycle "
            "state without a legacy archive cache",
            context,
        )
    elif archive_changed:
        log.info("State archive cache removed during %s", context)
    if merge_control:
        save_state(state, merge_control=True)
    else:
        save_state(state)
    if baseline_changed:
        log.info("State baseline cache removed during %s", context)
    return archive_changed is not None


def learning_exclusion_criticism(
    learning_excluded_by: str,
    learning_excluded_reason: str,
) -> SelfCriticism:
    """Controller-facing criticism for journaled-but-untrusted trials.

    SafetyGate can pass a trial while still marking it as `mad_noise`. Those
    entries are already excluded from Pareto/AP-22 learning, so their journal
    criticism must not say "keep" or "continue this surface"; that text is fed
    back into the planner and can create meta-action loops.
    """
    reason = learning_excluded_reason or "outcome was excluded from learning"
    if learning_excluded_by == "reproduction_confirmed":
        # Convergence, NOT a failure. The planner must read this as "the current
        # config is confirmed/converged on this surface" and move on — not as a
        # wasted/noisy trial demanding another attempt (which fuels the loop).
        return SelfCriticism(
            what_went_wrong="",
            why_it_happened=reason,
            what_should_change=(
                "Treat the current config as CONFIRMED / converged on this surface. "
                "Do NOT re-run this surface expecting a fresh win, and do NOT read "
                "repeated confirmations as a noisy or broken instrument — explore a "
                "different surface or idle cleanly until new signal is available."
            ),
            optimization_directions=[],
            keep_or_revert="excluded",
            keep_revert_reasoning=(
                "reproduction confirms the existing kept gain (convergence); no NEW "
                "Pareto point, but the config remains validated and trustworthy"
            ),
        )
    label = learning_excluded_by.replace("_", " ")
    return SelfCriticism(
        what_went_wrong=f"Trial excluded from learning: {label}",
        why_it_happened=reason,
        what_should_change=(
            "Do not treat this outcome as a keep or config-efficacy signal; "
            "require a clean, non-excluded metric trial before continuing this direction"
        ),
        optimization_directions=[],
        keep_or_revert="excluded",
        keep_revert_reasoning=(
            f"{reason}; archive/AP-22 learning skipped and planner trust excludes this trial"
        ),
    )


def _classify_meta_halt(journal: Any) -> str:
    """Classify a meta-action-loop halt as 'converged' vs 'stuck'.

    Convergence: the planner ran out of config moves because the recent metric
    trials REPRODUCED an already-established above-baseline config
    (``reproduction_confirmed``) rather than failing or corrupting. That is a
    benign terminal state — the planner should be pointed at a NEW surface, not
    treated as a malfunction or an instrument-noise problem (2026-05-31). Any
    other shape (bug-corruptions, kills, regressions, genuine gate-lock) is
    ``stuck`` and warrants operator review.
    """
    try:
        recent = journal.recent_hypotheses(n=5, exclude_bug_corrupted=False)
    except Exception:  # noqa: BLE001 — never let classification crash the halt
        return "stuck"
    repro = sum(
        1 for e in recent if getattr(e, "deficiency_category", "") == "reproduction_confirmed"
    )
    corrupt = sum(1 for e in recent if getattr(e, "bug_corrupted_by", ""))
    return "converged" if (repro >= 1 and corrupt == 0) else "stuck"


def load_blacklist() -> list[dict[str, Any]]:
    """Load failure blacklist from YAML."""
    return _load_blacklist_impl(BLACKLIST_PATH)


def load_model_signatures() -> dict[str, Any]:
    """Load descriptor-backed model signatures for planner context."""
    return _load_model_signatures_impl(_MODEL_SIGNATURES_PATH, _MODEL_DESCRIPTORS_PATH)


def append_blacklist(
    action: dict[str, Any],
    trial_id: int,
    reason: str,
    *,
    reason_class: str | None = None,
    ttl_days: int | None = None,
) -> None:
    """Auto-append a blacklist entry after rollback trigger."""
    _append_blacklist_impl(
        action,
        trial_id,
        reason,
        BLACKLIST_PATH,
        reason_class=reason_class,
        ttl_days=ttl_days,
    )


# ── Slot Memory Visibility (AM KV Compaction) ──────────────────


# Degraded fallback only. Runtime slot visibility should follow generated stack
# priors so the planner does not miss newly added live llama-server ports.
_FALLBACK_SLOT_QUERY_PORTS: dict[str, list[int]] = {
    "frontdoor": [8070],
    "coder": [8070],  # shares server with frontdoor (same Qwen3.6-35B Q8 GGUF)
    "worker": [8072],
    # 2026-09-27 ARCHITECT SWAP: architect_general = Flash-Next CPU :8074,
    # architect_critic = the MI210 27B :8083.
    "architect_general": [8074],
    "architect_critic": [8083],
}


def _slot_query_ports_from_stack_priors(
    stack_priors_path: Path = STACK_PRIORS_PATH,
) -> dict[str, list[int]]:
    """Return primary live llama-server ports by role from generated stack priors."""
    return live_stack_slot_query_ports(stack_priors_path)


def _slot_query_ports() -> dict[str, list[int]]:
    """Return live slot-query ports, falling back only when stack priors fail."""
    return _slot_query_ports_from_stack_priors() or dict(_FALLBACK_SLOT_QUERY_PORTS)


def _query_slot_memory() -> str:
    """Query llama-server /slots on production ports and return a summary.

    Returns a compact text block showing per-role slot memory usage so the
    controller can decide when slot_compact is worthwhile.
    """
    import httpx

    cached_lines: list[str] = []
    empty_ports: list[str] = []
    unavailable_ports: list[str] = []
    for role, ports in _slot_query_ports().items():
        for port in ports:
            try:
                resp = httpx.get(f"http://localhost:{port}/slots", timeout=3.0)
                if resp.status_code != 200:
                    unavailable_ports.append(f"{role}:{port} http {resp.status_code}")
                    continue
                slots = resp.json()
                if not isinstance(slots, list):
                    continue
                port_cached = 0
                for s in slots:
                    sid = s.get("id", "?")
                    state = s.get("state", "?")
                    n_past = s.get("n_past", 0)
                    if n_past > 0:
                        port_cached += int(n_past)
                        cached_lines.append(
                            f"  {role}:{port}/slot{sid} — {state}, {n_past} tokens cached"
                        )
                if port_cached == 0:
                    empty_ports.append(f"{role}:{port}")
            except Exception:
                unavailable_ports.append(f"{role}:{port}")
    lines = list(cached_lines)
    if empty_ports:
        lines.append("  healthy queried ports with empty KV cache: " + ", ".join(empty_ports))
    if unavailable_ports:
        lines.append("  unavailable configured replica ports: " + ", ".join(unavailable_ports))
        lines.append(
            "  note: unavailable replicas only affect slot_compact targeting; "
            "do not infer eval-instrument contamination from this line"
        )
    if not lines:
        return "  (no slot data returned; slot_compact has no target)"
    return "\n".join(lines)


# ── Claude CLI Controller ────────────────────────────────────────
# Controller invocation, action extraction, and AP-9 single-variable
# validation moved to controller_io.py (2026-05-22 refactor). Wrapper below
# preserves the original cwd=ORCH_ROOT semantics.


def invoke_controller(
    prompt: str,
    session_id: str | None = None,
    timeout: int = 300,
) -> tuple[str, str | None]:
    """Invoke Claude CLI for meta-reasoning."""
    return _invoke_controller_impl(prompt, session_id=session_id, timeout=timeout, cwd=ORCH_ROOT)


# ── Action Dispatch (moved to actions.py, 2026-05-22 refactor) ──
# dispatch_action() is re-imported from actions.py at the module top so the
# public name + signature are preserved for callers (run_loop, tests).


# ── Main Loop ────────────────────────────────────────────────────


def run_loop(
    max_trials: int | None = None,
    dry_run: bool = False,
    use_controller: bool = True,
    use_tui: bool = False,
) -> None:
    """Main optimization loop."""
    breadcrumb = ExitBreadcrumb()
    breadcrumb.write(
        "run_loop_started",
        max_trials=max_trials,
        dry_run=dry_run,
        use_controller=use_controller,
    )
    breadcrumb.register_atexit()
    # Optional TUI for live inference monitoring
    tui = None
    if use_tui:
        try:
            from autopilot_tui import AutoPilotTUI

            tui = AutoPilotTUI()
            tui.__enter__()
        except Exception as e:
            log.warning("TUI not available: %s", e)
            tui = None

    try:
        _run_loop_inner(max_trials, dry_run, use_controller, tui, breadcrumb)
    except BaseException as exc:
        breadcrumb.mark_terminal(
            "unhandled_exception",
            exception_type=type(exc).__name__,
            exception_message=str(exc)[:512],
        )
        raise
    else:
        breadcrumb.mark_terminal("run_loop_return")
    finally:
        if tui is not None:
            tui.__exit__(None, None, None)


def _make_eval_progress_callback(
    *,
    phase: PhaseTracker,
    tui: "AutoPilotTUI | None",
    trial_id: Callable[[], int],
    action: Callable[[], dict[str, Any] | None],
) -> Callable[[str], None]:
    def _callback(prompt: str) -> None:
        if tui is not None:
            tui.set_prompt(prompt)
        current_action = action() or {}
        phase.set(
            "dispatch_action",
            trial_id=trial_id(),
            action_type=current_action.get("type", ""),
            idle_reason="evaluating question",
            prompt_preview=str(prompt)[:240],
        )

    return _callback


def _startup_archive_from_current_era_payload(
    state: dict[str, Any],
    archive_payload: dict[str, Any] | None,
) -> tuple[ParetoArchive, bool]:
    """Load current-era journal authority and retire a completed empty rebase.

    A deliberate rebase is only a bootstrap escape hatch while the current-era
    journal has no archive-bearing rows. Once its first point exists, journal
    authority wins over the stale flag so a later restart cannot discard it.
    """
    if archive_payload is not None:
        archive = ParetoArchive.from_archive_payload(archive_payload, read_only=False)
    else:
        archive = ParetoArchive()
    rebase_completed = bool(
        state.get("_allow_empty_frontier_rebase")
        and any(archive.frontier_size(tier) > 0 for tier in archive.tiers())
    )
    if rebase_completed:
        state.pop("_allow_empty_frontier_rebase", None)
        bootstrap_key = (
            "eval_instrument_empty_frontier_bootstrap"
            if isinstance(state.get("eval_instrument_empty_frontier_bootstrap"), dict)
            else "e8_empty_frontier_bootstrap"
        )
        bootstrap = state.get(bootstrap_key)
        if isinstance(bootstrap, dict):
            bootstrap["status"] = "completed"
            bootstrap["completion_condition"] = (
                "next AutoPilot startup observed at least one current-era Pareto point"
            )
        else:
            state[bootstrap_key] = {
                "status": "completed",
                "completion_condition": (
                    "next AutoPilot startup observed at least one current-era Pareto point"
                ),
            }
    return archive, rebase_completed


def _make_eval_batch_progress_callback(
    *,
    phase: PhaseTracker,
    trial_id: Callable[[], int],
    action: Callable[[], dict[str, Any] | None],
) -> Callable[[dict[str, Any]], None]:
    def _callback(progress: dict[str, Any]) -> None:
        current_action = action() or {}
        phase.set(
            "dispatch_action",
            trial_id=trial_id(),
            action_type=current_action.get("type", ""),
            idle_reason="evaluating question",
            eval_label=progress.get("label"),
            eval_completed_questions=progress.get("completed_questions"),
            eval_total_questions=progress.get("total_questions"),
            eval_correct_questions=progress.get("correct_questions"),
            eval_correct_pct=progress.get("correct_pct"),
            eval_concurrency=progress.get("concurrency"),
        )

    return _callback


def _run_loop_inner(
    max_trials: int | None,
    dry_run: bool,
    use_controller: bool,
    tui: "AutoPilotTUI | None" = None,
    breadcrumb: ExitBreadcrumb | None = None,
) -> None:
    """Inner loop (separated to ensure TUI cleanup via run_loop's finally)."""
    state = load_state()
    if MULTITIER_PROMOTION_ENABLED:
        policy_state = state.get("multitier_promotion_policy")
        ready, reason = _multitier_required_baselines_ready(state)
        if (
            not isinstance(policy_state, dict)
            or policy_state.get("enabled") is not True
            or policy_state.get("policy_version") != MULTITIER_POLICY_VERSION
            or not ready
        ):
            raise RuntimeError(
                "staged multi-tier promotion is enabled in the launcher but its "
                f"ratified state/baseline bundle is not ready: {reason}"
            )
    # Defect #1/#3/#4: seed the eval_quality instrument-era fence on first startup after the
    # boundary opened (code-path migration, never a hand-edit of the human-owned registry).
    # Persist immediately so a crash before the first save cannot lose the fence.
    if _migrate_eval_quality_era(state):
        save_state(state)
    # Fail-closed startup check (mirrors the speed axis): raise NOW if the quality fence is
    # half-declared (era present, exclude timestamp missing/invalid) rather than silently
    # running unfenced mid-loop.
    eval_quality_era, quality_exclude_before_ts = _quality_epoch_params_from_state(state)
    journal = ExperimentJournal()
    _sync_segment_snapshot_scope(journal, state)  # W3: scope rollover snapshots from the start
    _deinfl_ts, _deinfl_factor, _exclude_ts = _archive_epoch_params_from_state(state)
    archive_payload = _journal_archive_payload_for_authority(
        journal,
        deinflate_before_ts=_deinfl_ts,
        deinflate_factor=_deinfl_factor,
        exclude_before_ts=_exclude_ts,
        objective_policy=_live_objective_policy_from_state(state),
    )
    archive, rebase_completed = _startup_archive_from_current_era_payload(
        state,
        archive_payload,
    )
    # Gate-frontier: install BEFORE SafetyGate(...) so no promotion check ever runs on the
    # legacy fallback scope in the live loop.
    _install_promotion_guard_scope(journal, state)
    # RTG-55 MHS-3 operability: the leakage guard fails CLOSED (every mutation rejected)
    # when the eval-id vocabulary cannot be built. Make that loud without blocking start:
    # one ERROR + journal event now, a rate-limited operator alarm after N consecutive
    # rejections, cleared when the vocabulary builds again. Runbook:
    # docs/guides/meta-harness-operator-guide.md.
    leakage_monitor = EvalLeakageMonitor(
        journal=journal, state=state, prompt_forge_module=_prompt_forge_module
    ).install()
    if not leakage_monitor.preflight():
        save_state(state)
    # Clear the deliberate-rebase bypass ONLY once the frontier has actually rebuilt
    # (a prior run admitted >=1 point). Clearing it at startup while the frontier is
    # still empty would re-arm the frontier-lost guard before the bootstrap lands —
    # crash-looping any restart that happens before the first trial is admitted. Once
    # a point exists the guard never fires anyway, so this safely re-arms it.
    if rebase_completed:
        save_state(state)
        log.info(
            "Rebase complete (frontier rebuilt) — cleared _allow_empty_frontier_rebase; guard re-armed."
        )
    gate = SafetyGate(
        consecutive_failures=state.get("consecutive_failures", 0),
        quality_history=state.get("quality_history", []),
        quality_history_by_tier=state.get("quality_history_by_tier", {}),
        quality_history_provenance_by_tier=state.get("quality_history_provenance_by_tier"),
        baseline_state=_baseline_state_for_startup_gate(state, journal),
        eval_quality_era=eval_quality_era,
        # 2026-08-03: the SPEED axis's era, from the same active_instrument_eras registry
        # entry (autopilot_speed) already used above to fence the Pareto frontier. Without
        # it the gate's throughput floor had NO provenance at all — a post-v8 trial could be
        # charged against a floor derived from a pre-v8 frontdoor_speed and nothing recorded
        # would ever reveal it.
        autopilot_speed_era=_numeric_swarm_epoch_label_from_state(state),
        quality_exclude_before_ts=quality_exclude_before_ts,
    )
    tower = EvalTower(
        url=ORCHESTRATOR_URL,
    )
    meta = MetaOptimizer()

    # Phase 5 — OrchestratorWatcher for exogenous-restart detection.
    # Single instance shared across the trial loop, the seeder, and the
    # eval tower. Disabled via AUTOPILOT_WATCHER_DISABLED=1 for tests/dev
    # without fleet markers; in that mode all methods no-op safely.
    try:
        from orchestrator_watch import OrchestratorWatcher

        watcher = OrchestratorWatcher(api_url=ORCHESTRATOR_URL)
        # Attach to the tower so its _eval_question call can pass it down
        # into call_orchestrator_forced (Phase 4 reads via
        # getattr(self, "watcher", None)).
        tower.watcher = watcher
        log.info(
            "OrchestratorWatcher attached (disabled=%s, marker dir=/mnt/raid0/llm/tmp)",
            watcher.disabled,
        )
    except Exception as exc:
        log.warning(
            "OrchestratorWatcher init failed (%s); falling back to legacy retry-less path", exc
        )
        watcher = None

    seeder = Seeder(
        url=ORCHESTRATOR_URL,
        dry_run=dry_run,
    )
    swarm = NumericSwarm(epoch_label=_numeric_swarm_epoch_label_from_state(state))
    forge = PromptForge(auto_commit=not dry_run)
    lab = StructuralLab(orchestrator_url=ORCHESTRATOR_URL)
    evo = EvolutionManager(use_local_model=not use_controller)

    # AP-22: Short-term memory (accumulated learnings across trials)
    memory = ShortTermMemory()
    memory.refresh_from_journal(journal)
    last_criticism_text = "(first trial — no prior criticism)"

    # B1: Strategy store for species memory
    strategy_store: StrategyStore | None = None
    try:
        strategy_store = StrategyStore()
        try:
            strategy_health = strategy_store.search_index_health()
        except Exception as health_exc:
            strategy_health = {
                "healthy": False,
                "summary": f"health check unavailable: {health_exc}",
            }
        log.info(
            "Strategy store loaded (%d entries; search indexes: %s)",
            strategy_store.count(),
            strategy_health.get("summary", "unknown"),
        )
        if not strategy_health.get("healthy", True):
            log.warning(
                "StrategyStore search indexes are degraded; planner retrieval may miss rows. %s",
                strategy_health.get("repair_hint", "Run StrategyStore.rebuild_search_indexes()."),
            )
    except Exception as e:
        log.warning("Strategy store unavailable: %s", e)
    _install_planner_convention_bindings(strategy_store, journal)

    # B2: Failure blacklist
    blacklist = load_blacklist()

    # Load species budget from state
    if "species_budget" in state:
        b = state["species_budget"]
        meta.budget = SpeciesBudget(**b)
    # AP-37 diversity-stall state was removed 2026-08-01; _normalize_state_before_save
    # drops any residual key from pre-existing state files.

    # Restore seeder convergence state. Prefer the explicit state shape; fall
    # back to legacy td_errors-only persistence so existing state files still
    # reconstruct batch_count + convergence streak sensibly.
    seeder.restore_state(state.get("seeder_state") or {"td_errors": state.get("td_errors", [])})

    trial_counter = state.get("trial_counter", 0)
    plot_paths: list[str] = []

    # 2026-05-23 Phase 6b — autopilot self-crash recovery.
    # Check for an in_flight_trial marker left by a prior crashed instance.
    # Recovery handles two cases (handoff Section 5.7):
    #   (a) journal_max >= prior_in_flight.trial_id → trial DID get recorded;
    #       we died between journal.record (line 837) and save_state (line 930).
    #       Re-sync trial_counter + attempt Pareto re-import from the journal.
    #   (b) journal_max <  prior_in_flight.trial_id → trial died BEFORE
    #       journal.record. Write a placeholder JournalEntry tagged
    #       bug_corrupted_by=autopilot_killed_mid_trial so the planner sees
    #       the gap and excludes it from hypothesis chains. Skip gate + archive.
    _reject_in_flight_manifest_drift(state)
    trial_counter = _recover_from_in_flight_trial(
        state,
        journal,
        archive,
        trial_counter,
    )
    if breadcrumb is not None:
        breadcrumb.set_context(trial_id=trial_counter)
    _sync_startup_archive_from_journal_authority(state, journal, archive)
    # Bump the fleet-startup timestamp on every start (recovery path or
    # normal startup) so downstream watchers can detect autopilot
    # restarts the same way they detect orchestrator/llama restarts.
    state["autopilot_fleet_started_at"] = time.time()
    save_state(state)

    # Graceful shutdown handler
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if breadcrumb is not None:
            breadcrumb.write(
                "signal_received",
                signal_number=signum,
                signal_name=signal.Signals(signum).name,
            )
        log.info("Shutdown requested (signal %d)", signum)
        shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    log.info("AutoPilot starting (trial=%d, dry_run=%s)", trial_counter, dry_run)
    phase = PhaseTracker()
    async_tasks = AsyncTaskRunner()
    startup_attestation = _startup_attestation_payload()
    if startup_attestation["missing_or_mismatch"]:
        log.error(
            "AutoPilot startup gate env mismatch: %s",
            json.dumps(startup_attestation["missing_or_mismatch"], sort_keys=True),
        )
    log.info(
        "AutoPilot startup attestation: config_hash=%s gate_env=%s",
        startup_attestation["config_hash"],
        json.dumps(startup_attestation["gate_env"], sort_keys=True),
    )
    phase.set(
        "starting",
        trial_id=trial_counter,
        dry_run=dry_run,
        startup_attestation=startup_attestation,
    )
    current_action: dict[str, Any] | None = None
    eval_progress_callback = _make_eval_progress_callback(
        phase=phase,
        tui=tui,
        trial_id=lambda: trial_counter,
        action=lambda: current_action,
    )
    eval_batch_progress_callback = _make_eval_batch_progress_callback(
        phase=phase,
        trial_id=lambda: trial_counter,
        action=lambda: current_action,
    )
    tower.on_question = eval_progress_callback
    tower.on_progress = eval_batch_progress_callback
    seeder.on_question = eval_progress_callback

    # ── Plot freshness, decoupled from the trial loop ──────────────────────
    # Pre-2026-06-07, plots regenerated ONLY at `trial_counter % PLOT_INTERVAL
    # == 0`, fired from inside this loop. So any stop on a non-multiple-of-N
    # (operator pause, SIGTERM, max-trials, internal halt) froze the PNGs at the
    # last decade boundary — observed: a run reached trial 707/708 but the
    # dashboard PNGs stayed at the trial-700 render. We now ALSO refresh on a
    # wall-clock timer (mid-decade) and force a synchronous render on every
    # lifecycle transition so the operator's view always reflects the last
    # completed trial.
    plot_clock = {"last_ts": 0.0, "last_pause_trial": -1}

    # Survives across loop iterations, which is the whole point: the pre-2026-08-12
    # retry branch kept no counter at all, so an unbounded `sleep(); continue` on a
    # dead API looked identical on iteration 1 and iteration 10,000.
    health_escalation = HealthEscalationTracker(escalate_after=HEALTH_ESCALATE_AFTER)

    def _refresh_plots(*, sync: bool, reason: str) -> None:
        """Regenerate the dashboard PNGs from current on-disk state.

        ``sync=True`` blocks until the child ``autopilot.py plot`` completes —
        for lifecycle transitions where the process may exit right after.
        ``sync=False`` submits the existing non-blocking async subprocess
        (steady state). Both read freshly-saved state/journal from disk, so a
        ``sync=True`` caller must have ``save_state()``-ed first.
        """
        plot_clock["last_ts"] = time.time()
        cmd = [sys.executable, str(SCRIPT_DIR / "autopilot.py"), "plot"]
        if sync:
            try:
                subprocess.run(cmd, cwd=ORCH_ROOT, timeout=PLOT_SYNC_TIMEOUT_S, check=False)
                log.info("Plots refreshed synchronously (%s, trial=%d)", reason, trial_counter)
            except Exception as exc:
                log.warning("Synchronous plot refresh (%s) failed: %s", reason, exc)
        else:
            async_tasks.submit_subprocess(f"plots-trial-{trial_counter}", cmd, cwd=ORCH_ROOT)

    # `--max-trials N` means RUN N MORE, not "stop once the lifetime counter
    # reaches N". It used to compare the cumulative counter directly, so with a
    # counter of 1459 `--max-trials 1` exited immediately without running a
    # single trial — a bounded smoke run silently did nothing, and the only way
    # to get one trial was to pass 1460 and know the current count. A relative
    # budget is what every caller actually wants and is safe by default: too
    # small a number can no longer be a no-op.
    trial_budget_start = trial_counter
    trial_stop_at = trial_counter + max_trials if max_trials else None
    if trial_stop_at is not None:
        log.info(
            "Trial budget: %d more (trial %d -> %d)",
            max_trials,
            trial_budget_start,
            trial_stop_at,
        )

    while not shutdown_requested:
        async_tasks.reap(logger=log)
        phase.set("loop_start", trial_id=trial_counter)
        if trial_stop_at is not None and trial_counter >= trial_stop_at:
            log.info(
                "Trial budget spent: ran %d of %d requested (trial %d -> %d)",
                trial_counter - trial_budget_start,
                max_trials,
                trial_budget_start,
                trial_counter,
            )
            phase.set(
                "max_trials_reached",
                trial_id=trial_counter,
                max_trials=max_trials,
                ran=trial_counter - trial_budget_start,
            )
            break

        # 2026-05-24 pause-bug fix: re-read state from disk at the top of
        # every iteration so that an external `autopilot.py pause` actually
        # takes effect on a running autopilot. Pre-fix the loop's local
        # `state` was loaded once at startup and only refreshed inside the
        # paused branch; meanwhile `save_state(state)` after each trial wrote
        # the cached `paused=False` back to disk, clobbering any externally-
        # set True. Net effect: `pause` was silently a no-op on a running
        # autopilot. See `feedback_autopilot_pause_broken_use_sigterm` memory.
        # Re-loading here is cheap (one small JSON file) and merges any
        # externally-set fields (paused, _in_cache_flush) without losing the
        # in-memory trial counters that get written back at end of iteration.
        was_paused = bool(state.get("paused"))
        try:
            disk_state = load_state()
            for key in _EXTERNAL_CONTROL_FIELDS:
                if key in disk_state:
                    state[key] = disk_state[key]
        except Exception as _exc:
            log.warning("Failed to reload state at iteration top: %s", _exc)

        # Operator resumed (paused True→False): clear the halt latch here, in the
        # daemon, because this is the only process whose write of it lasts.
        if was_paused and not state.get("paused"):
            clear_halt_latch_on_resume_edge(state)

        if state.get("paused"):
            # Refresh plots once on ENTERING a pause episode. trial_counter is
            # frozen while paused, so the guard fires exactly once per episode
            # (not every PAUSE_POLL_S poll) and re-fires only if trials advanced
            # before a later re-pause. This covers operator `pause` AND internal
            # halts (meta-loop latch), which rarely land on a %N boundary.
            if plot_clock["last_pause_trial"] != trial_counter:
                plot_clock["last_pause_trial"] = trial_counter
                _refresh_plots(sync=True, reason="pause")
            log.info("AutoPilot paused, waiting...")
            phase.set(
                "paused",
                trial_id=trial_counter,
                idle_reason="autopilot paused",
                poll_s=PAUSE_POLL_S,
            )
            time.sleep(PAUSE_POLL_S)
            continue

        # Check orchestrator health
        phase.set("health_check", trial_id=trial_counter, url=ORCHESTRATOR_URL)
        _health = health_check(ORCHESTRATOR_URL, retries=2)
        if not dry_run and not _health:
            # We keep retrying forever, on purpose (operator ruling 2026-08-12):
            # latching off here would trade a silent spin for a silent stop, and
            # a blip during a long reload would need an overnight resume. What we
            # do NOT do any more is retry silently — after HEALTH_ESCALATE_AFTER
            # consecutive failures the phase heartbeat carries an escalation flag
            # that build_phase_health_report turns into a blocker and a non-ok
            # status, which is what the dashboard and the restart advisor read.
            _escalation = health_escalation.record_failure()
            if _escalation.escalated_now:
                # Once per episode. Re-logging this every backoff would be noise,
                # and noise is what taught everyone to ignore the last one.
                log.critical("AUTOPILOT HEALTH ESCALATION: %s", _escalation.message)
            log.error(
                "Orchestrator unhealthy [%s]: %s — waiting %.1fs... "
                "(consecutive failures: %d/%d%s)",
                _health.failure_reason,
                _health.failure_detail,
                HEALTH_BACKOFF_S,
                _escalation.consecutive,
                HEALTH_ESCALATE_AFTER,
                ", ESCALATED" if _escalation.escalated else "",
            )
            phase.set(
                "health_backoff",
                trial_id=trial_counter,
                idle_reason="orchestrator unhealthy",
                failure_reason=_health.failure_reason,
                failure_detail=_health.failure_detail,
                backoff_s=HEALTH_BACKOFF_S,
                **_escalation.phase_fields,
            )
            time.sleep(HEALTH_BACKOFF_S)
            continue

        # Health is back (or dry-run ignores it). Clear any streak, and say so —
        # an alarm that never visibly resets is an alarm nobody trusts. Returns
        # None when there was no streak, so the healthy steady state is silent.
        _recovery = health_escalation.record_success()
        if _recovery is not None:
            if _recovery.recovered:
                log.warning("AUTOPILOT HEALTH RECOVERED: %s", _recovery.message)
            else:
                log.info(
                    "Orchestrator health recovered after %d consecutive failure(s)",
                    _recovery.phase_fields.get("health_recovered_after_failures", 0),
                )
            phase.set(
                "health_recovered",
                trial_id=trial_counter,
                **_recovery.phase_fields,
            )

        # Check preflight diagnostics for stack-level issues
        phase.set("preflight", trial_id=trial_counter)
        if get_preflight_diagnostics is not None:
            try:
                _pf = get_preflight_diagnostics()
                _last = _pf.get("last_preflight", {})
                if _last.get("status") == "failed":
                    log.warning(
                        "Preflight failed [%s]: %s — %s",
                        _last.get("stage", "?"),
                        _last.get("failure_reason", "?"),
                        _last.get("failure_detail", ""),
                    )
            except Exception:
                pass  # Preflight diagnostics are optional

        # ── 1. Observe ───────────────────────────────────────────
        phase.set("observe", trial_id=trial_counter)
        memory_count = seeder.get_memory_count() if not dry_run else 0
        converged = seeder.is_converged
        hv_slope = archive.hypervolume_slope(50)

        # ── 2. Reason ────────────────────────────────────────────
        if tui is not None:
            tui.set_status("selecting next trial (controller)…")
        critic_fallback_skip: SkipOutcome | None = None
        critic_repeat_skip: SkipOutcome | None = None
        planner_decision: Any | None = None
        action: dict[str, Any] | None = None
        rationale: dict[str, Any] | None = None
        seq_fresh_eval_context: dict[str, Any] | None = None
        seq_baseline_draw_reference: dict[str, Any] | None = None
        seq_candidate_replay_context: dict[str, Any] | None = None
        multitier_validation_context: dict[str, Any] | None = None
        seq_due_bypassed_planner = False
        selectable_action_types: list[str] | None = None
        planner_evidence_text = ""
        outcome_progress_pressure_text = ""
        if action is None:
            (
                action,
                rationale,
                multitier_validation_context,
            ) = _maybe_force_multitier_due_action(
                state=state,
                blacklist=blacklist,
                trial_counter=trial_counter,
            )
            if (
                isinstance(multitier_validation_context, dict)
                and multitier_validation_context.get("stage") == "final_t1"
                and isinstance(multitier_validation_context.get("seq_context"), dict)
            ):
                seq_fresh_eval_context = {
                    "candidate": multitier_validation_context.get("candidate"),
                    "source_trial_id": multitier_validation_context.get("source_trial_id"),
                }
        if action is None:
            (
                action,
                rationale,
                seq_fresh_eval_context,
                seq_baseline_draw_reference,
                seq_candidate_replay_context,
            ) = _maybe_force_seq_due_action(
                state=state,
                journal=journal,
                tier=DEFAULT_FRONTIER_TIER,
                blacklist=blacklist,
                trial_counter=trial_counter,
                enabled=gate.use_sequential,
                lab=lab,
            )
        if action is None:
            preplanner_action, preplanner_rationale = _maybe_force_frontier_rerun_action(
                {"type": "seed_batch", "n_questions": SAFE_FALLBACK_SEED_N},
                state,
                journal=journal,
                archive=archive,
                blacklist=blacklist,
                rationale=rationale,
                trial_counter=trial_counter,
            )
            if preplanner_action != {"type": "seed_batch", "n_questions": SAFE_FALLBACK_SEED_N}:
                action = preplanner_action
                rationale = preplanner_rationale or {}
                phase.set(
                    "planner_bypassed_preemptive_gate",
                    trial_id=trial_counter,
                    gate="frontier_rerun",
                    action_type=action.get("type", ""),
                )
        if use_controller and action is None:
            phase.set(
                "planner_prompt_build",
                trial_id=trial_counter,
                idle_reason="building controller prompt",
            )
            # Load split controller guidance (W8): durable human policy plus
            # generated live system card. program.md remains historical only.
            constitution_text = _read_guidance_file(
                CONSTITUTION_PATH,
                "constitution.md",
            )
            system_card_text = _render_system_card(state)
            # B2: Format blacklist for controller. The full blacklist is always
            # enforced at dispatch by check_blacklist(). The prompt view shows
            # recent entries with reasons plus compact older patterns so the
            # planner/critic do not unknowingly re-propose hidden hard blocks.
            blacklist_text = _format_blacklist_for_prompt(blacklist)

            # AM compaction: query slot memory so controller can decide on compaction
            try:
                slot_memory_text = _query_slot_memory() if not dry_run else "  (dry run)"
            except Exception:
                slot_memory_text = "  (query failed)"

            # Load model signatures for hypothesis assessment.
            #
            # 2026-08-01: `load_model_signatures` now RAISES when the descriptor
            # artifact is unavailable, instead of silently falling back to
            # `model_quality_signatures.yaml` — a hand-maintained table that had
            # drifted three model generations (it described the fleet retired
            # 2026-05-08, at throughputs 1.4x-11x too low). Planning confidently
            # against a dead fleet is worse than not planning.
            #
            # But a REFUSAL and a CRASH are not the same thing. Unhandled, that
            # raise would tear down the controller loop mid-trial. Catch it here
            # and halt deliberately: the operator gets a named artifact and a
            # recompile command, and the trial ends cleanly rather than as a
            # traceback with whatever partial state was in flight.
            try:
                model_sigs = load_model_signatures()
            except ModelSignaturesUnavailableError as exc:
                log.critical(
                    "model signatures unavailable — halting the controller rather "
                    "than planning against unknown models: %s",
                    exc,
                )
                break
            model_signatures_text = format_model_signatures(model_sigs)

            # Adaptive-batch telemetry hint for the controller. Surfaces
            # recent seconds-per-question + recommended n_questions so the
            # planner can request batches that will actually fit the budget,
            # rather than asking for 10 every time and getting silently
            # scaled to 3 by _action_seed_batch's adaptive_batch_size().
            try:
                import sys as _sys

                _sys.path.insert(
                    0,
                    str(_REPO_ROOT / "scripts/benchmark"),
                )
                from seeding_telemetry import (
                    batch_summary as _batch_summary,
                    adaptive_batch_size as _adaptive_n,
                )

                _bs = _batch_summary()
                _rate = _bs.get("median_s_per_q")
                if _rate is None:
                    batch_telemetry_text = (
                        "(no batch history yet — first seed_batch will use "
                        "the requested n; subsequent batches will auto-adapt)"
                    )
                else:
                    _budget_s = float(os.environ.get("SEEDING_BATCH_BUDGET_S", "900"))
                    _max_n_now, _max_reason = _adaptive_n(100)
                    batch_telemetry_text = (
                        f"recent median: {_rate:.0f}s/question over "
                        f"{_bs['n_recent']} batches\n"
                        f"current budget: {_budget_s:.0f}s "
                        f"(SEEDING_BATCH_BUDGET_S)\n"
                        f"realistic max n_questions right now: ~{_max_n_now} "
                        f"(scaler reason: {_max_reason})\n"
                        f"recent batches: "
                        + json.dumps(_bs.get("recent", []), separators=(",", ":"))
                        + "\n"
                        + "IMPORTANT: request seed_batch sizes that fit the "
                        "budget. Asking for 10 questions when the realistic "
                        "max is 3 just gets silently scaled down — better to "
                        "ask for 3 directly so you keep clean attribution."
                    )
            except Exception as _exc:
                batch_telemetry_text = f"(telemetry unavailable: {_exc})"

            # 2026-05-23 planner enrichment — four new sections injected
            # into the prompt:
            #   1. pareto_geometry — frontier shape + blocking points + gaps
            #   2. journal_trustworthiness — bug-corrupted ratio + low-signal flag
            #   3. hypotheses_under_test — last 3 trustworthy trials' hypotheses
            #   4. insights_structured — per-action-type observations + confidence
            #   5. stagnation_signal + exploration_block — stagnation-gated
            #      lean/rich creative-exploration fragment (2026-05-23 upgrade)
            try:
                trust = journal.trustworthiness_score()
                trust_lines = [
                    f"total trials: {trust['total']}",
                    f"trustworthy:  {trust['trustworthy']}",
                    f"corrupted:    {trust['corrupted']}",
                    f"ratio:        {trust['ratio']:.1%}",
                ]
                if trust["corrupted_by"]:
                    trust_lines.append(
                        "corrupted-by breakdown: "
                        + ", ".join(f"{sha}:{n}" for sha, n in trust["corrupted_by"].items())
                    )
                if trust["low_signal"]:
                    trust_lines.append(
                        "WARNING: trustworthy < 5 — hypothesis-chain reasoning has low "
                        "signal. Prefer EXPLORE actions (seed_batch, structural_experiment, "
                        "tail-sampled creative actions) over EXPLOIT (rollback, "
                        "small numeric perturbations) until trustworthiness exceeds 5."
                    )
                # Convergence vs corruption (2026-05-31). reproduction_confirmed
                # trials REPRODUCE an already-kept above-baseline gain: they are
                # measurement-VALID confirmations (the planner has converged on a
                # surface), deliberately NOT counted in `corrupted` above. Surface
                # the count so the planner reads convergence, not "wasted trials".
                try:
                    _recent = journal.recent(10)
                    _repro = sum(
                        1
                        for e in _recent
                        if getattr(e, "deficiency_category", "") == "reproduction_confirmed"
                    )
                except Exception:
                    _repro = 0
                if _repro:
                    trust_lines.append(
                        f"reproduction_confirmed (convergence, last 10): {_repro} — these "
                        "reproduce an established above-baseline gain and are TRUSTWORTHY "
                        "confirmations (config converged on this surface), NOT noise or "
                        "corruption. Do not 'wait out a noise window' or re-run for a fresh "
                        "win; explore a NEW surface or idle cleanly."
                    )
                # Attribution guard (C, 2026-05-31): every exclusion above is a
                # measurement-VALID classification, NOT evidence of host noise or a
                # broken eval tower. The planner must not invent an "exogenous
                # host-load noise window" from exclusions alone (the 2026-05-31
                # meta-action loop did exactly that).
                trust_lines.append(
                    "ATTRIBUTION GUARD: corrupted-by tags are classifications "
                    "(mad_noise=within-noise improvement; reproduction_confirmed="
                    "convergence; a commit SHA=operator code-fix invalidation; "
                    "exogenous_operator_reload=service restart). NONE of these are "
                    "evidence of host/exogenous eval-noise or a broken instrument. Do "
                    "NOT claim a host-load 'noise window' or escalate 'eval tower stuck' "
                    "unless the host-health line below shows an actual throttle/cache/load "
                    "signal; absent that, treat repeated reproduction_confirmed as "
                    "CONVERGENCE and move to a new surface."
                )
                try:
                    from host_health import HostHealthState  # type: ignore

                    _hh = HostHealthState.snapshot()
                    _throttled, _trig = _hh.is_throttled()
                    _mem_warnings = _hh.memory_residency_warnings()
                    trust_lines.append(
                        "host-health: "
                        + (
                            "THROTTLED — " + "; ".join(_trig)
                            if _throttled
                            else "nominal (no CPU-throttle / page-cache / load signal "
                            "→ a host-noise narrative is UNSUPPORTED)"
                        )
                    )
                    if _mem_warnings:
                        trust_lines.append(
                            "memory-residency: ADVISORY — "
                            + "; ".join(_mem_warnings)
                            + "; do not use drop_caches for this class of RAM"
                        )
                except Exception:
                    trust_lines.append(
                        "host-health: not collected this turn — no positive evidence of "
                        "host noise; do not assume a noise window exists."
                    )
                journal_trustworthiness_text = "\n".join(trust_lines)
            except Exception as _exc:
                journal_trustworthiness_text = f"(trustworthiness unavailable: {_exc})"

            try:
                hyps = journal.recent_hypotheses(n=3, exclude_bug_corrupted=True)
                if not hyps:
                    hypotheses_text = "(no recent trustworthy hypotheses)"
                else:
                    hyp_lines: list[str] = []
                    for e in hyps:
                        outcome = f"q={e.quality:.3f} sp={e.speed:.1f} → {e.pareto_status}"
                        hyp_lines.append(
                            f"#{e.trial_id} ({e.species}/{e.action_type}):\n"
                            f"  Hypothesis: {(e.hypothesis or '(none)')[:240]}\n"
                            f"  Outcome:    {outcome}"
                            + (
                                f"\n  Self-criticism: {scrub_legacy_scale_text(e.self_criticism)[:200]}"
                                if e.self_criticism
                                else ""
                            )
                        )
                    hypotheses_text = "\n\n".join(hyp_lines)
            except Exception as _exc:
                hypotheses_text = f"(hypothesis chain unavailable: {_exc})"

            try:
                insights_structured_text = journal.insights_structured_text(
                    n=PLANNER_STRUCTURED_INSIGHTS_LIMIT,
                    exclude_bug_corrupted=True,
                )
            except Exception as _exc:
                insights_structured_text = f"(structured insights unavailable: {_exc})"

            try:
                pareto_geometry_text = archive.geometry_text(tier=DEFAULT_FRONTIER_TIER)
            except Exception as _exc:
                pareto_geometry_text = f"(geometry unavailable: {_exc})"

            # Stepping-stone lane (intake-772 Darwin Gödel Machine): append a diverse sample of
            # dominated-but-novel configs to the geometry block so the planner sees exploration
            # seeds beyond the frontier. Observe-only + fully guarded — never breaks the prompt.
            # Gated by AUTOPILOT_STEPPING_STONES (default on); set to "0" for the frontier-only
            # arm of the ablation in scripts/autopilot/STEPPING_STONE_ABLATION_PROTOCOL.md.
            if os.environ.get("AUTOPILOT_STEPPING_STONES", "1") != "0":
                try:
                    _stepping_text = archive.stepping_stones_text(
                        tier=DEFAULT_FRONTIER_TIER,
                        limit=8,
                    )
                    if _stepping_text:
                        pareto_geometry_text = f"{pareto_geometry_text}\n\n{_stepping_text}"
                except Exception as _exc:
                    pareto_geometry_text = (
                        f"{pareto_geometry_text}\n\n(stepping-stones unavailable: {_exc})"
                    )

            # AP-ME-2 parent utility: a COMPUTED, always-on parent-selection
            # signal (port of aira-evo `compute_parent_utilities`, shipped
            # weights score 1.0 / delta 0.4 / novelty 0.25). Complementary to
            # P17's stagnation-gated BT tiebreak — this one is proactive, so it
            # is rendered unconditionally (not only under hypervolume
            # stagnation). Observe-only + fully guarded — never breaks the
            # prompt. Gate flag follows the stepping-stones convention.
            if os.environ.get("AUTOPILOT_PARENT_UTILITY", "1") != "0":
                try:
                    _parent_util_text = archive.parent_utility_text(
                        tier=DEFAULT_FRONTIER_TIER,
                        limit=8,
                    )
                    if _parent_util_text:
                        pareto_geometry_text = f"{pareto_geometry_text}\n\n{_parent_util_text}"
                except Exception as _exc:
                    pareto_geometry_text = (
                        f"{pareto_geometry_text}\n\n(parent utility unavailable: {_exc})"
                    )

            try:
                planner_evidence_text = format_planner_evidence_section(
                    (asdict(entry) for entry in journal.entries_with_supersessions()),
                    exclude_before_ts=quality_exclude_before_ts,
                )
            except Exception as _exc:
                planner_evidence_text = f"(planner evidence unavailable: {_exc})"
            w8_replay_pressure_active = _w8_replay_pressure_active(planner_evidence_text)

            _refresh_planner_convention_bindings(
                strategy_store,
                journal,
                reason=f"planner_turn:{trial_counter}",
            )

            try:
                _known_actions = [
                    "seed_batch",
                    "numeric_trial",
                    "prompt_mutation",
                    "gepa_optimize",
                    "code_mutation",
                    "structural_experiment",
                    "consult_gate_probe",
                    "structural_prune",
                    "slot_compact",
                    "train_routing_models",
                    "distill_skillbank",
                    "reset_memories",
                    "deep_eval",
                    "rollback",
                    "distill_knowledge",
                ]
                (
                    action_availability_text,
                    viable_tail_actions,
                    selectable_action_types,
                ) = _build_action_availability(
                    journal=journal,
                    known_actions=_known_actions,
                    memory_count=memory_count,
                    converged=converged,
                    slot_memory_text=slot_memory_text,
                    blacklist=blacklist,
                    suppressed_numeric_surfaces=_PLANNER_SUPPRESSED_NUMERIC_SURFACES,
                    w8_replay_pressure_text=planner_evidence_text,
                )
                exploration_block, stagnation_signal = _build_exploration_block(
                    journal=journal,
                    archive=archive,
                    known_actions=viable_tail_actions,
                )
            except Exception as _exc:
                exploration_block = (
                    "Briefly enumerate up to 3 alternatives with one-line reject/accept "
                    "reasons before committing to your single action.\n"
                    f"(exploration-block assembly failed: {_exc})"
                )
                action_availability_text = "(action availability unavailable)"
                selectable_action_types = _known_actions
                stagnation_signal = "unknown"

            planner_strategy_hints_text = _build_planner_strategy_hints(
                strategy_store,
                journal,
            )
            higher_tier_pressure_text = _build_higher_tier_planner_pressure(
                archive,
                gate,
                w8_candidate_generation_active=w8_replay_pressure_active,
            )
            eval_coverage_pressure_text = _build_eval_coverage_pressure(
                journal,
                w8_candidate_generation_active=w8_replay_pressure_active,
            )
            outcome_progress_pressure_text = _build_outcome_progress_pressure()

            vidya_kvq_context = _build_vidya_kvq_context()
            prompt = (
                CONTROLLER_PROMPT_TEMPLATE.format(
                    constitution=constitution_text,
                    system_card=system_card_text,
                    pareto_summary=archive.summary_text(tier=DEFAULT_FRONTIER_TIER),
                    pareto_geometry=pareto_geometry_text,
                    planner_evidence=planner_evidence_text,
                    journal_trustworthiness=journal_trustworthiness_text,
                    hypotheses_under_test=hypotheses_text,
                    journal_summary=journal.summary_text(PLANNER_JOURNAL_SUMMARY_LIMIT),
                    seeder_status=json.dumps(seeder.convergence_status(), indent=2),
                    batch_telemetry=batch_telemetry_text,
                    species_effectiveness=json.dumps(journal.species_effectiveness(), indent=2),
                    health_status="OK" if not dry_run else "dry_run",
                    memory_count=memory_count,
                    converged=converged,
                    slot_memory=slot_memory_text,
                    action_availability=action_availability_text,
                    available_action_schemas=_format_available_action_schemas(
                        selectable_action_types
                    ),
                    model_gate_advisory=_build_model_gate_advisory(),
                    higher_tier_pressure=higher_tier_pressure_text,
                    eval_coverage_pressure=eval_coverage_pressure_text,
                    outcome_progress_pressure=outcome_progress_pressure_text,
                    planner_strategy_hints=planner_strategy_hints_text,
                    repo_readiness_advisory=_build_repo_readiness_advisory(),
                    budget=json.dumps(meta.budget.as_dict(), indent=2),
                    suite_quality_trends=_format_suite_trends(journal.suite_quality_trend(10)),
                    insights_structured=insights_structured_text,
                    stagnation_signal=stagnation_signal,
                    exploration_block=exploration_block,
                    short_term_memory=memory.refresh_from_journal(journal),  # W5 generated STM
                    prior_planner_decisions=_build_prior_planner_decision_digest(),
                    last_criticism=last_criticism_text,  # AP-23
                    model_signatures=model_signatures_text,
                    blacklist_text=blacklist_text,
                    operator_outbox_feedback=_build_operator_outbox_feedback(),
                    operator_hypotheses_block=_build_operator_hypotheses_block(blacklist),
                    vidya_settled_ground_block=_build_vidya_settled_ground_block(),
                    vidya_kvq_evidence_block=vidya_kvq_context["text"],
                    feature_flags_block=_build_feature_flags_block(
                        lab,
                        denylisted_flags=_PLANNER_DENYLISTED_FEATURE_FLAGS,
                    ),
                    last_invalid_feedback=(
                        _build_last_invalid_feedback(state)
                        + _build_rejected_config_feedback(journal)  # AP-53
                    ),
                    plot_paths="\n".join(f"  - {p}" for p in plot_paths) or "  (none yet)",
                )
                + peaf.peaf_prompt_addendum()
            )

            phase.set(
                "planner_invoke",
                trial_id=trial_counter,
                prompt_chars=len(prompt),
                session_id=state.get("session_id"),
                idle_reason="planner subprocess running",
            )
            planner_provider_state = state.setdefault("planner_providers", {})
            if not isinstance(planner_provider_state, dict):
                planner_provider_state = {}
                state["planner_providers"] = planner_provider_state
            planner_decision = plan_with_providers(
                prompt,
                session_id=state.get("session_id"),
                cwd=ORCH_ROOT,
                planner_state=planner_provider_state,
                stagnation_signal=stagnation_signal,
                allowed_action_types=selectable_action_types,
                action_feedback_state=state,
                trial_id=trial_counter,
                vidya_kvq_context=vidya_kvq_context,
            )
            phase.set(
                "planner_parse",
                trial_id=trial_counter,
                response_chars=len(planner_decision.canonical_text or ""),
                session_id=planner_decision.session_id,
                draft_provider=planner_decision.draft_provider,
                critic_provider=planner_decision.critic_provider,
                planner_mode=planner_decision.mode,
                degraded=planner_decision.degraded,
                fallback_reason=planner_decision.fallback_reason,
            )
            if planner_decision.fallback_reason:
                log.warning(
                    "Planner fallback/degraded mode: %s",
                    planner_decision.fallback_reason,
                )
            if planner_decision.critique is not None:
                log.info(
                    "Planner critique by %s: %s confidence=%.2f issues=%s",
                    planner_decision.critique.provider or "(none)",
                    planner_decision.critique.decision,
                    planner_decision.critique.confidence,
                    planner_decision.critique.issues,
                )
            state["session_id"] = planner_decision.session_id
            action = planner_decision.action
            predicted_objectives = planner_decision.predicted_objectives
            rationale = planner_decision.rationale

            # Degraded/uncritiqued safety gate (2026-06-07, tightened 2026-06-12):
            # two cases (see uncritiqued_dispatch_block_reason).
            #   (A) NO critique object — the PRIMARY failed and the draft fell back
            #       to the critic provider: uncritiqued AND untrusted, so dispatch
            #       only if observational, else PAUSE.
            #   (B) verdict "unavailable" — the PRIMARY drafted fine but the binding
            #       critic could not review it (timeout/empty/unparseable/circuit).
            #       RISK CLASS ALONE IS NOT A SUFFICIENT GUARD (the @708 failure was
            #       "low-risk" seed looping): HIGH → pause; seed_batch/passive →
            #       pause unless explicitly observational+one-shot; MEDIUM experiment
            #       → proceed ONLY IF novel and non-looping (not blacklisted, not a
            #       recurring-invalid signature, carries a real falsifier). The gate
            #       is pure, so supply the blacklist + invalid-signature context here.
            # Mirrors the critic_reject_loop / meta / skip durable halts.
            _gate_action = planner_decision.action
            _gate_is_bl = bool(
                isinstance(_gate_action, dict) and check_blacklist(_gate_action, blacklist)
            )
            if (
                _gate_is_bl
                and isinstance(_gate_action, dict)
                and _p0_3_retryable_blacklist_match(_gate_action, blacklist) is not None
            ):
                _gate_is_bl = False
            _gate_is_rep = bool(
                isinstance(_gate_action, dict)
                and _action_signature(_gate_action)
                in (state.get("invalid_signature_counts", {}) or {})
            )
            _uncritiqued_block = uncritiqued_dispatch_block_reason(
                planner_decision,
                is_blacklisted=_gate_is_bl,
                is_repeated=_gate_is_rep,
            )
            if _uncritiqued_block:
                state["paused"] = True
                state["_dispatch_deficiency"] = _uncritiqued_block
                save_state(state)
                log.error(
                    "Planner ran degraded with NO critic verdict and drafted a "
                    "non-observational action %r — pausing (%s) for operator review "
                    "instead of dispatching unreviewed (fallback_reason=%r).",
                    action.get("type") if isinstance(action, dict) else action,
                    _uncritiqued_block,
                    planner_decision.fallback_reason,
                )
                phase.set("critic_unavailable_halt", trial_id=trial_counter)
                break

            # draft_critique authority: a BINDING reject/revise substituted the
            # planner's draft (safe fallback / revised_action is what `action`
            # now holds). Record the rejected draft so it cannot bypass the
            # invalid-action feedback + blacklist — otherwise the planner could
            # re-draft the same rejected action forever while the critic
            # silently swaps in the fallback. The substituted `action` still
            # flows through meta/quota/blacklist/dispatch below unchanged.
            crit = planner_decision.critique
            draft_action = planner_decision.draft_action
            if (
                planner_decision.mode == "draft_critique"
                and crit is not None
                and crit.decision in ("reject", "revise")
                and draft_action
                and draft_action != action
            ):
                _record_rejected_draft(state, draft_action, crit, trial_counter)
                blacklist = load_blacklist()  # may have grown
                action, rationale, critic_fallback_skip, critic_rejection_repaired = (
                    _repair_critic_reject_fallback_for_w8(
                        action,
                        blacklist,
                        rationale,
                        trial_counter=trial_counter,
                        w8_replay_pressure_text=planner_evidence_text,
                    )
                )
                if critic_fallback_skip is not None:
                    state["paused"] = True
                    state["_dispatch_deficiency"] = "critic_reject_no_safe_fallback"
                    state["last_invalid_action"] = action
                    state["last_invalid_reason"] = critic_fallback_skip.reason
                    state["last_invalid_status"] = "critic_reject_no_safe_fallback"
                    save_state(state)
                    log.error(
                        "Critic rejected/revised planner draft but the safe fallback "
                        "is unavailable — pausing before dispatch instead of "
                        "journaling a skipped trial (%s).",
                        critic_fallback_skip.reason,
                    )
                    phase.set(
                        "critic_reject_no_safe_fallback_halt",
                        trial_id=trial_counter,
                    )
                    break
                if critic_rejection_repaired:
                    state["consecutive_rejected_drafts"] = 0
                if (
                    int(state.get("consecutive_rejected_drafts", 0))
                    >= MAX_CONSECUTIVE_REJECTED_DRAFTS
                ):
                    # Durable halt mirroring the meta/skip breakers: the planner
                    # keeps drafting actions the critic overrides — operator call.
                    state["paused"] = True
                    state["_dispatch_deficiency"] = "critic_reject_loop"
                    save_state(state)
                    log.error(
                        "Critic rejected/revised %d consecutive planner drafts — "
                        "pausing for operator review (stuck planner the critic "
                        "keeps overriding).",
                        int(state.get("consecutive_rejected_drafts", 0)),
                    )
                    phase.set("critic_reject_loop_halt", trial_id=trial_counter)
                    break
            else:
                # Draft accepted (approve / not critiqued / no substitution).
                state["consecutive_rejected_drafts"] = 0
                # The deterministic guard's breaker counts a RUN, so a single clean
                # dispatch clears it — otherwise isolated blocks spread across unrelated
                # trials would eventually halt a perfectly healthy planner.
                state["consecutive_planner_deterministic_blocks"] = 0
        elif not use_controller:
            # Autonomous mode: species selection by budget
            phase.set("autonomous_select", trial_id=trial_counter)
            species = meta.select_species()
            action = _auto_action(species, memory_count, converged, seeder)
            predicted_objectives = {}  # PEAF: autonomous mode has no controller forecast
            rationale = {"falsifier": "", "rubric_scores": {}}  # no controller call
            stagnation_signal = ""  # gate is controller-only; autonomous mode skips it
            state["consecutive_rejected_drafts"] = 0  # no critic in autonomous mode
            state["consecutive_planner_deterministic_blocks"] = 0  # no planner guard either
            action, rationale = _replace_blacklisted_autonomous_action(
                action,
                blacklist,
                rationale,
            )
        else:
            predicted_objectives = {}
            rationale = rationale or {}
            stagnation_signal = ""
            state["consecutive_rejected_drafts"] = 0
            seq_due_bypassed_planner = True
            phase.set(
                "planner_bypassed_seq_due",
                trial_id=trial_counter,
                action_type=action.get("type", ""),
                seq_promotion_fresh_eval=seq_fresh_eval_context is not None,
                seq_baseline_draw=seq_baseline_draw_reference is not None,
                seq_candidate_replay=seq_candidate_replay_context is not None,
                multitier_validation=multitier_validation_context is not None,
            )

        if not action:
            if getattr(planner_decision, "providers_unavailable", False):
                # BOTH planner models are offline (no usable response from any
                # attempted model). Don't loop a model-free seed_batch silently and
                # don't route through the LLM-critic gate (it's also offline). Pick a
                # DETERMINISTIC, statistically-grounded Optuna numeric_trial if the
                # numeric-swarm surfaces are available; otherwise pause for the
                # operator. (cross-model failover, 2026-06-12)
                _det_surface = None
                try:
                    from species.numeric_swarm import SURFACES as _NS_SURFACES

                    if _NS_SURFACES:
                        _surface_names = sorted(_NS_SURFACES.keys())
                        _det_surface = (
                            "memrl_retrieval"
                            if "memrl_retrieval" in _NS_SURFACES
                            else _surface_names[0]
                        )
                except Exception:
                    _det_surface = None

                if _det_surface is not None:
                    # numeric_trial without "params" → Optuna fills them at execution
                    # (trial 969 ran exactly this shape; the dispatch path handles it).
                    action = {"type": "numeric_trial", "surface": _det_surface}
                    log.warning(
                        "Both planner models offline → deterministic Optuna "
                        "numeric_trial fallback (surface=%s)",
                        _det_surface,
                    )
                else:
                    state["paused"] = True
                    state["_dispatch_deficiency"] = "planners_offline_no_deterministic_fallback"
                    save_state(state)
                    log.error(
                        "Both planner models offline and no deterministic fallback "
                        "available — pausing for operator."
                    )
                    phase.set("planners_offline_halt", trial_id=trial_counter)
                    break
            elif getattr(planner_decision, "deterministic_block_reason", ""):
                block_reason = planner_decision.deterministic_block_reason
                decision, consecutive_blocks = _planner_deterministic_block_decision(
                    state, planner_decision
                )
                if decision == "halt":
                    # A RUN of these is a genuinely stuck planner — halt for the operator,
                    # mirroring the critic-reject-loop breaker.
                    state["paused"] = True
                    state["_dispatch_deficiency"] = "planner_deterministic_guard"
                    save_state(state)
                    log.error(
                        "Planner action blocked deterministically %d consecutive times "
                        "— pausing for operator review. Last reason: %s",
                        consecutive_blocks,
                        block_reason,
                    )
                    phase.set("planner_deterministic_guard_halt", trial_id=trial_counter)
                    break
                save_state(state)
                log.warning(
                    "Planner action blocked deterministically (%d/%d consecutive): %s "
                    "— substituting seed_batch and continuing. The rejected draft still "
                    "feeds invalid-action feedback, so the planner learns from it.",
                    consecutive_blocks,
                    MAX_CONSECUTIVE_PLANNER_DETERMINISTIC_BLOCKS,
                    block_reason,
                )
                action = {"type": "seed_batch", "n_questions": SAFE_FALLBACK_SEED_N}
            else:
                log.warning("No action proposed, defaulting to seed_batch")
                action = {"type": "seed_batch", "n_questions": SAFE_FALLBACK_SEED_N}

        if not seq_due_bypassed_planner:
            critic_repeat_skip = _critic_rejected_signature_skip(action, state)

        if not seq_due_bypassed_planner and critic_repeat_skip is None:
            # Meta actions are allowed as occasional bookkeeping, but a repeated
            # metric-free action means the planner is avoiding the experiment loop.
            action, rationale = _force_metric_action_after_meta(
                action,
                state,
                rationale,
                blacklist,
            )

            # Experiment quota: once memory is large, cap consecutive passive
            # (seed/distill) actions so the planner cannot rationalize no-op work
            # forever — force a frontier-moving experiment instead.
            action, rationale = _enforce_experiment_quota(
                action,
                state,
                memory_count,
                rationale,
                trial_counter,
                blacklist,
            )

            action, rationale, seq_fresh_eval_context = _maybe_force_seq_promotion_fresh_eval(
                action,
                state=state,
                blacklist=blacklist,
                rationale=rationale,
                trial_counter=trial_counter,
                enabled=gate.use_sequential,
            )
            if seq_fresh_eval_context is None:
                action, rationale, seq_baseline_draw_reference = _maybe_force_seq_baseline_draw(
                    action,
                    state=state,
                    journal=journal,
                    tier=DEFAULT_FRONTIER_TIER,
                    blacklist=blacklist,
                    rationale=rationale,
                    trial_counter=trial_counter,
                    enabled=gate.use_sequential,
                )
            if seq_fresh_eval_context is None and seq_baseline_draw_reference is None:
                action, rationale, seq_candidate_replay_context = _maybe_force_seq_candidate_replay(
                    action,
                    state=state,
                    journal=journal,
                    tier=DEFAULT_FRONTIER_TIER,
                    blacklist=blacklist,
                    rationale=rationale,
                    trial_counter=trial_counter,
                    enabled=gate.use_sequential,
                    lab=lab,
                )
            if (
                seq_fresh_eval_context is None
                and seq_baseline_draw_reference is None
                and seq_candidate_replay_context is None
            ):
                action, rationale = _replace_w8_candidate_generation_deferral(
                    action,
                    blacklist,
                    rationale,
                    trial_counter=trial_counter,
                    w8_replay_pressure_text=planner_evidence_text,
                )
            if (
                seq_fresh_eval_context is None
                and seq_baseline_draw_reference is None
                and seq_candidate_replay_context is None
            ):
                action, rationale = _maybe_force_higher_tier_probe(
                    action,
                    state,
                    journal=journal,
                    archive=archive,
                    blacklist=blacklist,
                    rationale=rationale,
                    trial_counter=trial_counter,
                    w8_replay_pressure_text=planner_evidence_text,
                    outcome_progress_pressure_text=outcome_progress_pressure_text,
                )
            if (
                seq_fresh_eval_context is None
                and seq_baseline_draw_reference is None
                and seq_candidate_replay_context is None
            ):
                action, rationale = _maybe_force_outcome_progress_action(
                    action,
                    state,
                    blacklist=blacklist,
                    rationale=rationale,
                    trial_counter=trial_counter,
                    outcome_progress_pressure_text=outcome_progress_pressure_text,
                )

        if critic_repeat_skip is None:
            action, rationale = _maybe_force_frontier_rerun_action(
                action,
                state,
                journal=journal,
                archive=archive,
                blacklist=blacklist,
                rationale=rationale,
                trial_counter=trial_counter,
            )

        if critic_repeat_skip is None:
            action, rationale, seq_gate_preflight = _maybe_defer_seq_unreachable_candidate_action(
                action,
                state=state,
                journal=journal,
                blacklist=blacklist,
                rationale=rationale,
                trial_counter=trial_counter,
                tier=DEFAULT_FRONTIER_TIER,
                enabled=gate.use_sequential,
            )
            if seq_gate_preflight is not None:
                seq_fresh_eval_context = None
                seq_candidate_replay_context = None
                phase.set(
                    "seq_gate_preflight",
                    trial_id=trial_counter,
                    action_type=action.get("type", ""),
                    seq_gate_preflight_status=seq_gate_preflight.get("status"),
                    seq_gate_preflight_reason=seq_gate_preflight.get("reason"),
                )
                seq_gate_block_reason = _seq_gate_preflight_dispatch_block_reason(
                    seq_gate_preflight
                )
                if seq_gate_block_reason:
                    state["paused"] = True
                    state["_dispatch_deficiency"] = "seq_gate_preflight_blocked"
                    state["last_invalid_action"] = seq_gate_preflight.get(
                        "original_action",
                        action,
                    )
                    state["last_invalid_reason"] = seq_gate_block_reason
                    state["last_invalid_status"] = "seq_gate_preflight_blocked"
                    save_state(state)
                    log.error(
                        "Seq gate preflight blocked trial %d before dispatch (%s): %s",
                        trial_counter,
                        seq_gate_block_reason,
                        json.dumps(seq_gate_preflight, default=str),
                    )
                    phase.set(
                        "seq_gate_preflight_halt",
                        trial_id=trial_counter,
                        seq_gate_preflight_status=seq_gate_preflight.get("status"),
                        seq_gate_preflight_reason=seq_gate_preflight.get("reason"),
                    )
                    break

        # ── 3. Act ───────────────────────────────────────────────
        # B2: Check failure blacklist before dispatch
        pre_dispatch_skip: SkipOutcome | None = critic_fallback_skip or critic_repeat_skip
        if pre_dispatch_skip is not None:
            log.warning(
                "Trial %d: %s",
                trial_counter,
                pre_dispatch_skip.reason,
            )
        else:
            action, rationale = _replace_blacklisted_w8_candidate_action(
                action,
                blacklist,
                rationale,
                trial_counter=trial_counter,
                w8_replay_pressure_text=planner_evidence_text,
            )
            action, rationale = _replace_blacklisted_seed_fallback(
                action,
                blacklist,
                rationale,
                reason_label="pre-dispatch",
            )
            blocked_reason = check_blacklist(action, blacklist)
            if blocked_reason:
                retry_meta = _p0_3_retryable_blacklist_match(action, blacklist)
                if retry_meta is not None:
                    rationale = _record_p0_3_reexploration_rationale(rationale, retry_meta)
                    log.warning(
                        "Trial %d: action matches P0.3 retryable blacklist target %s; "
                        "dispatching audit-scoped re-exploration instead of invalid skip",
                        trial_counter,
                        retry_meta.get("target_key", "unknown"),
                    )
                else:
                    log.warning(
                        "Trial %d: action blacklisted (%s), recording invalid skip",
                        trial_counter,
                        blocked_reason,
                    )
                    pre_dispatch_skip = _blacklisted_action_skip(action, blocked_reason)

        log.info("Trial %d: %s", trial_counter, json.dumps(action))
        current_action = action
        phase.set(
            "action_selected",
            trial_id=trial_counter,
            action_type=action.get("type", ""),
        )

        # Update TUI with trial info
        if tui is not None:
            species_hint = action.get("type", "unknown")
            tui.set_trial(trial_counter, species_hint)
            # Show the action description as the "current prompt" in TUI
            prompt_preview = action.get("description", "")
            if not prompt_preview:
                prompt_preview = json.dumps(action, indent=2)[:500]
            tui.set_prompt(prompt_preview)

        # Phase 6b — write in_flight_trial marker BEFORE dispatch_action.
        # Atomic save_state (Phase 6a) guarantees this either lands fully
        # or not at all. The marker is cleared after the final save_state
        # at the end of the trial; if autopilot crashes in between, the
        # next startup's recovery block sees the marker and either:
        #   - finds the trial in the journal (case a) → re-syncs counter
        #   - finds nothing (case b) → writes AUTOPILOT_KILLED placeholder
        # Both cases prevent silent corruption of the planner's view.
        dispatch_infra_fingerprint: dict[str, Any] | None = None
        trial_served_content: dict[str, Any] | None = None  # set by mutation handlers
        if pre_dispatch_skip is None:
            dispatch_infra_fingerprint = _infra_fingerprint_for_trial()  # AP-55
            state["in_flight_trial"] = {
                "trial_id": trial_counter,
                "action": action,
                "run_manifest": build_run_manifest(
                    source_paths=_run_manifest_source_paths(),
                    task=action,
                    evaluator=_run_manifest_evaluator(),
                ),
                "started_at": time.time(),
                "host_pid": os.getpid(),
                "host_started_at": state.get("autopilot_fleet_started_at"),
            }
            save_state(state)

        if pre_dispatch_skip is not None:
            phase.set(
                "dispatch_precheck_skip",
                trial_id=trial_counter,
                action_type=pre_dispatch_skip.action_type,
                skip_status=pre_dispatch_skip.status,
            )
            eval_result = pre_dispatch_skip
            species_name = pre_dispatch_skip.action_type or action.get("type", "unknown")
        elif dry_run:
            phase.set(
                "dispatch_dry_run", trial_id=trial_counter, action_type=action.get("type", "")
            )
            eval_result = EvalResult(tier=0, quality=2.5, speed=15.0, cost=0.3, reliability=0.95)
            species_name = action.get("type", "unknown").split("_")[0]
        else:
            phase.set(
                "dispatch_action",
                trial_id=trial_counter,
                action_type=action.get("type", ""),
                idle_reason="running selected action",
            )
            # Operator decision 2026-09-16: a mutation handler leaves the sha of the file it
            # served; clear any stale record so this trial only ever sees its own.
            state.pop(SERVED_CONTENT_STATE_KEY, None)
            eval_result, species_name = dispatch_action(
                action,
                seeder,
                swarm,
                forge,
                lab,
                tower,
                gate,
                archive,
                journal,
                state,
                strategy_store=strategy_store,
                evo=evo,
                watcher=watcher,
                allowed_action_types=_dispatch_allowed_action_types(
                    selectable_action_types,
                    seq_due_bypassed_planner=seq_due_bypassed_planner,
                    seq_fresh_eval_context=seq_fresh_eval_context,
                    seq_baseline_draw_reference=seq_baseline_draw_reference,
                    seq_candidate_replay_context=seq_candidate_replay_context,
                    multitier_validation_context=multitier_validation_context,
                    seq_gate_preflight=seq_gate_preflight,
                ),
            )
            trial_served_content = state.pop(SERVED_CONTENT_STATE_KEY, None)
            if trial_served_content is None and isinstance(multitier_validation_context, dict):
                trial_served_content = _multitier_validation_served_content(
                    state, multitier_validation_context, trial_counter
                )
            phase.set(
                "dispatch_complete",
                trial_id=trial_counter,
                action_type=action.get("type", ""),
                species=species_name,
            )

        # ── 4. Evaluate ─────────────────────────────────────────
        if eval_result is None or isinstance(eval_result, SkipOutcome):
            action_type = action.get("type", "")
            if eval_result is None and action_type in META_NOOP_ACTIONS:
                # Meta/housekeeping action: it executed its effect but collected
                # no metrics, so it is NOT an independent trial — do not consume a
                # trial number. Track consecutive meta actions so a stuck planner
                # that only emits no-ops (gate-lock symptom) halts loudly instead
                # of spinning the counter forever.
                meta_streak = int(state.get("consecutive_meta_actions", 0)) + 1
                state["consecutive_meta_actions"] = meta_streak
                state["in_flight_trial"] = None
                save_state(state)
                log.info(
                    "Meta action '%s' executed (no eval, no metrics) — not counted "
                    "as a trial (consecutive meta=%d, trial stays at %d)",
                    action_type,
                    meta_streak,
                    trial_counter,
                )
                if meta_streak >= MAX_CONSECUTIVE_META:
                    # Durable, terminal-until-operator-resume halt. Previously
                    # this only `break`-ed the inner loop; an external supervisor
                    # then restarted the process straight back into the same
                    # gate-locked state (consecutive_meta_actions persisted on
                    # disk), so it re-halted on meta action #1 (2026-05-31).
                    # Setting paused=True LATCHES the halt: the iteration-top
                    # paused gate idles this process AND any restart until the
                    # operator explicitly resumes (clearing paused resets the
                    # counter — see top of loop), so a supervisor cannot re-enter
                    # the no-op loop.
                    halt_reason = _classify_meta_halt(journal)
                    state["paused"] = True
                    state["_dispatch_deficiency"] = "meta_action_loop"
                    state["_meta_halt_reason"] = halt_reason
                    save_state(state)
                    if halt_reason == "converged":
                        log.warning(
                            "Planner emitted %d consecutive metric-free meta actions; "
                            "recent metric trials are reproduction-confirmed — the planner "
                            "has CONVERGED on the current surface (benign, not a malfunction "
                            "and not instrument noise). Pausing cleanly; resume after "
                            "pointing it at a NEW surface.",
                            meta_streak,
                        )
                    else:
                        log.error(
                            "Planner emitted %d consecutive metric-free meta actions "
                            "without running a single trial — pausing for operator review "
                            "(likely a gate-lock or stuck planner).",
                            meta_streak,
                        )
                    phase.set(
                        "meta_loop_halt",
                        trial_id=trial_counter,
                        meta_streak=meta_streak,
                        halt_reason=halt_reason,
                    )
                    break
                continue

            # Non-executing action (invalid flags / AP-9 scope / dirty-tree
            # fence / unknown type / handler no-op). This was the blind spot: the
            # old path bumped the counter and `continue`d, discarding the reason —
            # so the planner re-sampled an impossible action 119× until max_trials
            # (the graph_router deadlock). Now it leaves durable residue: the
            # reason is journaled, fingerprinted, counted, fed back to the next
            # planner prompt, blacklisted on repeat, and circuit-broken on a run.
            if isinstance(eval_result, SkipOutcome):
                skip_status = eval_result.status
                skip_reason = eval_result.reason
                skip_bug_corrupted_by = getattr(eval_result, "bug_corrupted_by", "") or ""
                skip_bug_corrupted_reason = getattr(eval_result, "bug_corrupted_reason", "") or ""
            else:
                skip_status = "skipped"
                skip_reason = f"{action_type} returned no eval result (handler no-op)"
                skip_bug_corrupted_by = ""
                skip_bug_corrupted_reason = ""

            sig = _action_signature(action)
            sig_counts = state.setdefault("invalid_signature_counts", {})
            if skip_bug_corrupted_by:
                sig_seen = int(sig_counts.get(sig, 0))
                skip_streak = int(state.get("consecutive_skip_actions", 0))
            else:
                sig_counts[sig] = int(sig_counts.get(sig, 0)) + 1
                sig_seen = sig_counts[sig]
                skip_streak = int(state.get("consecutive_skip_actions", 0)) + 1
                state["consecutive_skip_actions"] = skip_streak
                state["last_invalid_action"] = action
                state["last_invalid_reason"] = skip_reason
                state["last_invalid_status"] = skip_status

            try:
                _record_skip_trial(
                    journal,
                    trial_counter,
                    action,
                    species_name,
                    skip_status,
                    skip_reason,
                    memory_count,
                    bug_corrupted_by=skip_bug_corrupted_by,
                    bug_corrupted_reason=skip_bug_corrupted_reason,
                    run_manifest=_in_flight_run_manifest(state, trial_counter),  # AP-63(a)
                )
            except Exception:
                log.debug("skip-trial journal write failed", exc_info=True)

            if skip_bug_corrupted_by:
                log.warning(
                    "Trial %d %s (%s): %s [bug_corrupted_by=%s; not counted "
                    "against planner signature pressure]",
                    trial_counter,
                    skip_status,
                    action_type,
                    skip_reason,
                    skip_bug_corrupted_by,
                )
            else:
                log.warning(
                    "Trial %d %s (%s): %s [signature seen %d×, consecutive skips=%d]",
                    trial_counter,
                    skip_status,
                    action_type,
                    skip_reason,
                    sig_seen,
                    skip_streak,
                )
            phase.set(
                "dispatch_skip",
                trial_id=trial_counter,
                action_type=action_type,
                skip_status=skip_status,
            )

            # Auto-blacklist a repeated INVALID signature (stable validator
            # reason → safe to pattern-match). "skipped" outcomes are NOT
            # blacklisted: their coarse pattern would over-match (e.g. one
            # scope-violating numeric_trial would ban all numeric_trials).
            if (
                not skip_bug_corrupted_by
                and skip_status == "invalid"
                and sig_seen >= INVALID_SIGNATURE_BLACKLIST_THRESHOLD
            ):
                append_blacklist(
                    action,
                    trial_counter,
                    f"Auto-blacklisted: {sig_seen}× invalid — {skip_reason[:80]}",
                    reason_class="invalid_repeat",
                )
                blacklist = load_blacklist()

            trial_counter += 1
            state["trial_counter"] = trial_counter
            state["in_flight_trial"] = None
            save_state(state)

            # Hard circuit-breaker mirroring MAX_CONSECUTIVE_META: a run of
            # non-executing actions means a stuck planner / impossible action
            # space. Latch paused=True so a supervisor restart cannot re-enter
            # the loop (same durable-halt semantics as the meta-action guard).
            if not skip_bug_corrupted_by and skip_streak >= MAX_CONSECUTIVE_SKIP:
                state["paused"] = True
                state["_dispatch_deficiency"] = "skip_action_loop"
                save_state(state)
                log.error(
                    "Planner emitted %d consecutive non-executing actions "
                    "(last: %s — %s); pausing for operator review (stuck planner "
                    "or impossible action space).",
                    skip_streak,
                    action_type,
                    skip_reason,
                )
                phase.set(
                    "skip_loop_halt",
                    trial_id=trial_counter,
                    skip_streak=skip_streak,
                )
                break
            continue

        # AP-13: Emit grep-parseable metrics
        phase.set("safety_gate", trial_id=trial_counter, species=species_name)
        log.info("\n%s", eval_result.to_grep_lines(trial_counter, species_name))

        # ── Pre-gate exogenous-reload classification (handoff Phase 5) ──
        # If the trial picked up an operator/external service reload (and at
        # least one question stayed unrecovered after retry), classify it as
        # bug-corrupted BEFORE running the safety gate or admitting it to the
        # Pareto archive. The aggregate quality/reliability numbers are
        # partially based on missing data; treating it as a real failure would
        # increment consecutive_failures + trigger spurious rollback + admit
        # a (0, 0, ?, ?) point to the archive.
        #
        # Trials that detected exogenous reloads but RECOVERED via retry are
        # sound — they just carry audit metadata in eval_details so the
        # operator can see this trial weathered a reload cleanly.
        has_exo_unrecovered = getattr(eval_result, "n_exogenous_unrecovered", 0) > 0
        has_exo_recovered = getattr(eval_result, "n_exogenous_recovered", 0) > 0
        seq_finalized = False
        seq_inputs: dict[str, Any] | None = None

        # EV-14e: pin the baseline identity HERE — before the gate runs and, critically,
        # before `gate.update_baseline()` (below) can move the reference. Captured at this
        # point the pin is exactly the reference the comparison is about to use; captured at
        # journal-write time it would be the POST-promotion reference, i.e. the candidate's
        # own number masquerading as its own incumbent. Recorded on the JournalEntry so the
        # delta is self-contained instead of being regex-recovered from failure prose.
        baseline_pin_record = _baseline_pin_for_trial(gate, eval_result)
        # AP-55: the regime this trial was measured in, compared with the regime of the
        # baseline reference it is about to be judged against (captured before
        # update_baseline() can replace that reference).
        trial_infra_fingerprint = _infra_fingerprint_for_trial()
        trial_comparability = _infra_comparability_for_trial(
            state,
            int(getattr(eval_result, "tier", 0) or 0),
            trial_infra_fingerprint,
            dispatch_fingerprint=dispatch_infra_fingerprint,
            baseline_revision=(
                baseline_pin_record.get("baseline_revision")
                if isinstance(baseline_pin_record, dict) else None
            ),
        )
        if isinstance(baseline_pin_record, dict) and baseline_pin_record:
            baseline_pin_record["infra_comparability"] = trial_comparability.get("status")

        if has_exo_unrecovered:
            # Bypass safety gate + archive update. Trial is journaled below
            # as a bug-corrupted placeholder for audit; the planner's
            # trustworthiness gate excludes it from learning surfaces.
            from safety_gate import SafetyVerdict  # type: ignore

            verdict = SafetyVerdict(passed=True)
            failure_analysis = (
                f"Excluded: unrecovered operator/service reload during trial "
                f"({eval_result.n_exogenous_unrecovered}/{eval_result.n_questions} "
                f"questions affected)"
            )
            log.warning(
                "Trial %d skipped safety gate + archive: %s",
                trial_counter,
                failure_analysis,
            )
        else:
            # Safety gate
            seq_inputs = _seq_inputs_for_trial(
                journal=journal,
                action=action,
                tier=eval_result.tier,
                candidate_override=(
                    str(seq_fresh_eval_context.get("candidate")) if seq_fresh_eval_context else None
                ),
                quality_exclude_before_ts=quality_exclude_before_ts,
            )
            try:
                verdict = gate.check(
                    eval_result,
                    question_results=list(getattr(eval_result, "question_results", []) or []),
                    # SEQ-B: paired with the incumbent comparator built in
                    # `_seq_inputs_for_trial`. NOT `task_rate_qph_from` — that divides the
                    # decision-partition question count by the full batch's wall clock.
                    task_rate=seq_task_rate_qph_from(eval_result),
                    baseline_profile=seq_inputs["baseline_profile"],
                    baseline_task_rate=seq_inputs["baseline_task_rate"],
                    prior_quality_obs=seq_inputs["prior_quality_obs"],
                    prior_rate_obs=seq_inputs["prior_rate_obs"],
                    candidate=seq_inputs["candidate"],
                    core_id=seq_inputs["core_id"],
                )
            except Exception as exc:  # noqa: BLE001 - SEQ-3b: seq inputs must never crash the loop
                # SEQ-3b: mirror the actions.py action-gate fallback
                # (_action_gate_check). A corrupt journal-derived seq input (e.g. an
                # out-of-range z that slipped past the rebuild guard) must fail THIS
                # trial safely, not tear down the whole optimization loop. Fall back to
                # the legacy (non-sequential) gate check: baseline_profile omitted =>
                # the e-process path is not taken => verdict.seq is None =>
                # seq_confirmed=None downstream (conservative: no baseline ratchet).
                log.warning(
                    "Main-loop safety gate fell back to legacy check "
                    "(sequential inputs raised %s: %s)",
                    type(exc).__name__,
                    exc,
                )
                verdict = gate.check(eval_result)
            if isinstance(getattr(verdict, "seq", None), dict):
                verdict.seq["alpha_wealth"] = seq_inputs.get("alpha_wealth")
            seq_finalized = _annotate_seq_promotion_finalization(
                verdict.seq,
                baseline_reference=seq_inputs.get("baseline_reference"),
                is_fresh_eval=seq_fresh_eval_context is not None,
                fresh_eval_context=seq_fresh_eval_context,
            )
            if (
                verdict.seq is not None
                and verdict.seq.get("baseline_reference_state") == "stale-reference"
            ):
                verdict.categories.append("seq_stale_reference")
            failure_analysis = gate.analyze_failure(eval_result, verdict)
            if not verdict:
                log.warning("Safety violations: %s", "; ".join(verdict.violations))
                if gate.should_rollback():
                    log.error("Consecutive failure limit reached, rolling back")
                    state["_dispatch_deficiency"] = "consecutive_failures"  # AP-14
                    # B2: Auto-append failing config to blacklist
                    append_blacklist(
                        action,
                        trial_counter,
                        f"Auto-blacklisted: 3 consecutive failures ending at trial {trial_counter}",
                        reason_class="safety_failure",
                    )
                    blacklist = load_blacklist()  # Reload after append
                    if strategy_store is not None:
                        try:
                            strategy_store.close()
                        except Exception as exc:
                            log.warning(
                                "StrategyStore close before rollback restore failed: %s", exc
                            )
                    restore_result = lab.restore_checkpoint()
                    memory = ShortTermMemory()
                    try:
                        strategy_store = StrategyStore()
                        _install_planner_convention_bindings(strategy_store, journal)
                    except Exception as exc:
                        strategy_store = None
                        log.warning(
                            "StrategyStore reload after rollback restore failed: %s",
                            exc,
                        )
                    log.info(
                        "Rollback restore complete; AP-22 and StrategyStore handles reloaded: %s",
                        restore_result,
                    )
                    gate.reset_failures()

        # ── 4b. Self-Criticism (AP-23/AP-24) ────────────────────
        phase.set("self_criticism", trial_id=trial_counter, species=species_name)
        # Get baseline and previous per-suite for comparison
        # Tier scales are intentionally incomparable. A missing T2/T3 baseline
        # means "no same-tier criticism reference", never "fall back to T1".
        baseline_q = (
            _strict_baseline_quality(gate.baseline, eval_result.tier) if gate.baseline else None
        )
        prev_suite = {}
        recent = journal.by_species(species_name)
        if recent:
            prev_details = recent[-1].eval_details
            if isinstance(prev_details, dict):
                prev_suite = prev_details.get("per_suite_quality", {})

        prior_criticism_text = last_criticism_text
        criticism = generate_self_criticism(
            action=action,
            eval_result=eval_result,
            verdict=verdict,
            failure_analysis=failure_analysis,
            baseline_quality=baseline_q or 0.0,
            prev_per_suite=prev_suite,
        )
        last_criticism_text = criticism.as_text()

        # ── 5. Record ────────────────────────────────────────────
        phase.set("record_trial", trial_id=trial_counter, species=species_name)
        # Classify whether this trial should be excluded from learning
        # surfaces (Pareto archive + AP-22 short-term memory). Non-benign
        # measurement contamination (unrecovered reloads, abandoned eval
        # requests) tags bug_corrupted_by so the planner's trustworthiness
        # gate excludes the trial and the Pareto frontier is not distorted.
        # Benign within-noise exclusions (mad_noise/reproduction/seq) suppress
        # learning without marking data corruption. See
        # classify_learning_exclusion() for priority order + reason strings.
        learning_excluded_by, learning_excluded_reason, exclusion_def_cat = (
            classify_learning_exclusion(verdict, eval_result)
        )
        multitier_tier_verdict = _record_multitier_validation_result(
            state,
            context=multitier_validation_context,
            eval_result=eval_result,
            verdict=verdict,
            trial_counter=trial_counter,
        )
        effective_action = _multitier_effective_action(action, multitier_validation_context)
        # Policy correction (2026-06-04 — see autopilot-continuous-optimization handoff):
        # a TRUSTED within-quality-noise measurement (mad_noise / reproduction_confirmed)
        # stays excluded from AP-22 / strategy learning, but must NOT be auto-excluded from
        # the MULTI-OBJECTIVE Pareto archive. The MAD test is quality-only; a trial flat on
        # quality can still be NON-DOMINATED on speed / cost / reliability and belongs on the
        # frontier. Admit ONE representative point per stable config fingerprint, dominance
        # tested on robust-MEDIAN objectives across the reproduction cluster (so a lucky
        # single-trial speed sample / host-throughput variance can't manufacture frontier
        # geometry). This also subsumes the old empty-frontier bootstrap: on an empty tier the
        # first trusted within-noise point becomes the seed via the same median path. AP-22
        # suppression remains tied to the mad_noise/reproduction_confirmed tag (criticism
        # below). Non-trusted exclusions (exogenous reload, etc.) still skip entirely.
        baseline_update = None
        # AP-55 (b)+(c): evaluated before any winner is named; binding only when the
        # operator arms AUTOPILOT_AP55_PROMOTION_GATE (enforce|strict).
        ap55_gate = _ap55_promotion_gate_for_trial(
            journal,
            tier=int(getattr(eval_result, "tier", 0) or 0),
            baseline_pin=baseline_pin_record,
            candidate_action=effective_action,
            question_results=list(getattr(eval_result, "question_results", []) or []),
            trial_fingerprint=trial_infra_fingerprint,
        )
        if isinstance(trial_comparability, dict):
            trial_comparability["promotion_gate"] = ap55_gate
        # W3 flip (2026-08-04): axis 1 of the live dominance vector is questions/hour, so a
        # trial that did not MEASURE a rate cannot be placed on the frontier — the pre-flip
        # `task_rate_qph_from` returned 0.0 for "unavailable" on 128 of 1466 journal rows,
        # and 0 qph is a real, maximally-bad throughput that would dominate the config out
        # on an axis it never scored. Absence is not zero.
        #
        # This gates the ARCHIVE WRITES ONLY. It deliberately does not wrap the whole
        # branch: `gate.update_baseline` sits inside it and promotes on QUALITY, an axis
        # that was measured perfectly well. An earlier version of this guard skipped the
        # branch wholesale and silently suppressed quality baseline promotion whenever the
        # rate was missing — coupling two independent axes.
        rate_measured = objectives_measurable(eval_result)
        # Decision (c), 2026-09-16: the row this trial journals is fixed NOW (same timestamp,
        # same objective-bearing fields) and handed to update_baseline, so the promotion
        # guard's live frontier includes the candidate. A clean trial is stamped as a
        # frontier representative and clusters with its config's reproductions.
        trial_journal_ts = datetime.now(timezone.utc).isoformat()
        trial_infra_digest = fingerprint_digest(trial_infra_fingerprint)
        trial_regime_digest = regime_digest(trial_infra_fingerprint)
        trial_served = _served_content_for_row(trial_served_content)
        # Re-review B1: only an action that names its served-config delta may join a
        # reproduction cluster; measurement actions (seed_batch, deep_eval, ...) stay raw
        # per-trial points exactly as before.
        clean_representative = (
            _clean_trial_representatives_enabled()
            and not learning_excluded_by
            and bool(verdict.passed)
            and rate_measured
            and eval_result.tier >= MIN_FRONTIER_EVAL_TIER
            and action_config_identity(effective_action, trial_regime_digest, trial_served)
            is not None
        )
        frontier_admission = FRONTIER_ADMISSION_REPRESENTATIVE if clean_representative else ""
        provisional_row = _provisional_trial_row(
            trial_id=trial_counter,
            timestamp=trial_journal_ts,
            eval_result=eval_result,
            action=effective_action,
            learning_excluded_by=learning_excluded_by,
            learning_excluded_reason=learning_excluded_reason,
            frontier_admission=frontier_admission,
            comparability=trial_comparability,
            infra_digest=trial_infra_digest,
            regime_digest=trial_regime_digest,
            served_content=trial_served,
        )
        # Re-review B2: the decision mutates gate.baseline in memory. Keep the pre-decision
        # baseline so a journal/decision mismatch can be rolled back before anything persists.
        baseline_before_decision = copy.deepcopy(gate.baseline)
        if not rate_measured:
            log.warning(
                "Trial %d: Pareto archive write SKIPPED — dominance axis unmeasured "
                "(no task rate: missing question ledger/eval_wall_s, or batch aborted "
                "below the s/question validity floor). NOT archived as 0 qph. Quality "
                "baseline promotion is unaffected and still evaluated below.",
                trial_counter,
            )
        if (
            learning_excluded_by in ("mad_noise", "reproduction_confirmed")
            and eval_result.tier >= MIN_FRONTIER_EVAL_TIER
        ):
            fingerprint = _config_fingerprint(effective_action)
            if rate_measured:
                pareto_status, rep_objs = archive.upsert_representative(
                    fingerprint,
                    eval_result.tier,
                    objectives_from(eval_result),
                    trial_id=trial_counter,
                    config_snapshot=effective_action,
                    species=species_name,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    memory_count=memory_count,
                    reasoning=json.dumps(effective_action),
                )
                log.info(
                    "Trial %d: Pareto representative admission %s (T%d fp=%s n=%d) "
                    "median_objs=%s — AP-22/strategy learning still excluded (%s)",
                    trial_counter,
                    pareto_status,
                    eval_result.tier,
                    fingerprint,
                    archive.reproduction_count(eval_result.tier, fingerprint),
                    [round(float(x), 3) for x in rep_objs],
                    learning_excluded_by,
                )
            else:
                pareto_status = "dominated"  # placeholder for JournalEntry only
            if (
                not MULTITIER_PROMOTION_ENABLED
                and rate_measured
                and bool(verdict)
                and ap55_gate.get("hold")
            ):
                log.info(
                    "Trial %d: within-noise baseline promotion HELD by AP-55 (%s, mode=%s)",
                    trial_counter,
                    ",".join(ap55_gate.get("hold_reasons") or []),
                    ap55_gate.get("mode"),
                )
            elif not MULTITIER_PROMOTION_ENABLED and rate_measured and bool(verdict):
                # AP-55 (b)+(c) hold is honoured by the branch above (train integration
                # 2026-09-16): without it an enforced hold is bypassed on this path.
                # Decision (c), 2026-09-16: a within-noise reproduction is exactly the
                # evidence a promotion requires (a >=3-member representative whose median
                # clears the baseline by a quantum). Without this call only a CLEAN trial
                # reached update_baseline, but the 2nd+ run of an above-baseline config is
                # labelled reproduction_confirmed, so its cluster could never promote.
                # (Multitier mode stages the same trial below and re-checks at final_t1.)
                seq_confirmed = (
                    bool(verdict.seq.get("baseline_promotion_finalized"))
                    if verdict.seq is not None
                    else None
                )
                baseline_update = gate.update_baseline(
                    eval_result,
                    source_trial_id=trial_counter,
                    seq_confirmed=seq_confirmed,
                    pending_journal_rows=(provisional_row,),
                )
                _log_baseline_update_result(trial_counter, baseline_update)
            criticism = learning_exclusion_criticism(learning_excluded_by, learning_excluded_reason)
        elif learning_excluded_by:
            pareto_status = "dominated"  # placeholder for JournalEntry only
            log.info(
                "Trial %d: archive.update SKIPPED (learning_excluded_by=%s)",
                trial_counter,
                learning_excluded_by,
            )
            criticism = learning_exclusion_criticism(learning_excluded_by, learning_excluded_reason)
        elif not verdict.passed:
            # Safety verdict FAILED and this is not a benign within-noise exclusion
            # (e.g. a genuine per-suite regression at adequate n, possibly co-tagged
            # mad_noise on the quality axis — see classify_learning_exclusion). A
            # failed trial must NEVER clean-update the Pareto archive or raise the
            # baseline. It is a real failed experiment, NOT corrupted data, so we
            # leave bug_corrupted_by unset; the deficiency category is taken from
            # verdict.categories below. (2026-06-06: previously such a trial fell
            # through to the clean-update branch and could admit/raise on a failure.)
            pareto_status = "dominated"  # placeholder for JournalEntry only
            log.info(
                "Trial %d: archive.update SKIPPED (safety verdict failed: %s)",
                trial_counter,
                ", ".join(verdict.categories) or "unspecified",
            )
        else:
            if rate_measured and clean_representative:
                # Decision (c): same median-representative admission as the replay applies
                # to the stamped row, so the live archive and journal authority agree.
                pareto_status, _rep_objs = archive.upsert_representative(
                    _config_fingerprint(effective_action),
                    eval_result.tier,
                    objectives_from(eval_result),
                    trial_id=trial_counter,
                    config_snapshot=effective_action,
                    species=species_name,
                    timestamp=trial_journal_ts,
                    memory_count=memory_count,
                    reasoning=json.dumps(effective_action),
                )
            elif rate_measured:
                pareto_status = archive.update(
                    ParetoEntry(
                        trial_id=trial_counter,
                        objectives=objectives_from(eval_result),
                        config_snapshot=effective_action,
                        species=species_name,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        eval_tier=eval_result.tier,
                        memory_count=memory_count,
                        reasoning=json.dumps(effective_action),
                    )
                )
            else:
                pareto_status = "dominated"  # placeholder for JournalEntry only
            allow_multitier_final = (
                MULTITIER_PROMOTION_ENABLED
                and isinstance(multitier_validation_context, dict)
                and multitier_validation_context.get("stage") == "final_t1"
            )
            if MULTITIER_PROMOTION_ENABLED and not allow_multitier_final:
                eligible, eligibility_reason = _multitier_candidate_is_eligible(
                    state=state,
                    gate=gate,
                    action=effective_action,
                    eval_result=eval_result,
                    verdict=verdict,
                    pareto_status=pareto_status,
                    served_content=trial_served_content,
                )
                if eligible:
                    pending = _start_multitier_validation(
                        state,
                        action=effective_action,
                        eval_result=eval_result,
                        verdict=verdict,
                        trial_counter=trial_counter,
                        served_content=trial_served_content,
                    )
                    log.info(
                        "Trial %d: T1 baseline promotion HELD for binding T2/T3 "
                        "validation (candidate=%s, next=T%s)",
                        trial_counter,
                        pending["candidate"],
                        pending["next_tier"],
                    )
                else:
                    log.info(
                        "Trial %d: baseline promotion HELD by %s (%s)",
                        trial_counter,
                        MULTITIER_POLICY_VERSION,
                        eligibility_reason,
                    )
            elif ap55_gate.get("hold"):
                log.info(
                    "Trial %d: baseline promotion HELD by AP-55 (%s, mode=%s)",
                    trial_counter,
                    ",".join(ap55_gate.get("hold_reasons") or []),
                    ap55_gate.get("mode"),
                )
            else:
                # Runs whether or not the RATE was measured: this promotes on QUALITY.
                seq_confirmed = (
                    bool(verdict.seq.get("baseline_promotion_finalized"))
                    if verdict.seq is not None
                    else None
                )
                baseline_update = gate.update_baseline(
                    eval_result,
                    source_trial_id=trial_counter,
                    seq_confirmed=seq_confirmed,
                    pending_journal_rows=(provisional_row,),
                )
                # B4 / SEQ-2: surface the update/refusal outcome — including a distinct
                # line for the gate's seq_inputs_unavailable refusal (seq_confirmed=None).
                _log_baseline_update_result(trial_counter, baseline_update)

        # A repeated final T1 draw commonly lands in the benign MAD/reproduction
        # branch above. It still needs to exercise the existing baseline gate;
        # otherwise the staged policy could validate T2/T3 correctly and then
        # reject forever merely because the final confirmation reproduced.
        if (
            MULTITIER_PROMOTION_ENABLED
            and isinstance(multitier_validation_context, dict)
            and multitier_validation_context.get("stage") == "final_t1"
            and baseline_update is None
            and bool(verdict)
            and not ap55_gate.get("hold")
        ):
            seq_confirmed = (
                bool(verdict.seq.get("baseline_promotion_finalized"))
                if verdict.seq is not None
                else None
            )
            baseline_update = gate.update_baseline(
                eval_result,
                source_trial_id=trial_counter,
                seq_confirmed=seq_confirmed,
                pending_journal_rows=(provisional_row,),
            )
            _log_baseline_update_result(trial_counter, baseline_update)

        # The same benign branch can be the first robust representative of a
        # T1 candidate. Stage it exactly as a clean frontier point; never let a
        # within-noise label bypass binding higher-tier validation.
        if (
            MULTITIER_PROMOTION_ENABLED
            and multitier_validation_context is None
            and not isinstance(state.get(MULTITIER_PENDING_STATE_KEY), dict)
            and bool(verdict)
        ):
            eligible, _eligibility_reason = _multitier_candidate_is_eligible(
                state=state,
                gate=gate,
                action=effective_action,
                eval_result=eval_result,
                verdict=verdict,
                pareto_status=pareto_status,
                served_content=trial_served_content,
            )
            if eligible:
                _start_multitier_validation(
                    state,
                    action=effective_action,
                    eval_result=eval_result,
                    verdict=verdict,
                    trial_counter=trial_counter,
                    served_content=trial_served_content,
                )

        if (
            MULTITIER_PROMOTION_ENABLED
            and isinstance(multitier_validation_context, dict)
            and multitier_validation_context.get("stage") == "final_t1"
        ):
            _finish_multitier_promotion(
                state,
                baseline_update=baseline_update,
                trial_counter=trial_counter,
            )

        _update_seq_promotion_fresh_eval_state(
            state,
            seq=getattr(verdict, "seq", None),
            action=action,
            eval_result=eval_result,
            trial_counter=trial_counter,
            is_fresh_eval=seq_fresh_eval_context is not None,
            finalized=seq_finalized if baseline_update is not None else False,
            baseline_update=baseline_update,
            seq_alpha_wealth=seq_inputs.get("alpha_wealth") if not has_exo_unrecovered else None,
        )

        # Extract hypothesis and expected mechanism from action/controller
        hypothesis = action.get("description", "")
        # AP-15: Fallback for species that don't provide description
        if not hypothesis:
            action_type = action.get("type", "")
            if action_type == "seed_batch":
                hypothesis = f"Seed {action.get('n_questions', 10)} questions across {action.get('suites', 'all')}"
            elif action_type == "numeric_trial":
                hypothesis = f"Optimize {action.get('surface', 'unknown')} surface"
            elif action_type == "structural_experiment":
                hypothesis = f"Toggle flags: {action.get('flags', {})}"
            elif action_type == "consult_gate_probe":
                hypothesis = "Probe targeted review-before-commit consult gate"
            elif action_type in ("train_routing_models", "distill_skillbank", "rollback"):
                hypothesis = action_type.replace("_", " ").title()
        expected_mechanism = (
            action.get("mutation", "") or action.get("surface", "") or action.get("type", "")
        )

        # B3: Capture execution traces for next PromptForge iteration
        recent_trace_text = state.get("last_traces", "")
        if not dry_run:
            recent_trace_text = tower.capture_recent_traces(50)
            state["last_traces"] = recent_trace_text

        # J9/HLE-4: observe-only harness metrics. Compute after SafetyGate and
        # ParetoArchive decisions so these diagnostics cannot affect current
        # trial acceptance or baseline mutation.
        if not getattr(eval_result, "harness_metrics", None):
            from hle_metrics import compute_hle_observe_payload  # type: ignore

            hle_payload = compute_hle_observe_payload(
                eval_result,
                action=action,
                verdict=verdict,
                failure_analysis=failure_analysis,
                prior_criticism=prior_criticism_text,
                recent_traces=recent_trace_text,
            )
            eval_result.metric_schema_version = hle_payload["metric_schema_version"]
            eval_result.harness_metrics = hle_payload["harness_metrics"]
            eval_result.oracle_adequacy = hle_payload["oracle_adequacy"]

        # Compute trial lineage (AP-3) before BSV observe so BSV-3 dependency rows
        # can carry the same parent-trial identity as the journal entry.
        parent_trial_id = None
        config_diff: dict[str, Any] = {}
        # AP-63(a): explicit, rule-named parent selection, stored on the row. Computed
        # before this trial is journaled, so a trial is never its own parent. Fails open
        # to an error record: lineage is provenance and must never lose a trial.
        try:
            trial_lineage = select_lineage_parent(
                journal, species_name, before_trial_id=trial_counter
            )
        except Exception as exc:  # noqa: BLE001
            trial_lineage = {"selection_error": f"{type(exc).__name__}: {exc}"[:200]}
        species_history = journal.by_species(species_name)
        if species_history:
            parent = species_history[-1]
            parent_trial_id = parent.trial_id
            # Compute config diff: keys that changed between parent and current
            prev_cfg = parent.config_snapshot
            for key in set(list(prev_cfg.keys()) + list(effective_action.keys())):
                old_val = prev_cfg.get(key)
                new_val = effective_action.get(key)
                if old_val != new_val:
                    config_diff[key] = {"old": old_val, "new": new_val}

        # J11/BSV-2/BSV-3: observe-only behavior-signature differential vs the last
        # frontier-accepted incumbent. Flag-gated (AUTOPILOT_BSV_OBSERVE=1), default-OFF.
        # Same observe-only zone as HLE (after SafetyGate + ParetoArchive) so it CANNOT affect
        # trial acceptance, Pareto promotion, routing, blacklists, or baseline mutation. Optional
        # BSV-3 conflict policy only governs BSV ledger/incumbent diagnostic-state promotion.
        bsv_payload: dict = {}
        if os.environ.get("AUTOPILOT_BSV_OBSERVE") == "1":
            try:
                from bsv_observe import (  # type: ignore
                    build_conflict_report,
                    build_mutation_dependency_entry,
                    compute_bsv_observe_payload,
                )

                incumbent_signature = state.get("bsv_incumbent_signature")
                bsv_payload = compute_bsv_observe_payload(
                    eval_result,
                    species_name=species_name,
                    trial_id=trial_counter,
                    archive_member_id=f"trial:{trial_counter}",
                    incumbent_signature=incumbent_signature,
                    incumbent_archive_member_id=state.get("bsv_incumbent_archive_member_id"),
                )
                # Incumbent = last frontier entry that ALSO PASSED the SafetyGate (verdict truthy =
                # SafetyVerdict.passed). archive.update runs even on gate-FAILED trials, so frontier
                # alone could promote a failed trial's signature as the incumbent (finding #1).
                if pareto_status == "frontier" and verdict:
                    archive_member_id = bsv_payload.get("archive_member_id")
                    signature = bsv_payload.get("signature")
                    dependency_entry = build_mutation_dependency_entry(
                        trial_id=trial_counter,
                        action=action,
                        parent_trial=parent_trial_id,
                        bsv_payload=bsv_payload,
                        incumbent_signature=incumbent_signature,
                        pareto_status=pareto_status,
                    )
                    existing_ledger = state.get("bsv_mutation_dependency_ledger", [])
                    if not isinstance(existing_ledger, list):
                        existing_ledger = []
                    conflict_report = build_conflict_report(dependency_entry, existing_ledger)
                    bsv_payload["mutation_dependency"] = dependency_entry
                    bsv_payload["conflict_report"] = conflict_report
                    conflict_policy = _bsv3_conflict_policy_decision(conflict_report)
                    if conflict_policy["enabled"]:
                        bsv_payload["conflict_policy"] = conflict_policy
                    if conflict_report.get("severity") in {"watch", "blocking"}:
                        log.warning(
                            "BSV-3 conflict review signal for trial %d: severity=%s conflicts=%s",
                            trial_counter,
                            conflict_report.get("severity"),
                            conflict_report.get("conflict_count"),
                        )
                    if conflict_policy["ledger_update_allowed"]:
                        state["bsv_mutation_dependency_ledger"] = [
                            *existing_ledger[-499:],
                            dependency_entry,
                        ]
                        state["bsv_incumbent_signature"] = signature
                        state["bsv_incumbent_archive_member_id"] = archive_member_id
                        if archive_member_id and signature:
                            archive_signatures = state.setdefault("bsv_archive_signatures", {})
                            archive_signatures[archive_member_id] = {
                                "trial_id": trial_counter,
                                "signature_hash": bsv_payload.get("signature_hash"),
                                "signature_confidence": bsv_payload.get("signature_confidence"),
                                "severity_vs_previous_incumbent": bsv_payload.get("severity"),
                                "reasons": list(bsv_payload.get("reasons") or [])[:8],
                                "conflict_severity": (bsv_payload.get("conflict_report") or {}).get(
                                    "severity"
                                ),
                                "conflict_count": (bsv_payload.get("conflict_report") or {}).get(
                                    "conflict_count"
                                ),
                                "signature": signature,
                            }
                    else:
                        log.warning(
                            "BSV-3 conflict policy withheld ledger/incumbent update "
                            "for trial %d: policy=%s severity=%s conflicts=%s",
                            trial_counter,
                            conflict_policy["policy"],
                            conflict_policy["severity"],
                            conflict_policy["conflict_count"],
                        )
            except Exception as _bsv_err:  # observe-only must never disrupt the trial loop
                log.debug("BSV observe skipped (trial %s): %s", trial_counter, _bsv_err)

        # W8 paired evidence observability: compare this trial's same-qid
        # outcomes with the latest marked baseline-reference draw. This is
        # journal-only and deliberately computed after gate/archive/baseline
        # decisions so it cannot affect current keep/revert or promotion.
        seq_paired_baseline_payload: dict[str, Any] = {}
        if seq_inputs is not None:
            try:
                seq_paired_baseline_payload = _seq_paired_baseline_diagnostics(
                    journal=journal,
                    tier=eval_result.tier,
                    candidate=str(seq_inputs.get("candidate") or ""),
                    candidate_trial_id=trial_counter,
                    question_results=list(getattr(eval_result, "question_results", []) or []),
                    candidate_fingerprint=trial_infra_fingerprint,
                )
            except Exception as _paired_err:  # observe-only must never disrupt the loop
                log.debug(
                    "Seq paired-baseline diagnostics skipped (trial %s): %s",
                    trial_counter,
                    _paired_err,
                )

        # Git tag
        phase.set("post_trial_artifacts", trial_id=trial_counter, species=species_name)
        git_tag = ""
        if not dry_run:
            git_tag = f"autopilot/trial-{trial_counter}"
            _git_tag(git_tag, f"Trial {trial_counter}: {species_name}/{action.get('type', '')}")

        # Build active_flags from action context
        active_flags_dict = action.get("flags", {})
        active_flags_list = (
            [f"{k}={v}" for k, v in active_flags_dict.items()] if active_flags_dict else []
        )

        # AP-14: Extract deficiency category from safety verdict + dispatch side channel
        deficiency_category = ""
        if not verdict.passed:
            deficiency_category = verdict.categories[0] if verdict.categories else ""
        if not deficiency_category:
            deficiency_category = state.pop("_dispatch_deficiency", "")

        # Apply the learning-exclusion decision computed above. True measurement
        # contamination (exogenous reload, prompt leak, kill/reload artifacts)
        # produces bug_corrupted_by + eval_details["learning_exclusion"]. Valid
        # negative evidence such as seq_refuted remains a learning exclusion but
        # must not be hidden as corrupted data.
        # Benign convergence exclusions (reproduction_confirmed) skip the Pareto
        # archive (via learning_excluded_by above) but must NOT populate
        # bug_corrupted_by — otherwise trustworthiness_score() and the journal
        # trust render would treat a valid confirmation like a kill / reload /
        # commit-invalidation, and the planner would narrate a "noisy instrument"
        # (2026-05-31 incident → meta-action loop).
        bug_corrupted_by, bug_corrupted_reason = _bug_tag_for_learning_exclusion(
            learning_excluded_by, learning_excluded_reason
        )
        _update_contrastive_trace_state(
            state,
            tower,
            trace_text=recent_trace_text,
            trial_id=trial_counter,
            species=species_name,
            action_type=action.get("type", ""),
            pareto_status=pareto_status,
            verdict=verdict,
            bug_corrupted_by=bug_corrupted_by,
            failure_analysis=failure_analysis,
            eval_result=eval_result,
        )
        metric_schema_version = getattr(eval_result, "metric_schema_version", 1)
        harness_metrics = getattr(eval_result, "harness_metrics", {}) or {}
        oracle_adequacy = getattr(eval_result, "oracle_adequacy", {}) or {}
        # Build each series with its OWN explicit builder. `objectives_from` now returns
        # the live tasks/hour vector, which has the same 4D SHAPE as the legacy
        # tokens/second one — calling it here would have relabelled the rate vector as
        # "legacy" with nothing to catch it, and would additionally RAISE on a trial with
        # no measured rate (this line runs for every trial, including archive-skipped ones).
        # RTG-23: the legacy and shadow builders also refuse to substitute 0.0 for an
        # axis they never measured, so they can raise / return None here too. This runs
        # for EVERY trial, including archive-skipped ones, so record the shadow series
        # as null rather than failing the journal write.
        try:
            legacy_objectives: list[float] | None = list(legacy_objectives_from(eval_result))
        except UnmeasuredObjectiveError:
            legacy_objectives = None
        _shadow_rate_objectives = task_rate_objectives_from(eval_result)
        task_rate_objectives = (
            list(_shadow_rate_objectives) if _shadow_rate_objectives is not None else None
        )
        try:
            live_objectives: list[float] | None = list(objectives_from(eval_result))
        except UnmeasuredObjectiveError:
            live_objectives = None
        eval_details_dict: dict[str, Any] = {
            "per_suite_quality": eval_result.per_suite_quality,
            "routing_distribution": eval_result.routing_distribution,
            "question_results": list(getattr(eval_result, "question_results", []) or []),
            "details": eval_result.details,
            "rlvr_reward": rlvr_reward_from_result(eval_result).as_dict(),
            "metric_schema_version": metric_schema_version,
            "harness_metrics": harness_metrics,
            "oracle_adequacy": oracle_adequacy,
            "bsv_observe": bsv_payload,  # J11/BSV-2 observe-only diff ({} unless AUTOPILOT_BSV_OBSERVE=1)
            # W3 flip 2026-08-04 (operator): tasks/hour is the LIVE dominance vector.
            # Legacy tokens/second is retained as shadow telemetry so the pre-flip series
            # stays readable; it no longer decides keep/revert.
            "objective_policy_live": RATE_4D_OBJECTIVE_POLICY,
            "objective_policy_shadow": LEGACY_OBJECTIVE_POLICY,
            "objectives_legacy_v1": legacy_objectives,
            "objectives_task_rate_v1": task_rate_objectives,
            # The vector dominance actually used, or None when the rate was unmeasured
            # (such a trial is skipped by the archive rather than entered as 0 qph).
            "objectives_live_v1": live_objectives,
            "task_rate_qph": task_rate_qph_from(eval_result),
            "goodput_qph": goodput_qph_from(eval_result),
            "speed_metric_mode": getattr(eval_result, "speed_metric_mode", "median_request_tps"),
            "median_request_speed": getattr(eval_result, "median_request_speed", 0.0),
            "aggregate_speed": getattr(eval_result, "aggregate_speed", 0.0),
            "eval_concurrency": getattr(eval_result, "eval_concurrency", 1),
            "eval_wall_s": getattr(eval_result, "eval_wall_s", 0.0),
            "sum_request_elapsed_s": getattr(eval_result, "sum_request_elapsed_s", 0.0),
            "ece": eval_result.ece,
            "auroc": eval_result.auroc,
            "calibration_violations": eval_result.calibration_violations,
            "gepa_ratio": state.get("gepa_ratio", 0.30),
            # Tool-use telemetry (2026-06-01) — surfaced at top level so the planner
            # can measure and incentivize productive tool use. tokens_generated/speed
            # already credit tool-turn generation; these are the explicit signal.
            "mean_tools_used": getattr(eval_result, "mean_tools_used", 0.0),
            "tool_use_rate": getattr(eval_result, "tool_use_rate", 0.0),
            "total_tool_calls": getattr(eval_result, "total_tool_calls", 0),
            # The decision-grade signal: marginal usefulness of tools (NaN until
            # enough sample). Steer by THIS, not raw tool_use_rate. Now computed
            # per-suite then averaged, so trivially-correct no-tool suites can't
            # contaminate the sign (was −0.4 cross-suite at cutover for all-passing
            # tool calls). Per-suite breakdown below for the suite under test.
            "tool_helpfulness": getattr(eval_result, "tool_helpfulness", float("nan")),
            "per_suite_tool_helpfulness": getattr(eval_result, "per_suite_tool_helpfulness", {}),
        }
        # D6 / FIELD-1 journal leg: persist the documented diversity_* / rubric_* /
        # reviewer_* / branching_density / instruction_token_* / avg_prompt_tokens /
        # compaction_events families that were previously dropped from the journal
        # payload (they only reached the METRIC grep-lines). Null-gated; see
        # _eval_details_from_result. These feed H-LB (AP-4 axes comment on EvalResult).
        eval_details_dict.update(_eval_details_from_result(eval_result))
        if seq_paired_baseline_payload:
            eval_details_dict["seq_paired_baseline"] = seq_paired_baseline_payload
        if learning_excluded_by:
            deficiency_category = exclusion_def_cat
            eval_details_dict["learning_exclusion"] = {
                "by": learning_excluded_by,
                "reason": learning_excluded_reason,
            }
        if has_exo_unrecovered or has_exo_recovered:
            # Surface audit info regardless — recovered trials carry this
            # so the operator can see "this trial weathered a reload"
            # without the planner downgrading it.
            eval_details_dict["exogenous_retries"] = {
                "n_recovered": getattr(eval_result, "n_exogenous_recovered", 0),
                "n_unrecovered": getattr(eval_result, "n_exogenous_unrecovered", 0),
                "n_external_restart": getattr(eval_result, "n_external_restart", 0),
                "question_ids": list(getattr(eval_result, "exogenous_question_ids", [])),
                "marker_observations": list(getattr(eval_result, "exogenous_marker_log", [])),
            }
        if seq_baseline_draw_reference is not None:
            eval_details_dict["seq_baseline_reference_draw"] = True
            eval_details_dict["seq_baseline_reference_reason"] = seq_baseline_draw_reference.get(
                "reason", ""
            )
        if seq_fresh_eval_context is not None:
            eval_details_dict["seq_promotion_fresh_eval"] = {
                "candidate": seq_fresh_eval_context.get("candidate"),
                "source_trial_id": seq_fresh_eval_context.get("source_trial_id"),
            }
        if seq_candidate_replay_context is not None:
            eval_details_dict["seq_candidate_replay"] = {
                "candidate": seq_candidate_replay_context.get("candidate"),
                "source_trial_id": seq_candidate_replay_context.get("source_trial_id"),
                "k": seq_candidate_replay_context.get("k"),
                "combined_E": seq_candidate_replay_context.get("combined_E"),
            }
        if multitier_validation_context is not None:
            eval_details_dict["multitier_validation"] = {
                "policy_version": MULTITIER_POLICY_VERSION,
                "stage": multitier_validation_context.get("stage"),
                "tier": multitier_validation_context.get("tier"),
                "candidate": multitier_validation_context.get("candidate"),
                "source_trial_id": multitier_validation_context.get("source_trial_id"),
                "verdict": multitier_tier_verdict,
            }

        # AP-55: eval-side consumers read eval_details, so stamp the regime there too.
        eval_details_dict["infra_fingerprint_digest"] = fingerprint_digest(trial_infra_fingerprint)
        eval_details_dict["infra_comparability"] = trial_comparability.get("status", "")
        eval_details_dict[INFRA_REGIME_DIGEST_KEY] = trial_regime_digest
        if trial_served:
            eval_details_dict[SERVED_CONTENT_KEY] = trial_served
        if frontier_admission:
            eval_details_dict[FRONTIER_ADMISSION_KEY] = frontier_admission
        if baseline_update is not None:
            # Which promotion rule the archive stage applied (frontier / empty_frontier_repro /
            # seed); also carried on the baseline_promotion ledger event.
            eval_details_dict["promotion_rule"] = getattr(baseline_update, "promotion_rule", "")
            # Re-review B3: the row is written BEFORE the promotion commits. "pending_commit"
            # is confirmed only by a baseline_promotion ledger event with this source_trial_id
            # (the commit record, appended with the final state save). A crash in between
            # leaves the row pending and the baseline unchanged, which is consistent: readers
            # must treat an unconfirmed pending row as NOT promoted; recovery needs no action.
            eval_details_dict["promotion_status"] = (
                "pending_commit" if getattr(baseline_update, "updated", False) else "refused"
            )
        # AP-54: carry the trial's eval-fence state beside the infra regime, so a
        # fenced and an unfenced run are never read as the same regime.
        _fence_summary = (
            eval_result.details.get("eval_fence")
            if isinstance(getattr(eval_result, "details", None), dict)
            else None
        )
        eval_details_dict["eval_fence_state"] = (
            str(_fence_summary.get("state") or "absent")
            if isinstance(_fence_summary, dict)
            else "absent"
        )
        eval_details_dict["eval_fence_enforcement"] = (
            str(_fence_summary.get("fence_enforcement") or "")
            if isinstance(_fence_summary, dict)
            else ""
        )
        eval_details_dict["ap55_promotion_gate"] = ap55_promotion_gate.gate_summary(
            trial_comparability.get("promotion_gate")
        )
        journal_entry = JournalEntry(
            trial_id=trial_counter,
            timestamp=trial_journal_ts,
            species=species_name,
            action_type=action.get("type", ""),
            tier=eval_result.tier,
            quality=eval_result.quality,
            speed=eval_result.speed,
            cost=eval_result.cost,
            reliability=eval_result.reliability,
            pareto_status=pareto_status,
            git_tag=git_tag,
            config_snapshot=effective_action,
            config_diff=config_diff,
            parent_trial=parent_trial_id,
            memory_count=memory_count,
            active_flags=active_flags_list,
            failure_analysis=failure_analysis,
            eval_details=eval_details_dict,
            metric_schema_version=metric_schema_version,
            harness_metrics=harness_metrics,
            oracle_adequacy=oracle_adequacy,
            seq=verdict.seq or {},
            reasoning=json.dumps(effective_action),
            hypothesis=hypothesis,
            expected_mechanism=expected_mechanism,
            deficiency_category=deficiency_category,
            instruction_token_count=eval_result.instruction_token_count,
            instruction_token_ratio=eval_result.instruction_token_ratio,
            self_criticism=criticism.as_text(),  # AP-23
            keep_revert_decision=criticism.keep_or_revert,  # AP-24
            optimization_directions=criticism.directions_text(),  # AP-24
            predicted_objectives=predicted_objectives,  # PEAF (intake-571 spike)
            surprise_score=peaf.compute_surprise(
                predicted_objectives, peaf.actual_objectives_from_eval(eval_result)
            ),
            falsifier=rationale.get("falsifier", ""),
            rubric_scores=rationale.get("rubric_scores", {}),
            stagnation_signal=stagnation_signal,
            bug_corrupted_by=bug_corrupted_by,
            bug_corrupted_reason=bug_corrupted_reason,
            baseline_pin=baseline_pin_record,  # EV-14e
            infra_fingerprint=trial_infra_fingerprint,  # AP-55
            comparability=trial_comparability,  # AP-55
            run_manifest=_in_flight_run_manifest(state, trial_counter),  # AP-63(a)
            lineage=trial_lineage,  # AP-63(a)
        )
        _sync_segment_snapshot_scope(journal, state)  # W3
        journal.record(journal_entry)
        baseline_update = _reconcile_promotion_with_journal(
            gate,
            baseline_update,
            baseline_before_decision,
            provisional_row,
            asdict(journal_entry),
            trial_counter,
        )
        _record_reproposal_rate_windows(journal)  # VB-AP53-RATE
        # Evidence is the already-durable trial, supplied here rather than by the
        # planner, so a resolution can never cite a trial that did not run.
        _record_operator_hypothesis_resolution(rationale, trial_counter)
        if strategy_store is not None:
            try:
                strategy_store.store_frontier_journal_entry(journal_entry)
                if hasattr(strategy_store, "store_consult_gate_journal_entry"):
                    strategy_store.store_consult_gate_journal_entry(journal_entry)
            except Exception as e:
                log.warning("Strategy store journal projection failed: %s", e)

        # AP-16: Track last instruction ratio for structural pruning comparison
        state["_last_instruction_ratio"] = eval_result.instruction_token_ratio

        # W5: rebuild AP-22 prompt memory from the folded append-only journal view
        # instead of mutating short_term_memory.md from in-memory trial state. The
        # renderer applies the same trust filters for corrupted, non-ok, and
        # learning-excluded rows, so excluded trials stay auditable in the ledger
        # without feeding planner memory.
        memory.refresh_from_journal(journal)

        # ── 6. Meta-learn ───────────────────────────────────────
        if meta.should_rebalance(trial_counter):
            meta.rebalance(
                species_effectiveness=journal.species_effectiveness(window=50),
                hv_slope=hv_slope,
                memory_count=memory_count,
                is_converged=converged,
            )
            state["species_budget"] = meta.budget.as_dict()

        # Context budget management: auto-checkpoint at intervals
        if trial_counter > 0 and trial_counter % 25 == 0 and not dry_run:
            log.info("Auto-checkpoint at trial %d", trial_counter)
            phase.set("checkpoint", trial_id=trial_counter)
            lab.checkpoint_state(
                trial_id=trial_counter,
                notes=f"Auto-checkpoint at trial {trial_counter}",
            )

        # Persist seeder convergence state on every metric-bearing trial so a
        # restart does not reset train_routing_models eligibility.
        state["seeder_state"] = seeder.export_state()
        state["td_errors"] = state["seeder_state"]["td_errors"]

        # Generate plots on a trial-count OR wall-clock cadence, so a long
        # mid-decade gap still refreshes instead of waiting for the next %N
        # boundary (which may never arrive if the run stops first).
        if (
            trial_counter % PLOT_INTERVAL == 0
            or (time.time() - plot_clock["last_ts"]) >= PLOT_MAX_AGE_S
        ):
            _refresh_plots(sync=False, reason="periodic")
            plot_paths = [
                str(PLOTS_DIR / name)
                for name in (
                    "hypervolume_trend.png",
                    "pareto_frontier_2d.png",
                    "species_effectiveness.png",
                    "per_suite_quality.png",
                    "memory_convergence.png",
                    "trial_timeline.png",
                )
            ]
            phase.set(
                "async_plots_scheduled",
                trial_id=trial_counter,
                action_type=action.get("type", ""),
            )

        # Save state
        phase.set("save_state", trial_id=trial_counter, species=species_name)
        trial_counter += 1
        state["trial_counter"] = trial_counter
        state["consecutive_meta_actions"] = 0  # a real metric-collecting trial ran
        # A trial executed and produced metrics — clear the non-executing-action
        # residue so the next prompt's "Last Non-Executing Action" feedback only
        # reflects an UNRESOLVED skip (the per-signature counts persist for the
        # blacklist threshold across the whole run).
        state["consecutive_skip_actions"] = 0
        state["last_invalid_action"] = None
        state["last_invalid_reason"] = ""
        state["last_invalid_status"] = ""
        state["consecutive_failures"] = gate.consecutive_failures
        state["quality_history"] = gate.quality_history
        state["quality_history_by_tier"] = gate.quality_history_by_tier
        # Defect #4: persist the provenance-bearing window (era/ts/core_id per sample) as the
        # authoritative shape; the float mirrors above stay for any external float reader.
        state["quality_history_provenance_by_tier"] = gate.quality_history_provenance_by_tier
        baseline_state = gate.baseline.to_state_dict()
        state["baseline_state"] = baseline_state
        _record_baseline_infra_fingerprint(state, baseline_update, trial_infra_fingerprint)
        _journal_baseline_ledger_event(
            state,
            journal=journal,
            baseline_update=baseline_update,
            eval_result=eval_result,
            source_trial_id=trial_counter - 1,
            pareto_status=pareto_status,
            baseline_state=baseline_state,
            infra_fingerprint=trial_infra_fingerprint,
        )
        # H4: the out-of-band control merge now happens UNDER the write lock
        # inside save_state (merge_control=True), so an operator/dashboard/
        # host_health pause set while this trial ran survives this whole-file save.
        _save_state_with_journal_archive_authority(
            state,
            journal,
            archive,
            context=f"trial {trial_counter} final save",
            merge_control=True,
        )

        # Phase 6b — clear in_flight_trial marker AFTER final save_state.
        # This is the closing half of the WAL pattern: a crash between
        # the pre-dispatch marker write and this clear leaves the marker
        # in place, which triggers the recovery branch on the next
        # startup. By the time we reach here both the journal and the
        # Pareto archive are durable on disk, so it is safe to clear.
        state["in_flight_trial"] = None
        _save_state_with_journal_archive_authority(
            state,
            journal,
            archive,
            context=f"trial {trial_counter} in-flight clear",
            merge_control=True,
        )

        # A newly accepted staged candidate becomes the next rollback anchor.
        # Checkpoint only after the WAL marker is cleared so restoring the
        # production-best snapshot can never resurrect a completed in-flight
        # trial.  Runtime numeric/flag values are carried separately by each
        # candidate's exact preimage; this checkpoint owns routing intelligence.
        checkpoint_due = state.get("multitier_production_checkpoint_due")
        if isinstance(checkpoint_due, dict):
            try:
                checkpoint_path = lab.checkpoint_state(
                    trial_id=int(checkpoint_due.get("trial_id") or trial_counter - 1),
                    hypervolume=float(archive.hypervolume()),
                    feature_flags=lab.current_flags(),
                    config_snapshot=dict(checkpoint_due.get("candidate_action") or {}),
                    notes=f"accepted by {MULTITIER_POLICY_VERSION}",
                    mark_production_best=True,
                )
                state["multitier_last_event"] = {
                    "event": "production_checkpoint_advanced",
                    "trial_id": checkpoint_due.get("trial_id"),
                    "candidate": checkpoint_due.get("candidate"),
                    "checkpoint": str(checkpoint_path),
                }
                state.pop("multitier_production_checkpoint_due", None)
            except Exception as exc:  # noqa: BLE001 - promotion must fail closed
                state["paused"] = True
                state["pause_reason"] = (
                    "accepted multitier candidate could not advance production checkpoint: "
                    f"{type(exc).__name__}: {exc}"
                )
                state["multitier_checkpoint_failure"] = {
                    **checkpoint_due,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                log.exception("Multitier production checkpoint advancement failed")
            _save_state_with_journal_archive_authority(
                state,
                journal,
                archive,
                context=f"trial {trial_counter} multitier checkpoint",
                merge_control=True,
            )

        log.info(
            "Trial %d complete: q=%.3f s=%.1f → %s (HV=%.4f)",
            trial_counter - 1,
            eval_result.quality,
            eval_result.speed,
            pareto_status,
            archive.hypervolume(),
        )

        # Daily digest: once per UTC day, append a markdown snapshot to
        # progress/YYYY-MM/YYYY-MM-DD-autopilot.md. Append-only and never
        # touches existing handoffs — review remains a manual step.
        if should_generate_today(state):
            try:
                state["last_digest_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                save_state(state)
                async_tasks.submit_subprocess(
                    f"digest-{state['last_digest_date']}",
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "autopilot.py"),
                        "digest",
                        "--no-state-update",
                    ],
                    cwd=ORCH_ROOT,
                )
                phase.set(
                    "async_digest_scheduled",
                    trial_id=trial_counter - 1,
                    digest_date=state["last_digest_date"],
                )
                log.info("Daily digest scheduled asynchronously")
            except Exception as e:
                # Digest failure must NEVER block the optimization loop.
                log.warning("Daily digest generation failed: %s", e)

    # Shutdown: checkpoint + save
    phase.set("shutting_down", trial_id=trial_counter)
    async_tasks.reap(logger=log)
    async_tasks.shutdown()
    log.info("AutoPilot shutting down (trial=%d)", trial_counter)
    _save_state_with_journal_archive_authority(
        state,
        journal,
        archive,
        context="shutdown",
        merge_control=True,
    )
    # Final synchronous plot render so the operator's last view reflects the
    # final completed trial. Covers SIGTERM/SIGINT and the max-trials break (both
    # exit the loop to here); the %N-boundary regen almost never lands on the
    # exact stopping trial. Skipped in dry_run (no real metrics produced).
    if not dry_run:
        _refresh_plots(sync=True, reason="shutdown")
    if strategy_store is not None:
        strategy_store.close()
    if not dry_run:
        lab.checkpoint_state(trial_id=trial_counter, notes="Shutdown checkpoint")
    phase.clear("autopilot process exiting")
    if breadcrumb is not None:
        breadcrumb.mark_terminal(
            "loop_exit",
            exit_trigger="signal" if shutdown_requested else "loop_completed",
            trial_id=trial_counter,
        )


def _format_suite_trends(
    trends: dict[str, list[tuple[int, float]]],
) -> str:
    """Format suite quality trends for the controller prompt."""
    if not trends:
        return "  (no suite data yet)"
    lines = []
    for suite, points in sorted(trends.items()):
        vals = [q for _, q in points]
        direction = ""
        if len(vals) >= 3:
            recent_avg = sum(vals[-3:]) / 3
            older_avg = sum(vals[:3]) / 3
            delta = recent_avg - older_avg
            if delta < -0.05:
                direction = " ↓ DECLINING"
            elif delta > 0.05:
                direction = " ↑ improving"
        trail = " → ".join(f"{q:.2f}" for _, q in points[-5:])
        lines.append(f"  {suite}: {trail}{direction}")
    return "\n".join(lines)


def _auto_action(
    species: str,
    memory_count: int,
    converged: bool,
    seeder: Seeder,
) -> dict[str, Any]:
    """Generate an action without LLM controller (autonomous fallback)."""
    if memory_count < 500 or species == "seeder":
        return {"type": "seed_batch", "n_questions": SAFE_FALLBACK_SEED_N}
    elif species == "numeric_swarm":
        return {"type": "numeric_trial", "surface": "memrl_retrieval"}
    elif species == "prompt_forge":
        # AP-21: GEPA ratio knob — dynamic proportion of GEPA vs LLM mutation.
        # Default 0.30 (30% GEPA). Tunable via autopilot_state["gepa_ratio"].
        # Set to 1.0 after AR-3 data confirms GEPA dominance on Pareto frontier.
        import random

        _state = load_state()
        gepa_ratio = float(_state.get("gepa_ratio", 0.30))
        if random.random() < gepa_ratio:
            return {"type": "gepa_optimize", "file": "frontdoor.md", "max_evals": 50}
        return {"type": "prompt_mutation", "file": "frontdoor.md", "mutation": "targeted_fix"}
    elif species == "structural_lab":
        if converged:
            return {"type": "train_routing_models", "min_memories": 500}
        return {"type": "structural_experiment", "flags": {"think_harder": True}}
    elif species == "evolution_manager":
        return {"type": "distill_knowledge", "last_n": 10}
    return {"type": "seed_batch", "n_questions": SAFE_FALLBACK_SEED_N}


def _git_tag(tag: str, message: str) -> None:
    """Create a git tag."""
    try:
        subprocess.run(
            ["git", "tag", "-a", tag, "-m", message],
            capture_output=True,
            timeout=10,
            cwd=str(ORCH_ROOT),
        )
    except Exception:
        log.debug("Git tagging failed", exc_info=True)


# ── CLI Commands ─────────────────────────────────────────────────


def cmd_start(args: argparse.Namespace) -> None:
    """Start the optimization loop."""
    _enforce_startup_gate_env()
    _enforce_episodic_integrity_gate()
    _assert_eval_execution_instrument_state(load_state())

    # Process lock
    lock_file = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ERROR: Another AutoPilot instance is running")
        sys.exit(1)

    # Durable earlyoom protection for THIS process. The autopilot is comm=python
    # (collides with runaway python evals, so it cannot be earlyoom --ignore'd by
    # name) and is a long-lived control-plane process; set oom_score_adj=-1000 now
    # that we hold the singleton lock (earlyoom skips exactly -1000 in both oom_score
    # and --sort-by-rss modes). Only this pid — the transient GEPA/planner subprocesses
    # must stay killable. Best-effort (sudo -n). See earlyoom-oom-protection.md.
    _set_oom_score_adj([os.getpid()])

    run_loop(
        max_trials=args.max_trials,
        dry_run=args.dry_run,
        use_controller=not args.no_controller,
        use_tui=args.tui,
    )


def _journal_rows_for_archive(journal: ExperimentJournal) -> list[dict[str, Any]]:
    rows = [asdict(entry) for entry in journal.all_entries()]
    if hasattr(journal, "supersession_events"):
        rows.extend(journal.supersession_events())
    return rows


def _archive_for_read_command(
    journal: ExperimentJournal,
    *,
    source: str = ARCHIVE_SOURCE_JOURNAL_ALL,
) -> tuple[ParetoArchive, str]:
    """Archive view for operator read commands.

    Journal reconstruction is the default read path. The explicit ``state``
    source remains a legacy fallback for one release.
    """
    if source == ARCHIVE_SOURCE_STATE:
        return ParetoArchive(), ARCHIVE_SOURCE_STATE
    if source not in ARCHIVE_SOURCE_CHOICES:
        raise ValueError(f"unknown archive source: {source}")

    state = load_state()
    deinflate_before_ts, deinflate_factor, exclude_before_ts = _archive_epoch_params_from_state(
        state
    )
    archive = pareto_archive_from_journal_rows(
        _journal_rows_for_archive(journal),
        None,
        current_run_only=(source == ARCHIVE_SOURCE_JOURNAL_CURRENT_RUN),
        deinflate_before_ts=deinflate_before_ts,
        deinflate_factor=deinflate_factor,
        exclude_before_ts=exclude_before_ts,
        objective_policy=_live_objective_policy_from_state(state),
    )
    if archive is None:
        return (
            _ConcreteParetoArchive.from_archive_payload({}, read_only=True),
            f"{source}->empty-fallback",
        )
    return archive, source


def _append_baseline_promotion_event(
    *,
    journal: ExperimentJournal,
    baseline_update: Any,
    eval_result: EvalResult,
    source_trial_id: int,
    pareto_status: str,
    baseline_state: dict[str, Any],
    infra_fingerprint: dict[str, Any] | None = None,
    run_manifest: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if baseline_update is None:
        return None
    if not baseline_update.updated:
        # AP-57 (operator ruled B, 2026-09-17): a refused promotion that re-anchored the
        # speed axis is still a baseline_state WRITE; journal its append-only receipt so the
        # reseed is auditable and the ledger fold can replay it.
        if not getattr(baseline_update, "speed_reseeded", False):
            return None
        reseed = dict(getattr(baseline_update, "speed_reseed", None) or {})
        return journal.append_speed_axis_reseed_event(
            source_trial_id=source_trial_id,
            tier=int(reseed.get("tier", baseline_update.tier)),
            previous_speed=reseed.get("previous_speed"),
            new_speed=float(reseed.get("new_speed", eval_result.speed)),
            previous_speed_era=str(reseed.get("previous_speed_era") or ""),
            new_speed_era=str(
                reseed.get("new_speed_era") or baseline_state.get("autopilot_speed_era") or ""
            ),
            refusal_reason=str(getattr(baseline_update, "ineligible_reason", "") or ""),
            eval_quality_era=str(baseline_state.get("eval_quality_era") or ""),
            run_manifest_sha256=str((run_manifest or {}).get("manifest_sha256") or ""),
            result_metrics={
                "quality": eval_result.quality,
                "speed": eval_result.speed,
                "cost": eval_result.cost,
                "reliability": eval_result.reliability,
                "n_questions": eval_result.n_questions,
                "pareto_status": pareto_status,
                "refusal_detail": baseline_update.reason,
            },
            baseline_state=baseline_state,
            infra_fingerprint=infra_fingerprint,
        )
    return journal.append_baseline_promotion_event(
        source_trial_id=source_trial_id,
        tier=baseline_update.tier,
        previous_quality=baseline_update.previous_quality,
        new_quality=baseline_update.new_quality,
        reason=baseline_update.reason,
        proof=baseline_update.proof,
        result_metrics={
            "quality": eval_result.quality,
            "speed": eval_result.speed,
            "cost": eval_result.cost,
            "reliability": eval_result.reliability,
            "n_questions": eval_result.n_questions,
            "pareto_status": pareto_status,
            "promotion_rule": getattr(baseline_update, "promotion_rule", ""),
        },
        baseline_state=baseline_state,
        infra_fingerprint=infra_fingerprint,
    )


SPEED_AXIS_RESEED_JOURNAL_ERROR_KEY = "speed_axis_reseed_journal_error"


def _journal_baseline_ledger_event(
    state: dict[str, Any],
    *,
    journal: ExperimentJournal,
    baseline_update: Any,
    eval_result: EvalResult,
    source_trial_id: int,
    pareto_status: str,
    baseline_state: dict[str, Any],
    infra_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Append the trial's baseline ledger event; a failure is loud and never blocks the save.

    The state write (``baseline_state``) always proceeds, matching the promotion path. A
    failed speed-axis reseed receipt (AP-57) additionally leaves
    ``state[SPEED_AXIS_RESEED_JOURNAL_ERROR_KEY]`` set, so the missing receipt is visible
    in persisted telemetry until a later receipt lands.
    """
    reseed = bool(
        baseline_update is not None
        and not getattr(baseline_update, "updated", False)
        and getattr(baseline_update, "speed_reseeded", False)
    )
    try:
        event = _append_baseline_promotion_event(
            journal=journal,
            baseline_update=baseline_update,
            eval_result=eval_result,
            source_trial_id=source_trial_id,
            pareto_status=pareto_status,
            baseline_state=baseline_state,
            infra_fingerprint=infra_fingerprint,
            run_manifest=_in_flight_run_manifest(state, source_trial_id),
        )
    except Exception as exc:  # noqa: BLE001
        if reseed:
            log.warning(
                "Speed-axis reseed event append FAILED for trial %d: %s — the reseeded "
                "baseline_state is still saved, but it has no ledger receipt (AP-57); "
                "state.%s is set",
                source_trial_id,
                exc,
                SPEED_AXIS_RESEED_JOURNAL_ERROR_KEY,
            )
            state[SPEED_AXIS_RESEED_JOURNAL_ERROR_KEY] = {
                "trial_id": source_trial_id,
                "error": f"{type(exc).__name__}: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            log.warning(
                "Baseline promotion event append failed for trial %d: %s",
                source_trial_id,
                exc,
            )
        return None
    if reseed and event is not None:
        state.pop(SPEED_AXIS_RESEED_JOURNAL_ERROR_KEY, None)
    return event


def _baseline_promotion_summary_lines(
    state: dict[str, Any],
    journal: ExperimentJournal,
) -> list[str]:
    """Read-only baseline-as-ledger preview for operator commands."""
    reconciliation = reconcile_baseline_ledger(
        journal.baseline_ledger_events(),  # AP-57: promotions + reseeds
        state.get("baseline_state"),
    )
    return format_baseline_ledger_summary(reconciliation)


def _frontier_rerun_summary_lines(
    state: dict[str, Any],
    journal: ExperimentJournal,
) -> list[str]:
    """Read-only frontier-rerun marker summary for operator commands."""
    marker = state.get("frontier_rerun_required")
    if not (isinstance(marker, dict) and marker.get("required")):
        return ["Frontier rerun: not required"]

    completed = _frontier_rerun_completed_numeric_trials(marker, journal)
    min_trials = _frontier_rerun_min_trials(marker)
    reason = str(marker.get("reason") or "frontier rerun required")
    lines = [
        f"Frontier rerun: required ({completed}/{min_trials} numeric trials complete)",
        f"Frontier rerun reason: {reason}",
    ]
    opened = marker.get("rerun_started_at") or marker.get("opened_at")
    if opened:
        lines.append(f"Frontier rerun opened: {opened}")

    pending = state.get("frontier_rerun_pending_clear")
    if isinstance(pending, dict) and pending.get("trial_id") is not None:
        action = pending.get("action") if isinstance(pending.get("action"), dict) else {}
        action_type = action.get("type", "unknown")
        surface = action.get("surface")
        action_label = f"{action_type}/{surface}" if surface else str(action_type)
        lines.append(f"Frontier rerun pending: trial #{pending.get('trial_id')} {action_label}")

    blocked = state.get("frontier_rerun_blocked")
    if isinstance(blocked, dict):
        lines.append(f"Frontier rerun blocked: {blocked.get('reason', 'unknown')}")

    return lines


def cmd_status(args: argparse.Namespace) -> None:
    """Show current status."""
    state = load_state()
    journal = ExperimentJournal()
    archive, archive_source = _archive_for_read_command(
        journal,
        source=getattr(args, "archive_source", ARCHIVE_SOURCE_JOURNAL_ALL),
    )

    print("AutoPilot Status")
    print("=" * 50)
    print(f"Trial counter: {state.get('trial_counter', 0)}")
    print(f"Paused: {state.get('paused', False)}")
    if state.get("paused") and state.get(PAUSE_OWNER_FIELD):
        print(f"Pause owner: {state.get(PAUSE_OWNER_FIELD)}")
    collision = state.get(PAUSE_COLLISION_FIELD)
    if isinstance(collision, dict):
        # A collision that is detected but silent is still a defect — surface it
        # on the operator's default read-out until the pause episode is resumed.
        print(
            f"Pause collision: {collision.get('new_owner')} superseded "
            f"{collision.get('superseded_owner')} at {collision.get('at')} "
            "(that operation will not resume AutoPilot)"
        )
    print(f"Session ID: {state.get('session_id', 'none')}")
    if archive_source != ARCHIVE_SOURCE_STATE:
        print(f"Archive source: {archive_source}")
    for line in _baseline_promotion_summary_lines(state, journal):
        print(line)
    for line in _frontier_rerun_summary_lines(state, journal):
        print(line)
    print()
    print(archive.summary_text(tier=DEFAULT_FRONTIER_TIER))
    print()
    print(journal.summary_text(10))


def cmd_pause(args: argparse.Namespace) -> None:
    # Operator control write: hold the H4 write lock across the WHOLE
    # load->modify->save so a concurrent daemon save can neither lose this pause
    # nor have its counters clobbered by our stale whole-file write. We OWN
    # `paused` here (this IS the out-of-band writer) — do NOT merge_control.
    #
    # The operator SUPERSEDES any automated pause lease held by an in-flight
    # config-apply or cache flush. Not a refusal and not a deferral: the pause
    # lands now (the loop honours it at the next iteration top — quiesce and drain
    # at a boundary), the in-flight apply is never aborted mid-operation, but it
    # loses the right to resume AutoPilot when it finishes. Before the lease, that
    # applicator's `finally:` silently resumed a stack the operator had just
    # stopped (2026-08-03 outage).
    with state_write_lock(STATE_PATH):
        state = load_state()
        collision = supersede_pause_lease(state, OPERATOR_PAUSE_OWNER)
        save_state(state, _lock=False)
    print("AutoPilot paused")
    if collision:
        # A detected-but-silent collision is still a defect: say it out loud, and
        # `pause_collision` keeps it visible in `autopilot status` afterwards.
        print(
            "WARNING: paused during an in-flight "
            f"{collision.get('superseded_owner')} operation. Your pause takes "
            "precedence and that operation will NOT resume AutoPilot; the stack "
            "may be mid-apply, so verify it before resuming."
        )


def cmd_resume(args: argparse.Namespace) -> None:
    # Operator control write — same single-writer discipline as cmd_pause: the
    # read-modify-write is one locked critical section; we own the control fields.
    #
    # The halt latch is a DIFFERENT question from `paused`. `consecutive_skip_actions`
    # / `last_invalid_*` / `consecutive_meta_actions` are daemon-owned: this CLI runs
    # in its own process, so clearing them here while the daemon lives is destroyed at
    # the daemon's next save and the breaker re-trips on the next skipped action. The
    # gated helper clears them only when that write survives, and says so otherwise —
    # the running daemon clears them itself when it sees the resume.
    with state_write_lock(STATE_PATH):
        state = load_state()
        state["paused"] = False
        clear_pause_lease(state)
        state.pop("pause_reason", None)
        latch = clear_halt_latch(STATE_PATH, state)
        save_state(state, _lock=False)
    print("AutoPilot resumed")
    message = halt_latch_message(latch)
    if message:
        print(message)


def cmd_report(args: argparse.Namespace) -> None:
    """Generate markdown report."""
    journal = ExperimentJournal()
    archive, archive_source = _archive_for_read_command(
        journal,
        source=getattr(args, "archive_source", ARCHIVE_SOURCE_JOURNAL_ALL),
    )

    print("# AutoPilot Optimization Report")
    print()
    print(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    if archive_source != ARCHIVE_SOURCE_STATE:
        print(f"Archive source: {archive_source}")
    print()
    print("## Baseline Promotion Ledger")
    for line in _baseline_promotion_summary_lines(load_state(), journal):
        print(line)
    print()
    print("## Frontier Rerun")
    for line in _frontier_rerun_summary_lines(load_state(), journal):
        print(line)
    print()
    print("## Summary")
    print(journal.summary_text())
    print()
    print("## Pareto Frontier")
    print(archive.summary_text(tier=DEFAULT_FRONTIER_TIER))
    print()
    print("## Species Effectiveness")
    eff = journal.species_effectiveness()
    for sp, stats in eff.items():
        print(f"  {sp}: {stats['pareto']:.0f}/{stats['total']:.0f} ({stats['rate']:.1%})")


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate plots."""
    journal = ExperimentJournal()
    archive, archive_source = _archive_for_read_command(
        journal,
        source=getattr(args, "archive_source", ARCHIVE_SOURCE_JOURNAL_ALL),
    )
    state = load_state()
    td_errors = [(i, e) for i, e in enumerate(state.get("td_errors", []))]
    try:
        paths = generate_all_plots(archive, journal, td_errors, raise_on_error=True)
    except Exception as e:
        # Exit non-zero so the async reaper (phase_status.reap) surfaces this as
        # "[async] plots-trial-N failed" instead of falsely logging "complete".
        # A swallowed ImportError (missing matplotlib) silently froze the
        # dashboard panels for days; never let that happen quietly again.
        log.error("Plot generation failed: %s", e)
        sys.exit(1)
    if archive_source != ARCHIVE_SOURCE_STATE:
        print(f"Archive source: {archive_source}")
    for p in paths:
        print(f"  {p}")


def cmd_checkpoint(args: argparse.Namespace) -> None:
    lab = StructuralLab()
    state = load_state()
    cp = lab.checkpoint_state(
        trial_id=state.get("trial_counter", 0),
        mark_production_best=args.production_best,
        notes="Manual checkpoint",
    )
    print(f"Checkpoint created: {cp}")


def cmd_restore(args: argparse.Namespace) -> None:
    lab = StructuralLab()
    path = Path(args.checkpoint) if args.checkpoint else None
    result = lab.restore_checkpoint(path)
    print(f"Restore result: {result}")
    print(
        "Restore rewinds AP-22 short-term memory and StrategyStore on disk; "
        "restart any running AutoPilot daemon to reload in-memory handles."
    )


def cmd_digest(args: argparse.Namespace) -> None:
    """Generate an autopilot digest snapshot on demand.

    Useful between automated daily generations, or to verify the digest
    writer works without waiting for the next trial-loop iteration.
    """
    state = load_state()
    journal = ExperimentJournal()
    archive, archive_source = _archive_for_read_command(
        journal,
        source=getattr(args, "archive_source", ARCHIVE_SOURCE_JOURNAL_ALL),
    )
    swarm = NumericSwarm(epoch_label=_numeric_swarm_epoch_label_from_state(state))
    lab = StructuralLab()
    path = generate_digest(
        swarm=swarm,
        lab=lab,
        archive=archive,
        state=state,
        journal=journal,
        archive_source=archive_source,
        output_root=Path(args.output_root) if getattr(args, "output_root", None) else None,
    )
    if not args.no_state_update:
        state["last_digest_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        save_state(state)
    print(f"Digest written: {path}")


def cmd_peaf(args: argparse.Namespace) -> None:
    """Print PEAF correlation report (intake-571 spike cheap-kill check).

    Walks the experiment journal for entries with non-None surprise_score
    and a parent_trial, computes Pearson r² between surprise and the
    (entry.quality - parent.quality) delta, and reports the cheap-kill
    decision threshold (r² < 0.10 over min_n predicted trials → abandon).
    """
    journal = ExperimentJournal()
    report = peaf.journal_peaf_correlation(journal.all_entries(), min_n=args.min_n)
    enabled = (
        "ON (default)"
        if peaf.is_peaf_enabled()
        else "OFF (EPYC_AUTOPILOT_PEAF explicitly disabled — set to 1 to re-enable)"
    )
    print(f"PEAF status: {enabled}")
    print(f"Total journal entries: {journal.count()}")
    print(f"Entries with PEAF prediction + parent: {report['n_predicted']}")
    if report["mean_surprise"] is not None:
        print(f"Mean surprise (L1, normalised): {report['mean_surprise']:.4f}")
    if report["r_squared"] is None:
        print(f"r²: n/a (need at least {args.min_n} predicted trials with a parent)")
    else:
        print(f"r² (surprise vs Δquality from parent): {report['r_squared']:.4f}")
    print(f"Decision: {report['decision']}")
    if report["decision"] == "abandon":
        print(
            "  → r² < 0.10 — PEAF signal does not correlate with config-quality gradient; consider abandoning."
        )
    elif report["decision"] == "continue":
        print(
            "  → r² ≥ 0.10 — PEAF signal correlates; consider promoting surprise as Pareto co-objective."
        )
    else:
        print("  → keep collecting.")


def _read_baseline_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def _drop_top_level_yaml_block(text: str, key: str) -> str:
    pattern = rf"(?ms)^({re.escape(key)}:\n(?:^[ \t]+.*\n?|^\n)*)"
    return re.sub(pattern, "", text).rstrip() + "\n"


def _format_baseline_tier_yaml(baseline: Baseline) -> str:
    data = {
        "baselines_by_tier": {
            int(tier): quality for tier, quality in sorted(baseline.baselines_by_tier.items())
        },
        "per_suite_quality_by_tier": {
            int(tier): suites for tier, suites in sorted(baseline.per_suite_quality_by_tier.items())
        },
        "per_suite_counts_by_tier": {
            int(tier): counts for tier, counts in sorted(baseline.per_suite_counts_by_tier.items())
        },
    }
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True).rstrip()


def _write_baseline_yaml_tiers(path: Path, baseline: Baseline) -> None:
    """Update YAML seed tier fields without dropping unrelated calibration tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        text = path.read_text()
        text = _drop_top_level_yaml_block(text, "baselines_by_tier")
        text = _drop_top_level_yaml_block(text, "per_suite_quality_by_tier")
        text = _drop_top_level_yaml_block(text, "per_suite_counts_by_tier")
        _atomic_write_text(
            path, text.rstrip() + "\n\n" + _format_baseline_tier_yaml(baseline) + "\n"
        )
        return

    data = {
        "quality": baseline.quality,
        "speed": baseline.speed,
        "cost": baseline.cost,
        "reliability": baseline.reliability,
        "frontdoor_speed": baseline.frontdoor_speed,
        "per_suite_quality": baseline.per_suite_quality,
        "baselines_by_tier": baseline.baselines_by_tier,
        "per_suite_quality_by_tier": baseline.per_suite_quality_by_tier,
        "per_suite_counts_by_tier": baseline.per_suite_counts_by_tier,
    }
    _atomic_write_text(path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def _migrate_flat_baseline_to_tier(
    baseline: Baseline,
    *,
    target_tier: int = 2,
) -> bool:
    """Move the historical flat quality seed into its documented eval tier."""
    if target_tier in baseline.baselines_by_tier:
        return False
    tier_quality = Baseline._validate_quality(
        baseline.quality,
        None,
        f"baseline migration T{target_tier}",
        baseline.source_path or DEFAULT_BASELINE_PATH,
    )
    if tier_quality is None:
        return False
    baseline.baselines_by_tier[target_tier] = tier_quality
    if target_tier not in baseline.per_suite_quality_by_tier:
        baseline.per_suite_quality_by_tier[target_tier] = dict(baseline.per_suite_quality)
    return True


def _calibration_eval_quality_era(result: EvalResult) -> str | None:
    """Era of the INSTRUMENT THAT PRODUCED ``result`` — fail-closed, never guessed.

    RTG-02 (2026-09-14). A baseline is only comparable to a later eval when both were
    measured on the same eval instrument, which is what ``baseline_state.eval_quality_era``
    records and what ``SafetyGate.quality_rebaseline_required`` enforces. The era to stamp is
    therefore the one the RESULT was measured under — stamped onto the result by
    ``eval_tower._stamp_eval_instrument`` at measurement time — and explicitly NOT
    ``active_instrument_eras.eval_quality`` read at write time: those two differ across
    exactly the boundary the fence exists to detect, and reading the second would stamp a
    provenance claim the measurement does not support.

    Returns the era id, or None for the unfenced single-era world (registry read fine, no
    ``eval_quality`` era open — the hold is inert, so there is nothing to stamp). RAISES when
    the result carries no era at all: an unstamped result is a measurement of unknown
    provenance, and inventing an era for it is the one failure mode worse than the hold.
    """
    details = getattr(result, "details", None) or {}
    era = str(details.get("eval_quality_era") or "").strip()
    if era:
        return era
    status = str(details.get("eval_quality_era_status") or "").strip()
    if status == "unfenced":
        return None
    raise ValueError(
        "calibration result carries no eval_quality era "
        f"(eval_quality_era_status={status or '<absent>'}): refusing to write a baseline whose "
        "instrument era is unknown. An unstamped baseline trips the eval-instrument re-baseline "
        "hold forever, and guessing the era current at write time would stamp a provenance "
        "claim the measurement does not support. Remediation: run the calibration through "
        "EvalTower (which stamps the era from orchestration/instrument_eras.yaml) and make sure "
        "that registry is readable — status 'unresolved' means it was not."
    )


def _apply_calibrated_baseline_result(baseline: Baseline, result: EvalResult) -> None:
    tier = int(result.tier)
    # RTG-02: resolve the era BEFORE any field is written, so a refusal leaves the baseline
    # untouched. This function used to write every baseline VALUE and never the era stamp, so
    # the one in-tree tool that produces a fresh baseline left the quality hold exactly where
    # it was (documented verbatim in operator_seed_e8_operational_baseline.py's header) —
    # AutoPilot could never promote quality again after an era boundary.
    era = _calibration_eval_quality_era(result)
    quality = Baseline._validate_quality(
        result.quality,
        None,
        f"T{tier} calibration result",
        baseline.source_path or DEFAULT_BASELINE_PATH,
    )
    if quality is None:
        raise ValueError(f"T{tier} calibration produced invalid quality: {result.quality!r}")
    if result.n_questions <= 0:
        raise ValueError("Baseline calibration requires a non-empty evaluation result")
    if result.reliability <= 0:
        raise ValueError(
            f"T{tier} calibration produced zero reliability; refusing to persist baseline"
        )
    baseline.baselines_by_tier[tier] = quality
    baseline.per_suite_quality_by_tier[tier] = dict(result.per_suite_quality)
    # Persist the per-suite question counts the baseline was measured at so the
    # per-suite regression gate's threshold knows the baseline's own sampling
    # resolution (3/n quantum); without this a calibration refresh leaves the
    # baseline-side count term inactive (2026-06-07).
    baseline.per_suite_counts_by_tier[tier] = dict(getattr(result, "per_suite_counts", {}) or {})
    # RTG-02: stamp the producing instrument's era onto the baseline it just produced, so the
    # eval-instrument fence closes on this evidence. Never CLEARS an existing stamp (the
    # unfenced case has nothing to say about provenance), matching update_baseline()'s rule.
    if era:
        baseline.eval_quality_era = era


def calibrate_baseline(
    *,
    tier: int = DEFAULT_FRONTIER_TIER,
    n: int | None = None,
    seed: int = 42,
    baseline_path: Path | None = None,
    migrate_only: bool = False,
    write: bool = True,
) -> tuple[Baseline, EvalResult | None, bool]:
    """Migrate flat baseline state and optionally seed a calibrated tier baseline."""
    path = baseline_path or DEFAULT_BASELINE_PATH
    state = load_state()
    baseline = Baseline.load(path, state=state.get("baseline_state") or {})
    migrated_t2 = _migrate_flat_baseline_to_tier(baseline, target_tier=2)

    result: EvalResult | None = None
    if not migrate_only:
        tower = EvalTower(url=ORCHESTRATOR_URL)
        if tier == 0:
            result = tower.eval_t0()
        elif tier == 1:
            result = tower.eval_t1(n=n or 100, seed=seed)
        elif tier == 2:
            result = tower.eval_t2(n=n or 500, seed=seed)
        elif tier == 3:
            result = tower.eval_t3(n=n or 160, seed=seed)
        else:
            raise ValueError(f"Unknown eval tier: {tier}")
        _apply_calibrated_baseline_result(baseline, result)

    state["baseline_state"] = baseline.to_state_dict()
    if write:
        _write_baseline_yaml_tiers(path, baseline)
        save_state(state)
    return baseline, result, migrated_t2


def cmd_calibrate_baseline(args: argparse.Namespace) -> None:
    baseline, result, migrated_t2 = calibrate_baseline(
        tier=args.tier,
        n=args.n,
        seed=args.seed,
        baseline_path=Path(args.baseline_path) if args.baseline_path else None,
        migrate_only=args.migrate_only,
        write=not args.dry_run,
    )
    write_label = "dry-run" if args.dry_run else "persisted"
    print(f"Baseline calibration {write_label}")
    print(f"  T2 flat migration: {'applied' if migrated_t2 else 'already present'}")
    if result is not None:
        print(
            f"  T{result.tier} calibrated: q={result.quality:.3f} "
            f"r={result.reliability:.3f} n={result.n_questions}"
        )
    print(
        "  baselines_by_tier: "
        + ", ".join(
            f"T{tier}={quality:.3f}" for tier, quality in sorted(baseline.baselines_by_tier.items())
        )
    )


# ── Entry Point ──────────────────────────────────────────────────


def main() -> None:
    log_path = ORCH_ROOT / "logs" / "autopilot.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # force=True overrides any basicConfig already called at import time
    # (seed_specialist_routing.py calls basicConfig at module level)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=_autopilot_logging_handlers(log_path),
        force=True,
    )

    parser = argparse.ArgumentParser(description="AutoPilot: Continuous recursive optimization")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start
    p_start = subparsers.add_parser("start")
    p_start.add_argument("--dry-run", action="store_true")
    p_start.add_argument("--max-trials", type=int, default=None)
    p_start.add_argument(
        "--no-controller",
        action="store_true",
        help="Use autonomous species selection instead of Claude CLI",
    )
    p_start.add_argument(
        "--tui", action="store_true", help="Live Rich TUI for inference monitoring (hang detection)"
    )
    p_start.set_defaults(func=cmd_start)

    # status
    p_status = subparsers.add_parser("status")
    p_status.add_argument(
        "--archive-source",
        choices=ARCHIVE_SOURCE_CHOICES,
        default=ARCHIVE_SOURCE_JOURNAL_ALL,
        help=(
            "Archive read source for this operator command only. "
            "Defaults to journal-all; state is a legacy fallback."
        ),
    )
    p_status.set_defaults(func=cmd_status)

    # pause / resume
    p_pause = subparsers.add_parser("pause")
    p_pause.set_defaults(func=cmd_pause)
    p_resume = subparsers.add_parser("resume")
    p_resume.set_defaults(func=cmd_resume)

    # report
    p_report = subparsers.add_parser("report")
    p_report.add_argument(
        "--archive-source",
        choices=ARCHIVE_SOURCE_CHOICES,
        default=ARCHIVE_SOURCE_JOURNAL_ALL,
        help=(
            "Archive read source for this operator command only. "
            "Defaults to journal-all; state is a legacy fallback."
        ),
    )
    p_report.set_defaults(func=cmd_report)

    # plot
    p_plot = subparsers.add_parser("plot")
    p_plot.add_argument(
        "--archive-source",
        choices=ARCHIVE_SOURCE_CHOICES,
        default=ARCHIVE_SOURCE_JOURNAL_ALL,
        help=(
            "Archive read source for this operator command only. "
            "Defaults to journal-all; state is a legacy fallback."
        ),
    )
    p_plot.set_defaults(func=cmd_plot)

    # checkpoint
    p_cp = subparsers.add_parser("checkpoint")
    p_cp.add_argument("--production-best", action="store_true")
    p_cp.set_defaults(func=cmd_checkpoint)

    # restore
    p_restore = subparsers.add_parser("restore")
    p_restore.add_argument("--checkpoint", type=str, default=None)
    p_restore.set_defaults(func=cmd_restore)

    # monitor — standalone TUI (read-only, doesn't own autopilot process)
    p_monitor = subparsers.add_parser(
        "monitor",
        help="Live TUI monitor (standalone, read-only — run in a separate terminal)",
    )
    p_monitor.set_defaults(func=cmd_monitor)

    # reset-memory — clear short-term memory (AP-22)
    p_reset_mem = subparsers.add_parser(
        "reset-memory",
        help="Clear short-term memory (start fresh for next session)",
    )
    p_reset_mem.set_defaults(func=cmd_reset_memory)

    # digest — generate an autopilot snapshot on demand
    p_digest = subparsers.add_parser(
        "digest",
        help="Generate an autopilot digest snapshot (progress/YYYY-MM/YYYY-MM-DD-autopilot.md)",
    )
    p_digest.add_argument(
        "--no-state-update",
        action="store_true",
        help="Do NOT update state['last_digest_date'] — useful for ad-hoc snapshots that should not delay the next automatic generation.",
    )
    p_digest.add_argument(
        "--archive-source",
        choices=ARCHIVE_SOURCE_CHOICES,
        default=ARCHIVE_SOURCE_JOURNAL_ALL,
        help=(
            "Archive read source for this operator command only. "
            "Defaults to journal-all; state is a legacy fallback."
        ),
    )
    p_digest.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Optional progress root override for this digest write. "
            "Useful for ad-hoc smoke snapshots outside the root progress tree."
        ),
    )
    p_digest.set_defaults(func=cmd_digest)

    # peaf — Prediction-Error-As-Feature cheap-kill correlation report (intake-571 spike)
    p_peaf = subparsers.add_parser(
        "peaf",
        help="PEAF cheap-kill report: Pearson r² between surprise_score and Δquality from parent trial. Abandon at r²<0.10 over min_n predicted trials.",
    )
    p_peaf.add_argument(
        "--min-n",
        type=int,
        default=200,
        help="Minimum predicted-trials sample size before computing r² (default: 200, per intake-571 cheap-kill criterion).",
    )
    p_peaf.set_defaults(func=cmd_peaf)

    # calibrate-baseline — one-shot YAML/state seed migration + tier calibration
    p_calibrate = subparsers.add_parser(
        "calibrate-baseline",
        help="Migrate flat baseline into T2 and optionally run a one-shot tier calibration.",
    )
    p_calibrate.add_argument(
        "--tier",
        type=int,
        default=DEFAULT_FRONTIER_TIER,
        help="Eval tier to calibrate (default: canonical T1).",
    )
    p_calibrate.add_argument(
        "--n",
        type=int,
        default=None,
        help="Question count override for the calibration eval.",
    )
    p_calibrate.add_argument("--seed", type=int, default=42)
    p_calibrate.add_argument(
        "--baseline-path",
        type=str,
        default=None,
        help="Baseline YAML path (defaults to orchestration/autopilot_baseline.yaml).",
    )
    p_calibrate.add_argument(
        "--migrate-only",
        action="store_true",
        help="Only migrate the flat baseline into T2; do not run EvalTower.",
    )
    p_calibrate.add_argument(
        "--dry-run",
        action="store_true",
        help="Run calibration logic without writing YAML or state.",
    )
    p_calibrate.set_defaults(func=cmd_calibrate_baseline)

    args = parser.parse_args()
    args.func(args)


def cmd_reset_memory(args: argparse.Namespace) -> None:
    """Clear short-term memory."""
    mem = ShortTermMemory()
    mem.clear()
    print("Short-term memory cleared.")


def cmd_monitor(args: argparse.Namespace) -> None:
    """Launch standalone TUI monitor (read-only)."""
    from autopilot_tui import AutoPilotTUI

    print("Starting standalone TUI monitor (read-only)...")
    print("Press Ctrl+C to exit.\n")
    with AutoPilotTUI():
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
