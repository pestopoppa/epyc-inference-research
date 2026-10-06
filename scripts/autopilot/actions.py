"""Autopilot action handlers and dispatcher.

Extracted from autopilot.py during the 2026-05-22 Tranche-5 refactor. Each
action type from the controller's response maps to one `_action_<type>`
handler; `dispatch_action()` is the public facade that:
  1. validates single-variable scope (AP-9)
  2. routes to the matching handler

Behavior is preserved verbatim — no semantic changes to any action. The
handler signatures pack the autopilot's many species/state objects into an
`_ActionContext` bundle so each handler stays readable.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from controller_io import suppressed_numeric_surfaces, validate_single_variable
import rejected_mutation_ledger
from mutation_author_capture import author_capture_kwargs
from safety_gate import EvalResult, SafetyGate
from species.prompt_forge import diversity_coverage_penalty
from src.autopilot_core.tier_specs import UnmeasuredObjectiveError, objectives_from

ORCH_ROOT = Path(__file__).resolve().parents[2]

# Operator decision 2026-09-16: a prompt / code / GEPA mutation is identified by the sha256
# of the file content actually SERVED. Handlers hash the served file(s) right after the
# mutation is written and immediately before the eval runs, and leave the record here; the
# trial loop pops it into ``eval_details.served_content``. It is never written onto the
# action, so a forced re-run (which copies the stored action) cannot inherit a stale sha,
# and action fingerprints / repeat detection are unchanged.
SERVED_CONTENT_STATE_KEY = "_trial_served_content"


def served_content_record(
    paths: Iterable[Path], *, root: Path | None = None
) -> dict[str, Any] | None:
    """``{"files": {path: sha256}}`` for the files as they are on disk now, or None."""
    import hashlib

    root = ORCH_ROOT if root is None else root

    files: dict[str, str] = {}
    for path in paths:
        path = Path(path)
        try:
            data = path.read_bytes()
        except OSError as exc:
            log.warning("served-content hash failed for %s: %s", path, exc)
            return None
        try:
            key = str(path.resolve().relative_to(Path(root).resolve()))
        except ValueError:
            key = str(path.resolve())
        files[key] = hashlib.sha256(data).hexdigest()
    if not files:
        return None
    return {"files": dict(sorted(files.items()))}


def _record_served_content(
    ctx: "_ActionContext",
    paths: Iterable[Path],
    *,
    restore: dict[str, Any] | None = None,
) -> None:
    """Leave the served-file record for the loop; ``restore`` is the multitier preimage.

    ``restore`` = ``{"kind": "prompt"|"code", "file": ..., "preimage": original text,
    "preimage_sha256": ...}``. It stays in loop state and the pending multitier record only;
    the journal row keeps the served shas alone.
    """
    record = served_content_record(paths)
    if record is None:
        ctx.state.pop(SERVED_CONTENT_STATE_KEY, None)
        return
    record["captured_at"] = datetime.now(timezone.utc).isoformat()
    if restore is not None:
        record["restore"] = restore
    ctx.state[SERVED_CONTENT_STATE_KEY] = record


def _mutation_restore(kind: str, mutation: Any) -> dict[str, Any] | None:
    import hashlib

    original = getattr(mutation, "original_content", None)
    if not isinstance(original, str):
        return None
    return {
        "kind": kind,
        "file": str(getattr(mutation, "file", "") or ""),
        "new_file": bool(getattr(mutation, "mutation_type", "") == "new_file" and not original),
        "preimage": original,
        "preimage_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
    }


MULTITIER_ROLLBACK_MAX_FAILURES = max(
    1, int(os.environ.get("AUTOPILOT_MULTITIER_ROLLBACK_MAX_FAILURES", "3") or 3)
)
ROLLBACK_EXTERNAL_CHANGE_ALARM_KEY = "autopilot-multitier-rollback-external-change"
ROLLBACK_STALLED_ALARM_KEY = "autopilot-multitier-rollback-stalled"


def _mutation_target_path(ctx: "_ActionContext", restore: dict[str, Any]) -> Path:
    target = str(restore.get("file") or "")
    if restore.get("kind") == "prompt":
        return Path(ctx.forge._resolve_prompt_path(target))
    return ORCH_ROOT / target


def _file_sha(path: Path) -> str | None:
    import hashlib

    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _mutation_served_sha(ctx: "_ActionContext", restore: dict[str, Any], served: Any) -> str | None:
    """The sha the candidate served for its target file (from the staged identity)."""
    files = (served or {}).get("files") if isinstance(served, dict) else None
    if not isinstance(files, dict):
        return None
    path = _mutation_target_path(ctx, restore).resolve()
    for rel, sha in files.items():
        if (ORCH_ROOT / rel).resolve() == path:
            return str(sha)
    return None


def _restore_mutation_preimage(
    ctx: "_ActionContext", restore: Any, served: Any = None
) -> dict[str, Any]:
    """Multitier rollback of a rejected mutation candidate: write the preimage back.

    Writes ONLY when the file still holds exactly the content the candidate served. If the
    file changed since staging (an operator edit, a pull, a merge), the change is not the
    candidate's to undo: nothing is written and ``status = "external_change"`` is returned
    (gate-frontier review fix 2). The final on-disk state is attested separately by
    :func:`_attest_mutation_restore`, after the checkpoint restore (fix 1).
    """
    if not isinstance(restore, dict) or not isinstance(restore.get("preimage"), str):
        return {"status": "error", "error": "mutation candidate lacks a restore preimage"}
    kind = restore.get("kind")
    target = str(restore.get("file") or "")
    if kind not in {"prompt", "code"}:
        return {"status": "error", "error": f"unknown mutation restore kind: {kind}"}
    try:
        path = _mutation_target_path(ctx, restore)
        expected = _mutation_served_sha(ctx, restore, served)
        current = _file_sha(path)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"mutation restore precheck failed: {exc}"}
    if expected is None:
        return {"status": "error", "error": f"no staged served sha for {target}"}
    if current != expected:
        return {
            "status": "external_change",
            "file": target,
            "served_sha256": expected,
            "current_sha256": current,
            "error": (
                f"{target} changed since the candidate was staged (served {expected[:12]}, "
                f"now {(current or 'missing')[:12]}); preimage NOT written"
            ),
        }
    try:
        if kind == "prompt":
            from species.prompt_forge import PromptMutation

            ctx.forge.revert_mutation(
                PromptMutation(
                    file=target,
                    mutation_type="multitier_rollback",
                    description="multitier rollback of a rejected mutation candidate",
                    original_content=restore["preimage"],
                )
            )
        else:
            from species.prompt_forge import CodeMutation

            new_file = bool(restore.get("new_file"))
            ctx.forge.revert_code_mutation(
                CodeMutation(
                    file=target,
                    mutation_type="new_file" if new_file else "multitier_rollback",
                    description="multitier rollback of a rejected mutation candidate",
                    original_content=restore["preimage"],
                )
            )
    except Exception as exc:  # noqa: BLE001 - reported as a rollback failure
        return {"status": "error", "error": f"mutation restore failed: {exc}"}
    return _attest_mutation_restore(ctx, restore)


def _attest_mutation_restore(ctx: "_ActionContext", restore: dict[str, Any]) -> dict[str, Any]:
    """The on-disk state of the candidate's file equals its preimage (or is absent)."""
    target = str(restore.get("file") or "")
    try:
        path = _mutation_target_path(ctx, restore)
        current = _file_sha(path)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"mutation restore attestation failed: {exc}"}
    if restore.get("new_file"):
        if current is not None:
            return {"status": "error", "error": f"new_file {target} still present"}
        return {"status": "ok", "file": target, "removed": True}
    if current != restore.get("preimage_sha256"):
        return {
            "status": "error",
            "error": f"mutation restore attestation failed for {target}: sha {current}",
        }
    return {"status": "ok", "file": target, "sha256": current}


def _public_pending_copy(pending: dict[str, Any]) -> dict[str, Any]:
    """A pending-candidate snapshot without the preimage text (fix 4)."""
    copy_ = dict(pending)
    copy_.pop("candidate_restore", None)
    return copy_


def _raise_rollback_alarm(key: str, severity: str, message: str, evidence: dict[str, Any]) -> None:
    try:
        from eval_leakage_monitor import AlarmChannelClient

        AlarmChannelClient().raise_alarm(message, evidence, key=key, severity=severity)
    except Exception as exc:  # noqa: BLE001 - an alarm must never break the rollback path
        log.error("multitier rollback alarm %s could not be raised: %s", key, exc)


def _served_prompt_path(ctx: "_ActionContext", filename: str) -> Path | None:
    try:
        return ctx.forge._resolve_prompt_path(filename)
    except Exception:  # noqa: BLE001 - identity is optional evidence, never a failure
        return None

SEQ_PROMOTION_RECENT_QID_TRIALS = int(
    os.environ.get("AUTOPILOT_SEQ_PROMOTION_RECENT_QID_TRIALS", "100")
)
SEQ_PROMOTION_RECENT_QID_DAYS = int(os.environ.get("AUTOPILOT_SEQ_PROMOTION_RECENT_QID_DAYS", "60"))


def _apply_params(*args, **kwargs):
    """Call apply_params via the autopilot module so tests' monkeypatches stick.

    Tests historically monkeypatch `autopilot.apply_params`; importing the
    function directly here would bypass that. Lazy lookup through sys.modules
    avoids a circular import (actions.py is imported by autopilot.py at the
    bottom of autopilot's imports, by which time `apply_params` is bound).
    """
    import sys

    # autopilot is imported as either 'autopilot' (normal load mode),
    # 'scripts.autopilot.autopilot' (package-path tests), or '__main__'
    # (direct script execution). Prefer the package path when both aliases are
    # present so tests that monkeypatch that module cannot leak into real
    # config application.
    for module_name in ("scripts.autopilot.autopilot", "autopilot", "__main__"):
        mod = sys.modules.get(module_name)
        if mod is not None and hasattr(mod, "apply_params"):
            return mod.apply_params(*args, **kwargs)

    # Fallback: import config_applicator directly (no monkeypatch in play).
    from config_applicator import apply_params as _ap

    return _ap(*args, **kwargs)


def _capture_param_preimage(params: dict[str, Any]) -> dict[str, Any]:
    """Lazy import keeps action-module test monkeypatching and load order intact."""
    try:
        from config_applicator import capture_reversible_param_preimage
    except ImportError:  # pragma: no cover - package-path invocation
        from scripts.autopilot.config_applicator import capture_reversible_param_preimage

    return capture_reversible_param_preimage(params)


def _restore_param_preimage(preimage: dict[str, Any]) -> dict[str, Any]:
    try:
        from config_applicator import restore_reversible_param_preimage
    except ImportError:  # pragma: no cover - package-path invocation
        from scripts.autopilot.config_applicator import restore_reversible_param_preimage

    return restore_reversible_param_preimage(preimage)


def _multitier_runtime_enabled() -> bool:
    return os.environ.get("AUTOPILOT_MULTITIER_PROMOTION", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _normalize_numeric_trial_params(
    surface: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Accept controller-friendly short param names for a numeric surface.

    NumericSwarm's internal/applicator names are fully qualified
    (``kv.keep_ratio``), but the controller often emits the user-facing knob
    name from the action schema (``keep_ratio``). Normalize when the short name
    is unambiguous within the selected surface and leave unknown keys untouched
    so the applicator can report an actionable error.
    """
    if not params:
        return {}
    try:
        from species.numeric_swarm import SURFACES
    except Exception:
        log.debug("Could not import NumericSwarm surfaces for param normalization", exc_info=True)
        return dict(params)

    specs = SURFACES.get(surface, [])
    full_names = {spec.name for spec in specs}
    short_to_full: dict[str, str] = {}
    for spec in specs:
        if "." not in spec.name:
            continue
        short = spec.name.split(".", 1)[1]
        if short in short_to_full:
            short_to_full.pop(short, None)
        else:
            short_to_full[short] = spec.name

    normalized: dict[str, Any] = {}
    for key, value in params.items():
        key_s = str(key)
        if key_s in full_names:
            normalized[key_s] = value
        elif key_s in short_to_full:
            normalized[short_to_full[key_s]] = value
        else:
            normalized[key_s] = value
    return normalized


def _numeric_apply_error_skip(
    *,
    surface: str,
    params: dict[str, Any],
    apply_result: dict[str, Any],
) -> SkipOutcome:
    errors = apply_result.get("errors") or []
    if isinstance(errors, str):
        errors = [errors]
    reason = "; ".join(str(error) for error in errors) or str(apply_result)
    unknown = apply_result.get("unknown_params") or []
    status = "invalid" if unknown or "unknown_params:" in reason else "skipped"
    infra = _numeric_apply_error_is_infra(reason, apply_result)
    return SkipOutcome(
        status,
        (
            f"numeric_trial params failed to apply for surface {surface}: {reason}; "
            f"params={dict(params)}"
        ),
        "numeric_trial",
        bug_corrupted_by="env_restart_apply_failure" if infra else "",
        bug_corrupted_reason=(
            "numeric_trial params were not applied because API/env restart failed" if infra else ""
        ),
    )


def _numeric_apply_error_is_infra(reason: str, apply_result: dict[str, Any]) -> bool:
    """True when params failed because orchestration reload infrastructure failed."""
    if apply_result.get("unknown_params") or "unknown_params:" in reason:
        return False
    return "env_restart:" in reason


def _numeric_apply_no_changes(apply_result: dict[str, Any]) -> bool:
    if apply_result.get("status") == "no_changes":
        return True
    nested_results = [
        apply_result.get("hot_swap_result"),
        apply_result.get("env_result"),
        apply_result.get("kv_compact_result"),
    ]
    present = [result for result in nested_results if isinstance(result, dict)]
    return bool(present) and all(result.get("status") == "no_changes" for result in present)


def _numeric_no_change_skip(
    *,
    surface: str,
    params: dict[str, Any],
) -> SkipOutcome:
    return SkipOutcome(
        "skipped",
        (
            f"numeric_trial params produced no live config changes for surface "
            f"{surface}; params={dict(params)}"
        ),
        "numeric_trial",
    )


if TYPE_CHECKING:
    from experiment_journal import ExperimentJournal
    from pareto_archive import ParetoArchive
    from eval_tower import EvalTower
    from species import (
        Seeder,
        NumericSwarm,
        PromptForge,
        StructuralLab,
        EvolutionManager,
    )
    from orchestration.repl_memory.strategy_store import StrategyStore

log = logging.getLogger("autopilot")
_SKILL_EFFICACY_GATE_ENV = "AUTOPILOT_SKILL_EFFICACY_GATE"
_BSV2_ACCEPT_GATE_ENV = "AUTOPILOT_BSV2_ACCEPT_GATE"
_BSV2_MIN_SHARED_QIDS_ENV = "AUTOPILOT_BSV2_MIN_SHARED_QIDS"
_BSV2_MAX_ACCURACY_REGRESSION_ENV = "AUTOPILOT_BSV2_MAX_ACCURACY_REGRESSION"
_PLANNER_HINTS_ENABLED = os.environ.get("AUTOPILOT_PLANNER_HINTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_MUTATION_DIVERSITY_COVERAGE_STATE_KEY = "_mutation_diversity_coverage"


@dataclass
class SkipOutcome:
    """Structured residue for an action that did NOT execute an eval.

    Returned (in the eval-result slot of the ``(result, species)`` tuple) instead
    of a bare ``None`` so the main loop can treat a non-executing action as a
    first-class trial outcome: journal it, fingerprint it, count it, and feed the
    reason back to the planner / blacklist. Before this existed, the dispatcher's
    ``return None`` path silently dropped the only actionable signal — which let
    the planner re-sample an impossible action forever (the 2026-06 graph_router
    deadlock: 119 identical invalid ``structural_experiment`` dispatches, each
    rejected with "graph_router feature requires specialist_routing feature",
    none of it ever reaching the planner).

    status: "invalid"  — failed pre-execution validation (actionable; e.g. a
                          feature flag whose dependency is not enabled). Eligible
                          for signature blacklisting because the reason is stable.
            "skipped"  — dropped by a dispatcher guard (AP-9 scope, dirty-tree
                          fence, unknown type, handler no-op). Counted + halted on
                          runaway, but NOT auto-blacklisted (the coarse pattern
                          would over-match, e.g. blacklisting all numeric_trials).
    reason:  human/planner-readable explanation (the validator/guard message).
    """

    status: str
    reason: str
    action_type: str = ""
    bug_corrupted_by: str = ""
    bug_corrupted_reason: str = ""


@dataclass
class _ActionContext:
    """Dependency bundle passed to each action handler."""

    seeder: "Seeder"
    swarm: "NumericSwarm"
    forge: "PromptForge"
    lab: "StructuralLab"
    tower: "EvalTower"
    gate: SafetyGate
    archive: "ParetoArchive"
    journal: "ExperimentJournal"
    state: dict[str, Any]
    strategy_store: "StrategyStore | None" = None
    evo: "EvolutionManager | None" = None
    # 2026-05-23 Phase 5 — OrchestratorWatcher for exogenous-restart
    # detection. None = legacy behavior (no retry, no metadata propagation).
    # Phase 6/7 deployments construct one in autopilot.py main() and pass
    # it through dispatch_action → _action_seed_batch → Seeder.run_batch.
    watcher: Any | None = None


# -----------------------------------------------------------------------------
# Per-action handlers — one per action_type
# -----------------------------------------------------------------------------


def _format_strategy_hint(entry: Any) -> str:
    title = str(getattr(entry, "title", "") or "").strip()
    description = str(getattr(entry, "description", "") or "").strip()
    insight = (
        str(getattr(entry, "generalized_content", "") or "").strip()
        or str(getattr(entry, "insight", "") or "").strip()
    )
    source = getattr(entry, "source_trial_id", "")
    species = str(getattr(entry, "species", "") or "").strip() or "unknown"
    prefix = f"Trial #{source} ({species})"
    if title:
        prefix = f"{prefix} {title}:"
    elif description:
        prefix = f"{prefix} {description}:"
    else:
        prefix = f"{prefix}:"
    return f"- {prefix} {insight}".strip()


def _prompt_forge_convention_guardrails(
    ctx: _ActionContext,
    *,
    limit: int = 8,
) -> str | None:
    """Return PromptForge convention rows that should not depend on RRF rank.

    RRF retrieval is query-specific and can miss broad operator guardrails.
    Convention rows are already journal/quarantine aware through
    ``StrategyStore.retrieve_conventions()``, so use that store-owned selector
    and keep the whole path behind the same startup gate as other planner hints.
    """
    if not _PLANNER_HINTS_ENABLED or ctx.strategy_store is None:
        return None
    if not hasattr(ctx.strategy_store, "retrieve_conventions"):
        log.warning(
            "Skipping PromptForge convention guardrails: StrategyStore lacks retrieve_conventions()"
        )
        return None
    try:
        conventions = ctx.strategy_store.retrieve_conventions(
            species="prompt_forge",
            journal=ctx.journal,
            limit=limit,
        )
    except TypeError:
        log.warning(
            "Skipping PromptForge convention guardrails: StrategyStore "
            "retrieve_conventions() has an incompatible signature"
        )
        return None
    if not conventions:
        return None

    lines = "\n".join(_format_strategy_hint(convention) for convention in conventions)
    return (
        "## PromptForge Convention Guardrails\n"
        f"{lines}\n\n"
        "These are operator-audited constraints from the strategy store. Treat "
        "them as hard guidance for proposal generation; do not reinterpret them "
        "as positive evidence for unrelated mutations."
    )


def _mutation_diversity_coverage_pressure(
    action: dict[str, Any],
    ctx: _ActionContext,
    *,
    k: int = 8,
) -> str | None:
    if ctx.strategy_store is None:
        return None
    if not hasattr(ctx.strategy_store, "retrieve_for_journal"):
        return None

    target = action.get("file", "")
    mutation_type = action.get("mutation", "targeted_fix")
    description = action.get("description", "")
    query = f"{target} {mutation_type} {description}".strip()
    result = diversity_coverage_penalty(
        query,
        ctx.strategy_store,
        journal=ctx.journal,
        k=k,
        species="prompt_forge",
    )
    status = str(result.get("status") or "unknown")
    if status not in {"ok", "sparse"}:
        log.debug(
            "Skipping PromptForge diversity coverage pressure: %s",
            result.get("reason") or status,
        )
        return None

    ctx.state[_MUTATION_DIVERSITY_COVERAGE_STATE_KEY] = result

    density = float(result.get("density") or 0.0)
    negative_log_density = float(result.get("negative_log_density") or 0.0)
    lines = [
        "## Diversity Coverage Pressure (AP-35/AP-36 observe-only)",
        (
            f"- strategy_density: {density:.6f}; "
            f"negative_log_density: {negative_log_density:.3f}; "
            f"nearby_strategy_count: {int(result.get('similar_count') or 0)}"
        ),
        (
            "- Use this as proposal-shaping pressure only: higher "
            "negative_log_density means this mutation target is under-covered "
            "in strategy memory. It is not an acceptance score or quality gate."
        ),
    ]
    if status == "sparse":
        lines.append("- No nearby folded strategy-memory entries were found for this target.")

    matches = result.get("top_matches")
    if isinstance(matches, list) and matches:
        lines.append("- Nearby strategy-memory entries:")
        for match in matches[:3]:
            if not isinstance(match, dict):
                continue
            source = match.get("source_trial_id")
            species = str(match.get("species") or "unknown")
            score = float(match.get("similarity_score") or 0.0)
            summary = str(match.get("description") or match.get("insight") or "").strip()
            if len(summary) > 160:
                summary = f"{summary[:157]}..."
            source_text = f"Trial #{source}" if source not in (None, "") else "strategy"
            lines.append(f"  - {source_text} ({species}) score={score:.6f}: {summary}")

    return "\n".join(lines)


def _discard_mutation_diversity_coverage(ctx: _ActionContext) -> None:
    ctx.state.pop(_MUTATION_DIVERSITY_COVERAGE_STATE_KEY, None)


def _record_mutation_diversity_coverage(
    eval_result: EvalResult,
    ctx: _ActionContext,
    *,
    artifact_kind: str,
    target: str,
    mutation_type: str,
    decision: str,
) -> None:
    coverage = ctx.state.pop(_MUTATION_DIVERSITY_COVERAGE_STATE_KEY, None)
    if not isinstance(coverage, dict):
        return

    payload = {
        "schema_version": "mutation_diversity_coverage.v1",
        "artifact_kind": artifact_kind,
        "target": target,
        "mutation_type": mutation_type,
        "decision": decision,
        "acceptance_effect": "none_observe_only",
        "status": coverage.get("status"),
        "reason": coverage.get("reason"),
        "query_text": coverage.get("query_text"),
        "density": coverage.get("density"),
        "negative_log_density": coverage.get("negative_log_density"),
        "penalty": coverage.get("penalty"),
        "similar_count": coverage.get("similar_count"),
        "top_matches": list(coverage.get("top_matches") or [])[:3],
        "interpretation": coverage.get("interpretation"),
    }
    eval_result.details["mutation_diversity_coverage"] = payload


def _seed_batch_strategy_hints(
    action: dict[str, Any],
    ctx: _ActionContext,
    *,
    k: int = 5,
) -> str | None:
    if not _PLANNER_HINTS_ENABLED or ctx.strategy_store is None:
        return None
    if not hasattr(ctx.strategy_store, "retrieve_for_journal"):
        log.warning(
            "Skipping seed_batch strategy hints: StrategyStore lacks "
            "journal-aware retrieve_for_journal()"
        )
        return None

    suites = action.get("suites") or []
    if isinstance(suites, list):
        suite_text = " ".join(str(suite) for suite in suites)
    else:
        suite_text = str(suites)
    query = f"seed_batch seeder {suite_text} n_questions={action.get('n_questions', '')}"
    try:
        strategies = ctx.strategy_store.retrieve_for_journal(
            query,
            journal=ctx.journal,
            k=k,
            species="seeder",
        )
    except TypeError:
        log.warning(
            "Skipping seed_batch strategy hints: StrategyStore "
            "retrieve_for_journal() does not support species filtering"
        )
        return None
    if not strategies:
        return None
    lines = "\n".join(_format_strategy_hint(strategy) for strategy in strategies)
    return (
        "AutoPilot planner hints for this seed batch:\n"
        f"{lines}\n\n"
        "Use these hints as operator guidance about what to sample, avoid, or "
        "watch for. Do not treat them as answer keys."
    )


def _planner_convention_bindings(
    ctx: _ActionContext,
    *,
    species: str,
) -> set[str]:
    """Return live audited convention bindings for a planner species.

    Operator-seeded convention rows carry explicit ``bind_identifiers`` after
    the identifier audit. Hard guards should use those identifiers only, not
    free-text descriptions, so future/context notes cannot disable live levers.
    """
    if not _PLANNER_HINTS_ENABLED or ctx.strategy_store is None:
        return set()
    if not hasattr(ctx.strategy_store, "retrieve_conventions"):
        log.warning(
            "Skipping %s convention bindings: StrategyStore lacks retrieve_conventions()",
            species,
        )
        return set()
    try:
        conventions = ctx.strategy_store.retrieve_conventions(
            species=species,
            journal=ctx.journal,
        )
    except TypeError:
        log.warning(
            "Skipping %s convention bindings: StrategyStore "
            "retrieve_conventions() has an incompatible signature",
            species,
        )
        return set()

    bindings: set[str] = set()
    for entry in conventions:
        metadata = getattr(entry, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            continue
        if str(metadata.get("bind_status", "")).strip().lower() != "live":
            continue
        raw_identifiers = metadata.get("bind_identifiers", [])
        if not isinstance(raw_identifiers, list):
            continue
        bindings.update(
            str(identifier).strip() for identifier in raw_identifiers if str(identifier).strip()
        )
    return bindings


def _action_seed_batch(action: dict[str, Any], ctx: _ActionContext):
    requested_n = int(action.get("n_questions", 10))
    suites = action.get("suites")

    # 2026-05-22: adaptive batch size — scale requested_n down to what
    # recent batch wall-clock suggests will fit within
    # SEEDING_BATCH_BUDGET_S (default 900s). Without this, the autopilot
    # asks for 10 questions every time and a 240s/question rate gives
    # ~40-min batches that crowd out other trials.
    try:
        import sys

        sys.path.insert(0, str(ORCH_ROOT / "scripts/benchmark"))
        from seeding_telemetry import (
            adaptive_batch_size as _adaptive_n,
            record_batch_duration as _record_batch,
        )

        adapted_n, reason = _adaptive_n(requested_n)
        if adapted_n != requested_n:
            log.warning(
                "[adaptive-batch] scaling seed_batch from %d → %d (%s)",
                requested_n,
                adapted_n,
                reason,
            )
        else:
            log.info("[adaptive-batch] keeping seed_batch n=%d (%s)", requested_n, reason)
        n = adapted_n
    except Exception as exc:
        log.warning(
            "[adaptive-batch] telemetry import failed (%s) — using requested n=%d", exc, requested_n
        )
        n = requested_n
        _record_batch = None  # type: ignore[assignment]

    import time as _time

    _batch_start = _time.perf_counter()
    # 2026-05-23 Phase 4: pass the watcher (if ctx supplies one) so the
    # seeder's per-role calls can detect exogenous service reloads. Phase 5
    # wires the watcher into ctx; for now ctx.watcher may be None (backward
    # compatible — Seeder.run_batch's watcher kwarg defaults to None).
    run_kwargs = {
        "n_questions": n,
        "suites": suites,
        "watcher": getattr(ctx, "watcher", None),
    }
    strategy_hints = _seed_batch_strategy_hints(action, ctx)
    if strategy_hints:
        log.info(
            "Seed-batch StrategyStore hints available for planner context; "
            "not injecting them into sampled question prompts."
        )
    seeder_result = ctx.seeder.run_batch(**run_kwargs)
    _batch_elapsed = _time.perf_counter() - _batch_start

    # Record duration so the next batch can adapt
    if _record_batch is not None:
        try:
            _record_batch(n, _batch_elapsed)
            log.info(
                "[adaptive-batch] recorded duration: %dq in %.0fs (%.0fs/q)",
                n,
                _batch_elapsed,
                _batch_elapsed / max(n, 1),
            )
        except Exception:
            pass

    # After seeding, run T0 eval
    eval_result = ctx.tower.hybrid_eval()

    # 2026-05-23 Phase 4 — merge seeding-phase exogenous-restart metadata
    # into the trial-level EvalResult. The seed phase happens BEFORE
    # tower.hybrid_eval; if a reload corrupted the seed phase but the
    # later eval came up clean, the EvalResult would (without this merge)
    # look entirely sound and the trial would not be tagged. Per the
    # handoff Section 5.4: aggregate counters via additive merge,
    # concatenate the per-question id list and marker log.
    if seeder_result is not None:
        eval_result.n_exogenous_recovered += seeder_result.n_exogenous_recovered
        eval_result.n_exogenous_unrecovered += seeder_result.n_exogenous_unrecovered
        eval_result.n_external_restart += seeder_result.n_external_restart
        # Avoid double-counting question ids if the eval phase also tagged
        # the same id (different question pool, but defensive).
        existing_ids = set(eval_result.exogenous_question_ids)
        for qid in seeder_result.exogenous_question_ids:
            if qid not in existing_ids:
                eval_result.exogenous_question_ids.append(qid)
                existing_ids.add(qid)
        eval_result.exogenous_marker_log.extend(seeder_result.exogenous_marker_log)

    return eval_result, "seeder"


def _action_numeric_trial(action: dict[str, Any], ctx: _ActionContext):
    surface = action.get("surface", "memrl_retrieval")
    explicit_params = _normalize_numeric_trial_params(surface, action.get("params", {}) or {})
    suppressed_surfaces = _planner_convention_bindings(ctx, species="numeric_swarm")
    suppressed_surfaces.update(suppressed_numeric_surfaces())
    if surface in suppressed_surfaces:
        return (
            SkipOutcome(
                "invalid",
                f"planner convention suppresses numeric surface: {surface}",
                "numeric_trial",
            ),
            "numeric_swarm",
        )

    # A staged candidate must survive T2/T3 replays and be exactly reversible
    # on rejection.  KV compaction mutates request-local cache state, while the
    # AP-3 role-restart surface does not yet expose a production-grade inverse;
    # neither may enter the binding promotion path.
    if _multitier_runtime_enabled() and surface in {
        "kv_compaction",
        "spec_decode_role_restart",
    }:
        return (
            SkipOutcome(
                "skipped",
                f"staged multi-tier policy requires a reversible surface: {surface}",
                "numeric_trial",
            ),
            "numeric_swarm",
        )

    if explicit_params:
        preimage = _capture_param_preimage(explicit_params) if _multitier_runtime_enabled() else None
        if preimage is not None and preimage.get("status") != "ok":
            return (
                SkipOutcome(
                    "skipped",
                    f"exact numeric restore preimage unavailable: {preimage}",
                    "numeric_trial",
                ),
                "numeric_swarm",
            )
        # Apply explicit params
        apply_result = _apply_params(explicit_params)
        if _numeric_apply_no_changes(apply_result):
            return (
                _numeric_no_change_skip(surface=surface, params=explicit_params),
                "numeric_swarm",
            )
        if apply_result.get("status") == "error":
            log.warning(
                "Skipping numeric trial eval; explicit params were not applied: %s",
                apply_result.get("errors") or apply_result,
            )
            return (
                _numeric_apply_error_skip(
                    surface=surface,
                    params=explicit_params,
                    apply_result=apply_result,
                ),
                "numeric_swarm",
            )
        action["params"] = dict(explicit_params)
        if preimage is not None:
            action["_multitier_restore_preimage"] = preimage
    else:
        # Let Optuna suggest
        trial = ctx.swarm.suggest_trial(surface)
        preimage = (
            _capture_param_preimage(dict(trial["params"]))
            if _multitier_runtime_enabled()
            else None
        )
        if preimage is not None and preimage.get("status") != "ok":
            reason = f"exact numeric restore preimage unavailable: {preimage}"
            ctx.swarm.mark_failed(surface, trial["trial_number"], reason)
            return SkipOutcome("skipped", reason, "numeric_trial"), "numeric_swarm"
        apply_result = _apply_params(trial["params"])
        if _numeric_apply_no_changes(apply_result):
            reason = "suggested params produced no live config changes"
            ctx.swarm.mark_failed(surface, trial["trial_number"], reason)
            return (
                _numeric_no_change_skip(surface=surface, params=dict(trial["params"])),
                "numeric_swarm",
            )
        if apply_result.get("status") == "error":
            reason = "; ".join(apply_result.get("errors", [])) or str(apply_result)
            if not _numeric_apply_error_is_infra(reason, apply_result):
                ctx.swarm.mark_failed(surface, trial["trial_number"], reason)
            log.warning(
                "Skipping numeric trial eval; suggested params were not applied: %s",
                reason,
            )
            return (
                _numeric_apply_error_skip(
                    surface=surface,
                    params=dict(trial["params"]),
                    apply_result=apply_result,
                ),
                "numeric_swarm",
            )
        action["params"] = dict(trial["params"])
        if preimage is not None:
            action["_multitier_restore_preimage"] = preimage
        ctx.state["_current_optuna_trial"] = {
            "surface": surface,
            "trial_number": trial["trial_number"],
        }

    eval_result = ctx.tower.hybrid_eval()
    if eval_result:
        eval_result.details.setdefault(
            "numeric_trial_applied_params", dict(action.get("params") or {})
        )
    # Report to Optuna if we have a trial. FIELD-1 tail: build the objective tuple
    # through the canonical tier_specs chokepoint (objectives_from) rather than the
    # raw EvalResult.objectives property — tier_specs is the single source of truth
    # for per-tier objective shape, and its own docstring forbids bypassing it with
    # ad-hoc tuple building at call sites. Behavior-identical for all current tiers
    # (the default spec returns the same (quality, speed, -cost, reliability) tuple).
    if "_current_optuna_trial" in ctx.state and eval_result:
        t = ctx.state.pop("_current_optuna_trial")
        # objectives_from() raises when a dominance axis is unmeasured — e.g. an
        # aborted / fast-rejected batch where seq_task_rate_qph_from() lands below
        # the 1.0 s/question validity floor and returns None. The trial has already
        # been popped, so letting that escape kills _action_numeric_trial and
        # strands an Optuna trial that is neither reported nor failed, permanently
        # occupying a slot in the study. autopilot.py guards the identical
        # condition at both of its objectives_from() call sites; this one was
        # missed. An unmeasured axis is a FAILED trial, not a crash.
        try:
            objectives = objectives_from(eval_result)
        except UnmeasuredObjectiveError:
            log.warning(
                "Numeric trial %s/%s: dominance axis unmeasured; marking the "
                "Optuna trial failed instead of reporting a fabricated objective",
                t["surface"],
                t["trial_number"],
            )
            ctx.swarm.mark_failed(t["surface"], t["trial_number"], "dominance axis unmeasured")
        else:
            ctx.swarm.report_result(t["surface"], t["trial_number"], objectives)
    return eval_result, "numeric_swarm"


def _verdict_detail(verdict: Any) -> str:
    violations = getattr(verdict, "violations", None) or []
    return "; ".join(str(v) for v in violations)[:500]


def _ledger_journal_dir(ctx: _ActionContext) -> Path | None:
    journal_dir = getattr(ctx.journal, "journal_dir", None)
    return Path(journal_dir) if isinstance(journal_dir, (str, Path)) else None


def _record_rejected_mutation(
    ctx: _ActionContext,
    *,
    mutation: Any,
    target: str,
    mutation_type: str,
    artifact_kind: str,
    rejecting_gate: str,
    eval_result: Any,
    description: str = "",
    gate_detail: str = "",
) -> None:
    """AP-53: harness-written rejection record. Fails open; never changes the trial."""
    try:
        journal_dir = _ledger_journal_dir(ctx)
        if journal_dir is None:
            return
        baseline_per_suite: dict[str, Any] | None = None
        tier = getattr(eval_result, "tier", None) if eval_result is not None else None
        baseline = getattr(ctx.gate, "baseline", None)
        if eval_result is not None and baseline is not None and hasattr(baseline, "pin_tier"):
            try:
                pin = baseline.pin_tier(int(tier or 0), register=False)
                baseline_per_suite = dict(getattr(pin, "per_suite_quality", None) or {})
            except Exception:  # noqa: BLE001 - deltas are optional context
                baseline_per_suite = None
        next_trial = getattr(ctx.journal, "next_trial_id", None)
        trial_id = ctx.state.get("trial_counter")
        if not isinstance(trial_id, int) and callable(next_trial):
            trial_id = next_trial()
        record = rejected_mutation_ledger.build_record(
            target=target,
            mutation_type=mutation_type,
            artifact_kind=artifact_kind,
            rejecting_gate=rejecting_gate,
            original_content=str(getattr(mutation, "original_content", "") or ""),
            mutated_content=str(getattr(mutation, "mutated_content", "") or ""),
            description=description,
            gate_detail=gate_detail,
            per_suite_quality=(
                getattr(eval_result, "per_suite_quality", None) if eval_result is not None else None
            ),
            baseline_per_suite_quality=baseline_per_suite,
            quality=getattr(eval_result, "quality", None) if eval_result is not None else None,
            tier=tier if isinstance(tier, int) else None,
            trial_id=trial_id if isinstance(trial_id, int) else None,
        )
        rejected_mutation_ledger.append_record(journal_dir, record)
    except Exception as exc:  # noqa: BLE001
        log.warning("AP-53 rejected-mutation record failed: %s", exc)


def _build_mutation_context(
    action: dict[str, Any],
    ctx: _ActionContext,
) -> tuple[str, dict | None]:
    """Shared failure-context + per-suite-quality assembly used by mutation handlers."""
    _discard_mutation_diversity_coverage(ctx)

    target = action.get("file", "")
    mutation_type = action.get("mutation", "targeted_fix")
    description = action.get("description", "")

    # Gather failure context from recent journal entries (AP-1)
    recent_failures = ctx.journal.recent_failures(species="prompt_forge", n=5)
    failure_context = "\n\n".join(
        f"Trial #{f.trial_id} ({f.action_type}):\n{ctx.journal.failure_analysis_for_prompt(f)}"
        for f in recent_failures
    )

    # B5: Cross-species fertilization — prepend insights from all species
    cross_insights = ctx.journal.insights_text(n=5)
    if cross_insights and cross_insights != "(no insights yet)":
        failure_context = f"## Cross-Species Insights\n{cross_insights}\n\n" + failure_context

    # B1: Strategy store retrieval — add past strategy insights
    if ctx.strategy_store is not None:
        query = f"{target} {mutation_type} {description}"
        if hasattr(ctx.strategy_store, "retrieve_for_journal"):
            strategies = ctx.strategy_store.retrieve_for_journal(
                query,
                journal=ctx.journal,
                k=3,
            )
        else:
            log.warning(
                "Skipping strategy retrieval: StrategyStore lacks journal-aware "
                "retrieve_for_journal()"
            )
            strategies = []
        if strategies:
            strategy_lines = "\n".join(
                f"- Trial #{s.source_trial_id} ({s.species}): {s.description} → {s.insight}"
                for s in strategies
            )
            failure_context = f"## Past Strategy Insights\n{strategy_lines}\n\n" + failure_context

        diversity_pressure = _mutation_diversity_coverage_pressure(action, ctx)
        if diversity_pressure:
            failure_context = f"{diversity_pressure}\n\n{failure_context}"

        convention_guardrails = _prompt_forge_convention_guardrails(ctx)
        if convention_guardrails:
            failure_context = f"{convention_guardrails}\n\n{failure_context}"

    # MH-11: Prefer structured trace IR when available. This is diagnostic
    # context only; mutation acceptance still uses the existing gates.
    critic_trace_ir_prompt = str(ctx.state.get("critic_trace_ir_prompt") or "").strip()
    if not critic_trace_ir_prompt and ctx.tower is not None:
        ir_formatter = getattr(ctx.tower, "format_critic_trace_ir", None)
        if callable(ir_formatter):
            try:
                critic_trace_ir_prompt = ir_formatter(ctx.state.get("critic_trace_ir"))
            except Exception as exc:  # trace feedback must never block mutation dispatch
                log.debug("Could not format critic trace IR: %s", exc)
                critic_trace_ir_prompt = ""

    # MH-7: Prefer labeled success/failure trace examples when available.
    contrastive_traces = ""
    if not critic_trace_ir_prompt:
        contrastive_traces = ctx.state.get("contrastive_traces", "")
    if not critic_trace_ir_prompt and not contrastive_traces and ctx.tower is not None:
        formatter = getattr(ctx.tower, "capture_contrastive_traces", None)
        if callable(formatter):
            try:
                contrastive_traces = formatter(
                    k_success=2,
                    k_failure=2,
                    trace_bank=ctx.state.get("contrastive_trace_bank"),
                )
            except Exception as exc:  # trace feedback must never block mutation dispatch
                log.debug("Could not format contrastive traces: %s", exc)
                contrastive_traces = ""
    if critic_trace_ir_prompt:
        failure_context = f"{critic_trace_ir_prompt}\n\n{failure_context}"
    elif contrastive_traces:
        failure_context = f"{contrastive_traces}\n\n{failure_context}"
    else:
        # B3 fallback: raw recent inference traces.
        last_traces = ctx.state.get("last_traces", "")
        if last_traces:
            failure_context = f"## Recent Execution Traces\n{last_traces}\n\n" + failure_context

    # AP-53: harness-written memory of rejected changes to THIS target. Placed first
    # so it survives any downstream truncation of the failure context.
    journal_dir = _ledger_journal_dir(ctx)
    if journal_dir is not None and target:
        rejected_block = rejected_mutation_ledger.render_for_prompt(journal_dir, target)
        if rejected_block:
            failure_context = f"{rejected_block}\n\n{failure_context}"

    # Get per-suite quality from most recent eval
    last_entries = ctx.journal.recent(1)
    last_per_suite = (
        last_entries[-1].eval_details.get("per_suite_quality") if last_entries else None
    )

    if mutation_type == "crossover":
        try:
            from crossover_diagnostics import crossover_context
            donor_context = crossover_context(ctx.journal, target)
            if donor_context:
                failure_context = f"{failure_context}\n\n{donor_context}"
        except Exception as exc:
            log.warning("Crossover donor context unavailable (%s)", type(exc).__name__)
    return failure_context, last_per_suite


def _autopilot_attr(name: str) -> Any | None:
    """Resolve helpers from the already-loaded autopilot module without import cycles."""
    import sys

    for module_name in ("autopilot", "scripts.autopilot.autopilot", "__main__"):
        mod = sys.modules.get(module_name)
        if mod is not None and hasattr(mod, name):
            return getattr(mod, name)
    return None


def _record_seq_action_gate(
    eval_result: EvalResult,
    *,
    applied: bool,
    reason: str = "",
    candidate: str = "",
    core_id: str = "",
) -> None:
    details = getattr(eval_result, "details", None)
    if isinstance(details, dict):
        details["seq_action_gate_check"] = {
            "enabled": True,
            "applied": applied,
            "reason": reason,
            "candidate": candidate,
            "core_id": core_id,
        }


def _action_gate_check(
    action: dict[str, Any],
    ctx: _ActionContext,
    eval_result: EvalResult,
):
    """Run the action-local revert gate, threading W4 seq inputs when available."""
    if not getattr(ctx.gate, "use_sequential", False):
        return ctx.gate.check(eval_result)

    seq_inputs_for_trial = _autopilot_attr("_seq_inputs_for_trial")
    # SEQ-B: the paired rate measurement, not the Pareto/goodput one. Must match the
    # incumbent comparator `_seq_inputs_for_trial` builds or an unchanged config scores a
    # rate regression on every trial.
    seq_task_rate_qph_from = _autopilot_attr("seq_task_rate_qph_from")
    if seq_inputs_for_trial is None or seq_task_rate_qph_from is None:
        _record_seq_action_gate(
            eval_result,
            applied=False,
            reason="autopilot_seq_helpers_unavailable",
        )
        log.warning("Sequential action gate fell back to legacy check: helpers unavailable")
        return ctx.gate.check(eval_result)
    if ctx.journal is None:
        _record_seq_action_gate(eval_result, applied=False, reason="journal_unavailable")
        log.warning("Sequential action gate fell back to legacy check: journal unavailable")
        return ctx.gate.check(eval_result)

    try:
        seq_inputs = seq_inputs_for_trial(
            journal=ctx.journal,
            action=action,
            tier=eval_result.tier,
        )
        verdict = ctx.gate.check(
            eval_result,
            question_results=list(getattr(eval_result, "question_results", []) or []),
            task_rate=seq_task_rate_qph_from(eval_result),
            baseline_profile=seq_inputs["baseline_profile"],
            baseline_task_rate=seq_inputs["baseline_task_rate"],
            prior_quality_obs=seq_inputs["prior_quality_obs"],
            prior_rate_obs=seq_inputs["prior_rate_obs"],
            candidate=seq_inputs["candidate"],
            core_id=seq_inputs["core_id"],
        )
        _record_seq_action_gate(
            eval_result,
            applied=verdict.seq is not None,
            reason="" if verdict.seq is not None else "seq_inputs_not_ready",
            candidate=seq_inputs.get("candidate", ""),
            core_id=seq_inputs.get("core_id", ""),
        )
        return verdict
    except Exception as exc:  # noqa: BLE001 - action revert gate must remain non-fatal
        _record_seq_action_gate(
            eval_result,
            applied=False,
            reason=f"seq_action_gate_error:{type(exc).__name__}",
        )
        log.warning("Sequential action gate fell back to legacy check: %s", exc)
        return ctx.gate.check(eval_result)


def _simplicity_check(
    mutation,
    eval_result,
    ctx: _ActionContext,
    *,
    kind: str,
    log_label: str,
) -> tuple[bool, str | None]:
    """AP-10 simplicity criterion. Returns (passed, deficiency_marker).

    `passed=False` means the caller should revert. `deficiency_marker` is set
    to "shrinkage" when catastrophic shrinkage is detected, else None.
    """
    orig_len = len(mutation.original_content)
    new_len = len(mutation.mutated_content)
    if orig_len <= 0:
        return True, None
    size_change = (new_len - orig_len) / orig_len
    last_quality = 0.0
    recent = ctx.journal.recent(1)
    if recent:
        last_quality = recent[-1].quality
    quality_delta = eval_result.quality - last_quality
    if size_change > 0.20 and quality_delta < 0.02:
        log.warning(
            "%s simplicity criterion: %s grew %.0f%% for %.3f quality gain, reverting",
            log_label,
            kind,
            size_change * 100,
            quality_delta,
        )
        return False, None
    if size_change < -0.50:
        log.warning(
            "%s simplicity criterion: %s shrank %.0f%% — likely destructive, reverting",
            log_label,
            kind,
            abs(size_change) * 100,
        )
        return False, "shrinkage"
    return True, None


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _skill_efficacy_without_result(ctx: _ActionContext) -> EvalResult | None:
    """Run the no-artifact arm for K-SKILL-1 when its gate is explicitly enabled."""
    if not _env_flag_enabled(_SKILL_EFFICACY_GATE_ENV):
        return None
    return ctx.tower.hybrid_eval()


def _bsv2_baseline_result(ctx: _ActionContext) -> EvalResult | None:
    """Run the pre-mutation BSV-2 baseline arm when explicitly enabled."""
    if not _env_flag_enabled(_BSV2_ACCEPT_GATE_ENV):
        return None
    return ctx.tower.hybrid_eval()


def _skill_efficacy_accepts(
    *,
    without_result: EvalResult | None,
    with_result: EvalResult,
    artifact_kind: str,
    target: str,
    mutation_type: str,
) -> bool:
    """Default-off EV-10a accept-path hook.

    ``without_result`` is the pre-mutation no-artifact arm; ``with_result`` is the
    post-mutation arm. When the env flag is off, ``without_result`` is ``None`` and
    this helper is a no-op.
    """
    if without_result is None:
        return True

    from skill_efficacy import evaluate_skill_efficacy

    verdict = evaluate_skill_efficacy(
        without_result.per_suite_quality,
        with_result.per_suite_quality,
    )
    detail = {
        "enabled": True,
        "artifact_kind": artifact_kind,
        "target": target,
        "mutation_type": mutation_type,
        "accept": verdict.accept,
        "aggregate_delta": verdict.aggregate_delta,
        "per_suite_delta": verdict.per_suite_delta,
        "regressed_suites": verdict.regressed_suites,
        "reason": verdict.reason,
        "without_per_suite_quality": dict(without_result.per_suite_quality),
        "with_per_suite_quality": dict(with_result.per_suite_quality),
    }
    with_result.details["skill_efficacy"] = detail
    if not verdict.accept:
        log.warning(
            "Skill efficacy gate rejected %s mutation on %s: %s",
            artifact_kind,
            target,
            verdict.reason,
        )
    return verdict.accept


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("Invalid integer %s=%r; using %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("Invalid float %s=%r; using %.6f", name, raw, default)
        return default


def _bsv2_eval_payload(
    result: EvalResult,
    *,
    label: str,
    artifact_kind: str,
    target: str,
    mutation_type: str,
) -> dict[str, Any]:
    details = dict(getattr(result, "details", {}) or {})
    details.setdefault("question_results", list(getattr(result, "question_results", []) or []))
    details.setdefault(
        "archive_member_id", f"bsv2:{label}:{artifact_kind}:{target}:{mutation_type}"
    )
    return {
        "tier": result.tier,
        "quality": result.quality,
        "speed": result.speed,
        "cost": result.cost,
        "reliability": result.reliability,
        "per_suite_quality": dict(result.per_suite_quality),
        "routing_distribution": dict(getattr(result, "routing_distribution", {}) or {}),
        "n_questions": getattr(result, "n_questions", 0),
        "question_results": list(getattr(result, "question_results", []) or []),
        "core_id": getattr(result, "core_id", ""),
        "avg_prompt_tokens": getattr(result, "avg_prompt_tokens", 0),
        "eval_details": details,
        "archive_member_id": details["archive_member_id"],
    }


def _bsv2_accepts(
    *,
    baseline_result: EvalResult | None,
    candidate_result: EvalResult,
    artifact_kind: str,
    target: str,
    mutation_type: str,
) -> bool:
    """Default-off BSV-2 behavior-signature accept hook.

    When enabled, the caller supplies a pre-mutation baseline eval and the
    post-mutation candidate eval. The existing paired-report backend decides
    whether the candidate is behaviorally safe to keep.
    """
    if baseline_result is None:
        return True

    detail: dict[str, Any] = {
        "enabled": True,
        "artifact_kind": artifact_kind,
        "target": target,
        "mutation_type": mutation_type,
    }
    candidate_result.details["bsv2_accept_gate"] = detail

    try:
        from bsv_paired_report import (
            DEFAULT_MIN_SHARED_QIDS,
            build_eval_result_pair_report,
        )

        report = build_eval_result_pair_report(
            _bsv2_eval_payload(
                baseline_result,
                label="baseline",
                artifact_kind=artifact_kind,
                target=target,
                mutation_type=mutation_type,
            ),
            _bsv2_eval_payload(
                candidate_result,
                label="candidate",
                artifact_kind=artifact_kind,
                target=target,
                mutation_type=mutation_type,
            ),
            baseline_label="baseline",
            candidate_label="candidate",
            min_shared_qids=_env_int(_BSV2_MIN_SHARED_QIDS_ENV, DEFAULT_MIN_SHARED_QIDS),
            max_accuracy_regression=_env_float(_BSV2_MAX_ACCURACY_REGRESSION_ENV, 0.0),
        )
    except Exception as exc:  # pragma: no cover - exact exception type is backend-owned
        detail.update(
            {
                "accept": False,
                "gate_decision": "block",
                "blockers": [f"paired report failed: {exc}"],
                "error": str(exc),
            }
        )
        log.warning(
            "BSV-2 accept gate failed closed for %s mutation on %s: %s",
            artifact_kind,
            target,
            exc,
        )
        return False

    signature_diff = dict(report.get("signature_diff") or {})
    blockers = list(report.get("blockers") or [])
    accept = report.get("gate_decision") == "pass" and signature_diff.get("severity") != "blocking"
    detail.update(
        {
            "accept": accept,
            "gate_decision": report.get("gate_decision"),
            "blockers": blockers,
            "paired_stats": report.get("paired_stats"),
            "signature_diff": signature_diff,
            "thresholds": report.get("thresholds"),
        }
    )
    if not accept:
        log.warning(
            "BSV-2 accept gate rejected %s mutation on %s: %s",
            artifact_kind,
            target,
            "; ".join(blockers) or signature_diff.get("severity") or "blocked",
        )
    return accept


def _action_prompt_mutation(action: dict[str, Any], ctx: _ActionContext):
    target = action.get("file", "frontdoor.md")
    mutation_type = action.get("mutation", "targeted_fix")
    description = action.get("description", "")
    failure_context, last_per_suite = _build_mutation_context(action, ctx)

    try:
        mutation = ctx.forge.propose_mutation(
            target_file=target,
            mutation_type=mutation_type,
            failure_context=failure_context,
            per_suite_quality=last_per_suite,
            description=description,
            **author_capture_kwargs(action, ctx),
        )
    except FileNotFoundError:
        log.warning("Prompt file not found: %s (may have been removed in refactoring)", target)
        _discard_mutation_diversity_coverage(ctx)
        return None, "prompt_forge"
    if not getattr(mutation, "safety_valid", True):
        log.warning(
            "Prompt mutation failed transfer safety, skipping: %s",
            getattr(mutation, "safety_reason", "unsafe"),
        )
        _record_rejected_mutation(
            ctx, mutation=mutation, target=target, mutation_type=mutation_type,
            artifact_kind="prompt", rejecting_gate="transfer_safety",
            eval_result=None, description=description,
            gate_detail=str(getattr(mutation, "safety_reason", "unsafe")),
        )
        _discard_mutation_diversity_coverage(ctx)
        return None, "prompt_forge"
    skill_without = _skill_efficacy_without_result(ctx)
    bsv2_baseline = _bsv2_baseline_result(ctx)
    ctx.forge.apply_mutation(mutation)
    served_path = _served_prompt_path(ctx, mutation.file)
    if served_path is not None:
        _record_served_content(ctx, [served_path], restore=_mutation_restore("prompt", mutation))
    eval_result = ctx.tower.hybrid_eval()

    # Revert if quality drops
    verdict = _action_gate_check(action, ctx, eval_result)
    if not verdict:
        _record_mutation_diversity_coverage(
            eval_result,
            ctx,
            artifact_kind="prompt",
            target=target,
            mutation_type=mutation_type,
            decision="reverted_safety_gate",
        )
        log.warning("Prompt mutation failed safety gate, reverting")
        _record_rejected_mutation(
            ctx, mutation=mutation, target=target, mutation_type=mutation_type,
            artifact_kind="prompt", rejecting_gate="safety_gate",
            eval_result=eval_result, description=description, gate_detail=_verdict_detail(verdict),
        )
        ctx.forge.revert_mutation(mutation)
        return eval_result, "prompt_forge"

    # AP-10 simplicity criterion
    passed, deficiency = _simplicity_check(
        mutation,
        eval_result,
        ctx,
        kind="prompt",
        log_label="Simplicity criterion:",
    )
    if not passed:
        _record_mutation_diversity_coverage(
            eval_result,
            ctx,
            artifact_kind="prompt",
            target=target,
            mutation_type=mutation_type,
            decision=f"reverted_simplicity:{deficiency or 'unknown'}",
        )
        _record_rejected_mutation(
            ctx, mutation=mutation, target=target, mutation_type=mutation_type,
            artifact_kind="prompt", rejecting_gate="simplicity",
            eval_result=eval_result, description=description, gate_detail=str(deficiency or ""),
        )
        ctx.forge.revert_mutation(mutation)
        if deficiency == "shrinkage":
            ctx.state["_dispatch_deficiency"] = "shrinkage"  # AP-14
        return eval_result, "prompt_forge"

    if not _skill_efficacy_accepts(
        without_result=skill_without,
        with_result=eval_result,
        artifact_kind="prompt",
        target=target,
        mutation_type=mutation_type,
    ):
        _record_mutation_diversity_coverage(
            eval_result,
            ctx,
            artifact_kind="prompt",
            target=target,
            mutation_type=mutation_type,
            decision="reverted_skill_efficacy",
        )
        _record_rejected_mutation(
            ctx, mutation=mutation, target=target, mutation_type=mutation_type,
            artifact_kind="prompt", rejecting_gate="skill_efficacy",
            eval_result=eval_result, description=description, gate_detail="",
        )
        ctx.forge.revert_mutation(mutation)
        return eval_result, "prompt_forge"

    if not _bsv2_accepts(
        baseline_result=bsv2_baseline,
        candidate_result=eval_result,
        artifact_kind="prompt",
        target=target,
        mutation_type=mutation_type,
    ):
        _record_mutation_diversity_coverage(
            eval_result,
            ctx,
            artifact_kind="prompt",
            target=target,
            mutation_type=mutation_type,
            decision="reverted_bsv2_accept_gate",
        )
        _record_rejected_mutation(
            ctx, mutation=mutation, target=target, mutation_type=mutation_type,
            artifact_kind="prompt", rejecting_gate="bsv2_accept_gate",
            eval_result=eval_result, description=description, gate_detail="",
        )
        ctx.forge.revert_mutation(mutation)
        return eval_result, "prompt_forge"

    _record_mutation_diversity_coverage(
        eval_result,
        ctx,
        artifact_kind="prompt",
        target=target,
        mutation_type=mutation_type,
        decision="kept",
    )
    # AP-7: Prompt change accepted — invalidate stale Optuna trials
    ctx.swarm.mark_epoch(f"prompt_mutation:{target}/{mutation_type}")
    return eval_result, "prompt_forge"


def _action_gepa_optimize(action: dict[str, Any], ctx: _ActionContext):
    # AP-19: GEPA evolutionary prompt optimization
    target = action.get("file", "frontdoor.md")
    max_evals = action.get("max_evals", 50)
    description = action.get("description", f"GEPA optimize {target}")

    log.info("GEPA optimize: %s (max_evals=%d)", target, max_evals)

    try:
        mutation = ctx.forge.propose_mutation(
            target_file=target,
            mutation_type="gepa",
            description=description,
            eval_tower=ctx.tower,
            gepa_max_evals=max_evals,
        )
    except FileNotFoundError:
        log.warning("Prompt file not found: %s", target)
        return None, "prompt_forge"

    # No-op mutation means GEPA failed
    if mutation.original_content == mutation.mutated_content:
        log.warning("GEPA produced no mutation for %s", target)
        eval_result = ctx.tower.hybrid_eval()
        return eval_result, "prompt_forge"

    skill_without = _skill_efficacy_without_result(ctx)
    bsv2_baseline = _bsv2_baseline_result(ctx)
    ctx.forge.apply_mutation(mutation)
    served_path = _served_prompt_path(ctx, mutation.file)
    if served_path is not None:
        _record_served_content(ctx, [served_path], restore=_mutation_restore("prompt", mutation))
    eval_result = ctx.tower.hybrid_eval()

    # Safety gate check
    verdict = _action_gate_check(action, ctx, eval_result)
    if not verdict:
        log.warning("GEPA mutation failed safety gate, reverting")
        _record_rejected_mutation(
            ctx, mutation=mutation, target=target, mutation_type="gepa",
            artifact_kind="prompt", rejecting_gate="safety_gate",
            eval_result=eval_result, description=description, gate_detail=_verdict_detail(verdict),
        )
        ctx.forge.revert_mutation(mutation)
        return eval_result, "prompt_forge"

    # AP-10 simplicity criterion
    passed, deficiency = _simplicity_check(
        mutation,
        eval_result,
        ctx,
        kind="prompt",
        log_label="GEPA",
    )
    if not passed:
        _record_rejected_mutation(
            ctx, mutation=mutation, target=target, mutation_type="gepa",
            artifact_kind="prompt", rejecting_gate="simplicity",
            eval_result=eval_result, description=description, gate_detail=str(deficiency or ""),
        )
        ctx.forge.revert_mutation(mutation)
        if deficiency == "shrinkage":
            ctx.state["_dispatch_deficiency"] = "shrinkage"
        return eval_result, "prompt_forge"

    if not _skill_efficacy_accepts(
        without_result=skill_without,
        with_result=eval_result,
        artifact_kind="prompt",
        target=target,
        mutation_type="gepa",
    ):
        _record_rejected_mutation(
            ctx, mutation=mutation, target=target, mutation_type="gepa",
            artifact_kind="prompt", rejecting_gate="skill_efficacy",
            eval_result=eval_result, description=description, gate_detail="",
        )
        ctx.forge.revert_mutation(mutation)
        return eval_result, "prompt_forge"

    if not _bsv2_accepts(
        baseline_result=bsv2_baseline,
        candidate_result=eval_result,
        artifact_kind="prompt",
        target=target,
        mutation_type="gepa",
    ):
        _record_rejected_mutation(
            ctx, mutation=mutation, target=target, mutation_type="gepa",
            artifact_kind="prompt", rejecting_gate="bsv2_accept_gate",
            eval_result=eval_result, description=description, gate_detail="",
        )
        ctx.forge.revert_mutation(mutation)
        return eval_result, "prompt_forge"

    ctx.swarm.mark_epoch(f"gepa_optimize:{target}")
    return eval_result, "prompt_forge"


def _action_code_mutation(action: dict[str, Any], ctx: _ActionContext):
    # Meta-Harness Tier 2: Python code mutation
    target = action.get("file", "")
    mutation_type = action.get("mutation", "targeted_fix")
    description = action.get("description", "")
    failure_context, last_per_suite = _build_mutation_context(action, ctx)

    try:
        mutation = ctx.forge.propose_code_mutation(
            target_file=target,
            mutation_type=mutation_type,
            failure_context=failure_context,
            per_suite_quality=last_per_suite,
            description=description,
            **author_capture_kwargs(action, ctx),
        )
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        log.error("Code mutation blocked: %s", e)
        _discard_mutation_diversity_coverage(ctx)
        return None, "prompt_forge"

    if not mutation.syntax_valid:
        log.warning("Code mutation failed syntax validation, skipping")
        _record_rejected_mutation(
            ctx, mutation=mutation, target=target, mutation_type=mutation_type,
            artifact_kind="code", rejecting_gate="syntax_validation",
            eval_result=None, description=description,
        )
        _discard_mutation_diversity_coverage(ctx)
        return None, "prompt_forge"
    if not getattr(mutation, "safety_valid", True):
        log.warning(
            "Code mutation failed transfer safety, skipping: %s",
            getattr(mutation, "safety_reason", "unsafe"),
        )
        _record_rejected_mutation(
            ctx, mutation=mutation, target=target, mutation_type=mutation_type,
            artifact_kind="code", rejecting_gate="transfer_safety",
            eval_result=None, description=description,
            gate_detail=str(getattr(mutation, "safety_reason", "unsafe")),
        )
        _discard_mutation_diversity_coverage(ctx)
        return None, "prompt_forge"
    if getattr(mutation, "mutated_content", None) == getattr(mutation, "original_content", None):
        log.warning("Code mutation produced no file changes, skipping eval")
        _discard_mutation_diversity_coverage(ctx)
        return (
            SkipOutcome(
                "skipped",
                "code_mutation produced no file changes",
                "code_mutation",
            ),
            "prompt_forge",
        )

    skill_without = _skill_efficacy_without_result(ctx)
    bsv2_baseline = _bsv2_baseline_result(ctx)
    applied = ctx.forge.apply_code_mutation(mutation)
    if isinstance(applied, dict) and applied.get("status") not in {"rejected"}:
        _record_served_content(
            ctx, [ORCH_ROOT / mutation.file], restore=_mutation_restore("code", mutation)
        )
    eval_result = ctx.tower.hybrid_eval()

    verdict = _action_gate_check(action, ctx, eval_result)
    if not verdict:
        _record_mutation_diversity_coverage(
            eval_result,
            ctx,
            artifact_kind="code",
            target=target,
            mutation_type=mutation_type,
            decision="reverted_safety_gate",
        )
        log.warning("Code mutation failed safety gate, reverting")
        _record_rejected_mutation(
            ctx, mutation=mutation, target=target, mutation_type=mutation_type,
            artifact_kind="code", rejecting_gate="safety_gate",
            eval_result=eval_result, description=description, gate_detail=_verdict_detail(verdict),
        )
        ctx.forge.revert_code_mutation(mutation)
        return eval_result, "prompt_forge"

    # AP-10 simplicity check (for code)
    passed, deficiency = _simplicity_check(
        mutation,
        eval_result,
        ctx,
        kind="code",
        log_label="Simplicity criterion:",
    )
    if not passed:
        _record_mutation_diversity_coverage(
            eval_result,
            ctx,
            artifact_kind="code",
            target=target,
            mutation_type=mutation_type,
            decision=f"reverted_simplicity:{deficiency or 'unknown'}",
        )
        _record_rejected_mutation(
            ctx, mutation=mutation, target=target, mutation_type=mutation_type,
            artifact_kind="code", rejecting_gate="simplicity",
            eval_result=eval_result, description=description, gate_detail=str(deficiency or ""),
        )
        ctx.forge.revert_code_mutation(mutation)
        if deficiency == "shrinkage":
            ctx.state["_dispatch_deficiency"] = "shrinkage"  # AP-14
        return eval_result, "prompt_forge"

    if not _skill_efficacy_accepts(
        without_result=skill_without,
        with_result=eval_result,
        artifact_kind="code",
        target=target,
        mutation_type=mutation_type,
    ):
        _record_mutation_diversity_coverage(
            eval_result,
            ctx,
            artifact_kind="code",
            target=target,
            mutation_type=mutation_type,
            decision="reverted_skill_efficacy",
        )
        _record_rejected_mutation(
            ctx, mutation=mutation, target=target, mutation_type=mutation_type,
            artifact_kind="code", rejecting_gate="skill_efficacy",
            eval_result=eval_result, description=description, gate_detail="",
        )
        ctx.forge.revert_code_mutation(mutation)
        return eval_result, "prompt_forge"

    if not _bsv2_accepts(
        baseline_result=bsv2_baseline,
        candidate_result=eval_result,
        artifact_kind="code",
        target=target,
        mutation_type=mutation_type,
    ):
        _record_mutation_diversity_coverage(
            eval_result,
            ctx,
            artifact_kind="code",
            target=target,
            mutation_type=mutation_type,
            decision="reverted_bsv2_accept_gate",
        )
        _record_rejected_mutation(
            ctx, mutation=mutation, target=target, mutation_type=mutation_type,
            artifact_kind="code", rejecting_gate="bsv2_accept_gate",
            eval_result=eval_result, description=description, gate_detail="",
        )
        ctx.forge.revert_code_mutation(mutation)
        return eval_result, "prompt_forge"

    _record_mutation_diversity_coverage(
        eval_result,
        ctx,
        artifact_kind="code",
        target=target,
        mutation_type=mutation_type,
        decision="kept",
    )
    ctx.swarm.mark_epoch(f"code_mutation:{target}/{mutation_type}")
    return eval_result, "prompt_forge"


def _action_structural_experiment(action: dict[str, Any], ctx: _ActionContext):
    flags = action.get("flags", {})
    denylisted_flags = _planner_convention_bindings(ctx, species="structural_lab")
    denied = sorted(set(flags) & denylisted_flags)
    if denied:
        return (
            SkipOutcome(
                "invalid",
                "planner convention denies feature flag(s): " + ", ".join(denied),
                "structural_experiment",
            ),
            "structural_lab",
        )

    noop_reason = _structural_noop_reason(flags, ctx.lab)
    if noop_reason:
        return (
            SkipOutcome("skipped", noop_reason, "structural_experiment"),
            "structural_lab",
        )

    validation = ctx.lab.propose_flag_experiment(flags)
    status = validation.get("status")
    if status != "valid":
        log.warning("Invalid flag experiment: %s", validation)
        # Surface the validator's reason instead of dropping it (graph_router
        # deadlock fix). The errors list carries the exact fix, e.g.
        # "graph_router feature requires specialist_routing feature".
        #
        # Map the validator status onto the SkipOutcome status so only STABLE
        # validation failures ("invalid") are blacklist-eligible. A transient
        # "error" (orchestrator unreachable, exception) must NOT be blacklisted,
        # or a momentary blip could permanently ban a perfectly valid flag.
        if status == "invalid":
            reason = "; ".join(validation.get("errors", [])) or "invalid flag experiment"
            return SkipOutcome("invalid", reason, "structural_experiment"), "structural_lab"
        reason = str(validation.get("error", "flag experiment error"))
        return SkipOutcome("skipped", reason, "structural_experiment"), "structural_lab"

    try:
        prior_flags = ctx.lab.current_flags()
    except Exception as exc:  # noqa: BLE001
        return (
            SkipOutcome(
                "skipped",
                f"live flag state unavailable before structural_experiment: {exc}",
                "structural_experiment",
            ),
            "structural_lab",
        )
    if not isinstance(prior_flags, dict):
        return (
            SkipOutcome(
                "skipped",
                "live flag state unavailable before structural_experiment",
                "structural_experiment",
            ),
            "structural_lab",
        )

    restore_flags: dict[str, bool] = {}
    missing_restore: list[str] = []
    for name in flags:
        prior_value = _coerce_bool(prior_flags.get(name))
        if prior_value is None:
            missing_restore.append(str(name))
        else:
            restore_flags[str(name)] = prior_value
    if missing_restore:
        return (
            SkipOutcome(
                "skipped",
                "refusing structural_experiment without exact flag restore snapshot: "
                + ", ".join(sorted(missing_restore)),
                "structural_experiment",
            ),
            "structural_lab",
        )

    apply_result = ctx.lab.apply_flag_experiment(flags)
    apply_attestation = apply_result.get("attestation") if isinstance(apply_result, dict) else {}
    apply_ok = (
        isinstance(apply_result, dict)
        and apply_result.get("status") == "ok"
        and isinstance(apply_attestation, dict)
        and apply_attestation.get("status") == "ok"
    )
    if not apply_ok:
        restore_result = ctx.lab.apply_flag_experiment(restore_flags)
        reason = (
            f"structural flag apply/attestation failed: {apply_result}; "
            f"restore_result={restore_result}"
        )
        return (
            SkipOutcome(
                "skipped",
                reason,
                "structural_experiment",
                bug_corrupted_by="structural_flag_apply_failure",
                bug_corrupted_reason=reason,
            ),
            "structural_lab",
        )

    eval_result = ctx.tower.hybrid_eval()
    eval_result.details.setdefault("flag_prior_values", restore_flags)
    eval_result.details.setdefault("flag_attestation", apply_result.get("attestation"))
    eval_result.details.setdefault("flag_apply_result", apply_result)

    # Revert if quality drops
    verdict = _action_gate_check(action, ctx, eval_result)
    if not verdict:
        log.warning("Structural experiment failed safety gate, reverting")
        revert_result = ctx.lab.apply_flag_experiment(restore_flags)
        eval_result.details["flag_revert_result"] = revert_result
        revert_attestation = (
            revert_result.get("attestation") if isinstance(revert_result, dict) else {}
        )
        revert_ok = (
            isinstance(revert_result, dict)
            and revert_result.get("status") == "ok"
            and isinstance(revert_attestation, dict)
            and revert_attestation.get("status") == "ok"
        )
        if not revert_ok:
            reason = (
                "structural flag revert failed after eval; trial and following "
                f"runtime state may be contaminated: {revert_result}"
            )
            setattr(eval_result, "bug_corrupted_by", "structural_flag_revert_failure")
            setattr(eval_result, "bug_corrupted_reason", reason)
            eval_result.details["flag_revert_failed"] = True
    else:
        # AP-7: Structural change accepted — invalidate stale Optuna trials
        ctx.swarm.mark_epoch(f"structural_experiment:{flags}")

    return eval_result, "structural_lab"


def _consult_gate_result_from_summary(
    summary: dict[str, Any],
    *,
    elapsed_s: float,
    tier: int,
) -> EvalResult:
    """Convert a J17 three-arm summary into a tiered EvalResult."""
    arms = summary.get("summary", {}) if isinstance(summary, dict) else {}
    gated = arms.get("gated", {}) if isinstance(arms, dict) else {}
    turns = int(gated.get("turns") or summary.get("turns_requested_per_arm") or 0)
    quality_0_1 = float(gated.get("quality") or 0.0)
    quality_0_3 = round(3.0 * quality_0_1, 4)
    elapsed_s = max(float(elapsed_s), 0.001)
    total_turns = sum(
        int(value.get("turns") or 0)
        for key, value in arms.items()
        if isinstance(value, dict) and key not in {"comparison", "gated_comparison"}
    )
    tasks_per_hour = round(3600.0 * total_turns / elapsed_s, 4) if total_turns else 0.0
    consult_calls = int(gated.get("consult_calls") or 0)
    consult_skips = int(gated.get("consult_skips") or 0)
    reruns = int(gated.get("rerun_requests") or 0)
    gate_reason_counts = gated.get("gate_reason_counts") or {}
    consult_decisions = max(1, consult_calls + consult_skips)
    cost = round(min(1.0, consult_calls / consult_decisions), 4)
    reliability = round((int(gated.get("passes") or 0) / max(1, turns)), 4)

    return EvalResult(
        tier=tier,
        quality=quality_0_3,
        speed=tasks_per_hour,
        cost=cost,
        reliability=reliability,
        per_suite_quality={"consult_gate_targeted": quality_0_3},
        per_suite_counts={"consult_gate_targeted": turns},
        n_questions=turns,
        core_id="internal_interaction_j17_targeted_gate_v1",
        # ETR-2: with zero gated turns in the summary, `quality_0_3` above is 0.0 only
        # because nothing ran — a placeholder, not a measured score. Say so explicitly so
        # SafetyGate refuses it on the quality axis instead of gating a fabricated 0.0.
        quality_measured=turns > 0,
        quality_unmeasured_reason="" if turns > 0 else "consult_gate_no_turns",
        details={
            "kind": "consult_gate_probe",
            "tier": tier,
            "summary": arms,
            "artifact_dir": summary.get("artifact_dir"),
            "rows_path": summary.get("rows_path"),
            "consult_calls": consult_calls,
            "consult_skips": consult_skips,
            "rerun_requests": reruns,
            "gate_reason_counts": gate_reason_counts
            if isinstance(gate_reason_counts, dict)
            else {},
            "quality_0_1": quality_0_1,
        },
        eval_wall_s=elapsed_s,
        speed_metric_mode="consult_gate_tasks_per_hour",
    )


def _action_consult_gate_probe(action: dict[str, Any], ctx: _ActionContext):
    task_suite = str(action.get("task_suite") or "targeted")
    turns = int(action.get("turns") or 10)
    tier = max(1, min(3, int(action.get("tier") or 3)))
    if task_suite not in {"targeted", "bep"}:
        return (
            SkipOutcome(
                "invalid",
                f"unsupported consult_gate_probe task_suite={task_suite!r}",
                "consult_gate_probe",
            ),
            "consult_gate",
        )
    turns = max(3, min(50, turns))
    script = ORCH_ROOT / "scripts" / "benchmark" / "internal_interaction_j17_live_ab.py"
    if not script.exists():
        return (
            SkipOutcome("skipped", f"missing consult gate harness: {script}", "consult_gate_probe"),
            "consult_gate",
        )
    started = datetime.now(timezone.utc)
    cmd = [
        sys.executable,
        str(script),
        "--apply",
        "--confirm-clean-window",
        "--allow-autopilot-active",
        "--task-suite",
        task_suite,
        "--turns",
        str(turns),
        "--arms",
        "baseline,consult,gated",
    ]
    timeout_s = max(900, turns * 240)
    proc = subprocess.run(
        cmd,
        cwd=str(ORCH_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    elapsed_s = (datetime.now(timezone.utc) - started).total_seconds()
    if proc.returncode != 0:
        return (
            SkipOutcome(
                "skipped",
                "consult_gate_probe harness failed "
                f"rc={proc.returncode}: {(proc.stderr or proc.stdout)[-1000:]}",
                "consult_gate_probe",
            ),
            "consult_gate",
        )

    artifact_dir: Path | None = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("wrote "):
            artifact_dir = Path(line.removeprefix("wrote ").strip())
            break
    if artifact_dir is None:
        return (
            SkipOutcome(
                "skipped",
                "consult_gate_probe did not report artifact directory",
                "consult_gate_probe",
            ),
            "consult_gate",
        )
    summary_path = artifact_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text())
    except Exception as exc:
        return (
            SkipOutcome(
                "skipped", f"consult_gate_probe summary unreadable: {exc}", "consult_gate_probe"
            ),
            "consult_gate",
        )
    summary["artifact_dir"] = str(artifact_dir)
    return _consult_gate_result_from_summary(
        summary, elapsed_s=elapsed_s, tier=tier
    ), "consult_gate"


def _structural_noop_reason(flags: dict[str, Any], lab: Any) -> str | None:
    if not flags or lab is None or not hasattr(lab, "current_flags"):
        return None
    try:
        current = lab.current_flags()
    except Exception:
        return None
    if not isinstance(current, dict) or not current:
        return None

    requested: dict[str, bool] = {}
    for name, value in flags.items():
        desired = _coerce_bool(value)
        actual = _coerce_bool(current.get(name))
        if desired is None or actual is None:
            return None
        requested[str(name)] = desired
        if actual != desired:
            return None

    rendered = ", ".join(
        f"{name}={str(value).lower()}" for name, value in sorted(requested.items())
    )
    return f"structural_experiment would not change live flag state: {rendered}"


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
    return None


def _action_structural_prune(action: dict[str, Any], ctx: _ActionContext):
    # AP-17: Block-level deletion from .md prompt files
    target = action.get("file", "")
    block_id = action.get("block", "")

    if not target or not block_id:
        log.warning("structural_prune requires 'file' and 'block'")
        return None, "structural_lab"

    # Only allow pruning .md files in prompts directory
    prompts_dir = Path(__file__).resolve().parents[2] / "orchestration" / "prompts"
    target_path = prompts_dir / target
    if not target_path.exists() or not target.endswith(".md"):
        log.warning("Prune target not found or not .md: %s", target_path)
        return None, "structural_lab"

    original_content = target_path.read_text()
    pruned_content = ctx.lab.prune_block(original_content, block_id)
    if pruned_content is None or pruned_content == original_content:
        log.warning("Block '%s' not found in %s", block_id, target)
        return None, "structural_lab"

    # Save deleted block in action for journal rollback
    deleted_lines = original_content.split("\n")
    pruned_lines = pruned_content.split("\n")
    action["_deleted_block"] = "\n".join(line for line in deleted_lines if line not in pruned_lines)

    # Apply pruning
    target_path.write_text(pruned_content)
    pre_ratio = ctx.state.get("_last_instruction_ratio", 0.0)

    eval_result = ctx.tower.hybrid_eval()

    # Acceptance: safety gate passes AND instruction_token_ratio decreased
    verdict_result = _action_gate_check(action, ctx, eval_result)
    ratio_decreased = eval_result.instruction_token_ratio < pre_ratio

    if not verdict_result or not ratio_decreased:
        reasons = []
        if not verdict_result:
            reasons.append(f"safety gate: {verdict_result.violations}")
        if not ratio_decreased:
            reasons.append(
                f"ratio not decreased: {eval_result.instruction_token_ratio:.4f} >= {pre_ratio:.4f}"
            )
        log.warning("Structural prune rejected: %s", "; ".join(reasons))
        target_path.write_text(original_content)
        return eval_result, "structural_lab"

    # Accepted — invalidate stale Optuna trials
    ctx.swarm.mark_epoch(f"structural_prune:{target}/{block_id}")
    return eval_result, "structural_lab"


def _action_train_routing_models(action: dict[str, Any], ctx: _ActionContext):
    min_mem = action.get("min_memories", 500)
    ctx.lab.checkpoint_state(
        trial_id=ctx.state.get("trial_counter", 0),
        notes="Pre-training checkpoint",
    )
    result = ctx.lab.train_routing_models(min_memories=min_mem)
    log.info("Training result: %s", result)
    eval_result = ctx.tower.hybrid_eval()
    return eval_result, "structural_lab"


def _action_distill_skillbank(action: dict[str, Any], ctx: _ActionContext):
    # Autonomous-teacher policy (operator, 2026-07-28): default is the metered
    # Claude CLI; the planned shift to the local teacher is ONE env line on the
    # supervisor — AUTOPILOT_DISTILL_TEACHER=local — no code change. An explicit
    # "teacher" in the action still wins.
    teacher = action.get("teacher") or os.environ.get("AUTOPILOT_DISTILL_TEACHER", "claude")
    categories = action.get("categories", ["routing"])
    ctx.lab.checkpoint_state(
        trial_id=ctx.state.get("trial_counter", 0),
        notes="Pre-distillation checkpoint",
    )
    result = ctx.lab.distill_skillbank(teacher=teacher, categories=categories)
    log.info("Distillation result: %s", result)
    if isinstance(result, dict) and result.get("status") == "not_available":
        return (
            SkipOutcome(
                "skipped",
                "distill_skillbank unavailable: DistillationPipeline not available",
                "distill_skillbank",
            ),
            "structural_lab",
        )
    eval_result = ctx.tower.hybrid_eval()
    return eval_result, "structural_lab"


def _action_reset_memories(action: dict[str, Any], ctx: _ActionContext):
    keep_seen = action.get("keep_seen", True)
    keep_skills = action.get("keep_skills", True)
    result = ctx.lab.reset_and_reseed(
        keep_seen=keep_seen,
        keep_skills=keep_skills,
        trial_id=ctx.state.get("trial_counter", 0),
    )
    log.info("Reset result: %s", result)
    return None, "structural_lab"


def _parse_entry_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _recent_eval_qids(
    journal: Any,
    *,
    limit: int = SEQ_PROMOTION_RECENT_QID_TRIALS,
    days: int = SEQ_PROMOTION_RECENT_QID_DAYS,
) -> set[str]:
    """Return recently seen compact eval qids for W8 promotion fresh draws."""
    if journal is None or limit <= 0:
        return set()
    try:
        if hasattr(journal, "entries_with_supersessions"):
            entries = list(journal.entries_with_supersessions())[-limit:]
        elif hasattr(journal, "all_entries"):
            entries = list(journal.all_entries())[-limit:]
        elif hasattr(journal, "recent"):
            entries = list(journal.recent(limit))
        else:
            entries = []
    except Exception:  # noqa: BLE001
        log.warning("Could not load recent journal entries for promotion eval qid exclusion")
        return set()

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, days))
    qids: set[str] = set()
    for entry in entries:
        timestamp = _parse_entry_timestamp(getattr(entry, "timestamp", None))
        # Missing timestamps are treated as recent so a malformed row cannot
        # silently re-enter a promotion draw.
        if timestamp is not None and timestamp < cutoff:
            continue
        details = getattr(entry, "eval_details", {}) or {}
        if not isinstance(details, dict):
            continue
        nested = details.get("details") or {}
        if not isinstance(nested, dict):
            nested = {}
        rows = (
            details.get("question_results")
            or nested.get("question_results")
            or getattr(entry, "question_results", None)
            or []
        )
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in ("qid", "stable_qid", "question_id", "id"):
                qid = str(row.get(key) or "").strip()
                if qid:
                    qids.add(qid)
    return qids


def _action_deep_eval(action: dict[str, Any], ctx: _ActionContext):
    tier = action.get("tier", 2)
    multitier_marker = ctx.state.pop("_multitier_candidate_validation", None)
    replay_marker = (
        multitier_marker
        if isinstance(multitier_marker, dict)
        else ctx.state.pop("_seq_promotion_candidate_replay", None)
    )
    candidate_action = replay_marker.get("action") if isinstance(replay_marker, dict) else None
    replay_detail: dict[str, Any] | None = None
    if isinstance(candidate_action, dict):
        candidate_type = str(candidate_action.get("type") or "")
        if candidate_type == "numeric_trial":
            params = candidate_action.get("params")
            if not isinstance(params, dict) or not params:
                return (
                    SkipOutcome(
                        "invalid",
                        "seq promotion candidate numeric_trial lacks replayable applied params",
                        "deep_eval",
                    ),
                    "seeder",
                )
            apply_result = _apply_params(params)
            if apply_result.get("status") == "error":
                reason = "; ".join(apply_result.get("errors", [])) or str(apply_result)
                return (
                    SkipOutcome(
                        "invalid",
                        f"seq promotion candidate params were not applied: {reason}",
                        "deep_eval",
                    ),
                    "seeder",
                )
            replay_detail = {
                "candidate_action_type": candidate_type,
                "surface": candidate_action.get("surface"),
                "applied_params": dict(params),
                "apply_result": apply_result,
            }
        elif candidate_type == "structural_experiment":
            flags = candidate_action.get("flags")
            if not isinstance(flags, dict) or not flags:
                return (
                    SkipOutcome(
                        "invalid",
                        "seq promotion candidate structural_experiment lacks replayable flags",
                        "deep_eval",
                    ),
                    "seeder",
                )
            validation = ctx.lab.propose_flag_experiment(flags)
            if validation.get("status") != "valid":
                reason = "; ".join(validation.get("errors", [])) or str(
                    validation.get("error") or validation
                )
                return (
                    SkipOutcome(
                        "invalid",
                        f"seq promotion candidate flags are invalid: {reason}",
                        "deep_eval",
                    ),
                    "seeder",
                )
            apply_result = ctx.lab.apply_flag_experiment(flags)
            replay_detail = {
                "candidate_action_type": candidate_type,
                "flags": dict(flags),
                "apply_result": apply_result,
            }
        else:
            return (
                SkipOutcome(
                    "invalid",
                    f"seq promotion candidate action is not replayable: {candidate_type or 'unknown'}",
                    "deep_eval",
                ),
                "seeder",
            )
    eval_kwargs: dict[str, Any] = {"tier": tier}
    if replay_detail is not None and not isinstance(multitier_marker, dict):
        eval_kwargs.update(
            {
                "promotion_eval": True,
                "trial_id": ctx.state.get("trial_counter"),
                "exclude_qids": _recent_eval_qids(ctx.journal),
            }
        )
    eval_result = ctx.tower.evaluate(**eval_kwargs)
    if replay_detail is not None:
        eval_result.details.setdefault("seq_promotion_candidate_replay", replay_detail)
    if isinstance(multitier_marker, dict):
        eval_result.details.setdefault(
            "multitier_candidate_validation",
            {
                "policy_version": multitier_marker.get("policy_version"),
                "stage": multitier_marker.get("stage"),
                "tier": multitier_marker.get("tier"),
                "candidate": multitier_marker.get("candidate"),
                "source_trial_id": multitier_marker.get("source_trial_id"),
                "candidate_replay": replay_detail,
            },
        )
    return eval_result, "seeder"


def _count_rollback_failure(ctx: "_ActionContext", reason: str) -> None:
    """Fix 6: bound the rollback retry loop; past the cap, stall loudly instead of spinning."""
    failures = int(ctx.state.get("multitier_rollback_failures") or 0) + 1
    ctx.state["multitier_rollback_failures"] = failures
    if failures < MULTITIER_ROLLBACK_MAX_FAILURES or ctx.state.get("multitier_rollback_stalled"):
        return
    ctx.state["multitier_rollback_stalled"] = {
        "failures": failures,
        "reason": reason[:500],
        "at": datetime.now(timezone.utc).isoformat(),
    }
    log.error(
        "MULTITIER ROLLBACK STALLED after %d failures: %s. Rollback is no longer re-forced "
        "and no new candidate is staged until an operator resolves it (remove "
        "state.multitier_rollback_stalled after restoring production state).",
        failures,
        reason,
    )
    _raise_rollback_alarm(
        ROLLBACK_STALLED_ALARM_KEY,
        "critical",
        f"AutoPilot multitier rollback failed {failures} times: {reason[:300]}. "
        "Rollback retries stopped; production state may still hold the rejected candidate.",
        {"failures": failures, "reason": reason[:300]},
    )


def _action_rollback(action: dict[str, Any], ctx: _ActionContext):
    to_cp = action.get("to_checkpoint", "production_best")
    runtime_restore: dict[str, Any] | None = None
    pending = ctx.state.get("multitier_pending_validation")
    if ctx.state.get("multitier_rollback_pending") and isinstance(pending, dict):
        candidate_action = pending.get("candidate_action") or {}
        candidate_type = str(candidate_action.get("type") or "")
        if candidate_type == "numeric_trial":
            preimage = candidate_action.get("_multitier_restore_preimage")
            runtime_restore = _restore_param_preimage(preimage) if isinstance(preimage, dict) else {
                "status": "error",
                "error": "numeric candidate lacks restore preimage",
            }
        elif candidate_type in {"prompt_mutation", "code_mutation", "gepa_optimize"}:
            runtime_restore = _restore_mutation_preimage(
                ctx, pending.get("candidate_restore"), pending.get("candidate_served_content")
            )
            if runtime_restore.get("status") == "external_change":
                # Fix 2: the file no longer holds the candidate's content, so there is
                # nothing of the candidate to undo and someone else's change must survive.
                # Touch nothing (no preimage write, no checkpoint restore) and retire the
                # candidate loudly.
                reason = "rejected_external_change: " + str(runtime_restore.get("error"))
                pending["status"] = "rejected_external_change"
                pending["blocked_reason"] = reason
                ctx.state["multitier_last_rejected"] = _public_pending_copy(pending)
                ctx.state.pop("multitier_rollback_pending", None)
                ctx.state.pop("multitier_pending_validation", None)
                ctx.state.pop("multitier_rollback_failures", None)
                ctx.state["multitier_last_event"] = {
                    "event": "rollback_skipped_external_change",
                    "candidate": pending.get("candidate"),
                    "runtime_restore": runtime_restore,
                }
                log.error(
                    "MULTITIER ROLLBACK NOT APPLIED — %s. The candidate is retired as "
                    "rejected_external_change; inspect the file and its history before "
                    "re-running anything.",
                    reason,
                )
                _raise_rollback_alarm(
                    ROLLBACK_EXTERNAL_CHANGE_ALARM_KEY,
                    "warning",
                    f"AutoPilot multitier rollback skipped: {runtime_restore.get('error')}. "
                    "The rejected mutation candidate was NOT restored because its file was "
                    "changed by someone else. Review the file, then clear this alarm.",
                    {
                        "file": runtime_restore.get("file"),
                        "served_sha256": runtime_restore.get("served_sha256"),
                        "current_sha256": runtime_restore.get("current_sha256"),
                        "candidate": pending.get("candidate"),
                    },
                )
                return SkipOutcome("skipped", reason, "rollback"), "structural_lab"
        elif candidate_type == "structural_experiment":
            restore_flags = candidate_action.get("_multitier_restore_flags")
            runtime_restore = (
                ctx.lab.apply_flag_experiment(restore_flags)
                if isinstance(restore_flags, dict) and restore_flags
                else {"status": "error", "error": "structural candidate lacks restore flags"}
            )
            attestation = (
                runtime_restore.get("attestation")
                if isinstance(runtime_restore, dict)
                else None
            )
            if (
                isinstance(runtime_restore, dict)
                and runtime_restore.get("status") == "ok"
                and (not isinstance(attestation, dict) or attestation.get("status") != "ok")
            ):
                runtime_restore = {
                    **runtime_restore,
                    "status": "error",
                    "error": "structural restore attestation failed",
                }
        else:
            runtime_restore = {
                "status": "error",
                "error": f"unsupported multitier rollback candidate type: {candidate_type}",
            }
        if runtime_restore.get("status") != "ok":
            reason = f"multitier runtime rollback failed: {runtime_restore}"
            _count_rollback_failure(ctx, reason)
            return (
                SkipOutcome(
                    "skipped",
                    reason,
                    "rollback",
                    bug_corrupted_by="multitier_rollback_failure",
                    bug_corrupted_reason=reason,
                ),
                "structural_lab",
            )
    mutation_candidate = (
        ctx.state.get("multitier_rollback_pending")
        and isinstance(pending, dict)
        and str((pending.get("candidate_action") or {}).get("type") or "")
        in {"prompt_mutation", "code_mutation", "gepa_optimize"}
    )
    # Fix 1: a mutation candidate's rollback restores only its own file (above). The
    # checkpoint's prompt copy would overwrite EVERY prompt, including unrelated operator
    # edits, so it is skipped; the rest of the checkpoint restore still runs.
    checkpoint_kwargs = {"restore_prompts": False} if mutation_candidate else {}
    if to_cp == "production_best":
        restore_result = ctx.lab.restore_checkpoint(**checkpoint_kwargs)
    else:
        restore_result = ctx.lab.restore_checkpoint(Path(to_cp), **checkpoint_kwargs)
    if mutation_candidate and isinstance(restore_result, dict) and restore_result.get("status") == "ok":
        # Attest the FINAL on-disk state, after everything the rollback wrote.
        final = _attest_mutation_restore(ctx, pending.get("candidate_restore") or {})
        runtime_restore = {**(runtime_restore or {}), "final_attestation": final}
        if final.get("status") != "ok":
            restore_result = {"status": "error", "error": final.get("error")}
    if ctx.state.get("multitier_rollback_pending"):
        if not isinstance(restore_result, dict) or restore_result.get("status") != "ok":
            reason = f"multitier rollback failed: {restore_result}"
            _count_rollback_failure(ctx, reason)
            return (
                SkipOutcome(
                    "skipped",
                    reason,
                    "rollback",
                    bug_corrupted_by="multitier_rollback_failure",
                    bug_corrupted_reason=reason,
                ),
                "structural_lab",
            )
        ctx.state.pop("multitier_rollback_pending", None)
        ctx.state.pop("multitier_pending_validation", None)
        ctx.state.pop("multitier_rollback_failures", None)
        ctx.state.pop("multitier_rollback_stalled", None)
        ctx.state["multitier_last_event"] = {
            "event": "rollback_applied",
            "result": restore_result,
            "runtime_restore": runtime_restore,
        }
    ctx.gate.reset_failures()
    eval_result = ctx.tower.hybrid_eval()
    return eval_result, "structural_lab"


def _action_distill_knowledge(action: dict[str, Any], ctx: _ActionContext):
    # Evolution Manager: knowledge distillation (no eval, no system change)
    last_n = action.get("last_n", 10)
    if ctx.evo is None or ctx.strategy_store is None:
        log.warning("distill_knowledge requires evo + strategy_store")
        return (
            SkipOutcome(
                "invalid",
                "distill_knowledge unavailable: missing evo or strategy_store",
                "distill_knowledge",
            ),
            "evolution_manager",
        )
    journal_entries = (
        ctx.journal.entries_with_supersessions()
        if hasattr(ctx.journal, "entries_with_supersessions")
        else ctx.journal.all_entries()
    )
    result = ctx.evo.distill(
        journal_entries=journal_entries,
        strategy_store=ctx.strategy_store,
        last_n=last_n,
        trial_id=ctx.state.get("trial_counter", 0),
    )
    log.info("Knowledge distillation: %s", result)
    if isinstance(result, dict) and result.get("status") == "failed":
        reason = result.get("reason") or "unknown failure"
        return (
            SkipOutcome(
                "invalid",
                f"distill_knowledge failed: {reason}",
                "distill_knowledge",
            ),
            "evolution_manager",
        )
    return None, "evolution_manager"


def _action_slot_compact(action: dict[str, Any], ctx: _ActionContext):
    # Expected Attention KV Compression: score and evict KV cache entries
    # Uses the kv_compress module for telemetry, gap guardrails, and structured results.
    port = action.get("port")
    if port is not None and (isinstance(port, bool) or not isinstance(port, int) or port <= 0):
        return (
            SkipOutcome(
                "invalid",
                "slot_compact port must be an explicit TCP port >= 1 or omitted "
                "to compact all generated production slots",
                "slot_compact",
            ),
            "slot_management",
        )

    from kv_compress import compress_slot, auto_compress_all

    slot_id = action.get("slot_id", 0)
    keep_ratio = action.get("keep_ratio", 0.5)
    keep_first = action.get("keep_first", 4)
    scorer = action.get("scorer", "expected_attention")
    layer_weights = action.get("layer_weights")
    n_future = action.get("n_future", 128)
    use_covariance = action.get("use_covariance", True)

    if port:
        # Single-port compression
        result = compress_slot(
            port=port,
            slot_id=slot_id,
            keep_ratio=keep_ratio,
            scorer=scorer,
            keep_first=keep_first,
            n_future=n_future,
            use_covariance=use_covariance,
            layer_weights=layer_weights,
        )
        if result.success:
            log.info(
                "KV compact port=%d slot=%d: evicted=%d keep=%.0f%% scorer=%s time=%.1fms",
                port,
                slot_id,
                result.n_evicted,
                keep_ratio * 100,
                scorer,
                result.elapsed_ms,
            )
        else:
            log.warning("KV compact failed on port %d: %s", port, result.error)
    else:
        # Compress all production slots
        results = auto_compress_all(
            threshold=action.get("threshold", 0.80),
            keep_ratio=keep_ratio,
            scorer=scorer,
            keep_first=keep_first,
            n_future=n_future,
            use_covariance=use_covariance,
            layer_weights=layer_weights,
        )
        for role, r in results.items():
            if r and r.success:
                log.info("KV compact %s: evicted=%d", role, r.n_evicted)

    # Evaluate quality after compaction to measure impact
    eval_result = ctx.tower.hybrid_eval()
    return eval_result, "slot_management"


# -----------------------------------------------------------------------------
# Reviewer control-plane actions (H8 AP-5). Plan-generation is default;
# execution is inference-gated. The default (and un-flagged) path enumerates the
# trial plan WITHOUT calling any backend and returns a SkipOutcome carrying the
# summary; the full plan dict is stashed on ctx.state for inspection/tests.
# Screening-tier live execution is wired through screening_tier_runner and still
# requires the documented env flag plus action dry_run=false.
# -----------------------------------------------------------------------------

_REVIEW_POLICY_TRIAL_INFERENCE_ENV = "AUTOPILOT_REVIEW_POLICY_TRIAL_INFERENCE"
_SCREENING_TIER_INFERENCE_ENV = "AUTOPILOT_SCREENING_TIER_INFERENCE"


def _action_review_policy_trial(action: dict[str, Any], ctx: _ActionContext):
    from review_policy_trials import plan_review_policy_trial

    plan, error = plan_review_policy_trial(action)
    if error is not None or plan is None:
        return (
            SkipOutcome("invalid", error or "review_policy_trial: no plan", "review_policy_trial"),
            "review_plane",
        )

    ctx.state["_review_policy_trial_plan"] = plan.to_dict()
    summary = (
        f"review_policy_trial plan: {plan.n_trials} trials over knobs "
        f"{plan.knobs} on corpus slice {plan.corpus_slice.get('corpus_id')}"
        f"/{plan.corpus_slice.get('domain')} (n={plan.corpus_slice.get('n_rows')}); "
        "execution inference-gated"
    )
    log.info("%s", summary)

    dry_run = bool(action.get("dry_run", True))
    if dry_run:
        return SkipOutcome("skipped", summary, "review_policy_trial"), "review_plane"

    if not _env_flag_enabled(_REVIEW_POLICY_TRIAL_INFERENCE_ENV):
        return (
            SkipOutcome(
                "skipped",
                "review_policy_trial live execution is inference-gated; set "
                f"{_REVIEW_POLICY_TRIAL_INFERENCE_ENV}=1 to attempt (still unimplemented). "
                f"Plan enumerated: {summary}",
                "review_policy_trial",
            ),
            "review_plane",
        )

    raise NotImplementedError(
        "review_policy_trial live eval-tower execution is not wired (H8 AP-5 is "
        f"plan-generation only). {_REVIEW_POLICY_TRIAL_INFERENCE_ENV} was set but the "
        "inference path is intentionally unimplemented under the zero-inference "
        "constraint; the enumerated plan is on ctx.state['_review_policy_trial_plan']."
    )


def _action_screening_tier_driver(action: dict[str, Any], ctx: _ActionContext):
    from review_policy_trials import (
        load_corpus_manifest,
        load_pool_gen_output,
        plan_screening_tier,
    )

    pool_gen_path = action.get("pool_gen_path")
    if not pool_gen_path:
        return (
            SkipOutcome(
                "invalid",
                "screening_tier_driver requires 'pool_gen_path' (reviewer_pool_gen.py output)",
                "screening_tier_driver",
            ),
            "review_plane",
        )
    pool_gen_output = load_pool_gen_output(Path(str(pool_gen_path)))
    if not pool_gen_output.get("pairings"):
        return (
            SkipOutcome(
                "invalid",
                f"screening_tier_driver: no pairings loadable from {pool_gen_path!r}",
                "screening_tier_driver",
            ),
            "review_plane",
        )

    corpus_path = action.get("corpus_manifest_path")
    manifest = load_corpus_manifest(Path(str(corpus_path)) if corpus_path else None)
    plan, error = plan_screening_tier(
        pool_gen_output,
        corpus_manifest=manifest,
        per_pairing_n=int(action.get("per_pairing_n", 12)),
        eval_tier=str(action.get("tier", "T0")),
        max_pairings=0,
        domain=action.get("domain"),
    )
    if error is not None or plan is None:
        return (
            SkipOutcome(
                "invalid", error or "screening_tier_driver: no plan", "screening_tier_driver"
            ),
            "review_plane",
        )

    ctx.state["_screening_tier_plan"] = plan.to_dict()
    summary = (
        f"screening_tier plan: {len(plan.queue)} pairings queued (of "
        f"{plan.pairings_considered}) at n={plan.per_pairing_n} {plan.eval_tier} "
        f"on {plan.corpus_slice.get('corpus_id')}/{plan.corpus_slice.get('domain')}; "
        "placement-queue dispatch, inference-gated"
    )
    log.info("%s", summary)

    dry_run = bool(action.get("dry_run", True))
    if dry_run:
        return SkipOutcome("skipped", summary, "screening_tier_driver"), "review_plane"

    if not _env_flag_enabled(_SCREENING_TIER_INFERENCE_ENV):
        return (
            SkipOutcome(
                "skipped",
                "screening_tier_driver live execution is inference-gated; set "
                f"{_SCREENING_TIER_INFERENCE_ENV}=1 to attempt. "
                f"Plan enumerated: {summary}",
                "screening_tier_driver",
            ),
            "review_plane",
        )

    from screening_tier_runner import run_screening_tier

    output_path_raw = action.get("results_path") or action.get("output_path")
    row_ids_path_raw = action.get("row_ids_path") or action.get("row_ids_file")
    result = run_screening_tier(
        plan.to_dict(),
        pool_gen_output,
        corpus_manifest=manifest,
        output_path=Path(str(output_path_raw)) if output_path_raw else None,
        row_ids_path=Path(str(row_ids_path_raw)) if row_ids_path_raw else None,
        cap_per_pairing=int(action.get("cap_per_pairing", 0)),
        max_pairings=int(action.get("max_pairings", 0)),
        prune_unfit=not bool(action.get("no_prune", False)),
        priority=not bool(action.get("no_priority", False)),
        seed=int(action.get("seed", 42)),
    )
    ctx.state["_screening_tier_result"] = result
    mode = str(result.get("mode") or "unknown")
    ran = bool(result.get("inference_ran"))
    n_jobs = int(result.get("n_jobs", 0) or 0)
    return (
        SkipOutcome(
            "skipped",
            f"screening_tier_driver {mode} complete: inference_ran={ran}, n_jobs={n_jobs}",
            "screening_tier_driver",
        ),
        "review_plane",
    )


# -----------------------------------------------------------------------------
# Dispatcher — maps action_type → handler
# -----------------------------------------------------------------------------

_ACTION_HANDLERS = {
    "seed_batch": _action_seed_batch,
    "numeric_trial": _action_numeric_trial,
    "prompt_mutation": _action_prompt_mutation,
    "gepa_optimize": _action_gepa_optimize,
    "code_mutation": _action_code_mutation,
    "structural_experiment": _action_structural_experiment,
    "consult_gate_probe": _action_consult_gate_probe,
    "structural_prune": _action_structural_prune,
    "train_routing_models": _action_train_routing_models,
    "distill_skillbank": _action_distill_skillbank,
    "reset_memories": _action_reset_memories,
    "deep_eval": _action_deep_eval,
    "rollback": _action_rollback,
    "distill_knowledge": _action_distill_knowledge,
    "slot_compact": _action_slot_compact,
    # Reviewer control-plane actions (H8 AP-5) — plan-generation, inference-gated.
    "review_policy_trial": _action_review_policy_trial,
    "screening_tier_driver": _action_screening_tier_driver,
}


# -----------------------------------------------------------------------------
# Dirty-tree fence — never let a file-mutating action commit (or even write on
# top of) pre-existing uncommitted work in its commit target.
#
# The forge stages differently per path, so the guard scope differs:
#   * code_mutation -> `git add <single file>`  => check that one file
#     or, for mutation=new_file, check the parent directory that will receive
#     the untracked file (the target itself does not exist yet)
#   * prompt_mutation / gepa_optimize
#                   -> `git add <prompts dir>`  => check the WHOLE prompts dir
#     (a dirty *sibling* prompt would otherwise be swept into the commit).
#   * structural_prune -> direct write to one prompt file => check that file
# Fires regardless of auto_commit (a mutation must not write over unrelated
# uncommitted work either) and fails CLOSED: any git error => treat as dirty
# and skip the mutation rather than risk committing someone else's work.
# -----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _REPO_ROOT / "orchestration" / "prompts"
_CODE_FILE_MUTATORS = {"code_mutation"}
_PROMPT_DIR_MUTATORS = {"prompt_mutation", "gepa_optimize"}
_PROMPT_FILE_MUTATORS = {"structural_prune"}


def _pathspec_pending_change_report(pathspec: Path) -> tuple[bool, str]:
    """Return (is_dirty, evidence) for pending changes under ``pathspec``.

    Fail-closed: git errors are dirty. The evidence string is intentionally
    short because it is surfaced in AutoPilot skip reasons.
    """
    path_text = str(pathspec)
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", path_text],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001 — fail closed on any git/subprocess error
        return True, f"pathspec={path_text}; git status raised {type(exc).__name__}: {exc}"
    if result.returncode != 0:
        stderr = result.stderr.strip() or "<no stderr>"
        return (
            True,
            f"pathspec={path_text}; git status rc={result.returncode}; stderr={stderr[:500]}",
        )
    status = result.stdout.strip()
    if status:
        lines = status.splitlines()
        sample = "\\n".join(lines[:8])
        if len(lines) > 8:
            sample += f"\\n... +{len(lines) - 8} more"
        return True, f"pathspec={path_text}; git status reported:\\n{sample}"
    return False, f"pathspec={path_text}; git status clean"


def _pathspec_has_pending_changes(pathspec: Path) -> bool:
    """True if ``git status --porcelain`` reports any change (modified, staged,
    or untracked) under ``pathspec``. Fail-closed: returns True on git error."""
    return _pathspec_pending_change_report(pathspec)[0]


def _mutation_dirty_target_reason(action: dict[str, Any]) -> str | None:
    """Return a skip reason if a file-mutating action would commit pre-existing
    uncommitted changes in its commit target; ``None`` if the action is
    non-mutating or its target is clean."""
    action_type = action.get("type", "")
    if action_type in _CODE_FILE_MUTATORS:
        target = action.get("file", "")
        if not target:
            return None  # missing-file is handled by the scope validator
        path = (_REPO_ROOT / target).resolve()
        pathspec = path.parent if action.get("mutation") == "new_file" else path
        is_dirty, evidence = _pathspec_pending_change_report(pathspec)
        if is_dirty:
            return (
                f"{action_type} target '{target}' has pre-existing uncommitted "
                "changes; skipping to avoid committing unrelated work "
                f"({evidence})"
            )
    elif action_type in _PROMPT_DIR_MUTATORS:
        # The prompt commit path stages the whole prompts dir, so any dirty
        # sibling prompt would be swept in — check the entire directory.
        is_dirty, evidence = _pathspec_pending_change_report(_PROMPTS_DIR)
        if is_dirty:
            return (
                f"{action_type} would stage the whole prompts dir, which has "
                "pre-existing uncommitted changes; skipping to avoid committing "
                f"unrelated work ({evidence})"
            )
    elif action_type in _PROMPT_FILE_MUTATORS:
        target = action.get("file", "")
        if not target:
            return None  # missing-file is handled by the scope validator
        path = (_PROMPTS_DIR / target).resolve()
        is_dirty, evidence = _pathspec_pending_change_report(path)
        if is_dirty:
            return (
                f"{action_type} target '{target}' has pre-existing uncommitted "
                "changes; skipping to avoid overwriting unrelated work "
                f"({evidence})"
            )
    return None


def _is_forced_seq_candidate_replay(
    action: dict[str, Any],
    state: dict[str, Any],
) -> bool:
    """Return True for the exact W8 replay action forced for the current trial."""
    marker = state.get("seq_candidate_replay_forced")
    if not isinstance(marker, dict):
        return False
    if marker.get("action") != action:
        return False
    try:
        marker_trial = int(marker.get("trial_id"))
        current_trial = int(state.get("trial_counter"))
    except (TypeError, ValueError):
        return False
    return marker_trial == current_trial


def dispatch_action(
    action: dict[str, Any],
    seeder: "Seeder",
    swarm: "NumericSwarm",
    forge: "PromptForge",
    lab: "StructuralLab",
    tower: "EvalTower",
    gate: SafetyGate,
    archive: "ParetoArchive",
    journal: "ExperimentJournal",
    state: dict[str, Any],
    strategy_store: "StrategyStore | None" = None,
    evo: "EvolutionManager | None" = None,
    watcher: Any | None = None,
    allowed_action_types: Iterable[str] | None = None,
) -> tuple[EvalResult | SkipOutcome | None, str]:
    """Execute an action and return (eval_result, species_name).

    The first element is an ``EvalResult`` for a metric-collecting trial, a
    ``SkipOutcome`` for an action that failed validation or was dropped by a
    dispatcher guard (so the main loop gets the reason as residue), or ``None``
    for a handler that ran a side effect but collected no metrics (meta no-op).

    `watcher`: OrchestratorWatcher (Phase 5). When non-None, action handlers
    that issue /chat traffic (seed_batch + variants) use it to detect operator
    reloads of the orchestrator or llama-servers and retry inline. None
    preserves pre-Phase-5 behavior exactly.
    """
    action_type = action.get("type", "")
    if allowed_action_types is not None:
        allowed = {str(item) for item in allowed_action_types if str(item)}
        if action_type not in allowed:
            reason = (
                f"action type {action_type!r} is not in the active live-loop "
                "allowlist; keep it in the shadow lane until action availability "
                "promotes it for dispatch"
            )
            log.warning("Live-loop allowlist: %s", reason)
            return SkipOutcome("skipped", reason, action_type), action_type

    if action_type == "numeric_trial" and action.get("surface") in suppressed_numeric_surfaces():
        surface = action.get("surface")
        reason = f"operator suppresses numeric surface: {surface}"
        log.warning("Numeric surface suppression: %s", reason)
        return SkipOutcome("skipped", reason, action_type), action_type

    # AP-9: Single-variable scope enforcement. Forced W8 candidate replays are
    # exact re-measurements of a journaled NumericSwarm candidate, not a new
    # planner-proposed multi-knob experiment.
    if _is_forced_seq_candidate_replay(action, state):
        log.info(
            "AP-9 scope check bypassed for forced seq candidate replay (trial=%s)",
            state.get("trial_counter"),
        )
    else:
        scope_err = validate_single_variable(action)
        if scope_err:
            log.warning("AP-9 scope violation: %s — skipping trial", scope_err)
            return SkipOutcome(
                "skipped", f"AP-9 scope violation: {scope_err}", action_type
            ), action_type
    # Dirty-tree fence (see _mutation_dirty_target_reason): a file-mutating
    # action must never commit — or write over — pre-existing uncommitted work.
    dirty_reason = _mutation_dirty_target_reason(action)
    if dirty_reason:
        log.warning("Dirty-tree fence: %s — skipping trial", dirty_reason)
        return SkipOutcome("skipped", f"Dirty-tree fence: {dirty_reason}", action_type), action_type

    log.info("Dispatching action: %s", action_type)

    handler = _ACTION_HANDLERS.get(action_type)
    if handler is None:
        log.warning("Unknown action type: %s", action_type)
        return SkipOutcome("skipped", f"unknown action type: {action_type}", action_type), "unknown"

    ctx = _ActionContext(
        seeder=seeder,
        swarm=swarm,
        forge=forge,
        lab=lab,
        tower=tower,
        gate=gate,
        archive=archive,
        journal=journal,
        state=state,
        strategy_store=strategy_store,
        evo=evo,
        watcher=watcher,
    )
    if hasattr(ctx.tower, "set_trial_context"):
        ctx.tower.set_trial_context(ctx.state.get("trial_counter"))
    return handler(action, ctx)
