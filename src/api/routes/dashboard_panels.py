"""Single source of truth for dashboard panels and their data sources.

This registry is the anti-regression core of the dashboard freshness work. It
declares, in ONE place, every operator-facing panel, the endpoint that feeds
it, and the producer file(s) whose staleness determines whether the panel is
showing current data. Two consumers read it:

  * the panel endpoints (``dashboard.py``) stamp their payload with a
    ``_freshness`` envelope built from the panel's declared sources, and
  * ``/dashboard/api/health`` + ``tests/unit/test_dashboard_panels.py`` fold
    the whole registry to assert every declared panel is fresh and every
    displayed panel has a registered, monitored source.

Historically the "which files back which panel" knowledge lived implicitly and
separately inside each endpoint, so when a producer died the affected panel
froze silently and nobody had a list to check. Centralising it means a new
panel cannot ship without declaring (and thereby monitoring) its source.

**live vs file-backed.** A panel marked ``live=True`` is recomputed from
``/proc`` / live HTTP fan-out on every request (topology, region locks,
contention) — it is fresh by construction, and its only real staleness signal
is ``generated_at`` ageing on the *client* when the transport dies. Its backing
files (stack state, contention matrix) are declared ``optional`` so they inform
but never gate. A file-backed panel (inference tap, autopilot phase/progress,
journal) freezes when its producer dies; those sources GATE the panel class.

Import-safe: depends only on ``dashboard_tap`` (stdlib-only) and the standard
library, never on ``dashboard`` — so ``dashboard`` can import it without a
cycle. The path literals are re-derived here the same way ``dashboard.py``
derives them; ``test_dashboard_panels.py`` asserts they stay identical, so the
duplication cannot drift.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.api.routes.dashboard_freshness import Source, mtime
from src.api.routes.dashboard_tap import (
    _INFERENCE_TAP_EVENTS_PATH,
    _INFERENCE_TAP_PATH,
    _REPL_TAP_PATH,
)

# --- Producer file locations (mirror dashboard.py; guarded by test) ----------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TMP_DIR = Path("/mnt/raid0/llm/tmp")
_LOG_DIR = _REPO_ROOT / "logs"

AUTOPILOT_PHASE_PATH = _TMP_DIR / "autopilot_phase.json"
AUTOPILOT_LOG_PATH = _LOG_DIR / "autopilot.log"
ORCHESTRATOR_STATE_PATH = _LOG_DIR / "orchestrator_state.json"
AUTOPILOT_STATE_PATH = _REPO_ROOT / "orchestration" / "autopilot_state.json"
AUTOPILOT_JOURNAL_PATH = _REPO_ROOT / "orchestration" / "autopilot_journal.jsonl"
CONTENTION_MATRIX_PATH = _REPO_ROOT / "orchestration" / "contention_matrix.yaml"
# G1 executor watchdog status (src/runtime/gpu_window_executor.py, cron every minute).
GPU_WINDOW_STATUS_PATH = _TMP_DIR / "gpu-window" / "mi210.json.executor-status.json"


@dataclass(frozen=True)
class SourceSpec:
    """A declared producer file for a panel, with its staleness thresholds.

    ``warn_s`` / ``stale_s`` reflect the producer's real update cadence. An
    ``optional`` source that is absent is ignored (not ``dead``) — used for
    informational backing files on live panels and for files that are
    legitimately empty until first use.
    """

    label: str
    path: Path
    warn_s: float
    stale_s: float
    optional: bool = False
    gating: bool = True  # False = informational only (reported, never flips the badge)
    # Override mtime resolution when a panel's freshness is not a single static
    # file — e.g. the journal, whose producer rotates across shards, so freshness
    # must track the NEWEST shard rather than the (frozen) base path.
    mtime_fn: "Callable[[], float | None] | None" = None

    def to_source(self) -> Source:
        m = self.mtime_fn() if self.mtime_fn is not None else mtime(self.path)
        return Source(
            label=self.label,
            mtime=m,
            warn_s=self.warn_s,
            stale_s=self.stale_s,
            optional=self.optional,
            gating=self.gating,
        )


def _latest_journal_mtime() -> float | None:
    """Newest mtime across ``autopilot_journal.jsonl`` + its ``_<n>`` rotations.

    The dashboard reads all journal shards (see
    ``dashboard._autopilot_journal_shards``); its freshness must therefore track
    whichever shard the live run is currently appending to, not the frozen base
    file — otherwise the gepa panel reports "stale" the moment the journal
    rotates even though it is being actively written.
    """
    base = AUTOPILOT_JOURNAL_PATH
    stem = base.stem
    shard_re = re.compile(rf"{re.escape(stem)}_(\d+)\.jsonl$")
    best: float | None = None
    try:
        candidates = list(base.parent.glob(f"{stem}*.jsonl"))
    except OSError:
        candidates = []
    for p in candidates:
        if p.name != base.name and not shard_re.match(p.name):
            continue
        m = mtime(p)
        if m is not None and (best is None or m > best):
            best = m
    return best


def _latest_tap_events_mtime() -> float | None:
    """Newest mtime across ``inference_tap_events.jsonl`` + its ``.<n>`` rotations.

    The tap writer rotates base → ``.1`` at 512 MB and only recreates the base
    on the NEXT append — so between rotation and the next event the base file
    does not exist and a plain mtime() would flip the panel to "dead" mid-write.
    Same bug class as the journal-rotation freeze; dot-suffix naming here.
    """
    base = _INFERENCE_TAP_EVENTS_PATH
    shard_re = re.compile(rf"{re.escape(base.name)}\.(\d+)$")
    best: float | None = None
    try:
        candidates = list(base.parent.glob(f"{base.name}*"))
    except OSError:
        candidates = []
    for p in candidates:
        if p.name != base.name and not shard_re.match(p.name):
            continue
        m = mtime(p)
        if m is not None and (best is None or m > best):
            best = m
    return best


REPO_READINESS_DIR = Path("/mnt/raid0/llm/epyc-root/data/repo_readiness")


def _latest_repo_readiness_mtime() -> float | None:
    """Newest mtime across the scorer's timestamped repo_readiness reports."""
    best: float | None = None
    try:
        candidates = list(REPO_READINESS_DIR.glob("repo_readiness_[0-9]*.json"))
    except OSError:
        candidates = []
    for p in candidates:
        m = mtime(p)
        if m is not None and (best is None or m > best):
            best = m
    return best


@dataclass(frozen=True)
class PanelSpec:
    key: str
    title: str
    endpoint: str
    mechanism: str  # "api" | "sse" | "snapshot"
    live: bool = False  # recomputed per request from /proc; fresh by construction
    sources: tuple[SourceSpec, ...] = field(default_factory=tuple)

    def live_sources(self) -> list[Source]:
        """Resolve declared sources to live-mtime :class:`Source` objects."""
        return [s.to_source() for s in self.sources]


# --- Threshold rationale ------------------------------------------------------
# tap:      structured events fire many/sec during generation; a single gen was
#           observed at ~138s, and there is setup between trials, so warn=90 /
#           stale=300 catches a DEAD producer without firing mid-generation.
# phase:    heartbeat refreshes on phase change + per eval-question; phase_status
#           uses DEFAULT_STALE_AFTER_S=900, so warn=300 / stale=900 aligns.
# state:    autopilot_state.json rewritten per trial; trials can be minutes.
# journal:  appended only on metric-bearing trials — legitimately sparse.
# log:      autopilot.log is chatty while alive.
# services/matrix: operator-curated / stack-change cadence — informational only.

PANELS: tuple[PanelSpec, ...] = (
    PanelSpec(
        key="topology",
        title="topology",
        endpoint="/dashboard/api/topology",
        mechanism="api",
        live=True,
        sources=(
            SourceSpec("stack_state", ORCHESTRATOR_STATE_PATH, 3600, 86400,
                       optional=True, gating=False),
        ),
    ),
    PanelSpec(
        key="topology_activity",
        title="topology activity overlay",
        endpoint="/dashboard/api/topology_activity",
        mechanism="api",
        live=True,
    ),
    PanelSpec(
        key="region_locks",
        title="regions lock",
        endpoint="/dashboard/api/region_locks",
        mechanism="api",
        live=True,
        sources=(
            SourceSpec("contention_matrix", CONTENTION_MATRIX_PATH, 86400, 30 * 86400,
                       optional=True, gating=False),
        ),
    ),
    PanelSpec(
        key="contention",
        title="contention gate",
        endpoint="/dashboard/api/contention",
        mechanism="api",
        live=True,
        sources=(
            SourceSpec("contention_matrix", CONTENTION_MATRIX_PATH, 86400, 30 * 86400,
                       optional=True, gating=False),
        ),
    ),
    PanelSpec(
        key="inference_tap",
        title="live inference",
        endpoint="/dashboard/api/inference_tap",
        mechanism="api",
        sources=(
            # Gating: the two live streams that back the panel. If BOTH go quiet
            # past the stale threshold while a trial should be running, the panel
            # is genuinely stale (producer died / wedged).
            SourceSpec("inference_tap", _INFERENCE_TAP_PATH, 120, 600),
            SourceSpec("structured_tap", _INFERENCE_TAP_EVENTS_PATH, 120, 600,
                       mtime_fn=_latest_tap_events_mtime),
            # Informational: legacy REPL tap is naturally old between uses —
            # reported for context, never flips the badge. (prompt_tap retired
            # 2026-07-05: orphaned file, writer removed long ago.)
            SourceSpec("repl_tap", _REPL_TAP_PATH, 300, 1800, optional=True, gating=False),
        ),
    ),
    PanelSpec(
        key="autopilot_progress",
        title="autopilot trial progress",
        endpoint="/dashboard/api/autopilot_progress",
        mechanism="api",
        sources=(
            SourceSpec("autopilot_state", AUTOPILOT_STATE_PATH, 300, 1800),
            SourceSpec("autopilot_phase", AUTOPILOT_PHASE_PATH, 300, 900),
        ),
    ),
    PanelSpec(
        key="process_status",
        title="autopilot process status",
        endpoint="/dashboard/api/process_status",
        mechanism="api",
        sources=(
            SourceSpec("autopilot_phase", AUTOPILOT_PHASE_PATH, 300, 900),
            SourceSpec("autopilot_log", AUTOPILOT_LOG_PATH, 120, 600, optional=True),
        ),
    ),
    PanelSpec(
        key="gepa",
        title="gepa progress",
        endpoint="/dashboard/api/gepa",
        mechanism="api",
        sources=(
            SourceSpec("autopilot_journal", AUTOPILOT_JOURNAL_PATH, 600, 3600,
                       mtime_fn=_latest_journal_mtime),
        ),
    ),
    PanelSpec(
        key="snapshot",
        title="live snapshot (topology + locks + activity)",
        endpoint="/dashboard/api/snapshot",
        mechanism="snapshot",
        live=True,
    ),
    # Previously-unregistered rendered panels (2026-07-05 audit): every panel
    # the page draws must be visible to /dashboard/api/health, or its producer
    # dies silently — the anti-whack-a-mole rule this registry exists for.
    PanelSpec(
        key="pareto",
        title="pareto frontier",
        endpoint="/dashboard/api/pareto",
        mechanism="api",
        sources=(
            SourceSpec("autopilot_journal", AUTOPILOT_JOURNAL_PATH, 600, 3600,
                       mtime_fn=_latest_journal_mtime),
        ),
    ),
    PanelSpec(
        key="repo_readiness",
        title="repo readiness queue",
        endpoint="/dashboard/api/repo_readiness",
        mechanism="api",
        sources=(
            # Advisory artifacts refresh on scorer/checkpoint runs (~daily);
            # age is context for the operator, never a health gate.
            SourceSpec("repo_readiness_report", REPO_READINESS_DIR, 3 * 86400, 14 * 86400,
                       optional=True, gating=False,
                       mtime_fn=_latest_repo_readiness_mtime),
        ),
    ),
    PanelSpec(
        key="optimization_brief",
        title="optimization brief",
        endpoint="/dashboard/api/optimization_brief",
        mechanism="api",
        sources=(
            # Synthesized from journal + strategy store; journal recency is
            # context (the endpoint already fails soft), not a gate.
            SourceSpec("autopilot_journal", AUTOPILOT_JOURNAL_PATH, 600, 3600,
                       optional=True, gating=False,
                       mtime_fn=_latest_journal_mtime),
        ),
    ),
    PanelSpec(
        key="decision_cockpit",
        title="decision cockpit (AP-50)",
        endpoint="/dashboard/api/decision_cockpit",
        mechanism="api",
        sources=(
            # Recomputed per request from every journal shard. Journal recency is
            # context, not a gate: AutoPilot is legitimately paused for long
            # stretches and DECLARES it (state.paused), which the payload's own
            # evidence_freshness + health fold carry. Input readability — not age —
            # is what degrades /dashboard/api/decision_cockpit/health.
            SourceSpec("autopilot_journal", AUTOPILOT_JOURNAL_PATH, 600, 3600,
                       optional=True, gating=False,
                       mtime_fn=_latest_journal_mtime),
            SourceSpec("autopilot_state", AUTOPILOT_STATE_PATH, 3600, 86400,
                       optional=True, gating=False),
        ),
    ),
    PanelSpec(
        key="gpu_window",
        title="MI210 window (G1 executor)",
        endpoint="/dashboard/api/gpu_window",
        mechanism="api",
        sources=(
            # Rewritten by the watchdog tick every minute. Optional so a branch
            # without the cron installed does not redden the global fold; the
            # panel's own probe (/dashboard/api/gpu_window/health) reports absence
            # as 503 "watchdog not running".
            SourceSpec("gpu_window_executor_status", GPU_WINDOW_STATUS_PATH, 120, 300,
                       optional=True),
        ),
    ),
    PanelSpec(
        key="insight_graph",
        title="insight graph",
        endpoint="/dashboard/api/insight_graph",
        mechanism="api",
        live=True,
    ),
    PanelSpec(
        key="build_rev",
        title="build revision",
        endpoint="/dashboard/api/version",
        mechanism="api",
        live=True,
    ),
)

PANELS_BY_KEY: dict[str, PanelSpec] = {p.key: p for p in PANELS}


def panel(key: str) -> PanelSpec:
    return PANELS_BY_KEY[key]
