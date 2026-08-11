#!/usr/bin/env python3
"""controls.py — the five-control harness, and the calibration block that feeds statistics.

WHY THIS MODULE EXISTS
----------------------
`P-AK-SEARCH-1` "Controls — four mandatory, plus one accept-side control run under a
declared contract" is the only part of the protocol whose subject is **the gate
itself** rather than a candidate. Everything else in the evaluator asks "is this
candidate good?"; this module asks "is the instrument that answers that question
still working?".

Four of the five test the gate's ability to **reject**. The fifth exists because
nothing tested its ability to **accept**, and the failure that motivated it is on
the record: AutoPilot ran **1,055 trials of which 8 were even of a promotable
type**, and 0 of 121 refutations came from futility rather than budget, while every
surface reported *"active, blockers: []"* (design §2.5 row 5, §12, decision AK-D27).
A gate that has quietly stopped passing anything is indistinguishable from an
exhausted search surface unless something known-good is replayed through it. That
is control 5, and *"a failure to promote is a gate defect, not a research finding"*.

WHICH PROTOCOL CLAUSES THIS FILE IMPLEMENTS
-------------------------------------------
`measurement/protocols/kernel-research.md` (Annex K, P-AK-SEARCH-1, RATIFIED
2026-08-03), by section name:

  * **"Controls — four mandatory, plus one accept-side control run under a declared
    contract"** — `CONTROL_DEFINITIONS` (the five, as data), `ControlHarness`,
    `ControlOutcome`, `ControlPanelResult`, `GateDefectFinding`, and the
    unavailable branch (`resolve_historical_win_replay`, `OperatorEscalation`).
    *"Control definitions, fixtures, expected directions, and seeds live inside the
    evaluator bundle under the measurement trust boundary and MUST NOT be modified
    by any process inside the loop"* is `ControlBundle`'s hash pin, not a comment.
  * **"Campaign calibration block — every threshold is derived, none is supplied"**
    — the CONTROL SIDE of it only: `run_calibration_block()` hands the A/A and
    neutral material to `statistics.solve_calibration()`, and
    `neutral_dispersion_check()` reads control 2's dispersion-vs-`phi` verdict back
    out so the neutral control's evaluator can consume it. The solve order, `phi`,
    `B_min`, the alpha budgets and the anchor-gate band are `statistics.py`'s and
    are NOT re-implemented here.
  * **"What voids a run"** — a failing A/A control reaches `api`'s
    `VOID_AA_CONTROL_FAILED` through `ControlPanelResult.panel`; a post-hoc change
    to the control definitions or the campaign's declared control bindings is
    detected by `verify_control_definitions()` and `ControlBundle.reverify()`.
  * **"Correctness precedence"** — the degraded-negative control is checked by
    calling `Verdict.rank_key()` and treating a *successful return* as the control
    failing. It tests the exact call a ranking loop makes.

Design context: `handoffs/active/autokernel-research-loop.md` §9.2 (statistical
machinery), §15.2 (five controls, and the acceptance criteria quoted verbatim in
`CONTROL_REQUIREMENTS`), §12 (failure/abuse rows), decision AK-D27.

WHAT THIS MODULE IS NOT
-----------------------
It runs NO control. It launches no inference, no benchmark and no build; it starts,
stops and signals no process; it writes no file. Controls are *observed* through the
`ControlRunner` seam and this module evaluates the observations.
`audit_no_write_or_process_paths()` proves that from this module's own AST, reusing
`api`'s auditor rather than forking it.

It also implements NO statistics and NO calibration solve. `statistics.py` owns both
(`solve_calibration`, `estimate_noise_floor`, `neutral_control_consistency`,
`resampled_crossing_rate`, `solve_mde`, `anchor_gate_band`), and this module calls
through to it. An earlier draft of this file carried its own solver behind an
estimator seam; that was a second source of truth for one protocol clause, and the
two would have drifted invisibly because both would have kept returning an accepted
calibration. One solve order, in the module whose subject is statistics.
"""
from __future__ import annotations

import math
from dataclasses import InitVar, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

from .. import schemas, storage
from . import api
from . import statistics as ak_statistics

__all__ = [
    # identity
    "CONTROL_IDS", "MANDATORY_CONTROL_IDS", "ACCEPT_SIDE_CONTROL_ID",
    "CONTROL_DEFINITIONS", "CONTROL_DEFINITIONS_DIGEST", "CONTROL_PREDICATES_DIGEST",
    "HISTORICAL_REPLAY_UNAVAILABLE",
    # errors
    "ControlsError", "ControlBundleDrift", "ControlWiringError", "ControlPanelForged",
    # definitions and the bundle
    "ControlDefinition", "SeedRotationSchedule", "AACadence", "ControlBundle",
    "resolve_control_bundle", "verify_control_definitions", "derive_control_seed",
    # A/A scheduling
    "AALedgerEntry", "AADueDecision", "AAScheduler",
    # historical-win replay contract
    "ReferenceBand", "HistoricalWinReplayDeclaration", "HistoricalWinResolution",
    "OperatorEscalation", "OPERATOR_DECISIONS", "resolve_historical_win_replay",
    # harness
    "ControlObservation", "ControlContext", "ControlRunContext", "ControlRunner",
    "ControlOutcome", "GateDefectFinding", "ControlPanelResult", "ControlHarness",
    "DISPOSITIONS",
    # the seam into api.WindowAttestations
    "window_control_attestations",
    # calibration (owned by statistics.py; this module supplies the material)
    "CALIBRATION_OWNER", "run_calibration_block", "neutral_dispersion_check",
    # audit
    "audit_no_write_or_process_paths",
]


# =============================================================================
# Errors — every one is a refusal, never a degraded result
# =============================================================================

class ControlsError(api.EvaluatorError):
    """Base class for every refusal this module makes."""


class ControlBundleDrift(ControlsError):
    """The control definitions do not hash to the value the campaign pinned.

    *"The controller MUST NOT modify this protocol, the evaluator bundle, the
    control definitions, the campaign objective, the calibrated thresholds, or any
    scoring contract."* This is how that is detected rather than trusted: the
    definitions are re-hashed at resolve time and compared against the pin, so a
    rebound module attribute, an edited literal, or a substituted bundle all fail
    closed at the same place.
    """


class ControlWiringError(ControlsError):
    """The harness was wired wrongly — a missing runner, a runner that answered for
    the wrong control, an observation of the wrong type.

    These raise rather than becoming findings: they are defects in the evaluator,
    not facts about the gate.
    """


class ControlPanelForged(ControlsError):
    """A `ControlPanelResult` was built by a route other than a control sweep, or
    its outcomes do not follow from the observations it carries.

    `ControlPanelResult.may_rank` is a LICENCE: it is the object the rest of the
    loop consults before ranking anything. Until this existed, one built by hand
    with five PASS `ControlOutcome`s answered `may_rank=True` with no control
    ever run — the same hole `api.Verdict` closes by re-deriving its status and
    refusing a stamped one, in the object that authorises ranking rather than the
    object that reports one candidate's score.

    Two locks, deliberately the same two `api.Verdict` uses:

      1. the module-private mint token, which only `ControlHarness.evaluate()`
         passes — this stops the ACCIDENT, and nothing more. It is not a
         capability: `_PANEL_MINT_TOKEN` is reachable by name from any module in
         the process, and `dataclasses.replace(result, mint=_PANEL_MINT_TOKEN)`
         constructs one. Lock 2 is the load-bearing one; and
      2. re-derivation of `outcomes`, `panel` and `blocked_reason` from the
         `observations` and `context` stored on the very same object, through the
         live `_EVALUATORS` — this stops the determined caller. Reaching in for
         the token buys nothing, because every PASS must now follow from an
         observation, and an observation that RAN must carry an `api.Verdict`,
         which only `api.compute_verdict()` mints. A fabricated all-PASS panel
         therefore requires fabricating five real verdicts first, which is the
         wall `api` already built.
    """


#: The mint token. Module-private, and never exported: `ControlPanelResult` is
#: constructible only by the harness that ran the sweep.
_PANEL_MINT_TOKEN = object()


# =============================================================================
# The five controls — data the actor cannot alter
# =============================================================================

CONTROL_POSITIVE = "positive"
CONTROL_NEUTRAL = "neutral"
CONTROL_DEGRADED_NEGATIVE = "degraded_negative"
CONTROL_AA = "aa"
CONTROL_HISTORICAL_WIN_REPLAY = "historical_win_replay"

#: Ordinal order, the protocol's own.
CONTROL_IDS = (
    CONTROL_POSITIVE, CONTROL_NEUTRAL, CONTROL_DEGRADED_NEGATIVE,
    CONTROL_AA, CONTROL_HISTORICAL_WIN_REPLAY,
)
MANDATORY_CONTROL_IDS = CONTROL_IDS[:4]
ACCEPT_SIDE_CONTROL_ID = CONTROL_HISTORICAL_WIN_REPLAY

#: Re-exported, never redefined: `api` owns the marker string that lands in the
#: record grammar, and two spellings of it would be two markers.
HISTORICAL_REPLAY_UNAVAILABLE = api.HISTORICAL_REPLAY_UNAVAILABLE

#: `api.ControlPanel`'s field name for each control. The panel is the object the
#: rest of the evaluator reads; this mapping is the only place the two vocabularies
#: meet.
_PANEL_FIELD_BY_CONTROL = {
    CONTROL_POSITIVE: "positive",
    CONTROL_NEUTRAL: "neutral",
    CONTROL_DEGRADED_NEGATIVE: "degraded_negative",
    CONTROL_AA: "aa",
    CONTROL_HISTORICAL_WIN_REPLAY: "historical_replay",
}

# What a failure of each control disposes of. These are the protocol's own
# consequences, not this module's policy.
DISPOSITION_SATISFIED = "satisfied"
DISPOSITION_GATE_DEFECT = "gate_defect"
DISPOSITION_BLOCKS_RANKING = "blocks_ranking"
DISPOSITION_VOIDS_WINDOW = "voids_window"
DISPOSITION_NOT_RUN = "not_run"
DISPOSITION_UNAVAILABLE_RECORDED = "unavailable_recorded"

DISPOSITIONS = (
    DISPOSITION_SATISFIED, DISPOSITION_GATE_DEFECT, DISPOSITION_BLOCKS_RANKING,
    DISPOSITION_VOIDS_WINDOW, DISPOSITION_NOT_RUN, DISPOSITION_UNAVAILABLE_RECORDED,
)

TESTS_REJECT = "reject"
TESTS_ACCEPT = "accept"


def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected a non-empty string, got {value!r}")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{label}: must be finite, got {value!r}")
    return float(value)


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label}: expected a positive int, got {value!r}")
    return value


@dataclass(frozen=True)
class ControlDefinition:
    """One control, as pure data.

    Every field is JSON-canonicalizable, because the definitions' content hash is
    what makes them tamper-evident. The *predicate* that evaluates a control lives
    in a module-level dispatch table and deliberately NOT on this object: a callable
    cannot be content-hashed, so a definition carrying its own evaluator would be a
    definition the pin does not cover.
    """

    control_id: str
    ordinal: int
    mandatory: bool
    tests_gate_ability_to: str
    purpose: str            # the protocol's own words for what the control IS
    requirement: str        # the protocol's own words for what it MUST do
    failure_disposition: str
    fixture_id: str
    expected_direction: str
    required_tiers: tuple = ()

    def __post_init__(self) -> None:
        if self.control_id not in CONTROL_IDS:
            raise ValueError(f"control_id {self.control_id!r} is not one of {list(CONTROL_IDS)}")
        _require_positive_int(self.ordinal, "control.ordinal")
        if not isinstance(self.mandatory, bool):
            raise TypeError("control.mandatory must be a bool")
        if self.tests_gate_ability_to not in (TESTS_REJECT, TESTS_ACCEPT):
            raise ValueError("control.tests_gate_ability_to must be 'reject' or 'accept'")
        for name in ("purpose", "requirement", "fixture_id", "expected_direction"):
            _require_nonempty_str(getattr(self, name), f"control.{name}")
        if self.failure_disposition not in (
                DISPOSITION_GATE_DEFECT, DISPOSITION_BLOCKS_RANKING, DISPOSITION_VOIDS_WINDOW):
            raise ValueError(
                f"control.failure_disposition {self.failure_disposition!r} must be one of "
                f"{[DISPOSITION_GATE_DEFECT, DISPOSITION_BLOCKS_RANKING, DISPOSITION_VOIDS_WINDOW]}")
        if not isinstance(self.required_tiers, tuple):
            raise TypeError("control.required_tiers must be a tuple")
        for tier in self.required_tiers:
            api.admit_tier(tier)

    def to_dict(self) -> dict:
        return {
            "control_id": self.control_id,
            "ordinal": self.ordinal,
            "mandatory": self.mandatory,
            "tests_gate_ability_to": self.tests_gate_ability_to,
            "purpose": self.purpose,
            "requirement": self.requirement,
            "failure_disposition": self.failure_disposition,
            "fixture_id": self.fixture_id,
            "expected_direction": self.expected_direction,
            "required_tiers": list(self.required_tiers),
        }


#: The five. `purpose` and `requirement` are the ratified text and §15.2's
#: acceptance text, kept verbatim so a journaled control result is auditable
#: against the protocol rather than against a paraphrase of it.
CONTROL_DEFINITIONS = (
    ControlDefinition(
        control_id=CONTROL_POSITIVE,
        ordinal=1,
        mandatory=True,
        tests_gate_ability_to=TESTS_REJECT,
        purpose="a known-correct optimization with a real, bounded mechanism",
        requirement="MUST rank above the anchor. Failure is a gate defect.",
        failure_disposition=DISPOSITION_GATE_DEFECT,
        fixture_id="ak3-control-positive/v1",
        expected_direction="ranks_above_anchor",
        # §15.2: "positive ranks above anchor under T1".
        required_tiers=("T1", "T1a", "T1b", "T1c"),
    ),
    ControlDefinition(
        control_id=CONTROL_NEUTRAL,
        ordinal=2,
        mandatory=True,
        tests_gate_ability_to=TESTS_REJECT,
        purpose="a correct change whose true effect is centred on zero",
        requirement=("MUST NOT advance, and its dispersion is checked against the "
                     "calibrated floor."),
        failure_disposition=DISPOSITION_BLOCKS_RANKING,
        fixture_id="ak3-control-neutral/v1",
        expected_direction="centred_on_zero",
    ),
    ControlDefinition(
        control_id=CONTROL_DEGRADED_NEGATIVE,
        ordinal=3,
        mandatory=True,
        tests_gate_ability_to=TESTS_REJECT,
        purpose=("deliberately fast-looking but wrong: cheating, silently falling back, "
                 "reducing work, or serving a cached result"),
        requirement="MUST receive no speed rank at all.",
        failure_disposition=DISPOSITION_BLOCKS_RANKING,
        fixture_id="ak3-control-degraded-negative/v1",
        expected_direction="no_speed_rank",
    ),
    ControlDefinition(
        control_id=CONTROL_AA,
        ordinal=4,
        mandatory=True,
        tests_gate_ability_to=TESTS_REJECT,
        purpose="the anchor measured against itself, through the full candidate pipeline",
        requirement=("Runs periodically on its declared cadence, not once per campaign: "
                     "it calibrates the false-positive rate and it is what detects host "
                     "drift mid-campaign. A failing A/A VOIDS the enclosing measurement "
                     "window."),
        failure_disposition=DISPOSITION_VOIDS_WINDOW,
        fixture_id="ak3-control-aa/v1",
        expected_direction="no_significant_effect",
    ),
    ControlDefinition(
        control_id=CONTROL_HISTORICAL_WIN_REPLAY,
        ordinal=5,
        mandatory=False,
        tests_gate_ability_to=TESTS_ACCEPT,
        purpose=("the accept-side control: a real past improvement, replayed end-to-end "
                 "through T0-T2 under a declared contract"),
        requirement=("it MUST promote. A failure to promote is a gate defect, not a "
                     "research finding: it halts the campaign and is escalated to the "
                     "operator."),
        failure_disposition=DISPOSITION_GATE_DEFECT,
        fixture_id="ak3-control-historical-win-replay/v1",
        expected_direction="must_promote",
    ),
)

def _definitions_payload(definitions: Sequence[ControlDefinition]) -> dict:
    return {
        "protocol": api.PROTOCOL_VERSIONED_ID,
        "ratified_utc": api.PROTOCOL_RATIFIED_UTC,
        "definitions": [d.to_dict() for d in definitions],
    }


def _current_definitions_digest() -> str:
    """Re-hash whatever `CONTROL_DEFINITIONS` currently is. Never cached."""
    return schemas.content_hash(_definitions_payload(CONTROL_DEFINITIONS))


#: The digest at import time, published so a campaign manifest can pin it.
#: It is NOT the baseline this module verifies against — see
#: `_make_control_definitions_verifier`: an actor that can rebind
#: `CONTROL_DEFINITIONS` can rebind a module constant next to it just as easily,
#: so the baseline the verifier compares against is captured in a closure cell at
#: import and is not reachable by attribute assignment.
CONTROL_DEFINITIONS_DIGEST = _current_definitions_digest()


def _predicates_payload() -> dict:
    """The identity of the code that decides whether each control passed.

    The five definitions are pure data and hash cleanly. The five *predicates* do
    not — and they are the operative half of a control definition, because a
    control whose predicate returns PASS unconditionally has been modified just as
    surely as one whose `requirement` string was edited. `_EVALUATORS` is a plain
    module-level dict: rebinding one entry is a strictly easier move than rebinding
    `CONTROL_DEFINITIONS`, and until this payload existed it was invisible to every
    digest. Function identity plus bytecode is the closest thing to a content hash
    a callable has.
    """
    entries = []
    for control_id, predicate in sorted(_EVALUATORS.items()):
        code = getattr(predicate, "__code__", None)
        entries.append({
            "control_id": control_id,
            "module": getattr(predicate, "__module__", None),
            "qualname": getattr(predicate, "__qualname__", None),
            # A callable with no bytecode (a bound object, a C function, a Mock)
            # cannot be hashed this way; naming its type is what makes the
            # substitution visible rather than silently equal to every other one.
            "code_sha256": (schemas.content_hash({"co_code": code.co_code.hex()})
                            if code is not None else None),
            "callable_type": None if code is not None else type(predicate).__name__,
        })
    return {"predicates": entries}


def _current_predicates_digest() -> str:
    """Re-hash whatever `_EVALUATORS` currently holds. Never cached."""
    return schemas.content_hash(_predicates_payload())


def _make_control_definitions_verifier(_baseline_definitions: str,
                                       _baseline_predicates: str):
    """Build the verifier around baselines held in closure cells, not attributes."""

    def verify_control_definitions(pinned_digest: Optional[str] = None) -> schemas.Check:
        """Re-derive the definitions digest and compare it against the pin.

        Called at window open AND window close. `pinned_digest=None` compares
        against the import-time baselines only, which catches in-process tampering
        but NOT a campaign whose manifest pinned a different bundle — so a caller
        with a manifest MUST pass its pin, and `resolve_control_bundle()` requires
        one.

        The predicate table is re-hashed on the same call. *"Control definitions,
        fixtures, expected directions, and seeds live inside the evaluator bundle
        under the measurement trust boundary and MUST NOT be modified by any
        process inside the loop"* covers the code that evaluates a control, not
        only the data that describes it.
        """
        current = _current_definitions_digest()
        current_predicates = _current_predicates_digest()
        reasons = []
        if current != _baseline_definitions:
            reasons.append(
                f"control definitions changed in-process: import-time digest "
                f"{_baseline_definitions[:12]}, current {current[:12]}")
        if current_predicates != _baseline_predicates:
            reasons.append(
                f"the control EVALUATORS changed in-process: import-time digest "
                f"{_baseline_predicates[:12]}, current {current_predicates[:12]}; a "
                "control whose predicate was substituted has been modified as surely "
                "as one whose definition was edited")
        if pinned_digest is not None:
            if not isinstance(pinned_digest, str) or not pinned_digest.strip():
                return schemas.Check(
                    schemas.COULD_NOT_CHECK,
                    ("a pinned control-definitions digest was supplied but is not a "
                     "usable string; an unreadable pin is not a satisfied pin",))
            if current != pinned_digest:
                reasons.append(
                    f"control definitions do not match the campaign pin: pinned "
                    f"{pinned_digest[:12]}, resolved {current[:12]}")
        if reasons:
            return schemas.Check(schemas.FAIL, tuple(reasons))
        return schemas.Check(schemas.PASS)

    return verify_control_definitions


#: Installed at the bottom of the module, once `_EVALUATORS` exists. Declared here
#: so every reader finds the contract where the constant is defined.
verify_control_definitions = None  # type: ignore[assignment]


# =============================================================================
# Seed rotation — "a never-rotated holdout is an evaluator coverage defect"
# =============================================================================

@dataclass(frozen=True)
class SeedRotationSchedule:
    """The declared rotation schedule for control seeds and confirmation shapes.

    *"Confirmation shapes and control seeds rotate on the schedule declared in the
    evaluator bundle."* `rotate_every_windows` must be finite and positive: an
    unbounded rotation interval is a schedule that never rotates, and design §12
    calls a never-rotated holdout an evaluator coverage defect rather than a
    tolerable simplification.
    """

    rotate_every_windows: int
    declared_at: str

    def __post_init__(self) -> None:
        _require_positive_int(self.rotate_every_windows, "seed_rotation.rotate_every_windows")
        _require_nonempty_str(self.declared_at, "seed_rotation.declared_at")

    def epoch_for(self, windows_completed: int) -> int:
        if isinstance(windows_completed, bool) or not isinstance(windows_completed, int) \
                or windows_completed < 0:
            raise ValueError("windows_completed must be a non-negative int")
        return windows_completed // self.rotate_every_windows

    def check_rotation(self, *, windows_completed: int, last_rotation_epoch: int) -> schemas.Check:
        """FAIL when the schedule says rotate and the recorded epoch has not moved."""
        if isinstance(last_rotation_epoch, bool) or not isinstance(last_rotation_epoch, int) \
                or last_rotation_epoch < 0:
            raise ValueError("last_rotation_epoch must be a non-negative int")
        due = self.epoch_for(windows_completed)
        if last_rotation_epoch < due:
            return schemas.Check(
                schemas.FAIL,
                (f"control seeds are at rotation epoch {last_rotation_epoch} but the "
                 f"declared schedule (every {self.rotate_every_windows} windows) is at "
                 f"epoch {due} after {windows_completed} windows; a never-rotated holdout "
                 f"is an evaluator coverage defect, not a tolerable simplification",))
        if last_rotation_epoch > due:
            return schemas.Check(
                schemas.FAIL,
                (f"control seeds are at rotation epoch {last_rotation_epoch}, ahead of the "
                 f"declared schedule's epoch {due}; rotating off-schedule is a post-hoc "
                 f"change to a declared rule",))
        return schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"rotate_every_windows": self.rotate_every_windows,
                "declared_at": self.declared_at}


def derive_control_seed(*, campaign_seed: str, control_id: str, epoch: int) -> str:
    """Deterministically derive one control's seed for one rotation epoch.

    Derived, not stored, so a seed cannot be chosen after seeing a result. Uses
    `schemas.content_hash` rather than a private hashing scheme so the derivation is
    reproducible by anything that can canonicalize JSON.
    """
    _require_nonempty_str(campaign_seed, "campaign_seed")
    if control_id not in CONTROL_IDS:
        raise ValueError(f"control_id {control_id!r} is not one of {list(CONTROL_IDS)}")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative int")
    return schemas.content_hash({
        "derivation": "ak3-control-seed/v1",
        "campaign_seed": campaign_seed,
        "control_id": control_id,
        "epoch": epoch,
    })


# =============================================================================
# A/A cadence — the periodic scheduling contract
# =============================================================================

@dataclass(frozen=True)
class AACadence:
    """The declared cadence of the A/A control.

    *"Runs periodically on its declared cadence, not once per campaign: it
    calibrates the false-positive rate and it is what detects host drift
    mid-campaign."* Both triggers are required and both are checked: a
    window-count-only cadence misses drift during a long single window, and a
    wall-clock-only cadence misses drift across a burst of short ones.

    `at_campaign_boundary` and `on_anchor_identity_change` are not options. The
    calibration clause says *"Every output is recomputed at each campaign boundary
    and whenever anchor identity changes"*, and φ — a calibration output — comes
    from this control, so a cadence that declined either would license using a φ
    measured against a different anchor.
    """

    every_n_windows: int
    every_n_seconds: float
    declared_at: str
    at_campaign_boundary: bool = True
    on_anchor_identity_change: bool = True

    def __post_init__(self) -> None:
        _require_positive_int(self.every_n_windows, "aa_cadence.every_n_windows")
        seconds = _require_finite(self.every_n_seconds, "aa_cadence.every_n_seconds")
        if seconds <= 0:
            raise ValueError("aa_cadence.every_n_seconds must be strictly positive; an "
                             "unbounded interval is a cadence that never fires")
        _require_nonempty_str(self.declared_at, "aa_cadence.declared_at")
        for name in ("at_campaign_boundary", "on_anchor_identity_change"):
            if getattr(self, name) is not True:
                raise ValueError(
                    f"aa_cadence.{name} must be True: the calibration outputs are "
                    "recomputed at each campaign boundary and whenever anchor identity "
                    "changes, and phi is derived from this control")

    def to_dict(self) -> dict:
        return {
            "every_n_windows": self.every_n_windows,
            "every_n_seconds": self.every_n_seconds,
            "declared_at": self.declared_at,
            "at_campaign_boundary": self.at_campaign_boundary,
            "on_anchor_identity_change": self.on_anchor_identity_change,
        }


@dataclass(frozen=True)
class AALedgerEntry:
    """One recorded A/A run. The ledger is the campaign's A/A history."""

    window_id: str
    ran_at_epoch_seconds: float
    windows_completed_at_run: int
    outcome: str
    anchor_short: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.window_id, "aa_ledger.window_id")
        _require_finite(self.ran_at_epoch_seconds, "aa_ledger.ran_at_epoch_seconds")
        if isinstance(self.windows_completed_at_run, bool) \
                or not isinstance(self.windows_completed_at_run, int) \
                or self.windows_completed_at_run < 0:
            raise ValueError("aa_ledger.windows_completed_at_run must be a non-negative int")
        if self.outcome not in (schemas.PASS, schemas.FAIL, schemas.COULD_NOT_CHECK):
            raise ValueError(f"aa_ledger.outcome {self.outcome!r} is not a Check outcome")
        _require_nonempty_str(self.anchor_short, "aa_ledger.anchor_short")

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "ran_at_epoch_seconds": self.ran_at_epoch_seconds,
            "windows_completed_at_run": self.windows_completed_at_run,
            "outcome": self.outcome,
            "anchor_short": self.anchor_short,
        }


@dataclass(frozen=True)
class AADueDecision:
    """Whether A/A is due, and every reason it is. Never a bare bool."""

    due: bool
    reasons: tuple
    windows_since_last: Optional[int]
    seconds_since_last: Optional[float]

    def to_dict(self) -> dict:
        return {"due": self.due, "reasons": list(self.reasons),
                "windows_since_last": self.windows_since_last,
                "seconds_since_last": self.seconds_since_last}


class AAScheduler:
    """The periodic scheduling contract for the A/A control.

    Two questions, deliberately separate:

      * `due()` — should A/A run before the next candidate window opens? This is a
        SCHEDULING answer and it is permissive about history: an empty ledger means
        due, not broken.
      * `check()` — may this window rank, given the A/A history? This is the
        `WindowAttestations.aa_cadence` attestation, and it is fail-closed: an A/A
        that has never run, is overdue, ran against a different anchor, or last
        returned anything but PASS makes it non-PASS.

    `drift_exposure()` reports the windows measured since the last PASSING A/A. It
    is ADVISORY and labelled so. The protocol voids *"the enclosing measurement
    window"*; extending that backwards is a controller/operator decision on the
    record, and this module reports the exposure rather than quietly widening a
    void it was not given the authority to widen.
    """

    def __init__(self, cadence: AACadence) -> None:
        if not isinstance(cadence, AACadence):
            raise TypeError("cadence must be an AACadence")
        self.cadence = cadence

    @staticmethod
    def _validate_ledger(ledger: Any) -> tuple:
        if isinstance(ledger, (str, bytes)) or not isinstance(ledger, (list, tuple)):
            raise TypeError("aa ledger must be a sequence of AALedgerEntry")
        for entry in ledger:
            if not isinstance(entry, AALedgerEntry):
                raise TypeError(
                    f"aa ledger entries must be AALedgerEntry, got {type(entry).__name__}; "
                    "an unreadable ledger is a wiring defect, not a cadence finding")
        return tuple(ledger)

    def due(self, *, ledger: Sequence[AALedgerEntry], windows_completed: int,
            now_epoch_seconds: float, anchor_short: str,
            campaign_boundary: bool = False) -> AADueDecision:
        entries = self._validate_ledger(ledger)
        _require_finite(now_epoch_seconds, "now_epoch_seconds")
        _require_nonempty_str(anchor_short, "anchor_short")
        if isinstance(windows_completed, bool) or not isinstance(windows_completed, int) \
                or windows_completed < 0:
            raise ValueError("windows_completed must be a non-negative int")

        if not entries:
            return AADueDecision(
                due=True,
                reasons=("no A/A control has run in this campaign; the A/A control runs "
                         "periodically on its declared cadence, not once per campaign",),
                windows_since_last=None, seconds_since_last=None)

        last = entries[-1]
        windows_since = windows_completed - last.windows_completed_at_run
        seconds_since = now_epoch_seconds - last.ran_at_epoch_seconds
        reasons = []
        if windows_since < 0 or seconds_since < 0:
            # The ledger is ahead of the counters it is being compared against, so
            # "how long since the last A/A" has no answer. Scheduling errs toward
            # running one; `check()` refuses to attest instead of reading the
            # negative interval as comfortably inside the cadence.
            reasons.append(
                f"the last A/A ({last.window_id}) is recorded ahead of this window "
                f"({windows_since} windows, {seconds_since:g}s); the A/A history cannot "
                f"be reconciled with the current window")
        if windows_since >= self.cadence.every_n_windows:
            reasons.append(
                f"{windows_since} windows since the last A/A, cadence is every "
                f"{self.cadence.every_n_windows}")
        if seconds_since >= self.cadence.every_n_seconds:
            reasons.append(
                f"{seconds_since:g}s since the last A/A, cadence is every "
                f"{self.cadence.every_n_seconds:g}s")
        if last.anchor_short != anchor_short:
            reasons.append(
                f"the last A/A ran against anchor {last.anchor_short}, this window's "
                f"anchor is {anchor_short}; a rebuilt anchor is a different anchor and "
                f"every calibration output is recomputed when anchor identity changes")
        if campaign_boundary and self.cadence.at_campaign_boundary:
            reasons.append("campaign boundary: every calibration output is recomputed here")
        if last.outcome != schemas.PASS:
            reasons.append(
                f"the last A/A returned {last.outcome}; a non-PASS A/A does not calibrate "
                f"anything and the next one is due immediately")
        return AADueDecision(due=bool(reasons), reasons=tuple(reasons),
                             windows_since_last=windows_since,
                             seconds_since_last=seconds_since)

    def check(self, *, ledger: Sequence[AALedgerEntry], windows_completed: int,
              now_epoch_seconds: float, anchor_short: str,
              campaign_boundary: bool = False) -> schemas.Check:
        """The `WindowAttestations.aa_cadence` attestation. Fail-closed.

        `campaign_boundary` is forwarded to `due()`. It was previously accepted by
        `due()` only, so the cadence's mandatory `at_campaign_boundary` trigger —
        *"Every output is recomputed at each campaign boundary"* — could not be
        expressed to the attestation that actually gates ranking.
        """
        entries = self._validate_ledger(ledger)
        if not entries:
            return schemas.Check(
                schemas.FAIL,
                ("no A/A control has run in this campaign, so there is no passing A/A "
                 "within its declared cadence; search-grade requires 'a passing A/A "
                 "control within its declared cadence'",))
        decision = self.due(ledger=entries, windows_completed=windows_completed,
                            now_epoch_seconds=now_epoch_seconds, anchor_short=anchor_short,
                            campaign_boundary=campaign_boundary)
        last = entries[-1]
        if last.outcome == schemas.COULD_NOT_CHECK:
            return schemas.Check(
                schemas.COULD_NOT_CHECK,
                (f"the last A/A ({last.window_id}) returned COULD_NOT_CHECK; the "
                 "false-positive rate is uncalibrated for this window, which is not the "
                 "same as a failed A/A and is not recorded as one",))
        if last.outcome == schemas.FAIL:
            # Checked before the reconcilability branch below: a failing A/A voids
            # the window whatever the ledger's counters say.
            return schemas.Check(
                schemas.FAIL,
                (f"the last A/A ({last.window_id}) FAILED; a failing A/A voids the "
                 "enclosing measurement window",))
        if windows_completed - last.windows_completed_at_run < 0 \
                or now_epoch_seconds - last.ran_at_epoch_seconds < 0:
            return schemas.Check(
                schemas.COULD_NOT_CHECK,
                (f"the last A/A ({last.window_id}) is recorded ahead of this window, so "
                 "whether a passing A/A ran within its declared cadence has no answer "
                 "here; an unreadable A/A history is not a satisfied cadence",))
        if decision.due:
            return schemas.Check(schemas.FAIL, decision.reasons)
        return schemas.Check(schemas.PASS)

    def drift_exposure(self, *, ledger: Sequence[AALedgerEntry],
                       windows_completed: int) -> dict:
        """Windows measured since the last PASSING A/A. Advisory, and labelled so."""
        entries = self._validate_ledger(ledger)
        if isinstance(windows_completed, bool) or not isinstance(windows_completed, int) \
                or windows_completed < 0:
            raise ValueError("windows_completed must be a non-negative int")
        last_pass = None
        for entry in entries:
            if entry.outcome == schemas.PASS:
                last_pass = entry
        if last_pass is None:
            exposed = windows_completed
            since = None
        else:
            exposed = max(0, windows_completed - last_pass.windows_completed_at_run)
            since = last_pass.window_id
        return {
            "label": "advisory",
            "authority_note": ("the protocol voids the ENCLOSING measurement window; "
                               "widening the void to the windows below is an operator "
                               "decision taken on the record, not this module's"),
            "windows_since_last_passing_aa": exposed,
            "last_passing_aa_window_id": since,
        }


# =============================================================================
# Control 5 — the historical-win replay contract
# =============================================================================

@dataclass(frozen=True)
class ReferenceBand:
    """The declared reference magnitude band for the replayed win."""

    low: float
    high: float

    def __post_init__(self) -> None:
        low = _require_finite(self.low, "reference_band.low")
        high = _require_finite(self.high, "reference_band.high")
        if low < 0:
            raise ValueError("reference_band.low must be non-negative; the band is over "
                             "|effect| and the direction is declared separately")
        if low >= high:
            raise ValueError(f"reference_band must be (low < high), got ({low}, {high})")

    def contains(self, magnitude: float) -> bool:
        value = _require_finite(magnitude, "magnitude")
        return self.low <= value <= self.high

    def to_dict(self) -> dict:
        return {"low": self.low, "high": self.high}


@dataclass(frozen=True)
class HistoricalWinReplayDeclaration:
    """The manifest's `historical_win_replay` entry.

    Exactly the protocol's fields: *"{win_id, backend, phase, reference direction,
    reference magnitude band, in-repo evidence locator, durability class}"*. The
    durability class is `schemas.DURABILITY_CLASSES` (§3.7), and the locator is
    resolved through `storage.verify_durability`, which is the module that already
    implements *"does not resolve in-repo per MEASUREMENT.md:146-156"*.
    """

    win_id: str
    backend: str
    phase: str
    reference_direction: str
    reference_band: ReferenceBand
    evidence_locator: str
    durability_class: str
    evidence_sha256: Optional[str] = None
    evidence_provenance: Optional[str] = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.win_id, "historical_win_replay.win_id")
        if self.backend not in schemas.BACKENDS:
            raise ValueError(f"historical_win_replay.backend {self.backend!r} is not one of "
                             f"{sorted(schemas.BACKENDS)}")
        _require_nonempty_str(self.phase, "historical_win_replay.phase")
        if self.reference_direction not in schemas.METRIC_DIRECTIONS:
            raise ValueError(
                f"historical_win_replay.reference_direction {self.reference_direction!r} "
                f"is not one of {sorted(schemas.METRIC_DIRECTIONS)}")
        if not isinstance(self.reference_band, ReferenceBand):
            raise TypeError("historical_win_replay.reference_band must be a ReferenceBand")
        _require_nonempty_str(self.evidence_locator, "historical_win_replay.evidence_locator")
        if self.durability_class not in schemas.DURABILITY_CLASSES:
            raise ValueError(
                f"historical_win_replay.durability_class {self.durability_class!r} is not "
                f"one of {sorted(schemas.DURABILITY_CLASSES)}; an unclassified citation "
                "cannot distinguish a defect from an expected absence")

    @classmethod
    def parse(cls, obj: Any):
        """Return `(declaration_or_None, reasons)`. Never raises on bad manifest input."""
        if not isinstance(obj, Mapping):
            return None, (f"historical_win_replay: expected a mapping, got "
                          f"{type(obj).__name__}",)
        band = obj.get("reference_band")
        try:
            if isinstance(band, Mapping):
                band = ReferenceBand(low=band.get("low"), high=band.get("high"))
            elif isinstance(band, (list, tuple)) and len(band) == 2:
                band = ReferenceBand(low=band[0], high=band[1])
            return cls(
                win_id=obj.get("win_id"),
                backend=obj.get("backend"),
                phase=obj.get("phase"),
                reference_direction=obj.get("reference_direction"),
                reference_band=band,
                evidence_locator=obj.get("evidence_locator"),
                durability_class=obj.get("durability_class"),
                evidence_sha256=obj.get("evidence_sha256"),
                evidence_provenance=obj.get("evidence_provenance"),
            ), ()
        except (ValueError, TypeError) as exc:
            return None, (str(exc),)

    def citation(self) -> dict:
        """The `storage.verify_durability` citation shape for this entry."""
        citation = {"path": self.evidence_locator, "durability_class": self.durability_class}
        if self.evidence_sha256 is not None:
            citation["sha256"] = self.evidence_sha256
        if self.evidence_provenance is not None:
            citation["provenance"] = self.evidence_provenance
        return citation

    def to_dict(self) -> dict:
        return {
            "win_id": self.win_id,
            "backend": self.backend,
            "phase": self.phase,
            "reference_direction": self.reference_direction,
            "reference_band": self.reference_band.to_dict(),
            "evidence_locator": self.evidence_locator,
            "durability_class": self.durability_class,
            "evidence_sha256": self.evidence_sha256,
            "evidence_provenance": self.evidence_provenance,
        }


OPERATOR_DECISION_PENDING = "pending"
OPERATOR_DECISION_PROCEED_ON_FOUR = "proceed_on_four_controls"
OPERATOR_DECISION_HALT = "halt"
OPERATOR_DECISIONS = (
    OPERATOR_DECISION_PENDING, OPERATOR_DECISION_PROCEED_ON_FOUR, OPERATOR_DECISION_HALT,
)


@dataclass(frozen=True)
class OperatorEscalation:
    """The operator's disposition of an unavailable control 5.

    *"Whether the campaign proceeds on four controls is the operator's call, taken
    once, on the record — not the controller's."* `pending` is a real state and it
    blocks: a campaign that escalated and did not wait has not had the call taken.
    A decided escalation MUST name who decided and when, because "taken once, on the
    record" is unverifiable otherwise.
    """

    escalation_ref: str
    raised_at: str
    decision: str
    decided_at: Optional[str] = None
    decided_by: Optional[str] = None
    note: str = ""

    def __post_init__(self) -> None:
        _require_nonempty_str(self.escalation_ref, "operator_escalation.escalation_ref")
        _require_nonempty_str(self.raised_at, "operator_escalation.raised_at")
        if self.decision not in OPERATOR_DECISIONS:
            raise ValueError(f"operator_escalation.decision {self.decision!r} is not one of "
                             f"{list(OPERATOR_DECISIONS)}")
        if self.decision != OPERATOR_DECISION_PENDING:
            for name in ("decided_at", "decided_by"):
                value = getattr(self, name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"operator_escalation.{name} is required once a decision is taken; "
                        "'the operator's call, taken once, on the record' is not on the "
                        "record without who took it and when")

    def to_dict(self) -> dict:
        return {
            "escalation_ref": self.escalation_ref,
            "raised_at": self.raised_at,
            "decision": self.decision,
            "decided_at": self.decided_at,
            "decided_by": self.decided_by,
            "note": self.note,
        }


@dataclass(frozen=True)
class HistoricalWinResolution:
    """The result of resolving the accept-side control's declared contract."""

    backend: str
    available: bool
    declaration: Optional[HistoricalWinReplayDeclaration]
    check: schemas.Check
    marker: Optional[str] = None
    durability_outcome: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.check, schemas.Check):
            raise TypeError("historical resolution check must be a schemas.Check")
        if self.available and self.declaration is None:
            raise ValueError("an available historical-win replay must carry its declaration")
        if not self.available and self.marker != HISTORICAL_REPLAY_UNAVAILABLE:
            raise ValueError(
                "an unavailable historical-win replay MUST carry the "
                f"{HISTORICAL_REPLAY_UNAVAILABLE} marker; the unavailable branch is "
                "normative, not a silent skip")

    def reason(self) -> str:
        return "; ".join(self.check.reasons) if self.check.reasons else ""

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "available": self.available,
            "declaration": None if self.declaration is None else self.declaration.to_dict(),
            "outcome": self.check.outcome,
            "reasons": list(self.check.reasons),
            "marker": self.marker,
            "durability_outcome": self.durability_outcome,
        }


def resolve_historical_win_replay(
    *,
    declarations: Sequence[HistoricalWinReplayDeclaration],
    backend: str,
    tracked_index: storage.TrackedIndex,
) -> HistoricalWinResolution:
    """Resolve the manifest's control-5 entry for `backend` at run time.

    Unavailable in exactly the protocol's two cases — *"no entry, or an entry whose
    evidence locator does not resolve in-repo per `MEASUREMENT.md:146-156`"* — plus
    the case where the answer cannot be obtained, which is reported as
    COULD_NOT_CHECK and is unavailable-but-not-failed. Fail closed, never conflated.

    `tracked_index` is REQUIRED and there is no default. Deciding "does this
    evidence resolve in-repo?" without an index would mean guessing, and
    `storage.TrackedIndex` exists precisely because a tracked file misreported as
    untracked is the worst wrong answer available here.
    """
    if backend not in schemas.BACKENDS:
        raise ValueError(f"backend {backend!r} is not one of {sorted(schemas.BACKENDS)}")
    if not isinstance(tracked_index, storage.TrackedIndex):
        raise ControlsError(
            "resolve_historical_win_replay requires a storage.TrackedIndex; without one "
            "'the evidence locator resolves in-repo' cannot be answered, and a default "
            "that answered it anyway would bless evidence git does not carry")
    if isinstance(declarations, (str, bytes)) or not isinstance(declarations, (list, tuple)):
        raise TypeError("declarations must be a sequence of HistoricalWinReplayDeclaration")
    for decl in declarations:
        if not isinstance(decl, HistoricalWinReplayDeclaration):
            raise TypeError("declarations must all be HistoricalWinReplayDeclaration; parse "
                            "manifest mappings with HistoricalWinReplayDeclaration.parse()")

    matching = [d for d in declarations if d.backend == backend]
    if not matching:
        return HistoricalWinResolution(
            backend=backend, available=False, declaration=None,
            marker=HISTORICAL_REPLAY_UNAVAILABLE,
            check=schemas.Check(
                schemas.FAIL,
                (f"the campaign manifest declares no historical_win_replay entry for "
                 f"backend {backend!r}; this backend has no qualifying durable win",)))
    if len(matching) > 1:
        return HistoricalWinResolution(
            backend=backend, available=False, declaration=None,
            marker=HISTORICAL_REPLAY_UNAVAILABLE,
            check=schemas.Check(
                schemas.FAIL,
                (f"the campaign manifest declares {len(matching)} historical_win_replay "
                 f"entries for backend {backend!r}: "
                 f"{[d.win_id for d in matching]}; the contract names one supplier, and "
                 "choosing among several would be the evaluator selecting its own "
                 "accept-side test",)))

    declaration = matching[0]
    verdicts = storage.verify_durability([declaration.citation()],
                                         tracked_index=tracked_index)
    verdict = verdicts[0]
    if verdict.outcome == schemas.PASS:
        return HistoricalWinResolution(
            backend=backend, available=True, declaration=declaration,
            durability_outcome=verdict.outcome,
            check=schemas.Check(
                schemas.PASS,
                (f"historical win {declaration.win_id!r} resolves in-repo as "
                 f"{declaration.durability_class}",)))

    return HistoricalWinResolution(
        backend=backend, available=False, declaration=declaration,
        marker=HISTORICAL_REPLAY_UNAVAILABLE, durability_outcome=verdict.outcome,
        check=schemas.Check(
            verdict.outcome,
            (f"historical win {declaration.win_id!r} declares evidence locator "
             f"{declaration.evidence_locator!r}, which does not resolve in-repo "
             f"({verdict.outcome})",) + tuple(verdict.check.reasons)))


# =============================================================================
# The bundle — definitions plus the campaign's declared bindings, hash-pinned
# =============================================================================

@dataclass(frozen=True)
class ControlBundle:
    """The resolved control bundle: the five definitions plus the campaign's
    declared cadence, rotation schedule and control-5 suppliers.

    Two digests, because two different things must be tamper-evident:

      * `definitions_digest` — the five definitions and the tightening
        constructions. Fixed by the evaluator bundle. If it moves, the control
        definitions were modified, which the protocol forbids outright.
      * `campaign_digest` — the above PLUS the campaign's declared cadence,
        rotation schedule and control-5 declarations. If it moves mid-campaign,
        that is *"any post-hoc change to … the control definitions"* and the
        affected records are void.

    `__post_init__` RE-DERIVES both from the bundle's own contents, so a bundle
    carrying a digest that does not describe it is refused at construction. Same
    shape as `api.Verdict`: the stored value must be the one the object's own
    contents imply.
    """

    definitions: tuple
    definitions_digest: str
    campaign_digest: str
    aa_cadence: AACadence
    seed_rotation: SeedRotationSchedule
    historical_win_replays: tuple
    source_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.definitions, tuple) or not self.definitions:
            raise ValueError("bundle.definitions must be a non-empty tuple")
        for defn in self.definitions:
            if not isinstance(defn, ControlDefinition):
                raise TypeError("bundle.definitions must all be ControlDefinition")
        ids = tuple(d.control_id for d in self.definitions)
        if ids != CONTROL_IDS:
            raise ValueError(
                f"bundle must carry exactly {list(CONTROL_IDS)} in ordinal order, got "
                f"{list(ids)}; 'a campaign that cannot run controls 1-4 MUST NOT rank any "
                "candidate', so a bundle missing one is refused rather than run")
        if not isinstance(self.aa_cadence, AACadence):
            raise TypeError("bundle.aa_cadence must be an AACadence")
        if not isinstance(self.seed_rotation, SeedRotationSchedule):
            raise TypeError("bundle.seed_rotation must be a SeedRotationSchedule")
        if not isinstance(self.historical_win_replays, tuple):
            raise TypeError("bundle.historical_win_replays must be a tuple")
        for decl in self.historical_win_replays:
            if not isinstance(decl, HistoricalWinReplayDeclaration):
                raise TypeError("bundle.historical_win_replays must all be "
                                "HistoricalWinReplayDeclaration")
        _require_nonempty_str(self.source_label, "bundle.source_label")

        derived_defs = schemas.content_hash(_definitions_payload(self.definitions))
        if self.definitions_digest != derived_defs:
            raise ControlBundleDrift(
                f"bundle.definitions_digest {self.definitions_digest!r} does not describe "
                f"the definitions this bundle carries (derived {derived_defs!r}); a digest "
                "is derived from the contents, never supplied alongside them")
        derived_campaign = schemas.content_hash(self._campaign_payload(derived_defs))
        if self.campaign_digest != derived_campaign:
            raise ControlBundleDrift(
                f"bundle.campaign_digest {self.campaign_digest!r} does not describe this "
                f"bundle's declared bindings (derived {derived_campaign!r})")

    def _campaign_payload(self, definitions_digest: str) -> dict:
        return {
            "definitions_digest": definitions_digest,
            "aa_cadence": self.aa_cadence.to_dict(),
            "seed_rotation": self.seed_rotation.to_dict(),
            "historical_win_replays": [d.to_dict() for d in self.historical_win_replays],
        }

    def definition(self, control_id: str) -> ControlDefinition:
        for defn in self.definitions:
            if defn.control_id == control_id:
                return defn
        raise KeyError(f"no control definition for {control_id!r}")

    def scheduler(self) -> AAScheduler:
        return AAScheduler(self.aa_cadence)

    def seed_for(self, *, campaign_seed: str, control_id: str, windows_completed: int) -> str:
        epoch = self.seed_rotation.epoch_for(windows_completed)
        return derive_control_seed(campaign_seed=campaign_seed, control_id=control_id,
                                   epoch=epoch)

    def reverify(self, *, pinned_definitions_digest: str,
                 pinned_campaign_digest: Optional[str] = None) -> schemas.Check:
        """Window-close re-verification. Same pins, re-derived from live contents."""
        checks = [verify_control_definitions(pinned_definitions_digest)]
        if pinned_campaign_digest is not None:
            derived = schemas.content_hash(
                self._campaign_payload(schemas.content_hash(
                    _definitions_payload(self.definitions))))
            if derived != pinned_campaign_digest:
                checks.append(schemas.Check(
                    schemas.FAIL,
                    (f"campaign control bindings drifted: pinned "
                     f"{pinned_campaign_digest[:12]}, resolved {derived[:12]}; this is a "
                     "post-hoc change to the control definitions and voids every affected "
                     "record",)))
        outcome = schemas.PASS
        reasons: list = []
        for chk in checks:
            if chk.outcome == schemas.PASS:
                continue
            reasons.extend(chk.reasons)
            if chk.outcome == schemas.FAIL:
                outcome = schemas.FAIL
            elif outcome != schemas.FAIL:
                outcome = schemas.COULD_NOT_CHECK
        return schemas.Check(outcome, tuple(reasons))

    def to_dict(self) -> dict:
        return {
            "definitions": [d.to_dict() for d in self.definitions],
            "definitions_digest": self.definitions_digest,
            "campaign_digest": self.campaign_digest,
            "aa_cadence": self.aa_cadence.to_dict(),
            "seed_rotation": self.seed_rotation.to_dict(),
            "historical_win_replays": [d.to_dict() for d in self.historical_win_replays],
            "source_label": self.source_label,
        }


def resolve_control_bundle(*,
                           pinned_definitions_digest: str,
                           aa_cadence: AACadence,
                           seed_rotation: SeedRotationSchedule,
                           historical_win_replays: Sequence[HistoricalWinReplayDeclaration] = (),
                           source_label: str,
                           pinned_campaign_digest: Optional[str] = None) -> ControlBundle:
    """Resolve the hash-pinned control bundle, or raise.

    The pin is REQUIRED. The definitions are re-hashed from whatever the module
    currently holds and compared against it, so every route to a modified
    definition — an edited literal, a rebound module attribute, a monkeypatched
    dataclass — lands on the same refusal. There is no unpinned resolve, because an
    unpinned resolve is a resolve that cannot detect the thing the pin exists to
    detect.
    """
    _require_nonempty_str(pinned_definitions_digest, "pinned_definitions_digest")
    check = verify_control_definitions(pinned_definitions_digest)
    if check.outcome != schemas.PASS:
        raise ControlBundleDrift(
            "control definitions failed verification against the campaign pin: "
            + "; ".join(check.reasons)
            + ". The control definitions live inside the evaluator bundle under the "
              "measurement trust boundary and MUST NOT be modified by any process "
              "inside the loop.")

    definitions = tuple(CONTROL_DEFINITIONS)
    definitions_digest = schemas.content_hash(_definitions_payload(definitions))
    declarations = tuple(historical_win_replays)
    campaign_payload = {
        "definitions_digest": definitions_digest,
        "aa_cadence": aa_cadence.to_dict() if isinstance(aa_cadence, AACadence) else None,
        "seed_rotation": (seed_rotation.to_dict()
                          if isinstance(seed_rotation, SeedRotationSchedule) else None),
        "historical_win_replays": [
            d.to_dict() for d in declarations
            if isinstance(d, HistoricalWinReplayDeclaration)],
    }
    campaign_digest = schemas.content_hash(campaign_payload)
    bundle = ControlBundle(
        definitions=definitions,
        definitions_digest=definitions_digest,
        campaign_digest=campaign_digest,
        aa_cadence=aa_cadence,
        seed_rotation=seed_rotation,
        historical_win_replays=declarations,
        source_label=source_label,
    )
    if pinned_campaign_digest is not None and pinned_campaign_digest != campaign_digest:
        raise ControlBundleDrift(
            f"campaign control bindings do not match the manifest pin: pinned "
            f"{pinned_campaign_digest!r}, resolved {campaign_digest!r}")
    return bundle


# =============================================================================
# Observations — what a control run produced. Fixtures, never live runs, here.
# =============================================================================

@dataclass(frozen=True)
class ControlObservation:
    """One control's observed result: the verdict THE GATE gave it.

    The subject of a control is the evaluator, so a control's input is the
    evaluator's own output. `verdict` is an `api.Verdict`, which can only be minted
    by `api.compute_verdict()` — so an observation cannot carry a hand-stamped
    verdict any more than a record can.

    `ran=False` is a first-class state with a mandatory reason. *"A campaign that
    cannot run controls 1-4 MUST NOT rank any candidate"* — a control that did not
    run is COULD_NOT_CHECK, never an absent entry that reads as satisfied.
    """

    control_id: str
    ran: bool
    verdict: Optional[api.Verdict] = None
    could_not_run_reason: Optional[str] = None
    abs_effects: tuple = ()
    promoted: Optional[bool] = None
    observed_magnitude: Optional[float] = None
    observed_direction: Optional[str] = None
    evidence_ref: Optional[str] = None
    notes: tuple = ()

    def __post_init__(self) -> None:
        if self.control_id not in CONTROL_IDS:
            raise ValueError(f"observation.control_id {self.control_id!r} is not one of "
                             f"{list(CONTROL_IDS)}")
        if not isinstance(self.ran, bool):
            raise TypeError("observation.ran must be a bool")
        if self.ran:
            if not isinstance(self.verdict, api.Verdict):
                raise TypeError(
                    "observation.verdict must be an api.Verdict when the control ran; a "
                    "control observes the verdict the gate produced, and only "
                    "api.compute_verdict() mints one")
            if self.control_id == CONTROL_HISTORICAL_WIN_REPLAY and self.promoted is None:
                raise ValueError(
                    "the historical-win replay observation must state whether the win "
                    "PROMOTED; 'it MUST promote' is unevaluable without it")
        else:
            if self.verdict is not None:
                raise ValueError("observation.verdict must be None when the control did "
                                 "not run")
            _require_nonempty_str(self.could_not_run_reason,
                                  "observation.could_not_run_reason")
        if not isinstance(self.abs_effects, tuple):
            raise TypeError("observation.abs_effects must be a tuple")
        for value in self.abs_effects:
            magnitude = _require_finite(value, "observation.abs_effects[]")
            if magnitude < 0:
                raise ValueError("observation.abs_effects are magnitudes and must be "
                                 "non-negative")
        if self.promoted is not None and not isinstance(self.promoted, bool):
            raise TypeError("observation.promoted must be a bool or None")
        if self.observed_magnitude is not None:
            _require_finite(self.observed_magnitude, "observation.observed_magnitude")
        if self.observed_direction is not None \
                and self.observed_direction not in schemas.METRIC_DIRECTIONS:
            raise ValueError(f"observation.observed_direction "
                             f"{self.observed_direction!r} is not one of "
                             f"{sorted(schemas.METRIC_DIRECTIONS)}")
        if not isinstance(self.notes, tuple):
            raise TypeError("observation.notes must be a tuple")

    def to_dict(self) -> dict:
        return {
            "control_id": self.control_id,
            "ran": self.ran,
            "verdict_status": None if self.verdict is None else self.verdict.status,
            "verdict_tier": None if self.verdict is None else self.verdict.tier,
            "effect_resolution": (None if self.verdict is None
                                  else self.verdict.effect_resolution),
            "speed_rank_admissible": (None if self.verdict is None
                                      else self.verdict.speed_rank_admissible),
            "could_not_run_reason": self.could_not_run_reason,
            "abs_effect_count": len(self.abs_effects),
            "promoted": self.promoted,
            "observed_magnitude": self.observed_magnitude,
            "observed_direction": self.observed_direction,
            "evidence_ref": self.evidence_ref,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ControlContext:
    """What the evaluators need besides the observation itself."""

    campaign_id: str
    backend: str
    phase: str
    cell_class: str
    window_id: str
    historical: HistoricalWinResolution
    neutral_dispersion: schemas.Check
    calibration: Optional[api.CalibrationOutputs] = None

    def __post_init__(self) -> None:
        for name in ("campaign_id", "backend", "phase", "cell_class", "window_id"):
            _require_nonempty_str(getattr(self, name), f"control_context.{name}")
        if not isinstance(self.historical, HistoricalWinResolution):
            raise TypeError("control_context.historical must be a HistoricalWinResolution")
        if not isinstance(self.neutral_dispersion, schemas.Check):
            raise TypeError(
                "control_context.neutral_dispersion must be a schemas.Check; the neutral "
                "control's dispersion-vs-floor consistency check is computed by the "
                "calibration solver, and 'not supplied' must be sayable as "
                "COULD_NOT_CHECK rather than inferred as satisfied")
        if self.calibration is not None \
                and not isinstance(self.calibration, api.CalibrationOutputs):
            raise TypeError("control_context.calibration must be api.CalibrationOutputs "
                            "or None")


@dataclass(frozen=True)
class ControlRunContext:
    """What a `ControlRunner` is given. Carries no authority and no thresholds."""

    campaign_id: str
    backend: str
    phase: str
    cell_class: str
    window_id: str
    tier: str
    seed: str
    anchor: api.AnchorIdentity
    declaration: Optional[HistoricalWinReplayDeclaration] = None

    def __post_init__(self) -> None:
        for name in ("campaign_id", "backend", "phase", "cell_class", "window_id", "seed"):
            _require_nonempty_str(getattr(self, name), f"run_context.{name}")
        api.admit_tier(self.tier)
        if not isinstance(self.anchor, api.AnchorIdentity):
            raise TypeError("run_context.anchor must be an api.AnchorIdentity")
        if self.declaration is not None \
                and not isinstance(self.declaration, HistoricalWinReplayDeclaration):
            raise TypeError("run_context.declaration must be a "
                            "HistoricalWinReplayDeclaration or None")


class ControlRunner(Protocol):
    """The seam that actually runs a control. NOT implemented in this module.

    A conforming runner holds the resource claim, drives the candidate pipeline for
    the control's fixture, and hands back the `api.Verdict` the gate produced. This
    module never calls a builder, a benchmark, or a process; the tests supply a
    fixture runner and that is the only runner in the tree today.
    """

    runner_id: str

    def run_control(self, definition: ControlDefinition,
                    context: ControlRunContext) -> ControlObservation:
        ...


# =============================================================================
# Per-control evaluation — the protocol's five requirements, as predicates
# =============================================================================

def _window_was_void(verdict: api.Verdict) -> bool:
    return bool(verdict.void_findings)


def _rank_key_or_none(verdict: api.Verdict):
    """Call the exact method a ranking loop calls. Returns the key, or None.

    This is the structural form of *"MUST receive no speed rank at all"*: the
    degraded-negative control fails if this returns a key, whatever the rest of the
    record says.
    """
    try:
        return verdict.rank_key()
    except api.SpeedRankUnavailable:
        return None


def _void_could_not_check(control_id: str, verdict: api.Verdict) -> schemas.Check:
    return schemas.Check(
        schemas.COULD_NOT_CHECK,
        (f"the {control_id} control's own window was VOID "
         f"({[f.reason for f in verdict.void_findings]}); a voided window says nothing "
         "whatever about the gate, and 'MUST NOT be recorded as a candidate failure' "
         "applies to a control run exactly as it applies to a candidate",))


def _not_run(control_id: str, observation: ControlObservation) -> schemas.Check:
    return schemas.Check(
        schemas.COULD_NOT_CHECK,
        (f"the {control_id} control did not run: {observation.could_not_run_reason}",))


def _evaluate_positive(definition: ControlDefinition, observation: ControlObservation,
                       context: ControlContext) -> schemas.Check:
    if not observation.ran:
        return _not_run(definition.control_id, observation)
    verdict = observation.verdict
    if _window_was_void(verdict):
        return _void_could_not_check(definition.control_id, verdict)

    reasons = []
    if definition.required_tiers and verdict.tier not in definition.required_tiers:
        reasons.append(
            f"the positive control ran at tier {verdict.tier!r}; it is required to rank "
            f"above the anchor at {list(definition.required_tiers)}")
    rank = _rank_key_or_none(verdict)
    if rank is None:
        reasons.append(
            "the positive control received NO speed rank: "
            + verdict.speed_rank_withheld_reason())
    elif verdict.effect_resolution != api.EFFECT_IMPROVEMENT:
        reasons.append(
            f"the positive control was ranked but its effect resolved as "
            f"{verdict.effect_resolution!r}, not {api.EFFECT_IMPROVEMENT!r}; a known-correct "
            "optimization with a real, bounded mechanism MUST rank above the anchor")
    if reasons:
        reasons.append("Failure is a gate defect.")
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


def _evaluate_neutral(definition: ControlDefinition, observation: ControlObservation,
                      context: ControlContext) -> schemas.Check:
    if not observation.ran:
        return _not_run(definition.control_id, observation)
    verdict = observation.verdict
    if _window_was_void(verdict):
        return _void_could_not_check(definition.control_id, verdict)

    advanced = (verdict.speed_rank_admissible
                and verdict.effect_resolution == api.EFFECT_IMPROVEMENT)
    if advanced:
        return schemas.Check(
            schemas.FAIL,
            ("the neutral control ADVANCED: a correct change whose true effect is centred "
             f"on zero was ranked as an {api.EFFECT_IMPROVEMENT}; the gate is producing "
             "false positives",))

    if verdict.status != api.STATUS_PASS:
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"the neutral control did not advance, but its verdict status is "
             f"{verdict.status!r}, so 'did not advance' is not evidence that the gate "
             "discriminates — a control that was rejected for an unrelated reason has "
             "not tested the discrimination it exists to test",))

    dispersion = context.neutral_dispersion
    if dispersion.outcome == schemas.FAIL:
        return schemas.Check(
            schemas.FAIL,
            ("the neutral control's dispersion check against the calibrated floor "
             "FAILED; a neutral control materially exceeding the A/A floor FAILS the "
             "calibration rather than raising the floor",) + tuple(dispersion.reasons))
    if dispersion.outcome == schemas.COULD_NOT_CHECK:
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            ("the neutral control's dispersion could not be checked against the "
             "calibrated floor",) + tuple(dispersion.reasons))
    return schemas.Check(schemas.PASS)


def _evaluate_degraded_negative(definition: ControlDefinition,
                                observation: ControlObservation,
                                context: ControlContext) -> schemas.Check:
    if not observation.ran:
        return _not_run(definition.control_id, observation)
    verdict = observation.verdict

    rank = _rank_key_or_none(verdict)
    if rank is not None:
        # Checked BEFORE the void branch: a void window is an excuse for an absent
        # rank, never for a present one.
        return schemas.Check(
            schemas.FAIL,
            (f"the degraded-negative control RECEIVED a speed rank {rank!r}. It is "
             "deliberately fast-looking but wrong, and it MUST receive no speed rank at "
             "all — not a penalised one",))

    if _window_was_void(verdict):
        return _void_could_not_check(definition.control_id, verdict)

    caught_by = tuple(
        gate.gate_id for gate in verdict.gates
        if gate.gate_class in api.SPEED_BLOCKING_GATE_CLASSES
        and gate.check.outcome == schemas.FAIL
    )
    if caught_by:
        return schemas.Check(schemas.PASS)

    blocked_by = tuple(
        f"{gate.gate_id}:{gate.check.outcome}" for gate in verdict.gates
        if gate.gate_class in api.SPEED_BLOCKING_GATE_CLASSES
        and gate.check.outcome != schemas.PASS
    )
    if blocked_by:
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            ("the degraded-negative control received no speed rank, but no "
             "correctness-class gate FAILED it — the prior gates returned "
             f"{list(blocked_by)}. COULD_NOT_CHECK is not a rejection, so this run does "
             "not demonstrate that the gate can reject a fast-looking wrong candidate",))
    return schemas.Check(
        schemas.FAIL,
        ("the degraded-negative control received no speed rank, but NO "
         "correctness-class gate rejected it: the rank was withheld only because "
         f"{verdict.speed_rank_withheld_reason()}. A cheating candidate that happens to "
         "measure slow is not a caught cheating candidate, and this control exists to "
         "prove the gate detects the cheat rather than the slowness",))


def _evaluate_aa(definition: ControlDefinition, observation: ControlObservation,
                 context: ControlContext) -> schemas.Check:
    if not observation.ran:
        return _not_run(definition.control_id, observation)
    verdict = observation.verdict
    if _window_was_void(verdict):
        return _void_could_not_check(definition.control_id, verdict)

    if verdict.effect_resolution in (api.EFFECT_IMPROVEMENT, api.EFFECT_REGRESSION):
        return schemas.Check(
            schemas.FAIL,
            (f"the A/A control resolved as {verdict.effect_resolution!r}: the anchor was "
             "measured against itself and produced a significant effect at the declared "
             "rate. A failing A/A VOIDS the enclosing measurement window",))
    if verdict.status != api.STATUS_PASS:
        # Was: only INVALID was caught, so an A/A whose correctness gate FAILED, or
        # whose verdict was INCONCLUSIVE, returned PASS as long as no significant
        # effect was resolved — the gate reporting "no effect" precisely because it
        # never got a usable measurement. This is the same guard `_evaluate_neutral`
        # already applies, for the same reason.
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"the A/A control's verdict status is {verdict.status!r}, so the "
             "false-positive rate is uncalibrated for this window; that is not the same "
             "as a failing A/A and is not recorded as one",))
    if verdict.effect is None or verdict.effect_resolution == api.EFFECT_NOT_MEASURED:
        # "The anchor measured against itself, THROUGH THE FULL CANDIDATE PIPELINE
        # ... it calibrates the false-positive rate and it is what detects host
        # drift mid-campaign." A run that produced no effect estimate calibrated
        # nothing and detected nothing; passing it would let a campaign satisfy
        # control 4 by not measuring.
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            ("the A/A control produced no effect estimate "
             f"(resolution {verdict.effect_resolution!r}), so it calibrated no "
             "false-positive rate and detected no host drift; an A/A that measured "
             "nothing is not an A/A that found nothing",))
    return schemas.Check(schemas.PASS)


def _evaluate_historical(definition: ControlDefinition, observation: ControlObservation,
                         context: ControlContext) -> schemas.Check:
    resolution = context.historical
    if not resolution.available:
        # The unavailable branch is handled by the harness, which needs the operator
        # escalation to dispose of it. Reaching here with an observation means the
        # harness ran a control it had resolved as unavailable.
        raise ControlWiringError(
            "the historical-win replay was resolved as unavailable but an observation "
            f"was produced for it ({resolution.reason()}); the unavailable branch is "
            "recorded and escalated, not run")
    if not observation.ran:
        return _not_run(definition.control_id, observation)
    verdict = observation.verdict
    if _window_was_void(verdict):
        return _void_could_not_check(definition.control_id, verdict)

    declaration = resolution.declaration
    reasons = []
    unchecked = []
    if observation.promoted is not True:
        reasons.append(
            f"the historical win {declaration.win_id!r} did NOT promote through T0-T2")
    if observation.observed_direction is None:
        # Was: absent direction skipped the comparison silently, so the check could
        # be passed by deleting the thing it inspects — while an absent MAGNITUDE
        # was already COULD_NOT_CHECK. The manifest declares a reference direction
        # for control 5 precisely so the replay is compared against it.
        unchecked.append(
            f"the historical win {declaration.win_id!r} reported no direction, so it "
            f"cannot be compared against its declared reference direction "
            f"{declaration.reference_direction!r}")
    elif observation.observed_direction != declaration.reference_direction:
        reasons.append(
            f"the replayed win moved {observation.observed_direction!r} but the declared "
            f"reference direction is {declaration.reference_direction!r}")
    if observation.observed_magnitude is None:
        unchecked.append(
            f"the historical win {declaration.win_id!r} reported no magnitude, so it "
            "cannot be compared against its declared reference band "
            f"{declaration.reference_band.to_dict()}")
    elif not declaration.reference_band.contains(abs(observation.observed_magnitude)):
        reasons.append(
            f"the replayed win's magnitude {abs(observation.observed_magnitude):g} is "
            f"outside its declared reference band "
            f"[{declaration.reference_band.low:g}, {declaration.reference_band.high:g}]")
    if reasons:
        reasons.append(
            "A failure to promote is a gate defect, not a research finding: it halts the "
            "campaign and is escalated to the operator.")
        return schemas.Check(schemas.FAIL, tuple(reasons))
    if unchecked:
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(unchecked))
    return schemas.Check(schemas.PASS)


_EVALUATORS = {
    CONTROL_POSITIVE: _evaluate_positive,
    CONTROL_NEUTRAL: _evaluate_neutral,
    CONTROL_DEGRADED_NEGATIVE: _evaluate_degraded_negative,
    CONTROL_AA: _evaluate_aa,
    CONTROL_HISTORICAL_WIN_REPLAY: _evaluate_historical,
}

_MISSING_EVALUATORS = [cid for cid in CONTROL_IDS if cid not in _EVALUATORS]
if _MISSING_EVALUATORS:  # pragma: no cover - import-time contract assertion
    raise ImportError(f"controls {_MISSING_EVALUATORS} have no evaluator; a control with "
                      "no predicate would silently pass")

#: The predicate baseline, and the verifier that closes over both baselines. This
#: is the earliest point at which `_EVALUATORS` is complete.
CONTROL_PREDICATES_DIGEST = _current_predicates_digest()
verify_control_definitions = _make_control_definitions_verifier(
    CONTROL_DEFINITIONS_DIGEST, CONTROL_PREDICATES_DIGEST)


# =============================================================================
# Outcomes, gate defects, and the panel result
# =============================================================================

@dataclass(frozen=True)
class ControlOutcome:
    """One control's evaluated result, with the protocol's consequence attached."""

    control_id: str
    definition: ControlDefinition
    check: schemas.Check
    disposition: str
    detail: tuple = ()

    def __post_init__(self) -> None:
        if self.control_id not in CONTROL_IDS:
            raise ValueError(f"outcome.control_id {self.control_id!r} is unknown")
        if not isinstance(self.definition, ControlDefinition):
            raise TypeError("outcome.definition must be a ControlDefinition")
        if not isinstance(self.check, schemas.Check):
            raise TypeError("outcome.check must be a schemas.Check")
        if self.disposition not in DISPOSITIONS:
            raise ValueError(f"outcome.disposition {self.disposition!r} is not one of "
                             f"{list(DISPOSITIONS)}")
        # The disposition is a SUPPLIED field that `ControlPanelResult.gate_defects`
        # reads: an outcome stamped `unavailable_recorded` is skipped there. Without
        # this cross-check a FAILING control 1 carrying that stamp produced no gate
        # defect at all — a derived consequence defeated by a stamp, which is the one
        # thing this file's design says cannot happen.
        if self.disposition == DISPOSITION_SATISFIED \
                and self.check.outcome != schemas.PASS:
            raise ValueError(
                f"outcome for {self.control_id!r} is dispositioned "
                f"{DISPOSITION_SATISFIED!r} but its check returned "
                f"{self.check.outcome}; a control is satisfied because its check "
                "passed, never because it was labelled so")
        if self.disposition != DISPOSITION_SATISFIED \
                and self.check.outcome == schemas.PASS \
                and self.disposition != DISPOSITION_UNAVAILABLE_RECORDED:
            raise ValueError(
                f"outcome for {self.control_id!r} passed its check but is "
                f"dispositioned {self.disposition!r}; only the accept-side control's "
                "recorded-unavailable branch passes under a non-satisfied disposition")
        if self.disposition == DISPOSITION_UNAVAILABLE_RECORDED \
                and self.control_id != CONTROL_HISTORICAL_WIN_REPLAY:
            raise ValueError(
                f"only the {CONTROL_HISTORICAL_WIN_REPLAY!r} control has an "
                f"unavailable branch; control {self.control_id!r} cannot be "
                f"dispositioned {DISPOSITION_UNAVAILABLE_RECORDED!r}")

    def to_dict(self) -> dict:
        return {
            "control_id": self.control_id,
            "ordinal": self.definition.ordinal,
            "outcome": self.check.outcome,
            "reasons": list(self.check.reasons),
            "disposition": self.disposition,
            "requirement": self.definition.requirement,
            "tests_gate_ability_to": self.definition.tests_gate_ability_to,
            "detail": list(self.detail),
        }


@dataclass(frozen=True)
class GateDefectFinding:
    """A control failure the protocol classifies as a defect in the GATE.

    Only controls 1 and 5 produce one, because only those two are named that way in
    the ratified text — *"Failure is a gate defect"* (control 1) and *"A failure to
    promote is a gate defect, not a research finding"* (control 5). Controls 2 and 3
    failing block ranking and are recorded with their own words; this module does
    NOT upgrade them to gate defects, because inventing a consequence the protocol
    did not state is the same class of error as omitting one it did.
    """

    control_id: str
    protocol_phrase: str
    outcome: str
    detail: tuple

    def __post_init__(self) -> None:
        if self.control_id not in (CONTROL_POSITIVE, CONTROL_HISTORICAL_WIN_REPLAY):
            raise ValueError(
                f"control {self.control_id!r} failing is not a gate defect under the "
                "ratified text; only controls 1 and 5 are named that way")
        if self.outcome not in (schemas.FAIL, schemas.COULD_NOT_CHECK):
            raise ValueError("gate-defect outcome must be FAIL or COULD_NOT_CHECK")

    def to_dict(self) -> dict:
        return {"control_id": self.control_id, "protocol_phrase": self.protocol_phrase,
                "outcome": self.outcome, "detail": list(self.detail),
                "halts_campaign": True, "escalation_required": True}


# -----------------------------------------------------------------------------
# The single derivation of a control sweep's outcomes. Called twice on purpose:
# once by `ControlHarness.evaluate()` to produce them, and once by
# `ControlPanelResult.__post_init__` to prove the object's own contents imply
# them. Two call sites, ONE derivation — a second copy would drift, and both
# copies would keep returning a panel.
# -----------------------------------------------------------------------------

def _index_observations(observations: Sequence[ControlObservation]) -> dict:
    by_id: dict = {}
    for observation in observations:
        if not isinstance(observation, ControlObservation):
            raise ControlWiringError(
                f"observations must be ControlObservation, got "
                f"{type(observation).__name__}")
        if observation.control_id in by_id:
            raise ControlWiringError(
                f"two observations for control {observation.control_id!r}; a control "
                "has one result per window and choosing between two would be the "
                "harness selecting its own answer")
        by_id[observation.control_id] = observation
    return by_id


def _dispose_unavailable_control_5(definition: ControlDefinition,
                                   historical: HistoricalWinResolution,
                                   escalation: Optional[OperatorEscalation]):
    """Dispose of the unavailable branch. Never a silent skip, never the
    controller's call."""
    marker_reason = (f"{HISTORICAL_REPLAY_UNAVAILABLE}: backend "
                     f"{historical.backend}: {historical.reason()}")
    if escalation is None:
        return ControlOutcome(
            control_id=definition.control_id, definition=definition,
            check=schemas.Check(
                schemas.COULD_NOT_CHECK,
                (marker_reason,
                 "no operator escalation is on the record; the campaign MUST NOT "
                 "silently run four controls and report as though it ran five",)),
            disposition=DISPOSITION_NOT_RUN,
        ), ("control 5 is unavailable and was not escalated to the operator; whether "
            "the campaign proceeds on four controls is the operator's call, taken "
            "once, on the record — not the controller's")
    if escalation.decision == OPERATOR_DECISION_PENDING:
        return ControlOutcome(
            control_id=definition.control_id, definition=definition,
            check=schemas.Check(
                schemas.COULD_NOT_CHECK,
                (marker_reason,
                 f"escalated to the operator at {escalation.escalation_ref}; the "
                 "decision is still PENDING")),
            disposition=DISPOSITION_NOT_RUN,
        ), (f"control 5 is unavailable and escalation {escalation.escalation_ref} is "
            "pending an operator decision")
    if escalation.decision == OPERATOR_DECISION_HALT:
        return ControlOutcome(
            control_id=definition.control_id, definition=definition,
            check=schemas.Check(
                schemas.FAIL,
                (marker_reason,
                 f"the operator halted the campaign at {escalation.escalation_ref} "
                 f"({escalation.decided_by}, {escalation.decided_at})")),
            disposition=DISPOSITION_UNAVAILABLE_RECORDED,
        ), f"the operator halted this campaign at {escalation.escalation_ref}"
    return ControlOutcome(
        control_id=definition.control_id, definition=definition,
        check=schemas.Check(
            schemas.PASS,
            (marker_reason,
             f"the operator authorised proceeding on four controls at "
             f"{escalation.escalation_ref} ({escalation.decided_by}, "
             f"{escalation.decided_at}); every record emitted by this campaign "
             f"carries controls=4/5 ({HISTORICAL_REPLAY_UNAVAILABLE})")),
        disposition=DISPOSITION_UNAVAILABLE_RECORDED,
    ), None


def _build_panel(outcomes: Sequence[ControlOutcome],
                 historical: HistoricalWinResolution,
                 escalation: Optional[OperatorEscalation]) -> Optional[api.ControlPanel]:
    by_id = {o.control_id: o for o in outcomes}
    fields = {
        _PANEL_FIELD_BY_CONTROL[cid]: by_id[cid].check
        for cid in MANDATORY_CONTROL_IDS
    }
    if historical.available:
        fields["historical_replay"] = by_id[CONTROL_HISTORICAL_WIN_REPLAY].check
        return api.ControlPanel(**fields)
    if escalation is None or escalation.decision != OPERATOR_DECISION_PROCEED_ON_FOUR:
        # `api.ControlPanel` would happily take a reason and an escalation ref
        # here; refusing to build one at all is stronger, because a panel is a
        # licence to rank and the operator has not granted it.
        return None
    return api.ControlPanel(
        historical_replay=None,
        historical_replay_unavailable_reason=(
            f"backend {historical.backend}: {historical.reason()}"),
        operator_escalation_ref=escalation.escalation_ref,
        **fields)


def _derive_panel_result_parts(*, observations: Sequence[ControlObservation],
                               context: "ControlContext",
                               escalation: Optional[OperatorEscalation]) -> tuple:
    """Return `(outcomes, panel, blocked_reason)` — everything derivable.

    Reads the module-level `CONTROL_DEFINITIONS` and `_EVALUATORS`, deliberately
    NOT a bundle's copy of them: the definitions a result is scored against are
    the ones under the measurement trust boundary, and a bundle whose copy has
    drifted from them is what `verify_control_definitions()` is for.
    """
    if not isinstance(context, ControlContext):
        raise TypeError("context must be a ControlContext")
    by_id = _index_observations(observations)
    outcomes: list = []
    blocked_reason: Optional[str] = None
    for definition in CONTROL_DEFINITIONS:
        cid = definition.control_id
        if cid == CONTROL_HISTORICAL_WIN_REPLAY and not context.historical.available:
            outcome, blocked_reason = _dispose_unavailable_control_5(
                definition, context.historical, escalation)
            outcomes.append(outcome)
            continue
        observation = by_id.get(cid)
        if observation is None:
            outcomes.append(ControlOutcome(
                control_id=cid, definition=definition,
                check=schemas.Check(
                    schemas.COULD_NOT_CHECK,
                    (f"no observation was produced for the {cid} control; a control "
                     "with no result is not a control that passed",)),
                disposition=DISPOSITION_NOT_RUN))
            continue
        check = _EVALUATORS[cid](definition, observation, context)
        outcomes.append(ControlOutcome(
            control_id=cid, definition=definition, check=check,
            disposition=(DISPOSITION_SATISFIED if check.outcome == schemas.PASS
                         else definition.failure_disposition),
            detail=tuple(observation.notes)))

    panel = _build_panel(outcomes, context.historical, escalation)
    if panel is None and blocked_reason is None:
        blocked_reason = ("no control panel could be built for this window; see the "
                          "control outcomes for which control could not be disposed of")
    return tuple(outcomes), panel, blocked_reason


@dataclass(frozen=True)
class ControlPanelResult:
    """Everything one control sweep produced.

    `may_rank`, `halts_campaign`, `voids_window` and `marker` are **properties, not
    fields**. `api.Verdict` re-derives its stored status and raises on disagreement;
    the same guarantee is available more cheaply here by never storing the derived
    values at all — there is no attribute to disagree with, and
    `dataclasses.replace()` cannot manufacture a rankable panel.

    (`dataclasses.replace()` cannot manufacture a rankable panel *by accident* —
    it can with the mint token in hand, and lock 2 below is what refuses it then.)

    That reasoning was right about the *properties* and wrong about the *object*:
    nothing stopped a caller building the whole result by hand out of five PASS
    `ControlOutcome`s, and `may_rank` then answered True with no control ever run.
    `observations` and `context` are stored for that reason, and
    `__post_init__` re-derives `outcomes`, `panel` and `blocked_reason` from them
    and refuses on disagreement — so a PASS on this object is now backed by an
    `api.Verdict` per control, and those can only be minted by
    `api.compute_verdict()`. See `ControlPanelForged`.
    """

    outcomes: tuple
    panel: Optional[api.ControlPanel]
    historical: HistoricalWinResolution
    escalation: Optional[OperatorEscalation]
    aa_cadence: schemas.Check
    definitions_check: schemas.Check
    observations: tuple
    context: "ControlContext"
    blocked_reason: Optional[str] = None
    mint: InitVar[Any] = None

    def __post_init__(self, mint: Any) -> None:
        if mint is not _PANEL_MINT_TOKEN:
            raise ControlPanelForged(
                "ControlPanelResult is not constructible directly — it is the licence "
                "to rank, and it derives from a control sweep that actually ran. Call "
                "ControlHarness.evaluate().")
        if not isinstance(self.context, ControlContext):
            raise ControlPanelForged("result.context must be a ControlContext")
        if not isinstance(self.observations, tuple):
            raise ControlPanelForged("result.observations must be a tuple")
        ids = tuple(o.control_id for o in self.outcomes)
        if ids != CONTROL_IDS:
            raise ValueError(
                f"a control panel result must cover exactly {list(CONTROL_IDS)} in ordinal "
                f"order, got {list(ids)}; a missing control is not an absent row, it is a "
                "campaign that MUST NOT rank")
        if self.panel is not None and not isinstance(self.panel, api.ControlPanel):
            raise TypeError("result.panel must be an api.ControlPanel or None")
        for name, klass in (("historical", HistoricalWinResolution),
                            ("aa_cadence", schemas.Check),
                            ("definitions_check", schemas.Check)):
            if not isinstance(getattr(self, name), klass):
                raise TypeError(f"result.{name} must be a {klass.__name__}")
        if self.escalation is not None and not isinstance(self.escalation, OperatorEscalation):
            raise TypeError("result.escalation must be an OperatorEscalation or None")
        if self.historical != self.context.historical:
            raise ControlPanelForged(
                "result.historical is not the resolution the controls were scored "
                "against (result.context.historical); control 5's availability decides "
                "whether the unavailable branch or the replay predicate ran, so two "
                "answers to it would let a result be scored one way and reported another")

        # Lock 2. Everything derivable is re-derived from this object's OWN
        # observations and context, through the live predicate table. A stamped
        # PASS now has to survive being recomputed from a `ControlObservation`,
        # and an observation that ran carries an `api.Verdict` — which only
        # `api.compute_verdict()` mints.
        derived_outcomes, derived_panel, derived_blocked = _derive_panel_result_parts(
            observations=self.observations, context=self.context,
            escalation=self.escalation)
        for name, stored, derived in (
                ("outcomes", self.outcomes, derived_outcomes),
                ("panel", self.panel, derived_panel),
                ("blocked_reason", self.blocked_reason, derived_blocked)):
            if stored != derived:
                raise ControlPanelForged(
                    f"result.{name} does not follow from the observations this result "
                    f"carries: stored {stored!r}, derived {derived!r}. A control panel is "
                    "computed from control results; it is never supplied.")

    def outcome_for(self, control_id: str) -> ControlOutcome:
        for outcome in self.outcomes:
            if outcome.control_id == control_id:
                return outcome
        raise KeyError(control_id)

    @property
    def gate_defects(self) -> tuple:
        defects = []
        for outcome in self.outcomes:
            if outcome.check.outcome == schemas.PASS:
                continue
            if outcome.definition.failure_disposition != DISPOSITION_GATE_DEFECT:
                continue
            if outcome.disposition == DISPOSITION_UNAVAILABLE_RECORDED:
                continue
            if outcome.control_id == CONTROL_HISTORICAL_WIN_REPLAY \
                    and not self.historical.available:
                # A gate defect is *"a failure to promote"*. An entry the manifest
                # never declared, or one whose evidence does not resolve in-repo, is
                # the UNAVAILABLE branch — a different clause with a different
                # remedy. Recording it as a gate defect would journal "the gate is
                # broken" out of an absent manifest row, and `to_dict()` would stamp
                # the phrase "it MUST promote" on a win that was never replayed.
                # Ranking is still blocked: `may_rank` needs a panel, and
                # `_build_panel` refuses to build one without the operator's call.
                continue
            defects.append(GateDefectFinding(
                control_id=outcome.control_id,
                protocol_phrase=outcome.definition.requirement,
                outcome=outcome.check.outcome,
                detail=tuple(outcome.check.reasons)))
        return tuple(defects)

    @property
    def halts_campaign(self) -> bool:
        if self.escalation is not None \
                and self.escalation.decision == OPERATOR_DECISION_HALT:
            return True
        if not self.historical.available and (
                self.escalation is None
                or self.escalation.decision != OPERATOR_DECISION_PROCEED_ON_FOUR):
            # Unavailable and undecided: the campaign stops here until the
            # operator's call is on the record. Stated in its own right rather than
            # arriving as a side effect of a mislabelled gate defect.
            return True
        return bool(self.gate_defects)

    @property
    def voids_window(self) -> bool:
        return self.outcome_for(CONTROL_AA).check.outcome != schemas.PASS

    @property
    def may_rank(self) -> bool:
        """*"A campaign that cannot run controls 1-4 MUST NOT rank any candidate."*

        Search-grade also requires *"a passing A/A control within its declared
        cadence"*, and the cadence attestation is a separate fact from this
        window's A/A outcome — an in-window A/A can pass while the cadence
        attestation says none has run for twenty windows. This property already
        consulted `definitions_check`, which is likewise not one of controls 1-4,
        so omitting the cadence was an omission rather than a scoping decision.
        """
        if self.panel is None:
            return False
        if self.definitions_check.outcome != schemas.PASS:
            return False
        if self.aa_cadence.outcome != schemas.PASS:
            return False
        return (self.panel.check_1_to_4().outcome == schemas.PASS
                and self.panel.check_5().outcome == schemas.PASS)

    @property
    def marker(self) -> Optional[str]:
        """The grammar's `controls=` field, or None when no panel could be built."""
        return None if self.panel is None else self.panel.marker()

    def to_dict(self) -> dict:
        return {
            "outcomes": [o.to_dict() for o in self.outcomes],
            # The proof-of-run, journaled rather than asserted: every outcome above
            # is re-derived from these, and an observation that ran carries a minted
            # verdict. A reader can recompute the panel from this list.
            "observations": [o.to_dict() for o in self.observations],
            "panel": None if self.panel is None else self.panel.to_dict(),
            "historical": self.historical.to_dict(),
            "escalation": None if self.escalation is None else self.escalation.to_dict(),
            "aa_cadence": {"outcome": self.aa_cadence.outcome,
                           "reasons": list(self.aa_cadence.reasons)},
            "definitions_check": {"outcome": self.definitions_check.outcome,
                                  "reasons": list(self.definitions_check.reasons)},
            "gate_defects": [d.to_dict() for d in self.gate_defects],
            "halts_campaign": self.halts_campaign,
            "voids_window": self.voids_window,
            "may_rank": self.may_rank,
            "marker": self.marker,
            "blocked_reason": self.blocked_reason,
        }


class ControlHarness:
    """Runs the five controls through the `ControlRunner` seam and evaluates them.

    The harness owns NO runner of its own and has no default: a default runner would
    report an unrun control as having produced no failures, which is a fail-open
    panel — the same defect `api.TierDispatcher` refuses with `EvaluatorNotWired`.

    Control 5 is never *silently* skipped. When `resolve_historical_win_replay()`
    says unavailable, the harness records `HISTORICAL_REPLAY_UNAVAILABLE`, demands
    an `OperatorEscalation`, and refuses to produce a rankable panel until the
    operator's call is on the record.
    """

    def __init__(self, *, bundle: ControlBundle, runner: Any) -> None:
        if not isinstance(bundle, ControlBundle):
            raise TypeError("bundle must be a ControlBundle")
        if runner is None or not hasattr(runner, "run_control"):
            raise ControlWiringError(
                "ControlHarness requires a runner exposing run_control(definition, "
                "context); there is no default runner, because an unrun control with no "
                "observation would evaluate as no failures found")
        self.bundle = bundle
        self.runner = runner

    def seed_plan(self, *, campaign_seed: str, windows_completed: int) -> dict:
        """`{control_id: seed}` for this rotation epoch. Derived, never chosen.

        `derive_control_seed()`, `ControlBundle.seed_for()` and
        `SeedRotationSchedule.check_rotation()` were all declared, hashed into the
        campaign digest, and had NO CALLER: `run_all` handed one `run_context` —
        and therefore ONE seed — to all five controls. Five controls sharing a
        seed is five controls sharing a holdout, and *"a never-rotated holdout is
        an evaluator coverage defect"* applies with more force to a holdout that
        was never even per-control.

        This is the caller. `run_all` uses it; `execution.control_runner` reads it
        for the record's seed ledger, and gets the same values because it is the
        same derivation rather than a second one.
        """
        _require_nonempty_str(campaign_seed, "campaign_seed")
        if isinstance(windows_completed, bool) or not isinstance(windows_completed, int) \
                or windows_completed < 0:
            raise ValueError("windows_completed must be a non-negative int")
        plan = {
            definition.control_id: self.bundle.seed_for(
                campaign_seed=campaign_seed, control_id=definition.control_id,
                windows_completed=windows_completed)
            for definition in self.bundle.definitions
        }
        if len(set(plan.values())) != len(plan):
            # `derive_control_seed` keys on the control id, so collisions are not
            # reachable from the ratified derivation — which is exactly why a
            # collision here means the derivation is no longer the ratified one.
            raise ControlWiringError(
                "two controls were assigned the same seed; the seed derivation keys on "
                f"the control id, so this means it has been substituted: {plan!r}")
        return plan

    def run_all(self, *, run_context: ControlRunContext,
                historical: HistoricalWinResolution,
                campaign_seed: str,
                windows_completed: int) -> tuple:
        """Call the runner once per control the harness has decided to run.

        `campaign_seed` and `windows_completed` are REQUIRED and have no defaults.
        A default would restore the same-seed-for-everything behaviour this method
        used to have, silently, at exactly the call sites that forgot to rotate —
        and the resulting panel would still read as a clean five-control sweep.

        `run_context.seed` is OVERRIDDEN per control from `seed_plan()`. It is not
        merely ignored: a run context whose seed happens to equal a derived one is
        indistinguishable from a rotated sweep, so the placeholder must never
        reach a runner, and `test_control_runner` asserts it does not.
        """
        if not isinstance(run_context, ControlRunContext):
            raise TypeError("run_context must be a ControlRunContext")
        if not isinstance(historical, HistoricalWinResolution):
            raise TypeError("historical must be a HistoricalWinResolution")
        plan = self.seed_plan(campaign_seed=campaign_seed,
                              windows_completed=windows_completed)
        observations = []
        for definition in self.bundle.definitions:
            if definition.control_id == CONTROL_HISTORICAL_WIN_REPLAY \
                    and not historical.available:
                continue
            context = replace(run_context, seed=plan[definition.control_id])
            observation = self.runner.run_control(definition, context)
            if not isinstance(observation, ControlObservation):
                raise ControlWiringError(
                    f"runner returned {type(observation).__name__} for control "
                    f"{definition.control_id!r}; expected a ControlObservation")
            if observation.control_id != definition.control_id:
                raise ControlWiringError(
                    f"runner answered for control {observation.control_id!r} when asked "
                    f"for {definition.control_id!r}")
            observations.append(observation)
        return tuple(observations)

    def evaluate(self, *, observations: Sequence[ControlObservation],
                 context: ControlContext,
                 aa_cadence: schemas.Check,
                 escalation: Optional[OperatorEscalation] = None,
                 pinned_definitions_digest: Optional[str] = None,
                 pinned_campaign_digest: Optional[str] = None) -> ControlPanelResult:
        """Evaluate observations into outcomes, a panel, and the derived consequences."""
        if not isinstance(context, ControlContext):
            raise TypeError("context must be a ControlContext")
        if not isinstance(aa_cadence, schemas.Check):
            raise TypeError("aa_cadence must be a schemas.Check")
        observations = tuple(observations)
        _index_observations(observations)  # raises on a duplicate or a wrong type

        # `or` here treated a supplied-but-empty pin as "no pin supplied" and fell
        # back to the bundle's own digest — a PASS built out of a manifest field
        # that read as "". `verify_control_definitions` has a COULD_NOT_CHECK branch
        # for exactly that input ("an unreadable pin is not a satisfied pin"); the
        # fallback made it unreachable. Only a genuinely absent pin falls back now.
        definitions_check = self.bundle.reverify(
            pinned_definitions_digest=(self.bundle.definitions_digest
                                       if pinned_definitions_digest is None
                                       else pinned_definitions_digest),
            pinned_campaign_digest=pinned_campaign_digest)

        outcomes, panel, blocked_reason = _derive_panel_result_parts(
            observations=observations, context=context, escalation=escalation)
        return ControlPanelResult(
            outcomes=outcomes, panel=panel, historical=context.historical,
            escalation=escalation, aa_cadence=aa_cadence,
            definitions_check=definitions_check,
            observations=observations, context=context,
            blocked_reason=blocked_reason, mint=_PANEL_MINT_TOKEN)



# =============================================================================
# The seam into api.WindowAttestations
# =============================================================================

def window_control_attestations(result: ControlPanelResult) -> dict:
    """The `api.WindowAttestations` fields this control sweep is authoritative for.

    The mirror of `statistics.BlockReduction.window_checks`, and it exists for the
    same reason: three of the four facts a control sweep establishes have a home
    in the window attestations, and before this projection existed the caller had
    to remember which — so `definitions_check` reliably did not travel, and *"any
    post-hoc change to … the control definitions"* (What voids a run) could not
    reach `api.check_void_conditions` at all.

    Returns exactly the keyword arguments to splice into `api.WindowAttestations`:

      * `controls` — the `api.ControlPanel`, which `check_void_conditions` reads
        for `AA_CONTROL_FAILED` and `evaluate_search_grade` reads for conjuncts
        8 and 9;
      * `aa_cadence` — *"a passing A/A control within its declared cadence"*;
      * `control_definitions_immutable` — the definitions AND predicates digest
        verification, which voids the window on drift.

    **A result with no panel RAISES.** `ControlPanelResult.panel is None` means
    controls 1-4 could not be assembled, and *"A campaign that cannot run controls
    1-4 MUST NOT rank any candidate"*: there is no `api.ControlPanel` value that
    says "no panel", and inventing one — four `COULD_NOT_CHECK`s, say — would put
    a fabricated panel on the record. The caller journals `result.to_dict()` and
    `result.blocked_reason` and does not open a measurement window.
    """
    if not isinstance(result, ControlPanelResult):
        raise ControlWiringError(
            f"window_control_attestations() takes a ControlPanelResult, got "
            f"{type(result).__name__}")
    if result.panel is None:
        raise ControlWiringError(
            "this control sweep produced no api.ControlPanel"
            + (f" ({result.blocked_reason})" if result.blocked_reason else "")
            + ". A campaign that cannot run controls 1-4 MUST NOT rank any candidate, and "
            "there is no ControlPanel value meaning 'no panel' — a synthesised one would "
            "be a fabricated attestation on the record. Journal the result and do not "
            "open the window.")
    return {
        "controls": result.panel,
        "aa_cadence": result.aa_cadence,
        "control_definitions_immutable": result.definitions_check,
    }


# =============================================================================
# The calibration block — controls supply the material, statistics.py solves it
# =============================================================================

#: The module that OWNS the calibration solve. Not this one.
CALIBRATION_OWNER = ak_statistics.STATISTICS_MODULE_ID


def run_calibration_block(inputs: Any):
    """Delegate the calibration block to `statistics.solve_calibration()`.

    Deliberately a pass-through with a type gate. `statistics.py` implements the
    whole of *"Campaign calibration block — every threshold is derived, none is
    supplied"*: the normative solve order, `phi`, `B_min`, the alpha budgets, the
    once-only empirical validation with its tightening restart, and the
    anchor-gate band. **None of that is re-implemented here.**

    What belongs to the controls harness is the JOIN, and it is a real one: the
    calibration block *"runs on the campaign's own anchor … under the identical
    recipe, claim, interleaving and reduction discipline that candidate rounds
    will use"*, its `phi` is estimated *"from the A/A control"* (control 4), and
    its neutral consistency check is a property of control 2. The material comes
    from the controls; the solve does not.

    A second solve order in this module would be the failure mode this project
    keeps paying for: both implementations would keep returning an accepted
    calibration, so the drift between them would never surface as an error.
    """
    if not isinstance(inputs, ak_statistics.CalibrationInputs):
        raise ControlWiringError(
            f"run_calibration_block() takes a statistics.CalibrationInputs, got "
            f"{type(inputs).__name__}. The calibration solve is owned by "
            f"{CALIBRATION_OWNER}; this module supplies the A/A and neutral material "
            "and reads the neutral verdict back, and implements no solve of its own")
    return ak_statistics.solve_calibration(inputs)


def neutral_dispersion_check(solve: Any) -> schemas.Check:
    """Read control 2's dispersion-vs-floor verdict out of a calibration solve.

    *"The neutral control's `|effect|` distribution is compared against `phi` as a
    consistency check; a neutral control materially exceeding the A/A floor FAILS
    the calibration rather than raising the floor."* That comparison is computed
    inside the solve (it is part of output 1), and this is how it reaches
    `ControlContext.neutral_dispersion`, where the neutral control's own evaluator
    reads it.

    A solve that recorded no noise floor yields COULD_NOT_CHECK, not PASS: the
    comparison did not happen, and "did not happen" is the third outcome rather
    than a satisfied one.
    """
    if not isinstance(solve, ak_statistics.CalibrationSolve):
        raise ControlWiringError(
            f"neutral_dispersion_check() takes a statistics.CalibrationSolve, got "
            f"{type(solve).__name__}")
    for attempt in reversed(tuple(solve.attempts)):
        floor = getattr(attempt, "noise_floor", None)
        if floor is not None and isinstance(getattr(floor, "neutral_check", None),
                                            schemas.Check):
            return floor.neutral_check
    return schemas.Check(
        schemas.COULD_NOT_CHECK,
        ("the calibration solve recorded no noise floor, so the neutral control's "
         "dispersion was never compared against phi; an uncomputed consistency check "
         "is not a passing one",))


# =============================================================================
# Self-audit — reuses api's auditor rather than forking it
# =============================================================================

def audit_no_write_or_process_paths(source: Optional[str] = None) -> schemas.Check:
    """Prove from THIS module's AST that it cannot write, execute, or signal.

    Delegates to `api.audit_no_write_or_process_paths`, which already encodes the
    forbidden call and import sets. A second copy of those sets would drift from the
    first, and the drift would be invisible: both would still return PASS.

    COULD_NOT_CHECK when the source cannot be read — an unreadable module is not an
    audited one.
    """
    if source is None:
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except OSError as exc:
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 (f"could not read {__file__}: {exc}",))
    return api.audit_no_write_or_process_paths(source, module_id=MODULE_ID)
MODULE_ID = "autokernel.evaluator.controls/v1"
