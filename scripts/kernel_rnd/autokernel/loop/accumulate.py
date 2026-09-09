#!/usr/bin/env python3
"""Compound-then-gate: batch cheap bench keeps until their compounded gain clears the
serving floor, THEN spend the expensive serving gate once (R23-44, operator directive
2026-09-04: "collect llama-bench keeps until they compound to 2x-3x noise floor before
the llama-server final champion advancement gate").

WHY THIS EXISTS. R23-43 made the keep gate a llama-server measurement under the champion's
canonical recipe -- correct, because only serving performance matters -- but the serving
metric's noise floor is ~3.5% (per-request aggregate, R23-43 (2)), while an individual
bench keep is 1-3%. A per-keep serving gate therefore vetoes EVERY keep: each one is
smaller than the floor it must clear, so the loop can never advance. That is not the
serving gate being strict; it is a resolution mismatch.

THE FIX IS A TWO-TIER CHAMPION.
  * The ACCUMULATOR (a working champion) advances on every bench keep. The anchor tracks
    it, so successive keeps compound (each is measured marginal against the last), exactly
    as before. This is cheap: llama-bench only.
  * The CHAMPION OF RECORD -- the last serving-DEMONSTRATED commit, the one a promotion
    would ship -- advances ONLY when the accumulator's compounded bench gain over it
    reaches `fire_multiple` x the serving floor, and THEN only if the one serving gate
    fired at that point comes back decisive and positive.

So the serving gate runs RARELY (once per bundle, not once per keep) and only on a bundle
big enough that the ~3.5% floor can resolve it. `fire_multiple` defaults to 2.5 (the
midpoint of the operator's 2-3x), i.e. a bundle must compound past ~8.8% bench before the
serving gate is even attempted.

THE DIVERGENCE CASE IS THE POINT, NOT AN EDGE. 2026-09-04 proved bench can gain while
serving stays flat (dec-b4 +35% -> DFlash2 serving 0%). So a bundle CAN compound past the
threshold on bench and still fail the serving gate. When it does, the champion of record
does NOT move -- the operator's rule is that only a serving win advances it -- and the
bundle is recorded as a measured divergence. What happens to the accumulated commits then
is a policy choice the caller selects (`DivergenceAction`); this module computes the
decision and leaves the git/build mechanics to the loop.

R23-54 (operator 2026-09-08) ADDED A SECOND, MANDATORY TRIGGER. `fire_multiple` is a
heuristic over the proxy, and the gate's first firing showed the proxy failing at that
very job (+5.958% bench vs -2.18% serving, n=10). So the gate now also fires every
`SERVING_GATE_EVERY_KEEPS` = 4 keeps regardless of the compounded estimate, and every
gate record names its `trigger` -- "threshold", "cadence" or "both".

This module is pure: it holds NO build directories and runs NO measurements. The loop
injects the compounded-bench number and the serving-gate row; `accumulate` decides.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import enum
import fcntl
import json
import math
import os
import stat

from .. import journal
from . import status


#: R23-54 (operator ruling 2026-09-08): the serving gate is MANDATORY on a fixed KEEP
#: CADENCE -- it fires every N keeps REGARDLESS of the compounded bench estimate.
#:
#: WHY. Until R23-54 the gate fired on one trigger only: the accumulator's compounded
#: llama-bench estimate crossing `fire_multiple` x the serving floor. That trigger is a
#: HEURISTIC built on the proxy, and the gate's FIRST firing (2026-09-08) measured the
#: proxy failing at exactly the job the trigger gives it: the bench estimate said +5.958%
#: while serving said "cannot tell, probably slightly negative" (-2.18%, n=10). A proxy
#: that cannot decide whether a bundle gained cannot be trusted to decide WHEN to spend
#: the instrument that can. So the bench estimate keeps its trigger -- it is still the
#: cheap early signal -- but it no longer holds a veto over the schedule: after N keeps
#: the gate runs on cadence and the bundle gets a real serving reading either way.
#:
#: N = 4: four 1-3% keeps is the batch size at which the ~3.5% serving floor has a
#: plausible chance of resolving the bundle, and it bounds how far the accumulator can
#: drift from serving-demonstrated reality (at most 4 keeps of unverified bench gain).
SERVING_GATE_EVERY_KEEPS = 4
JOURNAL_DIRNAME = "journal"
MEASUREMENT_CURRENT = "current_snapshot"
MEASUREMENT_UNKNOWN_LEGACY = "unknown_legacy"
MEASUREMENT_STALE_TIP_ADVANCE = "stale_external_tip_advance"
BUNDLE_SCHEMA_V1 = journal.LOOP_BUNDLE_SNAPSHOT_SCHEMA_V1
BUNDLE_SCHEMA_V2 = journal.LOOP_BUNDLE_SNAPSHOT_SCHEMA_V2


class BundleRecoveryRequired(RuntimeError):
    """Authoritative accumulator state cannot be proved safe to resume."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"bundle recovery required: {reason}")


class Decision(enum.Enum):
    """What to do after a bench keep lands on the accumulator."""
    ACCUMULATE = "accumulate"      #: below the fire threshold -- keep batching, no serving spend
    FIRE_SERVING = "fire_serving"  #: compounded gain cleared the threshold -- run the serving gate


class Outcome(enum.Enum):
    """What the serving gate said about the whole bundle."""
    PROMOTE = "promote"        #: decisive positive serving win -- advance the champion of record
    DIVERGED = "diverged"      #: bundle cleared bench but serving did not confirm -- champion holds


class DivergenceAction(enum.Enum):
    """What the loop does with the accumulated commits when a bundle diverges. The caller
    picks one; this module only NAMES the choice so the decision is explicit and logged.

    Operator decision 2026-09-04: HOLD, and write the divergence into the JOURNAL as
    evidence the planner reads -- "keep batching but update journal evidence so the planner
    can strategize accordingly and potentially modify previously batched/bundled keeps".
    So divergence is not a dead end and not a blind pile-on: the bundle stays, the finding
    goes to the planner, and the planner may add keeps aimed at the serving gap OR revert /
    revise specific keeps already in the bundle (it authors in a worktree on the champion
    branch, so a revert is just another authored patch). `resolve` emits that evidence."""
    ROLLBACK = "rollback"   #: reset the accumulator to the champion of record, discard the bundle
    HOLD = "hold"           #: keep the bundle; the planner gets the evidence and may revise it
    #: (a future BISECT action -- automatically find which keeps transferred -- is deliberately
    #: not built: the operator's HOLD+evidence hands that judgement to the PLANNER, which sees
    #: the whole context bundle, rather than spending O(log n) serving gates to bisect blindly.)


@dataclass(frozen=True)
class AccumulatorPolicy:
    """The thresholds. `fire_multiple` is the operator's 2-3x; `on_divergence` is the
    caller's choice for the divergence case (default HOLD: the operator's 2026-09-04 ruling
    -- keep the bundle and hand the divergence to the planner as journal evidence, so it can
    add keeps aimed at the serving gap or revise the keeps already bundled).

    `every_keeps` is R23-54's mandatory cadence (operator 2026-09-08): the second,
    proxy-independent trigger. It is a field only so a test can shorten it; the ruling's
    value is the module constant `SERVING_GATE_EVERY_KEEPS`."""
    fire_multiple: float = 2.5
    on_divergence: DivergenceAction = DivergenceAction.HOLD  # operator 2026-09-04
    every_keeps: int = SERVING_GATE_EVERY_KEEPS              # operator 2026-09-08 (R23-54)

    def fire_threshold_pct(self, serving_floor_pct: float) -> float:
        return self.fire_multiple * serving_floor_pct


@dataclass
class Bundle:
    """The accumulator's state: the champion of record it builds on, its own tip, the
    keeps batched onto it, and the compounded bench gain of tip over champion-of-record
    (RE-MEASURED against the champion of record after each keep, never a product of
    marginal effects -- keeps interact, and the compounded number is what the serving gate
    will be asked to confirm)."""
    champion_of_record: str
    tip: str
    keeps: list = field(default_factory=list)
    compounded_bench_pct: float = 0.0
    #: R23-54: keeps landed since the serving gate last RAN (any outcome), not since it
    #: last promoted -- the cadence measures how long the accumulator has gone without a
    #: real serving reading. Durable like the rest of the bundle, so restarts (which run 29
    #: proved are guaranteed) cannot postpone the gate forever by resetting the count.
    keeps_since_serving_gate: int = 0
    #: Whether compounded_bench_pct was measured for this exact tip.  Legacy v1
    #: JSON predates the field and therefore maps to the matching historical
    #: snapshot; a later external tip advance makes the magnitude stale.
    measurement_validity: str = MEASUREMENT_CURRENT

    def add_keep(self, mechanism_id: str, tip: str, compounded_bench_pct: float) -> None:
        self.keeps.append(mechanism_id)
        self.tip = tip
        self.compounded_bench_pct = compounded_bench_pct
        self.keeps_since_serving_gate += 1
        self.measurement_validity = MEASUREMENT_CURRENT

    def mark_serving_gate_fired(self) -> None:
        """The serving gate RAN -- reset the cadence counter. Called on every outcome
        (PROMOTE, DIVERGED), because the counter tracks readings taken, not verdicts won:
        resetting only on a promote would make a diverging bundle re-fire the expensive
        gate on every subsequent keep."""
        self.keeps_since_serving_gate = 0

    def is_empty(self) -> bool:
        return not self.keeps

    #: The bundle is DURABLE STATE, not per-process state. Measured defect 2026-09-07: it
    #: was constructed fresh from the anchor at every startup, so each restart (a) reset the
    #: keeps to zero and (b) silently advanced the champion of record to the accumulated tip
    #: -- LAUNDERING bench-only keeps into the serving-demonstrated slot they had never
    #: reached. Five keeps and +6.13% were absorbed that way across run 29's restarts, and
    #: the serving gate has NEVER fired: the bundle peaked at +5.19% against an +8.84%
    #: threshold and was reset before it could get there. Persisting it is what makes
    #: "compound until the gate fires" survive the restarts that a long campaign guarantees.
    FILENAME = "accumulator-bundle.json"
    LEGACY_SCHEMA = BUNDLE_SCHEMA_V1
    SCHEMA = BUNDLE_SCHEMA_V2

    def to_dict(self) -> dict:
        return {"schema": self.SCHEMA, "champion_of_record": self.champion_of_record,
                "tip": self.tip, "keeps": list(self.keeps),
                "compounded_bench_pct": self.compounded_bench_pct,
                "keeps_since_serving_gate": int(self.keeps_since_serving_gate),
                "measurement_validity": self.measurement_validity}

    @classmethod
    def from_dict(cls, d: dict) -> "Bundle":
        if not isinstance(d, dict):
            raise ValueError("bundle must be a JSON object")
        schema = d.get("schema")
        if schema not in (cls.LEGACY_SCHEMA, cls.SCHEMA):
            raise ValueError(f"unknown bundle schema {d.get('schema')!r}")
        common = {"schema", "champion_of_record", "tip", "keeps",
                  "compounded_bench_pct"}
        allowed = (common | {"keeps_since_serving_gate"}
                   if schema == cls.LEGACY_SCHEMA else
                   common | {"keeps_since_serving_gate", "measurement_validity"})
        required = (common if schema == cls.LEGACY_SCHEMA else allowed)
        extra = sorted(set(d) - allowed)
        missing = sorted(required - set(d))
        if extra:
            raise ValueError(f"unknown bundle field(s) {extra}")
        if missing:
            raise ValueError(f"missing required bundle field(s) {missing}")
        for key in ("champion_of_record", "tip"):
            if not isinstance(d.get(key), str) or not d[key].strip():
                raise ValueError(f"{key} must be a non-empty string")
        keeps = d["keeps"]
        if (not isinstance(keeps, list)
                or any(not isinstance(value, str) or not value.strip()
                       for value in keeps)):
            raise ValueError("keeps must be a list of non-empty strings")
        gain = d["compounded_bench_pct"]
        if (not isinstance(gain, (int, float)) or isinstance(gain, bool)
                or not math.isfinite(float(gain))):
            raise ValueError("compounded_bench_pct must be a finite number")
        cadence = d.get("keeps_since_serving_gate", 0)
        if cadence is None and schema == cls.LEGACY_SCHEMA:
            cadence = 0
        if (not isinstance(cadence, int) or isinstance(cadence, bool)
                or cadence < 0):
            raise ValueError("keeps_since_serving_gate must be a non-negative integer")
        validity = (MEASUREMENT_UNKNOWN_LEGACY if schema == cls.LEGACY_SCHEMA
                    else d["measurement_validity"])
        if validity not in journal.LOOP_BUNDLE_MEASUREMENT_VALIDITIES:
            raise ValueError(f"unknown measurement_validity {validity!r}")
        # v1 predates explicit measurement validity.  It remains readable as
        # unknown legacy state, but only v2 can represent a current measurement.
        return cls(champion_of_record=d["champion_of_record"], tip=d["tip"],
                   keeps=list(keeps), compounded_bench_pct=float(gain),
                   keeps_since_serving_gate=cadence,
                   measurement_validity=validity)

    def save(self, store: Path) -> Path:
        store = Path(store)
        book = journal.Journal(str(store / JOURNAL_DIRNAME))
        book.initialize()
        snapshot = self.to_dict()
        with book.write_lock():
            _append_snapshot_locked(book, snapshot, provenance="current_snapshot")
            return status.write_json(store, self.FILENAME, snapshot,
                                     prefix=".accumulator-bundle-")


def _saved_payload(snapshot: dict, *, provenance: str) -> dict:
    return {
        "schema": journal.LOOP_BUNDLE_SAVED_SCHEMA,
        "snapshot": snapshot,
        "snapshot_sha256": journal.loop_bundle_snapshot_digest(snapshot),
        "provenance": provenance,
    }


def _last_saved_snapshot(entries: list) -> tuple[dict, str] | None:
    saved = []
    for entry in entries:
        if entry.kind != journal.KIND_LOOP_BUNDLE_SAVED:
            continue
        violations = journal.validate_loop_bundle_saved_payload(entry.payload)
        if violations:
            raise BundleRecoveryRequired(
                f"journal LOOP_BUNDLE_SAVED event {entry.event_id} is invalid: "
                + "; ".join(violations)
            )
        saved.append(entry.payload)
    if not saved:
        return None
    payload = saved[-1]
    return dict(payload["snapshot"]), str(payload["provenance"])


def _read_entries(book: journal.Journal) -> list:
    try:
        return book.read_all()
    except (journal.JournalCorruption, OSError) as exc:
        raise BundleRecoveryRequired(
            f"journal read failure prevents authoritative recovery: {exc}"
        ) from exc


@contextmanager
def _existing_shared_lock(book: journal.Journal):
    """Share an existing Journal lock without creating any path or inode."""
    lock_path = Path(book.root) / journal.LOCK_NAME
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock_path, flags)
    except OSError as exc:
        raise BundleRecoveryRequired(
            f"read-only recovery requires existing journal lock {lock_path}: {exc}"
        ) from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise BundleRecoveryRequired(
                f"read-only recovery journal lock is not a regular file: {lock_path}"
            )
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
        except OSError as exc:
            raise BundleRecoveryRequired(
                f"read-only recovery cannot acquire journal lock {lock_path}: {exc}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _append_snapshot_locked(book: journal.Journal, snapshot: dict, *,
                            provenance: str) -> None:
    entries = _read_entries(book)
    previous = _last_saved_snapshot(entries)
    if previous is not None:
        previous_snapshot, _ = previous
        try:
            if (Bundle.from_dict(previous_snapshot).to_dict()
                    == Bundle.from_dict(snapshot).to_dict()):
                return
        except (TypeError, ValueError) as exc:
            raise BundleRecoveryRequired(
                f"journal bundle snapshot cannot be restored: {exc}"
            ) from exc
    book.append(journal.KIND_LOOP_BUNDLE_SAVED,
                _saved_payload(snapshot, provenance=provenance))


def _read_legacy_projection(path: Path) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, "legacy Bundle JSON is missing"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema") != Bundle.LEGACY_SCHEMA:
            raise ValueError(
                f"only {Bundle.LEGACY_SCHEMA!r} is importable legacy state"
            )
        Bundle.from_dict(raw)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return None, f"legacy Bundle JSON is unreadable or invalid: {exc}"
    return raw, None


def _require_valid_ancestry(bundle: Bundle, anchor_commit: str, is_ancestor) -> None:
    try:
        cor_precedes_tip = bool(is_ancestor(bundle.champion_of_record, bundle.tip))
        tip_precedes_anchor = bool(is_ancestor(bundle.tip, anchor_commit))
    except Exception as exc:
        raise BundleRecoveryRequired(f"commit ancestry check failed: {exc}") from exc
    if not cor_precedes_tip:
        raise BundleRecoveryRequired(
            f"invalid COR/tip ancestry: champion of record "
            f"{bundle.champion_of_record[:12]} is not an ancestor of tip "
            f"{bundle.tip[:12]}"
        )
    if not tip_precedes_anchor:
        raise BundleRecoveryRequired(
            f"invalid tip/anchor ancestry: persisted tip {bundle.tip[:12]} is not "
            f"an ancestor of anchor {anchor_commit[:12]}"
        )


def load_bundle(store: Path, *, anchor_commit: str, is_ancestor,
                read_only: bool = False) -> tuple:
    """Restore authoritative journal state, importing legacy JSON only once.

    The JSON file is a projection. A write-capable caller may initialize an
    absent or empty store with the supplied baseline and no accumulated keeps.
    Missing state in a populated store, or invalid lineage/journal integrity,
    raises ``BundleRecoveryRequired``; existing history is never reset. Read-only
    consumers cannot invent a baseline for missing historical state.
    """
    store = Path(store)
    p = store / Bundle.FILENAME
    journal_root = store / JOURNAL_DIRNAME
    legacy, legacy_error = _read_legacy_projection(p)

    journal_exists = journal_root.exists()
    if not journal_exists and legacy is None:
        if read_only:
            raise BundleRecoveryRequired(f"no accumulator state: {legacy_error}")
        try:
            if store.is_symlink():
                raise BundleRecoveryRequired("new accumulator store cannot be a symlink")
            with os.scandir(store) as contents:
                if next(contents, None) is not None:
                    raise BundleRecoveryRequired(f"no accumulator state: {legacy_error}")
        except FileNotFoundError:
            pass  # A genuinely new store, not a lost snapshot in an existing history.
        except OSError as exc:
            raise BundleRecoveryRequired(f"cannot inspect new accumulator store: {exc}") from exc
        baseline = Bundle.from_dict(Bundle(
            champion_of_record=anchor_commit, tip=anchor_commit).to_dict())
        _require_valid_ancestry(baseline, anchor_commit, is_ancestor)
        book = journal.Journal(str(journal_root))
        book.initialize()
        with book.write_lock():
            # A concurrent initializer must not be overwritten by this baseline.
            # Nor may a file that appeared after the empty-store check be hidden.
            if _read_entries(book) or any(
                    item.name != JOURNAL_DIRNAME for item in store.iterdir()):
                raise BundleRecoveryRequired("accumulator store changed during initialization; retry recovery")
            snapshot = baseline.to_dict()
            _append_snapshot_locked(book, snapshot, provenance="current_snapshot")
            status.write_json(store, Bundle.FILENAME, snapshot,
                              prefix=".accumulator-bundle-")
        return baseline, "no persisted bundle — initialized a new empty baseline"

    if read_only and not journal_exists:
        # Dry consumers may inspect an importable v1 projection, but may not
        # create the journal, import it, or repair the projection.
        try:
            b = Bundle.from_dict(legacy)
        except (TypeError, ValueError) as exc:  # defensive; reader validated it
            raise BundleRecoveryRequired(f"legacy snapshot cannot be restored: {exc}") from exc
        _require_valid_ancestry(b, anchor_commit, is_ancestor)
        if b.tip != anchor_commit:
            b.tip = anchor_commit
            b.measurement_validity = MEASUREMENT_STALE_TIP_ADVANCE
        return b, (f"read-only legacy v1 state: restored {len(b.keeps)} keep(s) "
                   f"with {b.measurement_validity} measurement")

    book = journal.Journal(str(journal_root))
    if not journal_exists:
        book.initialize()

    lock = _existing_shared_lock(book) if read_only else book.write_lock()
    with lock:
        entries = _read_entries(book)
        saved = _last_saved_snapshot(entries)
        provenance = "current_snapshot"
        if saved is None:
            if legacy is None:
                raise BundleRecoveryRequired(
                    "journal contains no LOOP_BUNDLE_SAVED snapshot and "
                    f"{legacy_error}"
                )
            snapshot = legacy
            provenance = "imported_legacy_state"
        else:
            snapshot, provenance = saved
        try:
            b = Bundle.from_dict(snapshot)
        except (TypeError, ValueError) as exc:
            raise BundleRecoveryRequired(
                f"authoritative journal snapshot cannot be restored: {exc}"
            ) from exc

        _require_valid_ancestry(b, anchor_commit, is_ancestor)
        if saved is None and not read_only:
            # An invalid legacy lineage must fail above without ever becoming
            # authoritative journal state.
            _append_snapshot_locked(book, snapshot,
                                    provenance="imported_legacy_state")
        if b.tip != anchor_commit:
            # Preserve the provable COR and cadence/keep history, but an external tree
            # advance invalidates the old tip-vs-COR magnitude.  Keep that magnitude as
            # historical data and make validity the authority-bearing status.
            b.tip = anchor_commit
            b.measurement_validity = MEASUREMENT_STALE_TIP_ADVANCE
            if not read_only:
                _append_snapshot_locked(book, b.to_dict(), provenance="current_snapshot")
                status.write_json(store, Bundle.FILENAME, b.to_dict(),
                                  prefix=".accumulator-bundle-")
            return b, (f"restored {len(b.keeps)} keep(s) from {provenance}; tip advanced "
                       "to the anchor and compounded measurement marked stale")
        if not read_only:
            status.write_json(store, Bundle.FILENAME, b.to_dict(),
                              prefix=".accumulator-bundle-")
        return b, (f"restored {len(b.keeps)} keep(s), "
                   f"{b.compounded_bench_pct:+.2f}% vs cor "
                   f"{b.champion_of_record[:12]} from {provenance}")


def gate_trigger(bundle: Bundle, serving_floor_pct: float | None,
                 policy: AccumulatorPolicy) -> str | None:
    """WHY the serving gate should fire now, or None to keep accumulating.

    Two independent triggers (R23-54, operator 2026-09-08):
      * "threshold" -- the compounded bench estimate cleared `fire_multiple` x floor. The
        original R23-44 trigger: cheap, early, and a PROXY.
      * "cadence"   -- `every_keeps` keeps have landed since the gate last RAN. Mandatory,
        and deliberately blind to the bench estimate, because 2026-09-08 measured the
        estimate (+5.958%) disagreeing with serving (-2.18%, n=10) on the same bundle.
      * "both"      -- both hold; recorded distinctly so a reader is never left guessing
        which one carried the firing.

    THE FAIL-CLOSED GUARD IS UNCHANGED. An uncalibrated floor (None) blocks BOTH triggers:
    without a floor the gate's own `decisive` is None, so it can only ever return DIVERGED
    (`classify_serving`). Cadence-firing there would spend hours of llama-server time on a
    reading that cannot promote anything -- that is not strictness the ruling removed, it
    is a gate that cannot judge, which R23-43's grammar has always refused to spend."""
    if serving_floor_pct is None:
        return None
    threshold = (
        bundle.measurement_validity == MEASUREMENT_CURRENT
        and bundle.compounded_bench_pct >= policy.fire_threshold_pct(serving_floor_pct)
    )
    cadence = (policy.every_keeps > 0
               and bundle.keeps_since_serving_gate >= policy.every_keeps)
    if threshold and cadence:
        return "both"
    if threshold:
        return "threshold"
    if cadence:
        return "cadence"
    return None


def decide_after_keep(bundle: Bundle, serving_floor_pct: float,
                      policy: AccumulatorPolicy) -> Decision:
    """After a keep lands: fire the serving gate, or keep batching? Fires when EITHER
    trigger in `gate_trigger` holds -- the compounded bench estimate clearing
    `fire_multiple` x floor, OR R23-54's mandatory every-`every_keeps` cadence."""
    return (Decision.FIRE_SERVING
            if gate_trigger(bundle, serving_floor_pct, policy) is not None
            else Decision.ACCUMULATE)


def classify_serving(serving_row: dict) -> Outcome:
    """Read the serving A/B row (serving.compare output) for the whole bundle. PROMOTE
    only on a decisive, positive serving effect; anything else -- uncalibrated (decisive
    None), within-floor, or a regression -- is DIVERGED, and the champion of record holds.
    This is the same fail-closed grammar as R23-43's per-keep gate, applied to the bundle."""
    if not serving_row.get("decisive"):
        return Outcome.DIVERGED
    if serving_row.get("effect", 0.0) <= 0:
        return Outcome.DIVERGED
    return Outcome.PROMOTE


def resolve(bundle: Bundle, serving_row: dict, policy: AccumulatorPolicy) -> dict:
    """Fold the serving gate's verdict into a caller-actionable plan. Returns the outcome,
    the divergence action (only meaningful when DIVERGED), a one-line reason for the log,
    and the new champion of record the loop should record on a PROMOTE."""
    outcome = classify_serving(serving_row)
    if outcome is Outcome.PROMOTE:
        return {
            "outcome": outcome, "action": None,
            "new_champion_of_record": bundle.tip,
            "reason": (f"bundle of {len(bundle.keeps)} keep(s) (+{bundle.compounded_bench_pct:.2f}% "
                       f"bench) confirmed on serving: {serving_row.get('effect_pct', 0.0):+.3f}% "
                       f"(floor {serving_row.get('noise_floor_pct')}%) -> champion of record "
                       f"advances to {bundle.tip[:12]}")}
    return {
        "outcome": outcome, "action": policy.on_divergence,
        "new_champion_of_record": bundle.champion_of_record,
        "reason": (f"DIVERGENCE: bundle of {len(bundle.keeps)} keep(s) reached "
                   f"+{bundle.compounded_bench_pct:.2f}% bench but serving said "
                   f"{serving_row.get('effect_pct', 0.0):+.3f}% (decisive={serving_row.get('decisive')}, "
                   f"floor {serving_row.get('noise_floor_pct')}%); champion of record HOLDS at "
                   f"{bundle.champion_of_record[:12]}, action={policy.on_divergence.value}"),
        # The planner READS this next iteration (operator 2026-09-04): it is the evidence that
        # a bench-only gain did not transfer, and it names the keeps in the bundle so the planner
        # can revert or revise a specific one instead of only appending. NOT a grade -- a
        # strategy signal, so the planner stops mining a non-predictive vein.
        "planner_evidence": {
            "kind": "serving_divergence",
            "bundled_keeps": list(bundle.keeps),
            "compounded_bench_pct": bundle.compounded_bench_pct,
            "serving_effect_pct": serving_row.get("effect_pct", 0.0),
            "serving_decisive": serving_row.get("decisive"),
            "serving_floor_pct": serving_row.get("noise_floor_pct"),
            "hint": ("bench gains in this bundle did not transfer to serving; consider reverting or "
                     "revising one of the bundled keeps, or aim the next hypothesis at the serving "
                     "gap rather than the bench surface")}}


__all__ = ["SERVING_GATE_EVERY_KEEPS", "Decision", "Outcome", "DivergenceAction",
           "AccumulatorPolicy", "Bundle", "gate_trigger", "decide_after_keep",
           "classify_serving", "resolve"]
