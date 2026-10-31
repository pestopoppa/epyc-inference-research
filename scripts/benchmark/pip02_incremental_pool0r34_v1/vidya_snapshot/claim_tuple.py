"""The measurement ingestion contract: one grammar, one grader, many projections.

WHY THIS EXISTS. Different loops record measurements in different shapes — an autopilot trial row,
an AutoKernel `evaluation_event`, a sealed benchmark manifest, a llama-bench sweep. Each adapter
that arrived was written with its own reading of `MEASUREMENT_POLICY.md`, and on 2026-08-10 two of
them were caught disagreeing about the same case:

    record with no protocol and no attestation
      measurement_record.grade()  ->  Judged/T0
      sealed_manifest.grade()     ->  Judged/Located

Same constitution, same rule, two answers on the T axis. Neither is obviously wrong, which is the
point: a rule reimplemented per source becomes N dialects of itself, and the divergence surfaces as
unexplainable grade differences between corpora long after anyone remembers there were two
functions. The substrate exists to catch exactly this class of defect, so it may not contain it.

THE CONTRACT. An adapter's only job is **projection**: map its native record into the canonical
tuple below. It never grades, and it never invents an element it cannot find — a missing element is
reported and grades the claim down, which is a true statement about the measurement.

    native record  --project-->  ClaimTuple  --grade()-->  (Q, T, reasons)  --> frames

The vocabulary is not invented here either. It is AutoKernel's `claim_grammar`
(`epyc-inference-research` `scripts/kernel_rnd/autokernel/schemas.py`), which already enforces
`MEASUREMENT.md:13` as a REQUIRED schema block. Aligning on it means the strictest existing
producer defines the shape, rather than the newest adapter redefining it.

WHAT GRADING MEANS. Straight from the constitution's own words, one ladder for every source:

  * full tuple, artifact present and hashed  -> `Witnessed/Attested`  (decision-gating)
  * full tuple, artifact named but unhashed  -> `Witnessed/Anchored`  (re-derivable, not pinned)
  * protocol cited, tuple incomplete         -> `Verified/Located`    (a result, not gating)
  * no protocol citation                     -> `Judged/…`            (an OBSERVATION)

The last row is load-bearing. An observation is worth recording — it is what hypotheses are made of
— and grading it honestly at `Judged` is what stops it being cited later as though it had gated a
decision.

One post-step sits after that ladder, for verifier-class tuples only (SC56): a verifier result
whose decided proposition is not bound to the claim it is cited for is an observation about
something else, and caps at `Judged`; a bound one caps at `Verified`, per §4.5. T is never touched
— traceability is a different axis, and an unbound receipt can still be perfectly located.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]

# Borrowed verbatim from AutoKernel schemas.py. Kept as literals rather than imported: the research
# repo is a sibling working tree that may be absent, and a conformance test pins that the two
# vocabularies stay identical, which is stronger than an import that silently follows a rename.
CATEGORIES = frozenset({"OPTIMUM", "BASELINE", "CANDIDATE"})
METRIC_DIRECTIONS = frozenset({"higher_better", "lower_better"})

# The source classes. Declared HERE rather than beside their prose (below, with the SC57 verifier
# contract) only because `ClaimTuple.source_class` defaults to one of them and a dataclass default
# is evaluated at class-creation time. The rationale for each class lives where it is argued.
MEASUREMENT_CLASS = "measurement"
LITERATURE_CLASS = "literature"
VERIFIER_CLASS = "verifier"
SOURCE_CLASSES = frozenset({MEASUREMENT_CLASS, LITERATURE_CLASS, VERIFIER_CLASS})

# SC56 -- how a verifier's decided proposition is bound to the claim it is cited for.
#
#   ""          no binding recorded. The honest default, and the one that grades down.
#   "identity"  the proposition and the claim are the SAME statement, machine-checkable by
#               normalized string equality (`propositions_are_identical` below).
#   "attested"  a person judged that the claim follows from the proposition, and `binding_ref`
#               points at that judgment.
#
# `attested` has NO PRODUCER yet: SC61 (`claim_statement_binding/v1` -- the frame type, the review
# worksheet and the fold pass) owns it, and until that lands the kind is accepted and validated
# here but unreachable in practice. Deliberately filed apart: letting the producer ride inside SC56
# would have turned a one-function cap into a new frame type under one checkbox.
BINDING_NONE = ""
BINDING_IDENTITY = "identity"
BINDING_ATTESTED = "attested"
BINDING_KINDS = frozenset({BINDING_NONE, BINDING_IDENTITY, BINDING_ATTESTED})


class ProjectionError(ValueError):
    """An adapter produced something that is not a claim tuple."""


def normalize_proposition(text: str) -> str:
    """The ONLY normalization an `identity` binding is allowed to survive.

    Three operations, and this list is exhaustive on purpose:

      1. collapse all internal whitespace runs to one space (and strip the ends);
      2. casefold;
      3. strip trailing sentence punctuation.

    That is it. NO stemming, NO synonym or lemma handling, NO fuzzy ratio, NO token-set overlap,
    NO stopword removal. Every one of those would let two DIFFERENT statements normalize to the
    same string, and the whole load-bearing property of `identity` is that a machine can decide it
    with no judgment involved. The moment a similarity threshold appears here, "the checker decided
    this claim" becomes an opinion held by a string metric -- which is precisely the gap SC56
    exists to close (`intake-1307#00`: an automated check labelled 73.6% of proved artifacts
    non-trivial-and-correct where a manual audit put faithfulness far lower).

    A claim that genuinely follows from the proposition without being the same sentence is a real
    and common case. It is `binding_kind='attested'` -- a human judgment, with a reference -- and
    never a looser rule here.
    """
    return " ".join(str(text or "").split()).casefold().rstrip(".,;:!?")


def propositions_are_identical(decided: str, claim: str) -> bool:
    """True iff the decided proposition and the claim are the same statement (see above)."""
    left, right = normalize_proposition(decided), normalize_proposition(claim)
    return bool(left) and left == right


# --- applicability and observation: WHERE a claim holds and WHAT was measured --------------
#
# Both are inert to grading. `applicability` is the native scope (VB-APPLICABILITY, extended by
# the loop-belief redesign §4.1); `observation` is the measured content a planner needs -- the
# anchor, the relative effect, the floor it was judged against, the producer's own verdict and
# the mechanism. Neither is ever inferred on read: an adapter projects what the record states.

APPLICABILITY_KEYS = frozenset({
    "model_file", "quant", "backend", "device", "context_tokens", "kernel", "run_id",
    "run_expected_keys",
    # loop-belief redesign §4.1/§4.2 -- each only when the native record (or, for
    # `scope_basis="store_manifest"`, the store's pinned manifest) states it.
    "model_sha256", "build_sha", "epoch", "recipe", "np", "surface", "symbol", "op", "shape",
    "scope_basis", "role", "stack_regime", "config_key",
    # §12.2/§12.3 (knob grain): the knob family a record bears on (one name or a list), the flag
    # set it held fixed (`knobs`), the knob(s) its arms change (`varied`), and an AutoPilot
    # trial's role -> model_file map. All native declarations; inert to grading.
    "knob_family", "knobs", "varied", "role_models",
    # serving-floor key (VB-FLOOR-REGISTRY-WIRE): native declarations of the floor registry entry; inert to grading.
    "equivalence_key", "recipe_hash", "request_digest", "model_sha", "numerics_epoch", "claim_class",
    "threads", "co_tenancy",
})
# `native`: every scope key came from the record itself. `store_manifest`: the record carries no
# scope of its own and the store's manifest pinned it (the SOURCE_SCOPES basis, §3.1).
SCOPE_BASES = frozenset({"native", "store_manifest"})

OBSERVATION_KEYS = frozenset({
    "anchor_value", "effect_rel", "noise_floor_rel", "decisive", "n_pairs", "outcome",
    "outcome_native", "outcome_reason", "mechanism", "retrofit",
    # Lane C link (2026-10-21): an experiment-row claim names the capture-level serving claim
    # its numbers live on, instead of copying them. `legacy-serving-v2:<capture_id>` (or the
    # same with the `clm_` prefix) -- the join key is the capture id, nothing looser.
    "serving_claim_id",
})
_SERVING_CLAIM_ID = re.compile(r"(clm_)?legacy-serving-v2:[0-9a-f]{64}")
# The normalized outcome vocabulary (§4.1). `outcome_native` keeps the producer's own status
# verbatim beside it, so the normalization can always be audited against the record.
OBSERVATION_OUTCOMES = frozenset({
    "improved", "null", "regressed", "refused_correctness", "refused_scope", "divergence",
    "anchor", "floor",
})
MECHANISM_KEYS = frozenset({"id", "statement", "falsifier"})
_OBS_NUMBERS = ("anchor_value", "effect_rel", "noise_floor_rel")
_OBS_TEXT_LIMITS = {"outcome_native": 120, "outcome_reason": 300}
_MECHANISM_LIMITS = {"id": 200, "statement": 240, "falsifier": 240}
_HEX64 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{7,64}")


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


FRAME_FLOAT_FORMAT = ".12g"


def frame_scalar(value: Any) -> Any:
    """The frame encoding of one observed number -- THE rule, used for every value-bearing key.

    `canonical.py` refuses floats outright (the certified path carries none), but a measured rate
    or a relative effect is a float. So:

      * a float is carried as a decimal STRING with 12 significant digits,
        `format(x, ".12g")` -- e.g. 0.0205 -> "0.0205", 81.1165241567493 -> "81.1165241567",
        81.0 -> "81", 1.5e-05 -> "1.5e-05" (Python's `g` may use an exponent);
      * an int stays a JSON int; bool, str and None stay themselves.

    Adapters put plain Python numbers in `value` / `observation`; only the carrier encodes. A
    reader recovers every numeric key with `float(x)` (an int or such a string), never guessing
    a precision.
    """
    if isinstance(value, float):
        return format(value, FRAME_FLOAT_FORMAT)
    return value


def frame_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """`observation` in its frame encoding (floats -> decimal strings, nested mechanism copied)."""
    out: dict[str, Any] = {}
    for key, value in observation.items():
        out[key] = dict(value) if isinstance(value, dict) else frame_scalar(value)
    return out


def _validate_observation(tup: "ClaimTuple") -> None:
    """Refuse an observation that is not what §4.1 declares. Missing keys are fine (a record
    states what it states); a wrong type, an unknown key or an over-long verbatim text is not."""
    obs = tup.observation
    if not isinstance(obs, dict):
        raise ProjectionError("observation must be a mapping")
    if not obs:
        return
    unknown = set(obs) - OBSERVATION_KEYS
    if unknown:
        raise ProjectionError(f"observation contains unknown fields: {sorted(unknown)}")
    for name in _OBS_NUMBERS:
        if name in obs and not _finite_number(obs[name]):
            raise ProjectionError(f"observation.{name} must be a finite number")
    for name in ("decisive", "retrofit"):
        if name in obs and type(obs[name]) is not bool:
            raise ProjectionError(f"observation.{name} must be a boolean")
    if "n_pairs" in obs and not (type(obs["n_pairs"]) is int and obs["n_pairs"] >= 1):
        raise ProjectionError("observation.n_pairs must be a positive integer")
    if "outcome" in obs and obs["outcome"] not in OBSERVATION_OUTCOMES:
        raise ProjectionError(
            f"observation.outcome must be one of {sorted(OBSERVATION_OUTCOMES)} "
            f"(got {obs['outcome']!r}); the producer's own word goes in outcome_native")
    for name, limit in _OBS_TEXT_LIMITS.items():
        if name in obs:
            value = obs[name]
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise ProjectionError(
                    f"observation.{name} must be a non-empty string of at most {limit} chars "
                    "(the adapter truncates the verbatim text; the carrier never does)")
    if "serving_claim_id" in obs and not (
            isinstance(obs["serving_claim_id"], str)
            and _SERVING_CLAIM_ID.fullmatch(obs["serving_claim_id"])):
        raise ProjectionError(
            "observation.serving_claim_id must be `legacy-serving-v2:<64-hex capture id>` "
            "(optionally `clm_`-prefixed)")
    if "mechanism" in obs:
        mech = obs["mechanism"]
        if not isinstance(mech, dict) or "id" not in mech or set(mech) - MECHANISM_KEYS:
            raise ProjectionError(
                f"observation.mechanism must be a mapping with an `id` and only "
                f"{sorted(MECHANISM_KEYS)}")
        for name, value in mech.items():
            if (not isinstance(value, str) or not value.strip()
                    or len(value) > _MECHANISM_LIMITS[name]):
                raise ProjectionError(
                    f"observation.mechanism.{name} must be a non-empty string of at most "
                    f"{_MECHANISM_LIMITS[name]} chars")
    # A value-bearing observation is about a NUMBER; the number it is about must be one.
    if not _finite_number(tup.value):
        raise ProjectionError(
            "a tuple carrying an observation must carry a finite numeric `value` -- the "
            "observation is emitted beside it on the evidence frame")
    if __package__:
        from .canonical import CanonicalizationError, canonical_bytes
    else:
        from canonical import CanonicalizationError, canonical_bytes
    try:
        canonical_bytes({"observation": frame_observation(obs), "value": frame_scalar(tup.value),
                         "unit": tup.unit, "metric": tup.metric})
    except CanonicalizationError as exc:
        raise ProjectionError(f"observation is not canonical JSON: {exc}") from exc


@dataclass(frozen=True)
class ClaimTuple:
    """`MEASUREMENT.md:13` — (metric, protocol-id, n/reps, date, attestation ref) — plus the two
    labels `:85-95` requires so a number is never unlabelled: `category` and `metric_direction`."""

    measurement_id: str
    metric: str
    value: Any
    date: str
    category: str
    claim: str
    metric_direction: str = "higher_better"
    protocol_id: str = ""
    reps: int | None = None
    # Whether `reps` counted what SCORED or merely what was ATTEMPTED. "n=55 attempted" and
    # "n=55 scored" are different claims; a tuple that cannot tell them apart overstates its sample.
    reps_basis: str = ""
    unit: str = ""
    attestation_path: str = ""
    attestation_sha256: str = ""
    attestation_locator: str = ""
    # Whether the attested artifact exists. `None` means "derive it from `attestation_path`", which
    # is right when the path IS the artifact. A projector that determines presence some other way
    # — a sealed manifest checks its `authority/*` files, not the manifest itself — sets it
    # explicitly. Without this the ladder silently downgraded every sealed run to Anchored.
    attestation_present: bool | None = None
    # SC69 — whether `attestation_sha256` was recomputed over the artifact at the ADAPTER/WRITE
    # boundary and matched. `grade()` is a pure function of the tuple and hashing is I/O, so the
    # check lives where attestations are constructed and its result is carried HERE. `None` (the
    # default) means "not checked": the digest is a recorded claim, nothing more, and never
    # reaches `Attested` — `Attested` means the artifact was re-read and matched, not that a
    # 64-character string was typed. `True` is set only by a write path that re-derived the
    # digest (`verify_attestation` is the tool). `False` is REFUSED: a tuple that records a
    # digest WHILE recording that the digest did not match asserts two contradictory facts.
    attestation_verified: bool | None = None
    source_kind: str = "measurement"
    # SC57 — the verifier-class element. WHAT THE CHECK ASSERTED, in the checker's own terms: the
    # theorem statement a prover discharged, the property a validator decided, the postcondition a
    # test pinned. A boolean records that SOMETHING passed; only this records WHAT. A machine-checked
    # verdict certifies the proposition the checker decided and binds it to nothing else — the
    # certificate "does not certify individual mathematical truth" (`intake-1307#05`), and in one
    # real pipeline an automated check labelled 73.6% of proved artifacts non-trivial-and-correct
    # while a manual audit put faithfulness far lower (`intake-1307#00`, admissible as an existence
    # proof that the gap is large, never as a rate). Empty on non-verifier classes, and NEVER
    # inferred on read: a proposition invented after the fact claims warrant the original check
    # never captured.
    decided_proposition: str = ""
    # SC56 -- which ladder this tuple is graded against, and whether its decided proposition is
    # bound to the claim. `source_class` is the ONLY thing the SC56 cap keys on: a tuple that
    # merely carries a `decided_proposition` is not thereby a verifier result, and scoping the cap
    # on the field rather than the class would have re-graded every adapter that ever records one.
    source_class: str = MEASUREMENT_CLASS
    binding_kind: str = BINDING_NONE
    # Where the `attested` judgment lives (SC61's `claim_statement_binding/v1` frame id, once that
    # exists). Required when, and only when, `binding_kind == "attested"`.
    binding_ref: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    # VB-APPLICABILITY: native declarations only, inert with respect to grading.
    # Run identity and expected keys, when declared, live in this same scope.
    applicability: dict[str, Any] = field(default_factory=dict)
    # Loop-belief redesign §4.1: WHAT was measured -- the anchor, the effect, the floor, the
    # producer's own outcome and the mechanism -- so a planner reads a number, not a sentence.
    # Inert to grading exactly like `applicability`: `grade()` never reads it, and a test pins
    # grade(t) == grade(replace(t, observation={})) for every fixture. `to_frames` emits it
    # (with metric/value/unit) into the EVIDENCE assertion only when present, so every tuple
    # that carries none keeps its byte-identical frames (the SC57 precedent).
    # Last field preserves existing positional constructors.
    observation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.applicability, dict):
            raise ProjectionError("applicability must be a mapping")
        if set(self.applicability) - APPLICABILITY_KEYS:
            raise ProjectionError("applicability contains unknown fields")
        for name, value in self.applicability.items():
            if name == "context_tokens" or name == "np":
                valid = type(value) is int and value > 0
            elif name == "kernel":
                valid = (isinstance(value, str) and bool(value.strip())) or (
                    isinstance(value, dict) and bool(value))
            elif name == "run_expected_keys":
                valid = isinstance(value, list) and bool(value) and all(
                    isinstance(key, str) and key.strip() for key in value)
                if valid:
                    valid = len(value) == len(set(value))
            elif name == "model_sha256":
                valid = isinstance(value, str) and _HEX64.fullmatch(value) is not None
            elif name == "build_sha":
                valid = isinstance(value, str) and _GIT_SHA.fullmatch(value) is not None
            elif name == "scope_basis":
                valid = value in SCOPE_BASES
            elif name == "threads":
                valid = type(value) is int and value > 0
            elif name == "knob_family":
                valid = (isinstance(value, str) and bool(value.strip())) or (
                    isinstance(value, list) and bool(value) and len(value) == len(set(value))
                    and all(isinstance(v, str) and v.strip() for v in value))
            elif name == "varied":
                valid = (isinstance(value, list) and bool(value) and len(value) == len(set(value))
                         and all(isinstance(v, str) and v.strip() for v in value))
            elif name in ("knobs", "role_models"):
                # str -> str | int | bool. Floats are refused: canonical frames carry none, and a
                # fixed flag value is a label, not a measurement.
                valid = isinstance(value, dict) and bool(value) and all(
                    isinstance(k, str) and k.strip()
                    and (type(v) in (int, bool) or (isinstance(v, str) and v.strip()))
                    for k, v in value.items())
            else:
                valid = isinstance(value, str) and bool(value.strip())
            if not valid:
                raise ProjectionError(f"applicability.{name} is malformed")
        if self.applicability:
            if __package__:
                from .canonical import CanonicalizationError, canonical_bytes
            else:
                from canonical import CanonicalizationError, canonical_bytes
            try:
                canonical_bytes(self.applicability)
            except CanonicalizationError as exc:
                raise ProjectionError(f"applicability is not canonical JSON: {exc}") from exc
        _validate_observation(self)
        # Only IDENTITY and LABELS are structural. `date`, `protocol_id`, `reps` and the
        # attestation are gradable elements: a measurement missing them is a real measurement with
        # a low grade, and refusing to represent it would delete the very thing the ladder exists
        # to describe. An earlier draft required `date` here and made dateless runs unrepresentable
        # — caught by the sealed-manifest tests, which is exactly what they are for.
        for name in ("measurement_id", "metric", "claim"):
            if not str(getattr(self, name) or "").strip():
                raise ProjectionError(f"{name} is required and must be non-empty")
        if self.category not in CATEGORIES:
            raise ProjectionError(
                f"category must be exactly one of {sorted(CATEGORIES)} (got {self.category!r}) — "
                "MEASUREMENT.md:85-95, an unlabelled measurement is not decision-grade")
        if self.metric_direction not in METRIC_DIRECTIONS:
            raise ProjectionError(
                f"metric_direction must be one of {sorted(METRIC_DIRECTIONS)} "
                f"(got {self.metric_direction!r}) — a number whose direction is unknown cannot be "
                "compared to anything")
        if self.reps is not None and (not isinstance(self.reps, int) or self.reps < 1):
            raise ProjectionError("reps must be a positive integer when present; zero reps is not "
                                  "a measurement")
        if self.attestation_sha256 and len(self.attestation_sha256) != 64:
            raise ProjectionError("attestation_sha256 must be a 64-character hex digest")
        # SC69. A carried verification without a digest to verify is a lie about a structural
        # fact (nothing was verified); a carried MISmatch records the digest AND that the digest
        # is wrong, which is the same lie with the contradiction stated. Missing (None) is the
        # honest default and grades down.
        if self.attestation_verified is not None and not self.attestation_sha256:
            raise ProjectionError(
                "attestation_verified without attestation_sha256: nothing was verified. Record "
                "the digest the artifact was checked against, or leave verification unclaimed "
                "(SC69)")
        if self.attestation_verified is False:
            raise ProjectionError(
                "attestation_verified=False is refused, never downgraded: recording a digest "
                "WHILE recording that it does not match the artifact asserts two contradictory "
                "facts. Re-verify against the current artifact, or record no digest and take "
                "the grade that describes what is known (SC69)")
        if not isinstance(self.decided_proposition, str):
            raise ProjectionError(
                "decided_proposition must be a string — the proposition the check actually "
                "decided, in the checker's own terms")
        if self.source_class not in SOURCE_CLASSES:
            raise ProjectionError(
                f"source_class must be one of {sorted(SOURCE_CLASSES)} (got "
                f"{self.source_class!r}) — a new class of warrant is never an accident")
        # SC56. The rest of this method distinguishes a MISSING element from a FALSE one, and the
        # distinction is the whole point:
        #
        #   missing binding  -> represented, reported in `reasons`, and GRADED DOWN. "we do not
        #                       know that this check decided the claim" is a true statement about
        #                       the measurement, and §4.7 requires it to be representable.
        #   false binding    -> REFUSED here, like `category`. Asserting `identity` between two
        #                       different statements is not a gap in the record, it is a lie about
        #                       a structural fact, and letting it decay into a downgrade would let
        #                       a wrong claim be recorded at a defensible-looking grade.
        #
        # Different failures, different answers.
        if self.binding_kind not in BINDING_KINDS:
            raise ProjectionError(
                f"binding_kind must be one of {sorted(BINDING_KINDS)} (got "
                f"{self.binding_kind!r}) — how the decided proposition reaches the claim is "
                "recorded, never assumed (SC56)")
        if self.binding_kind and not str(self.decided_proposition or "").strip():
            raise ProjectionError(
                f"binding_kind={self.binding_kind!r} with no `decided_proposition` — a binding "
                "with nothing on the other end of it. Record what the check decided, or record "
                "no binding and take the grade that describes what is actually known")
        if self.binding_kind == BINDING_ATTESTED and not str(self.binding_ref or "").strip():
            raise ProjectionError(
                "binding_kind='attested' requires `binding_ref` — an attested binding IS a human "
                "judgment, and a judgment nobody can point at is indistinguishable from none "
                "(SC61 owns the producer that mints these references)")
        if self.binding_kind == BINDING_IDENTITY and not propositions_are_identical(
                self.decided_proposition, self.claim):
            raise ProjectionError(
                "binding_kind='identity' asserts the checker decided THIS claim, but the two "
                f"statements differ: decided {self.decided_proposition!r} vs claim {self.claim!r}. "
                "Identity is string equality under a deliberately strict normalization; anything "
                "looser is paraphrase detection, which is a judgment and belongs in "
                "binding_kind='attested'")


def artifact_present(tup: ClaimTuple) -> bool:
    """True when the attestation names a file actually on disk.

    A hash over an artifact that no longer exists proves nothing — the constitution's own reason
    for making the top T level require the artifact to be present, not merely referenced.

    Containment is checked on the UNRESOLVED path: reject absolute paths and any `..` component.
    Resolving first would follow `repos/<name>` out to its real location under /mnt/raid0 (the
    working-tree symlink every repo here uses) and reject a legitimate sibling-repo artifact as an
    escape, while `../../etc/passwd` is still caught by the `..` test.
    """
    if tup.attestation_present is not None:
        return tup.attestation_present
    if not tup.attestation_path:
        return False
    rel = PurePosixPath(tup.attestation_path)
    if rel.is_absolute() or ".." in rel.parts:
        return False
    return (REPO_ROOT / rel).is_file()


def verify_attestation(tup: ClaimTuple) -> bool:
    """SC69 — the write-boundary check: re-read the artifact and compare its sha256.

    The ONLY function here that opens the artifact, and deliberately never called by `grade()`:
    grading must stay reproducible from a stored tuple alone, so the digest check happens where
    attestations are constructed and the RESULT is carried in `attestation_verified`.

    Resolves the path exactly as `artifact_present` does (same containment rules), hashes the raw
    bytes, and compares against `attestation_sha256`. False on any mismatch, absence, or escape —
    never raises.
    """
    if not tup.attestation_sha256 or not tup.attestation_path:
        return False
    rel = PurePosixPath(tup.attestation_path)
    if rel.is_absolute() or ".." in rel.parts:
        return False
    try:
        actual = hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()
    except OSError:
        return False
    return actual == tup.attestation_sha256.lower()


def _q_min(a: str, b: str) -> str:
    """The lower of two Q levels, on the lattice's own ordering.

    Local import for the same reason `to_frames` imports `frames` locally: it keeps this module
    importable by a projector that has no ledger on its path. `lattice` is the authority on the Q
    chain -- restating the order here would be a second copy of a vocabulary, which is the exact
    defect class this file was written to stop.
    """
    from lattice import Q_LEVELS

    return a if Q_LEVELS.index(a) <= Q_LEVELS.index(b) else b


def _apply_statement_binding_cap(
        tup: ClaimTuple, q: str, t: str, reasons: list[str]) -> tuple[str, str, list[str]]:
    """SC56 — cap a verifier-class grade by what its decided proposition is actually bound to.

    A POST-STEP on the ladder above, never a second ladder: `register_ladder` gains no entry, each
    source class still has exactly one rule, and `grade()` remains the single door (§4.7).

    Being a post-step is also what protects the T axis. The obvious implementation -- an early
    return inside the ladder, copying the shape of the `no protocol` branch -- returns
    `Judged, Located|T0` and so silently destroys traceability. Traceability is a DIFFERENT AXIS:
    an unbound verifier receipt can be perfectly located, hashed and present on disk, and all of
    that stays true when nobody recorded which claim the check decided. So T is computed normally
    and passed through untouched; only Q moves.

    Two caps, one line apart:

      * unbound  -> `Judged`. Not lower: the check really ran and really decided something, which
        is exactly an observation -- worth recording, never decision-gating.
      * bound    -> `Verified`. Already ratified doctrine, merely unenforced until now: spec §4.5
        caps a deterministic verifier result at `Q3` because "a verifier confirms, it does not
        measure", and names protocol-admissible measurement as the ONLY route to `Q4`. Without
        this line a full-tuple verifier record reaches `Witnessed` through the measurement ladder.
    """
    if tup.source_class != VERIFIER_CLASS:
        return q, t, reasons

    reasons = list(reasons)
    if not tup.binding_kind:
        decided = tup.decided_proposition.strip()
        # Name the proposition VERBATIM. A reader who is told "the binding is missing" learns
        # nothing actionable; a reader shown the sentence the checker actually decided can see
        # for themselves how far it is from the claim it was cited for.
        what = (f'the check decided: "{decided}"' if decided
                else "the projection recorded no decided proposition at all")
        reasons.append(
            f"verifier result is not bound to the claim it supports — {what}, and nothing binds "
            f'that to the claim "{tup.claim}". A machine-checked verdict certifies the proposition '
            "the checker decided and binds it to nothing else (intake-1307#05), so this is an "
            "OBSERVATION: capped at Judged until a binding is recorded (SC56)")
        return _q_min(q, "Judged"), t, reasons

    reasons.append(
        f"verifier class (binding_kind={tup.binding_kind!r}): a verifier confirms, it does not "
        "measure — the Q ceiling is Verified, and protocol-admissible measurement is the only "
        "route to Witnessed (docs/design/vidya-pilot-spec.md §4.5)")
    return _q_min(q, "Verified"), t, reasons


def grade(tup: ClaimTuple) -> tuple[str, str, list[str]]:
    """THE grading ladder. One implementation, every source.

    `reasons` names every missing element, so a low grade is self-explaining and nobody has to
    reverse-engineer why their measurement failed to reach Witnessed.

    SC56 adds one post-step, `_apply_statement_binding_cap`, which touches verifier-class tuples
    only. Everything below it is the ladder as it has always been.
    """
    return _apply_statement_binding_cap(tup, *_measurement_ladder(tup))


def _measurement_ladder(tup: ClaimTuple) -> tuple[str, str, list[str]]:
    """The ladder body, unchanged."""
    reasons: list[str] = []
    has_protocol = bool(tup.protocol_id.strip())
    has_ref = bool(tup.attestation_path or tup.attestation_locator)

    if not has_protocol:
        reasons.append("no protocol citation — this is an OBSERVATION, never decision-gating "
                       "(MEASUREMENT.md:13)")
        # An observation that names where it came from is Located; one that names nothing is T0.
        # This is the case the two former implementations disagreed on; resolved toward the more
        # informative reading, since a locator genuinely does locate the claim.
        return "Judged", ("Located" if has_ref else "T0"), reasons

    if tup.reps is None:
        reasons.append("no n/reps recorded")
    if not tup.date:
        reasons.append("no date recorded")
    if not has_ref:
        reasons.append("no attestation reference — a result, but not decision-gating")
    elif not tup.attestation_sha256:
        reasons.append("attestation named but not hashed")
    elif not artifact_present(tup):
        reasons.append("attestation hashed but the artifact is not on disk — a hash over a file "
                       "that no longer exists proves nothing")
    if tup.reps is not None and tup.reps_basis.startswith("attempted"):
        reasons.append(f"n is the ATTEMPTED count ({tup.reps_basis}), not the scored one")

    if not (tup.reps is not None and tup.date and has_ref):
        return "Verified", "Located", reasons
    if tup.attestation_sha256 and artifact_present(tup):
        if tup.attestation_verified is True:
            return "Witnessed", "Attested", reasons
        # SC69. Present and hashed is not the top rung: a hash claim nobody re-derived from the
        # artifact is "a 64-character string was typed", and Attested means the artifact was
        # re-read and matched. The verification happens at the adapter/write boundary and rides
        # in the tuple, so grading stays a pure function of the record.
        reasons.append(
            "attestation sha256 never verified against the artifact's bytes — Attested means "
            "the artifact was re-read and its digest matched, and this digest was recorded "
            "without a re-read (SC69)")
    return "Witnessed", "Anchored", reasons


# --- source classes: one carrier, one vocabulary, one ladder PER CLASS ----------------------
#
# The carrier is shared; the grading rule is not, and pretending otherwise would be its own
# category error. Two classes exist:
#
#   * `measurement` — graded by the constitution's claim rule (protocol / n / date / attestation).
#     Reaches `Witnessed`, because a protocol-admissible measurement with durable attestation is
#     exactly what that level means.
#   * `literature`  — graded by verification status (anchored against the primary source, dive
#     verified or overturned). CAPS at `Verified` by construction: an intake entry is a record of
#     what someone else reported, and no amount of careful reading turns it into a measurement.
#
# What must be unified across classes is the lattice vocabulary, the projection discipline, the
# identity scheme and frame emission. What must NOT be unified is the ladder. So each class
# registers exactly one, and a conformance test fails if a second appears — the check that would
# have caught `measurement_record` and `sealed_manifest` drifting apart on 2026-08-10.

Ladder = Callable[..., tuple]
_LADDERS: dict[str, tuple[str, Ladder]] = {}


def register_ladder(source_class: str, module: str) -> Callable[[Ladder], Ladder]:
    """Declare THE grading ladder for a source class. A second registration is an error."""

    def deco(fn: Ladder) -> Ladder:
        existing = _LADDERS.get(source_class)
        if existing and existing[1] is not fn:
            raise ProjectionError(
                f"source class {source_class!r} already has a ladder in {existing[0]!r}; a second "
                "implementation of one rule becomes two dialects of it")
        _LADDERS[source_class] = (module, fn)
        return fn

    return deco


def ladders() -> dict[str, tuple[str, Ladder]]:
    return dict(_LADDERS)


register_ladder("measurement", "scripts/vidya/claim_tuple.py")(grade)


# --- the verifier class: what the check ASSERTED, never just whether it passed (SC57) ---------
#
# A third adapter class, and the only one with a projection precondition of its own. A verifier --
# a prover, a property checker, a schema validator, a contract test -- answers a question, and the
# answer is a boolean. The boolean is not the finding. `intake-1307#05`: a machine-checked
# certificate "does not certify individual mathematical truth"; it certifies THE PROPOSITION THE
# CHECKER DECIDED, which is a different sentence and is the one nobody records. `intake-1307#00` is
# the existence proof that the resulting gap is large -- an automated check labelled 73.6% of proved
# artifacts non-trivial-and-correct where a manual audit put faithfulness far lower (a reweighted
# projection from a 45-example single-annotator audit: admissible that the gap exists, never as a
# rate).
#
# So a verifier-class adapter MUST project `decided_proposition`, and the registry refuses one that
# emits pass/fail alone. This is a PROJECTION rule, not a grading rule: no verifier ladder is
# registered and none may be added here, because `claim_tuple.grade()` still decides (spec §4.7).
#
# What the recorded proposition LICENSES is SC56, landed 2026-09-07 as
# `_apply_statement_binding_cap` -- a post-step on the one ladder, not a second one. An unbound
# verifier tuple caps at `Judged`; a bound one caps at `Verified`, which is spec §4.5's already
# ratified "a verifier confirms, it does not measure" finally enforced in code.

# (`MEASUREMENT_CLASS` / `LITERATURE_CLASS` / `VERIFIER_CLASS` / `SOURCE_CLASSES` are declared at
# the top of the module, because `ClaimTuple.source_class` defaults to one of them.)

# What a check ANSWERS, as opposed to what it ASSERTS. Named explicitly so the refusal is
# mechanical rather than a comment somebody has to remember -- the same reason `frames.py` names
# its grade-bearing keys instead of describing them.
_BARE_VERDICT_TOKENS = frozenset({
    "0", "1", "accept", "accepted", "error", "fail", "failed", "failure", "false", "green",
    "invalid", "no", "not ok", "ok", "pass", "passed", "proved", "red", "reject", "rejected",
    "sat", "success", "true", "unknown", "unsat", "unverified", "valid", "verified", "yes",
})


def check_decided_proposition(value: object, *, where: str) -> str:
    """Refuse a verifier projection that records a verdict instead of a proposition.

    Two failures, one rule. An empty field says the adapter never captured what was decided, and
    that can never be recovered later: a proposition invented on read claims warrant the original
    check never captured. A bare verdict token is the same hole with a value in it -- "pass" is the
    answer, not the statement, and it is exactly what a downstream citer would have to guess at.
    """
    if not isinstance(value, str) or not value.strip():
        raise ProjectionError(
            f"{where}: a verifier-class projection must carry `decided_proposition` -- the "
            "proposition the checker actually decided. A boolean records that something passed; "
            "only the proposition records WHAT was decided, and it cannot be recovered on read "
            "(docs/design/vidya-pilot-spec.md §4.7)")
    normalized = value.strip().lower().rstrip(".")
    if normalized in _BARE_VERDICT_TOKENS:
        raise ProjectionError(
            f"{where}: `decided_proposition` is the bare verdict {value!r} -- that is what the "
            "check ANSWERED, not what it ASSERTED. Record the statement the checker discharged, "
            "in the checker's own terms")
    return value.strip()


# --- the projection registry ----------------------------------------------------------------

Projection = Callable[[Any], ClaimTuple]
_REGISTRY: dict[str, Projection] = {}
# The UNWRAPPED function per name. Duplicate detection must compare what the adapter wrote, not
# what the registry wrapped it in, or re-importing a verifier adapter would refuse itself.
_ORIGINALS: dict[str, Projection] = {}
_SOURCE_CLASS: dict[str, str] = {}


def _verifier_guard(name: str, fn: Projection, native_field: str) -> Projection:
    """Post-condition on a verifier projection: every tuple it emits names what was decided, AND
    is graded as the class the adapter registered under.

    The second half is SC56's. `source_class` is what the statement-binding cap keys on, so a
    verifier adapter whose tuples went out carrying the default `measurement` class would be
    graded by the uncapped ladder -- the cap would be present, tested, and inert, which is worse
    than absent because it reads as enforced.

    The class is STAMPED rather than refused, deliberately. The declared class is a fact about the
    REGISTRATION, not about the record: an adapter that left the field at its default has asserted
    nothing contrary, so refusing it would only push a re-declaration onto every verifier adapter
    and make the guarantee depend on each of them remembering. Stamping makes the registration
    authoritative and closes the route for every present and future verifier adapter at once. (A
    FALSE binding is a different matter and is refused outright in `__post_init__` -- there the
    record does assert something, and it is wrong.)
    """
    import functools
    from dataclasses import replace

    def pin(item):
        if not isinstance(item, ClaimTuple):
            return item
        where = f"projection {name!r} (verifier class, native field {native_field!r})"
        check_decided_proposition(item.decided_proposition, where=where)
        if item.source_class == VERIFIER_CLASS:
            return item
        return replace(item, source_class=VERIFIER_CLASS)

    @functools.wraps(fn)
    def guarded(*args, **kwargs):
        out = fn(*args, **kwargs)
        if isinstance(out, list):
            return [pin(item) for item in out]
        if isinstance(out, tuple):
            return tuple(pin(item) for item in out)
        return pin(out)

    guarded.__vidya_source_class__ = VERIFIER_CLASS
    guarded.__vidya_decided_proposition_field__ = native_field
    return guarded


def register(
    name: str,
    *,
    source_class: str = MEASUREMENT_CLASS,
    decided_proposition_field: str | None = None,
) -> Callable[[Projection], Projection]:
    """Register a source's projection. The registry is what makes the contract checkable: a
    conformance test can enumerate every source and assert each one produces a valid tuple,
    which is impossible when each adapter grades privately.

    `source_class="verifier"` (SC57) is REFUSED unless the adapter also names the native field its
    decided proposition is read from. The declaration is refused at import time, so an adapter that
    emits pass/fail alone never reaches the registry; the guard it installs is the backstop, so a
    field that is declared but left empty is refused at projection time instead of grading a
    boolean as though it were a finding.

    The class does NOT change grading. `claim_tuple.grade()` still decides, and no verifier ladder
    exists -- project, never grade (spec §4.7).
    """
    if source_class not in SOURCE_CLASSES:
        raise ProjectionError(
            f"projection {name!r}: unknown source_class {source_class!r}; declare one of "
            f"{sorted(SOURCE_CLASSES)} deliberately -- a new class of warrant is never an accident")
    if source_class == VERIFIER_CLASS and not (decided_proposition_field or "").strip():
        raise ProjectionError(
            f"projection {name!r} registers as verifier-class without naming the native field its "
            "decided proposition comes from: a verifier adapter that emits pass/fail alone is "
            "refused. Pass decided_proposition_field=<field on the native record> and project it "
            "into ClaimTuple.decided_proposition")
    if source_class != VERIFIER_CLASS and decided_proposition_field:
        raise ProjectionError(
            f"projection {name!r}: decided_proposition_field belongs to the verifier class only; "
            f"{source_class!r} records a measurement, not a decided proposition")

    def deco(fn: Projection) -> Projection:
        existing = _ORIGINALS.get(name)
        if existing is not None and existing is not fn:
            raise ProjectionError(f"projection {name!r} is already registered")
        _ORIGINALS[name] = fn
        _SOURCE_CLASS[name] = source_class
        _REGISTRY[name] = (
            _verifier_guard(name, fn, decided_proposition_field or "")
            if source_class == VERIFIER_CLASS
            else fn
        )
        return _REGISTRY[name]

    return deco


def registered() -> dict[str, Projection]:
    return dict(_REGISTRY)


def source_classes() -> dict[str, str]:
    """Projection name -> declared adapter class. Reported, so the contract is observable."""
    return dict(_SOURCE_CLASS)


def to_frames(tup: ClaimTuple, *, as_of: str, adapter_id: str,
              authority: str = "measurement") -> list[dict]:
    """Emit source + claim + supporting-evidence frames for one graded tuple.

    Shared so that every source lands in the ledger with the same shape. An adapter that emitted
    its own frames could quietly use a different claim-id scheme, and identity divergence is how
    three A/B arms merged into one belief earlier in this program.
    """
    from frames import make_frame  # local import: keeps this module importable without the ledger

    q, t, reasons = grade(tup)
    ident = tup.measurement_id
    source_id, claim_id = f"src_{ident}", f"clm_{ident}"
    # SC57: the decided proposition rides in the CLAIM frame, next to the claim text it is meant to
    # be read against -- that adjacency is the whole point. SC56's binding rides beside it, in the
    # same frame, for the same reason: a reader sees both what the check decided and what licenses
    # reading it as this claim. All three keys are conditional, so a tuple carrying none of them
    # emits the byte-identical assertion -- and therefore the identical frame_id -- it always did.
    claim_assertion = {"claim_id": claim_id, "display_text": tup.claim, "source_id": source_id}
    if tup.decided_proposition:
        claim_assertion["decided_proposition"] = tup.decided_proposition
    if tup.binding_kind:
        claim_assertion["binding_kind"] = tup.binding_kind
    if tup.binding_ref:
        claim_assertion["binding_ref"] = tup.binding_ref
    source_assertion = {"source_id": source_id,
                        "locator": tup.attestation_locator or tup.attestation_path or f"measurement:{ident}",
                        "source_kind": tup.source_kind, "title": tup.metric,
                        "revision_observed": tup.date}
    if tup.applicability:
        source_assertion["applicability"] = dict(tup.applicability)
    evidence_assertion = {"claim_id": claim_id, "evidence_id": f"evd_{ident}",
                          "grade": {"Q": q, "T": t}, "source_id": source_id,
                          "protocol_id": tup.protocol_id, "reps": tup.reps,
                          "category": tup.category, "metric_direction": tup.metric_direction}
    # Loop-belief redesign §4.2: the measured content rides on the EVIDENCE frame, because it is
    # a property of the measurement. All four keys are conditional on `observation`, so a tuple
    # without one emits the byte-identical assertion -- and frame_id -- it always did (the SC57
    # precedent above). `fold` reads claim_id/grade/evidence_id/source_id from this frame and
    # ignores the rest, so none of them can move a grade, a verdict or a gate outcome.
    retrofit = False
    if tup.observation:
        evidence_assertion.update(metric=tup.metric, value=frame_scalar(tup.value),
                                  unit=tup.unit, observation=frame_observation(tup.observation))
        # §6: a projection of a record that predates its producer's hook says so in the
        # envelope, where the intake adapter already says it.
        retrofit = tup.observation.get("retrofit") is True
    return [
        make_frame(
            frame_type="epyc.vidya/frame/source_observed/v1",
            assertion=source_assertion,
            provenance={"method": adapter_id, "about": ident, "retrofit": retrofit},
            actor=adapter_id, authority_scope=authority, created_at=as_of),
        make_frame(
            frame_type="epyc.vidya/frame/claim_proposed/v1",
            assertion=claim_assertion,
            provenance={"method": adapter_id, "derived_from": source_id, "about": ident},
            actor=adapter_id, authority_scope=authority, created_at=as_of),
        make_frame(
            frame_type="epyc.vidya/frame/evidence_supports_claim/v1",
            assertion=evidence_assertion,
            provenance={"evidence": f"evd_{ident}", "about": claim_id, "method": adapter_id,
                        "grade_reasons": reasons, "reps_basis": tup.reps_basis},
            actor=adapter_id, authority_scope=authority, created_at=as_of),
    ]
