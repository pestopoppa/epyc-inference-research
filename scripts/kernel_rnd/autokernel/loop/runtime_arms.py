"""Declared runtime arms for the existing strict runtime-admission path.

WHAT THIS IS
------------
A campaign can DECLARE a finite set of runtime-configuration arms -- OpenMP
placement (``OMP_PLACES``), load-thread caps (``GGML_REPACK_THREADS`` /
``OMP_NUM_THREADS``), kernel env switches (``GGML_IQK_Q8_0``), thread counts,
CPU lists, NUMA policies -- and the loop A/Bs them itself and adopts a winner
under its own gates, instead of escalating each knob to the operator
(operator directive 2026-09-27).

Nothing here measures, grades or admits anything. Each declared arm becomes an
ordinary runtime ``Hypothesis`` whose ``RuntimeArmPair`` is built by the SAME
constructor a planner proposal uses (``actors._runtime_pair``) against the
current runtime anchor. From there the unchanged path owns everything:

* ``run.gate_for`` -- installed runtime fields only, env keys only from the
  campaign's environment policy, owned CPU allocation, mandatory op
  correctness on the candidate recipe;
* ``RuntimeAdmission.compare`` -- the original A/A + neutral calibration of the
  anchor (``calibration_block_count`` from the prospective declaration), the
  measured control panel, a paired, order-randomized selection window whose
  length the calibrated stopping rule decides, and a confirmation window;
  ``DirectGates`` applies the evaluator's correctness, determinism and output
  coherence gates to every original output (a non-bit-exact arm passes only
  through the evaluator's own tolerance branch, which admits differing outputs
  solely when the anchor is measured ``bitwise_unstable``);
* the keep -> ``RuntimeAdmission.retain`` -> recipe switch in ``run.commit_pooled``.

Keep-grade evidence (``--runtime-arm-evidence keep_grade``, the default) instead
measures a declared arm with the matched serving instrument against the current
recipe's matched floor, exactly as a source candidate's keep A/B, and admits only
declarations restricted to bit-exact arms. Its attempts are recorded in
``runtime-arms-attempts.json`` and an adoption writes a keep-grade selection record
the next batch restores (``restore_keep_grade_selection``).

"Adopt the best one" is champion/challenger: every arm is a single-field delta
applied to the CURRENT recipe. After an adoption the remaining arms become
unsettled again (their anchor surface changed) and are compared against the
new champion recipe, so the surviving recipe is the one no declared arm beat.

SETTLEDNESS
-----------
An arm is settled for a runtime surface once the runtime-selection state holds
a finished attempt of that exact dimension against an anchor with that surface.
The surface is the launch minus build identity (executable/DSO digests, build
paths, ``LD_LIBRARY_PATH``), so a source keep does not re-open an arm the
current runtime recipe already answered, while a runtime adoption does. A
pending attempt (calibration or a window interrupted by the per-invocation
launch budget) is continued, never re-drawn as new. An arm that is served but
never reaches an attempt (refused before measurement) is served at most
``MAX_UNATTEMPTED_SERVES`` times per surface, durably, so a refused arm cannot
consume every one-iteration batch forever.

EPOCH
-----
``runtime_recipe_surface_digest`` is the runtime recipe's contribution to the
measurement epoch (A3.1 Clause 1a: every input that decides what is run). It is
folded into the epoch inputs only while a runtime selection is carried, so a
campaign that never adopts a runtime recipe keeps its exact historical epoch.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Any, Callable, Mapping

SCHEMA = "epyc.autokernel.runtime_arm_declaration.v1"
ADOPTION_SCHEMA = "epyc.autokernel.runtime_recipe_adoption.v1"
LEDGER_SCHEMA = "epyc.autokernel.runtime_arm_ledger.v1"
LEDGER_NAME = "runtime-arms-ledger.json"
ADOPTION_DIR = "runtime-adoptions"
ATTEMPTS_SCHEMA = "epyc.autokernel.runtime_arm_attempts.v1"
ATTEMPTS_NAME = "runtime-arms-attempts.json"
KEEP_GRADE_SELECTION_SCHEMA = "epyc.autokernel.keep_grade_runtime_selection.v1"
KEEP_GRADE_SELECTION_NAMESPACE = "keep-grade-runtime-selection"
STRICT_SELECTION_SCHEMA = "epyc.autokernel.direct_runtime_selection.v1"
#: How a declared arm is judged. `keep_grade` (default): the same matched, order-
#: randomized paired serving A/B a source candidate clears, against the current recipe's
#: matched serving floor, bit-exact arms only. `strict`: the prospective runtime frame
#: (`RuntimeAdmission`: A/A + neutral calibration, control panel, e-process selection and
#: confirmation windows).
EVIDENCE_MODES = ("keep_grade", "strict")
KINDS = ("threads", "cpu_list", "numa_policy", "load_threads", "env")
NUMERICS = ("bit_exact", "not_bit_exact")
#: ``bit_exact_only`` refuses a declaration naming a non-bit-exact arm.
#: ``evaluator_coherence_gate`` admits one, and it then faces the evaluator's
#: unchanged output-coherence gate; nothing here widens a tolerance.
POLICIES = ("bit_exact_only", "evaluator_coherence_gate")
MAX_ARMS = 32
MAX_UNATTEMPTED_SERVES = 2
MAX_DECLARATION_BYTES = 64 * 1024
_ARM_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class ArmDeclarationRefused(ValueError):
    pass


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArmDeclarationRefused(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True)
class RuntimeArm:
    arm_id: str
    kind: str
    candidate: Any
    numerics: str
    rationale: str
    evidence: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.arm_id, str) or not _ARM_ID.match(self.arm_id):
            raise ArmDeclarationRefused(f"arm_id {self.arm_id!r} must match {_ARM_ID.pattern}")
        if self.kind not in KINDS:
            raise ArmDeclarationRefused(f"arm {self.arm_id}: kind {self.kind!r} is not an "
                                        f"installed runtime field {KINDS}")
        if self.numerics not in NUMERICS:
            raise ArmDeclarationRefused(f"arm {self.arm_id}: numerics must be one of {NUMERICS}")
        _text(self.rationale, f"arm {self.arm_id}: rationale")
        if self.evidence is not None:
            _text(self.evidence, f"arm {self.arm_id}: evidence")
        if self.kind == "env":
            if (not isinstance(self.candidate, Mapping) or set(self.candidate) != {"key", "value"}
                    or not isinstance(self.candidate["key"], str) or not self.candidate["key"]
                    or not (self.candidate["value"] is None
                            or isinstance(self.candidate["value"], str))):
                raise ArmDeclarationRefused(
                    f"arm {self.arm_id}: env candidate must be {{key: <str>, value: <str|null>}}")
            object.__setattr__(self, "candidate", {"key": self.candidate["key"],
                                                   "value": self.candidate["value"]})
        elif self.kind in ("threads", "load_threads"):
            if isinstance(self.candidate, bool) or not isinstance(self.candidate, int) \
                    or self.candidate <= 0:
                raise ArmDeclarationRefused(f"arm {self.arm_id}: {self.kind} must be a positive int")
        else:
            _text(self.candidate, f"arm {self.arm_id}: {self.kind} candidate")

    @classmethod
    def from_dict(cls, value: Any) -> "RuntimeArm":
        if not isinstance(value, Mapping):
            raise ArmDeclarationRefused("arm must be an object")
        required = {"arm_id", "kind", "candidate", "numerics", "rationale"}
        if not required <= set(value) or set(value) - required - {"evidence"}:
            raise ArmDeclarationRefused(f"arm fields must be {sorted(required)} (+ evidence)")
        return cls(value["arm_id"], value["kind"], value["candidate"], value["numerics"],
                   value["rationale"], value.get("evidence"))

    def to_dict(self) -> dict[str, Any]:
        row = {"arm_id": self.arm_id, "kind": self.kind, "candidate": self.candidate,
               "numerics": self.numerics, "rationale": self.rationale}
        if self.evidence is not None:
            row["evidence"] = self.evidence
        return row

    def treatment(self) -> dict[str, Any]:
        """The planner-proposal shape `actors._runtime_pair` accepts."""
        candidate = dict(self.candidate) if self.kind == "env" else self.candidate
        return {"kind": self.kind, "candidate": candidate}

    @property
    def mechanism_id(self) -> str:
        return "runtime-arm-" + self.arm_id

    def env_key(self) -> str | None:
        return self.candidate["key"] if self.kind == "env" else None


@dataclass(frozen=True)
class RuntimeArmDeclaration:
    campaign_id: str
    numerics_policy: str
    arms: tuple[RuntimeArm, ...]
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ArmDeclarationRefused("runtime arm declaration schema is unsupported")
        _text(self.campaign_id, "campaign_id")
        if self.numerics_policy not in POLICIES:
            raise ArmDeclarationRefused(f"numerics_policy must be one of {POLICIES}")
        if not self.arms or len(self.arms) > MAX_ARMS:
            raise ArmDeclarationRefused(f"a declaration names 1..{MAX_ARMS} arms")
        ids = [arm.arm_id for arm in self.arms]
        if len(ids) != len(set(ids)):
            raise ArmDeclarationRefused("runtime arm ids repeat")
        treatments = [_digest(arm.treatment()) for arm in self.arms]
        if len(treatments) != len(set(treatments)):
            raise ArmDeclarationRefused("two declared arms are the same treatment")
        if self.numerics_policy == "bit_exact_only":
            inexact = [arm.arm_id for arm in self.arms if arm.numerics != "bit_exact"]
            if inexact:
                raise ArmDeclarationRefused(
                    "numerics_policy bit_exact_only refuses non-bit-exact arm(s) "
                    + ", ".join(inexact) + "; declare evaluator_coherence_gate only where the "
                    "campaign's measurement policy admits the evaluator tolerance path")

    @classmethod
    def from_dict(cls, value: Any) -> "RuntimeArmDeclaration":
        if not isinstance(value, Mapping) or set(value) != {"schema", "campaign_id",
                                                            "numerics_policy", "arms"}:
            raise ArmDeclarationRefused(
                "declaration fields must be exactly schema, campaign_id, numerics_policy, arms")
        if not isinstance(value["arms"], list):
            raise ArmDeclarationRefused("arms must be a list")
        return cls(value["campaign_id"], value["numerics_policy"],
                   tuple(RuntimeArm.from_dict(row) for row in value["arms"]), value["schema"])

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "campaign_id": self.campaign_id,
                "numerics_policy": self.numerics_policy,
                "arms": [arm.to_dict() for arm in self.arms]}

    def digest(self) -> str:
        return _digest(self.to_dict())

    def env_keys(self) -> set[str]:
        return {arm.env_key() for arm in self.arms if arm.env_key() is not None}

    def preflight(self, *, campaign_id: str, runtime_env_keys: set[str],
                  max_candidates: int) -> None:
        """Refuse before any claim: wrong campaign, uninstalled env key, over budget."""
        if self.campaign_id != campaign_id:
            raise ArmDeclarationRefused(
                f"runtime arms belong to {self.campaign_id!r}, not {campaign_id!r}")
        outside = sorted(self.env_keys() - set(runtime_env_keys))
        if outside:
            raise ArmDeclarationRefused(
                "declared env arm key(s) " + ", ".join(outside) + " are not installed runtime "
                "keys for this launch (the campaign environment policy must list them and the "
                "loop's backend allowlist must admit them)")
        if len(self.arms) > max_candidates:
            raise ArmDeclarationRefused(
                f"{len(self.arms)} declared arms exceed the statistics' max_candidates "
                f"{max_candidates}")


def load(path: Path) -> RuntimeArmDeclaration:
    raw = Path(path).read_bytes()
    if len(raw) > MAX_DECLARATION_BYTES:
        raise ArmDeclarationRefused("runtime arm declaration exceeds 64 KiB")
    return RuntimeArmDeclaration.from_dict(json.loads(raw))


# ------------------------------------------------------------------ surfaces

def _recipe_dict(recipe: Any) -> Mapping[str, Any]:
    return recipe if isinstance(recipe, Mapping) else recipe.to_dict()


def runtime_surface(recipe: Any) -> dict[str, Any]:
    """What the launch RUNS, minus which build it runs.

    The normalized execution identity (`CanonicalResolvedRecipe`'s execution
    dict) without the executable/DSO artifacts: build paths, the executable
    path and `LD_LIBRARY_PATH` move with every rebuilt anchor, the runtime
    surface does not.
    """
    row = _recipe_dict(recipe)
    command = list(row["command_argv"])[1:]
    replacements = {"-m": "<model:%s>" % row["model"]["sha256"], "--port": "<listen-port>"}
    if row.get("drafter"):
        replacements["-md"] = "<drafter:%s>" % row["drafter"]["sha256"]
    for index, token in enumerate(command[:-1]):
        if token in replacements:
            command[index + 1] = replacements[token]
    env = dict(row["launch_env"])
    env.pop("LD_LIBRARY_PATH", None)
    policy = row.get("environment_policy") or {}
    return {"backend": row["backend"], "command_argv_tail": command,
            "topology_prefix": list(row["topology_prefix"]), "launch_env": env,
            "absent_environment": sorted(row.get("absent_environment") or ()),
            "relevant_environment": row.get("relevant_environment"),
            "workload": row.get("workload"),
            "environment_policy_version": policy.get("version"),
            "template": row.get("template")}


def surface_digest(recipe: Any) -> str:
    return _digest(runtime_surface(recipe))


#: Restatements of `launch_env` (the policy's per-key present/absent view): still
#: part of the surface digest, left out of the human-readable change summary.
_DERIVED_SURFACE_KEYS = frozenset({"relevant_environment", "absent_environment"})


def _surface_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    diff = {}
    for key in sorted((set(before) | set(after)) - _DERIVED_SURFACE_KEYS):
        if before.get(key) != after.get(key):
            if key == "launch_env":
                left, right = before.get(key) or {}, after.get(key) or {}
                diff[key] = {name: {"before": left.get(name), "after": right.get(name)}
                             for name in sorted(set(left) | set(right))
                             if left.get(name) != right.get(name)}
            else:
                diff[key] = {"before": before.get(key), "after": after.get(key)}
    return diff


# ------------------------------------------------------------------ settledness

def _json(path: Path, limit: int) -> Any:
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ValueError(f"{path.name} exceeds {limit} bytes")
    return json.loads(raw)


def attempts(store_root: Path):
    """Every (pair, finished) row: strict runtime-selection state files and the
    keep-grade attempt ledger."""
    root = Path(store_root)
    if not root.is_dir():
        return
    ledger = root / ATTEMPTS_NAME
    if ledger.exists():
        body = _json(ledger, 4 * 1024 * 1024)
        if body.get("schema") != ATTEMPTS_SCHEMA or not isinstance(body.get("attempts"), list):
            raise ValueError("keep-grade runtime arm attempt ledger schema differs")
        for row in body["attempts"]:
            if isinstance(row.get("pair"), Mapping):
                yield row["pair"], True
    for path in sorted(root.glob("runtime-selection-*.json")):
        state = _json(path, 128 * 1024)
        for row in state.get("attempts", ()):
            pair = row.get("pair")
            if isinstance(pair, Mapping):
                yield pair, row.get("result") is not None


def _matches(pair: Mapping[str, Any], arm: RuntimeArm, surface: str) -> bool:
    dimension = pair.get("dimension") or {}
    if dimension.get("kind") != arm.kind:
        return False
    candidate = dimension.get("candidate")
    if arm.kind == "env":
        if not isinstance(candidate, Mapping) or dict(candidate) != arm.candidate:
            return False
    elif candidate != arm.candidate:
        return False
    try:
        return surface_digest(pair["anchor"]) == surface
    except (KeyError, TypeError):
        return False


def arm_state(store_root: Path, declaration: RuntimeArmDeclaration,
              anchor: Any) -> dict[str, str]:
    """``arm_id -> settled | pending | open`` against this anchor's runtime surface."""
    surface = surface_digest(anchor)
    rows = list(attempts(store_root))
    state = {}
    for arm in declaration.arms:
        finished = [done for pair, done in rows if _matches(pair, arm, surface)]
        state[arm.arm_id] = ("settled" if any(finished) else
                             "pending" if finished else "open")
    return state


def record_attempt(store_root: Path, *, pair: Mapping[str, Any], comparison: Mapping[str, Any],
                   declaration: "RuntimeArmDeclaration") -> None:
    """A completed keep-grade comparison settles its arm for the anchor's surface."""
    from . import status
    path = Path(store_root) / ATTEMPTS_NAME
    body = (_json(path, 4 * 1024 * 1024) if path.exists()
            else {"schema": ATTEMPTS_SCHEMA, "attempts": []})
    if body.get("schema") != ATTEMPTS_SCHEMA:
        raise ValueError("keep-grade runtime arm attempt ledger schema differs")
    body["attempts"].append({
        "recorded_at": _now(), "pair": dict(pair), "declaration_sha256": declaration.digest(),
        "anchor_surface_digest": surface_digest(pair["anchor"]),
        "comparison": {key: comparison.get(key) for key in (
            "effect", "effect_pct", "pairs", "decisive", "noise_floor_pct", "floor_sha256",
            "anchor_samples", "candidate_samples", "admission")}})
    status.write_json(path.parent, path.name, body, prefix=".runtime-arms-attempts-")


def is_declared(pair: Any, declaration: "RuntimeArmDeclaration | None") -> bool:
    """The pair IS one of the declared arms (mechanism id and exact treatment)."""
    if declaration is None or pair is None:
        return False
    dimension = pair.dimension
    for arm in declaration.arms:
        if dimension.dimension_id != arm.mechanism_id or dimension.kind != arm.kind:
            continue
        candidate = dimension.candidate
        if arm.kind == "env":
            candidate = {"key": candidate["key"], "value": candidate["value"]}
        if candidate == arm.candidate:
            return True
    return False


class Ledger:
    """Durable serve counts for arms that never reached an attempt."""

    def __init__(self, store_root: Path):
        self.path = Path(store_root) / LEDGER_NAME

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema": LEDGER_SCHEMA, "served": {}}
        body = _json(self.path, 256 * 1024)
        if body.get("schema") != LEDGER_SCHEMA or not isinstance(body.get("served"), dict):
            raise ValueError("runtime arm ledger schema differs")
        return body

    def count(self, surface: str, arm_id: str) -> int:
        return int(self.read()["served"].get(surface, {}).get(arm_id, 0))

    def increment(self, surface: str, arm_id: str) -> int:
        from . import status
        body = self.read()
        row = body["served"].setdefault(surface, {})
        row[arm_id] = int(row.get(arm_id, 0)) + 1
        status.write_json(self.path.parent, self.path.name, body, prefix=".runtime-arms-")
        return row[arm_id]


class DeclaredArmPlanner:
    """Serve the next unsettled declared arm before any planner proposal.

    While an arm is open or pending the planner is not consulted, so no source
    keep can move the anchor under an in-progress runtime frame (the strict
    calibration is bound to its exact anchor and spans many launches). Once
    every arm is settled for the current surface, the ordinary planner runs.
    """

    def __init__(self, ordinary, declaration: RuntimeArmDeclaration, *,
                 store_root: Callable[[], Path | None],
                 on_event: Callable[[dict], None] | None = None,
                 evidence: str = "strict"):
        if evidence not in EVIDENCE_MODES:
            raise ArmDeclarationRefused(f"runtime arm evidence must be one of {EVIDENCE_MODES}")
        self.ordinary = ordinary
        self.declaration = declaration
        self.evidence = evidence
        self._store_root = store_root
        self._on_event = on_event
        self._lock = threading.Lock()
        self._skipped: dict[tuple[str, str], str] = {}

    def __getattr__(self, name):
        # Only reached for attributes this wrapper does not define.
        return getattr(self.ordinary, name)

    def _event(self, **row) -> None:
        try:
            if self._on_event is not None:
                self._on_event({"observed_at": _now(), **row})
        except Exception:
            pass    # diagnostics are never a gate

    def next_arm(self, context: Mapping[str, Any]):
        """The Hypothesis for the next arm, or None to defer to the planner."""
        anchor = context.get("runtime_anchor")
        if anchor is None:
            return None
        # Planner-proposed runtime treatments stay observation-only under keep-grade
        # evidence; only the DECLARED arms carry keep-grade authority.
        if self.evidence == "strict" and context.get("runtime_observation_only", True):
            return None
        root = self._store_root()
        if root is None:
            return None
        from .actors import ProviderTransient, _runtime_pair
        from .loop import Hypothesis
        surface = surface_digest(anchor)
        with self._lock:
            state = arm_state(root, self.declaration, anchor)
            ledger = Ledger(root)
            for arm in self.declaration.arms:
                key = (surface, arm.arm_id)
                if state[arm.arm_id] == "settled" or key in self._skipped:
                    continue
                if state[arm.arm_id] == "open" \
                        and ledger.count(surface, arm.arm_id) >= MAX_UNATTEMPTED_SERVES:
                    self._skipped[key] = "served without reaching an attempt"
                    self._event(arm_id=arm.arm_id, surface=surface, status="skipped",
                                reason=self._skipped[key])
                    continue
                try:
                    pair = _runtime_pair(arm.treatment(), context, arm.mechanism_id)
                except ProviderTransient as exc:
                    # Typically the arm IS the current value (an exact no-op) or its
                    # key is not installed on this launch: skip, never fail the draw.
                    self._skipped[key] = str(exc)
                    self._event(arm_id=arm.arm_id, surface=surface, status="skipped",
                                reason=str(exc))
                    continue
                if state[arm.arm_id] == "open":
                    ledger.increment(surface, arm.arm_id)
                self._event(arm_id=arm.arm_id, surface=surface, status="served",
                            continuation=state[arm.arm_id] == "pending")
                return Hypothesis(
                    arm.mechanism_id,
                    f"Declared runtime arm {arm.arm_id}: {arm.rationale}",
                    "The original calibrated paired comparison does not admit an improvement "
                    "with the required correctness and control gates",
                    "runtime", arm.kind, runtime_pair=pair)
        return None

    def propose(self, context):
        hypothesis = self.next_arm(context)
        return hypothesis if hypothesis is not None else self.ordinary.propose(context)

    def author(self, hypothesis, context):
        return self.ordinary.author(hypothesis, context)


# ------------------------------------------------------------------ selection epoch

def keep_grade_selection(*, adopted: Any, previous: Any, runtime_pair: Mapping[str, Any],
                         comparison: Mapping[str, Any], declaration: RuntimeArmDeclaration,
                         current_source_commit: str) -> dict[str, Any]:
    """The retained keep-grade selection the next batch restores."""
    return {"schema": KEEP_GRADE_SELECTION_SCHEMA, "evidence": "keep_grade",
            "current_recipe": dict(_recipe_dict(adopted)),
            "previous_recipe_surface_digest": surface_digest(previous),
            "adopted_surface_digest": surface_digest(adopted),
            "runtime_pair": dict(runtime_pair), "declaration_sha256": declaration.digest(),
            "comparison": {key: comparison.get(key) for key in (
                "effect", "effect_pct", "pairs", "decisive", "noise_floor_pct", "floor_sha256",
                "request_digest", "anchor_samples", "candidate_samples", "admission")},
            "current_source_commit": current_source_commit}


def selection_current_recipe(store_root: Path, reference: Mapping[str, Any], *,
                             evidence: str | None = None) -> dict[str, Any]:
    """Read a retained runtime selection's current recipe BEFORE any claim.

    The same bytes `runtime_admission.restore_selection` later reopens and
    verifies under the claim; this read only derives the epoch input.
    """
    from .measurement_capture import ArtifactStore
    from .observation_binding import _plain
    if not isinstance(reference, Mapping) or not {"locator", "sha256"} <= set(reference):
        raise ArmDeclarationRefused("runtime recipe reference shape is invalid")
    root = Path(store_root) / "runtime-preparation"
    if not root.is_dir():
        raise ArmDeclarationRefused("runtime recipe reference names no runtime store")
    store = ArtifactStore(root)
    try:
        body = _plain(store.read(reference["locator"], reference["sha256"]))
    finally:
        store.close()
    schemas = {"keep_grade": {KEEP_GRADE_SELECTION_SCHEMA}, "strict": {STRICT_SELECTION_SCHEMA},
               None: {KEEP_GRADE_SELECTION_SCHEMA, STRICT_SELECTION_SCHEMA}}[evidence]
    if body.get("schema") not in schemas or not isinstance(body.get("current_recipe"), Mapping):
        raise ArmDeclarationRefused(
            "runtime recipe reference is not a runtime selection"
            + (f" of {evidence} evidence" if evidence else ""))
    return dict(body["current_recipe"])


def restore_keep_grade_selection(store, reference: Mapping[str, Any], *, build: Path,
                                 rebind: Callable[[Any, Path], Any]):
    """Reopen a retained keep-grade selection and rebind it to the current anchor build.

    The selection names the adopted recipe (captured on the build it was measured on);
    a later source keep moved the build, never the runtime surface. The rebound recipe
    must keep that exact surface and template, or the restore refuses."""
    from .observation_binding import _plain
    from .resolved_recipe import CanonicalResolvedRecipe
    if not isinstance(reference, Mapping) or not {"locator", "sha256"} <= set(reference):
        raise ArmDeclarationRefused("keep-grade runtime selection reference shape is invalid")
    body = _plain(store.read(reference["locator"], reference["sha256"]))
    if body.get("schema") != KEEP_GRADE_SELECTION_SCHEMA or body.get("evidence") != "keep_grade":
        raise ArmDeclarationRefused("runtime recipe reference is not a keep-grade selection")
    try:
        verified = store.verify(KEEP_GRADE_SELECTION_NAMESPACE, body)
    except Exception as exc:
        raise ArmDeclarationRefused(f"keep-grade runtime selection namespace differs: {exc}") from exc
    if (verified.locator, verified.sha256) != (reference["locator"], reference["sha256"]):
        raise ArmDeclarationRefused("keep-grade runtime selection namespace differs")
    adopted = CanonicalResolvedRecipe.from_dict(body["current_recipe"])
    if surface_digest(adopted) != body["adopted_surface_digest"]:
        raise ArmDeclarationRefused("keep-grade selection surface differs from its own record")
    current = (adopted if Path(adopted.build_dir).resolve() == Path(build).resolve()
               else rebind(adopted, Path(build)))
    if surface_digest(current) != body["adopted_surface_digest"] \
            or current.template.to_dict() != adopted.template.to_dict():
        raise ArmDeclarationRefused("keep-grade selection does not rebind to the current build "
                                    "with its runtime surface unchanged")
    return current


# ------------------------------------------------------------------ adoption receipt

def adoption_receipt(*, campaign_id: str, previous: Any, adopted: Any, admission: Any,
                     selection_reference: Any, runtime_pair: Mapping[str, Any] | None,
                     comparison: Mapping[str, Any], epoch: str, measurement_epoch: str,
                     anchor_commit: str, statistics_sha256: str | None,
                     declaration: RuntimeArmDeclaration | None,
                     invalidated_floor: Any, accumulator: Mapping[str, Any] | None,
                     adopted_at: str | None = None, evidence: str = "strict") -> dict[str, Any]:
    before, after = runtime_surface(previous), runtime_surface(adopted)
    mechanism = ((runtime_pair or {}).get("dimension") or {}).get("dimension_id")
    arm = None
    if declaration is not None and isinstance(mechanism, str) \
            and mechanism.startswith("runtime-arm-"):
        arm = next((row.to_dict() for row in declaration.arms
                    if row.mechanism_id == mechanism), None)
    return {
        "schema": ADOPTION_SCHEMA, "campaign_id": campaign_id,
        "adopted_at": adopted_at or _now(),
        "evidence": evidence,
        "authority": (("loop runtime-recipe adoption at keep-grade evidence (declared bit-exact "
                       "arm; matched, order-randomized paired serving A/B that cleared the current "
                       "recipe's matched serving floor; op correctness on the candidate recipe)"
                       if evidence == "keep_grade" else
                       "loop runtime-recipe adoption under the strict runtime admission "
                       "(calibrated paired selection + confirmation, measured control panel, "
                       "evaluator correctness/coherence gates)")
                      + "; experimental execution recipe only, never a production, registry or "
                        "serving-lineup change"),
        "previous_recipe": {"execution_digest": _recipe_dict(previous).get("execution_digest"),
                            "runtime_surface_digest": _digest(before)},
        "adopted_recipe": {"execution_digest": _recipe_dict(adopted).get("execution_digest"),
                           "runtime_surface_digest": _digest(after)},
        "surface_change": _surface_diff(before, after),
        "dimension": (runtime_pair or {}).get("dimension"),
        "declared_arm": arm,
        "declaration_sha256": None if declaration is None else declaration.digest(),
        "admission": admission, "selection_reference": selection_reference,
        "statistics_sha256": statistics_sha256,
        "comparison": {key: comparison.get(key) for key in (
            "effect", "pairs", "decisive", "noise_floor_pct", "floor_sha256", "admission",
            "anchor_samples", "candidate_samples", "runtime_status", "qualified")},
        "measured_under": {"epoch": epoch, "measurement_epoch": measurement_epoch,
                           "anchor_commit": anchor_commit},
        "epoch_transition": {
            "rule": ("the adopted runtime surface is a measured input (P-AK-SEARCH-1-A3.1 "
                     "Clause 1a): the next launch carries this selection and folds "
                     "runtime_recipe_surface_digest into its epoch inputs, opening a new "
                     "measurement epoch; rows before it stay in the epoch named in "
                     "measured_under and are cross-epoch (attempt-only) from the new one"),
            "next_epoch_input": {"runtime_recipe_surface_digest": _digest(after)}},
        "floors": {"source_comparison_floor_invalidated": (
                       None if invalidated_floor is None else str(invalidated_floor)),
                   "rule": ("floors are keyed on execution identity; the next source "
                            "comparison recalibrates the request-bound floor under the "
                            "adopted recipe before either arm launches")},
        "accumulator": accumulator,
    }


def write_adoption_receipt(store_root: Path, body: Mapping[str, Any]) -> Path:
    from . import status
    digest = _digest(body)
    name = f"runtime-recipe-adoption-{digest[:16]}.json"
    status.write_json(Path(store_root) / ADOPTION_DIR, name, dict(body),
                      prefix=".runtime-adoption-")
    return Path(store_root) / ADOPTION_DIR / name


__all__ = ["ADOPTION_SCHEMA", "ArmDeclarationRefused", "DeclaredArmPlanner", "EVIDENCE_MODES",
           "KEEP_GRADE_SELECTION_SCHEMA", "Ledger", "is_declared", "keep_grade_selection",
           "record_attempt", "restore_keep_grade_selection",
           "RuntimeArm", "RuntimeArmDeclaration", "SCHEMA", "adoption_receipt", "arm_state",
           "load", "runtime_surface", "selection_current_recipe", "surface_digest",
           "write_adoption_receipt"]
