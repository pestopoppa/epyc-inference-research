#!/usr/bin/env python3
"""MANDATORY Vidya belief context for the AutoKernel planner: the consumer half.

ONE SUBSTRATE (operator decision 2026-10-21). Agent loops lean on ONE belief reader,
ROOT's `scripts/vidya/loop_beliefs.py`, run here as a CHILD PROCESS (killed at this side's
deadline, never imported). It decides whether the canonical ledger is current (and runs the
incremental ingest when it is stale), verifies the hash chain, and selects + renders the claims
whose declared applicability scope equals the planner's target. Nothing here reads the ledger,
folds, gates or grades. This module keeps only the AutoKernel-specific half:

  * the target (`target_from_context`, from the `current_regime` block `run.py` builds);
  * the child-process calls (`prepare`, `check_seeds`, `raise_alarm`) and the consumer-side
    validation of a `ready` answer (`validate_ready`, a mirror of `loop_beliefs.validate_ready`);
  * the per-call receipt (`seal_receipt` / `write_receipt`, workers/actor-replies/
    belief-receipts.jsonl) and the explicit-reliance check (`reliance`);
  * degraded mode (`DegradedWork`): what may run while the ledger cannot be made current.

MANDATORY, NOT ADVISORY (operator rules 2026-10-21). There is no "off" mode for planning:

  1. `AgentPlanner.propose` calls `prepare` before it builds a prompt or touches any actor
     backend, and only a `ReadyAnswer` (constructible only from a validated `ready` answer)
     lets the backend call proceed (`require_ready`).
  2. A `refused` answer, a child timeout, a crash, garbage on stdout or a `ready` that fails
     validation raises `BeliefContextRefused` (reason + last known-good snapshot). The child
     already raised the operator alarm for a refusal it decided; for a consumer-side failure
     (timeout / crash / garbage / invalid) this side raises it with the `alarm` subcommand
     (short timeout, best effort). The planner model is never called.
  3. The loop then generates NO new hypothesis. It may run ALREADY-PLANNED work (queued resume
     checkpoints) and PRESEEDED, not-yet-tested hypotheses (`--preseeded-hypotheses`), each
     checked with `check-seeds` against the LAST KNOWN-GOOD snapshot first; a seed with no
     `seeded_against`, a flagged premise, or no known-good snapshot is skipped with its reason
     recorded. Every such work item is logged (belief-degraded.jsonl) and, on the next `ready`,
     re-checked against the fresh snapshot: a premise flagged since then gets a REVIEW FLAG
     (belief-recheck.jsonl). A result is never reversed automatically.

Config is argv-only (`run.py --belief-*`); the env var `EPYC_ROOT_REPO` only names a default
ROOT path, it never switches belief context off. ROOT resolves via `resolve_root_repo`.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

# AutoKernel is also invoked as `autokernel` with only scripts/kernel_rnd on
# PYTHONPATH. Locate the shared research helper from this source checkout.
_RESEARCH_ROOT = Path(__file__).resolve().parents[4]
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))
from scripts.benchmark.belief_capture import CaptureUnavailable, root_checkout

ROOT_REPO_ENV = "EPYC_ROOT_REPO"
READER_RELPATH = Path("scripts") / "vidya" / "loop_beliefs.py"
CONSUMER = "autokernel.planner"
#: The only planning mode. `--actor-belief-context` accepts nothing else.
MODES = ("required",)
ANSWER_SCHEMA = "epyc.vidya.loop_beliefs.v1"
LEDGER_STATE_SCHEMA = "epyc.vidya.ledger_state.v1"
SEED_CHECK_SCHEMA = "epyc.vidya.seed_check.v1"
# prepare reads the intake index, verifies/folds the ~30k-frame ledger and lists each source
# store, and may run an incremental ingest: measured 5.5 s with no source stores plus ~2.4 s
# per store, and 24-26 s (one 30 s TIMEOUT) with three stores at host load average ~600
# (2026-10-21). A 30 s deadline refused planning and raised the alarm on a healthy ledger;
# the planner call it gates takes minutes.
DEFAULT_READER_TIMEOUT_S = 120.0
DEFAULT_ALARM_TIMEOUT_S = 10.0
DEFAULT_ESCALATE_AFTER_S = 3600.0
RECEIPT_SCHEMA = "epyc.vidya.planner_evidence_receipt.v2"
RECEIPT_LOG = "belief-receipts.jsonl"
DEGRADED_SCHEMA = "epyc.autokernel.belief_degraded.v1"
DEGRADED_LOG = "belief-degraded.jsonl"
RECHECK_SCHEMA = "epyc.autokernel.belief_recheck.v1"
RECHECK_LOG = "belief-recheck.jsonl"
LOCK_NAME = ".belief-degraded.lock"
#: Env key carrying the presented-evidence summary to the metrics row (the
#: `AK_ACTOR_SEAT_ARM` precedent: a second channel into `_record_metrics`, harmless to the child).
ENV_KEY = "AK_ACTOR_BELIEF_CONTEXT"
#: ROOT renders the section under this header (`loop_beliefs.SECTION_HEADER`).
SECTION_HEADER = "## Belief-kernel evidence (Vidya ledger, read-only)"
RELIANCE_FIELD = "relies_on_claims"
#: The only GPU on this host is the MI210; the CPU target is the EPYC 9655 host.
DEVICE_BY_BACKEND = {"gpu": "gfx90a", "cpu": "epyc-9655"}
SEED_HYPOTHESIS_FIELDS = ("mechanism_id", "statement", "falsifier", "target_surface",
                          "target_symbol")
#: The child-process seam. Tests replace it (`autokernel/conftest.py` installs a canned
#: `ready` runner for every unit test) -- a fake answer is still validated, never bypassed.
RUNNER: Callable[..., Any] = subprocess.run


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_root_repo(explicit: str | Path | None = None, *,
                      env: Mapping[str, str] | None = None) -> Path:
    """Resolve the named ROOT checkout through the shared, fail-closed resolver.

    `--belief-root-repo` and `EPYC_ROOT_REPO` remain accepted for AutoKernel callers;
    `belief_capture.root_checkout` owns the actual checkout validation and refuses when
    neither is named. The older `/workspace` and hub-view guesses could silently select a
    different ledger than the operator intended.
    """
    environ = os.environ if env is None else env
    named = explicit if explicit is not None else environ.get(ROOT_REPO_ENV)
    if named is None:
        named = environ.get("EPYC_ROOT")
    if isinstance(named, str):
        named = named.strip()
    selected_env = {} if named is None else {"EPYC_ROOT": str(named)}
    return root_checkout(selected_env)


def root_repo(root: str | Path | None = None) -> Path:
    return resolve_root_repo(root)


def state_digest(state: Mapping[str, Any]) -> str:
    """Mirror of `loop_beliefs.state_digest`: the short identity of a ledger state."""
    body = {k: state.get(k) for k in ("ledger_path", "frame_count", "head_seq", "head_hash")}
    return "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _seeded_label(value: Any) -> Any:
    """A `seeded_against` value as a row carries it. A full ledger_state object is reduced to
    its digest (the row stays small); a digest string, or an operator / Fable-architect
    provenance object `{"kind", "author", "at"}` (loop_beliefs.check_seeds), is kept verbatim."""
    if isinstance(value, Mapping) and (value.get("schema") == LEDGER_STATE_SCHEMA
                                       or "head_hash" in value):
        return value.get("ledger_state_digest") or state_digest(value)
    if isinstance(value, Mapping):
        return dict(value)
    return value


# ------------------------------------------------------------------------------ target


#: A focus symbol is one kernel symbol name (an experiments row's `target_symbol`).
FOCUS_SYMBOLS_MAX = 64


def focus_symbol(text: Any) -> str:
    """One validated focus symbol (`--belief-focus-symbol`, owned-targets `focus_symbols`)."""
    value = text.strip() if isinstance(text, str) else ""
    if not 0 < len(value) <= 200 or any(ch.isspace() or ord(ch) < 32 or ch == "\x7f"
                                        for ch in value):
        raise ValueError(f"focus symbol must be one symbol name of 1-200 printable "
                         f"non-space characters: {text!r}")
    return value


def target_from_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """The planner's target, from the `current_regime` block `run.py` already builds. An
    incomplete target is not a refusal: ROOT answers `ready` with empty evidence
    (`inapplicable_target`), and the plan still names the verified ledger state."""
    regime = context.get("current_regime") or {}
    model = regime.get("model") or {}
    backend = str(regime.get("backend") or "").lower()
    return {"model_path": str(model.get("path") or "") if isinstance(model, Mapping) else "",
            "quant": regime.get("quant") or "", "backend": backend,
            "device": DEVICE_BY_BACKEND.get(backend, ""),
            "surface": regime.get("measurement_surface") or ""}


# ------------------------------------------------------------------------------ config


@dataclass(frozen=True)
class BeliefConfig:
    """Everything the child-process calls need; built from argv by `run.py`."""

    root: Path | None = None
    ledger: Path | None = None
    known_good: Path | None = None
    escalate_after_s: float = DEFAULT_ESCALATE_AFTER_S
    reader_timeout_s: float = DEFAULT_READER_TIMEOUT_S
    alarm_timeout_s: float = DEFAULT_ALARM_TIMEOUT_S
    #: Test isolation (and an operator override): the alarm channel's state/config files.
    alarm_state: Path | None = None
    alarm_config: Path | None = None
    index: Path | None = None
    #: The loop's own stores: their new serving A/B receipts are ingested by the reader before
    #: each plan (VB-LOOP-SCOPE producer-time ingest), so this loop's latest outcomes are beliefs.
    source_roots: tuple[Path, ...] = ()
    #: Belief redesign §7.5 (Phase 2): the campaign's owned target symbols (owned-targets.json
    #: `focus_symbols` -> run.py `--belief-focus-symbol`). ROOT's `Target.focus_symbols` ranks
    #: claims on these symbols first inside their group; a ranking boost, never a filter.
    focus_symbols: tuple[str, ...] = ()
    consumer: str = CONSUMER
    python: str = sys.executable

    def reader(self) -> Path:
        return root_repo(self.root) / READER_RELPATH

    def _alarm_args(self) -> list[str]:
        out = ["--escalate-after-s", str(float(self.escalate_after_s))]
        if self.alarm_state is not None:
            out += ["--alarm-state", str(self.alarm_state)]
        if self.alarm_config is not None:
            out += ["--alarm-config", str(self.alarm_config)]
        return out

    def _known_good_args(self) -> list[str]:
        return [] if self.known_good is None else ["--known-good", str(self.known_good)]


# ------------------------------------------------------------------------------ answers


class BeliefContextRefused(RuntimeError):
    """No validated `ready` answer: the planner must not be called.

    Deliberately NOT an `ActorTransient`: the loop handles it as "no planning possible"
    (degraded mode), never as a provider failure to retry."""

    def __init__(self, reason: str, detail: str = "", *,
                 last_known_good: Mapping[str, Any] | None = None,
                 answer: Mapping[str, Any] | None = None,
                 alarm: Mapping[str, Any] | None = None,
                 target: Mapping[str, Any] | None = None, elapsed_s: float | None = None):
        super().__init__(f"belief context refused ({reason})"
                         + (f": {detail}" if detail else ""))
        self.reason = str(reason)
        self.detail = str(detail)[:500]
        self.last_known_good = dict(last_known_good) if isinstance(last_known_good,
                                                                    Mapping) else None
        self.answer = dict(answer) if isinstance(answer, Mapping) else None
        self.alarm = dict(alarm) if isinstance(alarm, Mapping) else None
        self.target = dict(target) if isinstance(target, Mapping) else None
        self.elapsed_s = elapsed_s
        #: Set by the planner that raised it: the refusal receipt, and what may still run.
        self.receipt_id = ""
        self.degraded: DegradedWork | None = None

    def summary(self) -> dict[str, Any]:
        """What a degraded-mode experiments row carries."""
        kg = self.last_known_good or {}
        return {"reason": self.reason, "detail": self.detail[:300],
                "last_known_good_digest": kg.get("ledger_state_digest"),
                "last_known_good_as_of": kg.get("as_of"),
                "receipt_id": self.receipt_id}


def validate_ready(answer: Any) -> dict[str, Any]:
    """Mirror of `loop_beliefs.validate_ready` (the contract's consumer-side check). Raises
    ValueError for anything but a `ready` answer naming a verified, current ledger state."""
    if not isinstance(answer, Mapping) or answer.get("schema") != ANSWER_SCHEMA:
        raise ValueError("not a loop_beliefs answer")
    if answer.get("status") != "ready":
        raise ValueError(f"belief context refused: {answer.get('reason')}")
    state = answer.get("ledger_state")
    if (not isinstance(state, Mapping) or state.get("schema") != LEDGER_STATE_SCHEMA
            or not state.get("fresh") or not isinstance(state.get("frame_count"), int)
            or isinstance(state.get("frame_count"), bool)
            or not state.get("head_hash") or not isinstance(answer.get("evidence"), Mapping)):
        raise ValueError("belief context lacks a verified ledger state")
    return dict(answer)


_SEAL = object()


class ReadyAnswer:
    """A `ready` answer that passed `validate_ready`. Only `prepare` makes one; the planner's
    backend call (`AgentPlanner._proposal_from_agent`) refuses anything else."""

    __slots__ = ("answer", "elapsed_s", "target")

    def __init__(self, answer: Mapping[str, Any], *, elapsed_s: float,
                 target: Mapping[str, Any], _seal: object = None):
        if _seal is not _SEAL:
            raise TypeError("ReadyAnswer is made only by belief_context.prepare")
        self.answer = validate_ready(answer)
        self.elapsed_s = elapsed_s
        self.target = dict(target)

    @property
    def evidence(self) -> dict[str, Any]:
        return dict(self.answer.get("evidence") or {})

    @property
    def ledger_state(self) -> dict[str, Any]:
        return dict(self.answer["ledger_state"])

    @property
    def ledger_state_digest(self) -> str:
        return str(self.answer.get("ledger_state_digest") or state_digest(self.ledger_state))

    @property
    def section(self) -> str:
        return str(self.evidence.get("section") or "")


def require_ready(belief: Any) -> ReadyAnswer:
    """The construction guard every planner backend call passes through."""
    if not isinstance(belief, ReadyAnswer):
        raise BeliefContextRefused("unvalidated_answer",
                                   f"planner call without a validated ready answer "
                                   f"({type(belief).__name__})")
    try:
        validate_ready(belief.answer)
    except ValueError as exc:
        raise BeliefContextRefused("invalid_answer", str(exc)) from exc
    return belief


# ------------------------------------------------------------------------------ child calls


def _child(config: BeliefConfig, args: list[str], *, timeout_s: float,
           runner: Callable[..., Any] | None) -> Any:
    argv = [config.python, str(config.reader()), *args]
    return (runner or RUNNER)(argv, capture_output=True, text=True, timeout=timeout_s,
                              check=False)


def _json_object(stdout: Any) -> dict[str, Any]:
    text = stdout.strip() if isinstance(stdout, str) else ""
    body = json.loads(text) if text else None
    if not isinstance(body, dict):
        raise ValueError("child printed no JSON object")
    return body


def _fallback_alarm(config: BeliefConfig, reason: str, detail: str, *,
                    runner: Callable[..., Any] | None = None) -> dict[str, Any]:
    channel = root_repo(config.root) / "scripts" / "coordination" / "alarm_channel.py"
    out: dict[str, Any] = {"invoked": False, "fallback": True, "channel": str(channel)}
    if not channel.is_file():
        out["error"] = f"reader and alarm channel missing under {root_repo(config.root)}"
        print(f"BELIEF LEDGER ALARM UNDELIVERABLE ({reason}): {out['error']}", file=sys.stderr)
        return out
    argv = [config.python, str(channel)]
    if config.alarm_state is not None:
        argv += ["--state", str(config.alarm_state)]
    if config.alarm_config is not None:
        argv += ["--config", str(config.alarm_config)]
    argv += ["raise", "--severity", "critical", "--key",
             f"belief-ledger-not-current-escalated.{config.consumer}", "--message",
             f"{config.consumer}: belief reader missing ({reason}); no new hypotheses are "
             "being generated", "--evidence",
             json.dumps({"reason": reason, "detail": str(detail)[:300]})]
    try:
        done = (runner or RUNNER)(argv, capture_output=True, text=True,
                                  timeout=config.alarm_timeout_s, check=False)
        out.update(invoked=True, returncode=getattr(done, "returncode", None))
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def raise_alarm(config: BeliefConfig, reason: str, detail: str, *,
                runner: Callable[..., Any] | None = None) -> dict[str, Any]:
    """`loop_beliefs.py alarm` for a CONSUMER-side failure (timeout / crash / garbage).
    Best effort under its own short timeout; never raises. When the subcommand cannot run
    because the reader itself is missing, a CRITICAL alarm goes straight through ROOT's alarm
    channel instead."""
    try:
        done = _child(config, ["alarm", "--consumer", config.consumer, "--reason", reason,
                               "--detail", str(detail)[:500], *config._alarm_args()],
                      timeout_s=config.alarm_timeout_s, runner=runner)
        try:
            return {"invoked": True, **_json_object(done.stdout)}
        except ValueError:
            result = {"invoked": True, "returncode": getattr(done, "returncode", None)}
    except Exception as exc:  # noqa: BLE001 -- the refusal stands even if the alarm fails
        result = {"invoked": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    try:
        if not config.reader().is_file():
            result["fallback"] = _fallback_alarm(config, reason, detail, runner=runner)
    except CaptureUnavailable as exc:
        # No named ROOT means no trustworthy alarm locator either. Keep the refusal
        # and report the undeliverable critical alarm without guessing a checkout.
        result["fallback"] = {"invoked": False, "error": str(exc)}
        print(f"BELIEF LEDGER ALARM UNDELIVERABLE ({reason}): {exc}", file=sys.stderr)
    return result


def prepare(context: Mapping[str, Any], config: BeliefConfig | None = None, *,
            runner: Callable[..., Any] | None = None) -> ReadyAnswer:
    """The call before every planning step: a validated `ReadyAnswer`, or
    `BeliefContextRefused`. Never returns without a verified ledger state."""
    config = config or BeliefConfig()
    target = target_from_context(context)
    if config.focus_symbols:
        # Only when given: a loop without focus symbols sends the target it always sent.
        target["focus_symbols"] = sorted({focus_symbol(s) for s in config.focus_symbols})
    args = ["prepare", "--consumer", config.consumer,
            "--target-json", json.dumps(target, sort_keys=True),
            *config._alarm_args(), *config._known_good_args()]
    if config.ledger is not None:
        args += ["--ledger", str(config.ledger)]
    if config.index is not None:
        args += ["--index", str(config.index)]
    for source in config.source_roots:
        args += ["--source-root", str(source)]
    started = time.monotonic()

    def consumer_failure(reason: str, detail: str, answer=None) -> BeliefContextRefused:
        alarm = raise_alarm(config, reason, detail, runner=runner)
        last = (answer or {}).get("last_known_good") if isinstance(answer, Mapping) else None
        return BeliefContextRefused(reason, detail, last_known_good=last, answer=answer,
                                    alarm=alarm, target=target,
                                    elapsed_s=round(time.monotonic() - started, 3))

    try:
        done = _child(config, args, timeout_s=config.reader_timeout_s, runner=runner)
    except subprocess.TimeoutExpired:
        raise consumer_failure("reader_timeout",
                               f"loop_beliefs prepare exceeded {config.reader_timeout_s}s") \
            from None
    except Exception as exc:  # noqa: BLE001 -- e.g. the reader or interpreter is missing
        raise consumer_failure("reader_crash", f"{type(exc).__name__}: {exc}"[:300]) from exc
    try:
        answer = _json_object(done.stdout)
    except ValueError as exc:
        stderr = str(getattr(done, "stderr", "") or "").strip()[-300:]
        raise consumer_failure("reader_crash",
                               f"rc={getattr(done, 'returncode', None)}: {exc}"
                               + (f"; stderr: {stderr}" if stderr else "")) from None
    if answer.get("schema") == ANSWER_SCHEMA and answer.get("status") == "refused":
        # The child decided it, and already raised (or escalated) the operator alarm.
        raise BeliefContextRefused(str(answer.get("reason") or "refused"),
                                   str(answer.get("detail") or ""),
                                   last_known_good=answer.get("last_known_good"),
                                   answer=answer, alarm=answer.get("alarm"), target=target,
                                   elapsed_s=round(time.monotonic() - started, 3))
    try:
        if getattr(done, "returncode", 0) != 0:
            raise ValueError(f"ready answer with exit code {done.returncode}")
        return ReadyAnswer(answer, elapsed_s=round(time.monotonic() - started, 3),
                           target=target, _seal=_SEAL)
    except ValueError as exc:
        raise consumer_failure("invalid_answer", str(exc), answer) from None


def check_seeds(items: Iterable[Mapping[str, Any]], config: BeliefConfig | None = None, *,
                runner: Callable[..., Any] | None = None) -> dict[str, Any]:
    """`loop_beliefs.py check-seeds` against the known-good snapshot. Never raises: a failed
    call answers every item NOT admissible (`check_seeds_failed`) and carries `error`."""
    config = config or BeliefConfig()
    items = [{"id": str(i.get("id", "")), "premises": [str(p) for p in i.get("premises") or ()],
              "seeded_against": _seeded_label(i.get("seeded_against"))} for i in items]
    try:
        done = _child(config, ["check-seeds", "--consumer", config.consumer,
                               "--items-json", json.dumps(items, sort_keys=True),
                               *config._known_good_args()],
                      timeout_s=config.reader_timeout_s, runner=runner)
        body = _json_object(done.stdout)
        if body.get("schema") != SEED_CHECK_SCHEMA or not isinstance(body.get("items"), list):
            raise ValueError(f"not a seed check: {str(body)[:200]}")
        return body
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"[:300]
        return {"schema": SEED_CHECK_SCHEMA, "error": error, "checked_against": None,
                "checked_against_digest": None, "admissible_ids": [], "flagged_ids": [],
                "items": [{**item, "admissible": False, "reasons": ["check_seeds_failed"],
                           "flagged_premises": []} for item in items]}


# ------------------------------------------------------------------------------ receipt


def host_claim_ids(evidence: Mapping[str, Any]) -> list[str]:
    """Ids the reader labelled `tier: host` (device/backend-only background, DESIGN §7.1). An older
    answer has no `tier` on its claims, so it yields none: nothing is refused that was not labelled."""
    if not isinstance(evidence, Mapping):
        return []
    ids = [str(row["claim_id"]) for row in evidence.get("claims") or ()
           if isinstance(row, Mapping) and row.get("tier") == "host" and row.get("claim_id")]
    for key in ("host_claim_ids", "not_evidence_claim_ids"):  # lane E lists (evidence level)
        ids += [str(i) for i in evidence.get(key) or () if isinstance(i, str)]
    return sorted(set(ids))


def reliable_claim_ids(evidence: Mapping[str, Any]) -> list[str]:
    """The ids a hypothesis may declare reliance on: presented, minus host-tier background
    (DESIGN §7.4: target- and sibling-tier ids only)."""
    host = set(host_claim_ids(evidence))
    return [str(i) for i in evidence.get("claim_ids") or () if str(i) not in host]


def reliance(declared: Any, presented: Iterable[str],
             refused_host: Iterable[str] = ()) -> dict[str, Any]:
    """Shown is not relied on: only an explicit reply declaration counts, and only for ids that
    were presented as USABLE. Anything else is recorded as rejected, never raised. A host-tier id
    (`refused_host`) was presented but is background only: rejected as `host_tier`."""
    items = declared if isinstance(declared, list) else ([] if declared is None else [declared])
    presented = set(presented)
    host = set(refused_host)
    accepted, rejected = [], []
    for item in items:
        if isinstance(item, str) and item in presented and item not in host:
            if item not in accepted:
                accepted.append(item)
        else:
            rejected.append({"claim_id": item if isinstance(item, str) else repr(item)[:80],
                             "reason": ("host_tier" if isinstance(item, str) and item in host
                                        else "not_presented")})
    return {"declared": [i if isinstance(i, str) else repr(i)[:80] for i in items],
            "accepted": accepted, "rejected": rejected}


def _count(value: Any) -> int | None:
    """Counts may arrive as ints or decimal strings; None when neither."""
    if isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _tier_counts(value: Any) -> dict[str, int | None] | None:
    return ({str(k): _count(v) for k, v in value.items()} if isinstance(value, Mapping)
            else None)


def _v2_fields(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """The lane E answer keys, all under `evidence` (`banner` is mirrored there; the top-level
    copy is not needed). Tolerant of an older answer that has none of them."""
    index = evidence.get("tried_index")
    declared = _count(evidence.get("tried_index_count"))
    banner = evidence.get("banner")
    return {"tiers": _tier_counts(evidence.get("tiers")),
            "tiers_shown": _tier_counts(evidence.get("tiers_shown")),
            "nonempty_target": (evidence["nonempty_target"]
                                if isinstance(evidence.get("nonempty_target"), bool) else None),
            "tried_index_count": (declared if declared is not None else
                                  len(index) if isinstance(index, (list, tuple)) else None),
            "banner": banner[:300] if isinstance(banner, str) and banner else None,
            "host_claim_ids": host_claim_ids(evidence),
            "refuted_claim_ids": [str(i) for i in evidence.get("refuted_claim_ids") or ()
                                  if isinstance(i, str)]}


def metrics_summary(belief: ReadyAnswer) -> str:
    """The presented half, as the env value `_record_metrics` folds into the metrics row."""
    evidence = belief.evidence
    return json.dumps({"status": evidence.get("status"), "frontier": evidence.get("frontier"),
                       "ledger_state_digest": belief.ledger_state_digest,
                       "claim_ids": list(evidence.get("claim_ids") or []),
                       "contested_claim_ids": list(evidence.get("contested_claim_ids") or []),
                       "withdrawn_claim_ids": list(evidence.get("withdrawn_claim_ids") or []),
                       "runs": [{"source_kind": r.get("source_kind"), "run": r.get("run"),
                                 "status": r.get("status")} for r in evidence.get("runs") or []],
                       "reasons": list(evidence.get("reasons") or []),
                       "v2": _v2_fields(evidence)}, sort_keys=True)


def _module_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return ""


def seal_receipt(belief: ReadyAnswer | None, *, prompt: str | None, workspace: str | Path,
                 seat_arm: str | None, outcome: Mapping[str, Any],
                 relied: Mapping[str, Any] | None = None,
                 refusal: BeliefContextRefused | None = None,
                 config: BeliefConfig | None = None) -> dict[str, Any]:
    """One planner call's receipt: the ledger state it was made against (or the refusal that
    stopped it), what was presented/contested/withdrawn, the outcome and the reliance."""
    config = config or BeliefConfig()
    answer = belief.answer if belief is not None else dict((refusal.answer if refusal else None)
                                                           or {})
    evidence = belief.evidence if belief is not None else {}
    state = belief.ledger_state if belief is not None else None
    root_reader = None
    root_reader_error = None
    try:
        root_reader = str(config.reader())
    except CaptureUnavailable as exc:
        root_reader_error = str(exc)
    body = {
        "schema": RECEIPT_SCHEMA, "ts": _now(), "consumer": config.consumer, "mode": "required",
        "call": {"prompt_sha256": (hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                                   if prompt is not None else None),
                 "workspace": str(workspace), "seat_arm": seat_arm},
        "status": "ready" if belief is not None else "refused",
        "target": (belief.target if belief is not None
                   else (refusal.target if refusal is not None else None)),
        "reader": {"module": "scripts/kernel_rnd/autokernel/loop/belief_context.py",
                   "module_sha256": _module_sha256(), "root_reader": root_reader,
                   "root_reader_error": root_reader_error,
                   "consumer": config.consumer, "policy": answer.get("policy"),
                   "elapsed_s": (belief.elapsed_s if belief is not None
                                 else (refusal.elapsed_s if refusal is not None else None))},
        "ledger_state": state,
        "ledger_state_digest": belief.ledger_state_digest if belief is not None else None,
        "refusal": (None if refusal is None else
                    {"reason": refusal.reason, "detail": refusal.detail[:300],
                     "last_known_good": refusal.last_known_good, "alarm": refusal.alarm}),
        "frontier": evidence.get("frontier"), "as_of": evidence.get("as_of"),
        "run": list(evidence.get("runs") or []),
        "claim_ids": list(evidence.get("claim_ids") or []),
        "contested_claim_ids": list(evidence.get("contested_claim_ids") or []),
        "withdrawn_claim_ids": list(evidence.get("withdrawn_claim_ids") or []),
        "evidence_status": evidence.get("status"),
        "evidence_reasons": list(evidence.get("reasons") or []),
        "omitted": list(evidence.get("omitted") or []),
        "omitted_count": int(evidence.get("omitted_count") or 0),
        "section_sha256": evidence.get("section_sha256") or "",
        "section_bytes": int(evidence.get("section_bytes") or 0),
        # DESIGN §7.4 keys (None on an older answer): what the section held, by tier.
        "evidence_v2": _v2_fields(evidence),
        "outcome": dict(outcome),
        "reliance": dict(relied or reliance(None, ())),
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return {**body, "receipt_id": "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()}


def _append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_receipt(log_dir: str | Path, receipt: Mapping[str, Any]) -> None:
    """Append one receipt line. Evidence, never a reason to fail the planner call."""
    try:
        _append(Path(log_dir) / RECEIPT_LOG, receipt)
    except OSError:
        pass


# ------------------------------------------------------------------------------ degraded mode


def load_preseeded(path: str | Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """`--preseeded-hypotheses` JSONL: one hypothesis per line (the five Hypothesis fields)
    plus `premises` (claim ids) and `seeded_against` (a ledger_state object or its digest),
    optionally `id` (default: mechanism_id). Returns (rows, errors); a bad row is an error
    the caller records, never a crash."""
    if path is None:
        return [], []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [{"line": None, "error": f"{type(exc).__name__}: {exc}"[:300]}]
    rows, errors, seen = [], [], set()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("not an object")
            missing = [k for k in SEED_HYPOTHESIS_FIELDS
                       if not isinstance(raw.get(k), str) or not raw[k].strip()]
            if missing:
                raise ValueError(f"missing hypothesis fields {missing}")
            premises = raw.get("premises", [])
            if not isinstance(premises, list) or not all(isinstance(p, str) for p in premises):
                raise ValueError("premises must be a list of claim ids")
            seed_id = str(raw.get("id") or raw["mechanism_id"])
            if seed_id in seen:
                raise ValueError(f"duplicate seed id {seed_id!r}")
        except ValueError as exc:
            errors.append({"line": number, "error": str(exc)[:300]})
            continue
        seen.add(seed_id)
        rows.append({"id": seed_id, **{k: raw[k] for k in SEED_HYPOTHESIS_FIELDS},
                     "premises": list(premises), "seeded_against": raw.get("seeded_against")})
    return rows, errors


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


@dataclass(frozen=True)
class DegradedWork:
    """What may run while belief context is refused, and the recovery re-check.

    One per planner (lane); every lane's instance shares `log_dir` (workers/actor-replies),
    and every read-modify-append runs under one flock there, so two lanes never take the same
    seed and a work item is re-checked once."""

    log_dir: Path
    config: BeliefConfig
    preseeded: Path | None = None
    runner: Callable[..., Any] | None = None

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(self.log_dir) / LOCK_NAME, "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _log(self, row: Mapping[str, Any]) -> None:
        _append(Path(self.log_dir) / DEGRADED_LOG,
                {"schema": DEGRADED_SCHEMA, "ts": _now(), **row})

    def next_seed(self, refusal: Mapping[str, Any]) -> dict[str, Any] | None:
        """The next preseeded, not-yet-taken hypothesis that `check-seeds` admits against the
        LAST KNOWN-GOOD snapshot, claimed (logged `event: run`) before it is returned. Every
        candidate refused here is logged `event: skipped` with its reasons. None = nothing
        admissible."""
        if self.preseeded is None:
            return None
        rows, errors = load_preseeded(self.preseeded)
        with self._locked():
            for error in errors:
                self._log({"event": "skipped", "kind": "preseeded", "id": None,
                           "reasons": ["invalid_seed_row"], "detail": error,
                           "refusal": dict(refusal)})
            taken = {row.get("id") for row in _read_rows(Path(self.log_dir) / DEGRADED_LOG)
                     if row.get("event") == "run" and row.get("kind") == "preseeded"}
            candidates = [row for row in rows if row["id"] not in taken]
            if not candidates:
                return None
            check = check_seeds(candidates, self.config, runner=self.runner)
            verdicts = {str(item.get("id")): item for item in check.get("items") or []}
            chosen = None
            for row in candidates:
                verdict = verdicts.get(row["id"]) or {"admissible": False,
                                                      "reasons": ["check_seeds_missing_item"]}
                if verdict.get("admissible") is True and chosen is None:
                    chosen = row
                elif verdict.get("admissible") is not True:
                    self._log({"event": "skipped", "kind": "preseeded", "id": row["id"],
                               "mechanism_id": row["mechanism_id"],
                               "reasons": list(verdict.get("reasons") or []),
                               "flagged_premises": list(verdict.get("flagged_premises") or []),
                               "checked_against_digest": check.get("checked_against_digest"),
                               "check_error": check.get("error"), "refusal": dict(refusal)})
            if chosen is None:
                return None
            self._log({"event": "run", "kind": "preseeded", "id": chosen["id"],
                       "mechanism_id": chosen["mechanism_id"], "premises": chosen["premises"],
                       "seeded_against": _seeded_label(chosen["seeded_against"]),
                       "checked_against_digest": check.get("checked_against_digest"),
                       "refusal": dict(refusal)})
        return {**chosen, "seeded_against": _seeded_label(chosen["seeded_against"])}

    def _receipt_state(self, receipt_id: Any) -> str | None:
        """The ledger_state digest a planner receipt (v2) recorded, for a resume point whose
        hypothesis names the receipt it was planned under."""
        if not receipt_id:
            return None
        for row in _read_rows(Path(self.log_dir) / RECEIPT_LOG):
            if row.get("receipt_id") == receipt_id:
                return row.get("ledger_state_digest")
        return None

    def admit_resume(self, *, checkpoint_id: str, hypothesis: Mapping[str, Any],
                     refusal: Mapping[str, Any]) -> dict[str, Any]:
        """An already-planned checkpoint drawn in degraded mode, checked with `check-seeds`
        against the LAST KNOWN-GOOD snapshot BEFORE it runs. Skipped (``admissible: False``)
        only when a premise -- its declared `relies_on_claims` / `belief_premises` -- is
        flagged there: it was planned, so a missing `seeded_against` (pre-v2 receipts) or a
        failed check is recorded but does not block it, and the recovery re-check covers it.
        Logged `run` or `skipped` either way. Never raises."""
        premises = [str(p) for p in list(hypothesis.get("relies_on_claims") or [])
                    + list(hypothesis.get("belief_premises") or [])]
        try:
            seeded = (_seeded_label(hypothesis.get("seeded_against"))
                      or self._receipt_state(hypothesis.get("belief_receipt_id")))
            check = check_seeds([{"id": str(checkpoint_id), "premises": premises,
                                  "seeded_against": seeded}], self.config, runner=self.runner)
            verdict = next(iter(check.get("items") or []), {})
            flagged = list(verdict.get("flagged_premises") or [])
            row = {"kind": "resume", "id": str(checkpoint_id),
                   "mechanism_id": hypothesis.get("mechanism_id"), "premises": premises,
                   "seeded_against": seeded, "reasons": list(verdict.get("reasons") or []),
                   "flagged_premises": flagged,
                   "checked_against_digest": check.get("checked_against_digest"),
                   "check_error": check.get("error"), "refusal": dict(refusal)}
            with self._locked():
                self._log({"event": "skipped" if flagged else "run", **row})
            return {"admissible": not flagged, **row}
        except Exception as exc:  # noqa: BLE001 -- evidence; the planned work still runs
            print(f"belief degraded: resume check not logged ({exc})", file=sys.stderr)
            return {"admissible": True, "id": str(checkpoint_id), "reasons": ["log_failed"],
                    "flagged_premises": []}

    def recheck(self) -> list[dict[str, Any]]:
        """After a `ready`: every degraded-mode work item not yet re-checked is checked
        against the FRESH snapshot `prepare` just wrote; a premise flagged since then gets a
        review flag. Never reverses a result; never raises. Returns the rows appended."""
        try:
            with self._locked():
                done = {row.get("id") for row in _read_rows(Path(self.log_dir) / RECHECK_LOG)}
                pending: dict[str, dict[str, Any]] = {}
                for row in _read_rows(Path(self.log_dir) / DEGRADED_LOG):
                    if row.get("event") == "run" and row.get("id") not in done:
                        pending.setdefault(str(row.get("id")), row)
                if not pending:
                    return []
                check = check_seeds(
                    [{"id": key, "premises": row.get("premises") or [],
                      "seeded_against": row.get("seeded_against")}
                     for key, row in pending.items()], self.config, runner=self.runner)
                if check.get("error") or not check.get("checked_against"):
                    return []          # no fresh snapshot to judge against: retry next ready
                appended = []
                for item in check.get("items") or []:
                    row = pending.get(str(item.get("id")))
                    if row is None:
                        continue
                    flagged = list(item.get("flagged_premises") or [])
                    out = {"schema": RECHECK_SCHEMA, "ts": _now(), "id": row.get("id"),
                           "kind": row.get("kind"), "mechanism_id": row.get("mechanism_id"),
                           "premises": row.get("premises") or [],
                           "seeded_against": row.get("seeded_against"),
                           "rechecked_against_digest": check.get("checked_against_digest"),
                           "flagged_premises": flagged, "review_flag": bool(flagged),
                           "note": ("review only: a premise of this degraded-mode result is "
                                    "now flagged; the result is NOT reversed") if flagged
                           else "premises still unflagged"}
                    _append(Path(self.log_dir) / RECHECK_LOG, out)
                    appended.append(out)
                return appended
        except Exception as exc:  # noqa: BLE001 -- a re-check never blocks planning
            print(f"belief recheck: skipped ({type(exc).__name__}: {exc})", file=sys.stderr)
            return []


def with_root(config: BeliefConfig, root: str | Path | None) -> BeliefConfig:
    return config if root is None or config.root is not None else replace(config, root=Path(root))
