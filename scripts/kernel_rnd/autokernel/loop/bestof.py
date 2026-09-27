#!/usr/bin/env python3
"""Best-of-N concurrent authoring: N author calls for ONE accepted hypothesis.

Operator decision 2026-09-26: best-of-2, a MIXED pair -- author A thinking OFF, author
B thinking MEDIUM (`--actor-authors "off,medium"`; the list length is N). DS41 runs
10c/10g put the two failure modes side by side: thinking uncapped deliberated 74k
tokens and never edited, thinking off edited fast and wrote broken AVX-512. Racing the
two on the same hypothesis keeps whichever lands a checkable diff first.

SHAPE (one authoring round of `loop.iterate`)

    lane @ base (+ the lane's current diff, e.g. round 1's rejected patch)
      -> N scratch worktrees at the same base, each seeded with that diff
         (allocated through the run's one scratch registry, `<store>/scratch`)
      -> N author calls CONCURRENTLY on :8083, each in its own scratch tree
      -> the FIRST author whose diff passes the validator wins:
           its diff is applied to the real lane; every other call is ended through
           the actor stop path (DS41-C22: the process group is TERM'd) and its tree
           released at once
      -> no winner: the best diff by the existing rules goes to the lane (the normal
         gates then refuse it and hand the reason to the next round), or every
         failure is recorded and the round ends as the single path would

`iterate` sees ONE author: this panel is called where `planner.author` would be, returns
the lane paths (or `Abstain`, or raises the provider exception), and the lane holds the
selected diff. So critic pass 2, `validate_candidate`, the gates, the resume/critic2
checkpoint (whose retention reads the LANE: the winning diff) and every refusal row
are unchanged. N=1 never builds a panel: the single-author path is byte-identical.

POOL BUDGET. :8083 serves np4 on ONE unified KV pool (196,608 tokens, `--kv-unified`);
a full pool under MTP crashes llama-server. N concurrent author calls must fit in the
pool minus a 16,384-token reserve. The split is ASYMMETRIC by thinking mode
(`panel_budget`, operator 2026-09-26 after DS41 run 10h truncated the medium author's
edit step at a 16,384 output cap): off 65,536 context / 16,384 output, medium 114,688 /
40,960 (weights 4:7 of 180,224; `--actor-authors-budget` overrides per mode). Refused
when any member's output cannot leave the compaction headroom (opencode compacts at
context - output). An output above opencode's 32,000 wire cap reaches the wire through
`OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` (`actor_opencode_config.output_ceiling_env`).
`author_budget` is the symmetric floor((pool - reserve) / N) split. The planner and
critic limits are unchanged: they never run concurrently with the authors.

NO WINNER. A round in which no author produced a diff hands `iterate` one failure
record per member (`loop.author_failure_record`: outcome, reason, "authoring" or
"harness", the call's capped-step and ak-check evidence, the retained patch): as a
`loop.AuthoringFailure` when any member failed on its own account (it abstained, or
reported a change it never made on an uncapped final step), else by raising the
provider exception with the records attached (`author_failures`). A member whose final
step hit the output cap with no diff is recorded `failure_class: output_capped_empty`.

WALL BUDGET (operator 2026-09-27, after DS41 run 10j round 1 on `akm-q4k-x4t-avx512`:
a0-off gave up after 61 min, a1-medium hit the 7200 s actor timeout, was retried
in-round as an uncharged harness failure and spent another 6667 s -- 3.9 h for one
round, whose winning patch critic2 then proved wrong). `WallBudget`:

  * each member has a wall budget by thinking mode (off 2700 s, medium 4500 s, default
    4500 s; `--actor-authors-wall`) and the whole panel a hard wall (5400 s;
    `--actor-authors-panel-wall`), enforced as the member's stop predicate: the
    actor's own C22 path TERMs the call's process group. Nothing is killed by name.
    At its budget a member stops and its lane diff is salvaged (`integrity.
    lane_diff_report` against the base, only an edit made by this call; the validator
    then decides whether it is usable, exactly as for a report-missing reply);
  * a member is never retried in-round after a timeout or a budget stop (its seat runs
    `retry_timeouts=False`, and a budget stop ends `_with_backoff` as a stop): if
    another member produced a diff the loser simply loses; if none did the round is an
    uncharged `authoring_harness_failure` (`failure_class: member_budget_exhausted`;
    `loop.AUTHOR_HARNESS_FAILURE_CAP` still charges the third in a row);
  * early cancel (`--actor-authors-cancel-factor`, default 2.0, 0 = off): once one
    member FINISHES (a winner, a genuine abstention, or a diff that failed the
    validator), any member still authoring past factor x the finisher's wall without
    an ak-check-passing patch (its own sandbox log's last result) is ended, recorded
    `result: cancelled`, `reason: early_cancel`. A member already in the winner check
    is left to it (the panel wall still bounds it).

The panel row carries `walls` (the budgets, factor, panel wall and the round's wall),
and each member its `wall_budget_s`, `budget_stopped`, `stop_cause` and
`early_cancelled`.

DISK. Every scratch tree is allocated from, and released by, the flow-level scratch
registry (`scratch.py`): marker-owned, released on every exit path of the scope, the
loser early. This module never creates, sweeps or removes a worktree itself, and never
prunes. `ensure_free` refusing the space is a fallback to the single path, not an error.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

from . import actor_metrics, archive, integrity
from .loop import (MEMBER_BUDGET_EXHAUSTED, Abstain, ActorStopped, ActorTransient,
                   AuthoringFailure, AuthorReportMissing, author_failure_record)

#: :8083's unified KV pool (np4, --kv-unified, MTP draft) and the tokens that always stay
#: free for the other slots (operator 2026-09-26).
DEFAULT_POOL_TOKENS = 196_608
POOL_RESERVE = 16_384
#: Each concurrent author's `limit.output` (operator 2026-09-26).
AUTHOR_OUTPUT_LIMIT = 16_384
#: opencode compacts at context - output; below this much context left under the output
#: cap a concurrent author has no working room (the operator's 2026-09-26 headroom rule).
MIN_COMPACTION_HEADROOM = 32_768
#: :8083 is np4: more concurrent authors than slots only queue on the server.
MAX_AUTHORS = 4
#: Scratch bytes one author may cost: a lane checkout (~166 MB) plus an ak-check scratch
#: build (~850 MB), measured 2026-09-26. `ensure_free(N * this)` gates the panel.
SCRATCH_BYTES_PER_AUTHOR = 1_100_000_000

#: The thinking modes a panel member may name (the seat's `author_thinking` values).
#: `medium` lands with lane/ak-author-medium-20260926; run.py checks the installed set.
AUTHOR_MODES = ("default", "off", "medium")

PANEL_SCHEMA = "epyc.autokernel.author_panel.v1"
PANEL_LOG = "author-panel.jsonl"

SELECTION_FIRST_PASSING = "first_passing"
SELECTION_BEST_FAILING = "best_failing"
SELECTION_NONE = "none"
SELECTION_FALLBACK = "fallback_single"


class PoolBudgetRefused(ValueError):
    """N concurrent authors cannot fit the pool with a working compaction headroom."""


class PanelSetupRefused(RuntimeError):
    """The panel cannot run this round; the round falls back to the single author."""


class ValidationStopped(RuntimeError):
    """A validator was ended because the panel already has a winner (or a stop)."""


# ----------------------------------------------------------------- configuration


@dataclass(frozen=True)
class AuthorSpec:
    """One panel member: its label (`a0-off`) and the author thinking mode it runs."""
    label: str
    thinking: str


def parse_authors(spec: str, *, allowed: Sequence[str] = AUTHOR_MODES) -> tuple[AuthorSpec, ...]:
    """`"off,medium"` -> (a0-off, a1-medium). The list length is N; order is identity."""
    modes = [part.strip() for part in str(spec or "").split(",")]
    if not modes or any(not mode for mode in modes):
        raise ValueError(f"--actor-authors needs a comma list of thinking modes, got {spec!r}")
    unknown = sorted({mode for mode in modes if mode not in allowed})
    if unknown:
        raise ValueError(f"--actor-authors: unknown thinking mode(s) {unknown}; "
                         f"this build supports {list(allowed)}")
    if len(modes) > MAX_AUTHORS:
        raise ValueError(f"--actor-authors: {len(modes)} authors exceed :8083's "
                         f"{MAX_AUTHORS} slots (np4); extra calls only queue")
    return tuple(AuthorSpec(f"a{index}-{mode}", mode) for index, mode in enumerate(modes))


#: Per-thinking-mode (context WEIGHT, `limit.output`) of a concurrent author (operator
#: 2026-09-26, after DS41 run 10h). A thinking-medium author reasons inside each step
#: before it edits, so a 16,384-token output cap truncates its edit step: run 10h's
#: a1-medium hit the cap on 3 of 14 steps, the last one its edit, and the round was
#: lost as `report_missing` after 71 minutes. Off 4 : medium 7 of the 180,224 tokens
#: the pool leaves above its reserve is 65,536 / 114,688 of context, with outputs
#: 16,384 / 40,960 (each keeps `MIN_COMPACTION_HEADROOM` below it). "default" (thinking
#: uncapped) is budgeted like medium. `--actor-authors-budget` overrides per mode.
DEFAULT_MODE_BUDGETS: dict[str, tuple[int, int]] = {
    "off": (4, 16_384), "medium": (7, 40_960), "default": (7, 40_960)}
#: Member contexts are floored to this granule (so the split's sum stays in the pool).
CONTEXT_GRANULE = 1_024


@dataclass(frozen=True)
class MemberBudget:
    """One panel member's opencode limits: its share of the pool and its output cap."""
    label: str
    thinking: str
    weight: int
    context_limit: int
    output_limit: int

    @property
    def compaction_at(self) -> int:
        return self.context_limit - self.output_limit

    def to_dict(self) -> dict:
        return {"label": self.label, "thinking": self.thinking, "weight": self.weight,
                "context_limit": self.context_limit, "output_limit": self.output_limit,
                "compaction_at": self.compaction_at}


@dataclass(frozen=True)
class PoolBudget:
    """Per-author opencode limits for N concurrent authors on one unified pool.

    `members` (from `panel_budget`) may differ per thinking mode; `context_limit` /
    `output_limit` are then None. A symmetric budget (`author_budget`) carries them and
    no members: every author gets the same."""
    n: int
    pool_tokens: int
    reserve: int
    context_limit: int | None
    output_limit: int | None
    members: tuple = ()

    @property
    def compaction_at(self) -> int | None:
        if self.context_limit is None or self.output_limit is None:
            return None
        return self.context_limit - self.output_limit

    @property
    def concurrent_peak(self) -> int:
        if self.members:
            return sum(member.context_limit for member in self.members) + self.reserve
        return self.n * int(self.context_limit or 0) + self.reserve

    def for_member(self, spec: "AuthorSpec") -> MemberBudget:
        """The limits `spec` runs with (its own row, else the symmetric values)."""
        for member in self.members:
            if member.label == spec.label:
                return member
        if self.context_limit is None or self.output_limit is None:
            raise KeyError(f"no budget for panel member {spec.label}")
        return MemberBudget(spec.label, spec.thinking, 1, self.context_limit,
                            self.output_limit)

    def to_dict(self) -> dict:
        body = {"n": self.n, "pool_tokens": self.pool_tokens, "reserve": self.reserve,
                "context_limit": self.context_limit, "output_limit": self.output_limit,
                "compaction_at": self.compaction_at,
                "concurrent_peak": self.concurrent_peak}
        if self.members:
            body["members"] = [member.to_dict() for member in self.members]
        return body


def parse_mode_budgets(text: str | None) -> dict[str, tuple[int, int]]:
    """`--actor-authors-budget "off=4:16384,medium=7:40960"` -> {mode: (weight, output)},
    merged over `DEFAULT_MODE_BUDGETS` (a mode not named keeps its default). Empty or
    None is the defaults."""
    modes = dict(DEFAULT_MODE_BUDGETS)
    for part in [item.strip() for item in str(text or "").split(",") if item.strip()]:
        mode, sep, value = part.partition("=")
        weight, colon, output = value.partition(":")
        try:
            parsed = (int(weight), int(output))
        except ValueError:
            parsed = None
        if not sep or not colon or not mode.strip() or parsed is None:
            raise ValueError(f"--actor-authors-budget entry {part!r}: need "
                             "<mode>=<context weight>:<output tokens>")
        if parsed[0] < 1 or parsed[1] < 1:
            raise ValueError(f"--actor-authors-budget entry {part!r}: weight and output "
                             "must be positive")
        modes[mode.strip()] = parsed
    return modes


def author_budget(n: int, *, pool_tokens: int = DEFAULT_POOL_TOKENS,
                  reserve: int = POOL_RESERVE,
                  output_limit: int = AUTHOR_OUTPUT_LIMIT) -> PoolBudget:
    """floor((pool - reserve) / N) of context per author, or `PoolBudgetRefused`.

    N=2 on 196,608: context 90,112, output 16,384, compaction at 73,728. Refused when
    N is outside 1..MAX_AUTHORS, the pool does not exceed the reserve, or the output
    leaves less than MIN_COMPACTION_HEADROOM of context below it (opencode compacts
    at context - output, so a smaller gap compacts on every step)."""
    n, pool_tokens, reserve, output_limit = int(n), int(pool_tokens), int(reserve), int(output_limit)
    if not 1 <= n <= MAX_AUTHORS:
        raise PoolBudgetRefused(f"{n} concurrent authors: outside 1..{MAX_AUTHORS} "
                                f"(:8083 is np{MAX_AUTHORS})")
    if reserve < 0 or pool_tokens <= reserve:
        raise PoolBudgetRefused(f"pool {pool_tokens} tokens does not exceed the "
                                f"{reserve}-token reserve")
    if output_limit <= 0:
        raise PoolBudgetRefused("a concurrent author needs a positive output limit")
    context = (pool_tokens - reserve) // n
    if output_limit >= context - MIN_COMPACTION_HEADROOM:
        raise PoolBudgetRefused(
            f"{n} concurrent authors on a {pool_tokens}-token pool (reserve {reserve}) get "
            f"{context} tokens of context each; output {output_limit} must stay below "
            f"{context - MIN_COMPACTION_HEADROOM} (context - {MIN_COMPACTION_HEADROOM}) "
            "or opencode compacts with no working room")
    budget = PoolBudget(n, pool_tokens, reserve, context, output_limit)
    assert budget.n * budget.context_limit + budget.reserve <= budget.pool_tokens
    return budget


def panel_budget(specs: Sequence["AuthorSpec"], *, pool_tokens: int = DEFAULT_POOL_TOKENS,
                 reserve: int = POOL_RESERVE,
                 modes: Mapping[str, tuple[int, int]] | None = None) -> PoolBudget:
    """Per-member limits for a MIXED panel, or `PoolBudgetRefused`.

    The pool above the reserve is split by each member's mode weight (floored to
    `CONTEXT_GRANULE`), and each member takes its mode's output cap. Defaults
    (`DEFAULT_MODE_BUDGETS`) on 196,608: off 65,536 / 16,384, medium 114,688 / 40,960
    (sum 180,224 = pool - 16,384). Refused like `author_budget`: N outside
    1..MAX_AUTHORS, a pool not above its reserve, an unbudgeted mode, a non-positive
    output, or any member whose output leaves less than MIN_COMPACTION_HEADROOM of its
    context below it. An output above opencode's 32,000 wire cap is sent through
    `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` by the seat (`output_ceiling_env`)."""
    specs = tuple(specs)
    modes = dict(DEFAULT_MODE_BUDGETS if modes is None else modes)
    n, pool_tokens, reserve = len(specs), int(pool_tokens), int(reserve)
    if not 1 <= n <= MAX_AUTHORS:
        raise PoolBudgetRefused(f"{n} concurrent authors: outside 1..{MAX_AUTHORS} "
                                f"(:8083 is np{MAX_AUTHORS})")
    if reserve < 0 or pool_tokens <= reserve:
        raise PoolBudgetRefused(f"pool {pool_tokens} tokens does not exceed the "
                                f"{reserve}-token reserve")
    unknown = sorted({spec.thinking for spec in specs if spec.thinking not in modes})
    if unknown:
        raise PoolBudgetRefused(f"no author budget for thinking mode(s) {unknown} "
                                f"(budgeted: {sorted(modes)})")
    weights = [int(modes[spec.thinking][0]) for spec in specs]
    if any(weight < 1 for weight in weights):
        raise PoolBudgetRefused("a concurrent author needs a positive context weight")
    available, total = pool_tokens - reserve, sum(weights)
    members = []
    for spec, weight in zip(specs, weights):
        context = (available * weight // total) // CONTEXT_GRANULE * CONTEXT_GRANULE
        output = int(modes[spec.thinking][1])
        if output <= 0:
            raise PoolBudgetRefused("a concurrent author needs a positive output limit")
        if output >= context - MIN_COMPACTION_HEADROOM:
            raise PoolBudgetRefused(
                f"{spec.label} gets {context} tokens of context (weight {weight}/{total} of "
                f"{available}); output {output} must stay below "
                f"{context - MIN_COMPACTION_HEADROOM} (context - {MIN_COMPACTION_HEADROOM}) "
                "or opencode compacts with no working room")
        members.append(MemberBudget(spec.label, spec.thinking, weight, context, output))
    same = len({(m.context_limit, m.output_limit) for m in members}) == 1
    budget = PoolBudget(n, pool_tokens, reserve,
                        members[0].context_limit if same else None,
                        members[0].output_limit if same else None, tuple(members))
    assert budget.concurrent_peak <= budget.pool_tokens
    return budget


# ----------------------------------------------------------------- wall budget


#: Per-member wall budget by thinking mode, seconds (operator 2026-09-27, DS41 run 10j).
DEFAULT_MEMBER_WALL_S: dict[str, float] = {"off": 2700, "medium": 4500, "default": 4500}
#: The whole panel's hard wall, seconds: authoring AND the winner check.
DEFAULT_PANEL_WALL_S = 5400
#: A member still authoring past this multiple of a finisher's wall is ended (0 = off).
DEFAULT_CANCEL_FACTOR = 2.0

#: Why a member's call was ended by the panel (`_Member.stop_cause`).
STOP_RUN = "run_stop"
STOP_WINNER = "winner"
STOP_EARLY_CANCEL = "early_cancel"
STOP_MEMBER_BUDGET = "member_budget"
STOP_PANEL_WALL = "panel_wall"
BUDGET_STOPS = frozenset({STOP_MEMBER_BUDGET, STOP_PANEL_WALL})


class MemberBudgetExhausted(ActorTransient):
    """A panel member stopped at its wall budget with no usable diff. An
    `ActorTransient`, never an `ActorStopped`: `iterate` must record an uncharged
    authoring harness failure, not a run stop."""

    failure_class = MEMBER_BUDGET_EXHAUSTED

    def __init__(self, message: str, *, failure_class: str | None = MEMBER_BUDGET_EXHAUSTED):
        super().__init__(message)
        self.failure_class = failure_class or MEMBER_BUDGET_EXHAUSTED


@dataclass(frozen=True)
class WallBudget:
    """How long a panel may take: each member's budget by thinking mode, the panel's
    hard wall, and the early-cancel factor (0 = off)."""
    member_s: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_MEMBER_WALL_S))
    panel_s: float = DEFAULT_PANEL_WALL_S
    cancel_factor: float = DEFAULT_CANCEL_FACTOR

    def for_member(self, spec: "AuthorSpec") -> float:
        """The member's own budget, never above the panel wall. A mode with no entry
        runs to the panel wall."""
        own = self.member_s.get(spec.thinking)
        return float(self.panel_s if own is None else min(float(own), float(self.panel_s)))

    def to_dict(self) -> dict:
        return {"member_s": {mode: _num(value) for mode, value in sorted(self.member_s.items())},
                "panel_s": _num(self.panel_s), "cancel_factor": float(self.cancel_factor)}


def parse_member_walls(text: str | None, *,
                       allowed: Sequence[str] = AUTHOR_MODES) -> dict[str, float]:
    """`--actor-authors-wall "off=2700,medium=4500"` -> {mode: seconds}, merged over
    `DEFAULT_MEMBER_WALL_S` (a mode not named keeps its default). Empty or None is the
    defaults. Refused: an entry that is not <mode>=<seconds>, an unknown mode, a
    non-positive or non-integer number of seconds, a mode named twice."""
    walls = dict(DEFAULT_MEMBER_WALL_S)
    seen: set[str] = set()
    for part in [item.strip() for item in str(text or "").split(",") if item.strip()]:
        mode, sep, value = part.partition("=")
        mode = mode.strip()
        try:
            seconds = int(value.strip())
        except ValueError:
            seconds = None
        if not sep or not mode or seconds is None:
            raise ValueError(f"--actor-authors-wall entry {part!r}: need <mode>=<seconds>")
        if mode not in allowed:
            raise ValueError(f"--actor-authors-wall entry {part!r}: unknown thinking mode "
                             f"{mode!r}; this build supports {list(allowed)}")
        if seconds <= 0:
            raise ValueError(f"--actor-authors-wall entry {part!r}: seconds must be positive")
        if mode in seen:
            raise ValueError(f"--actor-authors-wall names {mode!r} twice")
        seen.add(mode)
        walls[mode] = seconds
    return walls


def parse_panel_wall(value: Any) -> int:
    """`--actor-authors-panel-wall 5400` -> seconds; a positive integer, else refused."""
    try:
        seconds = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"--actor-authors-panel-wall needs a positive integer number of "
                         f"seconds, got {value!r}") from None
    if seconds <= 0:
        raise ValueError(f"--actor-authors-panel-wall must be positive, got {seconds}")
    return seconds


def parse_cancel_factor(value: Any) -> float:
    """`--actor-authors-cancel-factor 2.0` -> the factor; 0 turns early cancel off.
    Refused: not a finite number, negative, or between 0 and 1 (a member would be
    cancelled before it had run as long as the finisher)."""
    try:
        factor = float(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"--actor-authors-cancel-factor needs a number, got {value!r}") from None
    if factor != factor or factor in (float("inf"), float("-inf")):
        raise ValueError(f"--actor-authors-cancel-factor must be finite, got {value!r}")
    if factor != 0 and factor < 1:
        raise ValueError(f"--actor-authors-cancel-factor must be 0 (off) or >= 1, got {factor}")
    return factor


def wall_budget(member_walls: str | None = None, panel_wall: Any = DEFAULT_PANEL_WALL_S,
                cancel_factor: Any = DEFAULT_CANCEL_FACTOR, *,
                allowed: Sequence[str] = AUTHOR_MODES) -> WallBudget:
    """The three knobs, parsed and refused together (run.py resolves them at startup)."""
    return WallBudget(parse_member_walls(member_walls, allowed=allowed),
                      parse_panel_wall(panel_wall), parse_cancel_factor(cancel_factor))


# ----------------------------------------------------------------- validation


@dataclass
class Validation:
    """A validator's verdict on one member's scratch diff.

    `score` orders FAILING diffs (higher got further: 1 = a diff inside the target
    surface, 2 = passed the host integrity screen, then one per external check that
    passed); the first PASSING diff wins regardless of score."""
    passed: bool
    score: int = 0
    reason: str = ""
    checks: dict = field(default_factory=dict)
    validator: str = ""

    def to_dict(self) -> dict:
        return {"passed": self.passed, "score": self.score, "reason": self.reason or None,
                "checks": self.checks, "validator": self.validator}


#: Where a member's ak-check build dir sits: a marked registry dir beside its tree.
CHECK_DIR_NAME = "ak-check"
#: The registry `kind` of that dir (ak_check.SCRATCH_KIND on lane/ak-sandbox-20260926).
CHECK_DIR_KIND = "ak-check-build"


def integrity_validator(hypothesis, workspace: Path, base: str, paths: Sequence[str],
                        should_stop: Callable[[], bool], context=None) -> Validation:
    """The lane-diff and integrity checks: the winner check without ak-check, and the
    first stage in front of it.

    A diff exists against `base` and names the hypothesis's target surface (no stray
    untracked file: `integrity.lane_diff_report`), and the declared paths pass the host
    integrity screen (`integrity.validate_candidate`: dirty set == declared, kernel
    allowlist, no protected file, reward-hack scan)."""
    checks: dict[str, Any] = {}
    try:
        derived = integrity.lane_diff_report(Path(workspace), base,
                                             target_surface=hypothesis.target_surface)
    except integrity.LaneDiffRefused as exc:
        checks["target_surface_diff"] = {"passed": False, "reason": str(exc)[:500]}
        return Validation(False, 0, f"target_surface_diff: {exc}"[:500], checks, "integrity")
    if not derived:
        checks["target_surface_diff"] = {"passed": False, "reason": "no diff against the base"}
        return Validation(False, 0, "target_surface_diff: no diff against the base",
                          checks, "integrity")
    checks["target_surface_diff"] = {"passed": True, "paths": list(derived)}
    try:
        integrity.validate_candidate(Path(workspace), list(paths))
    except integrity.IntegrityRefused as exc:
        checks["integrity"] = {"passed": False, "refusal_class": exc.refusal_class,
                               "reason": str(exc)[:500]}
        return Validation(False, 1, f"integrity ({exc.refusal_class}): {exc}"[:500],
                          checks, "integrity")
    checks["integrity"] = {"passed": True}
    return Validation(True, 2, "", checks, "integrity")


def command_validator(template: Sequence[str] | str, *, name: str = "ak-check",
                      timeout_s: int = 1800, inconclusive_exits: Sequence[int] = (),
                      score_output: Callable[[str], int] | None = None
                      ) -> Callable[..., Validation]:
    """An external check (ak-check: compile + `--op-test`) run in the scratch tree.

    `template` is an argv (or a shell-split string) whose placeholders are filled per
    member: `{worktree}`, `{base}`, `{paths}`, `{scratch}` (the member's marked check
    dir beside its tree) and `{build_dir}` (the target's anchor build, from the round's
    context). It runs with the scratch tree as cwd, in its own process group, and is
    ended the same way as an actor when the panel no longer needs it. Exit 0 passes.
    An exit in `inconclusive_exits` (ak-check's 2: refused, or the sandbox could not
    run -- NOT evidence about the patch) passes as inconclusive, so the winner check
    falls back to the stages before it. When its LAST stdout line is a JSON object,
    each `{"<check>": {"passed": bool}}` entry is recorded and passing entries raise
    the failing score; `score_output(stdout)` may add to it."""
    argv_template = shlex.split(template) if isinstance(template, str) else list(template)
    if not argv_template:
        raise ValueError("a command validator needs a command")

    def validate(hypothesis, workspace: Path, base: str, paths: Sequence[str],
                 should_stop: Callable[[], bool], context=None) -> Validation:
        from . import actors
        build_dir = actors._anchor_build_dir(context or {})
        values = {"worktree": str(workspace), "base": str(base), "paths": " ".join(paths),
                  "scratch": str(Path(workspace).parent / CHECK_DIR_NAME),
                  "build_dir": build_dir or ""}
        if any("{build_dir}" in part for part in argv_template) and not build_dir:
            return Validation(True, 0, f"{name} inconclusive: the target names no anchor "
                              "build dir", {name: {"passed": None, "inconclusive": True}},
                              name)
        argv = [part.format(**values) for part in argv_template]
        import tempfile
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as out, \
             tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as err:
            try:
                done = actors._run_stoppable(argv, out=out, err=err, timeout_s=timeout_s,
                                             cwd=Path(workspace), should_stop=should_stop,
                                             extra={})
            except actors._StoppedChild as exc:
                raise ValidationStopped(f"{name} ended (rc {exc.returncode}): "
                                        "the panel no longer needs it") from None
            except subprocess.TimeoutExpired:
                return Validation(False, 0, f"{name} exceeded {timeout_s}s",
                                  {name: {"passed": False, "timed_out": True}}, name)
            except OSError as exc:
                return Validation(False, 0, f"{name} could not run: {exc}",
                                  {name: {"passed": False, "error": str(exc)[:300]}}, name)
            out.seek(0)
            err.seek(0)
            stdout, stderr = out.read()[-20000:], err.read()[-4000:]
        stages: dict[str, Any] = {}
        tail = stdout.strip().splitlines()[-1:] if stdout.strip() else []
        if tail:
            try:
                parsed = json.loads(tail[0])
                if isinstance(parsed, dict):
                    stages = {key: value for key, value in parsed.items()
                              if isinstance(value, dict) and "passed" in value}
            except ValueError:
                stages = {}
        passed_stages = sum(1 for value in stages.values() if value.get("passed") is True)
        if score_output is not None:
            try:
                passed_stages += int(score_output(stdout))
            except Exception:      # noqa: BLE001 -- scoring is advisory
                pass
        record = {"passed": done.returncode == 0, "returncode": done.returncode,
                  "stages": stages, "stdout_tail": stdout[-2000:] or None,
                  "stderr_tail": stderr[-1000:] or None}
        if done.returncode == 0:
            return Validation(True, passed_stages, "", {name: record}, name)
        if done.returncode in tuple(inconclusive_exits):
            record.update(passed=None, inconclusive=True)
            return Validation(True, 0, f"{name} inconclusive (rc {done.returncode}): "
                              f"{(stdout or stderr).strip()[-300:]}", {name: record}, name)
        return Validation(False, passed_stages,
                          f"{name} failed (rc {done.returncode}): "
                          f"{(stderr or stdout).strip()[-400:]}", {name: record}, name)

    validate.__name__ = f"command_validator[{name}]"
    return validate


def chain_validators(*validators: Callable[..., Validation]) -> Callable[..., Validation]:
    """Run validators in order; the first failure stops the chain. The chain's failing
    score is 10 per stage passed plus the failing stage's own score."""
    def validate(hypothesis, workspace, base, paths, should_stop, context=None) -> Validation:
        checks: dict[str, Any] = {}
        names: list[str] = []
        for stage, validator in enumerate(validators):
            if should_stop():
                raise ValidationStopped("the panel no longer needs this validation")
            result = validator(hypothesis, workspace, base, paths, should_stop,
                               context=context)
            checks.update(result.checks)
            names.append(result.validator or getattr(validator, "__name__", f"stage{stage}"))
            if not result.passed:
                return Validation(False, 10 * stage + result.score, result.reason, checks,
                                  "+".join(names))
        return Validation(True, 10 * len(validators), "", checks, "+".join(names))
    validate.__name__ = "+".join(getattr(v, "__name__", "validator") for v in validators)
    return validate


def ak_check_validator(script: Path | str, *, python: str | None = None,
                       timeout_s: int = 1800) -> Callable[..., Validation]:
    """`ak-check --op-test` (lane/ak-sandbox-20260926) on a member's scratch tree: the
    compile check with the anchor build's own commands, a relink of the touched libraries
    and test-backend-ops against the CPU reference, built in the member's marked check
    dir. Exit 0 passes, 1 fails (a failure that reached the op test outranks a compile
    failure), 2 is inconclusive (refused / sandbox fault: never evidence about the
    patch), so the integrity screen in front of it decides."""
    import sys as _sys

    def reached_op_test(stdout: str) -> int:
        return 1 if "--- test-backend-ops" in stdout else 0

    return command_validator(
        [python or _sys.executable, str(script), "--op-test", "--lane", "{worktree}",
         "--build-dir", "{build_dir}", "--scratch", "{scratch}", "--base", "{base}"],
        name="ak-check", timeout_s=timeout_s, inconclusive_exits=(2,),
        score_output=reached_op_test)


# ----------------------------------------------------------------- the panel


@dataclass
class _Member:
    spec: AuthorSpec
    workspace: Path | None = None
    started: float | None = None
    finished: float | None = None
    outcome: str = "pending"      # diff | abstained | transient | report_missing | stopped | error
    result: str = "lost"          # won | lost | cancelled
    paths: tuple[str, ...] = ()
    report: dict | None = None
    reason: str = ""
    error: BaseException | None = None
    validation: Validation | None = None
    patch: dict | None = None
    patch_bytes: bytes | None = None
    metrics: dict = field(default_factory=dict)
    metrics_offset: int = 0
    panel_id: str = ""
    lane_log: Path | None = None
    released: bool = False
    retained: bool = False
    #: The member's other marked dirs, released with its tree, newest first (its
    #: ak-check dir, then its home dir holding the per-call opencode config).
    extra: list = field(default_factory=list)
    #: `actor_metrics.failure_evidence` of its last call (set by `_harvest_metrics`).
    evidence: dict = field(default_factory=dict)
    #: Its tree when its author call started (a salvage counts only this call's edit).
    before_tree: str | None = None
    #: Its wall budget (seconds from its start; `WallBudget.for_member`).
    wall_budget_s: float | None = None
    #: Why the panel ended its call (`STOP_*`), set once by its stop predicate.
    stop_cause: str | None = None
    budget_stopped: bool = False
    early_cancelled: bool = False
    timed_out: bool = False
    #: True while its diff is in the winner check (early cancel leaves it alone).
    validating: bool = False
    #: The lane-diff salvage of a budget-stopped / timed-out call ({usable, reason}).
    salvage: dict | None = None


class AuthorPanel:
    """Best-of-N authoring for one lane. Called in place of `planner.author`.

    `make_author(spec, workspace, should_stop)` returns an object with the planner's
    `author(hypothesis, context)` for one member (run.py: an `AgentPlanner` whose seat
    carries the member's thinking mode and `budget`'s limits, rooted at `workspace`).

    `scratch` is the run's scratch registry (`scratch.ScratchRegistry`): `scope(
    kind, name=...)` opens a marker-owned scope whose `worktree(repo, base_commit, name)`
    allocates a detached worktree released on every exit path (and `release(path)`, when
    it has one, releases one early); `ensure_free(bytes)` says whether the space exists.

    `walls` (`WallBudget`, default: the operator's 2026-09-27 budgets) bounds each
    member by thinking mode, the panel by its hard wall, and sets the early-cancel
    factor; `clock` is the monotonic clock those are measured on.

    `validator(hypothesis, workspace, base, paths, should_stop)` returns a `Validation`
    (default: `integrity_validator`; run.py chains ak-check in front of the build once
    it is wired). `retain(**patch)` publishes a member's patch bytes to the patch store
    (`archive.retain_patch_bytes` bound to the store) and returns its path.
    """

    def __init__(self, *, lane: str, specs: Sequence[AuthorSpec],
                 make_author: Callable[[AuthorSpec, Path, Callable[[], bool]], Any],
                 scratch: Any, budget: PoolBudget,
                 validator: Callable[..., Validation] = integrity_validator,
                 retain: Callable[..., Path | None] | None = None,
                 should_stop: Callable[[], bool] | None = None,
                 scratch_bytes_per_author: int = SCRATCH_BYTES_PER_AUTHOR,
                 walls: WallBudget | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if len(specs) < 2:
            raise ValueError("an author panel needs N >= 2; N=1 is the single-author path")
        if len(specs) != budget.n:
            raise ValueError(f"{len(specs)} authors but a budget for {budget.n}")
        for spec in specs:
            try:
                budget.for_member(spec)
            except KeyError as exc:
                raise ValueError(str(exc)) from None
        self.lane = lane
        self.specs = tuple(specs)
        self.make_author = make_author
        self.scratch = scratch
        self.budget = budget
        self.validator = validator
        self.retain = retain
        self.should_stop = should_stop or (lambda: False)
        self.scratch_bytes_per_author = int(scratch_bytes_per_author)
        self.walls = walls if walls is not None else WallBudget()
        self.clock = clock
        self._rounds = 0
        self._book = threading.RLock()

    # ------------------------------------------------------------- entry point

    def __call__(self, hypothesis, context: Mapping[str, Any], *,
                 lane: tuple[Path, str] | None,
                 solo: Callable[[Any, Mapping[str, Any]], Any],
                 record: Callable[[dict], None] | None = None):
        """One authoring round: the lane paths of the selected diff, `Abstain`, or the
        provider exception the single path would have raised."""
        self._rounds += 1
        panel_id = hashlib.sha256(
            f"{self.lane}|{getattr(hypothesis, 'mechanism_id', '')}|{time.time_ns()}|"
            f"{self._rounds}".encode()).hexdigest()[:16]
        row: dict[str, Any] = {
            "schema": PANEL_SCHEMA, "panel_id": panel_id, "lane": self.lane,
            "mechanism_id": getattr(hypothesis, "mechanism_id", None),
            "base": None if lane is None else str(lane[1] or ""),
            "n": len(self.specs), "pool": self.budget.to_dict(),
            "validator": getattr(self.validator, "__name__", type(self.validator).__name__),
            "selection": None, "winner": None, "fallback_reason": None,
            "walls": self.walls.to_dict(),
            "members": [], "scratch": {}, "recorded_at": _now()}
        emitted = [False]

        def emit() -> None:
            if emitted[0]:
                return
            emitted[0] = True
            self._write_row(lane, row)
            if record is not None:
                try:
                    record(dict(row))
                except Exception:      # noqa: BLE001 -- the record never fails the round
                    pass

        try:
            result = self._round(hypothesis, context, lane, row, panel_id)
        except _FallBack as fallback:
            row["selection"] = SELECTION_FALLBACK
            row["fallback_reason"] = str(fallback)
            emit()
            # The single-author path, byte-identical: the lane's own author on the lane.
            return solo(hypothesis, context)
        except BaseException:
            emit()
            raise
        emit()
        return result

    # ------------------------------------------------------------- one round

    def _round(self, hypothesis, context, lane, row, panel_id):
        if lane is None:
            raise _FallBack("no owner-reset lane for this draw")
        worktree, base = Path(lane[0]), str(lane[1] or "")
        if not base:
            raise _FallBack("the owner recorded no reset base for this lane")
        try:
            head = _git(worktree, "rev-parse", "HEAD").strip()
        except PanelSetupRefused as exc:
            raise _FallBack(f"lane unreadable: {exc}") from None
        if head != base:
            raise _FallBack(f"lane HEAD {head[:12]} is not the reset base {base[:12]}")
        wanted = self.scratch_bytes_per_author * len(self.specs)
        row["scratch"]["requested_bytes"] = wanted
        try:
            enough = bool(self.scratch.ensure_free(wanted))
        except Exception as exc:      # noqa: BLE001 -- an unreadable guard is a refusal
            raise _FallBack(f"scratch space check failed: {type(exc).__name__}: {exc}") from None
        if not enough:
            raise _FallBack(f"scratch space guard refused {wanted} bytes for "
                            f"{len(self.specs)} author trees")
        # The lane's current change (a later round: the previous round's rejected
        # patch, which the single author would edit on top of) seeds every member.
        try:
            seed = archive.capture_patch(worktree)
            lane_tree = integrity.candidate_tree(worktree)
        except Exception as exc:      # noqa: BLE001
            raise _FallBack(f"lane state unreadable: {type(exc).__name__}: {exc}") from None
        row["seed_patch_sha256"] = (hashlib.sha256(seed[1]).hexdigest()
                                    if seed is not None else None)
        lane_log = worktree.parent / actor_metrics.REPLY_DIR_NAME / actor_metrics.CALL_LOG_NAME
        members = [_Member(spec, panel_id=panel_id, lane_log=lane_log) for spec in self.specs]
        started_stats = _stats(self.scratch)
        try:
            with self.scratch.scope("call", name=f"{self.lane}-authors-{panel_id}") as scope:
                try:
                    self._allocate(scope, members, worktree, base, seed, panel_id)
                    return self._race(hypothesis, context, scope, members, worktree, base,
                                      lane_tree, row)
                finally:
                    # Every path: retain what exists, release what is still held (the
                    # scope exit releases anything this misses), describe every member.
                    for member in members:
                        self._retire(scope, member, hypothesis)
                    row["members"] = [self._describe(member, row) for member in members]
        finally:
            # After the scope closed: the registry's own created/removed/sweep counters.
            row["scratch"].update(_scratch_delta(started_stats, _stats(self.scratch)))

    def _allocate(self, scope, members, worktree: Path, base: str, seed,
                  panel_id: str = "") -> None:
        parents: set[Path] = set()
        for member in members:
            name = f"{self.lane}-{panel_id}-{member.spec.label}"
            try:
                if callable(getattr(scope, "dir", None)):
                    # A marked directory of its own per member, the worktree inside it:
                    # the actor's per-call config and reply dir land beside the tree
                    # (`workspace.parent`) and are released with it (worktree first,
                    # then the directory: the scope releases in reverse order).
                    home = Path(scope.dir("author", name))
                    path = Path(scope.worktree(worktree, base, name, at=home / "tree"))
                    # The member's ak-check build dir (the author's own sandbox calls
                    # and the winner check share it; released first, it is newest).
                    check = Path(scope.dir(CHECK_DIR_KIND, name, at=home / CHECK_DIR_NAME))
                    member.extra = [check, home]
                else:
                    path = Path(scope.worktree(worktree, base, name))
            except Exception as exc:      # noqa: BLE001
                raise _FallBack(f"scratch worktree for {member.spec.label} refused: "
                                f"{type(exc).__name__}: {exc}") from None
            member.workspace = path
            # The actor writes its per-call opencode config and its actor-replies beside
            # the workspace (`workspace.parent`): two members under one parent would
            # overwrite each other's config, i.e. run the wrong thinking mode.
            if path.parent in parents:
                raise _FallBack(f"scratch worktrees share the parent {path.parent}; each "
                                "author needs its own (per-call opencode config)")
            parents.add(path.parent)
            try:
                if _git(path, "rev-parse", "HEAD").strip() != base:
                    raise _FallBack(f"scratch tree for {member.spec.label} is not at {base[:12]}")
                if seed is not None:
                    _git(path, "apply", "--binary", "-", input_bytes=seed[1])
            except PanelSetupRefused as exc:
                raise _FallBack(f"scratch tree for {member.spec.label} unusable: {exc}") from None
            member.metrics_offset = _size(_call_log(path))

    def _race(self, hypothesis, context, scope, members, worktree, base, lane_tree, row):
        lock = threading.Lock()
        cancel = threading.Event()
        winner: list[_Member] = []
        order: list[_Member] = []
        origin = self.clock()
        row["started_at"] = _now()
        panel_s = float(self.walls.panel_s)
        factor = float(self.walls.cancel_factor or 0.0)
        #: The first GENUINE finish (not a winner: that cancels everyone anyway):
        #: {label, wall_s, threshold_s}; members still authoring past the threshold
        #: without an ak-check-passing patch are ended.
        finish: dict[str, Any] = {}
        for member in members:
            member.wall_budget_s = self.walls.for_member(member.spec)

        def elapsed() -> float:
            return self.clock() - origin

        def author_stop(member: _Member) -> Callable[[], bool]:
            """The member's actor stop predicate (the C22 path TERMs its process group
            when it turns True). The first cause is kept; it stays True after."""
            def stop() -> bool:
                if member.stop_cause is not None:
                    return True
                cause = None
                if self.should_stop():
                    cause = STOP_RUN
                elif cancel.is_set():
                    cause = STOP_WINNER
                else:
                    now = elapsed()
                    ran = now - (member.started or 0.0)
                    threshold = finish.get("threshold_s")
                    if now >= panel_s:
                        cause = STOP_PANEL_WALL
                    elif member.wall_budget_s is not None and ran >= member.wall_budget_s:
                        cause = STOP_MEMBER_BUDGET
                    elif threshold is not None and ran > threshold \
                            and not self._check_passing(member):
                        cause = STOP_EARLY_CANCEL
                if cause is None:
                    return False
                member.stop_cause = cause
                return True
            return stop

        def validate_stop(member: _Member) -> Callable[[], bool]:
            """The winner check's stop: a winner, a run stop, or the panel wall (never
            the member's own budget: a budget-stopped member's salvaged diff is
            checked, and early cancel leaves a member in its check alone)."""
            def stop() -> bool:
                if self.should_stop() or cancel.is_set():
                    return True
                if elapsed() >= panel_s:
                    if member.stop_cause is None:
                        member.stop_cause = STOP_PANEL_WALL
                    return True
                return False
            return stop

        def run(member: _Member) -> None:
            member.started = elapsed()
            try:
                self._author_one(member, hypothesis, context, base, author_stop(member))
                self._after_author(member, hypothesis, base)
                if member.outcome == "diff":
                    member.validating = True
                    try:
                        member.validation = self.validator(hypothesis, member.workspace, base,
                                                           member.paths, validate_stop(member),
                                                           context=context)
                    except ValidationStopped as exc:
                        member.validation = Validation(False, -1, str(exc), {"stopped": True},
                                                       "stopped")
                    except Exception as exc:      # noqa: BLE001 -- a validator fault is a failure
                        member.validation = Validation(
                            False, -1, f"validator failed: {type(exc).__name__}: {exc}"[:500],
                            {"error": True}, "error")
                    finally:
                        member.validating = False
            finally:
                member.finished = elapsed()
                self._harvest_metrics(member)
                with lock:
                    order.append(member)
                    # A passing diff wins even under a run stop: it lands on the lane
                    # and `iterate`'s next stage boundary keeps it (critic2 checkpoint).
                    won = (member.validation is not None and member.validation.passed
                           and not winner)
                    if won:
                        winner.append(member)
                        member.result = "won"
                        cancel.set()
                    elif factor > 0 and not winner and "threshold_s" not in finish \
                            and self._genuine_finish(member):
                        wall = round(max(0.0, member.finished - member.started), 3)
                        finish.update(label=member.spec.label, wall_s=wall,
                                      threshold_s=round(factor * wall, 3))
                    # Losers already finished when the winner lands, or this member
                    # finishing after it: retained for the record, released at once.
                    losers = ([m for m in order if m is not member] if won else
                              [member] if winner and winner[0] is not member else [])
                for loser in losers:
                    self._retire(scope, loser, hypothesis)

        threads = [threading.Thread(target=run, args=(member,), daemon=True,
                                    name=f"{self.lane}-{member.spec.label}")
                   for member in members]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        row["finished_at"] = _now()
        spans = [(m.started, m.finished) for m in members
                 if m.started is not None and m.finished is not None]
        row["overlap_s"] = (round(max(0.0, min(f for _s, f in spans) - max(s for s, _f in spans)), 3)
                            if len(spans) >= 2 else 0.0)
        walls = row.setdefault("walls", self.walls.to_dict())
        walls["panel_wall_s"] = round(elapsed(), 3)
        walls["panel_wall_hit"] = any(m.stop_cause == STOP_PANEL_WALL for m in members)
        walls["early_cancel"] = (dict(finish, cancelled=[m.spec.label for m in members
                                                          if m.early_cancelled])
                                 if finish else None)
        for member in members:
            ended = member.outcome == "stopped" or (
                member.validation is not None and member.validation.validator == "stopped")
            # A member that spent its budget simply LOST to the winner (it was not
            # cancelled by it).
            if winner and member.result != "won" and ended and not member.budget_stopped:
                member.result = "cancelled"
        if winner:
            selected = winner[0]
            row["selection"] = SELECTION_FIRST_PASSING
        else:
            with_diff = [m for m in order if m.outcome == "diff"]
            if not with_diff:
                row["selection"] = SELECTION_NONE
                # Retained first: a member that abstained over a diff it made (DS41
                # run 10h a0-off) hands the next attempt its patch path.
                for member in members:
                    self._retire(scope, member, hypothesis)
                return self._no_diff(members)
            # The existing rules: the diff that got furthest through the checks, ties to
            # the first finished; it goes to the lane and the normal gates judge it.
            selected = max(with_diff, key=lambda m: (m.validation.score if m.validation
                                                     else -1, -order.index(m)))
            selected.result = "won"
            row["selection"] = SELECTION_BEST_FAILING
        row["winner"] = selected.spec.label
        for member in members:
            if member is not selected:
                self._retire(scope, member, hypothesis)
        with self._book:
            self._retain(selected, hypothesis)
        paths = self._apply(selected, worktree, base, lane_tree)
        self._retire(scope, selected, hypothesis)
        from .actors import AuthorPaths
        report = dict(selected.report or {})
        report.update({"author_panel": row["panel_id"], "author": selected.spec.label})
        return AuthorPaths(paths, report=report)

    def _author_one(self, member: _Member, hypothesis, context, base: str,
                    should_stop: Callable[[], bool]) -> None:
        before = None
        try:
            before = integrity.candidate_tree(member.workspace)
        except Exception:      # noqa: BLE001 -- recovery then refuses
            before = None
        member.before_tree = before
        try:
            author = self.make_author(member.spec, member.workspace, should_stop)
            result = author.author(hypothesis, dict(context))
        except ActorStopped as exc:
            member.outcome, member.reason, member.error = "stopped", str(exc)[:500], exc
            return
        except AuthorReportMissing as missing:
            # The same recovery the single path makes on its lane (`loop.
            # _recover_author_report`), on this member's scratch tree.
            try:
                derived = integrity.lane_diff_report(member.workspace, base,
                                                     target_surface=hypothesis.target_surface,
                                                     before_tree=before)
            except Exception as exc:      # noqa: BLE001 -- LaneDiffRefused, git faults
                derived = None
                member.reason = f"{missing}; lane diff not usable: {exc}"[:500]
            if not derived:
                member.outcome, member.error = "report_missing", missing
                member.reason = member.reason or str(missing)[:500]
                return
            member.outcome, member.paths = "diff", tuple(derived)
            member.report = {"report_source": "lane_diff", "base": base,
                             "paths": list(derived),
                             "failure_class": getattr(missing, "failure_class", None),
                             "path_normalized": False, "reply_refusal": str(missing)[:500]}
            return
        except ActorTransient as exc:
            member.outcome, member.reason, member.error = "transient", str(exc)[:500], exc
            return
        except Exception as exc:      # noqa: BLE001 -- contained per member, re-raised if alone
            member.outcome, member.error = "error", exc
            member.reason = f"{type(exc).__name__}: {exc}"[:500]
            return
        if isinstance(result, Abstain):
            member.outcome, member.reason = "abstained", result.reason[:500]
            return
        member.outcome, member.paths = "diff", tuple(str(p) for p in result)
        report = getattr(result, "report", None)
        member.report = dict(report) if isinstance(report, Mapping) else None

    def _after_author(self, member: _Member, hypothesis, base: str) -> None:
        """Say how a member's call ended when the panel (or its budget) ended it.

        Budget stop (its own wall, or the panel's): `budget_stopped`, and its lane diff
        is salvaged; with none usable it is a `member_budget_exhausted` harness failure
        (an `ActorTransient`, never a run stop). Early cancel: `result: cancelled`,
        `reason: early_cancel`. A hard timeout (`ActorTimedOut`, never retried in a
        panel): its lane diff is salvaged the same way. (A seat's per-call
        `budget_exhausted` keeps its existing handling: a harness failure, no salvage.)"""
        cause = member.stop_cause
        if member.outcome == "stopped" and cause in BUDGET_STOPS:
            member.budget_stopped = True
            detail = member.reason
            budget = panel_s = float(self.walls.panel_s)
            if cause == STOP_MEMBER_BUDGET and member.wall_budget_s is not None:
                budget = member.wall_budget_s
            what = ("its wall budget" if cause == STOP_MEMBER_BUDGET
                    else "the panel's hard wall")
            self._salvage(member, hypothesis, base,
                          f"stopped at {what} ({_num(budget)} s)", detail)
            if member.outcome != "diff":
                member.reason = (f"{MEMBER_BUDGET_EXHAUSTED}: stopped at {what} "
                                 f"({_num(budget)} s; panel wall {_num(panel_s)} s) with no "
                                 f"usable diff ({(member.salvage or {}).get('reason')}); "
                                 f"{detail}")[:500]
                member.error = MemberBudgetExhausted(member.reason)
        elif member.outcome == "stopped" and cause == STOP_EARLY_CANCEL:
            member.early_cancelled = True
            member.result = "cancelled"
            member.reason = STOP_EARLY_CANCEL
        elif member.outcome == "transient" and getattr(member.error, "timed_out", False):
            member.timed_out = True
            self._salvage(member, hypothesis, base, "timed out", member.reason)

    def _salvage(self, member: _Member, hypothesis, base: str, how: str, detail: str) -> None:
        """A stopped member's own edit, as the report-missing recovery reads it: the
        lane diff against the base, only when THIS call changed the tree. Usable or
        not, the validator (and then the normal gates) decide what it is worth."""
        try:
            integrity.take_report_artifacts(Path(member.workspace))
        except Exception:      # noqa: BLE001 -- the diff read below then refuses a stray
            pass
        try:
            derived = integrity.lane_diff_report(member.workspace, base,
                                                 target_surface=hypothesis.target_surface,
                                                 before_tree=member.before_tree)
        except Exception as exc:      # noqa: BLE001 -- LaneDiffRefused, git faults
            member.salvage = {"usable": False, "reason": f"lane diff not usable: {exc}"[:400]}
            return
        if not derived:
            member.salvage = {"usable": False, "reason": "no diff made by this call"}
            return
        member.salvage = {"usable": True, "paths": list(derived)}
        member.outcome, member.paths = "diff", tuple(derived)
        member.reason = f"salvaged after it {how}: {detail}"[:500]
        member.report = {"report_source": "lane_diff", "base": base, "paths": list(derived),
                         "failure_class": (MEMBER_BUDGET_EXHAUSTED if member.budget_stopped
                                           else getattr(member.error, "failure_class", None)),
                         "path_normalized": False, "reply_refusal": str(detail)[:500],
                         "salvaged": how}
        member.error = None

    def _check_passing(self, member: _Member) -> bool:
        """Whether the member's own ak-check sandbox (`ak-check-<tree>.jsonl` in its reply
        dir) last PASSED: a member holding an ak-check-passing patch is never cancelled
        early. Evidence only: an unreadable log is "no"."""
        if member.workspace is None:
            return False
        workspace = Path(member.workspace)
        log = workspace.parent / actor_metrics.REPLY_DIR_NAME / f"ak-check-{workspace.name}.jsonl"
        try:
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return False
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            return isinstance(row, Mapping) and row.get("status") == "pass"
        return False

    @staticmethod
    def _genuine_finish(member: _Member) -> bool:
        """A finish that starts the early-cancel clock: a genuine abstention, or a diff
        the validator judged (not a budget stop, timeout, cancel or harness failure)."""
        if member.budget_stopped or member.timed_out or member.early_cancelled:
            return False
        if member.outcome == "abstained":
            return True
        return (member.outcome == "diff" and member.validation is not None
                and member.validation.validator not in ("stopped", "error"))

    def failure_record(self, member: _Member, panel_id: str = "") -> dict:
        """`loop.author_failure_record` for one member that produced no diff."""
        evidence = dict(member.evidence)
        if member.wall_budget_s is not None:
            evidence["wall_budget_s"] = _num(member.wall_budget_s)
        if member.budget_stopped:
            evidence.update(failure_class=MEMBER_BUDGET_EXHAUSTED, budget_stopped=True)
        if member.timed_out:
            evidence["timed_out"] = True
        if member.early_cancelled:
            evidence["early_cancelled"] = True
        return author_failure_record(
            label=member.spec.label, thinking=member.spec.thinking, outcome=member.outcome,
            reason=member.reason, evidence=evidence, patch=member.patch,
            validation=None if member.validation is None else member.validation.to_dict(),
            panel_id=panel_id or member.panel_id)

    def _no_diff(self, members: Sequence[_Member]):
        """No member produced a diff. A stop is a stop. Otherwise every member gets a
        failure record (`failure_record`): when ANY failed on its own account (an
        abstention, a report of a change it never made on an uncapped final step) the
        round returns `loop.AuthoringFailure` -- an AUTHORING failure of an accepted
        hypothesis, one attempt charged however many failed; when every failure was the
        harness's, the first provider exception is raised as before, carrying the
        records as `author_failures` (`iterate` then records a harness failure)."""
        summary = "; ".join(f"{m.spec.label}: {m.outcome}: {m.reason}"[:400] for m in members)
        if self.should_stop():
            raise ActorStopped(f"stop asked during the author panel ({summary})")
        records = [self.failure_record(member) for member in members]
        if any(record["class"] == "authoring" for record in records):
            reason = "; ".join(f"{r['label']}: {r['outcome']} ({r['class']}): {r['reason']}"
                               for r in records)
            return AuthoringFailure(reason[:4000] or "author panel: no diff",
                                    members=tuple(records))
        failure: BaseException | None = None
        for kind in ("report_missing", "transient", "stopped"):
            for member in members:
                if failure is None and member.outcome == kind and member.error is not None:
                    failure = _with_message(member.error, f"author panel: every author "
                                                          f"failed ({summary})")
        if failure is None:
            errors = [m.error for m in members if m.error is not None]
            failure = errors[0] if errors else ActorTransient(
                f"author panel: no author produced a diff ({summary})")
        if isinstance(failure, ActorStopped):
            # No run stop was asked (checked above): the panel ended these calls itself
            # (a budget, a cancel). `iterate` must read a harness failure, not a stop.
            failure = ActorTransient(f"author panel: every author failed ({summary})")
        try:
            failure.author_failures = records
        except Exception:      # noqa: BLE001 -- an exception type without __dict__
            pass
        raise failure

    def _apply(self, member: _Member, worktree: Path, base: str, lane_tree: str) -> tuple[str, ...]:
        """Put the selected member's full diff (seed included) on the real lane."""
        if member.patch_bytes is None:
            raise PanelSetupRefused(f"{member.spec.label}: no patch bytes to apply")
        if integrity.candidate_tree(worktree) != lane_tree:
            raise PanelSetupRefused("the lane changed while the author panel ran")
        # Back to the owner's reset state (the seed was retained by the round that made
        # it), then exactly the winner's bytes.
        _git(worktree, "checkout", "--detach", "--force", base)
        _git(worktree, "reset", "--hard", base)
        _git(worktree, "clean", "-fd", "ggml/", "src/", check=False)
        _git(worktree, "apply", "--check", "--binary", "-", input_bytes=member.patch_bytes)
        _git(worktree, "apply", "--binary", "-", input_bytes=member.patch_bytes)
        return member.paths

    # ------------------------------------------------------------- bookkeeping

    def _retire(self, scope, member: _Member, hypothesis) -> None:
        """Retain a finished member's patch, then release its tree (once, any thread)."""
        with self._book:
            self._retain(member, hypothesis)
            self._release(scope, member)

    def _retain(self, member: _Member, hypothesis) -> None:
        if member.retained or member.workspace is None or member.released \
                or member.started is None:
            return
        member.retained = True
        try:
            captured = archive.capture_patch(member.workspace)
        except Exception as exc:      # noqa: BLE001 -- evidence, never a failure
            member.patch = {"error": f"capture failed: {type(exc).__name__}: {exc}"[:300]}
            return
        if captured is None:
            member.patch = None
            return
        head, patch, additions = captured
        member.patch_bytes = patch
        member.patch = {"patch_sha256": hashlib.sha256(patch).hexdigest(),
                        "bytes": len(patch), "head": head}
        if self.retain is None:
            return
        try:
            kept = self.retain(patch=patch, head=head, lane=f"{self.lane}.{member.spec.label}",
                               mechanism_id=getattr(hypothesis, "mechanism_id", None) or "unnamed",
                               worktree=str(member.workspace), untracked_source_paths=additions)
            if kept is not None:
                member.patch["patch_file"] = str(Path(kept).resolve())
        except Exception as exc:      # noqa: BLE001
            member.patch["retain_error"] = f"{type(exc).__name__}: {exc}"[:300]

    def _release(self, scope, member: _Member) -> None:
        if member.released or member.workspace is None:
            return
        release = getattr(scope, "release", None)
        if release is None:
            return      # the scope exit releases it
        try:
            release(member.workspace)
            member.released = True
        except Exception:      # noqa: BLE001 -- the scope exit still releases it
            return
        # A finished loser's whole footprint goes now, not at the panel's end: its
        # check dir, then the home dir (tree first: the home dir holds it).
        for path in member.extra:
            try:
                release(path)
            except Exception:      # noqa: BLE001 -- the scope exit still releases it
                pass

    def _harvest_metrics(self, member: _Member) -> None:
        """Move this member's actor-call rows from beside its scratch tree into the
        LANE's `actor-calls.jsonl`, each metrics row annotated with the panel member it
        belongs to (joined to `author-panel.jsonl` by `panel_id` + `label`).

        Moved, not copied: the scratch reply dir is released with the tree, and a copy
        left behind would be counted twice by `actor_metrics.summarize`. The closed,
        self-hashed VB-AK-SEAT record rows are moved verbatim."""
        if member.workspace is None:
            return
        path = _call_log(member.workspace)
        rows: list[dict] = []
        raw_lines: list[str] = []
        try:
            with open(path, "rb") as handle:
                handle.seek(member.metrics_offset)
                raw_lines = handle.read().decode("utf-8", "replace").splitlines()
        except OSError:
            raw_lines = []
        annotation = {"panel_id": member.panel_id, "label": member.spec.label,
                      "thinking": member.spec.thinking, "lane": self.lane}
        moved: list[str] = []
        for line in raw_lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError:
                moved.append(line)
                continue
            if isinstance(value, dict):
                rows.append(value)
                if value.get("schema") == actor_metrics.METRICS_SCHEMA:
                    value = {**value, "author_panel": annotation}
                    line = json.dumps(value, sort_keys=True)
            moved.append(line)
        if moved and member.lane_log is not None:
            try:
                member.lane_log.parent.mkdir(parents=True, exist_ok=True)
                with open(member.lane_log, "a", encoding="utf-8") as handle:
                    handle.write("".join(item + "\n" for item in moved))
                with open(path, "r+b") as handle:
                    handle.truncate(member.metrics_offset)
            except OSError:
                pass
        metric_rows = [r for r in rows if r.get("schema") == actor_metrics.METRICS_SCHEMA]
        totals = [((r.get("opencode") or r.get("orchestrator") or {}).get("totals") or {})
                  for r in metric_rows]
        member.metrics = {
            "calls": len(metric_rows),
            "wall_s_actor": round(sum(float(r.get("wall_s") or 0) for r in metric_rows), 3),
            "steps": _sum(totals, "steps"), "decoded_tokens": _sum(totals, "decoded_tokens"),
            "compactions": _sum(totals, "compactions"),
            "failure_class": next((r.get("failure_class") for r in reversed(metric_rows)
                                   if r.get("failure_class")), None),
            "metrics_error": next((r.get("metrics_error") for r in reversed(metric_rows)
                                   if r.get("metrics_error")), None),
        }
        # How the LAST call ended (capped steps, limits, timeout, its ak-check usage):
        # what classifies a no-diff member as the author's failure or the harness's.
        last = actor_metrics.failure_evidence(metric_rows[-1]) if metric_rows else {}
        member.evidence = {**last, "steps": member.metrics["steps"],
                           "decoded_tokens": member.metrics["decoded_tokens"],
                           "failure_class": member.metrics["failure_class"]}
        for key in ("final_step_capped", "output_capped_steps", "output_limit",
                    "context_limit", "ak_check"):
            if last.get(key) is not None:
                member.metrics[key] = last[key]

    def _describe(self, member: _Member, row: Mapping[str, Any]) -> dict:
        wall = (round(member.finished - member.started, 3)
                if member.started is not None and member.finished is not None else None)
        body = {"label": member.spec.label, "thinking": member.spec.thinking,
                "result": member.result, "outcome": member.outcome,
                "reason": member.reason or None, "wall_s": wall,
                "started_offset_s": None if member.started is None else round(member.started, 3),
                "finished_offset_s": None if member.finished is None else round(member.finished, 3),
                "paths": list(member.paths),
                "report_source": (member.report or {}).get("report_source"),
                "validation": None if member.validation is None else member.validation.to_dict(),
                "patch": member.patch, "workspace": None if member.workspace is None
                else str(member.workspace), **member.metrics,
                "wall_budget_s": (None if member.wall_budget_s is None
                                  else _num(member.wall_budget_s)),
                "budget_stopped": member.budget_stopped,
                "early_cancelled": member.early_cancelled,
                "stop_cause": member.stop_cause, "timed_out": member.timed_out or None,
                "salvage": member.salvage}
        if member.budget_stopped:
            body["failure_class"] = MEMBER_BUDGET_EXHAUSTED
        try:
            body["budget"] = self.budget.for_member(member.spec).to_dict()
        except KeyError:
            pass
        if member.outcome not in ("diff", "pending"):
            # No diff from this member: whose failure it was, and a final step that
            # ended on the output cap with nothing to show surfaces as
            # `output_capped_empty` (DS41 run 10h a1-medium).
            record = self.failure_record(member)
            body["failure_class"] = record.get("failure_class")
            body["failure"] = {"class": record["class"],
                               "harness_reason": record.get("harness_reason")}
        return body

    def _write_row(self, lane, row: Mapping[str, Any]) -> None:
        """The panel row in the LANE's reply dir (the scratch trees are released)."""
        if lane is None:
            return
        target = Path(lane[0]).parent / actor_metrics.REPLY_DIR_NAME
        try:
            target.mkdir(parents=True, exist_ok=True)
            with open(target / PANEL_LOG, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        except OSError:
            pass


class _FallBack(Exception):
    """Internal: this round runs the single-author path instead (reason = message)."""


# ----------------------------------------------------------------- helpers


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git(repo: Path, *args: str, input_bytes: bytes | None = None, check: bool = True) -> str:
    env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(name, None)
    done = subprocess.run(["git", "-C", str(repo), *args], input=input_bytes,
                          capture_output=True, timeout=600, env=env)
    if check and done.returncode != 0:
        raise PanelSetupRefused(f"git {' '.join(args[:2])}: "
                                f"{done.stderr.decode('utf-8', 'replace').strip()[:400]}")
    return done.stdout.decode("utf-8", "replace")


def _call_log(workspace: Path) -> Path:
    return Path(workspace).parent / actor_metrics.REPLY_DIR_NAME / actor_metrics.CALL_LOG_NAME


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _num(value: Any) -> int | float:
    """Seconds as an int when whole (records read 2700, not 2700.0)."""
    value = float(value)
    return int(value) if value.is_integer() else round(value, 3)


def _sum(totals: Sequence[Mapping[str, Any]], key: str) -> int | None:
    values = [t.get(key) for t in totals if isinstance(t.get(key), (int, float))
              and not isinstance(t.get(key), bool)]
    return int(sum(values)) if values else None


def _stats(scratch: Any) -> dict:
    stats = getattr(scratch, "stats", None)
    if stats is None:
        return {}
    try:
        value = stats()
        return dict(value) if isinstance(value, Mapping) else {}
    except Exception:      # noqa: BLE001
        return {}


def _scratch_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict:
    """The registry's numeric counters moved by this round (bytes created/removed,
    sweeps), plus its final snapshot."""
    delta = {f"delta_{key}": after[key] - before.get(key, 0) for key in after
             if isinstance(after.get(key), (int, float)) and not isinstance(after.get(key), bool)
             and isinstance(before.get(key, 0), (int, float))}
    return {**delta, "registry": dict(after)} if after else delta


def _with_message(exc: BaseException, message: str) -> BaseException:
    """`exc` again with `message` in front (same type and failure_class when possible)."""
    failure_class = getattr(exc, "failure_class", None)
    text = f"{message}: {exc}"[:2000]
    try:
        new = type(exc)(text, failure_class=failure_class)      # type: ignore[call-arg]
    except TypeError:
        try:
            new = type(exc)(text)
        except Exception:      # noqa: BLE001
            return exc
    try:
        new.failure_class = failure_class      # type: ignore[attr-defined]
    except Exception:      # noqa: BLE001
        pass
    return new


__all__ = ["AUTHOR_MODES", "AUTHOR_OUTPUT_LIMIT", "AuthorPanel", "AuthorSpec",
           "DEFAULT_POOL_TOKENS", "MAX_AUTHORS", "MIN_COMPACTION_HEADROOM", "PANEL_LOG",
           "PANEL_SCHEMA", "POOL_RESERVE", "PanelSetupRefused", "PoolBudget",
           "PoolBudgetRefused", "SCRATCH_BYTES_PER_AUTHOR", "SELECTION_BEST_FAILING",
           "SELECTION_FALLBACK", "SELECTION_FIRST_PASSING", "SELECTION_NONE", "Validation",
           "ValidationStopped", "CHECK_DIR_KIND", "CHECK_DIR_NAME", "ak_check_validator",
           "author_budget", "chain_validators", "command_validator",
           "integrity_validator", "parse_authors", "CONTEXT_GRANULE", "DEFAULT_MODE_BUDGETS",
           "MemberBudget", "panel_budget", "parse_mode_budgets",
           "DEFAULT_CANCEL_FACTOR", "DEFAULT_MEMBER_WALL_S", "DEFAULT_PANEL_WALL_S",
           "MemberBudgetExhausted", "WallBudget", "parse_cancel_factor",
           "parse_member_walls", "parse_panel_wall", "wall_budget"]
