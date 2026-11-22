"""Planner authoring call (_author_plan and helpers), split verbatim from run.py (code-budget split)."""
from __future__ import annotations
from dataclasses import dataclass
from . import actor_opencode_config
from . import bestof


#: run.py's `--actor-authors` default (operator decision 2026-09-26: best-of-2, a mixed
#: pair). The library default (no panel) is the single-author path.
DEFAULT_ACTOR_AUTHORS = "off,medium"


#: `--actor-authors single`: one author, thinking from `--actor-author-thinking`.
SINGLE_AUTHOR = "single"


@dataclass(frozen=True)
class AuthorPlan:
    """What `--actor-authors` resolved to: a panel (N >= 2) or the single path."""
    specs: tuple = ()
    budget: object = None
    note: str = ""
    #: `bestof.WallBudget`: per-member wall by thinking mode, the panel's hard wall and
    #: the early-cancel factor (`--actor-authors-wall` / `-panel-wall` / `-cancel-factor`).
    walls: object = None

    @property
    def panel(self) -> bool:
        return len(self.specs) >= 2


def _shared_pool_lanes(args) -> int:
    """Lanes whose planner/author calls share the global planner's server (all of them
    unless `--lane-actor-models` moves a lane to another provider)."""
    from . import lane_actors
    workers = int(args.workers)
    return lane_actors.shared_pool_lanes(
        workers, lane_actors.parse(getattr(args, "lane_actor_models", None), workers=workers),
        getattr(args, "planner_model", None))


def _author_plan(args, planner_kind: str | None = None) -> AuthorPlan:
    """Resolve `--actor-authors` against this build and the pool, or raise ValueError.

    An EXPLICIT value that cannot run is refused. The run.py DEFAULT ("off,medium")
    degrades to the single path, loudly, when this build lacks a mode it names (the
    `medium` thinking mode lands with lane/ak-author-medium-20260926) or when several
    lanes would share :8083's pool. N=1 (`single`, or one mode) is the single path."""
    explicit = args.actor_authors is not None
    spec = args.actor_authors if explicit else DEFAULT_ACTOR_AUTHORS
    # The wall knobs are refused whatever N is (a typo must not wait for a panel run).
    # They bound best-of-N panels only: N=1 keeps the single path's timeout and retry.
    walls = bestof.wall_budget(
        getattr(args, "actor_authors_wall", None),
        getattr(args, "actor_authors_panel_wall", bestof.DEFAULT_PANEL_WALL_S),
        getattr(args, "actor_authors_cancel_factor", bestof.DEFAULT_CANCEL_FACTOR),
        allowed=tuple(actor_opencode_config.THINKING_CHOICES))
    if str(spec).strip() == SINGLE_AUTHOR:
        return AuthorPlan(note="single author (--actor-authors single)")
    try:
        specs = bestof.parse_authors(spec, allowed=actor_opencode_config.THINKING_CHOICES)
    except ValueError as exc:
        if explicit:
            raise
        return AuthorPlan(note=f"single author: default --actor-authors {spec!r} unavailable "
                               f"in this build ({exc})")
    if len(specs) == 1:
        return AuthorPlan(specs=specs, note=f"single author (thinking {specs[0].thinking})")
    if planner_kind is not None and planner_kind != "opencode":
        # The members differ ONLY by the per-call chat_template_kwargs of an opencode
        # author seat; on codex/claude/orchestrator they would be N identical calls.
        reason = (f"best-of-{len(specs)} races author thinking modes, which only an "
                  f"opencode author seat carries (planner backend: {planner_kind})")
        if explicit:
            raise ValueError(reason)
        return AuthorPlan(note=f"single author: {reason}")
    # Lanes whose planner/author run on ANOTHER provider (`--lane-actor-models`) do not
    # share :8083's pool: the panel budget still holds exactly one lane's authors there.
    if planner_kind == "opencode" and int(args.workers) > 1 and _shared_pool_lanes(args) > 1:
        reason = (f"{len(specs)} concurrent authors per lane with --workers {args.workers}: "
                  "the pool budget holds one lane's authors on :8083's unified pool, and "
                  "other lanes' calls would share it")
        if explicit:
            raise ValueError(reason)
        return AuthorPlan(note=f"single author: {reason}")
    budget = bestof.panel_budget(
        specs, pool_tokens=int(args.actor_pool_tokens),
        modes=bestof.parse_mode_budgets(getattr(args, "actor_authors_budget", None)))
    return AuthorPlan(specs=specs, budget=budget, walls=walls,
                      note=(f"best-of-{len(specs)} "
                            + " ".join(f"{m.label}(context={m.context_limit} "
                                       f"output={m.output_limit} compaction@{m.compaction_at} "
                                       f"wall={walls.for_member(m):.0f}s)"
                                       for m in budget.members)
                            + f" pool={budget.pool_tokens}-{budget.reserve}"
                            + f" panel_wall={walls.panel_s}s"
                            + f" cancel_factor={walls.cancel_factor:g}"))
