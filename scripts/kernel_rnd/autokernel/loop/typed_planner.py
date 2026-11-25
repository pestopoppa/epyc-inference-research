"""AK-EFF-P2-TYPED-PLANNER-S1: typed planner output over the controller hypothesis portfolio.

The planner chooses ONE option from a closed list instead of decoding a free-text proposal:
a portfolio row (mechanism enum = the row's `mechanism_id`), or one of two typed controls.
Free text (statement / falsifier / target) is allowed ONLY for the chosen row. The three
conditions from the 2026-11-07 offline replay (`/mnt/raid0/llm/tmp/ak-eff-p2-replay-20261107/`)
are part of the schema, not policy left to the prompt:

  (a) `exhausted` is a typed option distinct from `explore` ("none of these -> explore"): the
      replay's 18 abstentions only map onto the Choice through an explicit exhausted answer;
  (b) a reserved explore share >= 25%: `explore_required` forces the explore path whenever the
      running explore share would drop below the reserve (6 of 24 DS41 planner keeps had no
      sibling row);
  (c) do-not-repeat / retired rows stay SELECTABLE, flagged `premise_superseded` (the two
      biggest early DS41 keeps came from a class the portfolio listed as DEAD).

Pure functions: no actor call, no store write, no loop launch. Wiring into the live planner
call is the S2 A/B (granted window), not this module.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "epyc.autokernel.typed_planner.v1"
EXPLORE = "explore"
EXHAUSTED = "exhausted"
CONTROLS = (EXPLORE, EXHAUSTED)
EXPLORE_RESERVE = 0.25
MAX_OPTIONS = 255
CHOSEN_TEXT_FIELDS = ("statement", "falsifier", "target_surface", "target_symbol")
#: AK-SURFACE-SERVING-1: a row of this kind carries a closed `serving_knob` delta (built from a
#: seed by `serving_knobs.portfolio_rows`); the planner picks the arm, never writes an argv.
SERVING_KNOB = "serving-knob"
SERVING_KNOB_TEXT_FIELDS = ("statement", "falsifier")


class TypedPlannerRefused(ValueError):
    """The option list or the reply is not schema-valid; the message names why."""


def build_options(portfolio: Iterable[Mapping[str, Any]],
                  retired: Iterable[Mapping[str, Any]] = ()) -> list[dict]:
    """Option list: live portfolio rows, retired/DNR rows flagged premise_superseded
    (condition c: still selectable), then the two typed controls (condition a)."""
    options: list[dict] = []
    seen: set[str] = set()
    for rows, superseded in ((portfolio, False), (retired, True)):
        for row in rows:
            mid = str(row.get("mechanism_id") or "").strip()
            if not mid or mid in CONTROLS:
                raise TypedPlannerRefused(f"portfolio row lacks a usable mechanism_id: {row!r}"[:200])
            if mid in seen:
                continue
            seen.add(mid)
            opt = {"option": mid, "kind": "row", "premise_superseded": superseded,
                   "summary": str(row.get("summary") or row.get("statement") or "")[:300]}
            if row.get("kind") == SERVING_KNOB:
                if not isinstance(row.get("serving_knob"), Mapping):
                    raise TypedPlannerRefused(f"serving-knob row {mid!r} lacks its serving_knob delta")
                opt.update(kind=SERVING_KNOB, serving_knob=dict(row["serving_knob"]))
            options.append(opt)
    options += [{"option": o, "kind": "control", "premise_superseded": False, "summary": t} for o, t in (
        (EXPLORE, "none of these rows: explore source for a new mechanism"), (EXHAUSTED, "every row in force is spent: abstain, naming the rows"))]
    if len(options) > MAX_OPTIONS:
        raise TypedPlannerRefused(f"{len(options)} options exceed the {MAX_OPTIONS} Choice cap; "
                                  "de-duplicate the ledger by mechanism family")
    return options


def explore_required(prior_choices: Sequence[str], reserve: float = EXPLORE_RESERVE) -> bool:
    """Condition (b): True when taking a non-explore choice now would leave the explore share
    below `reserve`. `exhausted` counts as a call (it spends the reserve's denominator)."""
    if not 0 < reserve < 1:
        raise TypedPlannerRefused("explore reserve must be in (0, 1)")
    return sum(1 for c in prior_choices if c == EXPLORE) / (len(prior_choices) + 1) < reserve


def render_choice_prompt(options: Sequence[Mapping[str, Any]], *, must_explore: bool) -> str:
    lines = ["Choose exactly ONE option id below. Reply with ONE json object:",
             '{"choice": "<option id>", "statement": ..., "falsifier": ..., '
             '"target_surface": ..., "target_symbol": ...}',
             "Free-text fields are allowed ONLY for a row choice (or explore); for "
             f'"{EXHAUSTED}" give only {{"choice": "{EXHAUSTED}", "reason": "<rows that are spent>"}}.',
             "A row marked premise_superseded was retired under an earlier premise; it is still "
             "selectable if the current evidence revives it."]
    if must_explore:
        lines.append(f'The reserved exploration share is due: this call MUST choose "{EXPLORE}".')
    lines += [f"- {o['option']}{' [premise_superseded]' if o.get('premise_superseded') else ''}: {o.get('summary', '')}" for o in options]
    return "\n".join(lines)


def parse_choice(reply: str | Mapping[str, Any], options: Sequence[Mapping[str, Any]], *,
                 must_explore: bool = False) -> dict:
    """Validate one typed reply. Returns {"schema", "choice", "kind", "premise_superseded",
    "text": {...}}; refuses unknown ids, free text on `exhausted`, a missing row field, or a
    non-explore choice when the reserve is due."""
    body = json.loads(reply) if isinstance(reply, str) else dict(reply)
    if not isinstance(body, Mapping):
        raise TypedPlannerRefused("reply is not a json object")
    by_id = {o["option"]: o for o in options}
    choice = body.get("choice")
    if choice not in by_id:
        raise TypedPlannerRefused(f"choice {choice!r} is not an offered option")
    if must_explore and choice != EXPLORE:
        raise TypedPlannerRefused("reserved explore share is due; only explore is admissible")
    knob = by_id[choice].get("kind") == SERVING_KNOB
    fields = SERVING_KNOB_TEXT_FIELDS if knob else CHOSEN_TEXT_FIELDS
    text = {k: body.get(k) for k in fields if body.get(k) is not None}
    if choice == EXHAUSTED:
        if text:
            raise TypedPlannerRefused("exhausted carries no proposal free text")
        reason = str(body.get("reason") or "").strip()
        if not reason:
            raise TypedPlannerRefused("exhausted must name the spent rows")
        text = {"reason": reason}
    else:
        missing = [k for k in fields if not str(body.get(k) or "").strip()]
        if missing:
            raise TypedPlannerRefused(f"chosen option lacks {missing}")
    out = {"schema": SCHEMA, "choice": choice, "kind": by_id[choice]["kind"],
           "premise_superseded": bool(by_id[choice].get("premise_superseded")), "text": text}
    if knob:
        out["serving_knob"] = dict(by_id[choice]["serving_knob"])
    return out


def candidate_from_choice(choice: Mapping[str, Any], *, base: Any = None, recipe_argv: Sequence[str] = (),
                          base_recipe_hash: str | None = None, champion_sha: str | None = None) -> dict | None:
    """Where a validated planner choice becomes a candidate. A `serving-knob` choice is routed to
    `serving_knobs.emit_candidate` (the delta rendered over `recipe_argv` and the registry `base`); it needs `base`
    and `base_recipe_hash`, and a missing one is refused, never defaulted. Any other kind (row / control) has no
    knob candidate: returns None and the caller keeps its existing free-text path."""
    if choice.get("kind") != SERVING_KNOB:
        return None
    if base is None or not base_recipe_hash:
        raise TypedPlannerRefused("a serving-knob choice needs the registry base and the base recipe hash")
    from . import serving_knobs
    try:
        return serving_knobs.emit_candidate(choice, base, recipe_argv, base_recipe_hash=base_recipe_hash, champion_sha=champion_sha)
    except serving_knobs.KnobError as exc:
        raise TypedPlannerRefused(f"serving-knob choice {choice.get('choice')!r}: {exc}") from exc


def replay_yield(calls: Sequence[Mapping[str, Any]], keeps: Sequence[Mapping[str, Any]], *,
                 choice_wall_min: float = 5.0, explore_reserve: float = EXPLORE_RESERVE,
                 conditions: tuple[bool, bool, bool] = (True, True, True)) -> dict:
    """Offline re-validation on the stored replay table (RESULT.json `table` + `keeps_extension`).

    keep-candidate yield per planner hour, free text vs typed. Free text = recorded wall.
    Typed = mapped calls at `choice_wall_min` ([inferred] DESIGN cost), NONE proposals and
    sunk/invalid calls at their recorded wall (never credited). Keeps are counted reachable
    under typed only through an option the conditions provide: EXACT/FAMILY always, FAMILY-DNR
    only with (c), NONE only with (b) (the explore path). Abstentions map to a Choice only
    with (a). Every number is labelled; a 0/0 yield is reported as None (undetermined).
    """
    a, b, c = conditions
    free_h = sum(float(x["wall_min"]) for x in calls) / 60.0
    typed_min = 0.0
    for x in calls:
        cat = x.get("category")
        mapped = cat in ("EXACT", "FAMILY") or (a and cat == "ABSTAIN-MAPPED")
        typed_min += choice_wall_min if mapped else float(x["wall_min"])
    if b:  # the reserved explore share pays full exploration wall on that share of calls
        mapped_n = sum(1 for x in calls if x.get("category") in ("EXACT", "FAMILY", "ABSTAIN-MAPPED"))
        mean_free = (free_h * 60.0 / len(calls)) if calls else 0.0
        typed_min += explore_reserve * mapped_n * max(mean_free - choice_wall_min, 0.0)
    typed_h = typed_min / 60.0
    def _statuses(x):
        out = set()
        for item in x.get("downstream") or ():
            out.update(item if isinstance(item, (list, tuple)) else (item,))
        return {str(v) for v in out}
    window_keeps = sum(1 for x in calls if _statuses(x) & {"kept", "keep_candidate"})
    uniq: dict[str, str] = {}
    for k in keeps:
        uniq.setdefault(k["mechanism_id"], k["category"])
    reach = sum(1 for cat in uniq.values()
                if cat in ("EXACT", "FAMILY") or (c and cat == "FAMILY-DNR") or (b and cat == "NONE"))
    return {"schema": SCHEMA + ".replay", "conditions": {"a_exhausted": a, "b_explore_reserve": b,
                                                         "c_dnr_selectable": c},
            "calls": len(calls), "free_text_hours": round(free_h, 3),
            "typed_hours_inferred": round(typed_h, 3),
            "window_keep_candidates": window_keeps,
            "yield_per_hour_free": (window_keeps / free_h) if window_keeps and free_h else None,
            "yield_per_hour_typed": (window_keeps / typed_h) if window_keeps and typed_h else None,
            "keeps_reachable": reach, "keeps_unique": len(uniq),
            "reachable_fraction": (reach / len(uniq)) if uniq else None}


__all__ = ["SERVING_KNOB", "CONTROLS", "EXHAUSTED", "EXPLORE", "EXPLORE_RESERVE", "SCHEMA", "TypedPlannerRefused",
           "build_options", "candidate_from_choice", "explore_required", "parse_choice", "render_choice_prompt",
           "replay_yield"]
