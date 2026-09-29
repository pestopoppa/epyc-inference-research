#!/usr/bin/env python3
"""Per-lane actor models: lane k's planner and author on another model (2026-09-29).

WHY. Operator decision 2026-09-29, "2nd lane on external model". DS41 runs one lane:
the local 27B (`qwen-gpu/qwen3.8-27b`, :8083) plans and authors best-of-2, and the CPU
sits ~idle through every actor phase. A second lane on the SAME 27B would overflow
:8083's single 196,608-token unified KV pool (planner sessions reach 100-170k tokens;
a full pool with MTP has crashed that server). So lane 1 runs its planner and its
single author on an external opencode model (`deepseek/deepseek-flash`, already the
critic; its prompt egress off-host is accepted). Both lanes share the one serialized
build+measure tail (`pipeline.SerializedTail`), so CPU measurements never overlap.

`--lane-actor-models 1=deepseek/deepseek-flash[@effort],...`:

* lane 0 is never overridden (refused): its planner, best-of panel and seats are the
  global ones, byte for byte;
* an overridden lane's planner AND its author (`loop.iterate` authors through
  `planner.author`) use that model; effort defaults to `--planner-effort`;
* the critic stays global for every lane;
* an overridden lane authors SINGLE (no best-of panel), and its seat sends no
  qwen-gpu-template reasoning kwargs (`author_thinking="default"`: the author's
  `chat_template_kwargs` rides the served llama.cpp template only), and its planner keeps
  its reasoning history (`planner_reasoning_history="keep"`: the drop renames a message
  key only llama-server ignores; DeepSeek's API wants `reasoning_content` back inside a
  tool chain). Context/output
  limits and the planner/author wall budgets stay as the run's (the critic already runs
  deepseek with those same limits), and the planner salvage turn stays on: it continues
  the call's own opencode session, which is provider-agnostic.

Actor configuration is provenance, never identity: nothing here enters the launch or
measurement epoch (`epoch_aliases.launch_epoch_inputs` takes no argv), and the flag is
excluded from the serial continuation binding (`serial_run.POOL_ACTOR_FLAGS`).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Mapping

FLAG = "--lane-actor-models"
_LANE_NAME = re.compile(r"lane(\d+)")


@dataclass(frozen=True)
class LaneActor:
    """Lane `lane`'s planner/author model (an opencode `provider/model`) and effort."""
    lane: int
    model: str
    effort: str | None = None      # None: the run's --planner-effort

    def effort_or(self, default: str) -> str:
        return self.effort if self.effort else default

    def to_dict(self, default_effort: str | None = None) -> dict[str, Any]:
        return {"model": self.model,
                "effort": self.effort if self.effort else default_effort}


def parse(spec: str | None, *, workers: int) -> dict[int, LaneActor]:
    """`K=provider/model[@effort]` entries, comma-separated; {} for none. ValueError on
    anything that cannot run as written (an explicit knob never silently degrades)."""
    text = (spec or "").strip()
    if not text:
        return {}
    lanes: dict[int, LaneActor] = {}
    for raw in text.split(","):
        entry = raw.strip()
        index, equals, value = entry.partition("=")
        if not equals or not index.strip().isdigit() or not value.strip():
            raise ValueError(f"{FLAG} entry {entry!r} is not K=provider/model[@effort]")
        lane = int(index.strip())
        model, _at, effort = value.strip().partition("@")
        model, effort = model.strip(), effort.strip()
        if lane == 0:
            raise ValueError(f"{FLAG}: lane 0 keeps the run's --planner-model and panel "
                             "(it is never overridden)")
        if lane >= int(workers):
            raise ValueError(f"{FLAG}: lane {lane} does not exist with --workers {workers}")
        if lane in lanes:
            raise ValueError(f"{FLAG}: lane {lane} is named twice")
        if ("/" not in model or model.startswith(("orch:", "claude-"))
                or any(char.isspace() for char in model) or model.endswith("/")):
            raise ValueError(f"{FLAG}: lane {lane} model {model!r} must be an opencode "
                             "provider/model id")
        if _at and not effort:
            raise ValueError(f"{FLAG}: lane {lane} names an empty effort")
        lanes[lane] = LaneActor(lane, model, effort or None)
    return lanes


def lane_index(worker: Any) -> int | None:
    """The pool index of a `pipeline.Worker` (`pool.provision` names lanes `lane<i>`)."""
    match = _LANE_NAME.fullmatch(str(getattr(worker, "name", "")))
    return int(match.group(1)) if match else None


def for_worker(lanes: Mapping[int, LaneActor], worker: Any) -> LaneActor | None:
    index = lane_index(worker)
    return lanes.get(index) if index is not None else None


def provider(model: str | None) -> str | None:
    return model.split("/", 1)[0] if model and "/" in model else None


def shared_pool_lanes(workers: int, lanes: Mapping[int, LaneActor],
                      planner_model: str | None) -> int:
    """How many lanes' planner/author calls land on the global planner's server.

    An overridden lane on ANOTHER provider (deepseek vs qwen-gpu) does not share
    :8083's pool; one on the same provider does. With no overrides this is `workers`."""
    home = provider(planner_model)
    return sum(1 for index in range(int(workers))
               if index not in lanes or provider(lanes[index].model) == home)


def seat_for(lane: LaneActor | None, seat):
    """The lane's planner/author seat: the global seat unchanged (the same object) for
    a lane without an override; else no qwen-gpu reasoning kwargs on its author and no
    llama-server-only reasoning-history drop on its planner."""
    if lane is None or seat is None:
        return seat
    return replace(seat, author_thinking="default", planner_reasoning_history="keep")


def provenance(lanes: Mapping[int, LaneActor], default_effort: str | None) -> dict[str, Any]:
    """`actor_config` provenance (never identity): {} when no lane is overridden."""
    if not lanes:
        return {}
    return {"lane_actor_models": {str(index): lanes[index].to_dict(default_effort)
                                  for index in sorted(lanes)}}


__all__ = ["FLAG", "LaneActor", "for_worker", "lane_index", "parse", "provenance",
           "provider", "seat_for", "shared_pool_lanes"]
