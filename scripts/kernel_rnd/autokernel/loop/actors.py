#!/usr/bin/env python3
"""Real planner and critic, driven by an external coding agent.

The loop itself never shells out -- it takes `Planner` and `Critic` protocols. This
module is the concrete implementation, kept separate so the control flow stays
testable without an API key.

TWO LESSONS ARE BUILT IN.

**Backoff.** A codex 401 on 2026-08-26 produced 284 failures in 23 minutes because
the transient path retried with zero delay. Consecutive provider failures back off
exponentially and the streak is surfaced, not swallowed.

**The prompt is the product.** The old planner was a pure function of a context
bundle that was empty: no refusal reasons, no memory, no profile. Everything this
assembles is something the loop measured and previously discarded -- the hotspot
table `rocprofv3` produced on every attempt, the refusals it filtered on the wrong
status string, and the history every crash reset to zero.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

from .loop import ActorTransient, Hypothesis, Review

CODEX = "/usr/local/share/npm-global/bin/codex"
DEFAULT_TIMEOUT_S = 1800
#: 30s -> 1800s. The streak is what the operator needs to see, not each retry.
BACKOFF_S = (30, 120, 480, 1800)


class ProviderTransient(ActorTransient):
    """The actor provider failed in a way that is worth retrying.

    Subclasses the loop's own transient type so `iterate` ends the ITERATION rather
    than the run, without this module and the loop importing each other.
    """


def _run_agent(prompt: str, *, workspace: Path, timeout_s: int = DEFAULT_TIMEOUT_S,
               binary: str = CODEX) -> str:
    argv = [binary, "exec", "--skip-git-repo-check", "-C", str(workspace), prompt]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        # A hung container held a turn forever in v27; a bounded invocation is a
        # transient, not a terminal fault.
        raise ProviderTransient(f"actor exceeded {timeout_s}s") from exc
    if done.returncode != 0:
        raise ProviderTransient(
            f"actor exited {done.returncode}: {done.stderr[-400:]}")
    return done.stdout


def _with_backoff(call, *, attempts: int = len(BACKOFF_S),
                  sleep=time.sleep) -> tuple[Any, int]:
    """Retry a provider call, backing off. Returns (result, transient_streak)."""
    streak = 0
    last: Exception | None = None
    for index in range(attempts):
        try:
            return call(), streak
        except ProviderTransient as exc:
            last = exc
            streak += 1
            if index < attempts - 1:
                sleep(BACKOFF_S[min(index, len(BACKOFF_S) - 1)])
    raise ProviderTransient(
        f"actor failed {streak} consecutive times; last: {last}") from last


#: Phrases that only ever appear in OUR prompt template, never in an answer.
#: Deliberately specific: the first version of this guard listed a bare `"<"`, which
#: rejected every legitimate falsifier that said `delta < 0.97%` and every statement
#: that named `mul_mat_vec_q<Q4_K>`. It retired three consecutive hypotheses the
#: planner had answered correctly -- a guard that forbids its own compliant idiom,
#: which is the exact failure class this rebuild exists to remove.
_TEMPLATE_PHRASES = ("path you changed", "short-slug", "your path here",
                     "the function you will change")


def _is_placeholder(value: Any) -> bool:
    """True only for our own template echoed back, never for a real answer.

    Three signals, all of which a genuine reply avoids and an echo cannot:

      * the value is ENTIRELY an angle-bracket span (`<the function you will
        change>`) -- an echoed slot is the whole field, whereas a C++ template or a
        `<` comparison always sits inside surrounding prose;
      * it carries a phrase that exists only in the prompt;
      * it opens with `e.g.`, which introduces the prompt's illustration.
    """
    text = str(value).strip()
    if text.startswith("<") and text.endswith(">"):
        return True
    lowered = text.lower()
    return (lowered.startswith("e.g.")
            or any(phrase in lowered for phrase in _TEMPLATE_PHRASES))


def _extract_json(text: str) -> dict:
    """Pull the last JSON object out of an agent's stdout."""
    depth = 0
    start = None
    best = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:index + 1]
                try:
                    best = json.loads(candidate)
                except json.JSONDecodeError:
                    pass
    if best is None:
        raise ProviderTransient("actor produced no parseable JSON object")
    return best


def render_context(context: Mapping[str, Any], *, limit: int = 12) -> str:
    """The bundle, as the actor sees it. Everything here was previously discarded."""
    lines: list[str] = []

    # First, because it is the cheapest rejection: standing constraints and the
    # settled list. `program.md` carried "Already in v9: GGML_IQK, MMQ, HIP graphs"
    # for the whole of run 6 while the planner proposed exactly those and the critic
    # rejected all nine iterations for it -- a document nobody was wired to read.
    program = (context.get("program") or "").strip()
    if program:
        lines.append("## Standing constraints and settled questions (read this first)")
        lines.append(program)
        lines.append("")

    hotspots = context.get("kernel_hotspots") or []
    lines.append("## Where the device time actually goes (rocprofv3, current champion)")
    if hotspots:
        lines.append("| share | ns | calls | kernel |")
        lines.append("|---|---|---|---|")
        for row in list(hotspots)[:limit]:
            share = row.get("share_of_device_time") or row.get(
                "anchor_share_of_device_time") or 0.0
            lines.append(f"| {share * 100:.2f}% | {row.get('total_duration_ns')} | "
                         f"{row.get('calls')} | `{row.get('signature')}` |")
        lines.append("\nA mechanism aimed at a route with negligible share cannot move "
                     "the target runtime no matter how correct it is.")
    else:
        lines.append("(no profile yet — say so rather than guessing a target)")

    prior = context.get("prior_experiments") or []
    lines.append("\n## Already tried")
    if prior:
        for row in list(prior)[:limit]:
            stale = " [STALE EPOCH — the fact it was tried is usable, the NUMBER is not]" \
                if row.get("stale_epoch") else ""
            effect = row.get("effect_fraction")
            measured = f"{effect * 100:+.3f}%" if isinstance(effect, (int, float)) else "—"
            lines.append(f"- `{row.get('mechanism_id')}` → {row.get('status')} "
                         f"{measured}{stale}"
                         + (f"\n    refused: {row['refusal_reason']}"
                            if row.get("refusal_reason") else ""))
    else:
        lines.append("(nothing yet)")

    for label, key in (("Your hypothesis was rejected", "prior_hypothesis_rejections"),
                       ("Your patch was rejected", "prior_patch_rejections")):
        reasons = context.get(key) or []
        if reasons:
            lines.append(f"\n## {label} — answer these, do not re-derive")
            lines.extend(f"- {reason}" for reason in reasons)

    inbox = context.get("inbox") or []
    if inbox:
        lines.append("\n## Operator suggestions (async; use if relevant)")
        lines.extend(f"- {item}" for item in inbox)
    return "\n".join(lines)


_HYPOTHESIS_TASK = """You are proposing ONE kernel optimisation for llama.cpp on an \
AMD MI210 (gfx90a, ROCm 6.2).

{context}

Propose exactly one hypothesis. Reply with ONE json object and nothing else:
{{"mechanism_id": "akm-<short-slug>",
  "statement": "<what changes, mechanically, and why it should be faster>",
  "falsifier": "<the measurement that would prove this wrong>",
  "target_surface": "<one path under ggml/src/ggml-cuda/>",
  "target_symbol": "<the function you will change>"}}

Rules: attack a route near the top of the profile; name a MECHANISM, not a wish; \
state a falsifier that could actually fail."""


@dataclass
class CodexPlanner:
    """Proposes and authors through an external coding agent."""

    workspace: Path
    binary: str = CODEX
    timeout_s: int = DEFAULT_TIMEOUT_S
    transient_streak: int = 0

    def propose(self, context: Mapping[str, Any]) -> Hypothesis:
        prompt = _HYPOTHESIS_TASK.format(context=render_context(context))
        raw, streak = _with_backoff(
            lambda: _run_agent(prompt, workspace=self.workspace,
                               timeout_s=self.timeout_s, binary=self.binary))
        self.transient_streak = streak
        body = _extract_json(raw)
        missing = {"mechanism_id", "statement", "falsifier", "target_surface",
                   "target_symbol"} - set(body)
        if missing:
            raise ProviderTransient(f"hypothesis is missing {sorted(missing)}")
        echoed = sorted(key for key in body if _is_placeholder(body[key]))
        if echoed:
            raise ProviderTransient(
                f"hypothesis echoed the prompt template for {echoed}")
        return Hypothesis(
            mechanism_id=str(body["mechanism_id"]), statement=str(body["statement"]),
            falsifier=str(body["falsifier"]),
            target_surface=str(body["target_surface"]),
            target_symbol=str(body["target_symbol"]))

    def author(self, hypothesis: Hypothesis,
               context: Mapping[str, Any]) -> tuple[str, ...]:
        prompt = (
            f"Implement this hypothesis in the worktree at {self.workspace}.\n\n"
            f"mechanism: {hypothesis.mechanism_id}\n"
            f"statement: {hypothesis.statement}\n"
            f"file:      {hypothesis.target_surface}\n"
            f"symbol:    {hypothesis.target_symbol}\n\n"
            f"{render_context(context)}\n\n"
            "Edit the file directly. Keep the change minimal and confined to the "
            "named file.\n\n"
            "DO NOT BUILD, COMPILE, BENCHMARK OR TEST. The loop owns the build and "
            "the GPU; a build you start is unmeasured compute taken from another "
            "session and it will not be used. Make the edit and stop.\n\n"
            "Then reply with ONE json object naming the files you actually changed, "
            "using their real paths:\n"
            '{"paths": ["ggml/src/ggml-cuda/<file>"]}')
        raw, streak = _with_backoff(
            lambda: _run_agent(prompt, workspace=self.workspace,
                               timeout_s=self.timeout_s, binary=self.binary))
        self.transient_streak = streak
        paths = _extract_json(raw).get("paths")
        if not isinstance(paths, list) or not paths:
            raise ProviderTransient("authoring returned no changed paths")
        if any(_is_placeholder(item) for item in paths):
            raise ProviderTransient(
                f"authoring echoed the prompt template instead of answering: {paths}")
        # The ground truth is the worktree, not the reply. An actor that says it
        # changed a file and did not is the failure mode a self-reported path cannot
        # catch.
        dirty = subprocess.run(
            ["git", "-C", str(self.workspace), "status", "--porcelain", "--", *paths],
            capture_output=True, text=True, timeout=300).stdout.strip()
        if not dirty:
            raise ProviderTransient(
                f"authoring reported {paths} but the worktree is unchanged there")
        return tuple(str(item) for item in paths)


_REVIEW_TASK = """{subject}

{context}

Reply with ONE json object and nothing else:
{{"accepted": true|false, "reason": "<required when accepted is false>"}}

Reject when: {grounds}"""


@dataclass
class CodexCritic:
    """Two passes: the hypothesis before any patch, the diff before the build."""

    workspace: Path
    binary: str = CODEX
    timeout_s: int = DEFAULT_TIMEOUT_S

    def _review(self, subject: str, grounds: str,
                context: Mapping[str, Any]) -> Review:
        prompt = _REVIEW_TASK.format(subject=subject, grounds=grounds,
                                     context=render_context(context))
        raw, _ = _with_backoff(
            lambda: _run_agent(prompt, workspace=self.workspace,
                               timeout_s=self.timeout_s, binary=self.binary))
        body = _extract_json(raw)
        accepted = bool(body.get("accepted"))
        reason = str(body.get("reason") or "")
        if not accepted and not reason.strip():
            # The loop refuses a reasonless rejection at construction; make the
            # provider's omission explicit rather than crashing on it.
            reason = "critic rejected without stating a reason"
        return Review(accepted=accepted, reason=reason)

    def review_hypothesis(self, hypothesis: Hypothesis,
                          context: Mapping[str, Any]) -> Review:
        return self._review(
            f"Review this HYPOTHESIS before any patch is written:\n"
            f"  mechanism: {hypothesis.mechanism_id}\n"
            f"  statement: {hypothesis.statement}\n"
            f"  falsifier: {hypothesis.falsifier}\n"
            f"  target:    {hypothesis.target_surface}::{hypothesis.target_symbol}",
            "it was already measured; the mechanism is unsupported by the profile; "
            "there is no real falsifier; the target has negligible device-time share; "
            "or it is already present in production v9",
            context)

    def review_patch(self, hypothesis: Hypothesis, paths: Sequence[str],
                     context: Mapping[str, Any]) -> Review:
        diff = subprocess.run(["git", "-C", str(self.workspace), "diff", "--", *paths],
                              capture_output=True, text=True, timeout=300).stdout
        return self._review(
            f"Review this DIFF before it is built. It should implement "
            f"{hypothesis.mechanism_id}: {hypothesis.statement}\n\n"
            f"```diff\n{diff[:20000]}\n```",
            "it does not implement the accepted mechanism; it creeps beyond "
            f"{list(paths)}; it risks correctness; or it edits a file that must stay "
            "byte-identical to production",
            context)


__all__ = ["BACKOFF_S", "CODEX", "CodexCritic", "CodexPlanner", "ProviderTransient",
           "render_context"]
