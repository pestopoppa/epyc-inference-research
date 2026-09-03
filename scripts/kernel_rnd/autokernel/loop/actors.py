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

**Two backends, chosen per role (2026-09-03).** The planner runs Claude Fable 5.1 at
medium effort through the `claude` CLI; the critic stays on `gpt-5.6-sol` at high
through `codex exec`. Each is an external coding agent invoked headless in the lane's
detached worktree. Note for the Claude backend: the worktree carries the llama-tree
freeze overlay `CLAUDE.md`, which scopes its never-edit rule to the
`production-consolidated-*` branch -- measured 2026-09-03, Fable authors correctly in a
detached lane with that overlay loaded, and `--bare` (which would skip it) is not an
option because it refuses OAuth and this host has no API key. The sandbox note
appended to the system prompt makes the scoping explicit rather than inferred.
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
CLAUDE = "/home/node/.local/bin/claude"
DEFAULT_TIMEOUT_S = 1800
#: 30s -> 1800s. The streak is what the operator needs to see, not each retry.
BACKOFF_S = (30, 120, 480, 1800)

#: Appended to the Claude backend's system prompt. The lane worktree ships the
#: production freeze overlay; this states the scoping that overlay itself declares.
_CLAUDE_SANDBOX_NOTE = (
    "You are running headless as the AutoKernel planner inside a DETACHED git worktree "
    "of the champion kernel tree. This worktree exists to be edited: it is not the "
    "frozen production-consolidated branch, and the freeze rule in this tree's "
    "CLAUDE.md applies to that branch, not to this sandbox. Make the requested edits "
    "directly. Never build, compile, benchmark or test -- the loop owns the build and "
    "the GPU. Reply exactly as instructed.")


@dataclass(frozen=True)
class Backend:
    """One external coding agent: which binary, which model, how much reasoning.

    `argv` is the whole contract -- everything else in this module is backend-blind
    and only ever sees stdout. Keep the prompt LAST for both CLIs.
    """
    kind: str       # "codex" | "claude"
    model: str
    effort: str
    binary: str

    def argv(self, prompt: str, workspace: Path) -> list[str]:
        if self.kind == "codex":
            # `-c` takes TOML: the value must be quoted or codex rejects it.
            return [self.binary, "exec", "--skip-git-repo-check",
                    "-m", self.model, "-c", f'model_reasoning_effort="{self.effort}"',
                    "-C", str(workspace), prompt]
        if self.kind == "claude":
            return [self.binary, "-p", "--dangerously-skip-permissions",
                    "--no-session-persistence", "--output-format", "text",
                    "--model", self.model, "--effort", self.effort,
                    "--append-system-prompt", _CLAUDE_SANDBOX_NOTE, prompt]
        raise ValueError(f"unknown backend kind {self.kind!r}")

    def describe(self) -> str:
        return f"{self.kind}:{self.model}@{self.effort}"


def backend_for(model: str, effort: str) -> Backend:
    """`claude-*` models go through the claude CLI; everything else through codex."""
    if model.startswith("claude-"):
        return Backend("claude", model, effort, CLAUDE)
    return Backend("codex", model, effort, CODEX)


#: Operator choice, 2026-09-03: planner Fable 5.1 at medium, critic gpt-5.6-sol at high.
PLANNER_DEFAULT = backend_for("claude-fable-5-1", "medium")
CRITIC_DEFAULT = backend_for("gpt-5.6-sol", "high")


class ProviderTransient(ActorTransient):
    """The actor provider failed in a way that is worth retrying.

    Subclasses the loop's own transient type so `iterate` ends the ITERATION rather
    than the run, without this module and the loop importing each other.
    """


def _run_agent(prompt: str, *, workspace: Path, timeout_s: int = DEFAULT_TIMEOUT_S,
               backend: Backend = CRITIC_DEFAULT) -> str:
    argv = backend.argv(prompt, workspace)
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s,
                              cwd=str(workspace))
    except subprocess.TimeoutExpired as exc:
        # A hung container held a turn forever in v27; a bounded invocation is a
        # transient, not a terminal fault.
        raise ProviderTransient(f"actor exceeded {timeout_s}s") from exc
    if done.returncode != 0:
        # Both tails. `claude -p` reports its own errors ("Not logged in", usage
        # limits, refusals) on STDOUT with a non-zero exit and an EMPTY stderr --
        # run 27 logged 74 transients reading "actor exited 1: " and nothing else,
        # because this path used to throw the only channel that carried the reason.
        raise ProviderTransient(
            f"actor exited {done.returncode} [{backend.describe()}]: "
            f"stderr={done.stderr[-300:]!r} stdout={done.stdout[-300:]!r}")
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

    # Before anything else the planner is asked to look at candidates that were
    # FORMED AND NEVER MEASURED. A lane authors against champion C0; another lane's
    # keep advances it to C1; the first candidate is refused as superseded. That work
    # is only wasted if nobody looks at it again -- the hypothesis, its falsifier and
    # the champion it was formed against are all recorded, so it can be re-proposed
    # against the champion that displaced it. Cheaper than deriving a new one, and it
    # is a QUEUE, not a graveyard.
    superseded = [row for row in (context.get("prior_experiments") or [])
                  if row.get("status") == "superseded"][:limit]
    if superseded:
        lines.append("## Formed but never measured — consider these FIRST")
        lines.append("Each was authored against a champion that moved before it could "
                     "be measured. None was refuted. Re-propose one if it still "
                     "addresses the current profile; say why if it does not.")
        for row in superseded:
            lines.append(f"- `{row.get('mechanism_id')}` — {row.get('statement') or ''}"
                         + (f"\n    falsifier: {row['falsifier']}"
                            if row.get("falsifier") else "")
                         + (f"\n    {row['refusal_reason']}"
                            if row.get("refusal_reason") else ""))
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

    # Mechanisms already CHARACTERISED by repeated measurement. Run 15 spent 9 of its
    # 10 measurements re-sampling two unchanged patches: a near-floor result reads as
    # "almost", so the planner re-proposed it. But re-measuring unchanged code adds no
    # information -- it redraws the same noise. Pooling says what is actually known,
    # and a characterised mechanism is FINISHED unless the code changes.
    #
    # `comparable_measurement` is load bearing here and was MISSING. This pooled every
    # row carrying a number -- cross-epoch ones included -- into one median printed
    # under a heading that tells the planner not to re-measure. Against the live store,
    # through the first ~20 rows of epoch `6a4dccec`, it read "`akm-q4k-q8-sum-sidecar`:
    # measured 4x, median -8.814%" with all four magnitudes taken against a DIFFERENT
    # anchor and build: a cross-epoch magnitude deciding a mechanism was finished. That
    # is what `P-AK-SEARCH-1` denial 4 forbade and what `-A3` clause 2 still forbids --
    # A3 moved the ORDERING question only, never the comparability one. It bit at every
    # epoch transition, because a new epoch's first recall window is the old epoch's
    # tail. The marker was right there on the row and the loop read the number instead,
    # which is why `experiments.rank()` now deletes the number as well as marking it.
    #
    # BOTH markers, and a row carrying NEITHER is still pooled. `recall()` always writes
    # both -- `test_ranking.py` asserts that, which is what makes this the whole real
    # path -- so the two spellings are complements there and the only rows this default
    # reaches are hand-built ones with no provenance to judge. Requiring a positive
    # `comparable_measurement` instead was the first version and it silently switched
    # the block off for every synthetic context, `test_seed.py`'s five-sample run-15
    # regression included: a conformance fix that disables the feature it is protecting.
    repeats: dict[str, list[float]] = {}
    for row in prior:
        effect = row.get("effect_fraction")
        if (row.get("mechanism_id") and isinstance(effect, (int, float))
                and not row.get("stale_epoch")
                and row.get("comparable_measurement", True)):
            repeats.setdefault(row["mechanism_id"], []).append(effect * 100.0)
    characterised = {k: v for k, v in repeats.items() if len(v) >= 3}
    if characterised:
        lines.append("## Characterised — do NOT re-measure these")
        lines.append("Each was measured repeatedly on UNCHANGED code. Re-running one "
                     "redraws the same noise and tells you nothing new. Change the "
                     "mechanism or pick a different target.")
        for mechanism, values in characterised.items():
            values = sorted(values)
            median = values[len(values) // 2]
            positive = sum(1 for v in values if v > 0)
            lines.append(
                f"- `{mechanism}`: measured {len(values)}x, median {median:+.3f}%, "
                f"{positive}/{len(values)} positive "
                f"[{', '.join(f'{v:+.2f}' for v in values)}]")
        lines.append("")

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
class AgentPlanner:
    """Proposes and authors through an external coding agent (default: Fable 5.1)."""

    workspace: Path
    backend: Backend = PLANNER_DEFAULT
    timeout_s: int = DEFAULT_TIMEOUT_S
    transient_streak: int = 0

    def propose(self, context: Mapping[str, Any]) -> Hypothesis:
        prompt = _HYPOTHESIS_TASK.format(context=render_context(context))
        raw, streak = _with_backoff(
            lambda: _run_agent(prompt, workspace=self.workspace,
                               timeout_s=self.timeout_s, backend=self.backend))
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
                               timeout_s=self.timeout_s, backend=self.backend))
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
class AgentCritic:
    """Two passes: the hypothesis before any patch, the diff before the build
    (default: gpt-5.6-sol at high)."""

    workspace: Path
    backend: Backend = CRITIC_DEFAULT
    timeout_s: int = DEFAULT_TIMEOUT_S

    def _review(self, subject: str, grounds: str,
                context: Mapping[str, Any]) -> Review:
        prompt = _REVIEW_TASK.format(subject=subject, grounds=grounds,
                                     context=render_context(context))
        raw, _ = _with_backoff(
            lambda: _run_agent(prompt, workspace=self.workspace,
                               timeout_s=self.timeout_s, backend=self.backend))
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


__all__ = ["BACKOFF_S", "Backend", "CLAUDE", "CODEX", "CRITIC_DEFAULT",
           "PLANNER_DEFAULT", "AgentCritic", "AgentPlanner", "ProviderTransient",
           "backend_for", "render_context"]
