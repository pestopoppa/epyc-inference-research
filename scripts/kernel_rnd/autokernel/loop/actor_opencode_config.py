"""Per-run opencode config for the autokernel planner/author seat.

The first 27B-on-GPU proposal made 64 serial tool calls, read whole files, dumped
unbounded perf output, overflowed its 98,304-token slot in 26 minutes, compacted, and
took ~39 minutes per proposal without ever using a subagent. This module writes the
config that fixes the shape of that session, not the model: bounded MCP tools
(`actor_tools_mcp`), a global cap on built-in tool output, a step cap, a tool-discipline
system prompt, and (as an A/B knob) fan-out through the `task` tool.

The caller passes the file as ``OPENCODE_CONFIG`` and selects the agent with
``opencode run --agent <AGENT_NAMES[role]>``. opencode (1.18.31) loads the global
``~/.config/opencode/opencode.jsonc`` FIRST and merges ``OPENCODE_CONFIG`` on top, so
the global providers (``qwen-gpu``), the bash deny-list and the filesystem-containment
plugin all survive. This file deliberately sets no top-level ``permission``, no
``plugin`` list and no per-agent ``bash`` rule: an agent's permission is appended AFTER
the global rules and the last match wins, so an agent-level ``bash: allow`` would
silently re-allow every globally denied verb. (OAB-10/11, opt-in: the trim and
lane-guard knobs add a top-level ``permission`` block that holds only denies, plus
``external_directory`` allows that re-open just the anchor build dirs its own deny
closed; see ``seat_permission``. The PLAIN seat gets the same block through
``build_plain_config``.) Every config either builder returns also sets
``snapshot: false`` (``SNAPSHOT_OFF``), knobs on or off, bounded or plain.

Key names are verified against the opencode 1.18.31 binary's config schema
(``ConfigV1.Info`` / ``AgentConfig`` / ``McpLocalConfig`` / ``PermissionConfig``):

* ``tool_output.{max_lines,max_bytes}`` -- global truncation thresholds (defaults
  2000 lines / 51200 bytes); over either, the full text goes to disk and a preview
  is returned.
* ``agent.<name>.{description,mode,steps,prompt,permission}`` -- ``tools`` is
  deprecated in favour of ``permission`` (the decoder converts it), and ``maxSteps``
  in favour of ``steps``. A set ``prompt`` REPLACES opencode's provider system prompt
  rather than appending to it, so the prompt here is self-contained.
* ``mcp.<name>.{type,command,cwd,environment,enabled,timeout}`` -- ``cwd`` is
  supported (relative paths resolve from the workspace); ``environment`` is merged
  over the parent environment; ``timeout`` is per-request ms (default 5000).
* ``permission.task`` patterns match the ``subagent_type``; MCP tools are named
  ``<server>_<tool>`` with non ``[A-Za-z0-9_-]`` characters replaced by ``_``.
* ``compaction.prune`` -- prune old tool outputs (default false).
* ``snapshot`` -- top-level boolean, SINGULAR (``ConfigV1.Info``: "Enable or disable
  snapshot tracking ... Defaults to true"; the v2 translation reads ``t.snapshot`` into
  ``snapshots``, and ``Snapshot`` gates on ``config.snapshot !== false``). The docs'
  plural spelling is the v2 INTERNAL name and is not a v1 config key. See
  ``SNAPSHOT_OFF``.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

#: role -> the agent name the caller passes as ``opencode run --agent``.
AGENT_NAMES = {"planner": "autokernel-planner", "author": "autokernel-author"}

#: The fan-out subagent (``task`` tool ``subagent_type``); defined only when fan_out.
SCOUT_AGENT = "autokernel-scout"

#: MCP server key; its tools surface as ``autokernel-tools_<tool>``.
MCP_SERVER = "autokernel-tools"
MCP_MODULE = "scripts.kernel_rnd.autokernel.loop.actor_tools_mcp"
MCP_TOOLS = ("read_range", "grep", "outline", "code_search", "profile_top",
             "symbol_annotate")

#: The GPU llama-server runs -np 2: more concurrent subagents than this just queue.
MAX_CONCURRENT_SUBAGENTS = 2

#: Per-request MCP timeout. The 5 s default is too short for profile_top on a large
#: perf.data; the tool server caps its own output, not its wall time.
MCP_TIMEOUT_MS = 120_000

#: <repo>/scripts/kernel_rnd/autokernel/loop/this_file.py -> <repo>
RESEARCH_ROOT = Path(__file__).resolve().parents[4]

#: Every per-call config this module writes turns opencode's snapshot tracking off
#: (operator-approved 2026-09-25). With it on, opencode `git write-tree`s the lane into
#: `~/.local/share/opencode/snapshot/<project>/` on every step and stores per-step
#: snapshot diffs in opencode.db -- the growth behind the 10.8 GB store that the
#: reaper's 30-minute VACUUM keeps rewriting, and a VACUUM window is exactly what
#: killed DS41 run 9c's author call (07:59:47Z, SQLITE_BUSY behind "Unexpected error").
#: The actor never uses undo/revert: the loop owns the worktree and resets it itself.
#: Independent of every OAB knob, so the knobs-off plain seat stops bloating too.
SNAPSHOT_OFF = {"snapshot": False}

# --------------------------------------------------------------------------------------
# OAB-23: per-call CONTEXT and OUTPUT limits for the model the call runs on (2026-09-25).
#
# DS41 run 10 (18:49Z): the first planner call ran 61 min and reached 183,710 tokens of
# :8083's 196,608-token UNIFIED KV pool (np4, --kv-unified, MTP draft); a full unified
# pool under MTP crashes llama-server ("speculative batch index 8 is not inside the
# current sub-batch", reproduced 4/4). The global provider entry gives the model no
# `limit`, and opencode 1.18.31 then NEVER compacts proactively. Read from the installed
# binary (read-only):
#
# * Provider model parse: `limit:{context:C.limit?.context??_?.limit?.context??0,
#   input:C.limit?.input??..., output:C.limit?.output??_?.limit?.output??0}` -- a
#   config-only model (not in models.dev) defaults to context 0, output 0.
# * `ProviderTransform.maxOutputTokens(model, cap=OUTPUT_TOKEN_MAX=32000)` is
#   `Math.min(model.limit.output, cap) || cap`: output 0 -> 32000. It is passed as
#   `maxOutputTokens` on every LLM request, and `@ai-sdk/openai-compatible` sends it as
#   `max_tokens`. So today every step may decode 32,000 tokens (the observed ~30k-token
#   single reasoning turns); `limit.output: O` makes it `max_tokens: min(O, 32000)`.
# * `SessionCompaction.isOverflow`: false when `compaction.auto === false` or
#   `limit.context === 0`; otherwise true when the LAST finished assistant step's
#   `tokens.total` (input + output + cache) >= usable, where usable = `limit.input -
#   reserved` if `limit.input` is set, else `limit.context - maxOutputTokens(model)`.
#   The loop checks it before every step and runs an auto compaction first. Context 0
#   is why the seat only ever compacted REACTIVELY, on a server context-overflow error
#   -- which a unified pool never returns before it is full.
# * A step that ends on `finish == "length"` (the output cap) is not "tool-calls", so
#   the session loop EXITS after it: a step cut at O tokens ends the call (the reply is
#   whatever text it had; the loop's salvage/repair/transient paths take it from there).
#
# With C = 131,072 and O = 8,192: compaction fires once a step's total reaches C - O =
# 122,880; the next request is at most that plus one step's tool results plus O
# decoded, i.e. ~C + one step of tool output (bounded seat: <=12 KB per result; plain
# seat: opencode's 50 KB default) -- about 131k-150k of the 196,608 pool, leaving
# >= ~45k for the other three slots of :8083 (np4). O = 8,192 is twice the ~4,000-token
# analysis budget the concise rule states (`actors.CONCISE_RULE`).
# --------------------------------------------------------------------------------------

# Operator pool budget, 2026-09-26 (supersedes C = 131,072 and the O < C/2 rule): a
# FULL unified pool plus MTP crashes llama-server, so the bound is on the pool, not on
# O relative to C. :8083's unified KV pool is POOL_TOKENS = 196,608; every role's C is
# at most POOL_TOKENS - POOL_RESERVE (16,384 always stay free for the other :8083
# slots), and every O is below C - MIN_COMPACTION_HEADROOM (32,768), so compaction
# (at C - O) always leaves at least 32k of context before the cap. Defaults: C =
# 180,224 for all roles; author O = 40,960 (compaction ~139k), planner and critic O =
# 16,384 (compaction ~164k). run.py validates both bounds.
POOL_TOKENS = 196_608
POOL_RESERVE = 16_384
MAX_CONTEXT_LIMIT = POOL_TOKENS - POOL_RESERVE
MIN_COMPACTION_HEADROOM = 32_768

#: run.py defaults for `--actor-context-limit` / `--actor-output-limit`. 0 = opencode's
#: own default (context 0: no proactive compaction; output 0: max_tokens 32000).
DEFAULT_CONTEXT_LIMIT = 180_224
DEFAULT_OUTPUT_LIMIT = 8_192
#: Per-role `limit.output` (run.py `--actor-planner-output-limit` / `--actor-author-
#: output-limit`; the critic takes the planner's). DS41 run 10b (2026-09-25): the author
#: ended on ONE 8,192-token step with no report -- a file-write tool call's arguments are
#: output -- and the planner hit 8,192 once but still replied. The author's 40,960 is
#: above opencode's 32,000 ceiling, so its calls carry `output_ceiling_env`.
DEFAULT_PLANNER_OUTPUT_LIMIT = 16_384
DEFAULT_AUTHOR_OUTPUT_LIMIT = 40_960


def model_limits(model: str | None, *, context_limit: int = 0,
                 output_limit: int = 0) -> dict:
    """The `provider` block that sets `limit` on the call's `provider/model`, or {}.

    opencode deep-merges OPENCODE_CONFIG over the global config (plain objects merge
    key by key, `uW` -> mergeDeep), so this entry ADDS `limit` to the global
    `provider.<id>.models.<model>` and keeps its npm, name and options.baseURL. The
    config schema's `limit` requires both `context` and `output`, so both are always
    written; 0 keeps opencode's own default for that one."""
    for label, value in (("context_limit", context_limit), ("output_limit", output_limit)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative int, got {value!r}")
    if not (context_limit or output_limit):
        return {}
    if not model or "/" not in model:
        raise ValueError(f"model limits need a provider/model id, got {model!r}")
    if context_limit and output_limit and output_limit >= context_limit:
        raise ValueError(f"output_limit {output_limit} must be below context_limit "
                         f"{context_limit}: opencode compacts at context - output")
    provider, model_id = model.split("/", 1)
    return {"provider": {provider: {"models": {model_id: {
        "limit": {"context": context_limit, "output": output_limit}}}}}}


# --------------------------------------------------------------------------------------
# OAB-24: the AUTHOR's reasoning switch (operator, 2026-09-25: the local 27B author runs
# with thinking OFF, author calls only; the planner keeps reasoning on).
#
# DS41 runs 10b/10c: the author decoded 74,288 tokens over 13 steps in 2,700 s, hit the
# output cap twice re-deriving the block_q8_2_x4 layout inside <think>, and made zero
# edits. The switch is the served template's own `enable_thinking` kwarg, sent per
# request. The path, read from the installed opencode 1.18.31 binary (read-only):
#
# * `LLMRequestPrep.prepare`: `options = mergeDeep(mergeDeep(mergeDeep(base,
#   model.options), agent.options), variant)`, where `base` is `ProviderTransform.options`
#   (or `smallOptions` for small/title calls -- model.options is merged over BOTH).
# * `ProviderTransform.providerOptions(model, options)`: for `@ai-sdk/openai-compatible`
#   (no `sdkKey`) the key is `providerID.split(".")[0]`, i.e. `{"qwen-gpu": options}`.
# * `@ai-sdk/openai-compatible` chat `getArgs`: `...Object.fromEntries(Object.entries(
#   {...providerOptions[providerOptionsName], ...providerOptions[camelCase(name)]})
#   .filter(([k]) => !Object.keys(<its own option schema>.shape).includes(k)))` is spread
#   into the request BODY -- unknown keys pass through verbatim. opencode creates the SDK
#   with `name: providerID`, so `providerOptionsName` is "qwen-gpu".
#
# So `provider.<id>.models.<model>.options.chat_template_kwargs` in the per-call config
# (deep-merged over the global provider entry, like `limit`) becomes
# `"chat_template_kwargs":{"enable_thinking":false}` in every POST body of that call
# (proved against a recording mock server, `test_actor_author_thinking.py`). `provider.
# <id>.options` would NOT work: those go to the SDK factory (baseURL, headers), never the
# body. On the server, llama-server (--jinja) merges the request's `chat_template_kwargs`
# over its CLI defaults and parses `enable_thinking` into `inputs.enable_thinking`
# (`tools/server/server-common.cpp`), and the served template
# `epyc-qwen3x-v1-terse.jinja` reads `enable_thinking` (default true) and, when false,
# opens the assistant turn with an empty `<think>\n\n</think>` block.
#
# The kwarg rides the model entry of the AUTHOR's per-call config only, so it reaches
# every request of that call (its scouts and any compaction step included) and no
# planner or critic request. "default" writes nothing: the config is byte-identical.
# --------------------------------------------------------------------------------------

# Operator, 2026-09-26: "off" edited but wrote broken AVX-512 (five critic rejections in
# DS41 run 10g); thinking ON uncapped deliberated 74k tokens without editing. The author
# now runs thinking ON at MEDIUM reasoning effort (`reasoning_effort` rides the same
# per-request `chat_template_kwargs`), with the output cap at its maximum and the author
# action rule (`actors.AUTHOR_ACTION_RULE`) telling it to think briefly and act.
#: `--actor-author-thinking` choices; run.py defaults to "medium" (the operator's choice).
THINKING_CHOICES = ("default", "off", "medium")
DEFAULT_AUTHOR_THINKING = "medium"
#: The model `options` that turn the served template's reasoning off for one call.
THINKING_OFF_OPTIONS = {"chat_template_kwargs": {"enable_thinking": False}}
#: The model `options` for thinking ON at medium reasoning effort.
THINKING_MEDIUM_OPTIONS = {"chat_template_kwargs": {"enable_thinking": True,
                                                    "reasoning_effort": "medium"}}
THINKING_OPTIONS = {"off": THINKING_OFF_OPTIONS, "medium": THINKING_MEDIUM_OPTIONS}


def model_thinking(model: str | None, thinking: str = "default") -> dict:
    """The `provider` block that sets the call's reasoning kwargs on its model, or {}."""
    if thinking not in THINKING_CHOICES:
        raise ValueError(f"thinking must be one of {THINKING_CHOICES}, got {thinking!r}")
    if thinking == "default":
        return {}
    if not model or "/" not in model:
        raise ValueError(f"thinking={thinking!r} needs a provider/model id, got {model!r}")
    provider, model_id = model.split("/", 1)
    return {"provider": {provider: {"models": {model_id: {
        "options": json.loads(json.dumps(THINKING_OPTIONS[thinking]))}}}}}


#: opencode 1.18.31 sends `max_tokens = min(limit.output, OUTPUT_TOKEN_MAX)`, where
#: OUTPUT_TOKEN_MAX is `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` (a positive int) or 32000
#: (`ProviderTransform.maxOutputTokens`, read from the installed binary). A `limit.output`
#: above 32000 is silently clamped unless the call's env raises the ceiling, so a call
#: whose output limit exceeds it carries this env var set to that limit. Compaction also
#: reads it (threshold = context - maxOutputTokens).
OUTPUT_TOKEN_MAX_ENV = "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"
OPENCODE_OUTPUT_TOKEN_MAX = 32_000


def output_ceiling_env(output_limit: int) -> dict[str, str]:
    """The env that lets `output_limit` reach the wire, or {} at/below opencode's 32000."""
    if output_limit > OPENCODE_OUTPUT_TOKEN_MAX:
        return {OUTPUT_TOKEN_MAX_ENV: str(int(output_limit))}
    return {}


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for key, value in b.items():
        out[key] = (_merge(a[key], value)
                    if isinstance(a.get(key), dict) and isinstance(value, dict) else value)
    return out


def model_block(model: str | None, *, context_limit: int = 0, output_limit: int = 0,
                thinking: str = "default") -> dict:
    """`model_limits` and `model_thinking` for one call's model, merged ({} when both off)."""
    return _merge(model_limits(model, context_limit=context_limit, output_limit=output_limit),
                  model_thinking(model, thinking))


def limits_label(*, context_limit: int = 0, output_limit: int = 0) -> str:
    """`+ctx128k+out8k` style arm suffix for the limits that are on."""
    def k(value: int) -> str:
        return f"{value // 1024}k" if value % 1024 == 0 else str(value)
    return (("+ctx" + k(context_limit) if context_limit else "")
            + ("+out" + k(output_limit) if output_limit else ""))


def _tool(name: str) -> str:
    return f"{MCP_SERVER}_{name}"


_FINAL_JSON = ("6. Stop investigating as soon as you can answer. Print the requested single "
               "JSON object as the LAST thing in your reply, with nothing after it.")
_FINAL_SUMMARY = ("6. Stop as soon as you can answer. Reply with a summary of at most ~15 "
                  "lines with file:line references -- no file dumps.")


def _discipline(final: str = _FINAL_JSON) -> str:
    t = _tool
    return "\n".join([
        "Tool discipline -- your context is one ~98k-token slot and every byte a tool "
        "returns stays in it:",
        f"1. Locate before you read: {t('outline')} for a file's structure, "
        f"{t('grep')} or {t('code_search')} with a NARROW pattern, then "
        f"{t('read_range')} on only the lines you need.",
        "2. Never read more than ~200 lines at once and never read a whole file. If you "
        "use the built-in read tool, always pass offset and limit.",
        f"3. Profiles: use {t('profile_top')} and {t('symbol_annotate')}. Never run perf "
        "report/annotate/script or print a profile through bash.",
        "4. bash is for short, bounded commands only (e.g. `git log --oneline -n 20`, "
        "`git diff --stat`); bound every command's output.",
        "5. Do not re-read what you have already read; keep notes in your reasoning.",
        final,
    ])


def _fan_out() -> str:
    n = MAX_CONCURRENT_SUBAGENTS
    return "\n".join([
        "Parallel investigation -- keep your own context small:",
        f"- When questions are independent (e.g. one per candidate hotspot or file), "
        f"delegate them with the `task` tool, subagent_type \"{SCOUT_AGENT}\", issuing "
        f"up to {n} task calls in the SAME message so they run concurrently. The server "
        f"has {n} slots: never more than {n} subagents at once.",
        "- Give each scout one precise question and ask for a summary of at most ~15 "
        "lines with file:line references.",
        "- Work from the summaries; do not repeat a scout's reads yourself.",
    ])


def _role_intro(role: str) -> str:
    if role == "planner":
        return ("You are the autokernel PLANNER for a llama.cpp kernel lane worktree. "
                "You investigate the code and profiles and PROPOSE one change; you cannot "
                "edit files.")
    return ("You are the autokernel AUTHOR for a llama.cpp kernel lane worktree. You "
            "implement the requested change with minimal, targeted edits.")


def _scout_prompt() -> str:
    return "\n\n".join([
        "You are an autokernel SCOUT subagent. Answer the one question you were given "
        "by reading code and profiles; you cannot edit files and cannot start subagents.",
        _discipline(_FINAL_SUMMARY),
    ])


def build_actor_config(*, role: str, lane: Path, profiles: Sequence[Path] = (),
                       python: str = sys.executable,
                       research_root: Path = RESEARCH_ROOT, steps: int = 60,
                       tool_output_max_lines: int = 250,
                       tool_output_max_bytes: int = 12000,
                       fan_out: bool = True,
                       instructions_path: Path | None = None,
                       replace_system_prompt: bool = False,
                       trim_instructions: bool = False, trim_tools: bool = False,
                       lane_guard: bool = False,
                       build_dir: str | Path | None = None,
                       model: str | None = None, context_limit: int = 0,
                       output_limit: int = 0, thinking: str = "default",
                       author_sandbox: bool = False) -> dict:
    """The opencode config (a dict ready for ``json.dump``) for one actor run.

    By default the seat's guidance is ADDED to opencode's own system prompt through
    the top-level ``instructions`` file list (``instructions_path``; its text is
    ``actor_instructions(role, fan_out)``). An agent ``prompt`` REPLACES that system
    prompt, and opencode's default is the part that keeps the model terse: with it
    replaced (v1, ``replace_system_prompt=True``), the 27B planner decoded a median
    1,626 tokens per step against 266 on the plain seat and filled its 98k slot in 12
    steps instead of ~45 (DS41 seat A/B, 2026-09-24).

    `trim_instructions` / `trim_tools` / `lane_guard` (OAB-10/11, all off by default)
    add a TOP-LEVEL `permission` block of denies (`seat_permission`) that the global
    config's rules merge with; under the lane guard the author agent's own `edit` rule
    becomes the lane-only guard (an agent rule is evaluated after the top-level one).

    `context_limit` / `output_limit` (OAB-23, 0 = off) add `model_limits(model, ...)`;
    `thinking` "off"/"medium" (OAB-24) adds `model_thinking(model, thinking)` on the same model entry."""
    if role not in AGENT_NAMES:
        raise ValueError(f"unknown actor role {role!r}; expected one of "
                         f"{sorted(AGENT_NAMES)}")
    for label, value in (("steps", steps), ("tool_output_max_lines", tool_output_max_lines),
                         ("tool_output_max_bytes", tool_output_max_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive int, got {value!r}")

    root = Path(research_root)
    command = [str(python), "-m", MCP_MODULE, "--root", str(lane)]
    for profile in profiles:
        command += ["--profiles", str(profile)]

    mcp_tools = {f"{MCP_SERVER}_*": "allow"}
    author_edit = ({"*": "allow", **AUTHOR_EDIT_GUARD} if lane_guard else "allow")
    permission: dict = {"edit": "deny" if role == "planner" else author_edit, **mcp_tools}
    if fan_out:
        permission["task"] = {"*": "deny", SCOUT_AGENT: "allow"}
    else:
        permission["task"] = "deny"

    if not replace_system_prompt and instructions_path is None:
        raise ValueError("instructions_path is required unless replace_system_prompt=True")

    primary: dict = {
        "description": f"autokernel {role} seat (bounded tools, step-capped)",
        "mode": "primary",
        "steps": steps,
        "permission": permission,
    }
    if replace_system_prompt:
        primary["prompt"] = actor_instructions(role, fan_out, scout_note=False)
    agents = {AGENT_NAMES[role]: primary}
    if fan_out:
        scout: dict = {
            "description": ("Read-only investigator for ONE independent question about "
                            "the lane's code or profiles; returns a <=15-line summary."),
            "mode": "subagent",
            "hidden": True,
            "steps": max(1, min(20, steps)),
            "permission": {"edit": "deny", "task": "deny", **mcp_tools},
        }
        if replace_system_prompt:
            scout["prompt"] = _scout_prompt()
        agents[SCOUT_AGENT] = scout

    top = seat_permission(role, lane=lane, build_dir=build_dir,
                          trim_instructions=trim_instructions, trim_tools=trim_tools,
                          lane_guard=lane_guard, keep_task=fan_out, edit_rule=False,
                          author_sandbox=author_sandbox)
    limits = model_block(model, context_limit=context_limit, output_limit=output_limit,
                         thinking=thinking)
    return {
        "$schema": "https://opencode.ai/config.json",
        **SNAPSHOT_OFF,
        **limits,
        **({"permission": top} if top else {}),
        "tool_output": {"max_lines": tool_output_max_lines,
                        "max_bytes": tool_output_max_bytes},
        "compaction": {"prune": True},
        "mcp": {
            MCP_SERVER: {
                "type": "local",
                "command": command,
                "cwd": str(root),
                # Merged over the parent env by opencode. Both roots: the module is
                # launched by its repo-root path, and loop code imports `autokernel.*`.
                "environment": {"PYTHONPATH": os.pathsep.join(
                    [str(root), str(root / "scripts" / "kernel_rnd")])},
                "enabled": True,
                "timeout": MCP_TIMEOUT_MS,
            },
        },
        "agent": agents,
        **({} if replace_system_prompt else {"instructions": [str(instructions_path)]}),
    }


def actor_instructions(role: str, fan_out: bool, *, scout_note: bool = True) -> str:
    """The seat's guidance text: role, tool discipline, fan-out. As an ``instructions``
    file it is loaded by every agent in the run, scouts included, hence the note."""
    sections = [_role_intro(role), _discipline()]
    if fan_out:
        sections.append(_fan_out())
        if scout_note:
            sections.append(f"If you are a {SCOUT_AGENT} subagent: ignore the role and fan-out "
                            "text above; answer only your one question, read-only, in a summary "
                            "of at most ~15 lines with file:line references -- no file dumps.")
    return "\n\n".join(sections)


def write_actor_config(path: Path, **kw) -> Path:
    """Build the config and write it to ``path`` atomically (temp file + rename)."""
    path = Path(path)
    if kw.get("role") not in AGENT_NAMES:
        raise ValueError(f"unknown actor role {kw.get('role')!r}; expected one of "
                         f"{sorted(AGENT_NAMES)}")
    if not kw.get("replace_system_prompt"):
        instructions = path.with_name(path.stem + ".instructions.md")
        kw.setdefault("instructions_path", instructions)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(Path(kw["instructions_path"]),
                      actor_instructions(kw["role"], kw.get("fan_out", True)) + "\n")
    config = build_actor_config(**kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return path


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


# --------------------------------------------------------------------------------------
# OAB-10 / OAB-11: fixed-overhead trim and the lane guard (2026-09-25).
#
# HOW opencode 1.18.31 BUILDS THE FIXED PART OF EVERY CALL (read from the installed
# binary: `Instruction.systemPaths`, `SystemPrompt.skills`, `Permission.disabled`):
#
# * Instruction files. Global: the FIRST existing of `~/.config/opencode/AGENTS.md` and
#   `~/.claude/CLAUDE.md` (the latter unless OPENCODE_DISABLE_CLAUDE_CODE[_PROMPT]).
#   Project, unless OPENCODE_DISABLE_PROJECT_CONFIG: for the names AGENTS.md, CLAUDE.md,
#   CONTEXT.md in that order, `findUp(name, --dir, git worktree root)`; the FIRST name
#   with any hit wins and each hit is loaded as "Instructions from: <path>\n<text>". A
#   lane is its own git worktree, so this loads exactly the lane's AGENTS.md (8,857 chars
#   of ggml-org contribution policy; the lane's CLAUDE.md overlay is shadowed by it).
#   Config `instructions` entries are ADDED; nothing in the config removes a discovered
#   file, so the switch is the env var. It also skips project opencode.json and .opencode/
#   dirs (a llama.cpp lane has neither) and keeps OPENCODE_CONFIG, the global config and
#   its plugins. A `read` of a file in a SUBDIRECTORY still attaches that subdirectory's
#   AGENTS.md (`Instruction.resolve`, not gated); a llama.cpp lane has none.
# * The skill catalog. `SystemPrompt.skills` appends "Skills provide specialized
#   instructions..." and an <available_skills> block (name, description, location) for
#   every SKILL.md under ~/.claude/skills and ~/.agents/skills (and the project's), plus
#   the built-in `customize-opencode`. On this host: the 12 synced claude.ai skills (pdf,
#   docx, pptx, chrome-browser, ...; 8,149 description chars). It is omitted, and the
#   `skill` tool dropped, when the agent's permission disables `skill`.
# * Tool schemas. A tool is dropped from the request when the LAST rule matching its
#   permission has pattern "*" and action "deny" (edit/write/apply_patch share `edit`).
#
# PERMISSION SEMANTICS (same binary): findLast over defaults -> global config ->
# OPENCODE_CONFIG (deep-merged, global keys first) -> agent rules; `--auto` approves
# every `ask` no rule DENIES, so every fence here is an explicit deny. Bash is checked
# per parsed command (tree-sitter), matching the whole command text, heredoc body
# included, against a wildcard: `*` is any run of characters including newlines, `?` is
# ONE character, a trailing " *" makes the arguments optional, all else is literal.
# `external_directory` is asked for native read/grep/glob/edit/write paths outside
# --dir, but for bash only on cd/rm/cp/mv/mkdir/touch/chmod/chown/cat arguments and the
# workdir -- so bash reads of another tree need bash rules too.
# --------------------------------------------------------------------------------------

#: Env for a trimmed call: no project instruction files, no Claude-Code prompt or skills,
#: no external (~/.agents, project .claude/.agents) skills. Truthy is "1"/"true".
TRIM_ENV = {"OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "OPENCODE_DISABLE_CLAUDE_CODE": "1",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1"}

#: Tools the plain planner never called in the DS41 transcripts (C20c and the four OAB-9
#: calls used only bash/read/grep/glob; the 27B never delegated through `task` in 69
#: bounded steps). Denying them drops their schemas from every request.
UNUSED_TOOLS = ("task", "todowrite", "webfetch", "websearch", "question", "lsp")

#: Never build, compile, link, benchmark or test: the loop owns every build and measures
#: on this CPU. Prefix patterns match one parsed command; the " <tool> " forms catch
#: wrappers (`timeout 60 gcc ...`, `env CC=x g++ ...`) and are kept to names that do not
#: occur in ordinary greps or prose (`make`, `cc` do, so they are prefix-only).
BUILD_DENY = (
    "cmake*", "*/cmake *", "* cmake *", "ctest*",
    "make", "make *", "*/make *", "gmake*",
    "ninja*", "*/ninja *", "* ninja *",
    "gcc*", "*/gcc *", "* gcc *",
    "g++*", "*/g++ *", "* g++ *",
    "cc *", "*/cc *", "c++ *", "*/c++ *",
    "clang*", "*/clang *", "* clang *", "*/clang++ *", "* clang++ *",
    "ccache*", "* ccache *", "hipcc*", "*/hipcc *", "nvcc*", "*/nvcc *",
    "ld *", "ld.*", "*.o", "*.o *",
    "llama-bench*", "*/llama-bench *", "*/llama-bench",
    "llama-server*", "*/llama-server *", "*/llama-server",
    "llama-cli*", "*/llama-cli *", "*/llama-cli",
    "llama-perplexity*", "*/llama-perplexity *",
    "perf record*", "perf stat*", "perf top*", "*/perf record*", "*/perf stat*",
)

#: A read-only role (planner, critic) changes nothing anywhere: not its lane (a stray
#: planner edit rides into the author's diff; a critic's is caught only after the fact
#: by the tree check) and not outside it (`--auto` approves external writes).
READ_ONLY_DENY = (
    "rm *", "rmdir *", "mv *", "cp *", "mkdir *", "touch *", "tee *", "ln *",
    "chmod *", "install *", "dd *", "truncate *", "patch *",
    "sed -i*", "sed --in-place*", "perl -i*", "perl -pi*",
    "git apply*", "git am *", "git checkout*", "git switch*", "git restore*",
    "git reset*", "git stash*", "git commit*", "git clean*", "git merge*",
    "git rebase*", "git cherry-pick*", "git revert*", "git worktree*", "git push*",
    "git add*", "git rm *", "git mv *", "git tag *",
    "*> /tmp/*", "*>/tmp/*", "*>> /tmp/*", "*>>/tmp/*",
)

#: The author edits only inside its lane. `edit` patterns are paths RELATIVE to the git
#: worktree root, so anything outside the lane starts with "../". No "*": "allow" here:
#: the default already allows, and a top-level allow would re-open edit for the native
#: agents (plan/explore/compaction) whose defaults deny it.
AUTHOR_EDIT_GUARD = {"../*": "deny", ".git/*": "deny"}

#: `ak-check` (operator 2026-09-26, `ak_check.py`): the author's ONE sanctioned check, a
#: compile / op test of its patch in a loop-allocated scratch dir that refuses while the
#: campaign measures. The author gets exactly these command texts as allows, appended
#: AFTER every deny so they are the last match (everything else stays denied: `gcc ...`,
#: `ak-check; gcc ...` parse into separate commands). Planner and critic get the denies.
AK_CHECK_ALLOW = ("ak-check", "ak-check --op-test")
AK_CHECK_DENY = ("ak-check*", "* ak-check*", "*/ak-check*", "*ak_check*")

#: GitNexus under the lane guard. Every role is denied the subcommands that write an
#: index or the host's editor setup (a bare `gitnexus analyze` in a lane would index
#: the lane under the anchor's name; only /workspace/scripts/gitnexus-analyze.sh may
#: re-index). The AUTHOR is allowed exactly the two read forms its action rule names,
#: targeting the anchor by its ABSOLUTE path: the anchor is registered under the name
#: `llama.cpp`, which collides with the frozen production tree, and the author's cwd
#: (a lane worktree at another path) cannot disambiguate a name. Appended after every
#: deny (the last match wins), so the anchor-path denies still refuse anything else.
GITNEXUS_DENY = tuple(pattern for sub in ("analyze", "index", "clean", "remove", "setup",
                                          "uninstall", "wiki", "publish", "serve", "mcp",
                                          "eval-server", "group")
                      for pattern in (f"gitnexus {sub}*", f"* gitnexus {sub}*",
                                      f"*/gitnexus {sub}*"))
GITNEXUS_READS = ("context", "query")


def gitnexus_allow(root: str | Path) -> tuple[str, ...]:
    """The author's bash allows for `gitnexus context|query ... --repo <root>`."""
    return tuple(pattern for sub in GITNEXUS_READS
                 for pattern in (f"gitnexus {sub} * --repo {root}",
                                 f"gitnexus {sub} * --repo {root} *",
                                 f"gitnexus {sub} --repo {root} *"))


#: Roles the plain config knows (the bounded config has no critic seat).
PLAIN_ROLES = ("planner", "author", "critic")

#: What the author keeps from the lane AGENTS.md: its code-style lines. The rest of that
#: file is ggml-org's PR policy, and parts are actively wrong for this seat ("If you are
#: a fully autonomous agent ... do not contribute ... STOP", "Guide, don't solve",
#: "PAUSE and ask the user") -- a headless author has nobody to ask.
AUTHOR_STYLE_NOTE = "\n".join([
    "Code style for edits in this llama.cpp tree (from its AGENTS.md):",
    "- ASCII only in code and comments: no em dash, unicode arrows, x-sign or ellipsis.",
    "- Keep comments concise; never restate what the code says or narrate the task.",
    "- Reuse existing infrastructure and blend in with the surrounding code; no new "
    "subsystems.",
    "- Read the relevant code before you write any.",
])


def anchor_fence(build_dir: str | Path | None, lane: Path | None = None
                 ) -> tuple[Path | None, tuple[Path, ...], tuple[str, ...]]:
    """(anchor source root, its build dirs, its other top-level entry names).

    The root is the nearest ancestor of `build_dir` holding `.git` (the kernel tree),
    else `build_dir`'s parent; build dirs are the root's `build*` children (the measured
    build and siblings such as the instrumented `build-cpu-prof`). Nothing is fenced
    without a build dir, or when the lane sits inside the root (that would fence the
    lane itself)."""
    if not build_dir:
        return None, (), ()
    build = Path(build_dir)
    root = next((p for p in build.parents if (p / ".git").exists()), build.parent)
    if str(root) in ("/", ""):
        return None, (), ()
    if lane is not None:
        try:
            Path(lane).resolve().relative_to(root.resolve())
            return None, (), ()
        except ValueError:
            pass
    try:
        entries = sorted((p.name, p.is_dir()) for p in root.iterdir())
    except OSError:
        entries = []
    builds = {root / name for name, is_dir in entries if is_dir and name.startswith("build")}
    try:
        builds.add(root / build.relative_to(root).parts[0])
    except (ValueError, IndexError):
        pass
    others = tuple(name for name, _ in entries if root / name not in builds)
    return root, tuple(sorted(builds)), others


def seat_permission(role: str, *, lane: Path | None = None,
                    build_dir: str | Path | None = None,
                    trim_instructions: bool = False, trim_tools: bool = False,
                    lane_guard: bool = False, keep_task: bool = False,
                    edit_rule: bool = True, author_sandbox: bool = False,
                    read_roots: tuple = ()) -> dict:
    """The permission block OAB-10/11 add, for a TOP-LEVEL `permission` key.

    Denies only, except `external_directory` allows for the anchor's build dirs, which
    re-open only what this block's own anchor-root deny closed (the global config has no
    external_directory rule). `keep_task` leaves `task` alone (bounded fan-out owns it);
    `edit_rule=False` leaves `edit` to the caller (a bounded agent sets its own, and an
    agent rule is evaluated after this one).

    `read_roots` (the critic's, `actors._read_roots`) add `external_directory` ALLOWS for
    directories its context points into: the read-only critic has no `--auto`, so an
    `ask` there is auto-rejected and the rejection ends its session with no reply. The
    critic's `edit` stays denied (lane guard) or asked-and-rejected (no `--auto`), so
    this re-opens reads only."""
    if role not in PLAIN_ROLES:
        raise ValueError(f"unknown actor role {role!r}; expected one of {PLAIN_ROLES}")
    permission: dict = {}
    if trim_instructions:
        permission["skill"] = "deny"
    if trim_tools:
        for tool in UNUSED_TOOLS:
            if not (tool == "task" and keep_task):
                permission[tool] = "deny"
    if lane_guard:
        bash = {pattern: "deny" for pattern in BUILD_DENY}
        bash.update({pattern: "deny" for pattern in GITNEXUS_DENY})
        if role != "author":
            bash.update({pattern: "deny" for pattern in READ_ONLY_DENY})
        root, builds, others = anchor_fence(build_dir, lane)
        if root is not None:
            for pattern in (f"*{root}", f"*{root} *", f"*{root}/", f"*{root}/ *"):
                bash[pattern] = "deny"
            for name in others:
                bash[f"*{root}/{name}*"] = "deny"
            if role == "author":
                for pattern in gitnexus_allow(root):
                    bash[pattern] = "allow"     # after every deny: the last match wins
            external = {f"{root}/*": "deny"}
            external.update({f"{b}/*": "allow" for b in builds})
            permission["external_directory"] = external
        permission["bash"] = bash
        if edit_rule:
            permission["edit"] = dict(AUTHOR_EDIT_GUARD) if role == "author" else "deny"
    if read_roots:
        external = permission.setdefault("external_directory", {})
        for root in read_roots:
            root = str(root).rstrip("/")
            if root:
                external.pop(f"{root}/*", None)   # re-inserted LAST: the last match wins
                external[f"{root}/*"] = "allow"
    if author_sandbox:
        bash = permission.setdefault("bash", {})
        for pattern in (AK_CHECK_ALLOW if role == "author" else AK_CHECK_DENY):
            bash.pop(pattern, None)   # re-inserted LAST: the last match wins
            bash[pattern] = "allow" if role == "author" else "deny"
    return permission


def gitnexus_repo_for(build_dir: str | Path | None) -> str | None:
    """The `gitnexus --repo` target of the anchor tree holding `build_dir`: its git
    root's ABSOLUTE path, or None without one.

    Not the name: `gitnexus analyze` registers a tree under its git remote/repo name,
    and the DS41 anchor (`/mnt/raid0/llm/llama.cpp-experimental-fastload-ds41-...`) is
    registered as `llama.cpp`, colliding with the frozen production tree at
    `/mnt/raid0/llm/llama.cpp`; `--repo llama.cpp` is refused as ambiguous, and the
    author's cwd (a lane worktree at another path) cannot disambiguate it. The
    absolute path is GitNexus's own disambiguation, and the lane guard allows exactly
    `gitnexus context|query ... --repo <this path>` (`gitnexus_allow`)."""
    root, _builds, _others = anchor_fence(build_dir)
    return str(root) if root is not None else None


def seat_label(base: str, *, trim_instructions: bool = False, trim_tools: bool = False,
               author_sandbox: bool = False,
               lane_guard: bool = False, context_limit: int = 0, output_limit: int = 0,
               concise: bool = False, budget_s: int = 0, thinking_off: bool = False,
               thinking: str = "default", action_rule: bool = False) -> str:
    """`plain` / `bounded` plus one suffix per knob that is on (free text in VB-AK-SEAT).
    OAB-22/23 add `+ctx<C>+out<O>`, `+concise` and `+budget<B>s`; OAB-24 adds
    `+think-off` (or `+think-<thinking>`), and the author action rule `+act-rule`
    (all off: unchanged)."""
    if thinking_off:
        thinking = "off"
    return (base + "".join(suffix for on, suffix in (
        (trim_instructions, "+trim-instr"), (trim_tools, "+trim-tools"),
        (lane_guard, "+lane-guard"), (author_sandbox, "+ak-check")) if on)
        + limits_label(context_limit=context_limit, output_limit=output_limit)
        + (f"+think-{thinking}" if thinking and thinking != "default" else "")
        + ("+concise" if concise else "") + ("+act-rule" if action_rule else "")
        + (f"+budget{budget_s}s" if budget_s else ""))


def build_plain_config(*, role: str, lane: Path, build_dir: str | Path | None = None,
                       trim_instructions: bool = False, trim_tools: bool = False,
                       lane_guard: bool = False, author_sandbox: bool = False,
                       author_note_path: Path | None = None,
                       model: str | None = None, context_limit: int = 0,
                       output_limit: int = 0, thinking: str = "default",
                       read_roots: tuple = ()) -> dict:
    """The per-call `OPENCODE_CONFIG` for the PLAIN seat: `snapshot: false`, a permission
    block (plus the author's style note as an `instructions` file) and nothing else -- no
    agent, no MCP, no tool_output cap, so the plain seat stays the plain seat. With every
    knob off it is `{"$schema", "snapshot": false}` alone: the plain seat ALWAYS gets a
    per-call config, because snapshot tracking is on by default and bloats the store.
    `context_limit` / `output_limit` (OAB-23, 0 = off) add `model_limits(model, ...)`;
    `thinking` "off"/"medium" (OAB-24) adds `model_thinking(model, thinking)`."""
    permission = seat_permission(role, lane=lane, build_dir=build_dir,
                                 trim_instructions=trim_instructions,
                                 trim_tools=trim_tools, lane_guard=lane_guard,
                                 author_sandbox=author_sandbox, read_roots=tuple(read_roots))
    note = trim_instructions and role == "author" and author_note_path is not None
    config: dict = {"$schema": "https://opencode.ai/config.json", **SNAPSHOT_OFF,
                    **model_block(model, context_limit=context_limit,
                                  output_limit=output_limit, thinking=thinking)}
    if permission:
        config["permission"] = permission
    if note:
        config["instructions"] = [str(author_note_path)]
    return config


def write_plain_config(path: Path, **kw) -> Path:
    """Build the plain-seat config and write it (and the author note) atomically. Always
    writes: even the knobs-off config carries `snapshot: false`."""
    path = Path(path)
    if kw.get("trim_instructions") and kw.get("role") == "author":
        kw.setdefault("author_note_path", path.with_name(path.stem + ".instructions.md"))
    config = build_plain_config(**kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "instructions" in config:
        _atomic_write(Path(config["instructions"][0]), AUTHOR_STYLE_NOTE + "\n")
    _atomic_write(path, json.dumps(config, indent=2) + "\n")
    return path


__all__ = ["actor_instructions", "AGENT_NAMES", "AK_CHECK_ALLOW", "AK_CHECK_DENY", "DEFAULT_AUTHOR_OUTPUT_LIMIT",
           "DEFAULT_AUTHOR_THINKING", "THINKING_CHOICES", "THINKING_OFF_OPTIONS",
           "THINKING_MEDIUM_OPTIONS", "THINKING_OPTIONS", "OUTPUT_TOKEN_MAX_ENV",
           "OPENCODE_OUTPUT_TOKEN_MAX", "output_ceiling_env", "POOL_TOKENS",
           "POOL_RESERVE", "MAX_CONTEXT_LIMIT", "MIN_COMPACTION_HEADROOM", "gitnexus_repo_for",
           "GITNEXUS_DENY", "GITNEXUS_READS", "gitnexus_allow",
           "model_block", "model_thinking",
           "DEFAULT_CONTEXT_LIMIT", "DEFAULT_OUTPUT_LIMIT", "DEFAULT_PLANNER_OUTPUT_LIMIT", "limits_label", "model_limits", "AUTHOR_EDIT_GUARD",
           "AUTHOR_STYLE_NOTE",
           "BUILD_DENY", "MAX_CONCURRENT_SUBAGENTS", "MCP_SERVER", "PLAIN_ROLES",
           "READ_ONLY_DENY", "SCOUT_AGENT", "SNAPSHOT_OFF", "TRIM_ENV", "UNUSED_TOOLS", "anchor_fence",
           "build_actor_config", "build_plain_config", "seat_label", "seat_permission",
           "write_actor_config", "write_plain_config"]
