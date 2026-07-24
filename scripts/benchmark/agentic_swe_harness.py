#!/usr/bin/env python3
"""Agentic SWE-bench harness: multi-turn, no-oracle repo-fixing loop.

Tier above the single-turn ORACLE rung
(artifacts/architect-code-eval-20260724/build_swebench_prompts.py): the model
gets ONLY the problem statement + repo name ("checked out at /testbed") and
must explore, edit, and (optionally) run tests itself, mini-SWE-agent style.

Action protocol (one action per assistant turn):
    ACTION: bash            -> run one shell command in /testbed
    ACTION: edit            -> one or more SEARCH/REPLACE blocks (same
                               semantics as convert_sr_to_patch.py: exact
                               substring first, whitespace-normalized
                               line-sequence fallback, empty SEARCH = create)
    ACTION: done            -> finish

At loop end the model_patch is `git -C /testbed diff` (files the harness
created via empty-SEARCH edits are `git add -N`-staged first so they appear).
Predictions rows are {instance_id, model_name_or_path, model_patch}, directly
consumable by `swebench.harness.run_evaluation --predictions_path`.

BUILD-LEG CONTRACT (mirrors review_f1/harness.py):
  * Real transport = stdlib urllib against an OpenAI-compatible
    /v1/chat/completions (llama-server). Tests inject ReplayClient.
  * Real environment = `docker exec` into a SWE-bench instance container
    (repo at /testbed). Tests inject FakeEnv (in-memory dict filesystem +
    canned command outputs). Nothing in this file touches docker or the
    network unless main() wires the real implementations at run time.
  * Incremental persistence: per-instance trajectory JSONL written turn by
    turn; predictions.json rewritten (atomically) after every instance.
  * Results are labeled by ARM (model/quant label), never by role.
  * enable_thinking=False in the payload (Qwen3.x rule, chat-completions path);
    production-style sampling (temperature + fixed seed).

Run `--dry-run` to print the execution plan without contacting anything.
"""
from __future__ import annotations

import argparse
import difflib
import json
import posixpath
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TESTBED = "/testbed"
DEFAULT_MAX_TURNS = 30
DEFAULT_MAX_WALL_S = 1800.0
DEFAULT_CMD_TIMEOUT = 120
DEFAULT_MAX_TOKENS = 2048
OBS_TRUNCATE_CHARS = 4000
TIMEOUT_EXIT = 124

GIT_DIFF_CMD = f"git -C {TESTBED} diff"
GIT_ADD_N_PREFIX = f"git -C {TESTBED} add -N -- "

# SEARCH/REPLACE grammar + whitespace-normalized fallback: copied VERBATIM from
# artifacts/architect-code-eval-20260724/convert_sr_to_patch.py so the agentic
# and oracle rungs share one edit dialect. (Not imported: that module loads the
# 500-row dataset at import time from a path-relative location.)
SR = re.compile(r"<<<<<<<+\s*SEARCH\s*\n(.*?)\n?=======\s*\n(.*?)\n?>>>>>>>+\s*REPLACE\s*(\S*)",
                re.DOTALL)


def ws_norm_find(hay: str, needle: str) -> tuple[int, int] | None:
    """Find needle in hay comparing lines stripped of trailing ws; return char span."""
    h_lines = hay.split("\n"); n_lines = [l.rstrip() for l in needle.split("\n")]
    if not n_lines: return None
    stripped = [l.rstrip() for l in h_lines]
    for i in range(len(stripped) - len(n_lines) + 1):
        if stripped[i:i + len(n_lines)] == n_lines:
            start = sum(len(l) + 1 for l in h_lines[:i])
            length = sum(len(l) + 1 for l in h_lines[i:i + len(n_lines)]) - 1
            return start, start + length
    return None


def _timeout_msg(timeout: float) -> str:
    return f"[command timed out after {int(timeout)}s]"


def truncate_output(text: str, limit: int = OBS_TRUNCATE_CHARS) -> str:
    """Head+tail truncation: keep limit//2 chars from each end."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return (text[:half] + f"\n... [{len(text) - limit} chars truncated] ...\n"
            + text[-half:])


# --------------------------------------------------------------------------- #
# Model clients
# --------------------------------------------------------------------------- #
class ModelClient:
    """OpenAI-compatible /v1/chat/completions transport (stdlib urllib only).

    RUN-LEG ONLY: contacts a local llama-server. Never used by the tests.
    """

    def __init__(self, server_url: str, model: str, *, temperature: float = 0.6,
                 top_p: float = 0.95, top_k: int | None = 20, seed: int = 42,
                 timeout: int = 3600, enable_thinking: bool = False):
        self.url = server_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self.last_usage: dict | None = None

    def chat(self, messages: list[dict], max_tokens: int) -> str:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "max_tokens": max_tokens,
            "enable_thinking": self.enable_thinking,
        }
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=data, headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (local server)
            body = json.loads(resp.read().decode("utf-8"))
        self.last_usage = body.get("usage")
        return body["choices"][0]["message"].get("content") or ""


class ReplayClient:
    """Deterministic scripted client for tests. Records every call's messages
    (deep-copied at call time, so later history compaction can't rewrite what
    tests inspect)."""

    def __init__(self, responses: list[str], repeat_last: bool = False):
        self.responses = list(responses)
        self.repeat_last = repeat_last
        self.calls: list[list[dict]] = []
        self._i = 0

    def chat(self, messages: list[dict], max_tokens: int) -> str:
        self.calls.append(json.loads(json.dumps(messages)))
        if self._i >= len(self.responses):
            if self.repeat_last and self.responses:
                return self.responses[-1]
            raise AssertionError(
                f"ReplayClient exhausted after {len(self.responses)} responses")
        resp = self.responses[self._i]
        self._i += 1
        return resp


# --------------------------------------------------------------------------- #
# Environments
# --------------------------------------------------------------------------- #
class DockerEnv:
    """Instance environment backed by `docker exec` into a SWE-bench instance
    container (repo at /testbed).

    RUN-LEG ONLY: not exercised by the tests. Per-command timeout is enforced
    INSIDE the container via coreutils `timeout` (so the container-side process
    dies too), with an outer subprocess timeout as a safety net.
    """

    def __init__(self, container: str, docker_bin: str = "docker"):
        self.container = container
        self.docker_bin = docker_bin

    def _exec(self, cmd: str, timeout: float) -> tuple[int, str, str]:
        inner = f"timeout {int(timeout)}s bash -c {shlex.quote(cmd)}"
        argv = [self.docker_bin, "exec", "-w", TESTBED, self.container,
                "bash", "-c", inner]
        try:
            p = subprocess.run(argv, capture_output=True, text=True,
                               errors="replace", timeout=timeout + 15)
        except subprocess.TimeoutExpired:
            return TIMEOUT_EXIT, "", _timeout_msg(timeout)
        return p.returncode, p.stdout or "", p.stderr or ""

    def run(self, cmd: str, timeout: float) -> tuple[int, str]:
        """(exit_code, combined stdout+stderr) — the observation channel."""
        code, out, err = self._exec(cmd, timeout)
        combined = out + (("\n" + err) if err else "")
        if code == TIMEOUT_EXIT and _timeout_msg(timeout) not in combined:
            combined += ("\n" if combined else "") + _timeout_msg(timeout)
        return code, combined

    def run_stdout(self, cmd: str, timeout: float) -> tuple[int, str]:
        """(exit_code, stdout only) — used for patch extraction, where stderr
        noise (e.g. git CRLF warnings) must not pollute the diff."""
        code, out, _err = self._exec(cmd, timeout)
        return code, out

    def read_file(self, rel_path: str) -> str | None:
        argv = [self.docker_bin, "exec", self.container, "cat",
                posixpath.join(TESTBED, rel_path)]
        try:
            p = subprocess.run(argv, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            return None
        if p.returncode != 0:
            return None
        return p.stdout.decode("utf-8", errors="replace")

    def write_file(self, rel_path: str, content: str) -> bool:
        abs_path = posixpath.join(TESTBED, rel_path)
        parent = posixpath.dirname(abs_path) or TESTBED
        argv = [self.docker_bin, "exec", "-i", self.container, "bash", "-c",
                f"mkdir -p {shlex.quote(parent)} && cat > {shlex.quote(abs_path)}"]
        try:
            p = subprocess.run(argv, input=content.encode("utf-8"),
                               capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False
        return p.returncode == 0


class FakeEnv:
    """In-memory environment for tests: dict filesystem (repo-relative paths)
    + canned command outputs.

    canned: {exact command string: (exit_code, output)} or the sentinel string
    "TIMEOUT" (returns exit 124 + the same message DockerEnv would produce).
    The git stage/diff commands issued by extract_patch() are intercepted and
    answered from the dict filesystem (unified diff vs the construction-time
    baseline, a/<rel> b/<rel> labels like real `git -C /testbed diff`).
    """

    def __init__(self, files: dict[str, str] | None = None,
                 canned: dict[str, object] | None = None):
        self.baseline = {self._key(p): c for p, c in (files or {}).items()}
        self.files = dict(self.baseline)
        self.canned = dict(canned or {})
        self.calls: list[str] = []

    @staticmethod
    def _key(path: str) -> str:
        p = path.strip()
        if p.startswith(TESTBED + "/"):
            p = p[len(TESTBED) + 1:]
        return p.lstrip("/")

    def run(self, cmd: str, timeout: float) -> tuple[int, str]:
        self.calls.append(cmd)
        if cmd == GIT_DIFF_CMD:
            return 0, self.diff()
        if cmd.startswith(GIT_ADD_N_PREFIX):
            return 0, ""
        v = self.canned.get(cmd)
        if v is None:
            return 127, f"[FakeEnv] no canned output for: {cmd}"
        if v == "TIMEOUT":
            return TIMEOUT_EXIT, _timeout_msg(timeout)
        code, out = v  # type: ignore[misc]
        return int(code), str(out)

    def run_stdout(self, cmd: str, timeout: float) -> tuple[int, str]:
        return self.run(cmd, timeout)

    def read_file(self, rel_path: str) -> str | None:
        return self.files.get(self._key(rel_path))

    def write_file(self, rel_path: str, content: str) -> bool:
        self.files[self._key(rel_path)] = content
        return True

    def diff(self) -> str:
        parts = []
        for path in sorted(set(self.baseline) | set(self.files)):
            old = self.baseline.get(path, "")
            new = self.files.get(path, "")
            if old == new:
                continue
            parts.append("".join(difflib.unified_diff(
                old.splitlines(keepends=True), new.splitlines(keepends=True),
                fromfile=f"a/{path}", tofile=f"b/{path}")))
        return "".join(parts)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def build_system_prompt(repo: str, max_turns: int) -> str:
    return f"""You are an autonomous software engineer fixing a real bug in the {repo} repository, which is checked out at {TESTBED} inside a Linux container (a git work tree; your accumulated edits are extracted with `git diff` at the end).

You work in turns. EVERY reply must contain EXACTLY ONE action, in one of the three formats below. You may write brief reasoning before the ACTION line; everything after the ACTION line belongs to the action.

1. Run one shell command (explore, search, run tests):
ACTION: bash
<shell command>

2. Edit a file (SEARCH/REPLACE):
ACTION: edit
<<<<<<< SEARCH
[lines copied VERBATIM from the current file, preserving indentation]
=======
[replacement lines]
>>>>>>> REPLACE path/to/file.py

- SEARCH must match the current file content exactly (a whitespace-tolerant fallback exists, but copy verbatim).
- Keep SEARCH minimal: only the changing lines plus 2-3 anchor lines.
- To CREATE a new file: leave SEARCH empty and put the full content in REPLACE. Create files this way (not via bash) or they will be missing from the final diff.
- Several SEARCH/REPLACE blocks may follow a single ACTION: edit line.

3. Finish:
ACTION: done

Rules:
- One action per reply; put nothing after the action block.
- Command output is truncated to ~{OBS_TRUNCATE_CHARS} characters, so keep output small (grep -n, sed -n 'a,bp', head).
- Do NOT modify tests; fix the underlying bug with the MINIMAL change.
- Budget: {max_turns} turns total. Verify the fix by running the most relevant tests if feasible, then reply ACTION: done."""


def build_task_prompt(instance: dict) -> str:
    """No-oracle instance prompt: problem statement + repo name only."""
    return f"""The repository {instance['repo']} is checked out at {TESTBED}. Solve the following issue.

## Issue
{instance['problem_statement']}

Explore the repository, find the root cause, and make the minimal fix. Do not modify tests. When the fix is in place (ideally verified by running the relevant tests), reply with ACTION: done."""


NUDGE = """Your last reply contained no valid action. Reply again with EXACTLY ONE action, in one of these formats:
ACTION: bash
<command>
--- or ---
ACTION: edit
<<<<<<< SEARCH
[exact current lines]
=======
[replacement lines]
>>>>>>> REPLACE path/to/file.py
--- or ---
ACTION: done"""

WASTED_TURN_OBS = ("OBSERVATION: no valid action was found in your last two replies, "
                   "so this turn was wasted. Emit exactly one ACTION: bash / "
                   "ACTION: edit / ACTION: done next.")

ELIDED_OBS = "OBSERVATION: [an earlier observation was elided to fit the context budget]"


# --------------------------------------------------------------------------- #
# Action parsing
# --------------------------------------------------------------------------- #
# [ \t]* (not \s*) so the match can never swallow the newline and misread the
# NEXT line as a same-line remainder (e.g. a ``` fence under "ACTION: bash").
ACTION_RE = re.compile(r"^[ \t]*ACTION:[ \t]*(bash|edit|done)\b[ \t]*([^\n]*?)[ \t]*$",
                       re.MULTILINE | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:[\w+-]*)\n(.*?)```", re.DOTALL)


@dataclass
class Action:
    kind: str | None                 # "bash" | "edit" | "done" | None (malformed)
    command: str | None = None       # bash
    blocks: list[tuple[str, str, str]] = field(default_factory=list)  # edit
    error: str | None = None         # malformed reason


def parse_action(text: str) -> Action:
    matches = list(ACTION_RE.finditer(text or ""))
    if not matches:
        return Action(None, error="no ACTION line found")
    if len(matches) > 1:
        return Action(None, error="multiple ACTION lines; emit exactly one action per reply")
    m = matches[0]
    kind = m.group(1).lower()
    same_line = (m.group(2) or "").strip()
    body = text[m.end():]
    if kind == "done":
        return Action("done")
    if kind == "bash":
        fenced = _FENCE_RE.search(body)
        cmd = fenced.group(1).strip() if fenced else body.strip()
        if same_line:  # lenient: command started on the ACTION line itself
            cmd = (same_line + "\n" + cmd).strip()
        if not cmd:
            return Action(None, error="ACTION: bash with no command")
        return Action("bash", command=cmd)
    # kind == "edit"
    blocks = SR.findall((same_line + "\n" + body) if same_line else body)
    if not blocks:
        return Action(None, error="ACTION: edit with no parseable SEARCH/REPLACE block")
    return Action("edit", blocks=blocks)


# --------------------------------------------------------------------------- #
# Edit application (SR semantics shared with convert_sr_to_patch.py)
# --------------------------------------------------------------------------- #
def norm_rel_path(raw: str) -> str | None:
    """Normalize a REPLACE-line path to repo-relative, or None if unusable.

    Deliberately NOT convert_sr_to_patch.py's `lstrip("ab/")`, which is
    char-set stripping and mangles paths starting with 'a'/'b' (e.g.
    astropy/... -> stropy/...). We strip only a literal leading "a/" or "b/"
    diff prefix, "./", or the "/testbed/" root. Absolute paths outside
    /testbed are refused."""
    p = (raw or "").strip()
    if not p:
        return None
    if p.startswith(("a/", "b/")):
        p = p[2:]
    if p.startswith("./"):
        p = p[2:]
    if p.startswith("/"):
        if not p.startswith(TESTBED + "/"):
            return None
        p = p[len(TESTBED) + 1:]
    p = p.strip("/")
    if not p or ".." in p.split("/"):
        return None
    return p


def apply_one_block(env, rel_path: str, search: str, replace: str
                    ) -> tuple[bool, str, bool]:
    """Apply one SR block to the LIVE file via env. Returns
    (applied, how/why, created_new_file).

    Match order is identical to convert_sr_to_patch.apply_blocks: empty SEARCH
    = create/overwrite whole file; exact substring (first occurrence); then
    whitespace-normalized (rstrip) line-sequence match."""
    if search.strip() == "":
        existed = env.read_file(rel_path) is not None
        env.write_file(rel_path, replace)
        return True, ("file overwritten (empty SEARCH on existing file)"
                      if existed else "new file created"), not existed
    content = env.read_file(rel_path)
    if content is None:
        return False, ("file does not exist (to create a new file, use an "
                       "empty SEARCH section)"), False
    idx = content.find(search)
    if idx >= 0:
        env.write_file(rel_path, content[:idx] + replace + content[idx + len(search):])
        return True, "exact match", False
    span = ws_norm_find(content, search)
    if span:
        env.write_file(rel_path, content[:span[0]] + replace + content[span[1]:])
        return True, "whitespace-normalized match", False
    return False, "SEARCH block not found in the current file content", False


def apply_edit_blocks(env, blocks: list[tuple[str, str, str]]
                      ) -> tuple[str, int, int, list[str]]:
    """Apply all blocks of one edit action. Returns
    (observation_text, n_applied, n_failed, created_files)."""
    lines, created = [], []
    applied = failed = 0
    for i, (search, replace, raw_path) in enumerate(blocks, 1):
        rel = norm_rel_path(raw_path)
        if rel is None:
            failed += 1
            lines.append(f"- block {i}: FAILED — missing or invalid file path "
                         f"on the REPLACE line (got: {raw_path!r})")
            continue
        ok, how, new_file = apply_one_block(env, rel, search, replace)
        if ok:
            applied += 1
            if new_file:
                created.append(rel)
            lines.append(f"- block {i}: APPLIED to {rel} ({how})")
        else:
            failed += 1
            lines.append(f"- block {i}: FAILED on {rel} — {how}")
    obs = "Edit results:\n" + "\n".join(lines)
    if failed:
        obs += ("\nFor each FAILED block: re-read the current file content "
                "(e.g. ACTION: bash with sed -n 'START,ENDp' FILE) and retry "
                "with the SEARCH text copied exactly.")
    return obs, applied, failed, created


# --------------------------------------------------------------------------- #
# Patch extraction
# --------------------------------------------------------------------------- #
def extract_patch(env, created_files: list[str] | None = None,
                  timeout: float = 120) -> str:
    """model_patch = `git -C /testbed diff` via env. Files the harness itself
    created (empty-SEARCH edits) are intent-to-add staged first so the diff
    includes them; we deliberately do NOT `git add -N .` — instance images
    carry untracked build artifacts (*.egg-info etc.) that would poison the
    patch."""
    created = sorted(set(created_files or []))
    if created:
        env.run(GIT_ADD_N_PREFIX + " ".join(shlex.quote(p) for p in created),
                timeout)
    code, out = env.run_stdout(GIT_DIFF_CMD, timeout)
    return out if code == 0 else ""


# --------------------------------------------------------------------------- #
# History compaction
# --------------------------------------------------------------------------- #
def compact_history(messages: list[dict], max_chars: int, keep_recent: int) -> int:
    """If total content exceeds max_chars, elide the OLDEST user observations
    (never the system prompt, the task message, or the last keep_recent
    messages). Returns number of messages elided."""
    total = sum(len(m.get("content", "")) for m in messages)
    if total <= max_chars:
        return 0
    elided = 0
    cutoff = max(2, len(messages) - keep_recent)
    for i in range(2, cutoff):
        if total <= max_chars:
            break
        m = messages[i]
        if m.get("role") == "user" and not m.get("_elided") \
                and len(m.get("content", "")) > len(ELIDED_OBS):
            total -= len(m["content"]) - len(ELIDED_OBS)
            m["content"] = ELIDED_OBS
            m["_elided"] = True
            elided += 1
    return elided


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #
@dataclass
class AgentConfig:
    max_turns: int = DEFAULT_MAX_TURNS
    max_wall_s: float = DEFAULT_MAX_WALL_S
    cmd_timeout: float = DEFAULT_CMD_TIMEOUT
    max_tokens: int = DEFAULT_MAX_TOKENS
    obs_limit: int = OBS_TRUNCATE_CHARS
    max_history_chars: int = 150_000
    keep_recent: int = 6
    log_full_responses: bool = False


def _append_jsonl(path: Path | None, obj: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def run_instance(client, env, instance: dict, cfg: AgentConfig,
                 traj_path: Path | None = None, clock=time.monotonic) -> dict:
    """Drive one instance through the agent loop. Returns a summary dict
    including model_patch. Trajectory JSONL is appended turn-by-turn."""
    t0 = clock()
    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(instance["repo"], cfg.max_turns)},
        {"role": "user", "content": build_task_prompt(instance)},
    ]
    applied_total = failed_total = malformed_total = 0
    created_files: list[str] = []
    status = "turns_exhausted"
    turns_used = 0

    for turn in range(1, cfg.max_turns + 1):
        if clock() - t0 > cfg.max_wall_s:
            status = "wall_exhausted"
            break
        compact_history(messages, cfg.max_history_chars, cfg.keep_recent)

        resp = client.chat(messages, cfg.max_tokens)
        messages.append({"role": "assistant", "content": resp})
        act = parse_action(resp)
        nudged = False
        if act.kind is None:  # one-shot corrective nudge
            malformed_total += 1
            nudged = True
            messages.append({"role": "user", "content": NUDGE})
            resp = client.chat(messages, cfg.max_tokens)
            messages.append({"role": "assistant", "content": resp})
            act = parse_action(resp)

        turns_used = turn
        rec: dict = {"turn": turn, "nudged": nudged}
        if cfg.log_full_responses:
            rec["response"] = resp

        if act.kind is None:  # second failure in a row: wasted turn
            malformed_total += 1
            obs = WASTED_TURN_OBS
            rec.update({"action": "malformed", "command": None, "exit": None,
                        "error": act.error})
        elif act.kind == "done":
            status = "done"
            rec.update({"action": "done", "command": None, "exit": None,
                        "obs_len": 0, "wall": round(clock() - t0, 2)})
            _append_jsonl(traj_path, rec)
            break
        elif act.kind == "bash":
            code, out = env.run(act.command, cfg.cmd_timeout)
            obs = f"OBSERVATION (exit code {code}):\n{out}"
            rec.update({"action": "bash", "command": act.command[:400],
                        "exit": code})
        else:  # edit
            obs, n_ok, n_fail, created = apply_edit_blocks(env, act.blocks)
            applied_total += n_ok
            failed_total += n_fail
            created_files.extend(created)
            obs = "OBSERVATION:\n" + obs
            rec.update({"action": "edit",
                        "command": ";".join(norm_rel_path(b[2]) or "?" for b in act.blocks)[:400],
                        "exit": 0 if n_fail == 0 else 1,
                        "edits_applied": n_ok, "edits_failed": n_fail})

        obs = truncate_output(obs, cfg.obs_limit)
        messages.append({"role": "user", "content": obs})
        rec.update({"obs_len": len(obs), "wall": round(clock() - t0, 2)})
        _append_jsonl(traj_path, rec)

    patch = extract_patch(env, created_files, cfg.cmd_timeout)
    summary = {
        "instance_id": instance.get("instance_id", "?"),
        "status": status,
        "turns_used": turns_used,
        "edits_applied": applied_total,
        "edits_failed": failed_total,
        "malformed": malformed_total,
        "patch_chars": len(patch),
        "wall_s": round(clock() - t0, 2),
    }
    _append_jsonl(traj_path, {"summary": summary})
    return {**summary, "model_patch": patch}


# --------------------------------------------------------------------------- #
# Predictions output
# --------------------------------------------------------------------------- #
def write_predictions(path: Path, rows: list[dict]) -> None:
    """Atomic rewrite; rows are swebench.harness.run_evaluation-compatible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, indent=1))
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# CLI (run leg — wires DockerEnv + ModelClient; never reached by tests)
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Agentic (multi-turn, no-oracle) SWE-bench harness")
    p.add_argument("--dataset", required=True,
                   help="swebench_verified.json (instance rows)")
    p.add_argument("--instance-id", action="append", default=[],
                   help="instance to run (repeatable)")
    p.add_argument("--instances-file", default=None,
                   help="file with one instance_id per line (alternative to --instance-id)")
    p.add_argument("--container", default=None,
                   help="docker container id/name (single-instance smoke)")
    p.add_argument("--container-map", default=None,
                   help="JSON file {instance_id: container id/name} for batches")
    p.add_argument("--server-url", default="http://127.0.0.1:18072",
                   help="llama-server base URL (OpenAI-compatible)")
    p.add_argument("--model", default="local", help="model name for the payload")
    p.add_argument("--arm", required=True,
                   help="model_name_or_path label for predictions (model/quant label, never a role)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p.add_argument("--max-wall-s", type=float, default=DEFAULT_MAX_WALL_S)
    p.add_argument("--cmd-timeout", type=float, default=DEFAULT_CMD_TIMEOUT)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--request-timeout", type=int, default=3600)
    p.add_argument("--log-full-responses", action="store_true",
                   help="store full assistant text in trajectory records")
    p.add_argument("--no-resume", action="store_true",
                   help="rerun instances already present in predictions.json")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan; contact no server and no docker")
    return p


def _resolve_ids(args) -> list[str]:
    ids = list(args.instance_id)
    if args.instances_file:
        ids += [l.strip() for l in Path(args.instances_file).read_text().splitlines()
                if l.strip()]
    return ids


def _resolve_containers(args, ids: list[str]) -> dict[str, str]:
    if args.container_map:
        cmap = json.loads(Path(args.container_map).read_text())
        missing = [i for i in ids if i not in cmap]
        if missing:
            raise SystemExit(f"--container-map missing entries for: {missing}")
        return {i: cmap[i] for i in ids}
    if args.container:
        if len(ids) != 1:
            raise SystemExit("--container is single-instance only; use --container-map")
        return {ids[0]: args.container}
    raise SystemExit("need --container (single instance) or --container-map")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ids = _resolve_ids(args)
    if not ids:
        raise SystemExit("no instances given (--instance-id / --instances-file)")
    rows = {r["instance_id"]: r for r in json.load(open(args.dataset))}
    unknown = [i for i in ids if i not in rows]
    if unknown:
        raise SystemExit(f"instance ids not in dataset: {unknown}")

    out_dir = Path(args.out_dir)
    preds_path = out_dir / "predictions.json"
    cfg = AgentConfig(max_turns=args.max_turns, max_wall_s=args.max_wall_s,
                      cmd_timeout=args.cmd_timeout, max_tokens=args.max_tokens,
                      log_full_responses=args.log_full_responses)

    if args.dry_run:
        cmap = (json.loads(Path(args.container_map).read_text())
                if args.container_map else
                {ids[0]: args.container} if args.container else {})
        plan = {
            "arm": args.arm, "server_url": args.server_url, "model": args.model,
            "endpoint": "/v1/chat/completions", "enable_thinking": False,
            "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                         "top_k": args.top_k, "seed": args.seed},
            "budgets": {"max_turns": cfg.max_turns, "max_wall_s": cfg.max_wall_s,
                        "cmd_timeout": cfg.cmd_timeout, "max_tokens": cfg.max_tokens},
            "instances": ids,
            "containers": {i: cmap.get(i, "<unset>") for i in ids},
            "predictions": str(preds_path),
            "trajectories": str(out_dir / "trajectories"),
        }
        print("=== agentic_swe_harness DRY-RUN (no server, no docker) ===")
        print(json.dumps(plan, indent=2))
        return 0

    cmap = _resolve_containers(args, ids)
    existing: list[dict] = []
    if preds_path.exists() and not args.no_resume:
        try:
            existing = json.loads(preds_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = []
    by_id = {r["instance_id"]: r for r in existing}

    client = ModelClient(args.server_url, args.model,
                         temperature=args.temperature, top_p=args.top_p,
                         top_k=args.top_k, seed=args.seed,
                         timeout=args.request_timeout, enable_thinking=False)
    for iid in ids:
        if iid in by_id and not args.no_resume:
            print(f"{iid}: already in predictions.json — skipped (resume)")
            continue
        env = DockerEnv(cmap[iid])
        traj = out_dir / "trajectories" / f"{iid}.jsonl"
        res = run_instance(client, env, rows[iid], cfg, traj_path=traj)
        by_id[iid] = {"instance_id": iid, "model_name_or_path": args.arm,
                      "model_patch": res["model_patch"]}
        write_predictions(preds_path, list(by_id.values()))  # after EVERY instance
        print(f"{iid}: {res['status']} turns={res['turns_used']} "
              f"edits={res['edits_applied']}/{res['edits_applied'] + res['edits_failed']} "
              f"patch_chars={res['patch_chars']}")
    print(f"predictions -> {preds_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
