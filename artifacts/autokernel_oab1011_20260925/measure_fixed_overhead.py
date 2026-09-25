#!/usr/bin/env python3
"""Static measurement of the planner's fixed per-call overhead and what OAB-10/11 remove.

No inference: every count is the Qwen3.8-27B vocab via `llama-tokenize` (vocab-only,
pinned to CPUs 72-79), on text reconstructed exactly as opencode 1.18.31 assembles it.

1. TOTAL fixed overhead, from real calls: llama-server's first-step prompt tokens
   (`context_first_tokens` = input + cache.read of step 1, OAB-9 metrics rows) minus the
   tokenized prompt the planner sent. Two independent arms (inline, variable) must agree.
2. COMPONENTS, reconstructed from the installed binary's own code/strings:
   * the lane AGENTS.md block, exactly as `Instruction.system` renders it
     ("Instructions from: <path>\\n<text>");
   * the skill catalog, exactly as `SystemPrompt.skills` renders it (header, then
     `Skill.fmt(list, {verbose: true})`: <available_skills> with name, description,
     html-escaped location; the built-in `customize-opencode` included);
   * tool schemas that the knobs drop (skill, task, todowrite, webfetch; edit+write for
     the read-only roles). Descriptions are the binary's literal strings; the parameter
     JSON is reconstructed from the binary's schema annotations, so these rows are
     APPROXIMATE (+-10%), marked `approx`.
   * what the knobs ADD: the author's style note and the lane-guard prompt block.

usage: python3 measure_fixed_overhead.py [--out fixed_overhead.json]
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "scripts/kernel_rnd"), str(REPO)]
from autokernel.loop import actor_opencode_config as aoc, actors  # noqa: E402
from autokernel.loop import test_actor_context as fx  # noqa: E402

KERNEL = Path("/mnt/raid0/llm/kernels/production/cpu")
VOCAB = Path("/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf")
OPENCODE_BIN = Path("/usr/local/share/npm-global/lib/node_modules/opencode-ai/bin/opencode.exe")
AB = Path("/mnt/raid0/llm/tmp/ak-ctx-ab")
LANE = Path("/mnt/raid0/llm/tmp/ak-seat-ab/lane")     # the OAB-9 lane (link target)
LANE_LINK = AB / "lane"                                # what opencode was given as --dir
BUILD_DIR = "/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/build-cpu"
SKILL_ROOTS = [Path.home() / ".claude" / "skills", Path.home() / ".agents" / "skills"]


def tokens(text: str) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write(text)
        path = fh.name
    try:
        out = subprocess.run(
            ["taskset", "-c", "72-79", str(KERNEL / "llama-tokenize"), "-m", str(VOCAB),
             "-f", path, "--show-count", "--ids", "--log-disable", "--no-escape",
             "--no-bos"],
            capture_output=True, text=True, check=True,
            env={**os.environ, "LD_LIBRARY_PATH": str(KERNEL)}).stdout
    finally:
        os.unlink(path)
    line = [ln for ln in out.splitlines() if ln.startswith("Total number of tokens:")][-1]
    return int(line.split(":")[1])


def js_literal(blob: str, anchor: str) -> str:
    """The JS string literal that STARTS with `anchor` in the bundled source, unescaped."""
    i = blob.index(anchor)
    quote = blob[i - 1]
    assert quote in "`'\"", quote
    j, out = i, []
    while True:
        ch = blob[j]
        if ch == "\\":
            nxt = blob[j + 1]
            out.append({"n": "\n", "t": "\t"}.get(nxt, nxt))
            j += 2
            continue
        if ch == quote:
            return "".join(out)
        out.append(ch)
        j += 1


def skill_catalog(builtin_description: str) -> tuple[str, list[dict]]:
    skills = [{"name": "customize-opencode", "description": builtin_description,
               "location": "<built-in>"}]
    for root in SKILL_ROOTS:
        for path in sorted(glob.glob(str(root / "**" / "SKILL.md"), recursive=True)):
            text = Path(path).read_text(encoding="utf-8")
            if not text.startswith("---"):
                continue
            front = yaml.safe_load(text.split("---", 2)[1]) or {}
            if isinstance(front.get("name"), str):
                skills.append({"name": front["name"], "description": front.get("description"),
                               "location": path})
    rows = ["<available_skills>"]
    for s in sorted((s for s in skills if s["description"] is not None),
                    key=lambda s: s["name"]):
        rows += ["  <skill>", f"    <name>{s['name']}</name>",
                 f"    <description>{s['description']}</description>",
                 f"    <location>{html.escape(s['location'], quote=True).replace('&#x27;', '&#39;')}</location>",
                 "  </skill>"]
    rows.append("</available_skills>")
    body = "\n".join(["Skills provide specialized instructions and workflows for specific tasks.",
                      "Use the skill tool to load a skill when a task matches its description.",
                      "\n".join(rows)])
    return body, skills


def tool_json(name: str, description: str, properties: dict, required: list[str]) -> str:
    return json.dumps({"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required,
                       "additionalProperties": False}}}, ensure_ascii=False)


def s(desc: str, **kw) -> dict:
    return {"type": "string", "description": desc, **kw}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "fixed_overhead.json")
    args = parser.parse_args()
    blob = OPENCODE_BIN.read_bytes().decode("utf-8", errors="replace")

    # ---- 1. total fixed overhead from the OAB-9 calls -------------------------------
    inline_prompt = fx._real_prompt()
    bundle = Path(json.loads((AB / "result-p1-variable.json").read_text())["bundle"]["dir"])
    variable_prompt = actors._HYPOTHESIS_TASK.format(
        context=(bundle / "INDEX.md").read_text(encoding="utf-8"), **fx.CPU_FORMAT)
    calls = {}
    for arm, prompt in (("inline", inline_prompt), ("variable", variable_prompt)):
        firsts = [json.loads((AB / f"result-p{p}-{arm}.json").read_text())["metrics"]
                  ["context_first_tokens"] for p in (1, 2)]
        n = tokens(prompt)
        calls[arm] = {"prompt_chars": len(prompt), "prompt_tokens": n,
                      "first_step_context_tokens": firsts,
                      "fixed_overhead_tokens": [f - n for f in firsts]}

    # ---- 2. components ---------------------------------------------------------------
    agents_md = (LANE / "AGENTS.md").read_text(encoding="utf-8")
    agents_block = f"Instructions from: {LANE_LINK / 'AGENTS.md'}\n{agents_md}"
    builtin = js_literal(blob, "Use ONLY when the user is editing or creating opencode")
    catalog, skills = skill_catalog(builtin)
    skill_desc = js_literal(blob, "Load a specialized skill when the task at hand")
    task_desc = js_literal(blob, "Launch a new agent to handle complex, multistep tasks")
    todo_desc = js_literal(blob, "Create and maintain a structured task list")
    fetch_desc = js_literal(blob, "- Fetches content from a specified URL")
    edit_desc = js_literal(blob, "Performs exact string replacements in files.")
    write_desc = js_literal(blob, "Writes a file to the local filesystem.")
    general = js_literal(blob, "General-purpose agent for researching complex questions")
    explore = js_literal(blob, "Fast agent specialized for exploring codebases.")
    task_full = (task_desc + "\n\nAvailable agent types and the tools they have access to:\n"
                 f"- general: {general}\n- explore: {explore}")
    todo_item = {"type": "object", "properties": {
        "content": s("Brief description of the task"),
        "status": s("Current status of the task: pending, in_progress, completed, cancelled"),
        "priority": s("Priority level of the task: high, medium, low")},
        "required": ["content", "status", "priority"]}
    tools = {
        "skill": tool_json("skill", skill_desc,
                           {"name": s("The name of the skill from available_skills")}, ["name"]),
        "task": tool_json("task", task_full, {
            "description": s("A short (3-5 words) description of the task"),
            "prompt": s("The task for the agent to perform"),
            "subagent_type": s("The type of specialized agent to use for this task"),
            "task_id": s("Resume a previous task by passing its task_id")},
            ["description", "prompt", "subagent_type"]),
        "todowrite": tool_json("todowrite", todo_desc, {
            "todos": {"type": "array", "items": todo_item, "description": "The updated todo list"}},
            ["todos"]),
        "webfetch": tool_json("webfetch", fetch_desc, {
            "url": s("The URL to fetch content from"),
            "format": s("The format to return the content in (text, markdown, or html). "
                        "Defaults to markdown.", enum=["text", "markdown", "html"]),
            "timeout": {"type": "number", "description": "Optional timeout in seconds (max 120)"}},
            ["url", "format"]),
        "edit": tool_json("edit", edit_desc, {
            "filePath": s("The absolute path to the file to modify"),
            "oldString": s("The text to replace"),
            "newString": s("The text to replace it with (must be different from oldString)"),
            "replaceAll": {"type": "boolean",
                           "description": "Replace all occurrences of oldString (default false)"}},
            ["filePath", "oldString", "newString"]),
        "write": tool_json("write", write_desc, {
            "content": s("The content to write to the file"),
            "filePath": s("The absolute path to the file (must be absolute, not relative)")},
            ["content", "filePath"]),
    }
    tool_tokens = {name: tokens(text) for name, text in tools.items()}
    guard_block = actors._lane_block(
        "planner", LANE_LINK, {"target": {"recipe": {"build_dir": BUILD_DIR}}})

    components = {
        "agents_md_block": {"chars": len(agents_block), "tokens": tokens(agents_block),
                            "exact": True, "knob": "trim_instructions"},
        "skill_catalog": {"chars": len(catalog), "tokens": tokens(catalog), "exact": True,
                          "skills": len(skills), "knob": "trim_instructions"},
        **{f"tool_{name}": {"chars": len(tools[name]), "tokens": tool_tokens[name],
                            "exact": False, "approx": "+-10% (parameter JSON reconstructed)",
                            "knob": ("trim_instructions" if name == "skill" else
                                     "lane_guard (planner/critic)" if name in ("edit", "write")
                                     else "trim_tools")}
           for name in tools},
    }
    added = {"author_style_note": {"chars": len(aoc.AUTHOR_STYLE_NOTE) + len(
                 "Instructions from: /x/actor-opencode-plain-author.instructions.md\n"),
                 "tokens": tokens(f"Instructions from: {AB}/actor-opencode-plain-author."
                                  f"instructions.md\n{aoc.AUTHOR_STYLE_NOTE}\n"),
                 "role": "author", "knob": "trim_instructions"},
             "lane_guard_prompt_block": {"chars": len(guard_block) + 2,
                                         "tokens": tokens(guard_block + "\n\n"),
                                         "knob": "lane_guard"}}

    def removed(role: str, knobs: set[str]) -> int:
        total = 0
        for key, row in components.items():
            knob = row["knob"]
            if knob.startswith("lane_guard"):
                if "lane_guard" in knobs and role in ("planner", "critic"):
                    total += row["tokens"]
            elif knob in knobs:
                total += row["tokens"]
        return total

    all_knobs = {"trim_instructions", "trim_tools", "lane_guard"}
    base = calls["inline"]["first_step_context_tokens"][0]
    per_role = {}
    for role in ("planner", "author", "critic"):
        cut = removed(role, all_knobs)
        add = added["lane_guard_prompt_block"]["tokens"] + (
            added["author_style_note"]["tokens"] if role == "author" else 0)
        per_role[role] = {"removed_tokens": cut, "added_tokens": add, "net_tokens": cut - add}
    planner_instr = removed("planner", {"trim_instructions"})
    result = {
        "schema": "epyc.autokernel.oab1011_fixed_overhead.v1",
        "tokenizer": {"binary": str(KERNEL / "llama-tokenize"), "vocab": str(VOCAB),
                      "flags": "--no-escape --no-bos, vocab-only, taskset -c 72-79"},
        "opencode": {"binary": str(OPENCODE_BIN), "version": "1.18.31"},
        "oab9_calls": calls,
        "components": components,
        "added_by_knobs": added,
        "per_role_all_knobs_on": per_role,
        "planner_trim_instructions_only_tokens": planner_instr,
        "planner_first_step_projection": {
            "oab9_inline_first_step": base,
            "after_all_knobs": base - per_role["planner"]["net_tokens"],
            "fixed_overhead_before": calls["inline"]["fixed_overhead_tokens"][0],
            "fixed_overhead_after": calls["inline"]["fixed_overhead_tokens"][0]
                                   - per_role["planner"]["removed_tokens"],
        },
        "notes": [
            "context_first_tokens is llama-server's own count of step 1 (input + cache.read); "
            "the difference from the tokenized prompt includes ~5 chat-template tokens.",
            "Every call re-bills the fixed part: the saving recurs on each of a call's steps "
            "(23-24 steps per OAB-9 inline call), mostly as prefix-cache reads.",
        ],
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
