"""Code extraction and error classification utilities."""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from src.escalation import ErrorCategory

_log = logging.getLogger(__name__)

# Regex matching OpenAI-format tool_call JSON objects.  The model sometimes
# emits these instead of the REPL ``CALL()`` syntax because Qwen3-Coder's
# instruct training activates the chat-completion tool-calling format even
# in raw-completion mode.
_TOOL_CALL_RE = re.compile(
    r'"function"\s*:\s*\{\s*"(?:arguments|name)"\s*:'
    r'|'
    r'"id"\s*:\s*"call_[^"]+"\s*,\s*"function"\s*:',
)

_TAGGED_JSON_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)

# TU-TC-1: tool-call JSON that fails ``json.loads`` is repaired deterministically
# or refused visibly -- never executed with ``{}`` arguments.  No telemetry sink
# exists for the parser (no prometheus_client in src/), so outcomes land on this
# in-process counter plus a structured ``tool_call_json_repair outcome=...`` log.
_TRAILING_TAGS_RE = re.compile(r"(?:\s*</?[A-Za-z_][\w:.=-]*>)+\s*$")
_CALL_NAME_RE = re.compile(r'"name"\s*:\s*"([^"\\]+)"')
_RAW_ECHO_LIMIT = 200
_CLOSERS = {"{": "}", "[": "]"}
_repair_lock = threading.Lock()
TOOL_CALL_JSON_REPAIR_COUNTS: dict[str, int] = {"repaired": 0, "unrecoverable": 0}


class _RejectedToolCall(NamedTuple):
    """A tool call whose JSON could not be repaired; rendered as a visible refusal."""

    name: str
    raw: str


def _record_repair_outcome(outcome: str, raw: str) -> None:
    with _repair_lock:
        TOOL_CALL_JSON_REPAIR_COUNTS[outcome] += 1
    _log.info(
        "tool_call_json_repair outcome=%s bytes=%d raw=%r",
        outcome, len(raw), raw[:_RAW_ECHO_LIMIT],
    )


def _repair_json_text(raw: str) -> str | None:
    """Return JSON text that ``json.loads`` accepts, or ``None``.

    Valid JSON is returned unchanged (the same string).  Only when parsing
    fails: strip leaked trailing XML-ish tags (``</parameter>``,
    ``</tool_call>``...), close an unterminated string, and balance
    unclosed/over-closed/mismatched brackets with a structural scan.
    """
    try:
        json.loads(raw)
        return raw
    except (json.JSONDecodeError, ValueError):
        pass
    text = _TRAILING_TAGS_RE.sub("", raw).strip()
    out: list[str] = []
    stack: list[str] = []
    in_str = False
    escape = False
    for c in text:
        if in_str:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            out.append(c)
        elif c in _CLOSERS:
            stack.append(c)
            out.append(c)
        elif c in ("}", "]"):
            if not stack:
                continue  # over-closed: drop the stray closer
            # A mismatched closer is taken as the closer of the open container.
            out.append(_CLOSERS[stack.pop()])
        else:
            out.append(c)
    if in_str:
        if escape:
            out.pop()  # a dangling backslash would escape the closing quote
        out.append('"')
    repaired = "".join(out).rstrip()
    if stack:
        repaired = repaired.rstrip(",").rstrip()
        repaired += "".join(_CLOSERS[o] for o in reversed(stack))
    try:
        json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        return None
    return repaired


def _loads_with_repair(raw: str) -> tuple[bool, object]:
    """``json.loads`` gated repair; returns ``(ok, value)`` and records telemetry."""
    fixed = _repair_json_text(raw)
    if fixed is None:
        _record_repair_outcome("unrecoverable", raw)
        return False, None
    if fixed is not raw:
        _record_repair_outcome("repaired", raw)
    return True, json.loads(fixed)


def _rejection_from_raw(raw: str) -> _RejectedToolCall:
    m = _CALL_NAME_RE.search(raw)
    return _RejectedToolCall(m.group(1) if m else "<unknown>", raw)


def _extract_json_arrays(
    text: str, rejected: list[_RejectedToolCall] | None = None
) -> list[list[dict]]:
    """Extract JSON arrays from text by scanning for balanced brackets.

    Tool-call-shaped arrays that fail ``json.loads`` go through the gated
    repair; unrecoverable ones are appended to *rejected* when given.
    """
    arrays: list[list[dict]] = []
    i = 0
    while i < len(text):
        if text[i] == '[':
            # Scan forward tracking depth and string literals
            depth = 0
            in_str = False
            escape = False
            j = i
            while j < len(text):
                c = text[j]
                if escape:
                    escape = False
                elif c == '\\' and in_str:
                    escape = True
                elif c == '"' and not escape:
                    in_str = not in_str
                elif not in_str:
                    if c == '[':
                        depth += 1
                    elif c == ']':
                        depth -= 1
                        if depth == 0:
                            chunk = text[i:j + 1]
                            try:
                                arr = json.loads(chunk)
                            except (json.JSONDecodeError, ValueError):
                                arr = None
                                if _TOOL_CALL_RE.search(chunk):
                                    ok, arr = _loads_with_repair(chunk)
                                    if not ok and rejected is not None:
                                        rejected.append(_rejection_from_raw(chunk))
                            if isinstance(arr, list):
                                arrays.append(arr)
                            i = j + 1
                            break
                j += 1
            else:
                i += 1
        else:
            i += 1
    return arrays


def _render_call_code(calls: list[tuple[str, dict] | _RejectedToolCall]) -> str:
    """Render parsed tool calls as executable REPL ``CALL()`` statements.

    A ``_RejectedToolCall`` is rendered as an ``[ERROR: ...]`` result string
    (the REPL tools' error convention) so the model sees the refusal next turn.
    """
    lines: list[str] = []
    for i, call in enumerate(calls):
        var = f"result_{i}" if len(calls) > 1 else "result"
        if isinstance(call, _RejectedToolCall):
            raw = call.raw
            if len(raw) > _RAW_ECHO_LIMIT:
                raw = raw[:_RAW_ECHO_LIMIT] + "...[truncated]"
            msg = (
                f"[ERROR: tool call {call.name!r} NOT executed: its JSON arguments "
                f"are malformed and could not be repaired. Raw: {raw} -- re-emit "
                f"the call with valid JSON arguments.]"
            )
            lines.append(f"{var} = {json.dumps(msg)}")
            continue
        name, kwargs = call
        kw_parts = ", ".join(f'{k}={json.dumps(v)}' for k, v in kwargs.items())
        lines.append(f'{var} = CALL("{name}", {kw_parts})')

    # Print results so the REPL captures output for the next turn.
    if len(calls) == 1:
        lines.append("print(result)")
    else:
        for i in range(len(calls)):
            lines.append(f"print(result_{i})")
    return "\n".join(lines)


def _deduplicate_tool_calls(
    items: list[dict],
) -> list[tuple[str, dict] | _RejectedToolCall]:
    """Normalize direct and OpenAI-style JSON tool-call objects.

    An unparseable ``arguments`` string is repaired or becomes a
    ``_RejectedToolCall`` -- it never degrades to ``{}`` (TU-TC-1).
    """
    calls_seen: list[tuple[str, dict] | _RejectedToolCall] = []
    seen_keys: set[tuple[str, ...]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        # Jackrong v2/Coder templates use the direct {name, arguments} object;
        # OpenAI completions wrap the same fields under ``function``.
        func = item.get("function") or item
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if not isinstance(name, str) or not name:
            continue
        raw_args = func.get("arguments", {})
        if isinstance(raw_args, str):
            if not raw_args.strip():
                args = {}  # an empty arguments string is a no-argument call
            else:
                ok, args = _loads_with_repair(raw_args)
                if not ok:
                    rejected = _RejectedToolCall(name, raw_args)
                    if ("rejected", name, raw_args) not in seen_keys:
                        seen_keys.add(("rejected", name, raw_args))
                        calls_seen.append(rejected)
                    continue
        else:
            args = raw_args
        if not isinstance(args, dict):
            args = {}
        # JSON serialization gives a stable, hashable key even when an
        # argument value is itself a list or object.
        dedup_key = (name, json.dumps(args, sort_keys=True, separators=(",", ":")))
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        calls_seen.append((name, args))
    return calls_seen


def translate_openai_tool_calls(text: str) -> str | None:
    """Translate recognized JSON tool-call wire formats to REPL ``CALL()`` code.

    When a model emits ``[{"id":"call_...","function":{"name":"web_search",
    "arguments":"{\"query\":\"...\"}"},"type":"function"}]`` instead of
    ``CALL("web_search", query="...")``, the REPL cannot execute it.

    This function extracts the *unique* tool calls, deduplicates them, and
    returns equivalent Python code using ``CALL()`` syntax.  Returns ``None``
    if no tool_call JSON is detected.
    """
    # Jackrong v2 and Coder place a direct JSON object inside a tool_call tag:
    # <tool_call>{"name":"web_search","arguments":{"query":"..."}}</tool_call>.
    # Parse each tag independently: templates differ per model, but the wire
    # contract is pinned by these fixtures rather than inferred from Qwen XML.
    tagged_items: list[dict] = []
    tagged_rejected: list[_RejectedToolCall] = []
    for match in _TAGGED_JSON_TOOL_CALL_RE.finditer(text):
        body = match.group(1)
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            # Only JSON-shaped bodies are ours to repair; other tagged wire
            # formats (e.g. Qwen XML) keep falling through as before.
            if not body.lstrip().startswith(("{", "[")):
                continue
            ok, payload = _loads_with_repair(body)
            if not ok:
                tagged_rejected.append(_rejection_from_raw(body))
                continue
        if isinstance(payload, dict):
            tagged_items.append(payload)
        elif isinstance(payload, list):
            tagged_items.extend(item for item in payload if isinstance(item, dict))
    tagged_calls = _deduplicate_tool_calls(tagged_items) + tagged_rejected
    if tagged_calls:
        _log.info(
            "Translated %d tagged JSON tool_call(s) to CALL() syntax: %s",
            len(tagged_calls),
            [call[0] for call in tagged_calls],
        )
        return _render_call_code(tagged_calls)

    if not _TOOL_CALL_RE.search(text):
        return None

    # Extract all JSON arrays that look like tool_call lists.
    items: list[dict] = []
    array_rejected: list[_RejectedToolCall] = []

    # Find JSON array boundaries robustly.  The model emits space-separated
    # ``[{...}] [{...}]`` blocks; simple regex can't handle nested braces
    # in the "arguments" field, so we scan for ``[`` and find the matching
    # ``]`` by tracking brace/bracket depth.
    for arr in _extract_json_arrays(text, array_rejected):
        for item in arr:
            if isinstance(item, dict):
                items.append(item)

    calls_seen = _deduplicate_tool_calls(items) + array_rejected

    if not calls_seen:
        return None

    _log.info(
        "Translated %d OpenAI-format tool_call(s) to CALL() syntax: %s",
        len(calls_seen),
        [c[0] for c in calls_seen],
    )

    # Do NOT auto-wrap with FINAL(): the model inspects the tool result on the
    # following turn.
    return _render_call_code(calls_seen)


def _strip_import_lines(code: str) -> str:
    """Strip top-level import/from lines since safe modules are pre-loaded in REPL globals.

    Models frequently generate 'import math' or 'import os' even when told not to.
    Safe modules (math, json, re, collections, numpy, scipy, etc.) are pre-loaded
    in _build_globals(); unsafe modules would be blocked by _safe_import() anyway.

    Skips lines inside triple-quoted strings to avoid corrupting embedded code
    (e.g., USACO solutions wrapped in solution = \"\"\"import sys...\"\"\" ).
    """
    lines = code.split("\n")
    filtered = []
    in_triple_quote = False
    triple_char = None
    for line in lines:
        # Track triple-quoted string boundaries
        stripped = line.strip()
        if not in_triple_quote:
            # Check if this line opens a triple-quoted string
            for tq in ('"""', "'''"):
                count = stripped.count(tq)
                if count % 2 == 1:  # Odd number = we entered/exited
                    in_triple_quote = True
                    triple_char = tq
                    break
            # Only strip imports at top level (outside strings)
            if not in_triple_quote and (
                stripped.startswith("import ") or stripped.startswith("from ")
            ):
                continue
        else:
            # Inside a triple-quoted string — check if it closes
            if triple_char and triple_char in stripped:
                count = stripped.count(triple_char)
                if count % 2 == 1:
                    in_triple_quote = False
                    triple_char = None
        filtered.append(line)
    return "\n".join(filtered).strip()


def _is_valid_python(code: str) -> bool:
    """Check if *code* parses as valid Python (syntax only)."""
    import ast

    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_code_from_response(response: str) -> str:
    """Extract Python code from an LLM response.

    Handles responses that may be wrapped in markdown code blocks
    or contain explanatory text. Also strips import lines since
    all needed modules are pre-loaded in the REPL globals.
    """
    response = response.strip()

    # Strip <end_prompt> tokens (Gemma-3/4 emits before thinking tags)
    response = re.sub(r'<end_prompt>', '', response)

    # Strip thinking channel tags (Gemma-4 uses <|channel>thought\n<channel|>)
    # Remove ^ anchor so it also matches when <end_prompt> consumed the line start
    response = re.sub(r'<\|channel>thought\n<channel\|>', '', response, flags=re.MULTILINE)

    # Strip Qwen3 thinking tags (</thinking> or </anthinking> markers)
    response = re.sub(r'</(?:an)?thinking>\s*\n?', '', response, flags=re.MULTILINE)

    # Intercept Gemma-4 <|tool_call>...<tool_call|> format and translate to TOOL() syntax.
    # Format: <|tool_call>call:tool_use:tool_name{json_args}<tool_call|>
    gemma_tool_calls = re.findall(r'<\|tool_call>(.*?)<tool_call\|>', response)
    if gemma_tool_calls:
        translated = []
        for tc in gemma_tool_calls:
            # Parse: call:tool_use:get_eval_secret{name: 'alpha'}
            m = re.match(r'call:tool_use:(\w+)\s*\{(.*)\}', tc.strip(), re.DOTALL)
            if m:
                tool_name = m.group(1)
                args_str = m.group(2)
                # Convert JSON-like args to Python kwargs
                kwargs = []
                for kv in re.findall(r"(\w+)\s*:\s*'([^']*)'", args_str):
                    kwargs.append(f'{kv[0]}="{kv[1]}"')
                for kv in re.findall(r'(\w+)\s*:\s*"([^"]*)"', args_str):
                    kwargs.append(f'{kv[0]}="{kv[1]}"')
                for kv in re.findall(r'(\w+)\s*:\s*(\d+)', args_str):
                    kwargs.append(f'{kv[0]}={kv[1]}')
                if kwargs:
                    translated.append(f'result = TOOL("{tool_name}", {", ".join(kwargs)})')
                    translated.append('FINAL(result)')
        if translated:
            response = "\n".join(translated)

    # Intercept OpenAI-format tool_call JSON (Qwen3-Coder instruct artifact)
    # and translate to CALL() syntax before normal code extraction.
    translated = translate_openai_tool_calls(response)
    if translated is not None:
        return translated

    # Remove trailing backticks that aren't properly paired
    # (model sometimes outputs code followed by ``` without opening)
    if response.endswith("```"):
        # Check if there's a matching opening
        if response.count("```") % 2 == 1:
            response = response[:-3].rstrip()

    # Try to extract from markdown code block
    code_block_pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(code_block_pattern, response, re.DOTALL)

    if matches:
        return _strip_import_lines(matches[0].strip())

    # If no code block, try to find code-like content
    lines = response.split("\n")
    code_lines = []
    in_code = False

    # Include REPL tool functions as code starters
    code_starters = [
        "import ",
        "from ",
        "def ",
        "class ",
        "if ",
        "for ",
        "while ",
        "try:",
        "except",
        "with ",
        "return ",
        "print(",
        "FINAL(",
        "artifacts[",
        "result =",
        "answer =",
        "output =",
        # REPL tools
        "peek(",
        "grep(",
        "list_dir(",
        "file_info(",
        "ocr_document(",
        "analyze_figure(",
        "extract_figure(",
        "web_fetch(",
        "run_shell(",
        "recall(",
        "escalate(",
        "llm_call(",
        "llm_batch(",
        "CALL(",
    ]

    tool_call_starters = (
        "peek(",
        "grep(",
        "list_dir(",
        "file_info(",
        "ocr_document(",
        "analyze_figure(",
        "extract_figure(",
        "web_fetch(",
        "run_shell(",
        "recall(",
        "escalate(",
        "llm_call(",
        "llm_batch(",
        "CALL(",
        "TOOL(",
    )

    def _looks_like_code_line(
        raw_line: str, stripped: str, *, already_in_code: bool = False
    ) -> bool:
        if not stripped:
            return already_in_code
        if stripped.startswith("#"):
            return True
        if any(stripped.startswith(kw) for kw in code_starters):
            return True
        if stripped.startswith(tool_call_starters):
            return True
        if re.match(r"^\w+\s*=\s*(?:CALL|TOOL|peek|grep|llm_call|llm_batch)\s*\(", stripped):
            return True
        if ("=" in stripped or "()" in stripped) and not re.match(r"^[A-Z][a-z]+(?:\s+[a-z]+)+", stripped):
            return True
        # Bare triple-quote lines (closing a multi-line string) are valid code.
        if already_in_code and stripped in ('"""', "'''"):
            return True
        if already_in_code and (
            raw_line.startswith((" ", "\t"))
            or stripped.endswith(":")
            or stripped.startswith((")", "]", "}", "elif ", "else:", "except", "finally:"))
        ):
            return True
        return False

    for line in lines:
        stripped = line.strip()
        if _looks_like_code_line(line, stripped, already_in_code=False):
            in_code = True

        if in_code:
            # Once code starts, keep only lines that still look like code.
            if _looks_like_code_line(line, stripped, already_in_code=True):
                code_lines.append(line)
        elif stripped.startswith("#"):
            # Standalone comments before code — include as potential preamble
            code_lines.append(line)

    if code_lines:
        # Strip common leading whitespace from all lines
        code = "\n".join(code_lines)
        # Dedent the code to remove consistent leading whitespace
        import textwrap

        code = textwrap.dedent(code).strip()
        # Strip import lines - modules like json are pre-loaded in REPL globals
        code = _strip_import_lines(code)
        # Validate that the extracted text is actually Python, not echoed
        # prompt text that happened to contain a code_starter keyword.
        if _is_valid_python(code):
            return code
        # Salvage FINAL()/tool-call lines from mixed prose+code responses.
        salvage_lines: list[str] = []
        for ln in code.split("\n"):
            s = ln.strip()
            if (
                "FINAL(" in s
                or s.startswith(tool_call_starters)
                or re.match(r"^\w+\s*=\s*(?:CALL|TOOL|peek|grep|llm_call|llm_batch)\s*\(", s)
            ):
                salvage_lines.append(ln)
        if salvage_lines:
            salvaged = textwrap.dedent("\n".join(salvage_lines)).strip()
            salvaged = _strip_import_lines(salvaged)
            if _is_valid_python(salvaged):
                return salvaged
        _log.debug("code_starters extraction failed syntax check, using strict fallback")

    # Strict fallback: only return whole response if it parses as Python.
    import textwrap

    code = textwrap.dedent(response).strip()
    code = _strip_import_lines(code)
    if _is_valid_python(code):
        return code

    # Last-resort prose rescue: scan the ENTIRE raw response for CALL()/FINAL()
    # embedded anywhere in prose lines.  Models sometimes write tool calls as
    # part of their reasoning (e.g. "I should CALL("search_wikipedia", ...)")
    # without code blocks.  Extract just the executable parts.
    _INLINE_CALL_RE = re.compile(
        r'((?:CALL|TOOL)\s*\(\s*"[^"]+"\s*(?:,\s*\w+\s*=\s*(?:"[^"]*"|\'[^\']*\'|\d+|True|False|None))*\s*\))',
    )
    _INLINE_FINAL_RE = re.compile(
        r'(FINAL\s*\(\s*(?:"[^"]*"|\'[^\']*\'|[A-Za-z0-9_.]+)\s*\))',
    )
    rescued: list[str] = []
    for m in _INLINE_CALL_RE.finditer(response):
        expr = m.group(1)
        if _is_valid_python(expr):
            rescued.append(f"result = {expr}")
            rescued.append("print(result)")
    for m in _INLINE_FINAL_RE.finditer(response):
        expr = m.group(1)
        if _is_valid_python(expr):
            rescued.append(expr)
    if rescued:
        rescued_code = "\n".join(rescued)
        _log.info("Prose CALL/FINAL rescue: extracted %d expressions from raw output", len(rescued))
        return rescued_code

    return ""


def auto_wrap_final(code: str) -> str:
    """Auto-wrap code in FINAL() if it looks like a final answer.

    This is a deterministic wrapper that detects when the model has generated
    complete code but didn't wrap it in FINAL(). This allows models to generate
    code naturally while still signaling completion to the orchestrator.

    Args:
        code: Extracted code from the model's response.

    Returns:
        Code wrapped in FINAL() if it's a final answer, otherwise unchanged.
    """
    # Already has FINAL - check if it's reachable at module level.
    # Models often define main() with FINAL() inside, plus
    # `if __name__ == "__main__": main()` — but in a REPL exec() context
    # __name__ is NOT "__main__", so FINAL() is never reached.
    if "FINAL(" in code:
        # Check if FINAL() only appears inside indented (function) blocks
        has_toplevel_final = any(
            ln.startswith("FINAL(") for ln in code.split("\n")
            if ln.strip() and not ln.strip().startswith("#")
        )
        if has_toplevel_final:
            return code
        # FINAL() is only inside a function.  Ensure the function is called.
        # Common pattern: def main(): ... FINAL(...) + if __name__=="__main__": main()
        # In exec() context, __name__ != "__main__" so add a bare main() call.
        if re.search(r"^def main\s*\(", code, re.MULTILINE):
            has_main_call = re.search(
                r"^main\s*\(", code, re.MULTILINE
            )
            if not has_main_call:
                code = code.rstrip() + "\nmain()\n"
        return code

    # Has exploration/continuation functions - not a final answer
    exploration_patterns = [
        "peek(",  # Exploring context
        "grep(",  # Searching context
        "llm_call(",  # Delegating to sub-LM
        "llm_batch(",  # Batch delegation
        "artifacts[",  # Storing intermediate results
        # INF-78 OAB-7: pulls from a request's context bundle (`context.grep(` is
        # already caught by "grep(")
        "context.get(",
        "context.json(",
        "context.index(",
        "context.search(",  # UFH-12: returns pointers; the pull comes next turn
        "context[",
    ]
    for pattern in exploration_patterns:
        if pattern in code:
            return code

    # Get non-empty, non-comment lines
    lines = [
        line.strip()
        for line in code.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return code

    # Code starting with def/class is likely a complete answer —
    # UNLESS it also calls print() or has a trailing expression that
    # invokes the function.  In that case the model wants the code to
    # execute and produce output, not be returned as a string.
    if lines[0].startswith(("def ", "class ")):
        has_print = any("print(" in ln for ln in lines)
        # Trailing call: check original (unstripped) lines for a top-level
        # statement after the def/class (not indented = not part of the body)
        raw_nonempty = [
            ln for ln in code.split("\n")
            if ln.strip() and not ln.strip().startswith("#")
        ]
        # Check if ANY top-level line (after the opening def/class) calls
        # or assigns — indicating the model wants the code to execute.
        has_trailing_call = False
        for raw_ln in raw_nonempty[1:]:
            if raw_ln.startswith((" ", "\t")):
                continue  # inside a function/class body
            s = raw_ln.strip()
            if s.startswith(("def ", "class ")):
                continue  # another definition, not a call
            if "(" in s or "=" in s:
                has_trailing_call = True
                break
        if not has_print and not has_trailing_call:
            # Pure definition — the code IS the answer
            escaped_code = code.replace("'''", r"\'\'\'")
            return f"FINAL('''{escaped_code}''')"

    # Single expression/value is likely a final answer
    # Exclude control flow and imports
    if len(lines) == 1:
        first_line = lines[0]
        # Error payloads from upper layers often come as bracketed strings
        # like "[ERROR: ...]". Wrapping them as FINAL([ERROR: ...]) yields
        # invalid Python syntax, so coerce to a quoted string.
        if first_line.startswith("[ERROR:") and first_line.endswith("]"):
            safe = first_line.replace("\\", "\\\\").replace('"', '\\"')
            return f'FINAL("{safe}")'
        non_final_patterns = [
            "import ",
            "from ",
            "for ",
            "while ",
            "if ",
            "try:",
            "with ",
            # INF-78 OAB-7: FINAL(print(x)) is FINAL(None) -- the call ended with the
            # answer "None". A one-line print is output for the next turn (the RLM
            # pull-and-print move), never an answer.
            "print(",
        ]
        if not any(first_line.startswith(p) for p in non_final_patterns):
            return f"FINAL({first_line})"

    return code


# `FINAL(OK)` — the whole code is one call whose argument is one bare identifier.
_BARE_NAME_FINAL_RE = re.compile(r"^FINAL\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;?$")

# Template placeholders the root prompt itself uses (`FINAL(answer)`, `FINAL(result)`,
# `FINAL(solution)`). An undefined one of these is a model that forgot to assign
# its answer, not a one-word answer — leave that NameError alone.
_BARE_NAME_FINAL_PLACEHOLDERS = frozenset(
    {"answer", "result", "solution", "your_answer", "final_answer", "output", "response"}
)


def rescue_bare_name_final(code: str, error: str | None) -> str | None:
    """Return a quoted ``FINAL("word")`` retry for an unquoted one-word answer.

    Models asked for a one-word reply routinely emit ``FINAL(OK)`` / ``FINAL(yes)``
    instead of ``FINAL("OK")``, and a bare ``OK`` reply is auto-wrapped by
    :func:`auto_wrap_final` into the same shape. Executed, that is
    ``NameError: name 'OK' is not defined`` — no final answer and no output — so
    the turn yields nothing, and a caller that does not feed errors back (the
    ``/v1/chat/completions`` REPL loop) returned ``""`` with ``finish_reason=stop``.

    Fires only when the WHOLE code is that single call AND execution failed with a
    NameError for exactly that identifier, so a name the REPL does define (a
    variable from an earlier turn) still resolves normally and is never rewritten.

    Returns:
        Replacement code to execute, or None when the rescue does not apply.
    """
    if not error or not error.startswith("NameError"):
        return None
    lines = [
        ln.strip() for ln in code.split("\n") if ln.strip() and not ln.strip().startswith("#")
    ]
    if len(lines) != 1:
        return None
    m = _BARE_NAME_FINAL_RE.match(lines[0])
    if m is None:
        return None
    name = m.group(1)
    if name.lower() in _BARE_NAME_FINAL_PLACEHOLDERS:
        return None
    if f"name '{name}' is not defined" not in error:
        return None
    return f'FINAL("{name}")'


# Error classification utilities
def classify_error(error_message: str, gate_name: str = "") -> ErrorCategory:
    """Classify an error message into an ErrorCategory.

    Args:
        error_message: The error message to classify.
        gate_name: Optional gate name if error came from a gate.

    Returns:
        ErrorCategory for the error.
    """
    # Import here to avoid circular imports
    from src.escalation import ErrorCategory

    error_lower = error_message.lower()

    # Schema/format errors (from gates or parsing)
    if gate_name in ("schema", "format", "lint", "mdformat", "shfmt"):
        return ErrorCategory.FORMAT
    if "schema" in error_lower or "validation" in error_lower:
        return ErrorCategory.SCHEMA
    if "format" in error_lower or "style" in error_lower:
        return ErrorCategory.FORMAT

    # Code errors (syntax, type, import)
    code_keywords = [
        "syntaxerror",
        "indentationerror",
        "typeerror",
        "nameerror",
        "importerror",
        "modulenotfound",
        "attributeerror",
    ]
    if any(kw in error_lower for kw in code_keywords):
        return ErrorCategory.CODE

    # Logic errors (test failures, assertions)
    logic_keywords = ["assertionerror", "test failed", "expected", "actual"]
    if any(kw in error_lower for kw in logic_keywords):
        return ErrorCategory.LOGIC

    # Timeout errors
    if "timeout" in error_lower or "timed out" in error_lower:
        return ErrorCategory.TIMEOUT

    # Early abort (from generation monitor)
    if "early abort" in error_lower or "high entropy" in error_lower:
        return ErrorCategory.EARLY_ABORT

    return ErrorCategory.UNKNOWN
