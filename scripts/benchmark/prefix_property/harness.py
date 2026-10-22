#!/usr/bin/env python3
"""Tokenizer-and-template-only prefix-property harness (DS-LB1, UFH14-LB5; intake-2386, intake-2430).

NO server, NO weights, NO GPU. Two production-kernel tools, both vocab/template only:
  * `test-chat-template --no-common --json IN --output OUT TEMPLATE.jinja` : the v10 jinja engine
    (the one llama-server renders with) over a template file and a JSON context;
  * `llama-tokenize -m GGUF --stdin --no-escape --ids --no-bos` : loads the GGUF with
    `vocab_only=true` (header + tokenizer, ~100 MB RSS, <1 s even for a 27 GB file).
Template text, BOS/EOS strings come from the GGUF header read by a small stdlib-only parser here
(never touches tensor data, never imports gguf-py: that would write bytecode into the frozen tree).

THE PROPERTIES (why a prefix cache stays warm across agent turns)
  P1 reply-prefix  : render(turn k, add_generation_prompt) + <what the model generated> + <eog> must be a
                     BYTE prefix of render(turn k+1, add_generation_prompt). A template that rewrites the
                     previous assistant message (drops its think block, re-formats its tool call) breaks the
                     cache at that point on a hybrid model, where a mid-prompt divergence costs a full
                     re-prefill (no partial restore).
  P2 token-prefix  : tokens(served text) must be a TOKEN prefix of tokens(next render) -- the cache compares
                     token ids, and BPE merges can cross a text boundary.
  P3 shared trunk  : two sessions with the same system message/tools must share the tokens of that system block.
All three are STATIC guards for the prefix-stability assumption; they say nothing about quality.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

KERNEL_STORE_CPU = Path(os.environ.get("PP_KERNEL_STORE", "/mnt/raid0/llm/kernels/production/cpu"))
FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ------------------------------------------------------------------------------ GGUF header reader
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


def _read(f, fmt):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))[0]


def _read_str(f):
    n = _read(f, "<Q")
    return f.read(n).decode("utf-8", "replace")


def _read_value(f, t, keep):
    if t in _SCALAR:
        return _read(f, _SCALAR[t])
    if t == _STRING:
        return _read_str(f)
    if t == _ARRAY:
        et, n = _read(f, "<I"), _read(f, "<Q")
        if et == _STRING:
            vals = [_read_str(f) for _ in range(n)]
            return vals if keep else None
        sz = struct.calcsize(_SCALAR[et])
        if keep:
            return list(struct.unpack(f"<{n}{_SCALAR[et][1]}", f.read(sz * n)))
        f.seek(sz * n, 1)
        return None
    raise ValueError(f"unknown GGUF value type {t}")


def read_gguf_meta(path, array_keys=("tokenizer.ggml.tokens",)) -> dict:
    """Header KV pairs of a GGUF (first split file for split models). Arrays are kept only for
    `array_keys`. Reads no tensor data."""
    kv = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        _version, _n_tensors, n_kv = _read(f, "<I"), _read(f, "<Q"), _read(f, "<Q")
        for _ in range(n_kv):
            key, t = _read_str(f), _read(f, "<I")
            kv[key] = _read_value(f, t, keep=key in array_keys)
    return kv


def template_info(gguf_path) -> dict:
    """{'chat_template': str|None, 'bos': str, 'eos': str, 'arch': str} from the GGUF header."""
    kv = read_gguf_meta(gguf_path)
    toks = kv.get("tokenizer.ggml.tokens") or []

    def tok(key):
        i = kv.get(key)
        return toks[i] if isinstance(i, int) and 0 <= i < len(toks) else ""

    return {"chat_template": kv.get("tokenizer.chat_template"), "bos": tok("tokenizer.ggml.bos_token_id"),
            "eos": tok("tokenizer.ggml.eos_token_id"), "arch": kv.get("general.architecture")}


# ------------------------------------------------------------------------------ engines
class EngineUnavailable(RuntimeError):
    pass


def _tool(name):
    p = KERNEL_STORE_CPU / name
    if not p.exists():
        raise EngineUnavailable(f"{p} not found (kernel store)")
    return p


def _env():
    return {**os.environ, "LD_LIBRARY_PATH": str(KERNEL_STORE_CPU)}


class JinjaEngine:
    """The v10 jinja engine over one template (text). `render(messages, ...)` -> prompt string."""

    def __init__(self, template_text: str, bos: str = "", eos: str = "", tmpdir: Optional[str] = None):
        self.template_text, self.bos, self.eos = template_text, bos, eos
        self._tmp = tempfile.TemporaryDirectory(prefix="pp-", dir=tmpdir)
        self._tpl = Path(self._tmp.name) / "t.jinja"
        self._tpl.write_text(template_text)
        self.calls = 0

    def render(self, messages, *, enable_thinking=False, tools=None, add_generation_prompt=True, extra=None) -> str:
        ctx = {"messages": messages, "bos_token": self.bos, "eos_token": self.eos,
               "add_generation_prompt": add_generation_prompt, "enable_thinking": enable_thinking}
        if tools:
            ctx["tools"] = tools
        ctx.update(extra or {})
        inp, out = Path(self._tmp.name) / "in.json", Path(self._tmp.name) / "out.txt"
        inp.write_text(json.dumps(ctx))
        out.unlink(missing_ok=True)
        r = subprocess.run([str(_tool("test-chat-template")), "--no-common", "--json", str(inp), "--output",
                            str(out), str(self._tpl)], capture_output=True, text=True, env=_env())
        self.calls += 1
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"test-chat-template failed rc={r.returncode}: {(r.stdout + r.stderr)[-400:]}")
        return out.read_text()

    __call__ = render


class VocabTokenizer:
    """`llama-tokenize` over a GGUF (vocab_only). `__call__(text) -> list[int]`; special tokens parsed."""

    def __init__(self, gguf_path):
        self.gguf = str(gguf_path)
        self.calls = 0

    def __call__(self, text: str) -> list[int]:
        r = subprocess.run([str(_tool("llama-tokenize")), "-m", self.gguf, "--stdin", "--no-escape", "--ids",
                            "--no-bos"], input=text.encode(), capture_output=True, env=_env())
        self.calls += 1
        out = r.stdout.decode("utf-8", "replace").strip().splitlines()
        if r.returncode != 0 or not out or not out[-1].startswith("["):
            raise RuntimeError(f"llama-tokenize failed rc={r.returncode}: {r.stderr.decode('utf-8', 'replace')[-300:]}")
        return json.loads(out[-1])


# ------------------------------------------------------------------------------ fixtures
def load_fixture(name_or_path) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = FIXTURES / (name_or_path if str(name_or_path).endswith(".json") else f"{name_or_path}.json")
    return json.loads(p.read_text())


def all_fixtures() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(FIXTURES.glob("*.json"))]


# ------------------------------------------------------------------------------ the checks
@dataclass
class Finding:
    fixture: str
    turn: int                      # index of the assistant message the check is about
    ok: bool
    first_diff: Optional[int] = None
    served_tail: str = ""
    next_at_diff: str = ""
    kind: str = ""                 # "" | think_block_dropped | reformatted | other
    detail: dict = field(default_factory=dict)

    def brief(self) -> str:
        return "ok" if self.ok else f"DIVERGES at char {self.first_diff} ({self.kind}): served ...{self.served_tail!r} vs next ...{self.next_at_diff!r}"


def _classify(served: str, nxt: str, i: int) -> str:
    window = served[max(0, i - 40):i + 60]
    if "<think>" in window and "<think>" not in nxt[max(0, i - 40):i + 60]:
        return "think_block_dropped"
    if "<think>" in nxt[max(0, i - 40):i + 80] and "<think>" not in window:
        return "think_block_added"
    return "reformatted"


def assistant_turns(messages: Sequence[dict]) -> list[int]:
    """Assistant message indices that are followed by at least one more non-assistant message."""
    return [j for j, m in enumerate(messages) if m["role"] == "assistant" and j + 1 < len(messages)]


def _next_prompt_end(messages, j):
    """Index (exclusive) of the next request's last message: up to, not incl., the next assistant."""
    k = j + 1
    while k < len(messages) and messages[k]["role"] != "assistant":
        k += 1
    return k


def tool_wire(template_text: str) -> str:
    """Wire form the template teaches for tool calls: 'xml' (<function=..><parameter=..>) or 'json'."""
    return "xml" if "<function=" in template_text else "json"


def served_text(render: Callable, messages, j, eog: str, *, generated: Optional[str] = None,
                wire: str = "xml", **kw) -> str:
    """What the slot cache holds after turn j: the prompt that was sent, then the model's own tokens.
    `generated_<wire>` (the raw text a model of that tool wire would emit) wins over `generated`."""
    prompt = render(list(messages[:j]), add_generation_prompt=True, **kw)
    m = messages[j]
    gen = generated if generated is not None else m.get(f"generated_{wire}", m.get("generated", m.get("content") or ""))
    return prompt + gen + eog


def check_reply_prefix(render: Callable, fixture: dict, *, eog: str, enable_thinking=False, extra=None,
                       wire: str = "xml") -> list[Finding]:
    """P1 over every assistant turn of a fixture. `render(messages, add_generation_prompt=, enable_thinking=,
    tools=, extra=)`. A fixture assistant message may carry `generated` (the raw text the model emits,
    e.g. a tool call in the template's wire form) distinct from the structured `content`/`tool_calls`."""
    msgs, tools = fixture["messages"], fixture.get("tools")
    kw = {"enable_thinking": enable_thinking, "tools": tools, "extra": extra}
    out = []
    for j in assistant_turns(msgs):
        sv = served_text(render, msgs, j, eog, wire=wire, **kw)
        nxt = render(list(msgs[:_next_prompt_end(msgs, j)]), add_generation_prompt=True, **kw)
        if nxt.startswith(sv):
            out.append(Finding(fixture["id"], j, True))
            continue
        i = next((n for n, (a, b) in enumerate(zip(sv, nxt)) if a != b), min(len(sv), len(nxt)))
        out.append(Finding(fixture["id"], j, False, i, sv[max(0, i - 30):i + 30], nxt[max(0, i - 30):i + 30],
                           _classify(sv, nxt, i)))
    return out


def first_token_diff(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    """Index of the first position where `b` stops extending `a`, or None if `a` is a token prefix of `b`."""
    for i, x in enumerate(a):
        if i >= len(b) or b[i] != x:
            return i
    return None


def check_token_prefix(render: Callable, tokenize: Callable, fixture: dict, *, eog: str, enable_thinking=False,
                       extra=None, wire: str = "xml") -> list[Finding]:
    """P2: tokens(served text) is a token prefix of tokens(next render), for every assistant turn whose
    P1 holds at the text level (a text divergence already fails the cache; P2 isolates tokenizer effects)."""
    msgs, tools = fixture["messages"], fixture.get("tools")
    kw = {"enable_thinking": enable_thinking, "tools": tools, "extra": extra}
    out = []
    for j in assistant_turns(msgs):
        sv = served_text(render, msgs, j, eog, wire=wire, **kw)
        nxt = render(list(msgs[:_next_prompt_end(msgs, j)]), add_generation_prompt=True, **kw)
        if not nxt.startswith(sv):
            out.append(Finding(fixture["id"], j, False, kind="text_prefix_broken_see_P1",
                               detail={"skipped": True}))
            continue
        ta, tb = tokenize(sv), tokenize(nxt)
        d = first_token_diff(ta, tb)
        out.append(Finding(fixture["id"], j, d is None, d, kind="" if d is None else "token_boundary_merge",
                           detail={"served_tokens": len(ta), "next_tokens": len(tb)}))
    return out


def check_shared_trunk(render: Callable, tokenize: Callable, fixture_a: dict, fixture_b: dict, *,
                       enable_thinking=False, extra=None) -> Finding:
    """P3: two sessions that start with the same system message (and tools) share the tokens of that block.
    The trunk is the render of the system message alone; both sessions' renders must start with it as text
    AND as tokens (so the first session's cache serves the second's prefix)."""
    sysmsg = [m for m in fixture_a["messages"][:1] if m["role"] == "system"]
    assert sysmsg and fixture_b["messages"][:1] == sysmsg, "fixtures must open with the same system message"
    assert fixture_a.get("tools") == fixture_b.get("tools"), "fixtures must declare the same tools"
    kw = {"enable_thinking": enable_thinking, "tools": fixture_a.get("tools"), "extra": extra}
    # block ends before the first user message begins: render system + first user, cut where the user turn starts
    ra = render(list(fixture_a["messages"][:2]), add_generation_prompt=True, **kw)
    rb = render(list(fixture_b["messages"][:2]), add_generation_prompt=True, **kw)
    n_char = next((n for n, (x, y) in enumerate(zip(ra, rb)) if x != y), min(len(ra), len(rb)))
    ta, tb = tokenize(ra), tokenize(rb)
    shared_tok = first_token_diff(ta, tb)
    shared_tok = min(len(ta), len(tb)) if shared_tok is None else shared_tok
    # the first user message differs, so the shared span must reach at least into the user turn opener:
    sys_only = render(sysmsg, add_generation_prompt=False, **kw)
    sys_tokens = len(tokenize(sys_only))
    ok = ra.startswith(sys_only) and rb.startswith(sys_only) and shared_tok >= sys_tokens - 1
    return Finding(f"{fixture_a['id']}|{fixture_b['id']}", 0, ok, None if ok else n_char,
                   kind="" if ok else "system_block_not_shared",
                   detail={"shared_chars": n_char, "shared_tokens": shared_tok, "system_block_tokens": sys_tokens,
                           "text_prefix_a": ra.startswith(sys_only), "text_prefix_b": rb.startswith(sys_only)})


# ------------------------------------------------------------------------------ role resolution
def role_templates(registry_path) -> list[dict]:
    """Every distinct (served GGUF, template source) of the registry's `server_mode` roles, plus each
    GGUF's own embedded template as the no-override reference. -> [{id, roles, gguf, template: 'file'|'gguf',
    template_file, kwargs, served}]"""
    import yaml
    reg = yaml.safe_load(Path(registry_path).read_text())
    roles = reg["roles"]
    out, seen = [], {}
    for name, sm in reg["server_mode"].items():
        mr = sm.get("model_role")
        gguf = (roles.get(mr, {}).get("model") or {}).get("path") if mr else None
        if not gguf:
            # server_mode row without model_role (worker_vision): resolve through the role's own model
            gguf = (roles.get(name, {}).get("model") or {}).get("path")
        if not gguf:
            continue
        # an alias shares its target's server process, hence its template (worker -> frontdoor, ...)
        tf = sm.get("chat_template_file") or (reg["server_mode"].get(sm.get("alias_of")) or {}).get("chat_template_file")
        key = (gguf, tf)
        if key not in seen:
            seen[key] = {"id": f"{Path(gguf).stem}+{'file:' + Path(tf).name if tf else 'embedded'}", "roles": [],
                         "gguf": gguf, "template": "file" if tf else "gguf", "template_file": tf,
                         "kwargs": sm.get("chat_template_kwargs") or {}, "served": sm.get("device") != "none"}
            out.append(seen[key])
        seen[key]["roles"].append(name)
    for e in list(out):                      # the embedded reference of each file-overridden GGUF
        if e["template"] == "file" and (e["gguf"], None) not in seen:
            seen[(e["gguf"], None)] = {"id": f"{Path(e['gguf']).stem}+embedded", "roles": [f"(reference for {','.join(e['roles'])})"],
                                       "gguf": e["gguf"], "template": "gguf", "template_file": None, "kwargs": e["kwargs"],
                                       "served": False}
            out.append(seen[(e["gguf"], None)])
    return out


def engine_for(entry: dict) -> tuple[JinjaEngine, VocabTokenizer, str]:
    info = template_info(entry["gguf"])
    text = Path(entry["template_file"]).read_text() if entry["template"] == "file" else info["chat_template"]
    if not text:
        raise EngineUnavailable(f"{entry['gguf']}: no embedded chat template")
    return JinjaEngine(text, info["bos"], info["eos"]), VocabTokenizer(entry["gguf"]), info["eos"]
