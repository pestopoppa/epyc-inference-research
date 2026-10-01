"""INF-78 OAB-7: the context bundle is a REPL variable, not an inlined prompt.

Offline and inference-free. Acceptance (handoff autokernel-orchestrator-actor-backend.md,
OAB-7 + OAB-12):
  * the bundle round-trips losslessly (payload -> bundle -> payload; sections concatenate
    back to the caller's text byte for byte, including the real DS41 run-8 prompt);
  * the index reports exact sizes (chars, UTF-8 bytes, lines);
  * PULLS DO NOT ENTER THE ROOT PROMPT: every prompt the REPL loop sends to a model is
    recorded, and a sentinel the model pulls into a variable never appears in any of them;
  * the per-turn print cap bounds what a print() puts in the next root prompt;
  * pull accounting is exact per section and per turn (bytes pulled vs offered,
    unique coverage, printed vs shown, variable-preview bytes), and the pull budget
    refuses past its cap;
  * an unscoped / bundle-less request is unchanged.

Run: taskset -c 72-79 .venv/bin/python -m pytest tests/unit/test_oab7_context_bundle.py -q
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.api.models import ChatRequest
from src.api.models.responses import ChatResponse
from src.repl_environment import REPLEnvironment
from src.repl_environment.context_bundle import (
    BUNDLE_SCHEMA,
    PULLS_SCHEMA,
    ContextBundle,
    ContextPullBudgetExceeded,
    repl_view,
)

SENTINEL = "ZQX-SENTINEL-7f3a"
#: The fixed instruction line every REPL root prompt ends its task with
#: (PromptBuilder.build_root_lm_prompt).
ROOT_MARK = "Answer or write code. End with FINAL()."
DS41_FIXTURE = Path(
    "/mnt/raid0/llm/epyc-inference-research/scripts/kernel_rnd/autokernel/loop/fixtures/"
    "ds41-run8-planner-prompt-node-profile.txt"
)


def _payload(**over):
    target = {"recipe": {"template": {"model": "Qwen3.8-27B-Q8_0.gguf", "threads": 96},
                         "build_dir": "/mnt/raid0/llm/build"},
              "requests": ["a", "b"], "note": "héllo ✓"}
    sections = [
        {"name": "preamble", "text": "## Task\nPropose ONE kernel change.\n", "inline": True},
        {"name": "target", "kind": "json",
         "text": "## Selected target\n```json\n" + json.dumps(target, indent=2, sort_keys=True)
                 + "\n```\n",
         "description": "the target JSON"},
        {"name": "big", "text": "\n".join(f"row {i}: filler {'x' * 60}" for i in range(400))
                                + f"\nneedle: {SENTINEL}\n"},
        {"name": "unicode", "text": "ünïcödé — 三 lines\nsecond\nthird"},
    ]
    payload = {"schema": BUNDLE_SCHEMA, "sections": sections,
               "manifest": {"prompt_sha256": "0" * 64, "source": "test"}}
    payload.update(over)
    return payload


# ─────────────────────────────────────────────────────────────── round trip / index


def test_round_trip_is_lossless():
    payload = _payload()
    bundle = ContextBundle.from_payload(payload)
    again = ContextBundle.from_payload(bundle.to_payload())
    assert again.to_payload() == bundle.to_payload()
    assert again.sha256 == bundle.sha256
    assert bundle.full_text() == "".join(s["text"] for s in payload["sections"])
    # every section text survives verbatim, in order
    assert [s.text for s in bundle.sections] == [s["text"] for s in payload["sections"]]
    assert bundle.to_payload()["manifest"] == payload["manifest"]


def test_dict_form_keeps_order():
    bundle = ContextBundle.from_payload({"sections": {"b": "two", "a": {"text": "one"}}})
    assert [s.name for s in bundle.sections] == ["b", "a"]
    assert bundle.full_text() == "twoone"


@pytest.mark.skipif(not DS41_FIXTURE.is_file(), reason="research DS41 fixture not present")
def test_real_ds41_prompt_round_trips_byte_for_byte():
    text = DS41_FIXTURE.read_text(encoding="utf-8")
    # cut at the prompt's own H2 headers (the research splitter's rule, simplified)
    cuts = [0] + [m.start() for m in re.finditer(r"(?m)^## ", text) if m.start() > 0]
    parts = [text[a:b] for a, b in zip(cuts, cuts[1:] + [len(text)])]
    payload = {"sections": [{"name": f"s{i:02d}", "text": p} for i, p in enumerate(parts)]}
    bundle = ContextBundle.from_payload(payload)
    assert bundle.full_text() == text
    assert bundle.text_sha256 == __import__("hashlib").sha256(text.encode()).hexdigest()
    assert ContextBundle.from_payload(bundle.to_payload()).full_text() == text


def test_index_sizes_are_exact():
    bundle = ContextBundle.from_payload(_payload())
    rows = {r["name"]: r for r in bundle.index()}
    uni = "ünïcödé — 三 lines\nsecond\nthird"
    assert rows["unicode"]["chars"] == len(uni)
    assert rows["unicode"]["bytes"] == len(uni.encode("utf-8"))
    assert rows["unicode"]["lines"] == 3
    assert rows["preamble"]["in_prompt"] is True and rows["big"]["in_prompt"] is False
    assert rows["target"]["kind"] == "json" and rows["target"]["keys"] == ["note", "recipe", "requests"]
    assert bundle.total_bytes == sum(r["bytes"] for r in rows.values())
    # index() is metadata: no pull bytes
    acc = bundle.accounting()
    assert acc["totals"]["bytes_pulled"] == 0 and acc["totals"]["index_calls"] == 1


@pytest.mark.parametrize("bad, msg", [
    ({"sections": []}, "empty"),
    ({"sections": [{"name": "a.b", "text": "x"}]}, "must match"),
    ({"sections": [{"name": "a", "text": "x"}, {"name": "a", "text": "y"}]}, "repeated"),
    ({"sections": [{"name": "a", "text": 3}]}, "text must be a string"),
    ({"sections": [{"name": "a", "text": "no json here", "kind": "json"}]}, "json"),
    ({"sections": [{"name": "a", "text": "x", "extra": 1}]}, "unknown keys"),
    ({"schema": "other.v9", "sections": [{"name": "a", "text": "x"}]}, "schema"),
])
def test_malformed_payloads_are_refused(bad, msg):
    with pytest.raises(ValueError, match=msg):
        ContextBundle.from_payload(bad)


# ─────────────────────────────────────────────────────────────── API + accounting


def test_get_json_grep_and_accounting():
    bundle = ContextBundle.from_payload(_payload())
    big = bundle.sections[2].text
    assert bundle.get("big", max_chars=100) == big[:100]
    assert bundle.get("big", max_chars=50, offset=100) == big[100:150]
    assert bundle.get("target.recipe.template.model") == '"Qwen3.8-27B-Q8_0.gguf"'
    assert bundle.json("target.requests[1]") == "b"
    assert bundle.json("target/recipe/template/threads") == 96
    obj = bundle.item("target")
    obj["recipe"] = "mutated"                     # a fresh copy: the bundle is untouched
    assert bundle.json("target")["recipe"]["build_dir"] == "/mnt/raid0/llm/build"
    hits = bundle.grep(SENTINEL)
    assert hits == [{"section": "big", "line": 401, "text": f"needle: {SENTINEL}"}]

    acc = bundle.accounting()
    assert acc["schema"] == PULLS_SCHEMA
    big_acc = acc["sections"]["big"]
    assert big_acc["offered_bytes"] == len(big.encode())
    # 100 + 50 bytes of spans + the grep line
    assert big_acc["bytes_pulled"] == 150 + len(f"needle: {SENTINEL}")
    assert big_acc["unique_bytes"] == 150 + len(f"needle: {SENTINEL}")   # disjoint spans
    assert big_acc["pulls"] == 3
    # re-reading the same span adds bytes_pulled, not unique_bytes
    bundle.get("big", max_chars=100)
    acc2 = bundle.accounting()["sections"]["big"]
    assert acc2["bytes_pulled"] == big_acc["bytes_pulled"] + 100
    assert acc2["unique_bytes"] == big_acc["unique_bytes"]
    totals = bundle.accounting()["totals"]
    assert totals["bytes_pulled"] == sum(s["bytes_pulled"] for s in bundle.accounting()["sections"].values())


def test_pull_budget_refuses_past_cap():
    bundle = ContextBundle.from_payload(_payload(), pull_budget_bytes=500)
    bundle.get("big", max_chars=400)
    with pytest.raises(ContextPullBudgetExceeded, match="budget exhausted"):
        bundle.get("big", max_chars=400)
    acc = bundle.accounting()
    assert acc["totals"]["bytes_pulled"] == 400 and acc["totals"]["refused"] == 1


def test_print_cap_and_turn_records():
    bundle = ContextBundle.from_payload(_payload(), print_cap_bytes=256)
    bundle.begin_turn(1)
    bundle.get("big", max_chars=1000)
    shown = bundle.cap_output("é" * 1000)            # 2000 bytes, multi-byte chars
    assert shown.startswith("é" * 128) and "[print cap: showed 256 of 2000 bytes" in shown
    assert shown.split("\n[print cap")[0] == "é" * 128   # never splits a character
    bundle.begin_turn(2)
    assert bundle.cap_output("short") == "short"
    acc = bundle.accounting()
    t1, t2 = acc["turns"]
    assert (t1["turn"], t1["bytes_pulled"], t1["printed_bytes"], t1["shown_bytes"], t1["capped"]) == \
        (1, 1000, 2000, 256, True)
    assert (t2["turn"], t2["bytes_pulled"], t2["capped"]) == (2, 0, False)
    assert acc["totals"]["turns_capped"] == 1 and acc["totals"]["shown_bytes"] == 256 + 5


def test_view_exposes_only_the_counted_api():
    bundle = ContextBundle.from_payload(_payload())
    view = repl_view(bundle)
    assert not hasattr(view, "__dict__")
    assert [a for a in dir(view) if not a.startswith("__")] == \
        ["get", "grep", "index", "json", "keys"]
    assert "big" in view and len(view) == 4 and list(view) == view.keys()
    assert "4 sections" in repr(view)                 # repr is not a pull
    assert view["unicode"].startswith("ünïcödé")
    assert bundle.accounting()["sections"]["unicode"]["coverage"] == 1.0


# ─────────────────────────────────────────────────────────────── REPL integration


def _repl_with_bundle(**kw) -> tuple[REPLEnvironment, ContextBundle]:
    bundle = ContextBundle.from_payload(_payload(), **kw)
    repl = REPLEnvironment(context="root prompt text")
    repl.attach_context_bundle(bundle)
    return repl, bundle


def test_repl_context_is_the_bundle_and_legacy_helpers_are_counted():
    repl, bundle = _repl_with_bundle()
    r = repl.execute("x = context.get('big')\nprint(len(x))")
    assert r.error is None and r.output.strip() == str(len(bundle.sections[2].text))
    r = repl.execute("print(peek(20))")               # legacy peek reads the bundle now
    assert r.output.strip() == bundle.full_text()[:20].strip()
    r = repl.execute(f"print(grep('{SENTINEL}'))")
    assert SENTINEL in r.output
    assert repl._context_len() == bundle.total_chars
    acc = bundle.accounting()
    ops = [p["op"] for t in acc["turns"] for p in t["pulls"]]
    assert ops == ["get", "peek", "grep_legacy"]
    assert "context: bundle (4 sections" in repl.get_state()


def test_repl_print_cap_applies_before_spill():
    repl, bundle = _repl_with_bundle(print_cap_bytes=300)
    r = repl.execute("x = context['big']\nprint(x)")
    assert "[print cap: showed 300 of" in r.output
    assert "[Output:" not in r.output                 # no spill, no worker summary
    assert repl.bundle_output_preview_chars == 300 + 400


def test_state_preview_bytes_are_counted():
    repl, bundle = _repl_with_bundle()
    repl.execute("t = context.get('big')")
    state = repl.get_state()
    line = next(ln for ln in state.splitlines() if ln.startswith("  t (str)"))
    assert bundle.accounting()["totals"]["state_preview_bytes"] == len(line.encode())
    repl.get_state()                                    # re-rendered, counted once
    assert bundle.accounting()["totals"]["state_preview_bytes"] == len(line.encode())


def test_unbundled_repl_is_unchanged():
    repl = REPLEnvironment(context="plain context string")
    assert repl.context_bundle is None and repl.bundle_output_preview_chars is None
    r = repl.execute("print(context[:5], peek(5), context_len())")
    assert r.output.strip() == "plain plain 20"
    big = "y" * 20000
    r = repl.execute(f"print('{big}')")
    assert r.output.startswith("[Output:")             # the historical spill path
    assert repl.get_state().startswith("context: str (20 chars)")


# ─────────────────────────────────── pre-reload review fixes (F2 / F4 / F5 / F12 / F13)

#: The two escapes the review proved against the closure-based view: both reach the live
#: ContextBundle (read without accounting, or mutate its caps).
PROVEN_ESCAPES = [
    "s = '{0.get.__closure__[0].cell_contents._sections[2].text}'.format(type(context))\n"
    "print(len(s))",
    "import operator\n"
    "b = operator.attrgetter('get.__closure__')(type(context))[0].cell_contents\n"
    "b.print_cap_bytes = 65536",
]


@pytest.mark.parametrize("code", PROVEN_ESCAPES)
def test_proven_closure_escapes_are_refused(code):
    repl, bundle = _repl_with_bundle(print_cap_bytes=300)
    r = repl.execute(code)
    assert (r.error or "").startswith("Dangerous operation not allowed"), r.error
    assert r.output == "" and bundle.print_cap_bytes == 300
    assert bundle.accounting()["totals"]["bytes_pulled"] == 0


@pytest.mark.parametrize("code, why", [
    ("f.__closure__[0].cell_contents", "cell_contents"),
    ("c = cells[0].cell_contents", "cell_contents"),
    ("g = (i for i in []); g.gi_frame.f_globals", "gi_frame"),
    ("tb.tb_frame.f_back.f_locals", "tb_frame"),
    ("co.cr_frame", "cr_frame"),
    ("import operator\noperator.methodcaller('get', 'x')(obj)", "methodcaller"),
    ("from operator import attrgetter", "attrgetter"),
    ("import operator as op\nag = op.attrgetter", "attrgetter"),
    ("import string\nstring.Formatter().get_field('0.__class__', [1], {})", "get_field"),
    ("'{0.__class__}'.format(1)", "format"),
    ("'{x.__init__.__globals__}'.format_map({'x': 1})", "format"),
    ("t = '{0.get.__closure__}'\nprint(t.format(obj))", "format"),
    ("print(str.format('{0[__x]}', {}))", "format"),
    ("import string\nstring.Formatter().vformat('{0.__dict__}', [1], {})", "format"),
])
def test_ast_checker_refuses_introspection_routes(code, why):
    import ast as _ast

    from src.repl_environment.security import ASTSecurityVisitor

    visitor = ASTSecurityVisitor()
    visitor.visit(_ast.parse(code))
    assert any(why in v for v in visitor.violations), visitor.violations


@pytest.mark.parametrize("code", [
    "print('{} + {} = {}'.format(1, 2, 3))",
    "print('{name}: {v:.2f}'.format(name='x', v=1.5))",
    "print('{0[a]} {0[b]}'.format({'a': 1, 'b': 2}))",
    "print('{a}'.format_map({'a': 1}))",
    "import operator\nprint(operator.add(1, 2), sorted([(1, 'b'), (0, 'a')], key=operator.itemgetter(0)))",
    "from operator import itemgetter\nprint(itemgetter(1)([5, 6]))",
    "x = 3\nprint(f'{x} {x!r} {x:>4}')",
    "print('__main__', '%s' % 'dunder __init__ in text')",
    "import functools\nprint(functools.reduce(lambda a, b: a + b, [1, 2, 3]))",
    "def g():\n    yield 1\nprint(list(g()))",
    "import re\nprint(re.findall(r'__\\w+__', 'a __init__ b'))",
    "n = len(context.get('big', max_chars=50))\nprint('{} chars'.format(n))",
])
def test_normal_repl_code_still_passes(code):
    repl, _ = _repl_with_bundle()
    r = repl.execute(code)
    assert r.error is None, r.error
    assert r.output.strip()


def test_a_mutated_bundle_cannot_lift_the_print_cap():
    """The REPL caps from ITS copy of print_cap_bytes: even if code reached the live
    bundle and raised its cap, the next turn is still capped at the attach-time value,
    and the graph's preview size does not grow either."""
    repl, bundle = _repl_with_bundle(print_cap_bytes=300)
    bundle.print_cap_bytes = 65536                      # what an escape would do
    r = repl.execute("print(context['big'])")
    assert "[print cap: showed 300 of" in r.output
    assert repl.bundle_output_preview_chars == 300 + 400
    acc = bundle.accounting()["turns"][-1]
    assert acc["capped"] is True and acc["shown_bytes"] == 300


def test_print_cap_holds_on_the_error_path():
    """Review F4: output printed before an exception is capped too."""
    repl, bundle = _repl_with_bundle(print_cap_bytes=300)
    r = repl.execute("print(context['big'])\nraise ValueError('boom')")
    assert r.error and r.error.startswith("ValueError: boom")
    assert "[print cap: showed 300 of" in r.output
    assert len(r.output.split("\n[print cap")[0].encode()) <= 300
    assert SENTINEL not in r.output
    assert bundle.accounting()["turns"][-1]["capped"] is True


def test_exception_text_is_capped_like_printed_output():
    """Exception text reaches the root prompt too: with a bundle it goes through the same
    attach-time cap copy as printed output (a mutated bundle cannot lift it)."""
    repl, bundle = _repl_with_bundle(print_cap_bytes=300)
    bundle.print_cap_bytes = 65536                      # what an escape would do
    r = repl.execute("raise ValueError(context['big'])")
    assert r.error.startswith("ValueError: row 0: filler")
    assert "[print cap: showed 300 of" in r.error
    assert len(r.error.split("\n[print cap")[0].encode()) <= 300
    assert SENTINEL not in r.error
    assert bundle.accounting()["turns"][-1]["capped"] is True
    r = repl.execute("raise KeyError('short')")         # a small error is untouched
    assert r.error == "KeyError: 'short'"


def test_unbundled_exception_text_is_unchanged():
    repl = REPLEnvironment(context="plain context string")
    big = "z" * 20000
    r = repl.execute(f"raise ValueError('{big}')")
    assert r.error == f"ValueError: {big}"


def test_restore_and_reset_keep_the_bundle_bound():
    """Review F5: a checkpoint restore or reset() rebuilds the globals; ``context`` must
    stay the bundle view, not fall back to the root prompt string."""
    repl, _ = _repl_with_bundle(print_cap_bytes=300)
    repl.execute("keep = 41")
    ckpt = repl.checkpoint()
    repl.restore(ckpt)
    r = repl.execute("print(keep + 1, len(context))")
    assert r.error is None and r.output.split()[:2] == ["42", "4"]
    r = repl.execute("print(context['big'])")
    assert "[print cap: showed 300 of" in r.output
    repl.reset()
    r = repl.execute("print(sorted(context.keys()))")
    assert r.error is None and "'big'" in r.output and "root prompt" not in r.output
    assert repl.context == "root prompt text"           # the internals' string is intact


def test_unbundled_restore_and_reset_are_unchanged():
    repl = REPLEnvironment(context="plain context string")
    repl.restore(repl.checkpoint())
    assert repl._globals["context"] == "plain context string"
    repl.reset()
    assert repl._globals["context"] == "plain context string"


def test_lone_surrogates_and_deep_json_do_not_crash():
    """Review F12 / F13: a lone surrogate is sized (not a UnicodeEncodeError) and an
    absurdly nested json section is a ValueError (not a RecursionError)."""
    from src.repl_environment.context_bundle import cap_printed_output

    bundle = ContextBundle.from_payload(
        {"sections": [{"name": "odd", "text": "a\ud800b" * 10}]}, print_cap_bytes=256)
    assert bundle.total_bytes == 10 * 5
    shown, raw, shown_b, capped = cap_printed_output("x\udfff" * 200, 256)
    assert capped and raw == 800 and shown_b <= 256 and "[print cap" in shown
    deep = "[" * 200_000 + "]" * 200_000
    with pytest.raises(ValueError, match="nested too deeply"):
        ContextBundle.from_payload({"sections": [{"name": "j", "kind": "json", "text": deep}]})
    fenced = "## x\n```json\n" + deep + "\n```"
    with pytest.raises(ValueError, match="nested too deeply"):
        ContextBundle.from_payload({"sections": [{"name": "j", "kind": "json", "text": fenced}]})


# ─────────────────────────────────────────────────────────────── request contract


def test_request_contract():
    ok = ChatRequest(prompt="p", force_mode="repl", context_bundle=_payload(),
                     context_print_cap_bytes=1024, context_pull_budget_bytes=10_000)
    assert ok.context_bundle["schema"] == BUNDLE_SCHEMA
    with pytest.raises(ValidationError, match="force_mode='repl'"):
        ChatRequest(prompt="p", context_bundle=_payload())
    with pytest.raises(ValidationError, match="must match"):
        ChatRequest(prompt="p", force_mode="repl",
                    context_bundle={"sections": [{"name": "bad name", "text": "x"}]})
    with pytest.raises(ValidationError, match="requires context_bundle"):
        ChatRequest(prompt="p", context_print_cap_bytes=1024)
    with pytest.raises(ValidationError, match="requires context_bundle"):
        ChatRequest(prompt="p", context_pull_budget_bytes=5)
    plain = ChatRequest(prompt="p")
    assert plain.context_bundle is None and plain.context_pull_budget_bytes is None
    # the echo is omitted from a response without a bundle
    resp = ChatResponse(answer="a", turns=1, tokens_used=0, elapsed_seconds=0.0,
                        mock_mode=True, real_mode=False)
    assert "context_pulls" not in resp.model_dump(exclude_none=True)


# ─────────────────────────────────────────────────────────────── the loop itself


class _RecordingPrimitives:
    """Stands in for LLMPrimitives: records EVERY prompt any role is sent, and plays a
    scripted model for the root role."""

    def __init__(self, script: list[str]):
        self.script = list(script)
        self.calls: list[tuple[str, str]] = []
        self.total_tokens_generated = 0
        self.total_prompt_eval_ms = 0.0
        self.total_generation_ms = 0.0
        self.total_http_overhead_ms = 0.0
        self._last_predicted_tps = 0.0
        self._backends = {}
        self._early_stop_check = None
        self.mock_mode = False

    def llm_call(self, prompt, role=None, **kwargs):
        self.calls.append((str(role), prompt))
        # Only a REPL root prompt gets the scripted model; repair/summary/compaction
        # calls (any role) get a reply that parses as nothing.
        if ROOT_MARK in prompt and self.script:
            return self.script.pop(0)
        return "summary"

    def root_prompts(self) -> list[str]:
        return [p for _, p in self.calls if ROOT_MARK in p]

    def get_cache_stats(self):
        return {}

    def __getattr__(self, name):          # anything else the pipeline probes
        return MagicMock()


def _state():
    state = MagicMock()
    state.tool_registry = None
    state.script_registry = None
    state.hybrid_router = None
    state.session_store = None
    return state


async def _run(request, primitives):
    from src.api.routes.chat_pipeline.repl_executor import _execute_repl
    from src.api.routes.chat_utils import RoutingResult
    from src.roles import Role

    routing = RoutingResult(task_id="oab7-test", task_ir={}, use_mock=False,
                            routing_strategy="forced", formalization_applied=False,
                            document_result=None)
    return await _execute_repl(request=request, routing=routing, primitives=primitives,
                               state=_state(), start_time=time.perf_counter(),
                               initial_role=Role.FRONTDOOR)


@pytest.mark.asyncio
async def test_pulls_never_enter_any_prompt_the_loop_sends():
    script = [
        # turn 1: pull the WHOLE big section (sentinel included) into a variable, print
        # only a count
        "```python\nbig = context['big']\nn_hits = len(context.grep('needle'))\n"
        "print(len(big), n_hits)\n```",
        # turn 2: print a bounded, sentinel-free slice
        "```python\nprint(big[:40])\n```",
        "```python\nFINAL('{\"abstain\": \"nothing to change\"}')\n```",
    ]
    prims = _RecordingPrimitives(script)
    request = ChatRequest(prompt="Propose one change.", real_mode=True, mock_mode=False,
                          force_mode="repl", max_turns=6, context_bundle=_payload(),
                          context_print_cap_bytes=1024)
    resp = await _run(request, prims)
    assert resp.answer == '{"abstain": "nothing to change"}'
    root = prims.root_prompts()
    assert len(root) == 3
    # the index is in the root prompt; the pulled text is not, in ANY call to ANY role
    assert "[Context bundle: the REPL variable `context`]" in root[0]
    assert "| big | text |" in root[0]
    for role, prompt in prims.calls:
        assert SENTINEL not in prompt, f"pulled bytes leaked into a {role} prompt"
        assert "row 399: filler" not in prompt
    # what the model printed DID reach the next root prompt
    assert "row 0: filler" in root[2]
    acc = resp.context_pulls
    assert acc["schema"] == PULLS_SCHEMA
    big_bytes = acc["sections"]["big"]["offered_bytes"]
    assert acc["sections"]["big"]["bytes_pulled"] == big_bytes + len(f"needle: {SENTINEL}")
    assert acc["sections"]["big"]["coverage"] == 1.0
    assert acc["totals"]["shown_bytes"] == acc["totals"]["printed_bytes"]   # nothing capped
    assert [t["turn"] for t in acc["turns"]] == [1, 2, 3]
    assert [t["bytes_pulled"] > 0 for t in acc["turns"]] == [True, False, False]


@pytest.mark.asyncio
async def test_print_cap_bounds_the_next_root_prompt():
    script = [
        "```python\nprint(context['big'])\n```",      # ~26 KB printed
        "```python\nFINAL('{\"abstain\": \"nothing to change\"}')\n```",
    ]
    prims = _RecordingPrimitives(script)
    request = ChatRequest(prompt="Propose one change.", real_mode=True, mock_mode=False,
                          force_mode="repl", max_turns=4, context_bundle=_payload(),
                          context_print_cap_bytes=512)
    resp = await _run(request, prims)
    root = prims.root_prompts()
    assert "[print cap: showed 512 of" in root[1]
    assert SENTINEL not in root[1]                       # beyond the cap
    t1 = next(t for t in resp.context_pulls["turns"] if t["turn"] == 1)
    assert t1["capped"] and t1["shown_bytes"] == 512 and t1["printed_bytes"] > 20_000


@pytest.mark.asyncio
async def test_bundle_less_request_prompt_is_unchanged():
    prims = _RecordingPrimitives(["```python\nFINAL('ok')\n```"])
    request = ChatRequest(prompt="Plain task.", real_mode=True, mock_mode=False,
                          force_mode="repl", max_turns=3)
    resp = await _run(request, prims)
    assert resp.answer == "ok" and resp.context_pulls is None
    root = prims.root_prompts()
    assert "Context bundle" not in root[0] and "context: str (" in root[0]


def test_one_line_prints_and_bundle_pulls_are_not_auto_finalized():
    from src.prompt_builders import auto_wrap_final

    for code in ("print(big[:40])", "print(context.get('profile', max_chars=2000))",
                 "context.index()", "context['target']", "context.json('target.recipe')"):
        assert auto_wrap_final(code) == code, code
    # the historical single-expression heuristic is otherwise unchanged
    assert auto_wrap_final("42") == "FINAL(42)"


@pytest.mark.asyncio
@pytest.mark.parametrize("scoped", ["bundle", "task_root"])
async def test_proactive_delegation_never_intercepts_a_bundle_or_scoped_request(scoped, tmp_path):
    """The proactive stage runs BEFORE mode selection and ignores force_mode. The real
    DS41 run-8 planner prompt classifies COMPLEX, so with prod defaults
    (parallel_execution + architect_delegation on) it was answered by architect
    decomposition and the bundle / task_root never reached the REPL."""
    from unittest.mock import patch

    from src.api.routes.chat_pipeline.proactive_stage import _execute_proactive
    from src.api.routes.chat_utils import RoutingResult

    kw = ({"context_bundle": _payload()} if scoped == "bundle"
          else {"task_root": str(_scratch_root())})
    request = ChatRequest(prompt="Design and implement a multi-step kernel rewrite",
                          real_mode=True, mock_mode=False, force_mode="repl", **kw)
    routing = RoutingResult(task_id="oab7-proactive", task_ir={}, use_mock=False,
                            routing_decision=["frontdoor"], routing_strategy="forced")
    with patch("src.api.routes.chat_pipeline.proactive_stage.features") as feats, \
            patch("src.proactive_delegation.classify_task_complexity") as classify:
        feats.return_value.parallel_execution = True
        result = await _execute_proactive(request, routing, MagicMock(), MagicMock(), time.perf_counter())
    assert result is None
    classify.assert_not_called()


def _scratch_root() -> Path:
    root = Path("/mnt/raid0/llm/tmp") / "oab7-proactive-scope"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ─────────────────────────────────────────────────────────────── the actor CLI


def _bundle_file(tmp_path: Path) -> Path:
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    return path


def test_cli_sends_the_bundle_and_keeps_the_pull_accounting(tmp_path):
    from tests.unit.test_autokernel_actor_cli import MockChat, OK_SCHEMA, _schema_file, chat_response, run_main

    pulls = {"schema": PULLS_SCHEMA, "totals": {"bytes_pulled": 123}}
    sidecar = tmp_path / "prov.json"
    with MockChat(body=chat_response('{"ok": true}', context_pulls=pulls)) as mock:
        code, out, err = run_main(
            ["--root", str(tmp_path), "--read-only", "--schema", str(_schema_file(tmp_path, OK_SCHEMA)),
             "--url", mock.url, "--context-bundle", str(_bundle_file(tmp_path)),
             "--context-print-cap-bytes", "2048", "--context-pull-budget-bytes", "50000",
             "--provenance-out", str(sidecar)], prompt="index only")
    assert code == 0, err
    body = mock.requests[0]["body"]
    assert body["context_bundle"] == _payload()
    assert body["context_print_cap_bytes"] == 2048 and body["context_pull_budget_bytes"] == 50000
    assert body["force_mode"] == "repl" and body["prompt"] == "index only"
    record = json.loads(sidecar.read_text())
    assert record["context_bundle_acknowledged"] is True
    assert record["response"]["context_pulls"] == pulls
    ref = record["request"]["context_bundle"]
    assert ref["sections"] == 4 and len(ref["sha256"]) == 64
    assert "context_bundle" in record["request"]["fields_sent"]


def test_cli_fails_when_the_server_did_not_attach_the_bundle(tmp_path):
    from tests.unit.test_autokernel_actor_cli import MockChat, chat_response, run_main

    with MockChat(body=chat_response('{"ok": true}')) as mock:         # no context_pulls
        code, out, err = run_main(["--root", str(tmp_path), "--url", mock.url,
                                   "--context-bundle", str(_bundle_file(tmp_path))])
    assert code == 1 and "did not attach the context bundle" in err
    assert json.loads(out) == {"ok": True}                               # still salvageable


def test_cli_bundle_flag_misuse(tmp_path):
    from tests.unit.test_autokernel_actor_cli import run_main

    code, _, err = run_main(["--root", str(tmp_path), "--mode", "auto",
                             "--context-bundle", str(_bundle_file(tmp_path))])
    assert code == 1 and "requires --mode repl" in err
    code, _, err = run_main(["--root", str(tmp_path), "--context-print-cap-bytes", "1024"])
    assert code == 1 and "need --context-bundle" in err
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]")
    code, _, err = run_main(["--root", str(tmp_path), "--context-bundle", str(bad)])
    assert code == 1 and "not a bundle" in err


# ───────────────────────────── dict.get idiom (c95-smoke-orv, 2026-10-01 HTTP 500)


def test_repl_dict_get_idiom_with_empty_string_default_does_not_crash():
    """Regression: ``context.get('inbox', '')`` put ``''`` in ``max_chars`` and died in
    ``int('')`` -> ``ValueError: invalid literal for int() with base 10: ''``. The
    architect_critic retried the same line six times and the request ended in HTTP 500."""
    repl, bundle = _repl_with_bundle()
    r = repl.execute(
        "inbox_text = context.get('unicode', '').strip()\n"
        "missing = context.get('inbox', '')\n"
        "print(repr(missing), len(inbox_text))"
    )
    assert r.error is None, r.error
    assert r.output.strip() == f"'' {len(bundle.sections[3].text.strip())}"


def test_dict_get_idiom_semantics():
    bundle = ContextBundle.from_payload(_payload())
    unicode_text = bundle.sections[3].text
    # present text section -> the text (what context[name] returns), counted as a pull
    assert bundle.get("unicode", "") == unicode_text
    assert bundle.accounting()["sections"]["unicode"]["bytes_pulled"] == len(unicode_text.encode())
    # absent -> the default; a miss reads nothing and counts nothing
    pulled = bundle.accounting()["totals"]["bytes_pulled"]
    assert bundle.get("inbox", "") == ""
    sentinel = object()
    assert bundle.get("inbox", default=sentinel) is sentinel
    assert bundle.get("inbox", {}) == {}
    assert bundle.get("inbox", False) is False      # bool is a default, never a count
    assert bundle.get("target.recipe.nope", {}) == {}
    assert bundle.get("unicode.deep", "fallback") == "fallback"   # path into a text section
    assert bundle.accounting()["totals"]["bytes_pulled"] == pulled
    # present JSON section / field -> the parsed object, like context[name]
    assert bundle.get("target", {})["recipe"]["template"]["threads"] == 96
    assert bundle.get("target.requests[1]", "") == "b"


def test_char_count_forms_keep_paging_and_bad_counts_say_why():
    bundle = ContextBundle.from_payload(_payload())
    big = bundle.sections[2].text
    assert bundle.get("big", 10) == big[:10]
    assert bundle.get("big", "10") == big[:10]            # numeric string: legacy int() coercion
    assert bundle.get("big", 10.0) == big[:10]
    assert bundle.get("big", 5, 3) == big[3:8]
    with pytest.raises(TypeError, match="offset must be an int"):
        bundle.get("big", 5, "")
    with pytest.raises(TypeError, match="max_chars must be an int or None"):
        bundle.get("big", "", default="x")
    # strict lookup without a default still names the sections on a miss
    with pytest.raises(KeyError, match="no context section 'inbox'"):
        bundle.get("inbox")


def test_view_get_forwards_default():
    bundle = ContextBundle.from_payload(_payload())
    view = repl_view(bundle)
    assert view.get("inbox", "") == ""
    assert view.get("inbox", default=[]) == []
    assert view.get("big", 7) == bundle.sections[2].text[:7]
