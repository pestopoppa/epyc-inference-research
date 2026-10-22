"""LIVE (template + vocab only) audit pinned per registry role: UFH14-LB5 and DS-LB1.

Uses the v10 `test-chat-template` and `llama-tokenize` from the kernel store (vocab_only, no weights, no
server) and the real GGUF headers. Skipped when those are absent. The expected outcomes are PINNED with
their reason: a template or GGUF change that flips one fails here and must be re-read, not waved through.
Engine caveat: `--no-common` is the raw jinja engine; llama-server's `common_chat_templates_apply` path adds
its own tool/reasoning handling on top, so this is the static template guard, not a server trace.
"""
from pathlib import Path

import pytest

import harness as h

REGISTRY = Path("/workspace/repos/epyc-orchestrator/orchestration/model_registry.yaml")
LIVE = (REGISTRY.exists() and (h.KERNEL_STORE_CPU / "test-chat-template").exists()
        and (h.KERNEL_STORE_CPU / "llama-tokenize").exists())
pytestmark = pytest.mark.skipif(not LIVE, reason="registry or kernel-store template/tokenizer tools absent")

# id -> (P1, P2, P3). Every AS-SERVED template holds all three at enable_thinking=false.
# The stock Qwen3.6-35B embedded template is the one that does not: it DROPS the empty think block from
# earlier assistant replies (think_block_dropped), so a hybrid slot cache would diverge inside the previous
# reply on every follow-on turn -- the reason frontdoor/worker serve the terse override file instead.
EXPECT = {
    "Qwen3.6-35B-A3B-MTP-Q8_0+file:epyc-qwen3x-v1-terse.jinja": (True, True, True),
    "Qwen3.8-27B-Q8_0+file:epyc-qwen3x-v1-terse.jinja": (True, True, True),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003+embedded": (True, True, True),
    "Qwen3-VL-30B-A3B-Instruct-Q4_K_M+embedded": (True, True, True),
    "Qwen3.8-27B-Q8_0+embedded": (True, True, True),
    "Qwen3.6-35B-A3B-MTP-Q8_0+embedded": (False, False, True),
}


@pytest.fixture(scope="module")
def entries():
    return {e["id"]: e for e in h.role_templates(REGISTRY)}


@pytest.fixture(scope="module")
def audits(entries):
    import audit
    fixtures = h.all_fixtures()
    return {i: audit.audit_entry(e, fixtures, False) for i, e in entries.items()}


def test_every_registry_role_is_covered_by_a_pinned_row(entries):
    assert set(entries) == set(EXPECT)
    covered = {r for e in entries.values() for r in e["roles"] if not r.startswith("(")}
    assert covered >= {"frontdoor", "worker", "architect_critic", "architect_general", "coder_escalation",
                       "ingest_long_context", "worker_vision"}


@pytest.mark.parametrize("entry_id", sorted(EXPECT))
def test_pinned_prefix_properties(audits, entry_id):
    v = audits[entry_id]["verdict"]
    assert (v["P1"], v["P2"], v["P3"]) == EXPECT[entry_id], audits[entry_id]["fixtures"]


def test_the_failing_stock_template_fails_for_the_documented_reason(audits):
    fx = audits["Qwen3.6-35B-A3B-MTP-Q8_0+embedded"]["fixtures"]
    kinds = {f["kind"] for r in fx.values() for f in r["P1_reply_prefix"] if not f["ok"]}
    assert kinds == {"think_block_dropped"}
    # and P2 is skipped, not independently failing: the text prefix is already broken
    assert all(f["kind"] == "text_prefix_broken_see_P1" for r in fx.values() for f in r["P2_token_prefix"] if not f["ok"])


def test_served_templates_really_emit_the_empty_think_block_in_history(entries):
    """The mechanism behind a passing row: at enable_thinking=false the generation prompt opens with an
    empty think block and the template keeps that block on the earlier reply."""
    eng, tok, eog = h.engine_for(entries["Qwen3.8-27B-Q8_0+file:epyc-qwen3x-v1-terse.jinja"])
    fx = h.load_fixture("chat_multiturn")
    gp = eng.render(fx["messages"][:2], enable_thinking=False)
    assert gp.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    hist = eng.render(fx["messages"][:4], enable_thinking=False, add_generation_prompt=False)
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\nCanberra.<|im_end|>" in hist
    assert eog == "<|im_end|>" and tok("<|im_start|>assistant") == tok("<|im_start|>") + tok("assistant")


def test_two_sessions_share_the_system_trunk_tokens_on_every_role(audits):
    for i, a in audits.items():
        t = a["P3_shared_trunk"]
        assert t["ok"] and t["shared_tokens"] >= t["system_block_tokens"] - 1, (i, t)
