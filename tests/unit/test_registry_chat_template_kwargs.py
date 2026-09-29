"""J12: per-role chat_template_kwargs auto-injection from the registry."""
from __future__ import annotations
import importlib

rl = importlib.import_module("src.registry.registry_loader")

# Operator-signed 2026-09-22 lineup cutover (master registry 96651eae, lean
# registry 860b0b2d), ruling C1: architect_general (:8083 Qwen3.8-27B) runs
# thinking ON at MEDIUM reasoning effort. Ruling C2: ingest_long_context became
# an alias of architect_general and carries the same kwargs. Since the 2026-09-27
# ARCHITECT SWAP that :8083 process (and its kwargs) is architect_critic.
_ARCHITECT_THINKING = {"enable_thinking": True, "reasoning_effort": "medium"}


def test_chat_template_kwargs_for_role_reads_server_mode():
    # frontdoor declares enable_thinking=false; the :8083 27B declares thinking
    # on at medium effort (ruling C1). RI-23a (2026-09-29, OP-69 (a)):
    # coder_escalation, an alias on that same :8083 process, now declares the
    # host's kwargs too. Its old enable_thinking=false was dead on /completion and
    # would have gone LIVE (thinking OFF) under the thinking_roles_chat_lane flag.
    # 2026-09-27 ARCHITECT SWAP (operator-decided): the kwargs moved WITH the
    # process in the master registry — the 27B's thinking-on declaration now sits
    # under architect_critic, and architect_general (Flash-Next, :8074) carries
    # Flash-Next's own enable_thinking=false (its critic-era declaration).
    assert rl.chat_template_kwargs_for_role("frontdoor") == {"enable_thinking": False}
    assert rl.chat_template_kwargs_for_role("coder_escalation") == _ARCHITECT_THINKING
    assert rl.chat_template_kwargs_for_role("architect_critic") == _ARCHITECT_THINKING
    assert rl.chat_template_kwargs_for_role("architect_general") == {"enable_thinking": False}


def test_chat_template_kwargs_ingest_stays_thinking_on():
    # ingest_long_context must stay thinking-on (load-bearing for long-context
    # ingest; feedback_qwen3x_enable_thinking_false). Pre-cutover it ran on
    # Qwen3-Next-80B, whose template ignored the kwarg, so it declared NO
    # override (None). Post-cutover it is an alias on the Qwen3.8-27B, which
    # honours the kwarg, so thinking-on is now declared explicitly.
    ctk = rl.chat_template_kwargs_for_role("ingest_long_context")
    assert ctk == _ARCHITECT_THINKING
    assert ctk["enable_thinking"] is True


def test_chat_template_kwargs_unknown_role_is_none():
    assert rl.chat_template_kwargs_for_role("nonexistent_role_xyz") is None


def test_alias_thinking_kwarg_agrees_with_its_host():
    # RI-23a guard: a shared_with alias rides its host's process, and the stack
    # priors derive the alias's thinking state FROM THE HOST (model_descriptors
    # via server_mode.shared_with). If the alias's own request-side kwarg says
    # otherwise, the chat lane (thinking_roles_chat_lane) admits it as a
    # thinking role and then sends enable_thinking=false — the RI-23a defect.
    loader = rl.RegistryLoader(validate_paths=False)
    server_mode = loader._raw.get("server_mode") or {}
    checked = 0
    for host, cfg in server_mode.items():
        if not isinstance(cfg, dict):
            continue
        host_ctk = rl.chat_template_kwargs_for_role(host) or {}
        if "enable_thinking" not in host_ctk:
            continue
        for alias in cfg.get("shared_with") or []:
            alias_ctk = rl.chat_template_kwargs_for_role(alias)
            if alias_ctk is None or "enable_thinking" not in alias_ctk:
                continue
            checked += 1
            assert alias_ctk["enable_thinking"] == host_ctk["enable_thinking"], (
                f"{alias} (alias on {host}) declares enable_thinking="
                f"{alias_ctk['enable_thinking']} but its host declares "
                f"{host_ctk['enable_thinking']}"
            )
    assert checked, "no alias with a thinking kwarg found: the guard would be vacuous"
