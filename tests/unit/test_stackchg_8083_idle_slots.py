"""STACKCHG-8083BATCH-20261004 part A: `--no-cache-idle-slots` on :8083.

The declaration (`server_mode.<role>.cache_idle_slots`) must travel registry ->
compiled priors -> launcher argv -> runtime attestation, and the orchestrator's
prefix credit must model the loss path the flag introduces (an idle slot purged under
pool pressure has no RAM copy). Offline: synthetic priors, no server, no inference.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.server import orchestrator_stack as oss
from scripts.server import stack_commands
from src.backends import serving_calls
from src.backends.context_limits import ContextLimitResolver
from src.registry.stack_priors import _launch_runtime_record
from src.scheduling import prefix_history as ph

REPO = Path(__file__).resolve().parents[2]


# -- compiler ------------------------------------------------------------------------


def _record(server_cfg: dict) -> dict:
    return _launch_runtime_record(
        role="architect_critic",
        descriptor={},
        server_cfg=server_cfg,
        role_cfg={},
        launch_cfg={
            "effective_context_tokens": 8192,
            "launch": {
                "primary_roles": ["architect_critic"],
                "modes": ["default"],
                "requirements": {},
                "runtime": {},
            },
        },
    )


@pytest.mark.parametrize("declared", [False, True])
def test_declared_cache_idle_slots_is_compiled_verbatim(declared: bool) -> None:
    runtime = _record({"cache_ram": 65536, "cache_idle_slots": declared})
    assert runtime["flags"]["cache_idle_slots"] is declared
    assert runtime["flags"]["cache_ram"] == 65536


def test_undeclared_cache_idle_slots_is_absent_not_none() -> None:
    """Absent keeps every other role's compiled record byte-identical."""
    runtime = _record({"cache_ram": 65536})
    assert "cache_idle_slots" not in runtime["flags"]


# -- launcher ------------------------------------------------------------------------


def _write_prior(tmp_path: Path, flags: dict) -> Path:
    path = tmp_path / "stack_priors.yaml"
    path.write_text(yaml.safe_dump({"roles": {"architect_critic": {
        "deployment_status": "live_stack",
        "serving": {"launch": {"requirements": {}, "runtime": {
            "binary_path": "/prior/llama-server",
            "cache": {"context_tokens": 393216, "slots": 4, "ubatch": 2048},
            "flags": {"flash_attn": True, "jinja": True, "spec": {"enabled": False}, **flags},
        }}},
    }}}, sort_keys=False), encoding="utf-8")
    return path


def _role() -> SimpleNamespace:
    return SimpleNamespace(
        name="architect_critic",
        model=SimpleNamespace(full_path="/m/q38.gguf"),
        acceleration=SimpleNamespace(type="none", experts=None, draft_role=None),
    )


@pytest.mark.parametrize(
    ("declared", "present", "absent"),
    [
        (False, "--no-cache-idle-slots", "--cache-idle-slots"),
        (True, "--cache-idle-slots", "--no-cache-idle-slots"),
    ],
)
def test_launcher_emits_the_declared_direction(tmp_path, monkeypatch, declared, present, absent) -> None:
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", _write_prior(
        tmp_path, {"cache_ram": 65536, "cache_idle_slots": declared}))
    cmd = oss._build_role_command(_role(), port=9083, prepare_runtime_dirs=False)
    assert cmd.count(present) == 1 and absent not in cmd
    # it rides with the other declared serving flags, after --cache-ram
    assert cmd.index(present) > cmd.index("--cache-ram")


def test_launcher_emits_nothing_when_undeclared(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", _write_prior(tmp_path, {"cache_ram": 65536}))
    cmd = oss._build_role_command(_role(), port=9083, prepare_runtime_dirs=False)
    assert "--cache-idle-slots" not in cmd and "--no-cache-idle-slots" not in cmd


# -- runtime attestation -------------------------------------------------------------


def _info():
    return stack_commands.ProcessInfo(role="architect_critic", pid=321, port=8083,
                                      started_at="now", model_path="/models/m.gguf",
                                      log_file="a.log")


_BASE = ["/opt/llama/bin/llama-server", "-m", "/models/m.gguf", "--device", "ROCm0",
         "--cache-ram", "65536"]


def _contract(flags: dict) -> dict:
    return {"requirements": {"model_path": "/models/m.gguf"},
            "runtime": {"binary_path": "/opt/llama/bin/llama-server", "cache": {},
                        "flags": {"device": "ROCm0", "cache_ram": 65536, **flags}},
            "ports": [8083]}


def test_attestation_clean_when_live_matches() -> None:
    assert stack_commands._runtime_attestation_warnings(
        "architect_critic", _info(), _BASE + ["--no-cache-idle-slots"],
        _contract({"cache_idle_slots": False})) == []


def test_attestation_reports_a_server_still_on_the_default() -> None:
    """The pre-relaunch :8083 argv (no flag) resolves to ON: that is drift."""
    assert stack_commands._runtime_attestation_warnings(
        "architect_critic", _info(), _BASE, _contract({"cache_idle_slots": False})) == [
        "architect_critic pid 321 runtime cache_idle_slots expected False; live cmdline has True"
    ]


def test_live_resolution_rules() -> None:
    assert stack_commands._live_cache_idle_slots(["x"]) is True
    assert stack_commands._live_cache_idle_slots(["x", "--no-cache-idle-slots"]) is False
    assert stack_commands._live_cache_idle_slots(
        ["x", "--no-cache-idle-slots", "--cache-idle-slots"]) is True
    # the server disables it when there is no prompt cache
    assert stack_commands._live_cache_idle_slots(["x", "--cache-ram", "0"]) is False


def test_the_new_field_is_mapped_for_attestation() -> None:
    assert "runtime.flags.cache_idle_slots" in stack_commands._RUNTIME_FIELD_CHECKS


# -- context limits + prefix history ---------------------------------------------------

URL = "http://localhost:8083"


def _resolver(cis) -> ContextLimitResolver:
    return ContextLimitResolver(live=False, role_urls=lambda: {}, registry_facts=lambda: {
        8083: {"context_tokens": 393216, "slots": 4, "kv_unified": True,
               "cache_ram_mib": 65536, "cache_idle_slots": cis},
        8070: {"context_tokens": 262144, "slots": 4, "kv_unified": False,
               "cache_ram_mib": 32768, "cache_idle_slots": False},
    })


def test_idle_slot_residency_only_for_unified_pools_that_keep_idle_slots() -> None:
    assert _resolver(False).idle_slot_residency(URL) == (393216, 4)
    assert _resolver(True).idle_slot_residency(URL) is None    # moved to the cache instead
    assert _resolver(None).idle_slot_residency(URL) is None    # undeclared = server default (on)
    assert _resolver(False).idle_slot_residency("http://localhost:8070") is None  # split KV


def _ladder(text: str):
    return serving_calls.prefix_ladder_for_text(text)


def _hist(residency):
    return ph.PrefixHistory(clock=lambda: 1_000_000.0, cache_ram_mib=lambda u: 65536,
                            launch_id=lambda u: "L1", host_wide=False,
                            idle_residency=lambda u: residency)


CONV = "<|im_start|>system\nagent<|im_end|>\n" + "lorem ipsum dolor sit amet " * 4000


def test_credit_kept_while_the_pool_cannot_have_filled() -> None:
    hist = _hist((393216, 4))
    hist.remember(URL, _ladder(CONV), prompt_tokens=60_000)
    for i in range(3):
        hist.remember(URL, _ladder(f"other {i} " * 5000), prompt_tokens=60_000)
    match, reason = hist.lookup(URL, _ladder(CONV + "next turn"))
    assert reason is None and match is not None  # 60k + 3 x 60k = 240k <= 393216


def test_credit_withdrawn_once_an_idle_purge_was_possible() -> None:
    """A 262144-token ingest beside three idle ~60k conversations (60k x 3 + 262k =
    442k) overflows 393216:
    the server purges an idle slot with no cache copy, so the credit must go."""
    hist = _hist((393216, 4))
    hist.remember(URL, _ladder(CONV), prompt_tokens=60_000)
    hist.remember(URL, _ladder("a " * 5000), prompt_tokens=60_000)
    hist.remember(URL, _ladder("b " * 5000), prompt_tokens=60_000)
    hist.remember(URL, _ladder("ingest " * 9000), prompt_tokens=262_144)
    assert hist.lookup(URL, _ladder(CONV + "next turn")) == (None, "idle_purge_risk")


def test_default_caching_keeps_the_volume_rule_alone() -> None:
    hist = _hist(None)   # cache_idle_slots on / undeclared: idle slots go to the RAM cache
    hist.remember(URL, _ladder(CONV), prompt_tokens=60_000)
    hist.remember(URL, _ladder("a " * 5000), prompt_tokens=60_000)
    hist.remember(URL, _ladder("b " * 5000), prompt_tokens=60_000)
    hist.remember(URL, _ladder("ingest " * 9000), prompt_tokens=262_144)
    match, reason = hist.lookup(URL, _ladder(CONV + "next turn"))
    assert reason is None and match is not None   # 442k of a ~1M-token cache


# -- this tree: master declaration -> compiled priors (recompute, not a literal) -------


def test_this_trees_8083_priors_carry_the_master_declaration() -> None:
    lean = yaml.safe_load((REPO / "orchestration/model_registry.yaml").read_text())
    declared = (lean.get("server_mode") or {}).get("architect_critic", {}).get("cache_idle_slots")
    priors = yaml.safe_load((REPO / "orchestration/derived/stack_priors.yaml").read_text())
    flags = priors["roles"]["architect_critic"]["serving"]["launch"]["runtime"]["flags"]
    assert flags.get("cache_idle_slots") == declared
