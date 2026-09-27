"""Characterization tests for src.config module.

Tests the public API: env var helpers, config dataclass defaults,
ServerURLsConfig.as_dict(), MonitorConfigData tier_overrides,
TimeoutsConfig defaults, and get_config() singleton behavior.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
    EscalationConfigData,
    LLMConfig,
    MonitorConfigData,
    PathsConfig,
    REPLConfigData,
    ServerConfigData,
    ServerURLsConfig,
    TimeoutsConfig,
    _env_bool,
    _env_float,
    _env_int,
    get_config,
    reset_config,
)
from src.config import models as config_models
from src.config.models import reset_stack_prior_server_url_cache

_RETIRED_ARCHITECT_ROLE = "architect_" "coding"

# ----------------------------------------------------------------------------
# Topology-derived expectations.
#
# Serving ports are NOT restated here: they are read from the same source of
# truth the resolver reads (`scripts.server.stack_numa.NUMA_CONFIG`, itself a
# thin loader over orchestration/stack_topology.yaml). Hardcoding a fleet
# literal is how these tests went stale twice — first at the 2026-07-23
# big+quarters restoration, again at the 2026-07-30 quarters retirement
# (1 full + 2 halves), each time asserting ports the launcher had already freed.
# ----------------------------------------------------------------------------


def _topology_fleet_ports(role: str) -> tuple[int, list[int]]:
    """Return (full_port, sibling_ports) for a quarterable role."""
    from scripts.server.stack_numa import NUMA_CONFIG

    cfg = NUMA_CONFIG[role]
    instances = cfg["instances"]
    full_idx = cfg["full_instance_idx"]
    return instances[full_idx][1], [
        inst[1] for idx, inst in enumerate(instances) if idx != full_idx
    ]


def _quarterable_roles() -> tuple[str, ...]:
    """Roles the topology declares a MULTI-INSTANCE fleet for — derived, not listed.

    This was the literal ("frontdoor", "worker_general", "ingest_long_context"), and
    the 2026-09-22 cutover made two of its three entries aliases with no NUMA_CONFIG
    entry at all, so `_topology_fleet_ports` raised KeyError: 'worker_general'. A role
    is quarterable iff stack_topology.yaml gives it more than one instance and names
    which one is the full — exactly what the readers below need, and exactly what the
    old literal was standing in for.
    """
    from scripts.server.stack_numa import NUMA_CONFIG

    return tuple(
        sorted(
            role
            for role, cfg in NUMA_CONFIG.items()
            if len(cfg.get("instances") or ()) > 1 and cfg.get("full_instance_idx") is not None
        )
    )


def _lineup_host_of(role: str) -> str:
    """The role whose llama-server process `role` answers on (itself if it hosts one).

    Read from `role_launch_meta.<host>.shared_with_first_n`, which phase 2 DERIVES
    from `server_mode.<host>.shared_with` — so this follows a lineup change instead
    of restating one.
    """
    from scripts.server import stack_manifest

    for host, meta in stack_manifest.ROLE_LAUNCH_META.items():
        if isinstance(meta, dict) and role in (meta.get("shared_with_first_n") or []):
            return str(host)
    return role


def _expected_full_mode_fleet_url(role: str) -> str:
    """The `full:`-prefixed fleet URL string the topology implies for ``role``."""
    full_port, sibling_ports = _topology_fleet_ports(role)
    urls = [f"http://localhost:{p}" for p in [full_port, *sibling_ports]]
    if len(urls) > 1:
        urls[0] = f"full:{urls[0]}"
    return ",".join(urls)


def _expected_quarter_mode_fleet_url(role: str) -> str:
    """The fleet URL string for quarter mode: siblings only, no full, no prefix."""
    _, sibling_ports = _topology_fleet_ports(role)
    return ",".join(f"http://localhost:{p}" for p in sibling_ports)


# ============================================================================
# _env_bool
# ============================================================================


class TestEnvBool:
    """Tests for _env_bool() helper."""

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "Yes", "ON"])
    def test_truthy_values(self, value: str) -> None:
        with patch.dict(os.environ, {"TEST_BOOL": value}):
            assert _env_bool("TEST_BOOL") is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", "No", "OFF"])
    def test_falsy_values(self, value: str) -> None:
        with patch.dict(os.environ, {"TEST_BOOL": value}):
            assert _env_bool("TEST_BOOL") is False

    def test_missing_returns_default_false(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_BOOL_MISSING", None)
            assert _env_bool("TEST_BOOL_MISSING") is False

    def test_missing_returns_default_true(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_BOOL_MISSING2", None)
            assert _env_bool("TEST_BOOL_MISSING2", default=True) is True

    def test_unrecognized_value_returns_default(self) -> None:
        with patch.dict(os.environ, {"TEST_BOOL": "maybe"}):
            assert _env_bool("TEST_BOOL", default=True) is True
        with patch.dict(os.environ, {"TEST_BOOL": "maybe"}):
            assert _env_bool("TEST_BOOL", default=False) is False


# ============================================================================
# _env_int
# ============================================================================


class TestEnvInt:
    """Tests for _env_int() helper."""

    def test_valid_int(self) -> None:
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            assert _env_int("TEST_INT", 0) == 42

    def test_invalid_int_returns_default(self) -> None:
        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            assert _env_int("TEST_INT", 99) == 99

    def test_missing_returns_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_INT_MISSING", None)
            assert _env_int("TEST_INT_MISSING", 7) == 7

    def test_negative_parses(self) -> None:
        with patch.dict(os.environ, {"TEST_INT": "-5"}):
            assert _env_int("TEST_INT", 10) == -5

    def test_zero(self) -> None:
        with patch.dict(os.environ, {"TEST_INT": "0"}):
            assert _env_int("TEST_INT", 99) == 0


# ============================================================================
# _env_float
# ============================================================================


class TestEnvFloat:
    """Tests for _env_float() helper."""

    def test_valid_float(self) -> None:
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            assert _env_float("TEST_FLOAT", 0.0) == pytest.approx(3.14)

    def test_valid_int_as_float(self) -> None:
        with patch.dict(os.environ, {"TEST_FLOAT": "42"}):
            assert _env_float("TEST_FLOAT", 0.0) == pytest.approx(42.0)

    def test_invalid_float_returns_default(self) -> None:
        with patch.dict(os.environ, {"TEST_FLOAT": "abc"}):
            assert _env_float("TEST_FLOAT", 1.5) == pytest.approx(1.5)

    def test_missing_returns_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_FLOAT_MISSING", None)
            assert _env_float("TEST_FLOAT_MISSING", 2.5) == pytest.approx(2.5)

    def test_empty_string_returns_default(self) -> None:
        with patch.dict(os.environ, {"TEST_FLOAT": ""}):
            assert _env_float("TEST_FLOAT", 9.9) == pytest.approx(9.9)


# ============================================================================
# LLMConfig defaults
# ============================================================================


class TestLLMConfig:
    """Tests for LLMConfig dataclass defaults."""

    def test_defaults(self) -> None:
        cfg = LLMConfig()
        assert cfg.output_cap == 8192
        assert cfg.batch_parallelism == 4
        assert cfg.call_timeout == 600
        assert cfg.mock_response_prefix == "[MOCK]"
        assert cfg.max_recursion_depth == 5
        assert cfg.default_prompt_rate == pytest.approx(0.50)
        assert cfg.default_completion_rate == pytest.approx(1.50)
        assert cfg.qwen_stop_token == "<|im_end|>"
        assert cfg.depth_role_overrides == "1:worker_general,2:worker_math"
        assert cfg.depth_override_max_depth == 3


# ============================================================================
# EscalationConfigData defaults
# ============================================================================


class TestEscalationConfigData:
    """Tests for EscalationConfigData dataclass defaults."""

    def test_defaults(self) -> None:
        cfg = EscalationConfigData()
        assert cfg.max_retries == 2
        assert cfg.max_escalations == 2

    def test_optional_gates_content(self) -> None:
        cfg = EscalationConfigData()
        assert "typecheck" in cfg.optional_gates
        assert "integration" in cfg.optional_gates
        assert "shellcheck" in cfg.optional_gates
        assert isinstance(cfg.optional_gates, frozenset)


# ============================================================================
# REPLConfigData forbidden modules and builtins
# ============================================================================


class TestREPLConfigData:
    """Tests for REPLConfigData dataclass defaults."""

    def test_defaults(self) -> None:
        cfg = REPLConfigData()
        assert cfg.max_output_len == 10000
        assert cfg.timeout_seconds == 30

    def test_forbidden_modules_contains_dangerous(self) -> None:
        cfg = REPLConfigData()
        dangerous = {"os", "sys", "subprocess", "shutil", "socket", "pickle", "ctypes"}
        assert dangerous.issubset(cfg.forbidden_modules)
        assert isinstance(cfg.forbidden_modules, frozenset)

    def test_forbidden_builtins_contains_dangerous(self) -> None:
        cfg = REPLConfigData()
        dangerous = {"__import__", "eval", "exec", "compile", "open"}
        assert dangerous.issubset(cfg.forbidden_builtins)
        assert isinstance(cfg.forbidden_builtins, frozenset)


# ============================================================================
# ServerConfigData defaults
# ============================================================================


class TestServerConfigData:
    """Tests for ServerConfigData dataclass defaults."""

    def test_defaults(self) -> None:
        cfg = ServerConfigData()
        assert cfg.default_url == "http://localhost:8080"
        assert cfg.timeout == 600
        assert cfg.num_slots == 2
        assert cfg.connect_timeout == 5
        assert cfg.retry_count == 3
        assert cfg.retry_backoff == pytest.approx(0.5)


# ============================================================================
# PathsConfig default paths
# ============================================================================


class TestPathsConfig:
    """Tests for PathsConfig dataclass defaults."""

    def test_default_llm_root(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ORCHESTRATOR_PATHS_LLM_ROOT", None)
            os.environ.pop("ORCHESTRATOR_PATHS_PROJECT_ROOT", None)
            cfg = PathsConfig()
            assert str(cfg.llm_root) == "/mnt/raid0/llm"

    def test_default_project_root(self) -> None:
        # The default is derived from src/config/models.py's own __file__, so a
        # git worktree resolves to ITS checkout, not the main one.
        expected = Path(__file__).resolve().parents[2]
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ORCHESTRATOR_PATHS_LLM_ROOT", None)
            os.environ.pop("ORCHESTRATOR_PATHS_PROJECT_ROOT", None)
            cfg = PathsConfig()
            assert cfg.project_root == expected

    def test_raid_prefix_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ORCHESTRATOR_PATHS_RAID_PREFIX", None)
            cfg = PathsConfig()
            assert cfg.raid_prefix == "/mnt/raid0/"


# ============================================================================
# ServerURLsConfig.as_dict()
# ============================================================================


class TestServerURLsConfig:
    """Tests for ServerURLsConfig and as_dict()."""

    @pytest.fixture(autouse=True)
    def _ignore_live_runtime_facts(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ORCHESTRATOR_IGNORE_RUNTIME_STACK_FACTS", "1")
        reset_stack_prior_server_url_cache()
        yield
        reset_stack_prior_server_url_cache()

    def test_as_dict_excludes_service_urls(self) -> None:
        cfg = ServerURLsConfig()
        d = cfg.as_dict()
        assert "api_url" not in d
        assert "ocr_server" not in d
        assert "vision_api" not in d

    def test_as_dict_includes_role_urls(self) -> None:
        cfg = ServerURLsConfig()
        d = cfg.as_dict()
        # Derived, not restated: every serving role the resolver can resolve
        # (the last-resort fallback table) plus every compatibility alias the
        # resolver knows about, minus the three service URLs as_dict() drops.
        # A role added to those tables without a matching field now fails here
        # instead of silently vanishing from the LLMPrimitives backend map —
        # which is how `architect_critic` (W1, 2026-08-01) slipped past the old
        # hand-enumerated literal.
        service_urls = {"api_url", "ocr_server", "vision_api"}
        expected_keys = (
            set(config_models._LEGACY_SERVER_URL_FALLBACKS)
            | set(config_models._CANONICAL_SERVER_URL_ALIASES)
            | set(config_models._STACK_PRIOR_SERVER_URL_ALIASES)
            | set(config_models._RUNTIME_SELECTED_ROLE_ALIASES)
        ) - service_urls
        assert expected_keys == set(d.keys())
        assert not (service_urls & set(d.keys()))

    def test_as_dict_values_are_strings(self) -> None:
        cfg = ServerURLsConfig()
        d = cfg.as_dict()
        for v in d.values():
            assert isinstance(v, str)

    def test_default_frontdoor_url_matches_compiled_stack_priors(self) -> None:
        from src.registry.stack_priors import live_stack_serving_url_values

        cfg = ServerURLsConfig()
        expected = live_stack_serving_url_values(
            Path(config_models._get_default_stack_priors_path())
        )["frontdoor"]
        # Default resolution follows the compiled live lineup. It must not
        # manufacture the inactive aligned-full port merely because the static
        # topology also declares one.
        assert cfg.frontdoor == expected
        assert len(cfg.frontdoor.split(",")) > 1

    def test_default_architect_urls(self) -> None:
        cfg = ServerURLsConfig()
        # 2026-09-27 ARCHITECT SWAP: architect_general = Flash-Next CPU :8074,
        # architect_critic = the MI210 27B :8083 (host of coder_escalation).
        assert "http://localhost:8074" in cfg.architect_general
        assert "http://localhost:8083" in cfg.architect_critic
        assert cfg.coder_escalation == cfg.architect_critic
        assert _RETIRED_ARCHITECT_ROLE not in cfg.as_dict()

    def test_defaults_derive_from_stack_priors(self, tmp_path: Path) -> None:
        priors = tmp_path / "stack_priors.yaml"
        priors.write_text(
            """
roles:
  frontdoor:
    deployment_status: live_stack
    serving:
      ports: [9100, 9200]
  coder_escalation:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:9300
      ports: [9300]
  worker_general:
    deployment_status: live_stack
    serving:
      ports: [9400, 9500, 9600]
  worker_explore:
    deployment_status: live_stack
    serving:
      ports: [9400, 9500]
  worker_fast:
    deployment_status: live_stack
    serving:
      ports: [9902]
""".lstrip(),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"ORCHESTRATOR_PATHS_STACK_PRIORS_PATH": str(priors)}):
            reset_config()
            cfg = ServerURLsConfig()
            assert cfg.frontdoor == "full:http://localhost:9100,http://localhost:9200"
            # Fix A: coder delegates its URL default to frontdoor (shared GGUF), so
            # it inherits frontdoor's full fleet — NOT coder_escalation's 9300 port.
            assert cfg.coder == "full:http://localhost:9100,http://localhost:9200"
            assert cfg.coder == cfg.frontdoor
            assert cfg.worker == (
                "full:http://localhost:9400,http://localhost:9500,http://localhost:9600"
            )
            assert cfg.worker_explore == cfg.worker_general
            assert cfg.worker_fast == "http://localhost:9902"
            assert cfg.worker_coder == "http://localhost:9902"

        reset_config()

    def test_defaults_fall_back_when_stack_priors_missing(self, tmp_path: Path) -> None:
        with patch.dict(
            os.environ,
            {"ORCHESTRATOR_PATHS_STACK_PRIORS_PATH": str(tmp_path / "missing.yaml")},
        ):
            reset_config()
            cfg = ServerURLsConfig()
            assert "http://localhost:8080" in cfg.frontdoor
            # Fix A: coder delegates its URL default to frontdoor (shared GGUF).
            assert cfg.coder == cfg.frontdoor
            assert cfg.coder.startswith("full:http://localhost:8070")
            assert "8071" not in cfg.coder
            assert cfg.worker_explore == cfg.worker_general
            assert cfg.worker_fast == "http://localhost:8102"

        reset_config()

    def test_quarter_numa_mode_urls_skip_dead_full_ports(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        quarterable_roles = _quarterable_roles()
        assert quarterable_roles, "topology declares no multi-instance fleet"
        live_ports: set[int] = set()
        for role in quarterable_roles:
            _, sibling_ports = _topology_fleet_ports(role)
            live_ports.update(sibling_ports)

        # Hermetic fleet: the quarters/halves are up, the aligned fulls are
        # down — the exact "quarters-only fleet" shape ESC-8 Fix 5 exists to
        # survive. Patching the documented `_port_listening` seam means no real
        # socket is opened and the outcome does not depend on what happens to be
        # running on this shared host.
        monkeypatch.setattr(
            config_models, "_port_listening", lambda port: port in live_ports
        )

        with patch.dict(
            os.environ,
            {
                "ORCHESTRATOR_PATHS_STACK_PRIORS_PATH": str(tmp_path / "missing.yaml"),
                "ORCHESTRATOR_STACK_NUMA_MODE": "quarter",
            },
        ):
            reset_stack_prior_server_url_cache()
            cfg = ServerURLsConfig()
            for role in quarterable_roles:
                full_port, sibling_ports = _topology_fleet_ports(role)
                resolved = getattr(cfg, role)
                assert resolved == _expected_quarter_mode_fleet_url(role)
                # Teeth: the dead aligned-full port never appears, and the
                # "full:" prefix (which arms ConcurrencyAwareBackend's full slot)
                # is absent because no full instance was selected.
                assert str(full_port) not in resolved
                assert not resolved.startswith("full:")
                assert len(resolved.split(",")) == len(sibling_ports)

        reset_stack_prior_server_url_cache()

    def test_both_numa_mode_urls_keep_full_prefix(self, tmp_path: Path) -> None:
        with patch.dict(
            os.environ,
            {
                "ORCHESTRATOR_PATHS_STACK_PRIORS_PATH": str(tmp_path / "missing.yaml"),
                "ORCHESTRATOR_STACK_NUMA_MODE": "both",
                # HERMETIC. Priors missing is only HALF the degraded mode this test is
                # about: without this, live runtime facts from whatever happens to be
                # serving on this shared host win before the last-resort table is ever
                # reached, so the assertions below described the box rather than the
                # code. Runtime facts describe realized ports and are rightly preferred
                # in production; here they are exactly the thing being excluded.
                "ORCHESTRATOR_IGNORE_RUNTIME_STACK_FACTS": "1",
            },
        ):
            reset_stack_prior_server_url_cache()
            cfg = ServerURLsConfig()
            # Every host of a multi-instance fleet keeps the `full:` prefix and the
            # whole fleet, RECOMPUTED from the topology. The previous version pinned
            # frontdoor's :8070 fleet and worker_general's :8072 fleet as two separate
            # facts; the 2026-09-22 cutover moved the worker lane onto frontdoor's
            # process and the second pin became an assertion about a retired fleet.
            hosts = _quarterable_roles()
            assert hosts, "topology declares no multi-instance fleet"
            checked_aliases: list[str] = []
            for role in hosts:
                if not hasattr(cfg, role):
                    continue
                resolved = getattr(cfg, role)
                full_port, sibling_ports = _topology_fleet_ports(role)
                assert resolved == _expected_full_mode_fleet_url(role), role
                assert resolved.startswith("full:"), role
                # An ALIAS on that process launches no server of its own, so every
                # port it dials must be a port of its HOST's fleet. Asserted as a
                # SUBSET, not as equality: which of the host's instances an alias is
                # served from is a live disagreement between the launch view (the
                # tagged subset) and the serving view (WP-13 fleet convergence, whole
                # fleet), and this test is not the place to rule on it. The subset
                # form is what has teeth against the defect that actually happens --
                # an alias still dialling a RETIRED fleet (SSU-F8: worker_general on
                # :8072 after the 2026-09-22 cutover), which no subset can hide.
                host_ports = {
                    str(port) for port in [full_port, *sibling_ports]
                }
                for alias in ServerURLsConfig.__dataclass_fields__:
                    if alias == role or _lineup_host_of(alias) != role:
                        continue
                    alias_url = getattr(cfg, alias)
                    alias_ports = {
                        part.rsplit(":", 1)[-1]
                        for part in alias_url.removeprefix("full:").split(",")
                    }
                    assert alias_ports, alias
                    assert alias_ports <= host_ports, (
                        f"{alias} dials {sorted(alias_ports)}, which is not a subset "
                        f"of its host {role}'s declared fleet {sorted(host_ports)}"
                    )
                    checked_aliases.append(alias)
            assert checked_aliases, (
                "no alias of a multi-instance host is a ServerURLsConfig field — the "
                "alias arm of this test has gone vacuous"
            )

        reset_stack_prior_server_url_cache()


# ============================================================================
# MonitorConfigData tier_overrides structure
# ============================================================================


class TestMonitorConfigData:
    """Tests for MonitorConfigData dataclass defaults and tier_overrides."""

    def test_defaults(self) -> None:
        cfg = MonitorConfigData()
        assert cfg.entropy_threshold == pytest.approx(4.0)
        assert cfg.entropy_spike_threshold == pytest.approx(2.0)
        assert cfg.repetition_threshold == pytest.approx(0.3)
        assert cfg.min_tokens_before_abort == 50
        assert cfg.perplexity_window == 20
        assert cfg.ngram_size == 3

    def test_tier_overrides_has_expected_tiers(self) -> None:
        cfg = MonitorConfigData()
        assert set(cfg.tier_overrides.keys()) == {"worker", "coder", "architect", "ingest"}

    def test_tier_overrides_architect_has_higher_thresholds(self) -> None:
        cfg = MonitorConfigData()
        arch = cfg.tier_overrides["architect"]
        assert arch["entropy_threshold"] > cfg.entropy_threshold
        assert arch["min_tokens_before_abort"] > cfg.min_tokens_before_abort

    def test_task_overrides_has_expected_tasks(self) -> None:
        cfg = MonitorConfigData()
        assert "code" in cfg.task_overrides
        assert "reasoning" in cfg.task_overrides


# ============================================================================
# TimeoutsConfig defaults
# ============================================================================


class TestTimeoutsConfig:
    """Tests for TimeoutsConfig dataclass defaults and for_role()."""

    def test_for_role_known(self) -> None:
        cfg = TimeoutsConfig()
        # architect_general should be >= 600 (registry or fallback)
        result = cfg.for_role("architect_general")
        assert isinstance(result, int)
        assert result > 0

    def test_for_role_unknown_returns_default(self) -> None:
        cfg = TimeoutsConfig()
        result = cfg.for_role("nonexistent_role")
        assert result == cfg.default_request

    def test_role_timeouts_dict_keys(self) -> None:
        cfg = TimeoutsConfig()
        d = cfg.role_timeouts_dict()
        assert "frontdoor" in d
        assert "architect_general" in d
        assert "worker_fast" in d
        # Should not contain service timeouts
        assert "ocr_single_page" not in d

    def test_role_timeouts_dict_shares_canonical_worker_aliases(self) -> None:
        cfg = TimeoutsConfig()
        d = cfg.role_timeouts_dict()
        assert d["worker_explore"] == cfg.worker_general
        assert d["worker_fast"] == cfg.worker_fast
        assert cfg.for_role("worker_explore") == cfg.worker_general
        assert cfg.for_role("worker_fast") == cfg.worker_fast

    def test_service_timeouts_are_positive(self) -> None:
        cfg = TimeoutsConfig()
        assert cfg.ocr_single_page > 0
        assert cfg.ocr_pdf > 0
        assert cfg.health_check > 0


# ============================================================================
# get_config() singleton behavior
# ============================================================================


class TestGetConfig:
    """Tests for get_config() cached singleton."""

    def test_returns_orchestrator_config_data(self) -> None:
        get_config.cache_clear()
        cfg = get_config()
        from src.config import OrchestratorConfigData

        assert isinstance(cfg, OrchestratorConfigData)

    def test_returns_same_instance(self) -> None:
        get_config.cache_clear()
        before = get_config.cache_info().hits
        cfg1 = get_config()
        cfg2 = get_config()
        after = get_config.cache_info().hits
        assert cfg1 == cfg2
        assert after >= before + 1

    def test_cache_clear_yields_new_instance(self) -> None:
        get_config.cache_clear()
        cfg1 = get_config()
        get_config.cache_clear()
        cfg2 = get_config()
        # After clearing cache, a new object is created
        assert cfg1 is not cfg2

    def test_mock_mode_default_true(self) -> None:
        get_config.cache_clear()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ORCHESTRATOR_MOCK_MODE", None)
            get_config.cache_clear()
            cfg = get_config()
            assert cfg.mock_mode is True

    def test_has_nested_sections(self) -> None:
        get_config.cache_clear()
        cfg = get_config()
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.escalation, EscalationConfigData)
        assert isinstance(cfg.repl, REPLConfigData)
        assert isinstance(cfg.server, ServerConfigData)
        assert isinstance(cfg.server_urls, ServerURLsConfig)
        assert isinstance(cfg.timeouts, TimeoutsConfig)
        assert isinstance(cfg.monitor, MonitorConfigData)
        assert isinstance(cfg.paths, PathsConfig)


def test_server_urls_bridge_survives_a_field_the_settings_model_lacks():
    """A dataclass URL field absent from the Pydantic model must NOT become MISSING.

    The bridge used `getattr(settings.server_urls, name, f.default)`, and
    `f.default` is `dataclasses.MISSING` for every field declared with
    `default_factory`. So a field on the dataclass but not on the Pydantic model
    was passed MISSING *explicitly*, defeating its own factory.

    `architect_critic` hit this: added to the dataclass 2026-08-01 (W1), never
    added to the settings model. `_normalise_role_urls` calls `.split(",")` on
    every URL, so real-mode backend init raised and EVERY `real_mode=true`
    request 503'd — for every role, not just that one. AutoPilot INFRA_SKIP'd
    all 17 seeding calls against a stack whose servers were all healthy.

    Asserts the PROPERTY (every URL is a usable string), not one field name, so
    the next field added to only one of the two models is caught too.
    """
    import dataclasses

    from src.config import get_config, reset_config
    from src.config.models import ServerURLsConfig

    reset_config()
    urls = get_config().server_urls.as_dict()
    assert urls, "server_urls resolved to nothing"

    non_strings = {k: type(v).__name__ for k, v in urls.items() if not isinstance(v, str)}
    assert not non_strings, f"non-string server URLs (the MISSING-sentinel bug): {non_strings}"

    empties = [k for k, v in urls.items() if not v.strip()]
    assert not empties, f"empty server URLs: {empties}"

    # Every URL must survive the split the backend initialiser performs.
    for role, value in urls.items():
        parts = [u.strip() for u in value.split(",") if u.strip()]
        assert parts, f"{role} has no usable URL after splitting: {value!r}"

    # And no DECLARED field may hold the sentinel, whether or not it appears in
    # as_dict(). `ocr_server` / `vision_api` are service URLs outside the role
    # map, so membership in as_dict is not the invariant — being a real string
    # is. Checking the instance catches a broken bridge for any field.
    cfg = get_config().server_urls
    sentinels = {
        f.name: type(getattr(cfg, f.name)).__name__
        for f in dataclasses.fields(ServerURLsConfig)
        if not isinstance(getattr(cfg, f.name, None), str)
    }
    assert not sentinels, f"declared URL fields holding a non-string: {sentinels}"
