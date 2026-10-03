"""Tests for Phase 3 configuration consolidation.

Verifies that all new config sections have correct defaults matching
the hardcoded values they replace across the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

# Tests that assert /mnt/raid0/ paths only apply on the production machine
_on_raid = pytest.mark.skipif(
    not Path("/mnt/raid0").exists(),
    reason="Path assertions require /mnt/raid0 (production machine only)",
)

from src.config import (
    ChatPipelineConfig,
    DelegationConfig,
    LLMConfig,
    MonitorConfigData,
    OrchestratorConfigData,
    PathsConfig,
    ServerURLsConfig,
    ServicesConfig,
    TimeoutsConfig,
    VisionConfig,
    WorkerPoolPathsConfig,
    get_config,
    reset_config,
)


@pytest.fixture(autouse=True)
def _clean_config():
    """Reset config cache before and after each test."""
    reset_config()
    yield
    reset_config()


# ----------------------------------------------------------------------------
# Registry-derived expectations.
#
# Serving ports are read from `orchestration/model_registry.yaml` — the same
# declaration the resolver's tables are compiled from — so a role that is
# repointed (the 2026-08-01 W1 cutover moved coder_escalation frontdoor ->
# architect_general and collapsed vision_escalation onto worker_vision) updates
# the expectation instead of leaving a stale literal pinned to a freed port.
# ----------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _registry_server_mode() -> dict:
    import yaml

    registry = yaml.safe_load(
        (_REPO_ROOT / "orchestration" / "model_registry.yaml").read_text(encoding="utf-8")
    )
    return registry["server_mode"]


def _registry_serving_port(role: str) -> int:
    """Resolve a role to the port of the process that actually serves it.

    Follows `alias_of` (an alias row that launches no server of its own) and
    `shared_with` (a role co-hosted on another row's process).
    """
    server_mode = _registry_server_mode()
    seen: set[str] = set()
    while role not in seen:
        seen.add(role)
        entry = server_mode.get(role)
        if entry is not None:
            target = entry.get("alias_of")
            if target and target not in seen:
                role = target
                continue
            return int(entry["port"])
        hosts = [
            host
            for host, row in server_mode.items()
            if role in (row.get("shared_with") or [])
        ]
        assert len(hosts) == 1, f"role {role!r} is not hosted by exactly one row: {hosts}"
        role = hosts[0]
    raise AssertionError(f"role {role!r} has no resolvable server_mode port")


def _registry_host(role: str) -> str:
    """The server_mode row whose process physically serves ``role``.

    Same resolution as :func:`_registry_serving_port` (``alias_of`` then
    ``shared_with``), returning the host ROW instead of its port, so fleet facts
    (NUMA_CONFIG instances) can be looked up under the name the topology keys on.
    """
    server_mode = _registry_server_mode()
    seen: set[str] = set()
    while role not in seen:
        seen.add(role)
        entry = server_mode.get(role)
        if entry is not None:
            target = entry.get("alias_of")
            if target and target not in seen:
                role = target
                continue
            return role
        hosts = [
            host
            for host, row in server_mode.items()
            if role in (row.get("shared_with") or [])
        ]
        assert len(hosts) == 1, f"role {role!r} is not hosted by exactly one row: {hosts}"
        role = hosts[0]
    raise AssertionError(f"role {role!r} has no resolvable server_mode host")


def _topology_fleet_ports(role: str) -> tuple[int, list[int]]:
    """Return (full_port, sibling_ports) for a quarterable role."""
    from scripts.server.stack_numa import NUMA_CONFIG

    cfg = NUMA_CONFIG[role]
    instances = cfg["instances"]
    full_idx = cfg["full_instance_idx"]
    return instances[full_idx][1], [
        inst[1] for idx, inst in enumerate(instances) if idx != full_idx
    ]


# ── Module import ─────────────────────────────────────────────────────────


class TestConfigImports:
    """Verify all new sections are importable."""

    def test_config_module_imports(self):
        import src.config

        assert hasattr(src.config, "get_config")
        assert hasattr(src.config, "reset_config")

    def test_all_sections_importable(self):
        """Every new config section must be importable."""
        from src.config import (
            ServerURLsConfig,
            TimeoutsConfig,
            VisionConfig,
            ChatPipelineConfig,
            DelegationConfig,
            ServicesConfig,
            WorkerPoolPathsConfig,
        )

        for cls in [
            ServerURLsConfig,
            TimeoutsConfig,
            VisionConfig,
            ChatPipelineConfig,
            DelegationConfig,
            ServicesConfig,
            WorkerPoolPathsConfig,
        ]:
            assert cls is not None

    def test_config_is_leaf_module(self):
        """config.py must NOT import from src/ (prevents circular imports).

        Exception: Lazy imports inside try/except blocks are allowed for registry loading.
        """
        import src.config
        import inspect

        source = inspect.getsource(src.config)
        # Allow 'from src.' only inside docstrings/comments — not as actual imports
        # Look for actual import statements
        import ast

        # Allowed lazy imports (inside try/except, used for registry loading)
        allowed_imports = {"src.registry_loader"}

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("src.") and node.module not in allowed_imports:
                        assert False, (
                            f"config.py must be a leaf module: found 'from {node.module}'"
                        )


# ── get_config() and reset_config() ──────────────────────────────────────


class TestGetConfig:
    """Test config singleton and reset behavior."""

    def test_get_config_returns_dataclass(self):
        cfg = get_config()
        assert isinstance(cfg, OrchestratorConfigData)

    def test_get_config_is_cached(self):
        before = get_config.cache_info().hits
        cfg1 = get_config()
        cfg2 = get_config()
        after = get_config.cache_info().hits
        assert cfg1 == cfg2
        assert after >= before + 1

    def test_reset_config_clears_cache(self):
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2

    def test_get_config_all_sections_present(self):
        cfg = get_config()
        assert isinstance(cfg.server_urls, ServerURLsConfig)
        assert isinstance(cfg.timeouts, TimeoutsConfig)
        assert isinstance(cfg.vision, VisionConfig)
        assert isinstance(cfg.chat, ChatPipelineConfig)
        assert isinstance(cfg.delegation, DelegationConfig)
        assert isinstance(cfg.services, ServicesConfig)
        assert isinstance(cfg.worker_pool, WorkerPoolPathsConfig)
        assert isinstance(cfg.monitor, MonitorConfigData)
        assert isinstance(cfg.paths, PathsConfig)
        assert isinstance(cfg.llm, LLMConfig)


# ── ServerURLsConfig as single source of truth ─────────────────────────


class TestServerURLsDefaults:
    """Verify ServerURLsConfig is the single source of truth for server URLs."""

    @pytest.fixture(autouse=True)
    def _ignore_live_runtime_facts(self, monkeypatch: pytest.MonkeyPatch):
        # The defaults under test are the declared topology, not whatever lineup
        # the live host's runtime facts currently select (a sub-full fleet
        # resolves ingest_long_context to :8185 alone). Same seam as
        # tests/unit/test_config.py::TestServerURLsConfig.
        from src.config.models import reset_stack_prior_server_url_cache

        monkeypatch.setenv("ORCHESTRATOR_IGNORE_RUNTIME_STACK_FACTS", "1")
        reset_stack_prior_server_url_cache()
        yield
        reset_stack_prior_server_url_cache()

    def test_as_dict_contains_all_role_urls(self):
        """as_dict() must contain all expected orchestrator role URLs."""
        cfg = ServerURLsConfig()
        config_dict = cfg.as_dict()
        # All roles that LLMPrimitives consumers depend on
        expected_roles = {
            "frontdoor",
            "coder",
            "coder_escalation",
            "worker",
            "worker_general",
            "worker_explore",
            "worker_math",
            "worker_vision",
            "vision_escalation",
            "worker_coder",
            "worker_fast",
            "worker_summarize",
            "architect_general",
            "ingest_long_context",
        }
        for role in expected_roles:
            assert role in config_dict, f"Missing role: {role}"
            url = config_dict[role]
            # Multi-instance URLs use "full:" prefix for ConcurrencyAwareBackend
            assert url.startswith("http://") or url.startswith("full:http://"), (
                f"URL for {role} must be HTTP or full:HTTP: got {url}"
            )

    def test_as_dict_excludes_services(self):
        """as_dict() must NOT contain api_url, ocr_server, vision_api."""
        cfg = ServerURLsConfig()
        d = cfg.as_dict()
        assert "api_url" not in d
        assert "ocr_server" not in d
        assert "vision_api" not in d

    def test_service_urls_present(self):
        """Service URLs must be accessible on the config object."""
        cfg = ServerURLsConfig()
        assert cfg.api_url == "http://localhost:8000"
        assert cfg.ocr_server == "http://localhost:9001"
        assert cfg.vision_api == "http://localhost:8000/v1/vision/analyze"

    def test_specific_role_urls(self):
        """Spot-check specific role->URL mappings (multi-instance since ConcurrencyAware)."""
        cfg = ServerURLsConfig()
        # Multi-instance roles use "full:" prefix or comma-separated URLs.
        fd_full, fd_siblings = _topology_fleet_ports("frontdoor")
        assert cfg.frontdoor.startswith(f"full:http://localhost:{fd_full}")
        for port in fd_siblings:
            assert f"http://localhost:{port}" in cfg.frontdoor

        # 2026-09-22 cutover (orchestrator 860b0b2d): worker_general has no fleet
        # of its own any more — it is co-hosted on a registry host row, and its
        # NUMA_CONFIG entry was deleted. Look the fleet up under the HOST.
        wg_host = _registry_host("worker_general")
        assert wg_host != "worker_general", "worker_general regained its own row"
        wg_full, wg_siblings = _topology_fleet_ports(wg_host)
        assert wg_siblings, f"{wg_host} fleet has no siblings — check is vacuous"
        for port in (wg_full, *wg_siblings):
            assert f"http://localhost:{port}" in cfg.worker_explore
        assert cfg.worker_explore == cfg.worker_general
        # An alias carries its host's whole fleet string, not a copy of part of it.
        assert cfg.worker_general == getattr(cfg, wg_host)

        # ingest_long_context became an `alias_of` row (architect_general's MI210
        # process at the cutover): it must carry its host's URL string verbatim,
        # which includes the registry port of that host.
        ing_host = _registry_host("ingest_long_context")
        assert ing_host != "ingest_long_context", "ingest regained its own server"
        assert (
            f"http://localhost:{_registry_serving_port('ingest_long_context')}"
            in cfg.ingest_long_context
        )
        assert cfg.ingest_long_context == getattr(cfg, ing_host)

        # Single-process roles keep simple URLs; each is pinned to the port the
        # registry declares for the process that actually serves it (following
        # `alias_of` / `shared_with`), never to a hand-copied literal.
        assert cfg.worker_vision == f"http://localhost:{_registry_serving_port('worker_vision')}"
        assert cfg.vision_escalation == (
            f"http://localhost:{_registry_serving_port('vision_escalation')}"
        )
        assert cfg.architect_general == (
            f"http://localhost:{_registry_serving_port('architect_general')}"
        )
        assert cfg.architect_critic == (
            f"http://localhost:{_registry_serving_port('architect_critic')}"
        )
        # W1 cutover: coder_escalation is an alias row on the MI210 27B process —
        # it must resolve to its host's process, not to frontdoor's. The host is
        # architect_critic since the 2026-09-27 ARCHITECT SWAP (architect_general
        # before); resolve it from the registry rather than naming it.
        assert cfg.coder_escalation == (
            f"http://localhost:{_registry_serving_port('coder_escalation')}"
        )
        assert cfg.coder_escalation == getattr(cfg, _registry_host("coder_escalation"))
        assert cfg.worker_fast == "http://localhost:8102"
        # worker_summarize is declared in frontdoor.shared_with, so it must carry
        # frontdoor's whole fleet string, not just its own endpoint.
        assert _registry_serving_port("worker_summarize") == fd_full
        assert cfg.worker_summarize == cfg.frontdoor


# ── TimeoutsConfig defaults match ROLE_TIMEOUTS in chat_utils.py ─────────


class TestTimeoutsDefaults:
    """Verify TimeoutsConfig defaults match chat_utils.py ROLE_TIMEOUTS."""

    def test_for_role_matches_role_timeouts(self):
        """for_role() must return same values as ROLE_TIMEOUTS dict."""
        from src.api.routes.chat_utils import ROLE_TIMEOUTS, DEFAULT_TIMEOUT_S

        cfg = TimeoutsConfig()
        for role, expected_timeout in ROLE_TIMEOUTS.items():
            assert cfg.for_role(role) == expected_timeout, (
                f"Timeout mismatch for role {role}: "
                f"config={cfg.for_role(role)}, expected={expected_timeout}"
            )
        # Unknown role should return default
        assert cfg.for_role("nonexistent") == DEFAULT_TIMEOUT_S

    def test_role_timeouts_dict_matches(self):
        """role_timeouts_dict() must match ROLE_TIMEOUTS exactly."""
        from src.api.routes.chat_utils import ROLE_TIMEOUTS

        cfg = TimeoutsConfig()
        config_dict = cfg.role_timeouts_dict()
        for role, timeout in ROLE_TIMEOUTS.items():
            assert config_dict[role] == timeout

    def test_default_request_matches_default_timeout_s(self):
        from src.api.routes.chat_utils import DEFAULT_TIMEOUT_S

        cfg = TimeoutsConfig()
        assert cfg.default_request == DEFAULT_TIMEOUT_S

    def test_specific_timeout_values(self):
        """Spot-check critical timeout values — differentiated by role."""
        cfg = TimeoutsConfig()
        # Workers: short timeouts (fast circuit breaker)
        assert cfg.worker_explore == 60
        assert cfg.worker_math == 60
        assert cfg.worker_vision == 60
        assert cfg.worker_summarize == 120
        assert cfg.worker_fast == 30
        # Frontdoor/coder: medium
        assert cfg.frontdoor == 180
        assert cfg.coder_escalation == 120
        # Architects: long (complex reasoning)
        assert cfg.architect_general == 600
        # Backend: unified 600s timeout
        assert cfg.server_request == 600
        assert cfg.server_connect == 5

    def test_worker_explore_timeout_canonicalizes_to_worker_general(self):
        """worker_explore should inherit the live worker_general timeout."""
        from src.config import get_config

        def fake_registry_timeout(category: str, key: str, fallback: int | float) -> int | float:
            if category == "roles" and key == "worker_general":
                return 77
            if category == "roles" and key == "worker_explore":
                return 60
            return fallback

        with patch("src.config.models._registry_timeout", side_effect=fake_registry_timeout):
            try:
                get_config.cache_clear()
                cfg = TimeoutsConfig()
                assert cfg.worker_explore == 77
                assert cfg.worker_general == 77
                assert cfg.for_role("worker_explore") == 77
                assert cfg.role_timeouts_dict()["worker_explore"] == 77
            finally:
                get_config.cache_clear()


# ── MonitorConfigData defaults match generation_monitor.py ───────────────


class TestMonitorDefaults:
    """Verify MonitorConfigData defaults match generation_monitor.py MonitorConfig."""

    def test_entropy_threshold(self):
        cfg = MonitorConfigData()
        assert cfg.entropy_threshold == 4.0

    def test_entropy_spike_threshold(self):
        cfg = MonitorConfigData()
        assert cfg.entropy_spike_threshold == 2.0

    def test_min_tokens_before_abort(self):
        cfg = MonitorConfigData()
        assert cfg.min_tokens_before_abort == 50

    def test_perplexity_window(self):
        cfg = MonitorConfigData()
        assert cfg.perplexity_window == 20

    def test_repetition_threshold(self):
        cfg = MonitorConfigData()
        assert cfg.repetition_threshold == 0.3

    def test_ngram_size(self):
        cfg = MonitorConfigData()
        assert cfg.ngram_size == 3

    def test_combined_threshold(self):
        cfg = MonitorConfigData()
        assert cfg.combined_threshold == 0.7


# ── LLMConfig defaults match LLMPrimitivesConfig ─────────────────────────


class TestLLMConfigDefaults:
    """Verify LLMConfig defaults match llm_primitives.py LLMPrimitivesConfig."""

    def test_matches_llm_primitives_config(self):
        from src.llm_primitives import LLMPrimitivesConfig

        cfg = LLMConfig()
        ref = LLMPrimitivesConfig()

        assert cfg.output_cap == ref.output_cap
        assert cfg.batch_parallelism == ref.batch_parallelism
        assert cfg.call_timeout == ref.call_timeout
        assert cfg.mock_response_prefix == ref.mock_response_prefix
        assert cfg.max_recursion_depth == ref.max_recursion_depth
        assert cfg.default_prompt_rate == ref.default_prompt_rate
        assert cfg.default_completion_rate == ref.default_completion_rate

    def test_qwen_stop_token(self):
        from src.api.routes.chat_utils import QWEN_STOP

        cfg = LLMConfig()
        assert cfg.qwen_stop_token == QWEN_STOP


# ── ChatPipelineConfig defaults match chat_utils.py constants ────────────


class TestChatPipelineDefaults:
    """Verify ChatPipelineConfig defaults match THREE_STAGE_CONFIG etc."""

    def test_three_stage_config_values(self):
        from src.api.routes.chat_utils import THREE_STAGE_CONFIG

        cfg = ChatPipelineConfig()
        assert cfg.summarization_threshold_tokens == THREE_STAGE_CONFIG["threshold_tokens"]
        assert cfg.multi_doc_discount == THREE_STAGE_CONFIG["multi_doc_discount"]
        assert cfg.compression_enabled == THREE_STAGE_CONFIG["compression"]["enabled"]
        assert cfg.compression_min_chars == THREE_STAGE_CONFIG["compression"]["min_chars"]
        assert cfg.compression_target_ratio == THREE_STAGE_CONFIG["compression"]["target_ratio"]
        assert cfg.stage1_context_limit == THREE_STAGE_CONFIG["compression"]["stage1_context_limit"]

    def test_long_context_config_values(self):
        from src.api.routes.chat_utils import LONG_CONTEXT_CONFIG

        cfg = ChatPipelineConfig()
        assert cfg.long_context_enabled == LONG_CONTEXT_CONFIG["enabled"]
        assert cfg.long_context_threshold_chars == LONG_CONTEXT_CONFIG["threshold_chars"]
        assert cfg.long_context_max_turns == LONG_CONTEXT_CONFIG["max_turns"]

    def test_quality_detection_defaults(self):
        cfg = ChatPipelineConfig()
        assert cfg.repetition_unique_ratio == 0.5
        assert cfg.garbled_short_line_ratio == 0.6
        assert cfg.min_answer_length == 50

    def test_review_q_thresholds(self):
        cfg = ChatPipelineConfig()
        assert cfg.review_skip_q_threshold == 0.6
        # RI-18c removed the answer review gate and its threshold.
        assert not hasattr(cfg, "review_low_q_threshold")

    def test_plan_review_phase_defaults(self):
        cfg = ChatPipelineConfig()
        assert cfg.plan_review_phase_a_min == 50
        assert cfg.plan_review_phase_b_mean_q == 0.7
        assert cfg.plan_review_phase_b_min_q == 0.5
        assert cfg.plan_review_phase_c_min_q == 0.7
        assert cfg.plan_review_phase_c_min_total == 100
        assert cfg.plan_review_phase_c_skip_rate == 0.90

    def test_try_cheap_first_role_default(self):
        cfg = ChatPipelineConfig()
        assert cfg.try_cheap_first_role == "worker_general"

    def test_try_cheap_first_q_threshold_default(self):
        cfg = ChatPipelineConfig()
        assert cfg.try_cheap_first_q_threshold == 0.65

    def test_get_config_reads_try_cheap_first_q_threshold(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_CHAT_TRY_CHEAP_FIRST_Q_THRESHOLD", "0.78")
        reset_config()

        cfg = get_config()

        assert cfg.chat.try_cheap_first_q_threshold == 0.78

    def test_get_config_reads_legacy_quality_threshold_alias_for_q_threshold(self, monkeypatch):
        monkeypatch.delenv("ORCHESTRATOR_CHAT_TRY_CHEAP_FIRST_Q_THRESHOLD", raising=False)
        monkeypatch.setenv("ORCHESTRATOR_CHAT_TRY_CHEAP_FIRST_QUALITY_THRESHOLD", "0.74")
        reset_config()

        cfg = get_config()

        assert cfg.chat.try_cheap_first_q_threshold == 0.74


# ── VisionConfig defaults match src/vision/config.py ─────────────────────


class TestVisionDefaults:
    """Verify VisionConfig defaults match vision/config.py constants."""

    def test_processing_limits(self):
        cfg = VisionConfig()
        assert cfg.max_image_size_mb == 20
        assert cfg.max_image_dimension == 4096
        assert cfg.default_batch_size == 100
        assert cfg.max_concurrent_workers == 4
        assert cfg.default_video_fps == 1.0
        assert cfg.default_vl_max_tokens == 1024
        assert cfg.default_vl_threads == 8

    def test_thumbnail_settings(self):
        cfg = VisionConfig()
        assert cfg.thumb_size == (256, 256)
        assert cfg.thumb_quality == 85

    def test_face_detection(self):
        cfg = VisionConfig()
        assert cfg.face_min_confidence == 0.9
        assert cfg.face_embedding_dim == 512
        assert cfg.face_identification_threshold == 0.6

    def test_model_names(self):
        cfg = VisionConfig()
        assert cfg.arcface_model_name == "buffalo_l"
        assert cfg.clip_model_name == "ViT-B/32"
        assert cfg.sentence_transformer_model == "all-MiniLM-L6-v2"

    @_on_raid
    def test_paths_on_raid(self):
        cfg = VisionConfig()
        assert str(cfg.base_dir).startswith("/mnt/raid0/")
        assert str(cfg.llama_mtmd_cli).startswith("/mnt/raid0/")
        assert str(cfg.vl_model_path).startswith("/mnt/raid0/")

    def test_server_ports(self):
        cfg = VisionConfig()
        # Derived from the registry, which declares vision_escalation inside
        # worker_vision's `shared_with` (2026-08-01 W1 vision unification: the
        # standalone 7B on :8087 was retired and the role became an alias on the
        # 30B-A3B MI210 process). A pinned literal here would point the offline
        # batch analyzer at a port that no longer exists.
        assert cfg.vl_server_port == _registry_serving_port("worker_vision")
        assert cfg.vl_escalation_server_port == _registry_serving_port("vision_escalation")


# ── DelegationConfig defaults ────────────────────────────────────────────


class TestDelegationDefaults:
    def test_iteration_limits(self):
        cfg = DelegationConfig()
        assert cfg.max_iterations == 3
        assert cfg.max_total_iterations == 10
        assert cfg.max_concurrent_analysis == 4

    def test_token_limits(self):
        cfg = DelegationConfig()
        assert cfg.max_review_tokens == 128
        assert cfg.max_taskir_tokens == 256
        assert cfg.max_plan_review_tokens == 128


# ── ServicesConfig defaults ──────────────────────────────────────────────


class TestServicesDefaults:
    @_on_raid
    def test_ocr_model_paths_on_raid(self):
        cfg = ServicesConfig()
        assert str(cfg.lightonocr_model).startswith("/mnt/raid0/")
        assert str(cfg.lightonocr_mmproj).startswith("/mnt/raid0/")

    def test_ocr_max_tokens(self):
        cfg = ServicesConfig()
        assert cfg.lightonocr_max_tokens == 2048

    @_on_raid
    def test_draft_cache(self):
        cfg = ServicesConfig()
        assert str(cfg.draft_cache_dir).startswith("/mnt/raid0/")
        assert cfg.draft_cache_ttl_hours == 24.0

    def test_archive_limits(self):
        cfg = ServicesConfig()
        assert cfg.max_archive_size == 500 * 1024 * 1024
        assert cfg.max_extracted_size == 1024 * 1024 * 1024
        assert cfg.max_archive_files == 1000


# ── PathsConfig defaults ─────────────────────────────────────────────────


class TestPathsDefaults:
    @_on_raid
    def test_all_paths_on_raid(self):
        cfg = PathsConfig()
        for field_name in [
            "models_dir",
            "cache_dir",
            "tmp_dir",
            "registry_path",
            "tool_registry_path",
            "script_registry_dir",
            "project_root",
            "sessions_dir",
            "artifacts_dir",
            "llama_cpp_bin",
            "model_base",
            "log_dir",
        ]:
            path = getattr(cfg, field_name)
            assert str(path).startswith("/mnt/raid0/"), (
                f"PathsConfig.{field_name} = {path} is NOT on /mnt/raid0/"
            )

    @_on_raid
    def test_raid_prefix(self):
        cfg = PathsConfig()
        assert cfg.raid_prefix == "/mnt/raid0/"

    @_on_raid
    def test_specific_paths(self):
        cfg = PathsConfig()
        expected_project_root = Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])
            )
        )
        expected_llm_root = Path(
            os.environ.get("ORCHESTRATOR_PATHS_LLM_ROOT", "/mnt/raid0/llm")
        )
        expected_models_dir = Path(
            os.environ.get("ORCHESTRATOR_PATHS_MODEL_BASE", str(expected_llm_root / "models"))
        )
        expected_sessions_dir = Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_SESSIONS_DIR",
                str(expected_project_root / "orchestration" / "repl_memory" / "sessions"),
            )
        )
        assert cfg.project_root == expected_project_root
        assert cfg.models_dir == expected_models_dir
        assert cfg.sessions_dir == expected_sessions_dir


# ── WorkerPoolPathsConfig defaults ───────────────────────────────────────


class TestWorkerPoolPathsDefaults:
    @_on_raid
    def test_paths_on_raid(self):
        cfg = WorkerPoolPathsConfig()
        assert str(cfg.llama_server_path).startswith("/mnt/raid0/")
        assert str(cfg.log_dir).startswith("/mnt/raid0/")
        assert str(cfg.model_base).startswith("/mnt/raid0/")


# ── Environment variable overrides ───────────────────────────────────────


class TestEnvVarOverrides:
    """Test that environment variables properly override config defaults."""

    def test_mock_mode_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_MOCK_MODE": "0"}):
            reset_config()
            cfg = get_config()
            assert cfg.mock_mode is False

    def test_llm_output_cap_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_LLM_OUTPUT_CAP": "4096"}):
            reset_config()
            cfg = get_config()
            assert cfg.llm.output_cap == 4096

    def test_llm_call_timeout_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_LLM_CALL_TIMEOUT": "600"}):
            reset_config()
            cfg = get_config()
            assert cfg.llm.call_timeout == 600

    def test_llm_depth_role_overrides_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_LLM_DEPTH_ROLE_OVERRIDES": "1:worker_math"}):
            reset_config()
            cfg = get_config()
            assert cfg.llm.depth_role_overrides == "1:worker_math"

    def test_llm_depth_override_max_depth_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_LLM_DEPTH_OVERRIDE_MAX_DEPTH": "5"}):
            reset_config()
            cfg = get_config()
            assert cfg.llm.depth_override_max_depth == 5

    def test_timeout_role_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_TIMEOUTS_ARCHITECT_GENERAL": "600"}):
            reset_config()
            cfg = get_config()
            assert cfg.timeouts.architect_general == 600
            assert cfg.timeouts.for_role("architect_general") == 600

    def test_server_url_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_SERVER_URLS_FRONTDOOR": "http://custom:9999"}):
            reset_config()
            cfg = get_config()
            assert cfg.server_urls.frontdoor == "http://custom:9999"

    def test_dar_cost_tau_alias_override(self):
        with patch.dict(os.environ, {"DAR_COST_TAU": "1.7"}):
            reset_config()
            cfg = get_config()
            assert cfg.memrl_retrieval.cost_tau == 1.7

        with patch.dict(
            os.environ,
            {
                "DAR_COST_TAU": "1.7",
                "ORCHESTRATOR_MEMRL_RETRIEVAL_COST_TAU": "2.3",
            },
        ):
            reset_config()
            cfg = get_config()
            assert cfg.memrl_retrieval.cost_tau == 2.3

    def test_server_url_defaults_come_from_stack_priors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Isolation, not expectation-loosening: the stack-priors producer is the
        # LOWEST-precedence one. On a host where the real fleet is up, the
        # runtime-facts producer (which reads
        # /mnt/raid0/llm/tmp/orchestrator_runtime_facts.json and TCP-probes the
        # ports it names) legitimately wins and this test never reaches the
        # producer it is named after. Same isolation pattern as
        # tests/unit/test_config.py::TestServerURLsConfig.
        monkeypatch.setenv("ORCHESTRATOR_IGNORE_RUNTIME_STACK_FACTS", "1")
        monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
        priors = tmp_path / "stack_priors.yaml"
        priors.write_text(
            """
roles:
  frontdoor:
    deployment_status: live_stack
    serving:
      ports: [9100, 9200]
  worker_general:
    deployment_status: live_stack
    serving:
      ports: [9400, 9500]
""".lstrip(),
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"ORCHESTRATOR_PATHS_STACK_PRIORS_PATH": str(priors)}):
            reset_config()
            cfg = get_config()
            assert cfg.server_urls.frontdoor == "full:http://localhost:9100,http://localhost:9200"
            assert cfg.server_urls.worker == "full:http://localhost:9400,http://localhost:9500"

    def test_server_url_defaults_use_manifest_for_service_and_warm_compat(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from scripts.server import stack_manifest

        missing_priors = tmp_path / "missing-stack-priors.yaml"
        monkeypatch.setitem(stack_manifest.PORT_MAP, "worker_fast", 9902)
        monkeypatch.setitem(stack_manifest.PORT_MAP, "orchestrator", 9900)
        monkeypatch.setitem(stack_manifest.PORT_MAP, "document_formalizer", 9901)

        with patch.dict(
            os.environ,
            {"ORCHESTRATOR_PATHS_STACK_PRIORS_PATH": str(missing_priors)},
        ):
            reset_config()
            cfg = get_config()
            assert cfg.server_urls.worker_fast == "http://localhost:9902"
            assert cfg.server_urls.worker_coder == "http://localhost:9902"
            assert cfg.server_urls.worker_explore == cfg.server_urls.worker_general
            assert cfg.server_urls.api_url == "http://localhost:9900"
            assert cfg.server_urls.ocr_server == "http://localhost:9901"
            assert cfg.server_urls.vision_api == "http://localhost:9900/v1/vision/analyze"

    def test_monitor_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_MONITOR_ENTROPY_THRESHOLD": "5.0"}):
            reset_config()
            cfg = get_config()
            assert cfg.monitor.entropy_threshold == 5.0

    def test_chat_pipeline_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_CHAT_LONG_CONTEXT_MAX_TURNS": "16"}):
            reset_config()
            cfg = get_config()
            assert cfg.chat.long_context_max_turns == 16

    def test_chat_threshold_overrides(self):
        with patch.dict(
            os.environ,
            {
                "ORCHESTRATOR_CHAT_LONG_CONTEXT_THRESHOLD_CHARS": "64000",
                "ORCHESTRATOR_CHAT_SUMMARIZATION_THRESHOLD_TOKENS": "24000",
                "ORCHESTRATOR_CHAT_REVIEW_SKIP_Q_THRESHOLD": "0.72",
            },
        ):
            reset_config()
            cfg = get_config()
            assert cfg.chat.long_context_threshold_chars == 64000
            assert cfg.chat.summarization_threshold_tokens == 24000
            assert cfg.chat.review_skip_q_threshold == 0.72

    def test_manual_env_loader_reads_review_thresholds(self):
        from src.config import _load_from_env

        with patch.dict(
            os.environ,
            {
                "ORCHESTRATOR_CHAT_REVIEW_SKIP_Q_THRESHOLD": "0.76",
            },
        ):
            cfg = _load_from_env()
            assert cfg.chat.review_skip_q_threshold == 0.76

    def test_escalation_max_retries_override(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_ESCALATION_MAX_RETRIES": "5"}):
            reset_config()
            cfg = get_config()
            assert cfg.escalation.max_retries == 5


# ── End-to-end wiring verification ────────────────────────────────────────
# These tests verify that consumer modules ACTUALLY read from get_config().


class TestWiringChatUtils:
    """Verify chat_utils.py reads from config at module level."""

    def test_role_timeouts_from_config(self):
        """ROLE_TIMEOUTS dict must match config timeouts."""
        from src.api.routes.chat_utils import ROLE_TIMEOUTS

        cfg = get_config()
        for role, timeout in ROLE_TIMEOUTS.items():
            assert cfg.timeouts.for_role(role) == timeout

    def test_default_timeout_s_from_config(self):
        from src.api.routes.chat_utils import DEFAULT_TIMEOUT_S

        assert DEFAULT_TIMEOUT_S == get_config().timeouts.default_request

    def test_three_stage_config_sourced(self):
        from src.api.routes.chat_utils import THREE_STAGE_CONFIG

        cfg = get_config().chat
        assert THREE_STAGE_CONFIG["threshold_tokens"] == cfg.summarization_threshold_tokens

    def test_qwen_stop_from_config(self):
        from src.api.routes.chat_utils import QWEN_STOP

        assert QWEN_STOP == get_config().llm.qwen_stop_token


class TestWiringChatReview:
    """Verify chat_review.py reads from config."""

    def test_compute_plan_review_phase_uses_config(self):
        """_compute_plan_review_phase must use ChatPipelineConfig thresholds."""
        from src.api.routes.chat_review import _compute_plan_review_phase

        # Phase A: less than 50 reviews
        assert _compute_plan_review_phase({"total_reviews": 10}) == "A"
        # Phase A: exactly at threshold
        assert _compute_plan_review_phase({"total_reviews": 49}) == "A"
        # Phase A: >= 50 but no Q values
        assert _compute_plan_review_phase({"total_reviews": 50}) == "A"

    def test_detect_output_quality_uses_config_thresholds(self):
        """_detect_output_quality_issue uses repetition/garbled thresholds from config."""
        from src.api.routes.chat_review import _detect_output_quality_issue

        # A string with high repetition should be detected
        repeated = " ".join(["the same thing"] * 50)
        result = _detect_output_quality_issue(repeated)
        assert result is not None and "high_repetition" in result


class TestWiringChatVision:
    """Verify chat_vision.py reads URLs from config."""

    def test_vision_module_imports_config(self):
        """chat_vision.py must import get_config."""
        import src.api.routes.chat_vision as cv
        import inspect

        source = inspect.getsource(cv)
        assert "get_config" in source

    def test_execute_vision_tool_uses_config_ocr_url(self):
        """_execute_vision_tool references config for OCR URL."""
        import src.api.routes.chat_vision as cv
        import inspect

        source = inspect.getsource(cv._execute_vision_tool)
        assert "get_config" in source


class TestWiringRegistryLoader:
    """Verify registry_loader.py sources defaults from config."""

    @_on_raid
    def test_default_model_base_from_config(self):
        """RegistryLoader._model_base_path should match config.paths.model_base."""
        cfg = get_config()
        # The default should match — we can't instantiate RegistryLoader without
        # a valid YAML file, so we check the config value directly
        expected_llm_root = Path(
            os.environ.get("ORCHESTRATOR_PATHS_LLM_ROOT", "/mnt/raid0/llm")
        )
        expected_model_base = Path(
            os.environ.get(
                "ORCHESTRATOR_PATHS_MODEL_BASE",
                str(expected_llm_root / "models"),
            )
        )
        assert cfg.paths.model_base == expected_model_base

    @_on_raid
    def test_default_registry_path_from_config(self):
        cfg = get_config()
        assert cfg.paths.registry_path == (
            Path(__file__).resolve().parents[2] / "orchestration" / "model_registry.yaml"
        )


class TestWiringBuiltinTools:
    """Verify builtin_tools.py uses config for security prefix."""

    @_on_raid
    def test_raid_prefix_in_config(self):
        cfg = get_config()
        assert cfg.paths.raid_prefix == "/mnt/raid0/"

    def test_write_json_security_check_uses_config(self):
        """write_json tool should reference config for raid_prefix."""
        import src.builtin_tools as bt
        import inspect

        source = inspect.getsource(bt)
        assert "get_config" in source


class TestWiringBackends:
    """Verify backend modules source defaults from config."""

    def test_server_config_defaults_from_config(self):
        """ServerConfig fields should match config.server defaults."""
        from src.backends.llama_server import ServerConfig

        sc = ServerConfig()
        cfg = get_config().server
        assert sc.base_url == cfg.default_url
        assert sc.timeout == cfg.timeout
        assert sc.num_slots == cfg.num_slots

    def test_monitor_config_defaults_from_config(self):
        """MonitorConfig fields should match config.monitor defaults."""
        from src.generation_monitor import MonitorConfig

        mc = MonitorConfig()
        cfg = get_config().monitor
        assert mc.entropy_threshold == cfg.entropy_threshold
        assert mc.repetition_threshold == cfg.repetition_threshold
        assert mc.ngram_size == cfg.ngram_size

    def test_escalation_config_from_config(self):
        """EscalationConfig should match config.escalation."""
        from src.escalation import EscalationConfig

        ec = EscalationConfig()
        cfg = get_config().escalation
        assert ec.max_retries == cfg.max_retries
        assert ec.max_escalations == cfg.max_escalations


class TestWiringServices:
    """Verify service modules source defaults from config."""

    def test_document_client_urls_from_config(self):
        """document_client.py DEFAULT_OCR_URL must match config."""
        from src.services.document_client import DEFAULT_OCR_URL

        assert DEFAULT_OCR_URL == get_config().server_urls.ocr_server

    def test_vision_config_values_from_config(self):
        """vision/config.py constants must match VisionConfig."""
        from src.vision.config import MAX_IMAGE_SIZE_MB, MAX_IMAGE_DIMENSION

        cfg = get_config().vision
        assert MAX_IMAGE_SIZE_MB == cfg.max_image_size_mb
        assert MAX_IMAGE_DIMENSION == cfg.max_image_dimension
