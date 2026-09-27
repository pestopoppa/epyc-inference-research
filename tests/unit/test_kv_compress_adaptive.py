"""Tests for layer-adaptive KV compression (NIB2-20)."""

from scripts.autopilot.kv_compress import (
    compute_layer_adaptive_weights,
    LAYER_PROFILES,
    MODEL_LAYER_COUNT_ALIASES,
    MODEL_LAYER_COUNTS,
    PRODUCTION_PORTS,
    production_ports,
    production_ports_from_stack_priors,
    _layer_count_for_role,
    _stack_prior_layer_count_for_role,
    live_stack_role_records,
)
import scripts.autopilot.kv_compress as kv_compress

LEGACY_ARCHITECT_ROLE = "architect" "_coding"


class TestComputeLayerAdaptiveWeights:
    """Test per-layer weight computation."""

    def test_balanced_profile_28_layers(self):
        """28-layer model with balanced profile."""
        w = compute_layer_adaptive_weights(28, "balanced")
        assert len(w) == 28
        # Early third: weight 1.0
        assert w[0] == 1.0
        assert w[8] == 1.0
        # Mid third: weight 3.0
        assert w[10] == 3.0
        # Deep third: weight 10.0
        assert w[27] == 10.0

    def test_aggressive_profile(self):
        """Aggressive profile has lower early weights."""
        w = compute_layer_adaptive_weights(30, "aggressive")
        assert w[0] == 0.5  # early
        assert w[15] == 1.0  # mid
        assert w[29] == 5.0  # deep

    def test_conservative_profile(self):
        """Conservative profile has moderate weights."""
        w = compute_layer_adaptive_weights(30, "conservative")
        assert w[0] == 1.0  # early
        assert w[15] == 2.0  # mid
        assert w[29] == 5.0  # deep

    def test_unknown_profile_falls_back_to_balanced(self):
        """Unknown profile name uses balanced as default."""
        w = compute_layer_adaptive_weights(30, "unknown_profile")
        w_balanced = compute_layer_adaptive_weights(30, "balanced")
        assert w == w_balanced

    def test_small_model_3_layers(self):
        """Minimum viable: 3 layers (1 per zone)."""
        w = compute_layer_adaptive_weights(3, "balanced")
        assert len(w) == 3
        assert w[0] == 1.0   # early
        assert w[1] == 3.0   # mid
        assert w[2] == 10.0  # deep

    def test_single_layer(self):
        """Edge case: 1 layer."""
        w = compute_layer_adaptive_weights(1, "balanced")
        assert len(w) == 1

    def test_deep_layers_always_highest_weight(self):
        """Deep layers always get the highest weight in all profiles."""
        for profile in LAYER_PROFILES:
            w = compute_layer_adaptive_weights(30, profile)
            assert w[-1] >= w[0]
            assert w[-1] >= w[15]

    def test_all_known_models_produce_valid_weights(self):
        """Every model in MODEL_LAYER_COUNTS produces valid weights."""
        for role, n_layers in MODEL_LAYER_COUNTS.items():
            w = compute_layer_adaptive_weights(n_layers, "balanced")
            assert len(w) == n_layers
            assert all(v > 0 for v in w)

    def test_live_shared_role_aliases_reuse_current_layer_count(self):
        """Shared runtimes must not carry stale independent layer counts.

        The peer each alias points at is DERIVED from the live stack priors
        (the role it actually shares a serving endpoint with), not restated:
        shared runtimes get re-aliased between lineups, and a stale entry in
        the degraded fallback table silently builds a wrong-length weight
        vector for the alias role.
        """
        records = live_stack_role_records()

        def _endpoint(role: str) -> str:
            record = records.get(role) or {}
            serving = record.get("serving") or {}
            return str(serving.get("endpoint") or "")

        assert MODEL_LAYER_COUNT_ALIASES, "no shared-runtime aliases declared"
        for alias, declared_peer in MODEL_LAYER_COUNT_ALIASES.items():
            endpoint = _endpoint(alias)
            assert endpoint, f"{alias} is aliased but not live in stack priors"

            # Roles with their own fallback layer count that serve the SAME
            # llama-server instance as this alias.
            peers = sorted(r for r in MODEL_LAYER_COUNTS if _endpoint(r) == endpoint)
            assert peers, f"{alias} shares {endpoint} with no known-layer role"
            assert declared_peer in peers, (
                f"{alias} is aliased to {declared_peer!r} but serves {endpoint}, "
                f"shared with {peers}"
            )

            # Both paths (priors and degraded fallback) must agree.
            stack_prior_layers = _stack_prior_layer_count_for_role(alias)
            assert stack_prior_layers == MODEL_LAYER_COUNTS[declared_peer]
            assert _layer_count_for_role(alias) == stack_prior_layers

    def test_current_live_roles_expose_stack_prior_layer_counts(self):
        """Current live KV-adaptive roles should use generated stack-prior metadata."""
        assert _stack_prior_layer_count_for_role("frontdoor") == MODEL_LAYER_COUNTS["frontdoor"]
        # 2026-09-27 ARCHITECT SWAP: the 64-layer MI210 27B host is architect_critic.
        assert (
            _stack_prior_layer_count_for_role("architect_critic")
            == MODEL_LAYER_COUNTS["architect_critic"]
        )
        assert (
            _stack_prior_layer_count_for_role("ingest_long_context")
            == MODEL_LAYER_COUNTS["ingest_long_context"]
        )

    def test_retired_architect_role_has_no_active_layer_count(self):
        """Retired roles fall back to uniform KV compression."""
        assert LEGACY_ARCHITECT_ROLE not in MODEL_LAYER_COUNTS
        assert LEGACY_ARCHITECT_ROLE not in MODEL_LAYER_COUNT_ALIASES
        assert _layer_count_for_role(LEGACY_ARCHITECT_ROLE) is None

    def test_layer_count_prefers_stack_prior_metadata(self, tmp_path):
        """Stack priors are the canonical path for live model architecture metadata."""
        priors = tmp_path / "stack_priors.yaml"
        priors.write_text(
            """
roles:
  frontdoor:
    deployment_status: live_stack
    model:
      n_layers: 64
      attention_layers: 16
""".lstrip(),
            encoding="utf-8",
        )

        assert _stack_prior_layer_count_for_role("frontdoor", priors) == 16
        assert _layer_count_for_role("frontdoor", priors) == 16

    def test_layer_count_falls_back_when_stack_prior_metadata_missing(self, tmp_path):
        priors = tmp_path / "stack_priors.yaml"
        priors.write_text(
            """
roles:
  frontdoor:
    deployment_status: live_stack
    model: {}
""".lstrip(),
            encoding="utf-8",
        )

        assert _stack_prior_layer_count_for_role("frontdoor", priors) is None
        assert _layer_count_for_role("frontdoor", priors) == MODEL_LAYER_COUNTS["frontdoor"]

    def test_production_ports_use_live_role_names(self):
        """One entry per PHYSICAL server, keyed by the live host role name.

        2026-09-26: this pinned ``ingest_long_context -> 8085``, its own fleet.
        The operator-signed 2026-09-22 cutover (orchestrator 860b0b2d) made it
        an alias of architect_general's :8083 GPU process (and worker_general
        an alias of frontdoor's :8070 fleet), so like every other alias it must
        NOT get its own compress-all entry. The expected hosts and ports are
        now read from the registry's server_mode instead of restated.
        """
        from pathlib import Path

        import yaml

        registry = Path(__file__).resolve().parents[2] / "orchestration" / "model_registry.yaml"
        server_mode = yaml.safe_load(registry.read_text(encoding="utf-8"))["server_mode"]
        aliases = {r for r, row in server_mode.items() if row.get("alias_of")}
        assert {"ingest_long_context", "coder_escalation"} <= aliases, aliases  # non-vacuity

        assert "coder" not in PRODUCTION_PORTS
        assert "worker" not in PRODUCTION_PORTS
        assert LEGACY_ARCHITECT_ROLE not in PRODUCTION_PORTS
        for alias in aliases:
            assert alias not in PRODUCTION_PORTS, f"alias {alias} got its own port"
        # shared_with members ride their host too.
        for row in server_mode.values():
            for member in row.get("shared_with") or []:
                assert member not in PRODUCTION_PORTS, member
        assert "worker_general" not in PRODUCTION_PORTS
        assert "worker_vision" not in PRODUCTION_PORTS
        assert "vision_escalation" not in PRODUCTION_PORTS
        # Every entry is a registry HOST row, on the registry's own port.
        assert {"frontdoor", "architect_general"} <= set(PRODUCTION_PORTS), PRODUCTION_PORTS
        for role, port in PRODUCTION_PORTS.items():
            assert role in server_mode and role not in aliases, role
            assert port == server_mode[role]["port"], role

    def test_production_ports_from_stack_priors_use_primary_physical_ports(self, tmp_path):
        priors = tmp_path / "stack_priors.yaml"
        priors.write_text(
            """
roles:
  frontdoor:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8070
      binary: llama.cpp
      launch:
        runtime:
          cache:
            slot_save_path: /tmp/frontdoor-slots
        entries:
          - {port: 8070, alias: false}
          - {port: 8080, alias: false}
  coder_escalation:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8070
      binary: llama.cpp
      launch:
        runtime:
          cache:
            slot_save_path: /tmp/frontdoor-slots
        entries:
          - {port: 8070, alias: true}
  worker_general:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8072
      binary: ik-pr1744
      launch:
        runtime:
          binary_path: /mnt/raid0/llm/ik_llama.cpp/build/bin/llama-server
          cache:
            slot_save_path: null
        entries:
          - {port: 8072, alias: false}
  candidate:
    deployment_status: benchmark_only
    serving:
      endpoint: http://localhost:8099
      binary: llama.cpp
      launch:
        entries:
          - {port: 8099, alias: false}
  embedder:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8090
      binary: embedding-server
      launch:
        entries:
          - {port: 8090, alias: false}
""".lstrip(),
            encoding="utf-8",
        )

        assert production_ports_from_stack_priors(priors) == {
            "frontdoor": 8070,
        }

    def test_production_ports_from_stack_priors_can_include_aliases(self, tmp_path):
        priors = tmp_path / "stack_priors.yaml"
        priors.write_text(
            """
roles:
  frontdoor:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8070
      binary: llama.cpp
      launch:
        runtime:
          cache:
            slot_save_path: /tmp/frontdoor-slots
        entries:
          - {port: 8070, alias: false}
  coder_escalation:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8070
      binary: llama.cpp
      launch:
        runtime:
          cache:
            slot_save_path: /tmp/frontdoor-slots
        entries:
          - {port: 8070, alias: true}
  worker_math:
    deployment_status: live_stack
    serving:
      endpoint: http://localhost:8072 ik-pr1744
      binary: ik-pr1744
      launch:
        runtime:
          binary_path: /mnt/raid0/llm/ik_llama.cpp/build/bin/llama-server
          cache:
            slot_save_path: null
        entries:
          - {port: 8072, alias: true}
""".lstrip(),
            encoding="utf-8",
        )

        assert production_ports_from_stack_priors(priors, include_aliases=True) == {
            "coder_escalation": 8070,
            "frontdoor": 8070,
        }

    def test_fallback_production_ports_derive_from_stack_manifest(self, monkeypatch):
        monkeypatch.setattr(kv_compress, "HOT_ROLES", ("frontdoor", "coder_escalation", "worker_general"))
        monkeypatch.setattr(
            kv_compress,
            "PORT_MAP",
            {"frontdoor": 8070, "coder_escalation": 8070, "worker_general": 8072},
        )
        monkeypatch.setattr(
            kv_compress,
            "ROLE_LAUNCH_META",
            {
                "frontdoor": {"mode": "default"},
                "coder_escalation": {"mode": "default"},
                "worker_general": {"mode": "default"},
            },
        )

        assert kv_compress._fallback_production_ports_from_stack_manifest() == {
            "frontdoor": 8070,
            "coder_escalation": 8070,
            "worker_general": 8072,
        }

    def test_production_ports_falls_back_when_stack_priors_missing(self, monkeypatch):
        monkeypatch.setattr(
            "scripts.autopilot.kv_compress.production_ports_from_stack_priors",
            lambda include_aliases=False: {},
        )

        assert production_ports() == kv_compress.degraded_production_ports()
