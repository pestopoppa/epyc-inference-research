from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts.autopilot.species.structural_lab import StructuralLab
from scripts.server import orchestrator_stack
from src import features as feature_module
from src.api.routes.config import attest_config, update_config
from src.features import (
    Features,
    feature_sources,
    features,
    get_features,
    reset_features,
    runtime_flag_overrides,
    write_runtime_flag_overrides,
)


def test_runtime_flag_file_overrides_env(monkeypatch, tmp_path) -> None:
    runtime_path = tmp_path / "runtime_flags.json"
    monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(runtime_path))
    monkeypatch.setenv("ORCHESTRATOR_SPECIALIST_ROUTING", "0")
    reset_features()

    write_runtime_flag_overrides(
        {"specialist_routing": True},
        set_by="unit-test",
    )

    assert get_features().specialist_routing is True
    assert runtime_flag_overrides() == {"specialist_routing": True}
    assert feature_sources()["specialist_routing"].startswith("runtime_file:")
    payload = json.loads(runtime_path.read_text())
    assert payload["flags"]["specialist_routing"]["set_by"] == "unit-test"


def test_expired_runtime_flag_uses_registry_baseline_and_warns(monkeypatch, tmp_path, caplog) -> None:
    runtime_path = tmp_path / "runtime_flags.json"
    runtime_path.write_text(json.dumps({
        "version": 1,
        "flags": {
            "repl_embedding_pool": {
                "value": True,
                "set_by": "experiment",
                "ts": "2026-01-01T00:00:00+00:00",
                "expires_at": "2000-01-01T00:00:00Z",
            }
        },
    }))
    monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(runtime_path))
    monkeypatch.delenv("ORCHESTRATOR_REPL_EMBEDDING_POOL", raising=False)
    monkeypatch.setenv("ORCHESTRATOR_FEATURE_REPL_EMBEDDING_POOL", "1")
    reset_features()

    assert get_features().repl_embedding_pool is True
    assert feature_sources()["repl_embedding_pool"] == "ORCHESTRATOR_FEATURE_REPL_EMBEDDING_POOL"
    monkeypatch.delenv("ORCHESTRATOR_FEATURE_REPL_EMBEDDING_POOL")
    assert get_features().repl_embedding_pool is False
    assert feature_sources()["repl_embedding_pool"] == "default_test"
    assert "Ignoring expired runtime flag override repl_embedding_pool" in caplog.text


def test_unexpired_runtime_flag_survives_simulated_restart(monkeypatch, tmp_path) -> None:
    runtime_path = tmp_path / "runtime_flags.json"
    expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    runtime_path.write_text(json.dumps({
        "version": 1,
        "flags": {
            "repl_embedding_pool": {
                "value": True,
                "set_by": "experiment",
                "ts": datetime.now(timezone.utc).isoformat(),
                "expires_at": expiry,
            }
        },
    }))
    monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(runtime_path))
    reset_features()

    assert get_features().repl_embedding_pool is True
    reset_features()  # a new process reads the same unexpired runtime record
    assert get_features().repl_embedding_pool is True


def test_expiring_runtime_flag_invalidates_feature_cache_without_file_write(monkeypatch, tmp_path) -> None:
    runtime_path = tmp_path / "runtime_flags.json"
    now = [datetime(2030, 1, 1, tzinfo=timezone.utc)]
    monotonic = [10.0]
    runtime_path.write_text(json.dumps({
        "version": 1,
        "flags": {
            "repl_embedding_pool": {
                "value": True,
                "set_by": "experiment",
                "ts": now[0].isoformat(),
                "expires_at": (now[0] + timedelta(seconds=2)).isoformat(),
            }
        },
    }))
    monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(runtime_path))
    monkeypatch.setattr(feature_module, "_utc_now", lambda: now[0])
    monkeypatch.setattr(feature_module.time, "monotonic", lambda: monotonic[0])
    monkeypatch.setattr(feature_module, "RUNTIME_FLAGS_TTL_S", 0.0)
    original_get_features = feature_module.get_features
    rebuilds = [0]

    def counted_get_features(**kwargs):
        rebuilds[0] += 1
        return original_get_features(**kwargs)

    monkeypatch.setattr(feature_module, "get_features", counted_get_features)
    reset_features()

    assert features().repl_embedding_pool is True
    now[0] += timedelta(seconds=3)
    monotonic[0] += 3
    assert features().repl_embedding_pool is False
    assert features().repl_embedding_pool is False
    assert rebuilds[0] == 2


def test_feature_namespace_env_avoids_legacy_settings_collision(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(
        "ORCHESTRATOR_RUNTIME_FLAGS_PATH",
        str(tmp_path / "missing-runtime-flags.json"),
    )
    monkeypatch.setenv("ORCHESTRATOR_FEATURE_REPL", "0")
    monkeypatch.delenv("ORCHESTRATOR_REPL", raising=False)
    reset_features()

    assert get_features().repl is False
    assert feature_sources()["repl"] == "ORCHESTRATOR_FEATURE_REPL"


def test_features_singleton_reloads_runtime_file_after_ttl(monkeypatch, tmp_path) -> None:
    runtime_path = tmp_path / "runtime_flags.json"
    monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(runtime_path))
    monkeypatch.setattr(feature_module, "RUNTIME_FLAGS_TTL_S", 0.0)
    reset_features()

    assert features().model_fallback is False
    write_runtime_flag_overrides({"model_fallback": True}, set_by="unit-test")

    assert features().model_fallback is True


def test_config_post_writes_runtime_file_and_attests(monkeypatch, tmp_path) -> None:
    runtime_path = tmp_path / "runtime_flags.json"
    monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(runtime_path))
    reset_features()

    class Request:
        client = SimpleNamespace(host="127.0.0.1")

        async def json(self):
            return {"model_fallback": True, "unknown_flag": True}

    response = asyncio.run(update_config(Request(), current=Features()))
    assert response["status"] == "ok"
    assert response["features"]["model_fallback"] is True

    attestation = asyncio.run(attest_config(current=features()))
    assert attestation["flags"]["model_fallback"] is True
    assert attestation["sources"]["model_fallback"].startswith("runtime_file:")


def test_experiment_pool_enable_gets_default_expiry_and_restore_clears_it(monkeypatch, tmp_path) -> None:
    runtime_path = tmp_path / "runtime_flags.json"
    monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(runtime_path))
    reset_features()

    class Request:
        client = SimpleNamespace(host="127.0.0.1")

        def __init__(self, body):
            self.body = body

        async def json(self):
            return self.body

    before = datetime.now(timezone.utc)
    enabled = asyncio.run(update_config(Request({"repl_embedding_pool": True, "specialist_routing": True}), current=Features()))
    all_records = json.loads(runtime_path.read_text())["flags"]
    record = all_records["repl_embedding_pool"]
    assert "expires_at" not in all_records["specialist_routing"]
    expiry = feature_module._parse_runtime_expiry(record["expires_at"])
    assert enabled["features"]["repl_embedding_pool"] is True
    assert expiry is not None
    assert before + timedelta(seconds=feature_module.EXPERIMENT_FLAG_TTL_S - 2) <= expiry
    assert expiry <= datetime.now(timezone.utc) + timedelta(seconds=feature_module.EXPERIMENT_FLAG_TTL_S + 2)

    asyncio.run(update_config(Request({"repl_embedding_pool": False}), current=features()))
    restored = json.loads(runtime_path.read_text())["flags"]["repl_embedding_pool"]
    assert restored["value"] is False
    assert "expires_at" not in restored


def test_explicit_runtime_ttl_replaces_experiment_default(monkeypatch, tmp_path) -> None:
    runtime_path = tmp_path / "runtime_flags.json"
    monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(runtime_path))
    reset_features()

    class Request:
        client = SimpleNamespace(host="127.0.0.1")

        async def json(self):
            return {"repl_embedding_pool": True, "ttl_s": 120}

    before = datetime.now(timezone.utc)
    asyncio.run(update_config(Request(), current=Features()))
    record = json.loads(runtime_path.read_text())["flags"]["repl_embedding_pool"]
    expiry = feature_module._parse_runtime_expiry(record["expires_at"])
    assert expiry is not None
    assert before + timedelta(seconds=118) <= expiry
    assert expiry <= datetime.now(timezone.utc) + timedelta(seconds=122)


def test_runtime_flag_writer_allows_long_ttls_and_rejects_datetime_overflow(tmp_path) -> None:
    runtime_path = tmp_path / "runtime_flags.json"
    feature_module.write_runtime_flag_overrides(
        {"specialist_routing": True}, ttl_s=30 * 24 * 60 * 60, path=runtime_path
    )
    record = json.loads(runtime_path.read_text())["flags"]["specialist_routing"]
    assert feature_module._parse_runtime_expiry(record["expires_at"]) > datetime.now(timezone.utc) + timedelta(days=29)

    with pytest.raises(ValueError, match="datetime range"):
        feature_module.write_runtime_flag_overrides(
            {"specialist_routing": True}, ttl_s=1e100, path=runtime_path
        )


def test_stack_production_feature_env_is_complete_and_wave_gated() -> None:
    env = orchestrator_stack._production_feature_env()

    assert env["ORCHESTRATOR_FEATURE_SPECIALIST_ROUTING"] == "1"
    assert env["ORCHESTRATOR_FEATURE_MODEL_FALLBACK"] == "1"
    assert env["ORCHESTRATOR_FEATURE_PLAN_REVIEW"] == "0"
    assert env["ORCHESTRATOR_FEATURE_ARCHITECT_DELEGATION"] == "0"
    assert env["ORCHESTRATOR_FEATURE_PARALLEL_EXECUTION"] == "0"
    assert env["ORCHESTRATOR_FEATURE_UNIFIED_STREAMING"] == "0"
    assert env["ORCHESTRATOR_FEATURE_ROUTING_CLASSIFIER"] == "0"
    assert env["ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING"] == "0"
    assert "ORCHESTRATOR_FEATURE_LANGGRAPH_ARCHITECT_CODING" not in env
    assert "ORCHESTRATOR_REPL" not in env
    assert "ORCHESTRATOR_LANGGRAPH_ARCHITECT_CODING" not in env
    for spec in feature_module._FEATURE_REGISTRY:
        assert f"ORCHESTRATOR_FEATURE_{spec.env_var}" in env


def test_stack_production_feature_env_preserves_launch_override() -> None:
    env = {"ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING": "1"}

    orchestrator_stack._apply_production_feature_env(env)

    assert env["ORCHESTRATOR_FEATURE_EVAL_BATCH_SERVING"] == "1"
    assert env["ORCHESTRATOR_FEATURE_SPECIALIST_ROUTING"] == "1"
    assert env["ORCHESTRATOR_FEATURE_PLAN_REVIEW"] == "0"


def test_stack_live_langgraph_env_excludes_retired_architect_coding() -> None:
    assert "ORCHESTRATOR_LANGGRAPH_ARCHITECT" in (
        orchestrator_stack.LANGGRAPH_PHASE3_LIVE_ENV_VARS
    )
    assert "ORCHESTRATOR_LANGGRAPH_ARCHITECT_CODING" not in (
        orchestrator_stack.LANGGRAPH_PHASE3_LIVE_ENV_VARS
    )


def test_retired_architect_coding_feature_env_is_ignored(monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRATOR_FEATURE_LANGGRAPH_ARCHITECT_CODING", "1")
    reset_features()

    current = get_features()

    assert "langgraph_architect_coding" not in current.summary()
    assert "langgraph_architect_coding" not in feature_sources()


def test_retired_architect_coding_runtime_flag_is_ignored(monkeypatch, tmp_path) -> None:
    runtime_path = tmp_path / "runtime_flags.json"
    runtime_path.write_text(
        json.dumps(
            {
                "flags": {
                    "model_fallback": {"value": True, "set_by": "unit-test"},
                    "langgraph_architect_coding": {"value": True, "set_by": "old-state"},
                }
            }
        )
    )
    monkeypatch.setenv("ORCHESTRATOR_RUNTIME_FLAGS_PATH", str(runtime_path))
    reset_features()

    current = get_features()

    assert current.model_fallback is True
    assert "langgraph_architect_coding" not in current.summary()
    assert "langgraph_architect_coding" not in feature_sources()


def test_structural_lab_uses_attest_for_current_flags(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"pid": 123, "flags": {"model_fallback": True}}

    monkeypatch.setattr("httpx.get", lambda *a, **k: Response())

    assert StructuralLab().current_flags() == {"model_fallback": True}


def test_apply_flag_experiment_returns_attestation(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"status": "ok", "features": {"model_fallback": True}}

    lab = StructuralLab()
    monkeypatch.setattr("httpx.post", lambda *a, **k: Response())
    monkeypatch.setattr(
        lab,
        "attest_flags",
        lambda expected: {"status": "ok", "expected": expected},
    )

    result = lab.apply_flag_experiment({"model_fallback": True})
    assert result["attestation"] == {
        "status": "ok",
        "expected": {"model_fallback": True},
    }
