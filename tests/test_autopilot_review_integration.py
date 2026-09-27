"""Reviewer control-plane <-> autopilot integration tests (H8 AP-1/4/5/6/7/8).

All synthetic/stubbed — zero inference, no live autopilot, no network. Every
assertion targets the additive + default-behavior-preserving contract: absent
review-plane data must leave existing autopilot behavior byte-identical.
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
for _p in (str(ROOT), str(AUTOPILOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

config_applicator = importlib.import_module("config_applicator")
review_policy_trials = importlib.import_module("review_policy_trials")
safety_gate = importlib.import_module("safety_gate")
actions = importlib.import_module("actions")
planner_providers = importlib.import_module("planner_providers")
digest = importlib.import_module("digest")

ca = config_applicator
rpt = review_policy_trials
EvalResult = safety_gate.EvalResult


def _review_grammar_available() -> bool:
    return rpt._load_review_grammar() is not None


def _fish_json_available() -> bool:
    return rpt._load_fish_json() is not None


# ══════════════════════════════════════════════════════════════════════════════
# AP-1 — knob registration + apply_params plumbing
# ══════════════════════════════════════════════════════════════════════════════

_EXPECTED_KNOBS = {
    "review_trigger_complexity_threshold",
    "max_review_iterations",
    "reminder_cadence",
    "per_subtask_review_enabled",
    "review_majority_k",
    "request_evidence_round_budget",
    "review_token_multiplier",
}

_EXPECTED_AXA3_KNOBS = {
    "teleport_enabled",
    "long_running_trigger_tokens",
    "rate_window_tokens",
    "min_resident_remaining_tokens",
    "min_speedup",
    "lease_interactive_weight",
    "lease_batch_weight",
    "lease_eval_weight",
}

_EXPECTED_AP3_ROLE_RESTART_KNOBS = {
    "frontdoor_spec_type",
    "frontdoor_draft_max",
    "frontdoor_draft_min",
    "frontdoor_draft_p_min",
    "frontdoor_draft_p_split",
    "frontdoor_ngram_mod_n_min",
    "frontdoor_ngram_mod_n_max",
    "frontdoor_ngram_mod_n_match",
    "frontdoor_kv_profile",
    "worker_spec_type",
    "worker_draft_max",
    "worker_draft_min",
    "worker_draft_p_min",
    "worker_draft_p_split",
    "worker_threads_draft",
    "worker_ngram_mod_n_min",
    "worker_ngram_mod_n_max",
    "worker_ngram_mod_n_match",
    "worker_kv_profile",
    "architect_spec_type",
    "architect_draft_max",
    "architect_draft_min",
    "architect_draft_p_min",
    "architect_draft_p_split",
    "architect_ngram_mod_n_min",
    "architect_ngram_mod_n_max",
    "architect_ngram_mod_n_match",
    "architect_kv_profile",
}


def test_ap1_all_class1_knobs_registered_with_bounds_and_restart_cost() -> None:
    assert set(ca.REVIEW_PLANE_KNOB_SPECS) == _EXPECTED_KNOBS
    for name, spec in ca.REVIEW_PLANE_KNOB_SPECS.items():
        assert spec.restart_cost in {"none", "api_restart", "role_restart"}
        assert spec.apply_key == f"delegation.{name}"
        if spec.kind != "bool":
            assert spec.lo is not None and spec.hi is not None
            assert spec.lo <= spec.hi


def test_ap1_normalize_translates_bare_knob_to_dotted_apply_key() -> None:
    out = ca.normalize_review_plane_params({"max_review_iterations": 4})
    assert out == {"delegation.max_review_iterations": 4}


def test_ap1_normalize_is_noop_for_non_review_params_regression() -> None:
    legacy = {"memrl_retrieval.q_weight": 0.5}
    out = ca.normalize_review_plane_params(legacy)
    # Same object returned (identity) => zero disturbance to legacy callers.
    assert out is legacy


def test_ap1_classify_routes_knob_to_env_restart_surface() -> None:
    dotted = ca.normalize_review_plane_params({"review_majority_k": 3})
    classified = ca.classify_params(dotted)
    assert classified["env_restart"] == {"delegation.review_majority_k": 3}
    assert classified["unknown"] == {}


def test_ap1_env_changes_map_to_orchestrator_env_var() -> None:
    dotted = ca.normalize_review_plane_params({"reminder_cadence": 7})
    env = ca.EnvRestartApplicator(restart=False).env_changes_for(dotted)
    assert env == {"ORCHESTRATOR_DELEGATION_REMINDER_CADENCE": "7"}


def test_ap1_validate_accepts_in_bounds_rejects_out_of_bounds() -> None:
    assert ca.validate_review_plane_params({"max_review_iterations": 4}) is None
    err = ca.validate_review_plane_params({"max_review_iterations": 99})
    assert err and "max 5" in err
    # dotted form validates too
    assert ca.validate_review_plane_params({"delegation.review_majority_k": 0}) is not None


def test_ap1_apply_params_dry_run_out_of_bounds_errors_without_applying() -> None:
    res = ca.apply_params({"max_review_iterations": 99}, dry_run=True)
    assert res["status"] == "error"
    assert any("review_plane" in e for e in res["errors"])
    # never classified/applied
    assert "classified" in res


def test_ap1_apply_params_dry_run_in_bounds_classifies_env_restart() -> None:
    res = ca.apply_params({"review_token_multiplier": 2.0}, dry_run=True)
    assert res["status"] == "ok"
    assert res["classified"]["env_restart"] == {"delegation.review_token_multiplier": 2.0}


def test_ap1_apply_params_regression_legacy_param_unaffected() -> None:
    res = ca.apply_params({"memrl_retrieval.q_weight": 0.5}, dry_run=True)
    assert res["status"] == "ok"
    assert res["classified"]["env_restart"] == {"memrl_retrieval.q_weight": 0.5}


def test_ap1_manifest_loader_graceful_missing_file_returns_builtins() -> None:
    merged = ca.load_review_plane_knob_manifest(Path("/does/not/exist.yaml"))
    assert set(merged) == _EXPECTED_KNOBS
    # built-in bounds aligned to W2c manifest v1
    assert merged["max_review_iterations"]["hi"] == 5


def test_ap1_manifest_loader_overlays_bounds_dict_form(tmp_path: Path) -> None:
    manifest = tmp_path / "review_plane_knobs.yaml"
    manifest.write_text(
        "knobs:\n  max_review_iterations:\n    hi: 20\n    default: 4\n",
        encoding="utf-8",
    )
    merged = ca.load_review_plane_knob_manifest(manifest)
    assert merged["max_review_iterations"]["hi"] == 20
    assert merged["max_review_iterations"]["default"] == 4
    # untouched knobs keep built-in bounds
    assert merged["review_majority_k"]["hi"] == 5


def test_ap1_manifest_loader_parses_w2c_list_format(tmp_path: Path) -> None:
    """Real W2c convention: list entries, review_plane. prefix, low/high/param_type."""
    manifest = tmp_path / "review_plane_knobs.yaml"
    manifest.write_text(
        "knobs:\n"
        "  - name: review_plane.review_majority_k\n"
        "    param_type: int\n"
        "    low: 1\n"
        "    high: 7\n"
        "    default: 1\n"
        "    restart_cost: none\n",
        encoding="utf-8",
    )
    merged = ca.load_review_plane_knob_manifest(manifest)
    # prefix stripped, low/high -> lo/hi, restart_cost declared "none"
    assert merged["review_majority_k"]["lo"] == 1
    assert merged["review_majority_k"]["hi"] == 7
    assert merged["review_majority_k"]["restart_cost"] == "none"


def test_ap1_loader_reads_real_w2c_manifest_if_present() -> None:
    real = ca.DEFAULT_REVIEW_PLANE_KNOB_MANIFEST
    if not real.exists():
        pytest.skip("W2c manifest not present in this tree")
    merged = ca.load_review_plane_knob_manifest()
    # every built-in AP-1 knob is represented; W2c declares restart_cost none.
    for knob in _EXPECTED_KNOBS:
        assert knob in merged


def test_ap1_seed_fixture_shape_not_written_to_store() -> None:
    entries = rpt.review_plane_seed_strategies()
    assert len(entries) == len(_EXPECTED_KNOBS)
    for entry in entries:
        assert entry["species"] == "numeric_swarm"
        assert entry["slug"].startswith("review-plane-")
        assert entry["bind_identifiers"][0].startswith("delegation.")


# ══════════════════════════════════════════════════════════════════════════════
# AP-2 — AXA-3 GPU placement / teleport policy registration
# ══════════════════════════════════════════════════════════════════════════════


def test_ap2_all_class3_knobs_registered_default_off() -> None:
    assert set(ca.AXA3_POLICY_KNOB_SPECS) == _EXPECTED_AXA3_KNOBS
    enabled = ca.AXA3_POLICY_KNOB_SPECS["teleport_enabled"]
    assert enabled.default is False
    assert enabled.apply_key == "placement_policy.teleport_enabled"
    for name, spec in ca.AXA3_POLICY_KNOB_SPECS.items():
        assert spec.section == "placement_policy"
        assert spec.restart_cost in {"none", "api_restart", "role_restart"}
        if spec.kind != "bool":
            assert spec.lo is not None and spec.hi is not None
            assert spec.lo <= spec.hi


def test_ap2_apply_params_dry_run_classifies_token_rate_break_even_and_weights() -> None:
    res = ca.apply_params(
        {
            "long_running_trigger_tokens": 256,
            "rate_window_tokens": 128,
            "min_resident_remaining_tokens": 250,
            "min_speedup": 1.25,
            "lease_interactive_weight": 1.5,
            "lease_batch_weight": 0.5,
            "lease_eval_weight": 0.2,
        },
        dry_run=True,
    )

    assert res["status"] == "ok"
    assert res["classified"]["env_restart"] == {
        "placement_policy.long_running_trigger_tokens": 256,
        "placement_policy.rate_window_tokens": 128,
        "placement_policy.min_resident_remaining_tokens": 250,
        "placement_policy.min_speedup": 1.25,
        "placement_policy.lease_interactive_weight": 1.5,
        "placement_policy.lease_batch_weight": 0.5,
        "placement_policy.lease_eval_weight": 0.2,
    }


def test_ap2_env_changes_map_to_placement_policy_env_vars() -> None:
    dotted = ca.normalize_axa3_policy_params(
        {
            "long_running_trigger_tokens": 192,
            "min_resident_remaining_tokens": 300,
            "lease_eval_weight": 0.4,
        }
    )
    env = ca.EnvRestartApplicator(restart=False).env_changes_for(dotted)

    assert env == {
        "ORCHESTRATOR_PLACEMENT_POLICY_LONG_RUNNING_TRIGGER_TOKENS": "192",
        "ORCHESTRATOR_PLACEMENT_POLICY_MIN_RESIDENT_REMAINING_TOKENS": "300",
        "ORCHESTRATOR_PLACEMENT_POLICY_LEASE_EVAL_WEIGHT": "0.4",
    }


def test_ap2_teleport_enable_true_requires_operator_env(monkeypatch) -> None:
    monkeypatch.delenv("AUTOPILOT_AXA3_TELEPORT_ENABLE", raising=False)

    res = ca.apply_params({"teleport_enabled": True}, dry_run=True)

    assert res["status"] == "error"
    assert "AUTOPILOT_AXA3_TELEPORT_ENABLE" in res["errors"][0]


def test_ap2_manifest_loader_reads_placement_policy_block(tmp_path: Path) -> None:
    manifest = tmp_path / "review_plane_knobs.yaml"
    manifest.write_text(
        "placement_policy_knobs:\n"
        "  - name: placement_policy.min_resident_remaining_tokens\n"
        "    param_type: int\n"
        "    low: 80\n"
        "    high: 900\n"
        "    default: 200\n",
        encoding="utf-8",
    )

    merged = ca.load_axa3_policy_knob_manifest(manifest)

    assert set(_EXPECTED_AXA3_KNOBS).issubset(merged)
    assert merged["min_resident_remaining_tokens"]["lo"] == 80
    assert merged["min_resident_remaining_tokens"]["hi"] == 900
    assert merged["min_resident_remaining_tokens"]["default"] == 200


# ══════════════════════════════════════════════════════════════════════════════
# AP-3 — spec-dec / KV launch role-restart registration
# ══════════════════════════════════════════════════════════════════════════════


def test_ap3_all_role_restart_knobs_registered() -> None:
    assert set(ca.AP3_ROLE_RESTART_KNOB_SPECS) == _EXPECTED_AP3_ROLE_RESTART_KNOBS
    for name, spec in ca.AP3_ROLE_RESTART_KNOB_SPECS.items():
        assert spec.section == "role_restart"
        assert spec.restart_cost == "role_restart"
        assert spec.apply_key == f"role_restart.{name}"
        # 2026-09-27 ARCHITECT SWAP: the architect_* launch knobs follow the
        # MI210 27B process, now architect_critic.
        assert spec.role in {"frontdoor", "worker_general", "architect_critic"}
        if spec.kind == "enum":
            assert spec.allowed_values
        else:
            assert spec.lo is not None and spec.hi is not None


def test_ap3_apply_params_dry_run_classifies_role_restart() -> None:
    res = ca.apply_params(
        {
            "worker_draft_max": 4,
            "worker_draft_min": 1,
            "worker_draft_p_min": 0.1,
            "worker_draft_p_split": 0.2,
            "worker_threads_draft": 24,
            "worker_ngram_mod_n_min": 16,
            "worker_ngram_mod_n_max": 32,
            "worker_ngram_mod_n_match": 8,
            "worker_spec_type": "ngram-mod,draft-mtp",
            "worker_kv_profile": "f16_f16",
        },
        dry_run=True,
    )

    assert res["status"] == "ok"
    assert res["classified"]["role_restart"] == {
        "role_restart.worker_draft_max": 4,
        "role_restart.worker_draft_min": 1,
        "role_restart.worker_draft_p_min": 0.1,
        "role_restart.worker_draft_p_split": 0.2,
        "role_restart.worker_threads_draft": 24,
        "role_restart.worker_ngram_mod_n_min": 16,
        "role_restart.worker_ngram_mod_n_max": 32,
        "role_restart.worker_ngram_mod_n_match": 8,
        "role_restart.worker_spec_type": "ngram-mod,draft-mtp",
        "role_restart.worker_kv_profile": "f16_f16",
    }


def test_ap3_invalid_spec_type_rejected_before_classification_apply() -> None:
    res = ca.apply_params({"worker_spec_type": "mtp"}, dry_run=True)

    assert res["status"] == "error"
    assert "role_restart" in res["errors"][0]
    assert "draft-mtp" in res["errors"][0]


def test_ap3_live_role_restart_requires_operator_env(monkeypatch) -> None:
    monkeypatch.delenv("AUTOPILOT_AP3_ROLE_RESTART_ENABLE", raising=False)

    res = ca.apply_role_restart_params({"role_restart.worker_draft_max": 4})

    assert res["status"] == "error"
    assert "AUTOPILOT_AP3_ROLE_RESTART_ENABLE" in res["errors"][0]
    assert res["per_role"] == {}


def test_ap3_role_restart_builds_registry_overrides_and_groups_by_role(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    ordering: list[str] = []
    monkeypatch.setenv("AUTOPILOT_AP3_ROLE_RESTART_ENABLE", "1")

    def fake_restart_role(**kwargs):
        ordering.append("restart")
        calls.append(kwargs)
        return {
            "status": "ok",
            "role": kwargs["role"],
            "registry_overrides": kwargs["registry_overrides"],
        }

    monkeypatch.setattr(ca, "restart_role", fake_restart_role)
    monkeypatch.setattr(
        ca,
        "_ap3_prewarm_role_targets",
        lambda *args, **kwargs: (
            ordering.append("prewarm")
            or {
                "status": "ok",
                "targets": [],
                "policy": "test",
            }
        ),
    )

    res = ca.apply_role_restart_params(
        {
            "worker_draft_max": 4,
            "worker_draft_min": 1,
            "worker_spec_type": "ngram-mod,draft-mtp",
            "worker_draft_p_split": 0.25,
            "worker_ngram_mod_n_match": 16,
            "worker_kv_profile": "f16_f16",
        },
        smoke_check=lambda _role, _affected_roles: {"status": "ok"},
    )

    assert res["status"] == "ok"
    assert len(calls) == 1
    assert ordering == ["prewarm", "restart"]
    assert calls[0]["role"] == "worker_general"
    assert calls[0]["pause_dispatch"] is True
    assert calls[0]["require_smoke_check"] is True
    assert calls[0]["boundary_evidence"] == {
        "ap3_prewarm": {"status": "ok", "targets": [], "policy": "test"}
    }
    assert calls[0]["registry_overrides"] == {
        "server_mode.worker.acceleration.draft_max": 4,
        "server_mode.worker.acceleration.draft_min": 1,
        "server_mode.worker.acceleration.spec_type": "ngram-mod,draft-mtp",
        "server_mode.worker.acceleration.draft_p_split": 0.25,
        "server_mode.worker.acceleration.ngram_mod_n_match": 16,
        "server_mode.worker.kv_quant": {"k": "f16", "v": "f16"},
    }


def test_ap3_prewarm_failure_prevents_registry_restart(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_AP3_ROLE_RESTART_ENABLE", "1")
    monkeypatch.setattr(
        ca,
        "_ap3_prewarm_role_targets",
        lambda *args, **kwargs: {"status": "error", "error": "unreadable model", "targets": []},
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        ca, "restart_role", lambda **kwargs: calls.append(kwargs) or {"status": "ok"}
    )

    result = ca.apply_role_restart_params({"role_restart.worker_draft_max": 4})

    assert result["status"] == "error"
    assert calls == []
    assert result["per_role"]["worker_general"]["ap3_prewarm"]["error"] == "unreadable model"


def test_ap3_readiness_probe_is_one_token_and_failure_is_strict(monkeypatch) -> None:
    prewarm = {"status": "ok", "targets": [{"ports": [8072], "path": "/m.gguf"}]}
    calls: list[tuple[str, dict[str, object]]] = []

    class Response:
        def raise_for_status(self) -> None:
            raise RuntimeError("not ready")

    def fake_post(url: str, *, json: dict[str, object], timeout: int):
        calls.append((url, json))
        return Response()

    monkeypatch.setattr(ca.httpx, "post", fake_post)
    check = ca._ap3_strict_smoke_with_readiness(
        smoke_check=lambda *_: {"status": "ok"}, prewarm=prewarm
    )

    result = check("worker_general", ["worker_general"])

    assert result["status"] == "error"
    assert calls == [
        ("http://127.0.0.1:8072/completion", {"prompt": "ready", "n_predict": 1, "temperature": 0})
    ]
    assert result["readiness"][0]["timing_excluded"] is True
    assert result["readiness"][0]["error_type"] == "RuntimeError"
    assert result["base_smoke"]["status"] == "ok"


def test_ap3_readiness_skips_probe_when_smoke_fails(monkeypatch) -> None:
    prewarm = {"status": "ok", "targets": [{"ports": [8072]}]}
    monkeypatch.setattr(
        ca.httpx, "post", lambda *args, **kwargs: pytest.fail("must not probe after smoke failure")
    )
    check = ca._ap3_strict_smoke_with_readiness(smoke_check=lambda *_: False, prewarm=prewarm)

    result = check("worker_general", ["worker_general"])

    assert result["status"] == "error"


def test_ap3_readiness_dedupes_ports_and_returns_evidence_reference(monkeypatch) -> None:
    prewarm = {
        "status": "ok",
        "evidence_ref": "ap3-numa-prewarm:worker_general:2026-07-27T00:00:00+00:00",
        "targets": [{"ports": [8072, 8072]}, {"ports": [8072, 8085]}],
    }
    calls: list[str] = []

    class Response:
        status_code = 200
        content = b'{"tokens_predicted":1,"content":"x"}'

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"tokens_predicted": 1, "content": "x"}

    def fake_post(url: str, **_kwargs: object) -> Response:
        calls.append(url)
        return Response()

    monkeypatch.setattr(ca.httpx, "post", fake_post)
    check = ca._ap3_strict_smoke_with_readiness(
        smoke_check=lambda *_: {"status": "ok"}, prewarm=prewarm
    )

    result = check("worker_general", ["worker_general"])

    assert result["status"] == "ok"
    assert calls == [
        "http://127.0.0.1:8072/completion",
        "http://127.0.0.1:8085/completion",
    ]
    assert result["prewarm_evidence_ref"] == prewarm["evidence_ref"]
    assert all(probe["timing_excluded"] for probe in result["readiness"])
    assert all(probe["token_count"] == 1 for probe in result["readiness"])
    assert all(probe["http_status"] == 200 for probe in result["readiness"])
    assert all(
        probe["prompt_sha256"] == sha256(b"ready").hexdigest() for probe in result["readiness"]
    )
    assert all(
        probe["response_sha256"] == sha256(Response.content).hexdigest()
        for probe in result["readiness"]
    )
    assert all(probe["elapsed_s"] >= 0 for probe in result["readiness"])
    assert result["base_smoke"]["status"] == "ok"


@pytest.mark.parametrize(
    ("payload", "error_type"),
    [
        ({"tokens_predicted": 0}, "ValueError"),
        ({"tokens_predicted": "one"}, "ValueError"),
    ],
)
def test_ap3_readiness_rejects_200_without_one_generated_token(
    monkeypatch, payload: dict[str, object], error_type: str
) -> None:
    class Response:
        status_code = 200
        content = b"bounded-response"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return payload

    monkeypatch.setattr(ca.httpx, "post", lambda *_args, **_kwargs: Response())
    check = ca._ap3_strict_smoke_with_readiness(
        smoke_check=lambda *_: {"status": "ok"},
        prewarm={"status": "ok", "targets": [{"ports": [8072]}]},
    )

    result = check("worker_general", ["worker_general"])

    assert result["status"] == "error"
    assert result["base_smoke"]["status"] == "ok"
    probe = result["readiness"][0]
    assert probe["port"] == 8072
    assert probe["prompt_sha256"] == sha256(b"ready").hexdigest()
    assert probe["http_status"] == 200
    assert probe["response_sha256"] == sha256(b"bounded-response").hexdigest()
    assert probe["status"] == "error"
    assert probe["error_type"] == error_type
    assert probe["elapsed_s"] >= 0


def test_ap3_readiness_rejects_malformed_json_with_bounded_evidence(monkeypatch) -> None:
    class Response:
        status_code = 200
        content = b"not-json"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            raise json.JSONDecodeError("bad", "not-json", 0)

    monkeypatch.setattr(ca.httpx, "post", lambda *_args, **_kwargs: Response())
    check = ca._ap3_strict_smoke_with_readiness(
        smoke_check=lambda *_: {"status": "ok"},
        prewarm={"status": "ok", "targets": [{"ports": [8072]}]},
    )

    result = check("worker_general", ["worker_general"])

    assert result["status"] == "error"
    probe = result["readiness"][0]
    assert probe["error_type"] == "JSONDecodeError"
    assert probe["response_sha256"] == sha256(b"not-json").hexdigest()
    assert "not-json" not in str(probe)


def test_ap3_prewarm_failure_writes_pre_reload_boundary_evidence(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_AP3_ROLE_RESTART_ENABLE", "1")
    monkeypatch.setattr(
        ca,
        "_ap3_prewarm_role_targets",
        lambda *args, **kwargs: {
            "status": "error",
            "error": "unreadable model",
            "affected_roles": ["worker_general"],
            "targets": [],
        },
    )
    events: list[dict[str, object]] = []

    class Journal:
        def append_role_restart_boundary_event(self, **kwargs: object) -> dict[str, object]:
            events.append(kwargs)
            return kwargs

    result = ca.apply_role_restart_params({"role_restart.worker_draft_max": 4}, journal=Journal())

    assert result["status"] == "error"
    assert events[0]["command"] == "ap3-numa-prewarm"
    assert events[0]["evidence"] == {
        "ap3_prewarm": result["per_role"]["worker_general"]["ap3_prewarm"]
    }


def test_ap3_manifest_loader_reads_role_restart_block(tmp_path: Path) -> None:
    manifest = tmp_path / "review_plane_knobs.yaml"
    manifest.write_text(
        "role_restart_knobs:\n"
        "  - name: role_restart.worker_draft_max\n"
        "    param_type: int\n"
        "    low: 1\n"
        "    high: 12\n"
        "    default: 3\n"
        "    role: worker_general\n"
        "    registry_path: server_mode.worker.acceleration.draft_max\n",
        encoding="utf-8",
    )

    merged = ca.load_ap3_role_restart_knob_manifest(manifest)

    assert set(_EXPECTED_AP3_ROLE_RESTART_KNOBS).issubset(merged)
    assert merged["worker_draft_max"]["hi"] == 12
    assert merged["worker_draft_max"]["default"] == 3
    assert merged["worker_draft_max"]["role"] == "worker_general"


# ══════════════════════════════════════════════════════════════════════════════
# AP-4 — optional Pareto/quality axes
# ══════════════════════════════════════════════════════════════════════════════


def _base_eval() -> "EvalResult":
    return EvalResult(tier=1, quality=2.0, speed=10.0, cost=0.5, reliability=0.9)


def test_ap4_absent_axes_leave_objectives_and_grep_identical() -> None:
    r = _base_eval()
    assert r.objectives == (2.0, 10.0, -0.5, 0.9)
    grep = r.to_grep_lines(trial_id=1)
    # D4 METRIC contract v2 (audit MET-1): unavailable axes are emitted as
    # explicit null, never omitted — absence-vs-zero must be unambiguous.
    for axis in rpt.REVIEW_PARETO_AXES:
        assert f"METRIC {axis}: null" in grep
    assert rpt.reviewer_quality_axes(r) == {}


def test_ap4_present_axes_surface_in_extractor_and_grep() -> None:
    r = EvalResult(
        tier=1,
        quality=2.0,
        speed=10.0,
        cost=0.5,
        reliability=0.9,
        reviewer_fa_rate=0.1,
        reviewer_fr_rate=0.2,
        review_decision_latency_ms=250.0,
    )
    axes = rpt.reviewer_quality_axes(r)
    assert axes == {
        "reviewer_fa_rate": 0.1,
        "reviewer_fr_rate": 0.2,
        "review_decision_latency_ms": 250.0,
    }
    # objectives 4-tuple still unchanged (axes are NOT folded into it)
    assert r.objectives == (2.0, 10.0, -0.5, 0.9)
    grep = r.to_grep_lines(trial_id=2)
    assert "METRIC reviewer_fa_rate: 0.1000" in grep
    # D4 contract v2: never-set -> explicit null, not absent (audit MET-1)
    assert "METRIC reviewer_fa_fr_ratio: null" in grep


def test_ap4_calibration_from_decisions() -> None:
    cal = rpt.reviewer_calibration_from_decisions(
        [
            {"decision": "approve", "gate": "fail", "latency_ms": 100},  # FA
            {"decision": "reject", "gate": "pass", "latency_ms": 200},  # FR
            {"decision": "approve", "gate": "pass"},  # correct
            {"decision": "reject", "gate": None},  # inconclusive
        ]
    )
    assert cal["reviewer_fa_rate"] == 1.0  # 1 FA / 1 gate-fail
    assert cal["reviewer_fr_rate"] == 0.5  # 1 FR / 2 gate-pass
    assert cal["reviewer_fa_fr_ratio"] == 2.0
    assert cal["review_decision_latency_ms"] == 150.0


def test_ap4_calibration_rc11_always_keys_every_side() -> None:
    # RC-11: an unmeasured side is None, never absent, never 0.0 — and the
    # denominator counts ride along so undercounted rates are visible.
    one_sided = rpt.reviewer_calibration_from_decisions(
        [
            {"decision": "approve", "gate": "fail", "latency_ms": 100},  # FA
            {"decision": "approve", "gate": "fail", "latency_ms": 200},  # FA
        ]
    )
    assert one_sided["reviewer_fa_rate"] == 1.0
    assert one_sided["reviewer_fr_rate"] is None
    assert one_sided["reviewer_fa_fr_ratio"] is None
    assert one_sided["n_gate_fail"] == 2
    assert one_sided["n_gate_pass"] == 0
    assert one_sided["excluded_no_conclusive_gate"] == 0

    no_latency = rpt.reviewer_calibration_from_decisions([{"decision": "approve", "gate": "pass"}])
    assert no_latency["review_decision_latency_ms"] is None
    assert no_latency["n_gate_pass"] == 1

    with_causes = rpt.reviewer_calibration_from_decisions(
        [{"decision": "approve", "gate": None}],
        excluded_causes={"observation_gold_confidence": 7},
    )
    assert with_causes["excluded_no_conclusive_gate"] == 1
    assert with_causes["excluded_causes"] == {"observation_gold_confidence": 7}
    # caller's dict is never mutated
    caller = {"observation_gold_confidence": 7}
    rpt.reviewer_calibration_from_decisions([], excluded_causes=caller)
    assert caller == {"observation_gold_confidence": 7}


def test_ap4_instrument_era_row_is_observation_only() -> None:
    row = rpt.instrument_era_row()
    assert row["protocol_id"] == "P-REV-1"
    assert list(rpt.REVIEW_PARETO_AXES) == row["axes"]
    assert "observation" in row["status"]


# ══════════════════════════════════════════════════════════════════════════════
# AP-5 — new actions (plan-generation, inference-gated)
# ══════════════════════════════════════════════════════════════════════════════


def _ctx(state: dict | None = None):
    return actions._ActionContext(
        seeder=None,
        swarm=None,
        forge=None,
        lab=None,
        tower=None,
        gate=None,
        archive=None,
        journal=None,
        state=state if state is not None else {},
    )


def test_ap5_actions_registered_in_dispatch_table() -> None:
    assert actions._ACTION_HANDLERS["review_policy_trial"] is actions._action_review_policy_trial
    assert (
        actions._ACTION_HANDLERS["screening_tier_driver"] is actions._action_screening_tier_driver
    )


def test_ap5_review_policy_trial_dry_run_enumerates_plan() -> None:
    plan, err = rpt.plan_review_policy_trial(
        {"knobs": ["review_majority_k"], "grid_points": 3},
        corpus_manifest={"corpus_id": "nearmiss-v1", "total_rows": 100, "counts": {}},
    )
    assert err is None
    d = plan.to_dict()
    assert d["kind"] == "review_policy_trial_plan"
    assert d["surface"] == "review_plane"
    assert d["n_trials"] == len(d["grid"]) >= 2
    assert d["inference_required"] is True
    assert all("delegation.review_majority_k" in combo for combo in d["grid"])


def test_ap5_review_policy_trial_rejects_unknown_knob() -> None:
    plan, err = rpt.plan_review_policy_trial({"knobs": ["not_a_knob"]})
    assert plan is None and "unknown knobs" in err


def test_ap5_review_policy_trial_rejects_out_of_bounds_override() -> None:
    plan, err = rpt.plan_review_policy_trial(
        {"knobs": ["max_review_iterations"], "params": {"max_review_iterations": 99}}
    )
    assert plan is None and "max 5" in err


def test_ap5_review_policy_trial_handler_dry_run_stashes_plan() -> None:
    ctx = _ctx()
    res, species = actions._action_review_policy_trial(
        {"knobs": ["review_majority_k"], "grid_points": 2}, ctx
    )
    assert species == "review_plane"
    assert isinstance(res, actions.SkipOutcome)
    assert res.status == "skipped"
    assert "_review_policy_trial_plan" in ctx.state
    assert ctx.state["_review_policy_trial_plan"]["n_trials"] >= 2


def test_ap5_review_policy_trial_handler_live_gated_no_flag() -> None:
    ctx = _ctx()
    res, _ = actions._action_review_policy_trial(
        {"knobs": ["review_majority_k"], "dry_run": False}, ctx
    )
    assert res.status == "skipped"
    assert "inference-gated" in res.reason


def test_ap5_review_policy_trial_handler_live_with_flag_raises(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_REVIEW_POLICY_TRIAL_INFERENCE", "1")
    ctx = _ctx()
    with pytest.raises(NotImplementedError):
        actions._action_review_policy_trial({"knobs": ["review_majority_k"], "dry_run": False}, ctx)


def test_ap5_review_policy_trial_handler_invalid_knob() -> None:
    ctx = _ctx()
    res, _ = actions._action_review_policy_trial({"knobs": ["bogus"]}, ctx)
    assert res.status == "invalid"


def test_ap5_screening_tier_plan_from_synthetic_pool() -> None:
    pool = {
        "pairings": [
            {
                "pairing_id": "arch__rev__grad",
                "architect": "arch",
                "reviewer": "rev",
                "grader": "grad",
                "cross_family_preferred": True,
                "self_review": False,
                "anchor_arm": None,
            }
        ],
        "provenance": {"schema_version": "1", "registry_sha256": "abc"},
    }
    plan, err = rpt.plan_screening_tier(
        pool,
        corpus_manifest={"corpus_id": "nearmiss-v1", "total_rows": 100, "counts": {}},
        per_pairing_n=8,
    )
    assert err is None
    d = plan.to_dict()
    assert d["n_queued"] == 1
    assert d["queue"][0]["dispatch"] == "placement_queue"  # NOT /chat
    assert d["queue"][0]["n"] == 8
    assert d["inference_required"] is True


def test_ap5_screening_tier_rejects_empty_corpus() -> None:
    pool = {"pairings": [{"pairing_id": "a__b__c"}]}
    plan, err = rpt.plan_screening_tier(
        pool, corpus_manifest={"corpus_id": "x", "total_rows": 0, "counts": {}}
    )
    assert plan is None and "empty" in err


def test_ap5_screening_handler_dry_run_stashes_plan(tmp_path: Path) -> None:
    pool = {
        "pairings": [{"pairing_id": "a__b__c", "architect": "a", "reviewer": "b", "grader": "c"}],
        "provenance": {"schema_version": "1"},
    }
    pool_file = tmp_path / "pool.json"
    pool_file.write_text(json.dumps(pool), encoding="utf-8")
    corpus = tmp_path / "manifest.json"
    corpus.write_text(
        json.dumps({"corpus_id": "c1", "total_rows": 50, "counts": {}}), encoding="utf-8"
    )

    ctx = _ctx()
    res, species = actions._action_screening_tier_driver(
        {"pool_gen_path": str(pool_file), "corpus_manifest_path": str(corpus)}, ctx
    )
    assert species == "review_plane"
    assert res.status == "skipped"
    assert ctx.state["_screening_tier_plan"]["n_queued"] == 1


def test_ap5_screening_handler_requires_pool_path() -> None:
    ctx = _ctx()
    res, _ = actions._action_screening_tier_driver({}, ctx)
    assert res.status == "invalid" and "pool_gen_path" in res.reason


def test_ap5_screening_handler_live_with_flag_calls_runner(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_SCREENING_TIER_INFERENCE", "1")
    pool = {
        "pairings": [{"pairing_id": "a__b__c", "architect": "a", "reviewer": "b", "grader": "c"}],
        "provenance": {"schema_version": "1"},
    }
    pool_file = tmp_path / "pool.json"
    pool_file.write_text(json.dumps(pool), encoding="utf-8")
    corpus = tmp_path / "manifest.json"
    corpus.write_text(
        json.dumps({"corpus_id": "c1", "total_rows": 50, "counts": {}}), encoding="utf-8"
    )
    row_ids = tmp_path / "rows.txt"
    row_ids.write_text("row-a\nrow-b\n", encoding="utf-8")
    output = tmp_path / "results.jsonl"

    captured: dict = {}

    def _fake_run_screening_tier(plan, pool_gen_output, **kwargs):
        captured["plan"] = plan
        captured["pool_gen_output"] = pool_gen_output
        captured["kwargs"] = kwargs
        return {"mode": "execute", "inference_ran": True, "n_jobs": 1, "results": []}

    monkeypatch.setitem(
        sys.modules,
        "screening_tier_runner",
        type(
            "FakeScreeningTierRunner",
            (),
            {"run_screening_tier": staticmethod(_fake_run_screening_tier)},
        ),
    )

    ctx = _ctx()
    res, species = actions._action_screening_tier_driver(
        {
            "pool_gen_path": str(pool_file),
            "corpus_manifest_path": str(corpus),
            "row_ids_path": str(row_ids),
            "results_path": str(output),
            "max_pairings": 7,
            "cap_per_pairing": 3,
            "seed": 99,
            "dry_run": False,
        },
        ctx,
    )

    assert species == "review_plane"
    assert res.status == "skipped"
    assert "execute complete" in res.reason
    assert ctx.state["_screening_tier_result"]["inference_ran"] is True
    assert captured["kwargs"]["output_path"] == output
    assert captured["kwargs"]["row_ids_path"] == row_ids
    assert captured["kwargs"]["max_pairings"] == 7
    assert captured["kwargs"]["cap_per_pairing"] == 3
    assert captured["kwargs"]["seed"] == 99


# ══════════════════════════════════════════════════════════════════════════════
# AP-6 — codex-critic dogfooding
# ══════════════════════════════════════════════════════════════════════════════


def test_ap6_planner_result_has_optional_review_decision_field() -> None:
    r = planner_providers.PlannerProviderResult(provider="codex", role="critique")
    assert r.review_decision is None


@pytest.mark.skipif(not _review_grammar_available(), reason="review_grammar unavailable")
@pytest.mark.parametrize(
    "native,expected",
    [("approve", "approve"), ("revise", "request_changes"), ("reject", "reject")],
)
def test_ap6_critique_decision_maps_to_review_decision(native, expected) -> None:
    text = json.dumps({"decision": native, "confidence": 0.6, "issues": ["x"]})
    obj, failure = rpt.derive_review_decision_from_critique(text)
    assert failure is None
    assert obj["decision"] == expected
    assert obj["blocking"]["tripwire"] is False  # advisory reviewer never hard-stops


@pytest.mark.skipif(not _review_grammar_available(), reason="review_grammar unavailable")
def test_ap6_native_review_decision_passes_through() -> None:
    text = '{"decision":"approve","confidence":0.9,"blocking":{"tripwire":false}}'
    obj, failure = rpt.derive_review_decision_from_critique(text)
    assert failure is None and obj["decision"] == "approve"


def test_ap6_parse_failure_counted_on_garbage() -> None:
    obj, failure = rpt.derive_review_decision_from_critique("no json anywhere")
    assert obj is None and failure is not None


# ── TD-21.18: shared fish_json replaces the local balanced-brace scanner ──────
#
# No repair TURN is added at this site: the only caller
# (planner_providers._emit_codex_review_decision) feeds text from the `codex
# exec` EXTERNAL CLI, never a reachable local server, so per this repo's
# transport discipline (external CLI => deterministic fish only) there is no
# live model to spend a repair call against, and this module's own docstring
# ("nothing here spends inference") stays true. These tests instead prove the
# UPGRADED deterministic extraction: a malformed-but-fenced critique block the
# OLD scanner could never recover (it required the JSON to close within the
# text) IS now recovered, purely deterministically, and a call counter proves
# zero calls are ever spent either way.


@pytest.mark.skipif(not _fish_json_available(), reason="src.structured_output.repair unavailable")
def test_td21_18_happy_path_fenced_critique_zero_calls() -> None:
    """A well-formed, fenced critique block extracts cleanly -- 0 calls, by
    construction (this module has no completer to call at all)."""
    text = (
        "```json:autopilot_critique\n"
        '{"decision": "approve", "confidence": 0.8, "issues": []}\n'
        "```"
    )
    obj = rpt._extract_critique_block(text)
    assert obj == {"decision": "approve", "confidence": 0.8, "issues": []}
    review_decision, failure = rpt.derive_review_decision_from_critique(text)
    assert failure is None
    assert review_decision["decision"] == "approve"


@pytest.mark.skipif(not _fish_json_available(), reason="src.structured_output.repair unavailable")
def test_td21_18_malformed_truncated_fence_is_recovered_deterministically() -> None:
    """A critique block truncated INSIDE its own fence (missing the closing
    ``]`` and ``}`` for `issues`) is unrecoverable by the OLD scanner -- brace
    depth never returns to 0, so it finds no candidate at all -- but IS
    recovered by the shared fish_json's fenced-block + bracket-repair path
    (src/prompt_builders/code_utils._repair_json_text closes the dangling
    array/object). Never a live call: this is pure deterministic recovery."""
    text = (
        "```json:autopilot_critique\n"
        "```json\n"
        '{"decision": "revise", "confidence": 0.7, "issues": ["needs tests"\n'
        "```"
    )
    # Pin the OLD behavior as the regression baseline: the pre-TD-21.18 local
    # scanner (still shipped in review_grammar.py, unchanged) genuinely cannot
    # recover this -- depth never returns to 0.
    from src.proactive_delegation.review_grammar import _extract_json_object as _old_scanner

    marker = "```json:autopilot_critique"
    body = text[text.find(marker) + len(marker):]
    assert _old_scanner(body) is None  # the bug TD-21.18 closes

    obj = rpt._extract_critique_block(text)
    assert obj == {"decision": "revise", "confidence": 0.7, "issues": ["needs tests"]}

    review_decision, failure = rpt.derive_review_decision_from_critique(text)
    assert failure is None
    assert review_decision["decision"] == "request_changes"  # native "revise" mapped


def test_td21_18_unrepairable_garbage_is_typed_failure_counted() -> None:
    """No JSON anywhere (inside or outside a fence): a typed ParseFailure,
    never a silent default, and CritiqueEmissionStats counts it (AP-6 already
    wires this; TD-21.18 only upgrades the extractor feeding it)."""
    text = "```json:autopilot_critique\nthe critic just wrote prose, no braces at all\n```"
    assert rpt._extract_critique_block(text) is None

    stats = rpt.CritiqueEmissionStats()
    planner_providers.CODEX_REVIEW_DECISION_STATS = stats
    result = planner_providers.PlannerProviderResult(
        provider="codex", role="critique", ok=True, text=text,
    )
    planner_providers._emit_codex_review_decision(result)
    assert result.review_decision is None
    assert stats.parse_failures == 1
    assert stats.emitted == 0


@pytest.mark.skipif(not _review_grammar_available(), reason="review_grammar unavailable")
def test_ap6_emit_attaches_and_counts_success() -> None:
    stats = rpt.CritiqueEmissionStats()
    planner_providers.CODEX_REVIEW_DECISION_STATS = stats
    result = planner_providers.PlannerProviderResult(
        provider="codex_critic",
        role="critique",
        ok=True,
        text=json.dumps({"decision": "approve", "confidence": 0.8, "issues": []}),
    )
    planner_providers._emit_codex_review_decision(result)
    assert result.review_decision is not None
    assert stats.emitted == 1 and stats.parse_failures == 0


def test_ap6_emit_counts_parse_failure_and_preserves_behavior() -> None:
    stats = rpt.CritiqueEmissionStats()
    planner_providers.CODEX_REVIEW_DECISION_STATS = stats
    result = planner_providers.PlannerProviderResult(
        provider="codex",
        role="critique",
        ok=True,
        text="totally not json",
    )
    planner_providers._emit_codex_review_decision(result)
    assert result.review_decision is None  # fell back, unchanged
    assert stats.parse_failures == 1


def test_ap6_emit_disabled_by_flag(monkeypatch) -> None:
    monkeypatch.setenv("AUTOPILOT_CODEX_REVIEW_DECISION_EMIT", "0")
    stats = rpt.CritiqueEmissionStats()
    planner_providers.CODEX_REVIEW_DECISION_STATS = stats
    result = planner_providers.PlannerProviderResult(
        provider="codex",
        role="critique",
        ok=True,
        text=json.dumps({"decision": "approve", "confidence": 0.8, "issues": []}),
    )
    planner_providers._emit_codex_review_decision(result)
    assert result.review_decision is None
    assert stats.emitted == 0 and stats.parse_failures == 0


# ══════════════════════════════════════════════════════════════════════════════
# AP-7 — journal schema + checkpoint compatibility
# ══════════════════════════════════════════════════════════════════════════════


def test_ap7_event_type_constants() -> None:
    assert rpt.REVIEW_DECISION_EVENT_TYPE == "review_decision"
    assert rpt.REVIEW_POLICY_TRIAL_EVENT_TYPE == "review_policy_trial"
    assert set(rpt.REVIEW_JOURNAL_EVENT_TYPES) == {"review_decision", "review_policy_trial"}


def test_ap7_review_decision_event_to_event_shape() -> None:
    ev = rpt.ReviewDecisionEvent(
        decision="approve",
        confidence=0.8,
        tripwire=False,
        source="codex_critic",
        latency_ms=120.0,
    ).to_event()
    assert ev["type"] == "review_decision"
    assert ev["decision"] == "approve"
    assert ev["latency_ms"] == 120.0


def test_ap7_review_decision_event_omits_nan_latency() -> None:
    ev = rpt.ReviewDecisionEvent(
        decision="reject", confidence=0.5, tripwire=False, source="x"
    ).to_event()
    assert "latency_ms" not in ev


def test_ap7_checkpoint_compat_minimal_state_gains_defaults() -> None:
    state: dict = {}
    rpt.ensure_review_state_defaults(state)
    assert set(rpt.REVIEW_STATE_DEFAULTS).issubset(state)
    assert state["review_policy_trial_count"] == 0


def test_ap7_checkpoint_compat_does_not_overwrite_existing() -> None:
    state = {"review_policy_trial_count": 5, "trial_counter": 42}
    rpt.ensure_review_state_defaults(state)
    assert state["review_policy_trial_count"] == 5  # preserved
    assert state["trial_counter"] == 42  # untouched
    assert state["review_decision_shadow_count"] == 0  # default injected


# ══════════════════════════════════════════════════════════════════════════════
# AP-8 — digest reviewer-calibration section
# ══════════════════════════════════════════════════════════════════════════════


def test_ap8_no_ledger_renders_no_data_gracefully() -> None:
    class EmptyLedger:
        @staticmethod
        def calibration_summary():
            return {}

    section = digest._reviewer_calibration_section(
        datetime.now(timezone.utc),
        ledger_module=EmptyLedger,
        emission_stats=rpt.CritiqueEmissionStats(),  # fresh/empty => no dogfood line
    )
    assert section[0].startswith("### Reviewer calibration")
    assert any("no reviewer-calibration data yet" in line for line in section)


def test_ap8_renders_ledger_summary() -> None:
    class FakeLedger:
        @staticmethod
        def calibration_summary():
            return {"n_decisions": 12, "reviewer_fa_rate": 0.05, "reviewer_fr_rate": 0.11}

    section = digest._reviewer_calibration_section(
        datetime.now(timezone.utc),
        ledger_module=FakeLedger,
        emission_stats=rpt.CritiqueEmissionStats(),
    )
    text = "\n".join(section)
    assert "n decisions" in text and "reviewer fa rate" in text
    assert "no reviewer-calibration data yet" not in text


def test_ap8_renders_dogfood_emission_stats() -> None:
    stats = rpt.CritiqueEmissionStats()
    stats.record_success()
    stats.record_failure("no_json")
    section = digest._reviewer_calibration_section(
        datetime.now(timezone.utc), ledger_module=None, emission_stats=stats
    )
    text = "\n".join(section)
    assert "codex dogfood emission: emitted=1, parse_failures=1" in text


# ══════════════════════════════════════════════════════════════════════════════
# B2 — digest pre-materialize hook (events -> review_ledger refresh)
# ══════════════════════════════════════════════════════════════════════════════
def _seed_review_event(db_path, subtask_id, decision, gold, ts):
    """Emit ONE REVIEW_DECISION event via the REAL emit path (no inference)."""
    from src.trace.store import Event, EventCategory, EventSource, detail_to_json
    from src.trace.emit import emit

    detail = {
        "mode": "review",
        "subtask_id": subtask_id,
        "decision": decision,
        "confidence": 0.8,
        "tripwire": False,
        "latency_ms": 100.0,
        "tokens": {"tokens_out": 20, "chars_out": 80},
    }
    emit(
        Event(
            ts_utc=ts,
            source=EventSource.REVIEW_PLANE,
            source_path="",
            source_line=None,
            session_id="sess-b2",
            trial_id=1,
            role="architect_general",
            category=EventCategory.REVIEW_DECISION,
            status=decision,
            summary=f"review {subtask_id}",
            detail_json=detail_to_json(detail),
        ),
        db_path=db_path,
    )


def test_b2_refresh_swallows_raising_materializer(tmp_path):
    from src.trace.store import ensure_schema

    events_db = tmp_path / "events.sqlite"
    ensure_schema(events_db).close()  # exists so we reach the (raising) materializer

    class RaisingMaterializer:
        @staticmethod
        def read_review_events(*a, **k):
            raise RuntimeError("materializer exploded")

    # Best-effort: the raising materializer is swallowed, digest hook returns False.
    assert (
        digest._refresh_review_ledger(
            events_db=events_db,
            ledger_path=tmp_path / "review_ledger.sqlite",
            materializer=RaisingMaterializer,
        )
        is False
    )


def test_b2_refresh_missing_events_db_is_noop(tmp_path):
    assert (
        digest._refresh_review_ledger(
            events_db=tmp_path / "nope.sqlite", ledger_path=tmp_path / "review_ledger.sqlite"
        )
        is False
    )


def test_b2_refresh_materializes_events(tmp_path):
    events_db = tmp_path / "events.sqlite"
    _seed_review_event(events_db, "cand-A", "approve", "fail", "2026-07-16T10:00:00+00:00")
    ledger_path = tmp_path / "review_ledger.sqlite"
    assert digest._refresh_review_ledger(events_db=events_db, ledger_path=ledger_path) is True
    # write_ledger_sqlite writes <ledger dir>/review_ledger.sqlite
    from src.trace.review_ledger import calibration_summary

    s = calibration_summary(db_path=ledger_path)
    assert s["n_decisions"] == 1


def test_b2_render_digest_survives_refresh_failure(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("refresh hook exploded")

    monkeypatch.setattr(digest, "_refresh_review_ledger", _boom)
    out = digest.render_digest(
        swarm=object(), lab=object(), archive=object(), state={}, journal=None
    )
    assert isinstance(out, str)
    assert "Reviewer calibration" in out
