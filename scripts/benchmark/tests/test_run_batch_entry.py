#!/usr/bin/env python3
"""Self-contained tests for the clean-window / benchmark execution bridge (B2).

Runs under pytest if available, and — because the research repo ships no pytest —
also stands alone via the stdlib runner in ``__main__``:

    /mnt/raid0/llm/epyc-inference-research/.venv/bin/python \
        scripts/benchmark/tests/test_run_batch_entry.py

Every test is inference-free. They exercise resolution, canonical dry-run command
construction, the topology-hash gate, the B4-attestation path, and the blocked /
skipped paths. ``--execute`` is never set — a dedicated test proves the gated
execute code is never entered on the default path.
"""
from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _BENCHMARK_DIR.parent
_RESEARCH_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_RESEARCH_ROOT), str(_SCRIPTS_DIR), str(_BENCHMARK_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import clean_window_manifest as cwm  # noqa: E402
import run_batch_entry as rbe  # noqa: E402


class _Skip(Exception):
    """Raised by a test to signal 'skipped' to the stdlib runner."""


# ---------------------------------------------------------------------------
# Helpers / fixtures (stdlib only, so they work under both runners)
# ---------------------------------------------------------------------------

PY = "/mnt/raid0/llm/epyc-inference-research/.venv/bin/python"
RB = str(_BENCHMARK_DIR / "run_benchmark.py")

SERVER_CMD = (
    f"{PY} {RB} --model architect_general --suite omniscience "
    f"--new-run --server-mode --skip-speed-tests"
)
ROPE_CMD = (
    f"{PY} {_BENCHMARK_DIR / 'rope_position_probe.py'} --api chat --host 127.0.0.1 "
    f"--port 8081 --context-length 4096 --n-samples 100 --seed 42 "
    f"--out /tmp/clean_window/rope_probe/frontdoor/ctx_4096.json"
)


def synthetic_manifest(topo_hash: str = "HASH_A", artifact: str = "/nonexistent/registry.yaml") -> dict:
    return {
        "topology": {
            "required_topology_hash": topo_hash,
            "topology_artifact": artifact,
            "live_registry_artifact": "/nonexistent/live.yaml",
        },
        "entries": [
            {
                "package": "G10",
                "kind": "run_benchmark_suite",
                "role": "architect_general",
                "suite": "omniscience",
                "status": "ready",
                "command": SERVER_CMD,
                "notes": [],
                "model": {"model_path": "/tmp/fake-model.gguf"},
            },
            {
                "package": "K-ROPE-1",
                "kind": "rope_position_probe",
                "role": "frontdoor",
                "context_length": 4096,
                "status": "ready",
                "command": ROPE_CMD,
                "notes": ["server port from registry.server_mode"],
                "model": {"model_path": "/tmp/frontdoor.gguf"},
            },
            # duplicate package/kind/role to trigger the ambiguity guard
            {
                "package": "G11",
                "kind": "run_benchmark_suite",
                "role": "worker_general",
                "suite": "omniscience",
                "status": "ready",
                "command": SERVER_CMD.replace("architect_general", "worker_general"),
                "notes": [],
                "model": {"model_path": "/tmp/worker.gguf"},
            },
            {
                "package": "G11",
                "kind": "run_benchmark_suite",
                "role": "worker_general",
                "suite": "tulving_episodic",
                "status": "ready",
                "command": SERVER_CMD.replace("architect_general", "worker_general").replace(
                    "omniscience", "tulving_episodic"
                ),
                "notes": [],
                "model": {"model_path": "/tmp/worker.gguf"},
            },
        ],
    }


def ok_runner(argv, *, timeout_s):  # noqa: ARG001
    return 0, "dry-run OK (no inference)", ""


def fail_runner(argv, *, timeout_s):  # noqa: ARG001
    return 1, "", "CanonicalRecipeViolation: host drift\nfix: reboot"


def ok_stack_gate(resolved, *, runner):  # noqa: ARG001
    return rbe.StackContractGateResult(required=False, ok=True, warnings=[], reasons=[])


def ok_matrix_gate(resolved, *, runner):  # noqa: ARG001
    return rbe.ContentionMatrixGateResult(
        required=False,
        ok=True,
        command=None,
        exit_code=None,
        reasons=[],
    )


def stale_matrix_gate(resolved, *, runner):  # noqa: ARG001
    return rbe.ContentionMatrixGateResult(
        required=True,
        ok=False,
        command="check_contention_matrix_fresh.py",
        exit_code=2,
        reasons=["contention matrix freshness gate failed (exit 2): stale"],
    )


def drift_stack_gate(resolved, *, runner):  # noqa: ARG001
    return rbe.StackContractGateResult(
        required=True,
        ok=False,
        warnings=["frontdoor pid 123 runtime spec.type expected draft-mtp; live cmdline has none"],
        reasons=["live stack launch contract has 1 warning(s); refusing to measure a drifted/non-optimized stack"],
    )


@contextlib.contextmanager
def swap_attr(obj, name, value):
    sentinel = object()
    old = getattr(obj, name, sentinel)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if old is sentinel:
            delattr(obj, name)
        else:
            setattr(obj, name, old)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_find_entry_unique():
    m = synthetic_manifest()
    e = rbe.find_clean_window_entry(
        m, {"package": "G10", "kind": "run_benchmark_suite", "role": "architect_general"}
    )
    assert e["role"] == "architect_general"
    assert rbe.derive_entry_id(e) == "G10:run_benchmark_suite:architect_general:omniscience"


def test_find_entry_missing_raises():
    m = synthetic_manifest()
    try:
        rbe.find_clean_window_entry(m, {"package": "NOPE"})
    except rbe.BatchEntryError as exc:
        assert "no clean-window entry" in str(exc)
    else:
        raise AssertionError("expected BatchEntryError for missing selector")


def test_find_entry_ambiguous_raises():
    m = synthetic_manifest()
    try:
        rbe.find_clean_window_entry(m, {"package": "G11", "kind": "run_benchmark_suite"})
    except rbe.BatchEntryError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("expected BatchEntryError for ambiguous selector")


def test_classify_server_suite():
    assert (
        rbe.classify_exec_path(
            driver=rbe.DRIVER_CLEAN_WINDOW,
            kind="run_benchmark_suite",
            command=SERVER_CMD,
            model_path="/tmp/x.gguf",
        )
        == rbe.PATH_SERVER_SUITE
    )


def test_classify_resolved_command():
    assert (
        rbe.classify_exec_path(
            driver=rbe.DRIVER_CLEAN_WINDOW,
            kind="rope_position_probe",
            command=ROPE_CMD,
            model_path="/tmp/x.gguf",
        )
        == rbe.PATH_RESOLVED_COMMAND
    )


def test_classify_llama_bench_inferred_from_bare_model():
    # command driver, no command, only a model path -> raw llama-bench request
    assert (
        rbe.classify_exec_path(
            driver=rbe.DRIVER_COMMAND, kind=None, command=None, model_path="/tmp/m.gguf"
        )
        == rbe.PATH_LLAMA_BENCH
    )


# ---------------------------------------------------------------------------
# Canonical dry-run command construction
# ---------------------------------------------------------------------------


def test_dry_run_llama_bench_appends_dry_run_and_no_execute():
    entry = {"driver": rbe.DRIVER_COMMAND, "model_path": "/tmp/m.gguf", "exec_path": "llama_bench"}
    resolved = rbe.resolve_entry(entry)
    dry = rbe.build_dry_run_command(resolved)
    assert dry is not None
    assert dry[0] == "bash"
    assert str(rbe.BENCH_CANONICAL_SH) in dry
    assert "-m" in dry and "/tmp/m.gguf" in dry
    assert dry[-1] == "--dry-run"
    # the EXECUTE form the bridge would run carries NO --execute / no --dry-run
    assert "--execute" not in resolved.command_resolved
    assert "--dry-run" not in resolved.command_resolved
    # extra flags land behind the `--` passthrough, dry-run before it
    entry2 = {
        "driver": rbe.DRIVER_COMMAND,
        "model_path": "/tmp/m.gguf",
        "exec_path": "llama_bench",
        "bench": {"n_gen": 128, "reps": 3, "extra_flags": ["-ctk", "q8_0"]},
    }
    dry2 = rbe.build_dry_run_command(rbe.resolve_entry(entry2))
    assert dry2[dry2.index("-n") + 1] == "128"
    assert dry2[dry2.index("-r") + 1] == "3"
    assert dry2.index("--dry-run") < dry2.index("--")
    assert dry2[-2:] == ["-ctk", "q8_0"]


def test_dry_run_server_suite_appends_dry_run():
    m = synthetic_manifest()
    entry = {
        "driver": rbe.DRIVER_CLEAN_WINDOW,
        "selector": {"package": "G10", "kind": "run_benchmark_suite", "role": "architect_general"},
    }
    resolved = rbe.resolve_entry(entry, manifest=m)
    assert resolved.exec_path == rbe.PATH_SERVER_SUITE
    dry = rbe.build_dry_run_command(resolved)
    assert dry[-1] == "--dry-run"
    assert "--server-mode" in dry and "--skip-speed-tests" in dry
    assert "run_benchmark.py" in " ".join(dry)


def test_server_suite_baseline_run_swaps_new_run():
    m = synthetic_manifest()
    entry = {
        "driver": rbe.DRIVER_CLEAN_WINDOW,
        "selector": {"package": "G10", "kind": "run_benchmark_suite", "role": "architect_general"},
        "baseline_run": "run-20260717",
    }
    resolved = rbe.resolve_entry(entry, manifest=m)
    assert "--new-run" not in resolved.command_resolved
    assert "--baseline-run run-20260717" in resolved.command_resolved
    dry = rbe.build_dry_run_command(resolved)
    assert "--new-run" not in dry
    assert dry[dry.index("--baseline-run") + 1] == "run-20260717"


def test_dry_run_resolved_command_is_none():
    m = synthetic_manifest()
    entry = {
        "driver": rbe.DRIVER_CLEAN_WINDOW,
        "selector": {"package": "K-ROPE-1", "kind": "rope_position_probe", "role": "frontdoor"},
    }
    resolved = rbe.resolve_entry(entry, manifest=m)
    assert resolved.exec_path == rbe.PATH_RESOLVED_COMMAND
    assert rbe.build_dry_run_command(resolved) is None


def test_predict_artifacts_parses_out_flag():
    m = synthetic_manifest()
    entry = {
        "driver": rbe.DRIVER_CLEAN_WINDOW,
        "selector": {"package": "K-ROPE-1", "kind": "rope_position_probe", "role": "frontdoor"},
    }
    resolved = rbe.resolve_entry(entry, manifest=m)
    arts = rbe.predict_artifacts(resolved)
    assert "/tmp/clean_window/rope_probe/frontdoor/ctx_4096.json" in arts


def test_command_driver_raw_argv():
    entry = {"driver": rbe.DRIVER_COMMAND, "command": ["bash", "some_probe.sh", "--flag"]}
    resolved = rbe.resolve_entry(entry)
    assert resolved.exec_path == rbe.PATH_RESOLVED_COMMAND
    assert resolved.command_resolved == "bash some_probe.sh --flag"
    assert resolved.command_argv == ["bash", "some_probe.sh", "--flag"]


def test_full_manifest_command_entry_resolution_and_live_stack_requirement():
    entry = {
        "task_id": "EV-4-calibration-baseline",
        "preconditions": {
            "models": ["worker_general", "frontdoor"],
            "topology": {"required_topology_hash": "HASH_A"},
        },
        "execution": {
            "command": ".venv/bin/python scripts/benchmark/eval_batch_serving_evaltower_window.py --api-url http://localhost:8000 --apply",
            "concurrency_mode": "same_trial_eval_fanout",
            "cwd": "/mnt/raid0/llm/epyc-orchestrator",
        },
    }
    resolved = rbe.resolve_entry(entry)
    assert resolved.entry_id == "EV-4-calibration-baseline"
    assert resolved.required_topology_hash == "HASH_A"
    assert resolved.cwd == "/mnt/raid0/llm/epyc-orchestrator"
    assert resolved.requires_live_stack_contract is True
    assert "eval_batch_serving_evaltower_window.py" in resolved.command_resolved


# ---------------------------------------------------------------------------
# Topology-hash gate + B4 attestation
# ---------------------------------------------------------------------------


def _resolved_for_topology(topo_hash="HASH_A", artifact="/nonexistent/registry.yaml"):
    m = synthetic_manifest(topo_hash=topo_hash, artifact=artifact)
    entry = {
        "driver": rbe.DRIVER_CLEAN_WINDOW,
        "selector": {"package": "G10", "kind": "run_benchmark_suite", "role": "architect_general"},
    }
    return rbe.resolve_entry(entry, manifest=m)


def test_topology_match_but_unverified_refuses():
    resolved = _resolved_for_topology()
    with tempfile.TemporaryDirectory() as d:
        gate = rbe.topology_gate(
            resolved, attestation_dir=Path(d) / "missing", live_hash_override="HASH_A"
        )
    assert gate.hash_match is True
    assert gate.verified is False  # no attestation -> execute refused
    assert any("attestation" in r for r in gate.reasons)


def test_topology_mismatch_blocks():
    resolved = _resolved_for_topology(topo_hash="HASH_A")
    with tempfile.TemporaryDirectory() as d:
        gate = rbe.topology_gate(resolved, attestation_dir=Path(d), live_hash_override="HASH_B")
    assert gate.hash_match is False
    assert any("mismatch" in r for r in gate.reasons)
    assert gate.verified is False


def test_topology_verified_with_attestation():
    resolved = _resolved_for_topology(topo_hash="HASH_A")
    with tempfile.TemporaryDirectory() as d:
        att = Path(d) / "attest-20260717.json"
        att.write_text(
            json.dumps({"topology_hash": "HASH_A", "live_affinity_verified": True, "status": "ok"})
        )
        gate = rbe.topology_gate(resolved, attestation_dir=Path(d), live_hash_override="HASH_A")
    assert gate.hash_match is True
    assert gate.verified is True
    assert gate.reasons == []
    assert gate.attestation_path is not None


def test_topology_reuses_clean_window_hashing():
    # Confirm the gate's live hash equals clean_window_manifest._file_sha256 on the
    # real research registry (reuse of the manifest's hashing, per the plan).
    reg = _RESEARCH_ROOT / "orchestration" / "model_registry.yaml"
    if not reg.exists():
        raise _Skip("research registry not present")
    expected = cwm._file_sha256(reg)
    m = synthetic_manifest(topo_hash=expected, artifact=str(reg))
    entry = {
        "driver": rbe.DRIVER_CLEAN_WINDOW,
        "selector": {"package": "G10", "kind": "run_benchmark_suite", "role": "architect_general"},
    }
    resolved = rbe.resolve_entry(entry, manifest=m)
    with tempfile.TemporaryDirectory() as d:
        gate = rbe.topology_gate(resolved, attestation_dir=Path(d))  # real hasher
    assert gate.live_hash == expected
    assert gate.hash_match is True


# ---------------------------------------------------------------------------
# Preflight + gated-execute discipline
# ---------------------------------------------------------------------------


def test_preflight_server_suite_dry_run_ok():
    resolved = _resolved_for_topology()
    with tempfile.TemporaryDirectory() as d:
        res = rbe.run_preflight(
            resolved, attestation_dir=Path(d), runner=ok_runner,
            stack_contract_checker=ok_stack_gate,
            contention_matrix_checker=ok_matrix_gate,
            live_hash_override="HASH_A",
        )
    assert res["phase"] == "preflight"
    assert res["dry_run_ok"] is True
    assert res["dry_run_mode"] == "canonical_subprocess"
    assert res["exit_code"] is None  # preflight never executes
    assert res["command_resolved"]
    # topology matched but unverified -> a blocking reason remains for execute
    assert any("attestation" in r for r in res["blocking_reasons"])
    for key in (
        "entry_id",
        "phase",
        "dry_run_ok",
        "blocking_reasons",
        "command_resolved",
        "artifacts",
        "wall_clock_s",
        "exit_code",
    ):
        assert key in res


def test_preflight_dry_run_failure_captured():
    resolved = _resolved_for_topology()
    with tempfile.TemporaryDirectory() as d:
        res = rbe.run_preflight(
            resolved, attestation_dir=Path(d), runner=fail_runner,
            stack_contract_checker=ok_stack_gate,
            contention_matrix_checker=ok_matrix_gate,
            live_hash_override="HASH_A",
        )
    assert res["dry_run_ok"] is False
    assert res["dry_run_exit_code"] == 1
    assert any("dry-run failed" in r for r in res["blocking_reasons"])


def test_preflight_resolved_command_is_resolution_only():
    m = synthetic_manifest()
    entry = {
        "driver": rbe.DRIVER_CLEAN_WINDOW,
        "selector": {"package": "K-ROPE-1", "kind": "rope_position_probe", "role": "frontdoor"},
    }
    resolved = rbe.resolve_entry(entry, manifest=m)

    def explode(*a, **k):  # must NOT be called for resolution-only entries
        raise AssertionError("runner should not be invoked for resolution_only path")

    with tempfile.TemporaryDirectory() as d:
        res = rbe.run_preflight(
            resolved, attestation_dir=Path(d), runner=explode,
            stack_contract_checker=ok_stack_gate,
            contention_matrix_checker=ok_matrix_gate,
            live_hash_override="HASH_A",
        )
    assert res["dry_run_mode"] == "resolution_only"
    assert res["dry_run_ok"] is True


def test_execute_default_off_never_enters_gated_path():
    """The core safety guarantee: with execute defaulting off, the gated execute
    code is never entered — proven two ways."""
    m = synthetic_manifest()
    entry = {
        "driver": rbe.DRIVER_CLEAN_WINDOW,
        "selector": {"package": "G10", "kind": "run_benchmark_suite", "role": "architect_general"},
    }
    # (1) module sentinel stays False
    rbe._EXECUTE_INVOKED = False
    with tempfile.TemporaryDirectory() as d:
        res = rbe.run_batch_entry(
            entry,
            manifest=m,
            attestation_dir=Path(d),
                runner=ok_runner,
                stack_contract_checker=ok_stack_gate,
                contention_matrix_checker=ok_matrix_gate,
                live_hash_override="HASH_A",
            )
    assert rbe._EXECUTE_INVOKED is False
    assert res["phase"] == "preflight"
    assert res["exit_code"] is None

    # (2) monkeypatched _execute_resolved is never called on the default path
    called = {"n": 0}

    def spy(*a, **k):
        called["n"] += 1
        raise AssertionError("gated execute must not run with execute default-off")

    with swap_attr(rbe, "_execute_resolved", spy):
        with tempfile.TemporaryDirectory() as d:
            rbe.run_batch_entry(
                entry, manifest=m, attestation_dir=Path(d), runner=ok_runner,
                stack_contract_checker=ok_stack_gate,
                contention_matrix_checker=ok_matrix_gate,
                live_hash_override="HASH_A",
            )
    assert called["n"] == 0


def test_execute_requested_but_unverified_still_refuses():
    """Even if execute=True, an unverified topology (no B4 attestation) refuses to
    enter the gated path and returns the preflight result."""
    m = synthetic_manifest()
    entry = {
        "driver": rbe.DRIVER_CLEAN_WINDOW,
        "selector": {"package": "G10", "kind": "run_benchmark_suite", "role": "architect_general"},
    }
    called = {"n": 0}

    def spy(*a, **k):
        called["n"] += 1

    with swap_attr(rbe, "_execute_resolved", spy):
        with tempfile.TemporaryDirectory() as d:  # empty attestation dir -> unverified
            res = rbe.run_batch_entry(
                entry, manifest=m, attestation_dir=Path(d), runner=ok_runner,
                execute=True,
                stack_contract_checker=ok_stack_gate,
                contention_matrix_checker=ok_matrix_gate,
                live_hash_override="HASH_A",
            )
    assert called["n"] == 0
    assert res["phase"] == "preflight"
    assert res["topology"]["verified"] is False


def test_continue_on_error_captures_resolution_failure():
    m = synthetic_manifest()
    entry = {"driver": rbe.DRIVER_CLEAN_WINDOW, "selector": {"package": "DOES-NOT-EXIST"}}
    res = rbe.run_batch_entry(entry, manifest=m, continue_on_error=True)
    assert res["phase"] == "skipped"
    assert res["exit_code"] is None
    assert any("resolution failed" in r for r in res["blocking_reasons"])


def test_resolution_failure_raises_without_continue_on_error():
    m = synthetic_manifest()
    entry = {"driver": rbe.DRIVER_CLEAN_WINDOW, "selector": {"package": "DOES-NOT-EXIST"}}
    try:
        rbe.run_batch_entry(entry, manifest=m, continue_on_error=False)
    except rbe.BatchEntryError:
        pass
    else:
        raise AssertionError("expected BatchEntryError to propagate")


def test_result_dict_is_jsonl_serialisable():
    resolved = _resolved_for_topology()
    with tempfile.TemporaryDirectory() as d:
        res = rbe.run_preflight(
            resolved, attestation_dir=Path(d), runner=ok_runner,
            stack_contract_checker=ok_stack_gate,
            contention_matrix_checker=ok_matrix_gate,
            live_hash_override="HASH_A",
        )
    line = json.dumps(res, sort_keys=True)
    assert json.loads(line)["schema_version"] == rbe.RESULT_SCHEMA_VERSION


def test_live_stack_contract_warning_blocks_execute_even_with_verified_topology():
    resolved = _resolved_for_topology(topo_hash="HASH_A")
    resolved.requires_live_stack_contract = True
    called = {"n": 0}

    def spy(*a, **k):
        called["n"] += 1
        raise AssertionError("execute must not run on launch-contract drift")

    with tempfile.TemporaryDirectory() as d:
        att = Path(d) / "attest-20260720.json"
        att.write_text(
            json.dumps({"topology_hash": "HASH_A", "live_affinity_verified": True, "status": "ok"})
        )
        with swap_attr(rbe, "_execute_resolved", spy):
            res = rbe.run_batch_entry(
                {
                    "driver": rbe.DRIVER_CLEAN_WINDOW,
                    "selector": {
                        "package": "G10",
                        "kind": "run_benchmark_suite",
                        "role": "architect_general",
                    },
                },
                manifest=synthetic_manifest(),
                attestation_dir=Path(d),
                runner=ok_runner,
                stack_contract_checker=drift_stack_gate,
                contention_matrix_checker=ok_matrix_gate,
                execute=True,
                live_hash_override="HASH_A",
            )
    assert called["n"] == 0
    assert res["phase"] == "preflight"
    assert res["stack_contract"]["ok"] is False
    assert any("launch contract" in r for r in res["blocking_reasons"])


def test_eval_fanout_preflight_blocks_on_stale_contention_matrix():
    entry = {
        "task_id": "EV-4-calibration-baseline",
        "preconditions": {
            "models": ["worker_general", "frontdoor"],
            "topology": {
                "required_topology_hash": "HASH_A",
                "contention_matrix": "v7-recert-required",
            },
        },
        "execution": {
            "command": ".venv/bin/python scripts/benchmark/eval_batch_serving_evaltower_window.py --api-url http://localhost:8000 --apply",
            "concurrency_mode": "same_trial_eval_fanout",
            "cwd": "/mnt/raid0/llm/epyc-orchestrator",
        },
    }
    resolved = rbe.resolve_entry(entry)
    with tempfile.TemporaryDirectory() as d:
        att = Path(d) / "attest-20260720.json"
        att.write_text(
            json.dumps({"topology_hash": "HASH_A", "live_affinity_verified": True, "status": "ok"})
        )
        res = rbe.run_preflight(
            resolved,
            attestation_dir=Path(d),
            runner=ok_runner,
            stack_contract_checker=ok_stack_gate,
            contention_matrix_checker=stale_matrix_gate,
            live_hash_override="HASH_A",
        )

    assert res["phase"] == "preflight"
    assert res["topology"]["verified"] is True
    assert res["contention_matrix"]["required"] is True
    assert res["contention_matrix"]["ok"] is False
    assert any("contention matrix freshness" in r for r in res["blocking_reasons"])


def test_eval_fanout_not_required_contention_matrix_is_blocked():
    entry = {
        "task_id": "EV-4-calibration-baseline",
        "preconditions": {
            "models": ["worker_general"],
            "topology": {
                "required_topology_hash": "HASH_A",
                "contention_matrix": "not_required",
            },
        },
        "execution": {
            "command": ".venv/bin/python scripts/benchmark/eval_batch_serving_evaltower_window.py --api-url http://localhost:8000 --apply",
            "concurrency_mode": "same_trial_eval_fanout",
        },
    }
    resolved = rbe.resolve_entry(entry)

    def runner(argv, *, timeout_s, cwd=None):  # noqa: ARG001
        return 0, "OK: contention matrix is fresh", ""

    gate = rbe.contention_matrix_gate(resolved, runner=runner)

    assert gate.required is True
    assert gate.ok is False
    assert any("not_required" in reason for reason in gate.reasons)


def test_never_regenerates_manifest_when_supplied():
    """If a manifest is supplied, the (heavy) generator/builder is never called."""
    m = synthetic_manifest()

    def boom_builder():
        raise AssertionError("builder must not run when manifest is supplied")

    entry = {
        "driver": rbe.DRIVER_CLEAN_WINDOW,
        "selector": {"package": "G10", "kind": "run_benchmark_suite", "role": "architect_general"},
    }
    resolved = rbe.resolve_entry(entry, manifest=m, builder=boom_builder)
    assert resolved.entry_id.startswith("G10:")


# ---------------------------------------------------------------------------
# End-to-end against the real clean-window generator (read-only, no inference)
# ---------------------------------------------------------------------------


def test_end_to_end_real_generator():
    try:
        manifest = cwm.build_manifest()
    except Exception as exc:  # noqa: BLE001
        raise _Skip(f"clean-window generator unavailable: {exc}")
    suite_entries = [e for e in manifest["entries"] if e["kind"] == "run_benchmark_suite"]
    if not suite_entries:
        raise _Skip("no run_benchmark_suite entries in live manifest")
    src = suite_entries[0]
    entry = {
        "driver": rbe.DRIVER_CLEAN_WINDOW,
        "selector": {"package": src["package"], "kind": src["kind"], "role": src["role"],
                     "suite": src.get("suite")},
    }
    resolved = rbe.resolve_entry(entry, manifest=manifest)
    assert resolved.exec_path == rbe.PATH_SERVER_SUITE
    dry = rbe.build_dry_run_command(resolved)
    assert dry is not None and dry[-1] == "--dry-run"
    # the manifest's required_topology_hash flows into the gate
    assert resolved.required_topology_hash == manifest["topology"]["required_topology_hash"]


# ---------------------------------------------------------------------------
# Stdlib runner (used when pytest is not installed)
# ---------------------------------------------------------------------------


def _run_all() -> int:
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    passed = failed = skipped = 0
    failures: list[str] = []
    for name, fn in tests:
        try:
            fn()
        except _Skip as exc:
            skipped += 1
            print(f"SKIP {name}: {exc}")
        except AssertionError as exc:
            failed += 1
            failures.append(f"{name}: {exc}")
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS {name}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
