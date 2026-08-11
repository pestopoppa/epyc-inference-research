#!/usr/bin/env python3
"""Fully OFFLINE tests for the E5 multi-server NUMA x np sweep harness.

Runs under pytest if available, and — because the research repo ships no pytest —
also stands alone via the stdlib runner in ``__main__``:

    /mnt/raid0/llm/epyc-inference-research/.venv/bin/python \
        scripts/benchmark/tests/test_server_numa_np_sweep.py

Every test is inference-free and process-free: subprocess.Popen/run are guarded
or mocked, the schema-owner interface (e5_cell_manifests) is stubbed, and all
driver/summarizer math runs on synthetic fixtures. No real port, process, or
server is ever touched. ``--execute`` paths are exercised only with every
launch/kill primitive monkeypatched.
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _BENCHMARK_DIR.parent
_RESEARCH_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_RESEARCH_ROOT), str(_SCRIPTS_DIR), str(_BENCHMARK_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import server_numa_np_sweep as sns  # noqa: E402
from canonical_recipe import CANONICAL_OMP_ENV, LLVM20_LIBDIR  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers / fixtures (stdlib only, so they work under both runners)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def patched(obj, name, value):
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


@contextlib.contextmanager
def patched_many(*triples):
    with contextlib.ExitStack() as stack:
        for obj, name, value in triples:
            stack.enter_context(patched(obj, name, value))
        yield


SCHEMA_STUB = SimpleNamespace(
    SCHEMA_VERSION="e5-cell-manifest/1",
    validate_cell_manifest=lambda manifest: [],
)

QIDS = ["q001", "q002", "q003", "q004", "q005", "q006"]


def make_manifest(**overrides) -> dict:
    manifest = {
        "schema_version": "e5-cell-manifest/1",
        "protocol_id": "P-BENCH-3",
        "era": {
            "cpu_kernel": "E6-cpu-kernel",
            "eval_instrument": "E7-eval-instrument",
            "source": "epyc-orchestrator/orchestration/instrument_eras.yaml",
        },
        "window": "W1",
        "cell_id": "testmodel_q8_0-C2-np2",
        "model_key": "testmodel_q8_0",
        "model_path": "/nonexistent/testmodel.gguf",
        "quant": "Q8_0",
        "architecture": "testarch",
        "config_id": "C2",
        "instances": [
            {"cpu_list": "24-47,120-143", "port": 19080, "threads": 48, "numactl_policy": "none"},
            {"cpu_list": "48-71,144-167", "port": 19081, "threads": 48, "numactl_policy": "none"},
        ],
        "np": 2,
        "per_stream_ctx": 2048,
        "ctx": 8192,
        "ubatch_size": 512,
        "prompt_caps": {
            "n_predict": 256,
            "max_prompt_chars": 4096,
            "max_total_in_flight": 43,
        },
        "prompt_batch": {
            "source": "/nonexistent/question_pool.jsonl",
            "selection": "pinned_qids",
            "qids": list(QIDS),
            "pinned_from": "data/batched_decode/e1-pbench3-clean-20260703T1912Z/selected_prompts.jsonl",
            "tier": 1,
            "seed": 42,
            "limit": len(QIDS),
        },
        "spec_dec": {
            "enabled": True,
            "spec_type": "draft-mtp",
            "draft_model_path": None,
            "draft_max": 4,
            "draft_min": None,
            "draft_p_min": None,
            "draft_p_split": 0,
            "threads_draft": None,
            "ngram_mod": None,
            "device_draft": "none",
            "record_accept_rate": True,
            "disabled_reason": None,
        },
        "kv": {"type_k": "q8_0", "type_v": "q8_0", "flash_attn": True, "kv_unified": False},
        "env_expectation": {
            "ggml_iqk": "1",
            "omp_source": "scripts/lib/canonical_recipe.CANONICAL_OMP_ENV",
            "kmp_blocktime": "10",
        },
        "mlock": True,
        "jinja": True,
        "warmup": {"prompts": 1, "n_predict": 32},
        "decision_grade_intent": True,
        "notes": "",
    }
    manifest.update(overrides)
    return manifest


def write_manifest(directory: Path, manifest: dict, name: str = "cell.json") -> Path:
    path = directory / name
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def write_pool(directory: Path, qids=None, long_qid: str | None = None) -> Path:
    path = directory / "question_pool.jsonl"
    rows = [{"__pool_metadata__": True, "generated_at": "2026-07-21T08:45:59Z"}]
    for index, qid in enumerate(qids or QIDS):
        prompt = "x" * 101_655 if qid == long_qid else f"prompt for {qid} number {index}"
        rows.append({"id": qid, "suite": "synthetic", "tier": 1, "prompt": prompt})
    rows.append({"id": "unpinned-extra", "suite": "synthetic", "tier": 1, "prompt": "extra"})
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def load_cell(directory: Path, manifest: dict | None = None) -> sns.Cell:
    return sns.load_cell_manifest(write_manifest(directory, manifest or make_manifest()))


class DummyProc:
    """Offline stand-in for subprocess.Popen — never a real process."""

    def __init__(self, pid: int = 4242, wait_raises: int = 0, returncode: int = 0):
        self.pid = pid
        self.returncode = None
        self._wait_raises = wait_raises
        self._final_rc = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self._wait_raises > 0:
            self._wait_raises -= 1
            raise subprocess.TimeoutExpired(cmd="dummy", timeout=timeout or 0)
        self.returncode = self._final_rc
        return self.returncode


def make_fake_send(records_out: list):
    def fake_send(*, cell, instance, prompt, stream_id, request_index, n_predict, timeout_s):
        record = sns.StreamRequestRecord(
            cell_id=cell.cell_id,
            qid=prompt.qid,
            suite=prompt.suite,
            request_index=request_index,
            stream_id=stream_id,
            instance_port=instance.port,
            success=True,
            start_s=float(max(request_index, 0)),
            first_token_s=float(max(request_index, 0)) + 0.1,
            end_s=float(max(request_index, 0)) + 1.0,
            ttft_ms=100.0,
            latency_ms=1000.0,
            predicted_tokens=64,
            prompt_tokens=100,
            predicted_tps=30.0,
            draft_n=100,
            draft_n_accepted=70,
            http_status=200,
            error="",
            response_text=f"resp-{prompt.qid}",
            timings={"predicted_n": 64, "draft_n": 100, "draft_n_accepted": 70},
        )
        records_out.append(record)
        return record

    return fake_send


def _popen_guard(*args, **kwargs):
    raise AssertionError(f"subprocess.Popen called in an offline test: {args!r}")


def _idle_throttle_guard(freqs=None):
    raise AssertionError(
        "under-load throttle gate must not run on the idle host (review F1)"
    )


CLEAN_ATTESTATION = {
    "created_at": "2026-07-23T00:00:00+00:00",
    "host": "testhost",
    "uptime_seconds": 1000.0,
    "numa_balancing": "0",
    "existing_llama_processes": [],
}


# ---------------------------------------------------------------------------
# Env composition
# ---------------------------------------------------------------------------


def test_no_private_default_env():
    # audit 2a: the harness must import canonical_recipe, never keep a copy.
    assert not hasattr(sns, "DEFAULT_ENV")


def test_build_cell_env_composition():
    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "bin" / "llama-server"
        binary.parent.mkdir(parents=True)
        env = sns.build_cell_env(binary)
    for key, value in CANONICAL_OMP_ENV.items():
        assert env[key] == value, f"canonical OMP var {key} missing/wrong"
    assert env["GGML_IQK"] == "1"  # v7 iqk runtime gate (audit C1)
    assert env["KMP_BLOCKTIME"] == "10"  # E1 idle-spin fix preserved
    # V4_GATE_EXTRA_ENV must never be applied: the harness may not ADD
    # GGML_NUMA_WEIGHTS beyond whatever the ambient shell already carries.
    assert env.get("GGML_NUMA_WEIGHTS") == os.environ.get("GGML_NUMA_WEIGHTS")
    ld_parts = env["LD_LIBRARY_PATH"].split(":")
    assert LLVM20_LIBDIR in ld_parts
    assert str(binary.parent) == ld_parts[0]
    sns.assert_canonical_env(env)  # drift tripwire passes on composed env


def test_env_expectation_mismatch_detected():
    with tempfile.TemporaryDirectory() as tmp:
        cell = load_cell(Path(tmp))
    bad_env = {"KMP_BLOCKTIME": "10"}  # no GGML_IQK
    errors = sns.check_env_expectation(bad_env, cell)
    assert errors and "GGML_IQK" in errors[0]
    good_env = {"GGML_IQK": "1", "KMP_BLOCKTIME": "10"}
    assert sns.check_env_expectation(good_env, cell) == []


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def test_command_construction_quarter_taskset_only():
    with tempfile.TemporaryDirectory() as tmp:
        cell = load_cell(Path(tmp))
    inst = cell.instances[0]
    cmd = sns.build_instance_command(binary=Path("/fake/llama-server"), cell=cell, inst=inst)
    # taskset pinning first, NO numactl for a quarter/half (first-touch locality).
    assert cmd[:3] == ["taskset", "-c", "24-47,120-143"]
    assert "numactl" not in cmd
    assert cmd[3] == "/fake/llama-server"
    bin_index = cmd.index("/fake/llama-server")
    for flag, value in (
        ("-m", cell.model_path),
        ("--port", "19080"),
        ("-np", "2"),
        ("-c", "8192"),
        ("-t", "48"),
        ("-ub", "512"),
        ("-ctk", "q8_0"),
        ("-ctv", "q8_0"),
        ("--flash-attn", "on"),
        ("--spec-type", "draft-mtp"),
        ("--spec-draft-n-max", "4"),
        ("--draft-p-split", "0"),
        ("--device", "none"),
        ("--device-draft", "none"),
        ("--log-colors", "off"),
    ):
        index = cmd.index(flag, bin_index)  # skip taskset's own -c
        assert cmd[index + 1] == value, f"{flag} != {value}: {cmd}"
    assert "--jinja" in cmd and "--mlock" in cmd
    assert "-kvu" not in cmd  # kv_unified false in the manifest
    assert "-md" not in cmd  # self-draft (NEXTN/MTP same-file)
    assert "--spec-draft-n-min" not in cmd and "--draft-p-min" not in cmd


def test_command_construction_full_interleave_ordering():
    manifest = make_manifest(
        cell_id="gemma-C1-np8",
        config_id="C1",
        instances=[
            {"cpu_list": "0-95", "port": 19090, "threads": 96, "numactl_policy": "interleave=all"},
        ],
        np=8,
        ctx=16384,
        spec_dec={
            "enabled": True,
            "spec_type": "draft-mtp",
            "draft_model_path": None,
            "draft_max": 2,
            "draft_min": None,
            "draft_p_min": 0.0,
            "draft_p_split": 0,
            "threads_draft": 16,
            "ngram_mod": None,
            "device_draft": "none",
            "record_accept_rate": True,
            "disabled_reason": None,
        },
    )
    with tempfile.TemporaryDirectory() as tmp:
        cell = load_cell(Path(tmp), manifest)
    cmd = sns.build_instance_command(
        binary=Path("/fake/llama-server"), cell=cell, inst=cell.instances[0]
    )
    # interleave ONLY where the instance says so, taskset BEFORE numactl.
    assert cmd[:5] == ["taskset", "-c", "0-95", "numactl", "--interleave=all"]
    assert cmd[5] == "/fake/llama-server"
    assert cmd[cmd.index("--spec-draft-n-max") + 1] == "2"
    assert cmd[cmd.index("--draft-p-min") + 1] == "0.0"
    assert cmd[cmd.index("--threads-draft") + 1] == "16"


def test_command_kvu_flag_per_manifest():
    manifest = make_manifest(
        kv={"type_k": "q8_0", "type_v": "q8_0", "flash_attn": True, "kv_unified": True}
    )
    with tempfile.TemporaryDirectory() as tmp:
        cell = load_cell(Path(tmp), manifest)
    cmd = sns.build_instance_command(
        binary=Path("/fake/llama-server"), cell=cell, inst=cell.instances[0]
    )
    assert "-kvu" in cmd


def test_command_spec_disabled_no_draft_flags():
    manifest = make_manifest(
        spec_dec={"enabled": False, "disabled_reason": "documented wedge probe"}
    )
    with tempfile.TemporaryDirectory() as tmp:
        cell = load_cell(Path(tmp), manifest)
    cmd = sns.build_instance_command(
        binary=Path("/fake/llama-server"), cell=cell, inst=cell.instances[0]
    )
    assert "--spec-type" not in cmd and "--device-draft" not in cmd
    assert cmd[cmd.index("--device") + 1] == "none"  # GPU guard stays


def test_command_md_only_when_draft_path_differs():
    base = make_manifest()
    base["spec_dec"]["draft_model_path"] = base["model_path"]
    with tempfile.TemporaryDirectory() as tmp:
        cell = load_cell(Path(tmp), base)
        cmd = sns.build_instance_command(
            binary=Path("/fake/llama-server"), cell=cell, inst=cell.instances[0]
        )
        assert "-md" not in cmd  # same file = self-draft, omit -md
        other = make_manifest()
        other["spec_dec"]["draft_model_path"] = "/nonexistent/other-draft.gguf"
        cell2 = load_cell(Path(tmp), other)
        cmd2 = sns.build_instance_command(
            binary=Path("/fake/llama-server"), cell=cell2, inst=cell2.instances[0]
        )
        assert cmd2[cmd2.index("-md") + 1] == "/nonexistent/other-draft.gguf"


# ---------------------------------------------------------------------------
# Refusal paths
# ---------------------------------------------------------------------------


def test_refuses_port_below_bench_range():
    manifest = make_manifest()
    manifest["instances"][0]["port"] = 8082  # a PROD port
    with tempfile.TemporaryDirectory() as tmp:
        cell = load_cell(Path(tmp), manifest)
    errors = sns.harness_refusals([cell])
    assert any("8082" in error and "bench range" in error for error in errors)


def test_refuses_mixed_model_keys():
    with tempfile.TemporaryDirectory() as tmp:
        cell_a = load_cell(Path(tmp), make_manifest())
        cell_b = sns.load_cell_manifest(
            write_manifest(
                Path(tmp), make_manifest(model_key="othermodel_q4"), name="cell_b.json"
            )
        )
    errors = sns.harness_refusals([cell_a, cell_b])
    assert any("mixed model_keys" in error for error in errors)


def test_refuses_schema_and_protocol_drift():
    with tempfile.TemporaryDirectory() as tmp:
        bad_schema = load_cell(Path(tmp), make_manifest(schema_version="e5-cell-manifest/2"))
        bad_protocol = sns.load_cell_manifest(
            write_manifest(
                Path(tmp), make_manifest(protocol_id="P-BENCH-1"), name="cell_p.json"
            )
        )
    errors = sns.harness_refusals([bad_schema])
    assert any("schema_version" in error for error in errors)
    errors = sns.harness_refusals([bad_protocol])
    assert any("protocol_id" in error for error in errors)


def test_refuses_unpinned_selection_and_stream_cap():
    manifest = make_manifest()
    manifest["prompt_batch"]["selection"] = "tier_seed_resample"
    with tempfile.TemporaryDirectory() as tmp:
        cell = load_cell(Path(tmp), manifest)
        errors = sns.harness_refusals([cell])
        assert any("pinned_qids" in error for error in errors)
        capped = make_manifest()
        capped["prompt_caps"]["max_total_in_flight"] = 3  # 2 instances * np2 = 4 > 3
        cell2 = load_cell(Path(tmp), capped)
        errors2 = sns.harness_refusals([cell2])
        assert any("max_total_in_flight" in error for error in errors2)


def test_revalidate_uses_schema_owner_interface():
    with tempfile.TemporaryDirectory() as tmp:
        cell = load_cell(Path(tmp))
    seen = []
    stub = SimpleNamespace(
        SCHEMA_VERSION="e5-cell-manifest/1",
        validate_cell_manifest=lambda manifest: seen.append(manifest) or ["synthetic violation"],
    )
    with patched(sns, "_manifest_interface", lambda: stub):
        errors = sns.revalidate_cells([cell])
    assert seen == [cell.manifest]
    assert any("synthetic violation" in error for error in errors)

    def unavailable():
        raise ImportError("no module")

    with patched(sns, "_manifest_interface", unavailable):
        errors = sns.revalidate_cells([cell])
    assert any("not importable" in error for error in errors)

    drifted = SimpleNamespace(SCHEMA_VERSION="e5-cell-manifest/9", validate_cell_manifest=lambda m: [])
    with patched(sns, "_manifest_interface", lambda: drifted):
        errors = sns.revalidate_cells([cell])
    assert any("does not match harness expectation" in error for error in errors)


# ---------------------------------------------------------------------------
# Pinned prompt replay
# ---------------------------------------------------------------------------


def test_pinned_qid_replay_order_and_exactness():
    with tempfile.TemporaryDirectory() as tmp:
        pool = write_pool(Path(tmp))
        cell = load_cell(Path(tmp))
        prompts = sns.load_pinned_prompts(pool, cell)
    assert [prompt.qid for prompt in prompts] == QIDS  # manifest order, extras excluded
    assert all(prompt.prompt for prompt in prompts)


def test_prompt_over_max_chars_refused():
    # The rebuilt pool's 101,655-char tulving prompt must trip the fail-closed cap.
    with tempfile.TemporaryDirectory() as tmp:
        pool = write_pool(Path(tmp), long_qid="q003")
        cell = load_cell(Path(tmp))
        try:
            sns.load_pinned_prompts(pool, cell)
        except RuntimeError as exc:
            assert "max_prompt_chars" in str(exc)
        else:
            raise AssertionError("oversized pinned prompt was not refused")


def test_missing_pinned_qid_refused():
    with tempfile.TemporaryDirectory() as tmp:
        pool = write_pool(Path(tmp), qids=QIDS[:-1])  # q006 absent from pool
        cell = load_cell(Path(tmp))
        try:
            sns.load_pinned_prompts(pool, cell)
        except RuntimeError as exc:
            assert "missing from pool" in str(exc)
        else:
            raise AssertionError("missing pinned qid was not refused")


# ---------------------------------------------------------------------------
# Dry-run default + execute double-gate
# ---------------------------------------------------------------------------


def test_dry_run_default_never_spawns():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pool = write_pool(tmp_path)
        manifest_path = write_manifest(tmp_path, make_manifest())
        with patched_many(
            (sns, "_manifest_interface", lambda: SCHEMA_STUB),
            (subprocess, "Popen", _popen_guard),
            (sns, "start_server", _popen_guard),
        ):
            rc = sns.main(
                [
                    "--cell-manifest", str(manifest_path),
                    "--question-pool", str(pool),
                    "--output-root", str(tmp_path / "out"),
                    "--run-id", "dryrun-test",
                ]
            )
        assert rc == 0
        manifest = json.loads(
            (tmp_path / "out" / "dryrun-test" / "manifest.json").read_text()
        )
        assert manifest["dry_run"] is True
        assert manifest["protocol_id"] == "P-BENCH-3"
        assert manifest["prompt_batch"]["qids"] == QIDS


def test_execute_requires_operator_grant():
    try:
        sns.parse_args(["--cell-manifest", "/tmp/x.json", "--execute"])
    except SystemExit as exc:
        assert exc.code == 2  # argparse usage error
    else:
        raise AssertionError("--execute without --i-have-operator-grant was accepted")


def test_execute_refuses_on_host_health_warning():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pool = write_pool(tmp_path)
        binary = tmp_path / "bin" / "llama-server"
        binary.parent.mkdir(parents=True)
        binary.write_text("")
        model = tmp_path / "model.gguf"
        model.write_text("")
        manifest_path = write_manifest(tmp_path, make_manifest(model_path=str(model)))
        stale = dict(CLEAN_ATTESTATION, uptime_seconds=8 * 24 * 3600.0)
        with patched_many(
            (sns, "_manifest_interface", lambda: SCHEMA_STUB),
            (sns, "ensure_clean_runtime", lambda: None),
            (sns, "collect_attestation", lambda: stale),
            (sns, "cpu_freq_static_warnings", lambda: []),
            (sns, "cpu_freq_throttle_warnings", _idle_throttle_guard),
            (sns, "start_server", _popen_guard),
            (subprocess, "Popen", _popen_guard),
        ):
            try:
                sns.main(
                    [
                        "--cell-manifest", str(manifest_path),
                        "--question-pool", str(pool),
                        "--output-root", str(tmp_path / "out"),
                        "--run-id", "refuse-test",
                        "--llama-server", str(binary),
                        "--execute", "--i-have-operator-grant",
                    ]
                )
            except RuntimeError as exc:
                assert "host-health" in str(exc)
            else:
                raise AssertionError("host-health warning did not refuse the run")


def _execute_args(tmp_path: Path, manifest_path: Path, pool: Path, binary: Path, run_id: str):
    return [
        "--cell-manifest", str(manifest_path),
        "--question-pool", str(pool),
        "--output-root", str(tmp_path / "out"),
        "--run-id", run_id,
        "--llama-server", str(binary),
        "--affinity-preflight", str(tmp_path / "fake_preflight.py"),
        "--execute", "--i-have-operator-grant",
    ]


def test_execute_happy_path_outputs_and_attestation():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pool = write_pool(tmp_path)
        binary = tmp_path / "bin" / "llama-server"
        binary.parent.mkdir(parents=True)
        binary.write_text("")
        model = tmp_path / "model.gguf"
        model.write_text("")
        manifest_path = write_manifest(tmp_path, make_manifest(model_path=str(model)))
        sent: list = []
        pids = iter([5001, 5002])
        stops: list = []

        def fake_preflight(**kwargs):
            artifact = {"live_affinity_verified": True}
            kwargs["artifact_path"].parent.mkdir(parents=True, exist_ok=True)
            kwargs["artifact_path"].write_text(json.dumps(artifact))
            return 0, artifact, "all cells matched"

        def fake_stop(proc, **kwargs):
            stops.append(proc.pid)
            return {"pid": proc.pid, "signal": "SIGTERM", "ps_verified_dead": True}

        with patched_many(
            (sns, "_manifest_interface", lambda: SCHEMA_STUB),
            (sns, "ensure_clean_runtime", lambda: None),
            (sns, "collect_attestation", lambda: dict(CLEAN_ATTESTATION)),
            (sns, "cpu_freq_static_warnings", lambda: []),
            (sns, "cpu_freq_throttle_warnings", _idle_throttle_guard),
            (sns, "run_capture", lambda cmd, timeout=10.0: "version: 10098 (fake)"),
            (sns, "start_server", lambda cmd, env, log: DummyProc(pid=next(pids))),
            (sns, "wait_for_health", lambda port, timeout, proc: None),
            (sns, "run_affinity_preflight", fake_preflight),
            (sns, "stop_instance", fake_stop),
            (sns, "send_streaming_completion", make_fake_send(sent)),
            (subprocess, "Popen", _popen_guard),
        ):
            rc = sns.main(_execute_args(tmp_path, manifest_path, pool, binary, "happy-test"))
        assert rc == 0
        run_dir = tmp_path / "out" / "happy-test"
        rows = [
            json.loads(line)
            for line in (run_dir / "cells.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        row = rows[0]
        assert row["success_count"] == len(QIDS)
        assert row["cell_error"] is None
        assert row["decision_grade"] is True
        assert row["live_affinity_verified"] is True
        assert row["ggml_iqk"] == "1"
        assert row["kv_unified"] is False
        assert abs(row["draft_accept_rate"] - 0.7) < 1e-9
        # per-cell precondition + under-load throttle records (review F1/F2/F7)
        assert row["host_health_warnings_at_cell"] == []
        assert row["throttle_check"]["status"] == "not_sampled"
        assert row["throttle_check"]["warnings"] == []
        # sampling regime always recorded (review F4)
        assert row["sampling"]["temperature"] == 0.0
        assert row["sampling"]["source"] == "default_e1_parity"
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["decision_grade"] is True
        assert manifest["attestation"]["ggml_iqk"] == "1"
        assert manifest["attestation"]["omp_env"]["KMP_BLOCKTIME"] == "10"
        assert manifest["attestation"]["binary_version"].startswith("version: 10098")
        assert manifest["attestation"]["api"].startswith("n/a")
        assert manifest["attestation"]["kv_unified_per_cell"] == {
            "testmodel_q8_0-C2-np2": False
        }
        # every raw response persisted for offline B7 scoring
        responses = [
            json.loads(line)
            for line in (run_dir / "responses.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert sorted(response["qid"] for response in responses) == sorted(QIDS)
        assert all(response["response_text"] for response in responses)
        requests = [
            json.loads(line)
            for line in (run_dir / "requests.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(requests) == len(QIDS)
        assert all("response_text" not in request for request in requests)
        # ps-verified teardown ran for both instances
        assert sorted(stops) == [5001, 5002]
        assert (run_dir / "summary.csv").exists()
        assert (run_dir / "selected_prompts.jsonl").exists()


def test_preflight_failure_aborts_cell():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pool = write_pool(tmp_path)
        binary = tmp_path / "bin" / "llama-server"
        binary.parent.mkdir(parents=True)
        binary.write_text("")
        model = tmp_path / "model.gguf"
        model.write_text("")
        manifest_path = write_manifest(tmp_path, make_manifest(model_path=str(model)))
        pids = iter([6001, 6002])
        stops: list = []

        def fake_preflight(**kwargs):
            return 1, {"live_affinity_verified": False}, "thread union mismatch"

        def fake_stop(proc, **kwargs):
            stops.append(proc.pid)
            return {"pid": proc.pid, "signal": "SIGTERM", "ps_verified_dead": True}

        def driver_guard(**kwargs):
            raise AssertionError("driver ran despite failed affinity preflight")

        with patched_many(
            (sns, "_manifest_interface", lambda: SCHEMA_STUB),
            (sns, "ensure_clean_runtime", lambda: None),
            (sns, "collect_attestation", lambda: dict(CLEAN_ATTESTATION)),
            (sns, "cpu_freq_static_warnings", lambda: []),
            (sns, "cpu_freq_throttle_warnings", _idle_throttle_guard),
            (sns, "run_capture", lambda cmd, timeout=10.0: "version"),
            (sns, "start_server", lambda cmd, env, log: DummyProc(pid=next(pids))),
            (sns, "wait_for_health", lambda port, timeout, proc: None),
            (sns, "run_affinity_preflight", fake_preflight),
            (sns, "stop_instance", fake_stop),
            (sns, "run_cell_driver", driver_guard),
            (subprocess, "Popen", _popen_guard),
        ):
            rc = sns.main(_execute_args(tmp_path, manifest_path, pool, binary, "preflight-test"))
        assert rc == 0  # the run records the failed cell and finishes honestly
        run_dir = tmp_path / "out" / "preflight-test"
        rows = [
            json.loads(line)
            for line in (run_dir / "cells.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        assert rows[0]["cell_error"] == "affinity_preflight_failed"
        assert rows[0]["decision_grade"] is False
        assert rows[0]["live_affinity_verified"] is False
        # both instances were torn down (no warn-and-continue path)
        assert sorted(stops) == [6001, 6002]
        # decision-grade-intended cell failed its hard gate => run demoted
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["decision_grade"] is False


# ---------------------------------------------------------------------------
# ps-verified kill
# ---------------------------------------------------------------------------


def test_stop_instance_sigterm_clean():
    proc = DummyProc(pid=7001)
    kills: list = []
    with patched_many(
        (sns.os, "killpg", lambda pid, sig: kills.append((pid, sig))),
        (sns, "_pid_alive", lambda pid: False),
        (time, "sleep", lambda seconds: None),
    ):
        result = sns.stop_instance(proc, timeout_s=0.1, poll_timeout_s=0.5, poll_interval_s=0.01)
    assert result["signal"] == "SIGTERM"
    assert result["killed"] is False
    assert result["ps_verified_dead"] is True
    assert kills and kills[0][1] == sns.signal.SIGTERM


def test_stop_instance_escalates_stubborn_pid():
    proc = DummyProc(pid=7002, wait_raises=1)  # SIGTERM wait times out once
    kills: list = []
    alive_sequence = iter([True, True, False])
    with patched_many(
        (sns.os, "killpg", lambda pid, sig: kills.append((pid, sig))),
        (sns, "_pid_alive", lambda pid: next(alive_sequence)),
        (time, "sleep", lambda seconds: None),
    ):
        result = sns.stop_instance(proc, timeout_s=0.1, poll_timeout_s=1.0, poll_interval_s=0.01)
    signals = [sig for _pid, sig in kills]
    assert signals[0] == sns.signal.SIGTERM
    assert sns.signal.SIGKILL in signals  # escalated
    assert result["killed"] is True
    assert result["ps_verified_dead"] is True


def test_stop_instance_raises_when_pid_never_dies():
    proc = DummyProc(pid=7003, wait_raises=2)
    with patched_many(
        (sns.os, "killpg", lambda pid, sig: None),
        (sns, "_pid_alive", lambda pid: True),
        (time, "sleep", lambda seconds: None),
    ):
        try:
            sns.stop_instance(proc, timeout_s=0.01, poll_timeout_s=0.05, poll_interval_s=0.01)
        except RuntimeError as exc:
            assert "still visible" in str(exc)
        else:
            raise AssertionError("immortal pid did not raise")


def test_stop_instance_recycled_pid_not_signaled():
    # Review F6: once our own wait() reaped the child it cannot appear in ps,
    # so a visible pid is a RECYCLED pid on an unrelated process — the poll
    # must never SIGKILL it, and the instance counts as verifiably dead.
    proc = DummyProc(pid=7004)  # SIGTERM wait succeeds -> child reaped
    kills: list = []
    with patched_many(
        (sns.os, "killpg", lambda pid, sig: kills.append((pid, sig))),
        (sns, "_pid_alive", lambda pid: True),  # recycled pid visible forever
        (time, "sleep", lambda seconds: None),
    ):
        result = sns.stop_instance(proc, timeout_s=0.1, poll_timeout_s=0.5, poll_interval_s=0.01)
    assert result["ps_verified_dead"] is True
    assert "recycled" in result["note"]
    signals = [sig for _pid, sig in kills]
    assert signals == [sns.signal.SIGTERM]  # never SIGKILL after reap


def test_teardown_cell_attempts_all_then_raises():
    good = DummyProc(pid=7010)
    bad = DummyProc(pid=7011)
    inst_a = sns.Instance(cpu_list="0-47,96-143", port=19080, threads=96, numactl_policy="none")
    inst_b = sns.Instance(cpu_list="48-95,144-191", port=19081, threads=96, numactl_policy="none")
    stopped: list = []

    def fake_stop(proc, **kwargs):
        stopped.append(proc.pid)
        if proc.pid == 7010:
            raise RuntimeError("still visible after SIGKILL escalation")
        return {"pid": proc.pid, "ps_verified_dead": True}

    with patched(sns, "stop_instance", fake_stop):
        try:
            sns.teardown_cell([(inst_a, good), (inst_b, bad)])
        except RuntimeError as exc:
            assert "surviving" in str(exc)
        else:
            raise AssertionError("surviving instance did not raise")
    assert stopped == [7010, 7011]  # the second instance was still attempted


# ---------------------------------------------------------------------------
# Driver math
# ---------------------------------------------------------------------------


def test_stream_instance_assignment_math():
    assignment = sns.stream_instance_assignment(4, 8)
    assert len(assignment) == 32
    for instance_index in range(4):
        assert assignment.count(instance_index) == 8
    assert assignment[:5] == [0, 1, 2, 3, 0]
    assert sns.stream_instance_assignment(0, 8) == []


def test_driver_round_robin_covers_each_prompt_exactly_once():
    manifest = make_manifest()
    with tempfile.TemporaryDirectory() as tmp:
        cell = load_cell(Path(tmp), manifest)  # 2 instances x np2 = 4 streams
    prompts = [
        sns.PromptSpec(qid=f"q{index:03d}", suite="synthetic", prompt=f"p{index}")
        for index in range(43)
    ]
    lock = threading.Lock()
    seen: list = []
    active = {"now": 0, "max": 0}
    expected_ports = {0: 19080, 1: 19081, 2: 19080, 3: 19081}

    def send_fn(*, stream_id, instance, prompt, request_index):
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        assert instance.port == expected_ports[stream_id]  # permanent binding
        time.sleep(0.001)
        with lock:
            active["now"] -= 1
            seen.append((stream_id, prompt.qid, request_index))
        return sns.StreamRequestRecord(
            cell_id=cell.cell_id, qid=prompt.qid, suite=prompt.suite,
            request_index=request_index, stream_id=stream_id, instance_port=instance.port,
            success=True, start_s=0.0, first_token_s=0.1, end_s=1.0, ttft_ms=100.0,
            latency_ms=1000.0, predicted_tokens=1, prompt_tokens=1, predicted_tps=1.0,
        )

    records = sns.run_cell_driver(cell=cell, prompts=prompts, send_fn=send_fn)
    assert len(records) == 43
    assert sorted(record.qid for record in records) == sorted(p.qid for p in prompts)
    assert len({record.qid for record in records}) == 43  # exactly once each
    assert active["max"] <= 4  # never more than N x K in flight
    assert [record.request_index for record in records] == list(range(43))


def test_trimmed_aggregate_math():
    def record(start, end):
        return sns.StreamRequestRecord(
            cell_id="c", qid=f"q{start}", suite="s", request_index=int(start),
            stream_id=0, instance_port=19080, success=True, start_s=start,
            first_token_s=start, end_s=end, ttft_ms=1.0, latency_ms=(end - start) * 1000,
            predicted_tokens=1, prompt_tokens=1, predicted_tps=1.0,
        )

    records = [record(0.0, 10.0), record(12.0, 20.0), record(15.0, 30.0), record(40.0, 50.0)]
    trimmed = sns.trimmed_aggregate(records)
    # ramp_end = 10 (first completion), drain_start = 40 (last start)
    assert trimmed["steady_count"] == 2
    assert abs(trimmed["window_seconds"] - 30.0) < 1e-9
    assert abs(trimmed["tasks_per_hour_trimmed"] - (2 / 30.0 * 3600.0)) < 1e-6
    assert "steady-state" in trimmed["trim_definition"]
    assert sns.trimmed_aggregate([])["tasks_per_hour_trimmed"] == 0.0


def test_empty_trimmed_window_demotes_decision_grade_cell(tmp_path):
    cell = load_cell(tmp_path)
    records = [
        sns.StreamRequestRecord(
            cell_id=cell.cell_id, qid=f"q{index}", suite="s", request_index=index,
            stream_id=index, instance_port=19080, success=True, start_s=0.0,
            first_token_s=0.1, end_s=1.0, ttft_ms=100.0, latency_ms=1000.0,
            predicted_tokens=1, prompt_tokens=1, predicted_tps=1.0,
        )
        for index in range(4)
    ]
    row = sns.summarize_cell(
        cell=cell, records=records, wall_s=2.0, env={"GGML_IQK": "1"},
        instance_pids={19080: 1}, affinity={"live_affinity_verified": True},
        run_overrides_active=False, host_warnings=[], throttle_check={"warnings": []},
    )
    assert row["tasks_per_hour_trimmed"] == 0.0
    assert row["trimmed_window_ready"] is False
    assert row["decision_grade"] is False
    assert row["decision_grade_blockers"] == [
        "empty_trimmed_window: raw ramp+drain fallback is observation-only"
    ]


def test_aggregate_decode_tps_sums_per_stream_rates():
    # Operator ruling 2026-07-30: the PRIMARY metric is aggregate decode tok/s,
    # taken from llama.cpp's own predicted_n/predicted_ms — never wall clock.
    # Two streams, 2 requests each. Stream 0: 100 tok in 10s -> 10 tok/s.
    # Stream 1: 300 tok in 10s -> 30 tok/s. Aggregate = 40 tok/s (system-wide,
    # the SUM), per-stream = 400/20 = 20 tok/s (token-weighted mean per slot).
    # Wall time is deliberately 1000s so any wall-clock contamination shows up.
    def record(stream_id, index, predicted_n, predicted_ms):
        return sns.StreamRequestRecord(
            cell_id="c", qid=f"q{index}", suite="s", request_index=index,
            stream_id=stream_id, instance_port=19080, success=True,
            start_s=float(index), first_token_s=float(index) + 0.1,
            end_s=float(index) + 5.0, ttft_ms=100.0, latency_ms=5000.0,
            predicted_tokens=predicted_n, prompt_tokens=10,
            predicted_tps=predicted_n / (predicted_ms / 1000.0),
            timings={"predicted_n": predicted_n, "predicted_ms": predicted_ms},
        )

    records = [
        record(0, 0, 50, 5000.0), record(0, 1, 50, 5000.0),
        record(1, 2, 150, 5000.0), record(1, 3, 150, 5000.0),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        cell = load_cell(Path(tmp))
    row = sns.summarize_cell(
        cell=cell, records=records, wall_s=1000.0, env={"GGML_IQK": "1"},
        instance_pids={19080: 1}, affinity={"live_affinity_verified": True},
        run_overrides_active=False, host_warnings=[], throttle_check={"warnings": []},
    )
    assert abs(row["aggregate_decode_tps"] - 40.0) < 1e-9
    assert abs(row["per_stream_decode_tps"] - 20.0) < 1e-9
    assert row["decode_tokens_total"] == 400
    assert abs(row["decode_seconds_total"] - 20.0) < 1e-9
    assert row["decode_stream_count"] == 2
    # per-stream rates reported ALONGSIDE the aggregate, and they sum to it
    per_stream = row["per_stream"]
    assert abs(per_stream["0"]["decode_tps"] - 10.0) < 1e-9
    assert abs(per_stream["1"]["decode_tps"] - 30.0) < 1e-9
    assert abs(
        sum(entry["decode_tps"] for entry in per_stream.values())
        - row["aggregate_decode_tps"]
    ) < 1e-9
    # NEVER wall clock: the wall-clock rate is a different, much smaller number
    assert abs(row["aggregate_wallclock_tps"] - 0.4) < 1e-9
    # tasks/hour is still computed and persisted, as a secondary diagnostic
    assert row["tasks_per_hour_raw"] > 0
    assert "tasks_per_hour_trimmed" in row
    # a malformed/missing timings block drops the request from BOTH sums
    broken = records + [
        sns.StreamRequestRecord(
            cell_id="c", qid="qx", suite="s", request_index=9, stream_id=2,
            instance_port=19080, success=True, start_s=0.0, first_token_s=0.1,
            end_s=1.0, ttft_ms=1.0, latency_ms=1000.0, predicted_tokens=999,
            prompt_tokens=1, predicted_tps=999.0, timings={"predicted_n": 999},
        )
    ]
    row2 = sns.summarize_cell(
        cell=cell, records=broken, wall_s=1000.0, env={"GGML_IQK": "1"},
        instance_pids={19080: 1}, affinity={"live_affinity_verified": True},
        run_overrides_active=False, host_warnings=[], throttle_check={"warnings": []},
    )
    assert abs(row2["aggregate_decode_tps"] - 40.0) < 1e-9
    assert row2["decode_tokens_total"] == 400


def test_cell_throughput_tps_prefers_decode_then_falls_back():
    # Reader tolerance (append-only data): prefer the new key, fall back to the
    # old ones so historical runs stay rankable — with the basis recorded.
    fresh = {"aggregate_decode_tps": 42.0, "aggregate_wallclock_tps": 11.0}
    assert sns.cell_throughput_tps(fresh) == 42.0
    assert sns.throughput_basis(fresh) == "decode"
    renamed = {"aggregate_wallclock_tps": 11.0}
    assert sns.cell_throughput_tps(renamed) == 11.0
    assert sns.throughput_basis(renamed) == "wallclock_fallback"
    legacy = {"aggregate_predicted_tps": 9.0}  # pre-2026-07-29 key
    assert sns.cell_throughput_tps(legacy) == 9.0
    assert sns.throughput_basis(legacy) == "wallclock_fallback"
    # tasks/hour never feeds the primary metric, however large it is
    tasks_only = {"tasks_per_hour_trimmed": 5000.0, "tasks_per_hour_raw": 5000.0}
    assert sns.cell_throughput_tps(tasks_only) == 0.0
    assert sns.throughput_basis(tasks_only) == "none"
    # ...but it is still readable as the labelled secondary quantity
    assert sns.cell_aggregate(tasks_only) == 5000.0
    assert sns.aggregate_basis(tasks_only) == "trimmed"


def test_summary_row_fixture_drives_the_primary_metric():
    # ANTI-SHORT-CIRCUIT GUARD (operator ruling 2026-07-30, and the 2026-07-30
    # fixture defect where a test wrote `tasks_per_hour` while cell_aggregate
    # read `tasks_per_hour_raw`, so both sides were 0 and the pair verdict
    # short-circuited to "insufficient_data" without ever reaching the ranking).
    #
    # Here the two orderings DISAGREE on purpose: tasks/hour ranks LEFT first,
    # tok/s ranks RIGHT first. A verdict computed on the demoted metric — or a
    # fixture that stops writing the field the code reads (the wall-clock
    # fallback preserves the tasks/hour ordering) — cannot pass this test.
    left = summary_row(
        "C1b", 4, 120.0, 24000.0,
        aggregate_decode_tps=40.0, per_stream_decode_tps=10.0,
        aggregate_wallclock_tps=20.0,
    )
    right = summary_row(
        "C3", 2, 100.0, 15000.0,
        aggregate_decode_tps=90.0, per_stream_decode_tps=45.0,
        aggregate_wallclock_tps=45.0,
    )
    assert sns.cell_aggregate(left) > sns.cell_aggregate(right)          # tasks/hour
    assert sns.cell_throughput_tps(left) < sns.cell_throughput_tps(right)  # tok/s

    pair = sns._pair_verdict(left, right, label="metric-ruling", prefer_on_tie="C1b")
    assert pair["metric"] == "aggregate_decode_tps"
    assert pair["status"] == "winner"          # 55.6% tok/s margin, not a tie
    assert pair["winner_config"] == "C3"       # tok/s wins; tasks/hour would say C1b
    assert pair["cells"][right["cell_id"]] == 90.0
    # tasks/hour is NOT deleted — it rides along, explicitly labelled secondary
    assert pair["cells_tasks_per_hour_secondary"][left["cell_id"]] == 120.0
    assert pair["cells_tasks_per_hour_secondary"][right["cell_id"]] == 100.0

    rows = [left, right]
    r2 = sns.evaluate_r2(rows)["models"]["testmodel_q8_0+Q8_0"]
    assert r2["peak_cell"] == right["cell_id"]          # peak on tok/s
    assert r2["peak_decode_tps"] == 90.0
    assert r2["peak_tasks_per_hour_secondary"] == 100.0
    assert [entry["cell_id"] for entry in r2["pareto"]][0] == right["cell_id"]

    r4 = sns.evaluate_r4(rows)[0]
    # >= 90% of peak tok/s (81) admits only C3@2; on tasks/hour (>=108) it
    # would have admitted only C1b@4 and recommended the opposite shape.
    assert r4["recommended"]["config_id"] == "C3"
    assert r4["recommended"]["aggregate_decode_tps"] == 90.0
    assert r4["recommended"]["per_stream_decode_tps"] == 45.0
    assert r4["recommended"]["aggregate_tasks_per_hour"] == 100.0  # secondary kept
    assert "tok/s" in r4["recommended"]["rule"]


def test_percentile_math():
    values = [float(value) for value in range(1, 101)]
    assert sns.percentile(values, 0.50) == 50.0
    assert sns.percentile(values, 0.95) == 95.0
    assert sns.percentile([], 0.95) == 0.0
    assert sns.percentile([7.0], 0.95) == 7.0


def test_parse_sse_line():
    assert sns.parse_sse_line('data: {"content": "hi", "stop": false}') == {
        "content": "hi",
        "stop": False,
    }
    assert sns.parse_sse_line("data: [DONE]") is None
    assert sns.parse_sse_line(": keepalive") is None
    assert sns.parse_sse_line("") is None


def test_gemma_reasoning_only_chat_stream_fails_closed_without_answer_text(tmp_path):
    # llama.cpp OpenAI-compatible streaming schema. This is the W0 Gemma
    # survivor shape: generated tokens arrive in reasoning_content, but no
    # answer-text delta appears before the token budget runs out.
    # Re-attributed 2026-07-29 (research 5d6a17f2): the budget is consumed
    # because reasoning is ON — the harness emitted no `--reasoning` flag and
    # gemma4 defaults to `auto` (= on) — not because of the scout cap. This
    # test asserts the fail-close, which detects the shape; `--reasoning off`
    # is what prevents it.
    chunk = {
        "choices": [{"delta": {"reasoning_content": "Let me work this out."}}]
    }
    assert sns.stream_chunk_text(chunk) == ("", "Let me work this out.")
    error = sns.response_capture_error(
        status=200, predicted_tokens=64, response_text=""
    )
    assert error.startswith("response_capture_missing_answer_text:")

    cell = load_cell(tmp_path)
    record = sns.StreamRequestRecord(
        cell_id=cell.cell_id, qid="gemma-reasoning-only", suite="s", request_index=0,
        stream_id=0, instance_port=19080, success=False, start_s=0.0,
        first_token_s=None, end_s=1.0, ttft_ms=None, latency_ms=1000.0,
        predicted_tokens=64, prompt_tokens=100, predicted_tps=30.0,
        http_status=200, error=error, response_text="",
        reasoning_text="Let me work this out.", timings={"predicted_n": 64},
    )
    row = sns.summarize_cell(
        cell=cell, records=[record], wall_s=1.0, env={"GGML_IQK": "1"},
        instance_pids={19080: 1}, affinity={"live_affinity_verified": True},
        run_overrides_active=False, host_warnings=[], throttle_check={"warnings": []},
    )
    assert row["response_capture_failure_count"] == 1
    assert row["decision_grade"] is False
    assert row["decision_grade_blockers"] == [
        "empty_trimmed_window: raw ramp+drain fallback is observation-only",
        "response_capture_failure: 1 generated response(s) lacked answer-text SSE deltas",
    ]


# ---------------------------------------------------------------------------
# Throttle gate (review F1: idle-valid static checks + under-load sampling)
# ---------------------------------------------------------------------------


def _write_sysfs_cpu(base: Path, index: int, cur_freq: int, max_freq: int = 3_700_000,
                     core_key: str | None = None):
    cpu = base / f"cpu{index}"
    (cpu / "cpufreq").mkdir(parents=True)
    (cpu / "cpufreq" / "scaling_cur_freq").write_text(str(cur_freq))
    (cpu / "cpufreq" / "scaling_max_freq").write_text(str(max_freq))
    if core_key is not None:
        (cpu / "topology").mkdir()
        (cpu / "topology" / "core_cpus_list").write_text(core_key + "\n")


def test_cpu_freq_underload_gate_semantics():
    assert sns.cpu_freq_throttle_warnings([2_600_000] * 96) == []
    warnings = sns.cpu_freq_throttle_warnings([2_000_000] * 96)
    assert warnings and "under load" in warnings[0]
    # 79 boosting < FREQ_BOOST_MIN_CORES (80-of-96 physical semantics)
    assert sns.cpu_freq_throttle_warnings([2_600_000] * 79 + [2_000_000] * 17)
    assert sns.cpu_freq_throttle_warnings([])  # unreadable => warning, never silent


def test_underload_gate_scopes_to_pinned_cores():
    # Operator-ratified 2026-07-29. A C1/C2 cell pins 48 of 96 physical cores;
    # the idle remainder parks near base clock, so a machine-wide count can
    # never reach 80 and the gate failed a healthy host. Reproduces the exact
    # W0 shape: 48 pinned cores boosting, 48 unpinned idle.
    freqs = {core: 2_600_000 for core in range(48)}
    freqs.update({core: 2_000_000 for core in range(48, 96)})
    pinned = set(range(48))

    # machine-wide scope: 48/96 boosting -> the false failure being fixed
    assert sns.cpu_freq_throttle_warnings(freqs)
    # cell-pinned scope: 48/48 boosting -> healthy
    assert sns.cpu_freq_throttle_warnings(freqs, pinned) == []

    # the SAME ratio still binds inside the pinned set: 39/48 < ceil(.833*48)=40
    throttled = {core: 2_600_000 for core in range(39)}
    throttled.update({core: 2_000_000 for core in range(39, 48)})
    throttled.update({core: 2_600_000 for core in range(48, 96)})
    warnings = sns.cpu_freq_throttle_warnings(throttled, pinned)
    assert warnings and "39/48" in warnings[0] and "need 40" in warnings[0]


def test_full_machine_gate_is_unchanged_by_scoping():
    # A 96-core cell must still need exactly 80 — the scoping change must not
    # loosen any gate that was already correct (C1b/C3 passed W0 cleanly).
    assert sns.freq_boost_min_cores(96) == sns.FREQ_BOOST_MIN_CORES == 80
    all_cores = set(range(96))
    at_79 = {core: 2_600_000 for core in range(79)}
    at_79.update({core: 2_000_000 for core in range(79, 96)})
    assert sns.cpu_freq_throttle_warnings(at_79, all_cores)
    at_80 = dict(at_79)
    at_80[79] = 2_600_000
    assert sns.cpu_freq_throttle_warnings(at_80, all_cores) == []


def test_throttle_check_persists_per_core_vector():
    # Without the per-core vector a throttle verdict is not re-derivable
    # offline — the retention gap that made the pre-fix W0 gate un-rescoreable.
    sampler = sns.FreqSampler(
        interval_s=0.005,
        read_fn=lambda: {c: 2_600_000 for c in range(96)},
        pinned_cores=set(range(48)),
    )
    sampler.start()
    time.sleep(0.05)
    sampler.stop()
    result = sampler.result()
    assert result["scope"] == "cell_pinned"
    assert result["n_physical_cores"] == 48          # scoped denominator
    assert result["n_physical_cores_host"] == 96     # host-wide evidence kept
    assert result["min_boosting_cores"] == 40        # same 80/96 ratio
    assert result["pinned_physical_cores"] == list(range(48))
    # full host vector retained, not just the aggregate count
    assert len(result["per_core_khz"]) == 96
    assert result["per_core_khz"]["95"] == 2_600_000


def test_resolve_pinned_physical_cores_from_cpusets():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for logical, key in ((0, "0,96"), (96, "0,96"), (1, "1,97"), (97, "1,97")):
            _write_sysfs_cpu(base, logical, 2_600_000, core_key=key)
        # a cpuset naming only the HIGH sibling still pins the physical core
        assert sns.resolve_pinned_physical_cores(["96"], base) == {0}
        assert sns.resolve_pinned_physical_cores(["0-1,96-97"], base) == {0, 1}
        # multi-instance cells (C3 = 4 quarters) union their cpusets
        assert sns.resolve_pinned_physical_cores(["0,96", "1,97"], base) == {0, 1}


def test_parse_cpu_list_ranges_and_singletons():
    assert sns.parse_cpu_list("0-3") == {0, 1, 2, 3}
    assert sns.parse_cpu_list("0-47,96-143") == set(range(48)) | set(range(96, 144))
    assert sns.parse_cpu_list("5") == {5}
    assert sns.parse_cpu_list("") == set()


def test_read_physical_core_freqs_dedupes_smt_siblings():
    # 80-of-96 is calibrated for PHYSICAL cores; SMT siblings share one clock
    # domain and must not be double-counted (192 logical entries, review F1).
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_sysfs_cpu(base, 0, 2_600_000, core_key="0,96")
        _write_sysfs_cpu(base, 96, 2_000_000, core_key="0,96")  # parked sibling
        _write_sysfs_cpu(base, 1, 2_550_000, core_key="1,97")
        _write_sysfs_cpu(base, 97, 1_900_000, core_key="1,97")
        freqs = sns.read_physical_core_freqs(base)
    # keyed by representative (lowest) logical cpu of each sibling group
    assert freqs == {0: 2_600_000, 1: 2_550_000}  # max per sibling group


def test_cpu_freq_static_warnings_synthetic_sysfs():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_sysfs_cpu(base, 0, 2_000_000, max_freq=3_700_000)
        _write_sysfs_cpu(base, 1, 2_000_000, max_freq=3_700_000)
        # idle-parked CUR freqs alone are NOT a warning (the F1 false-fail)
        assert sns.cpu_freq_static_warnings(base) == []
        # a max-freq cap below the boost threshold IS static throttle state
        (base / "cpu1" / "cpufreq" / "scaling_max_freq").write_text("2000000")
        warnings = sns.cpu_freq_static_warnings(base)
        assert warnings and "scaling_max_freq" in warnings[0]
        # global boost flag 0 is static throttle state
        (base / "cpufreq").mkdir()
        (base / "cpufreq" / "boost").write_text("0\n")
        warnings = sns.cpu_freq_static_warnings(base)
        assert any("boost flag" in warning for warning in warnings)


def test_freq_sampler_best_sample_and_not_sampled():
    # stopped before the first interval elapses: no idle-time false reading
    sampler = sns.FreqSampler(interval_s=60.0, read_fn=lambda: [2_600_000] * 96)
    sampler.start()
    sampler.stop()
    result = sampler.result()
    assert result["status"] == "not_sampled" and result["warnings"] == []
    # sampled under load, healthy
    sampler = sns.FreqSampler(interval_s=0.005, read_fn=lambda: [2_600_000] * 96)
    sampler.start()
    time.sleep(0.05)
    sampler.stop()
    result = sampler.result()
    assert result["status"] == "ok" and result["samples"] >= 1
    assert result["boosting_physical_cores"] == 96
    # sampled under load, throttled
    sampler = sns.FreqSampler(interval_s=0.005, read_fn=lambda: [2_000_000] * 96)
    sampler.start()
    time.sleep(0.05)
    sampler.stop()
    result = sampler.result()
    assert result["status"] == "warning"
    assert result["warnings"] and "throttled" in result["warnings"][0]


def test_execute_percell_host_flip_refuses_cell():
    # Review F2/F7: protocol decision 5 makes host health a PER-CELL gate — a
    # mid-run numa_balancing flip must refuse the cell before any launch.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pool = write_pool(tmp_path)
        binary = tmp_path / "bin" / "llama-server"
        binary.parent.mkdir(parents=True)
        binary.write_text("")
        model = tmp_path / "model.gguf"
        model.write_text("")
        manifest_path = write_manifest(tmp_path, make_manifest(model_path=str(model)))
        attestations = iter(
            [dict(CLEAN_ATTESTATION), dict(CLEAN_ATTESTATION, numa_balancing="1")]
        )
        with patched_many(
            (sns, "_manifest_interface", lambda: SCHEMA_STUB),
            (sns, "ensure_clean_runtime", lambda: None),
            (sns, "collect_attestation", lambda: next(attestations)),
            (sns, "cpu_freq_static_warnings", lambda: []),
            (sns, "cpu_freq_throttle_warnings", _idle_throttle_guard),
            (sns, "run_capture", lambda cmd, timeout=10.0: "version"),
            (sns, "start_server", _popen_guard),  # flip detected BEFORE any launch
            (subprocess, "Popen", _popen_guard),
        ):
            rc = sns.main(_execute_args(tmp_path, manifest_path, pool, binary, "flip-test"))
        assert rc == 0  # the run records the refused cell and finishes honestly
        run_dir = tmp_path / "out" / "flip-test"
        rows = [
            json.loads(line)
            for line in (run_dir / "cells.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        assert "host-health" in rows[0]["cell_error"]
        assert rows[0]["decision_grade"] is False
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["decision_grade"] is False
        events = [
            json.loads(line)
            for line in (run_dir / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert any(event["event"] == "cell_precondition_warning" for event in events)


def test_execute_throttle_warning_during_driver_demotes():
    # An under-load throttle warning sampled during the driver demotes the
    # cell (and the run) instead of passing silently (review F1/F7).
    class FakeThrottleSampler:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def result(self):
            return {
                "status": "warning",
                "samples": 3,
                "boosting_physical_cores": 40,
                "n_physical_cores": 96,
                "warnings": ["only 40/96 physical cores boosting under load"],
            }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pool = write_pool(tmp_path)
        binary = tmp_path / "bin" / "llama-server"
        binary.parent.mkdir(parents=True)
        binary.write_text("")
        model = tmp_path / "model.gguf"
        model.write_text("")
        manifest_path = write_manifest(tmp_path, make_manifest(model_path=str(model)))
        pids = iter([8001, 8002])

        def fake_preflight(**kwargs):
            artifact = {"live_affinity_verified": True}
            kwargs["artifact_path"].parent.mkdir(parents=True, exist_ok=True)
            kwargs["artifact_path"].write_text(json.dumps(artifact))
            return 0, artifact, "all cells matched"

        with patched_many(
            (sns, "_manifest_interface", lambda: SCHEMA_STUB),
            (sns, "ensure_clean_runtime", lambda: None),
            (sns, "collect_attestation", lambda: dict(CLEAN_ATTESTATION)),
            (sns, "cpu_freq_static_warnings", lambda: []),
            (sns, "FreqSampler", FakeThrottleSampler),
            (sns, "run_capture", lambda cmd, timeout=10.0: "version"),
            (sns, "start_server", lambda cmd, env, log: DummyProc(pid=next(pids))),
            (sns, "wait_for_health", lambda port, timeout, proc: None),
            (sns, "run_affinity_preflight", fake_preflight),
            (sns, "stop_instance", lambda proc, **kw: {"pid": proc.pid, "ps_verified_dead": True}),
            (sns, "send_streaming_completion", make_fake_send([])),
            (subprocess, "Popen", _popen_guard),
        ):
            rc = sns.main(_execute_args(tmp_path, manifest_path, pool, binary, "throttle-test"))
        assert rc == 0
        run_dir = tmp_path / "out" / "throttle-test"
        rows = [
            json.loads(line)
            for line in (run_dir / "cells.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        assert rows[0]["cell_error"] is None  # the cell ran; its grade demoted
        assert rows[0]["throttle_check"]["status"] == "warning"
        assert rows[0]["decision_grade"] is False
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["decision_grade"] is False


# ---------------------------------------------------------------------------
# Summarizer (R1-R4) on synthetic fixtures
# ---------------------------------------------------------------------------


def summary_row(config_id, np_level, aggregate, p95, **overrides):
    """Synthetic summarizer row.

    ``aggregate`` populates the PRIMARY ranked metric — ``aggregate_decode_tps``
    (operator ruling 2026-07-30) — AND the secondary tasks/hour fields, so a
    fixture can never silently rank on the demoted metric. Guarded by
    test_summary_row_fixture_drives_the_primary_metric: if the code stops
    reading aggregate_decode_tps, or if this fixture stops writing it, that
    test fails instead of every ranking assertion quietly short-circuiting on
    a 0-vs-0 comparison (the exact fixture/field-name mismatch that made a
    pair verdict short-circuit to "insufficient_data" on 2026-07-30).
    """
    row = {
        "cell_id": f"testmodel_q8_0-{config_id}-np{np_level}",
        "model_key": "testmodel_q8_0",
        "quant": "Q8_0",
        "config_id": config_id,
        "np": np_level,
        "success_count": 43,
        "total_count": 43,
        "wall_seconds": 900.0,
        # PRIMARY — what every R1/R2/R4 decision ranks on
        "aggregate_decode_tps": aggregate,
        "per_stream_decode_tps": aggregate / max(np_level, 1),
        "decode_tokens_total": 43 * 256,
        "decode_seconds_total": 43 * 256 / aggregate if aggregate else 0.0,
        "decode_stream_count": np_level,
        # SECONDARY / DIAGNOSTIC — still emitted, never ranked on
        "tasks_per_hour_raw": aggregate,
        "tasks_per_hour_trimmed": aggregate,
        "aggregate_wallclock_tps": aggregate * 0.5,
        "p50_latency_ms": p95 * 0.6,
        "p95_latency_ms": p95,
        "ttft_p50_ms": 500.0,
        "ttft_p95_ms": 900.0,
        "draft_accept_rate": 0.7,
        "kv_unified": False,
        "per_stream_ctx": 2048,
        "ctx": max(8192, 2048 * np_level),
        "kv": {"type_k": "q8_0", "type_v": "q8_0", "flash_attn": True, "kv_unified": False},
        "spec_dec": {"enabled": True, "spec_type": "draft-mtp", "draft_max": 4},
        "decision_grade": True,
        "cell_error": None,
    }
    row.update(overrides)
    return row


def r1_fixture_rows():
    return [
        summary_row("C1", 1, 30.0, 20000.0),
        summary_row("C1", 4, 60.0, 22000.0),
        summary_row("C1", 8, 110.0, 26000.0),
        summary_row("C1", 16, 90.0, 30000.0),
        summary_row("C1b", 4, 120.0, 24000.0),
        summary_row("C1b", 8, 105.0, 28000.0),
        summary_row("C2", 8, 80.0, 26000.0),
        summary_row("C3", 1, 25.0, 10000.0),
        summary_row("C3", 2, 100.0, 15000.0),
        summary_row("C3", 4, 100.0, 21000.0),
    ]


def test_summarizer_r1_winner_tie_and_kstar():
    rules = sns.evaluate_r1(r1_fixture_rows())
    pairs = {pair["pair"]: pair for pair in rules["iso_t_pairs"]}
    t8 = pairs["whole-machine T=8: C1b@4 vs C3@2"]
    assert t8["status"] == "winner" and t8["winner_config"] == "C1b"
    t16 = pairs["whole-machine T=16: C1b@8 vs C3@4"]
    assert t16["status"] == "tie" and t16["preferred"] == "C3"  # <10% => quarters win ties
    t32 = pairs["whole-machine T=32: C1b@16 vs C3@8"]
    assert t32["status"] == "insufficient_data"
    h16 = pairs["half-machine T=16: C1@16 vs C2@8"]
    assert h16["status"] == "winner" and h16["winner_config"] == "C1"
    assert rules["k_star_roofline_flip"] == 8  # C1@8=110 > C3@2=100
    scaling = {entry["k"]: entry for entry in rules["scaling_pairs"]}
    assert abs(scaling[4]["c1b_over_c1"] - 2.0) < 1e-9
    assert scaling[8]["status"] == "ok"
    assert rules["anchors"]["C1@1"] == "testmodel_q8_0-C1-np1"
    assert rules["anchors"]["C3@1"] == "testmodel_q8_0-C3-np1"


def test_summarizer_r2_lanes_real_and_not():
    real_rows = [
        summary_row("C3", 1, 20.0, 10000.0),
        summary_row("C3", 4, 100.0, 25000.0),
        summary_row("C3", 8, 120.0, 45000.0),  # peak: p95 > 3x K=1 p95
    ]
    rules = sns.evaluate_r2(real_rows)
    verdict = rules["models"]["testmodel_q8_0+Q8_0"]
    assert verdict["sla_violated_at_peak"] is True
    assert verdict["holder_cell"] == "testmodel_q8_0-C3-np4"  # 100 >= 70% of 120, within SLA
    assert verdict["lanes_real"] is True

    flat_rows = [
        summary_row("C3", 1, 20.0, 10000.0),
        summary_row("C3", 8, 120.0, 25000.0),  # peak within 3x K=1
    ]
    rules = sns.evaluate_r2(flat_rows)
    verdict = rules["models"]["testmodel_q8_0+Q8_0"]
    assert verdict["sla_violated_at_peak"] is False
    assert verdict["lanes_real"] is False
    # Pareto front is (aggregate desc, p95 asc) non-dominated
    pareto_ids = [entry["cell_id"] for entry in verdict["pareto"]]
    assert "testmodel_q8_0-C3-np8" in pareto_ids


def test_summarizer_r1_gemma_c1full_whole_machine_pairs():
    # Review F1/F2: gemma has no C1b/C2 — the pre-registered whole-machine
    # pairs are {C1full@T vs C3@T/4} under the same 10%-margin/tie->C3 rule.
    rows = []
    for np_level, aggregate in ((8, 100.0), (16, 130.0), (32, 150.0)):
        row = summary_row(
            "C1", np_level, aggregate, 30000.0,
            cell_id=f"gemma-C1-np{np_level}",
            stage_b_families=[f"whole_machine_T{np_level}"],
        )
        row["instances"] = [
            {"cpu_list": "0-95", "port": 19380, "threads": 96,
             "numactl_policy": "interleave=all"},
        ]
        rows.append(row)
    rows += [
        summary_row("C3", 2, 80.0, 15000.0, cell_id="gemma-C3-np2"),
        summary_row("C3", 4, 125.0, 20000.0, cell_id="gemma-C3-np4"),
        summary_row("C3", 8, 140.0, 25000.0, cell_id="gemma-C3-np8"),
    ]
    rules = sns.evaluate_r1(rows)
    assert rules["c1_whole_machine"] is True
    pairs = {pair["pair"]: pair for pair in rules["iso_t_pairs"]}
    t8 = pairs["whole-machine T=8: C1@8 vs C3@2"]
    assert t8["status"] == "winner" and t8["winner_config"] == "C1"  # 20% margin
    t16 = pairs["whole-machine T=16: C1@16 vs C3@4"]
    assert t16["status"] == "tie" and t16["preferred"] == "C3"  # ~3.8% margin
    t32 = pairs["whole-machine T=32: C1@32 vs C3@8"]
    assert t32["status"] == "tie" and t32["preferred"] == "C3"  # ~6.7% margin
    # qwen-style families (with C1b) keep the C1b-vs-C3 pairing
    assert sns.evaluate_r1(r1_fixture_rows())["c1_whole_machine"] is False


def test_summarizer_variant_rows_excluded_and_probes_emitted():
    # Review F3/F5: the -kvu and -scout-full paired-probe variants must never
    # be conflated into canonical (config, np) picks; they surface as
    # explicit scout-probe comparisons instead (M06 >=5% kvu escalation).
    rows = [
        summary_row("C1", 16, 100.0, 30000.0, cell_id="qwen36_q8_0-C1-np16-scout"),
        summary_row(
            "C1", 16, 120.0, 30000.0, cell_id="qwen36_q8_0-C1-np16-scout-kvu",
            kv_unified=True, stage_b_families=["scout_kvu_probe"],
        ),
        summary_row("C1", 1, 30.0, 20000.0, cell_id="qwen36_27b_q8-C1-np1-scout"),
        summary_row(
            "C1", 1, 50.0, 15000.0, cell_id="qwen36_27b_q8-C1-np1-scout-full",
            stage_b_families=["scout_dense_c1_shape_pair"],
        ),
    ]
    # _find never returns the higher-aggregate variant
    assert sns._find(rows, "C1", 16)["cell_id"] == "qwen36_q8_0-C1-np16-scout"
    assert sns._find(rows, "C1", 1)["cell_id"] == "qwen36_27b_q8-C1-np1-scout"
    probes = sns.evaluate_scout_probes(rows)
    kvu = probes["kvu_split_pairs"][0]
    assert kvu["status"] == "ok"
    assert kvu["split_cell"] == "qwen36_q8_0-C1-np16-scout"
    # escalation delta is computed on tok/s (operator ruling 2026-07-30)
    assert kvu["metric"] == "aggregate_decode_tps"
    assert kvu["kvu_decode_tps"] == 120.0 and kvu["split_decode_tps"] == 100.0
    assert abs(kvu["delta_fraction"] - 0.2) < 1e-9
    assert kvu["escalate_to_operator"] is True  # >= 5% delta (M06)
    # tasks/hour is not deleted, just demoted to a labelled secondary readout
    assert kvu["kvu_tasks_per_hour"] == 120.0
    shape = probes["dense_c1_shape_pairs"][0]
    assert shape["status"] == "ok"
    assert shape["winner_shape"] == "full"
    assert shape["full_decode_tps"] == 50.0 and shape["half0_decode_tps"] == 30.0
    assert shape["half_cell"] == "qwen36_27b_q8-C1-np1-scout"
    # R4 recommended pick also excludes variants (kvu 120 > split 100)
    r4 = sns.evaluate_r4(rows)
    assert r4[0]["recommended"]["config_id"] == "C1"
    assert r4[0]["recommended"]["np"] == 16
    assert r4[0]["recommended"]["throughput_basis"] == "decode"
    assert r4[0]["recommended"]["aggregate_decode_tps"] == 100.0
    assert r4[0]["recommended"]["aggregate_tasks_per_hour"] == 100.0  # secondary


def test_summarizer_mixed_metric_basis_flagged():
    # Review F3: trimmed-vs-raw_fallback comparisons must be flagged, never
    # silently mixed (raw includes ramp+drain and understates steady-state).
    left = summary_row("C1b", 4, 120.0, 24000.0)
    right = summary_row(
        "C3", 2, 100.0, 15000.0, cell_id="testmodel_q8_0-C3-np2-rawonly",
        tasks_per_hour_trimmed=0.0,  # empty steady window -> raw fallback
    )
    pair = sns._pair_verdict(left, right, label="x", prefer_on_tie="C3")
    assert pair["mixed_metric_basis"] is True
    assert pair["status"] == "winner_caveated"
    assert pair["winner_config"] == "C1b"
    assert pair["metric_basis"][right["cell_id"]] == "raw_fallback"
    assert pair["metric_basis"][left["cell_id"]] == "trimmed"
    assert "caveated" in pair["basis_note"]
    same = sns._pair_verdict(
        left, summary_row("C3", 2, 100.0, 15000.0), label="y", prefer_on_tie="C3"
    )
    assert same["mixed_metric_basis"] is False
    assert same["status"] == "winner"
    assert "basis_note" not in same
    observation_pair = sns._pair_verdict(
        summary_row(
            "C1", 32, 120.0, 30000.0,
            decision_grade=False,
            tasks_per_hour_trimmed=0.0,
        ),
        summary_row(
            "C2", 16, 100.0, 30000.0,
            decision_grade=False,
            tasks_per_hour_trimmed=0.0,
        ),
        label="observation-only",
        prefer_on_tie="C2",
    )
    assert observation_pair["mixed_metric_basis"] is False
    assert observation_pair["decision_grade"] is False
    assert observation_pair["status"] == "winner_caveated"
    assert "observation-only" in observation_pair["grade_note"]


def test_r2_r4_demote_status_on_mixed_basis_like_r1_does():
    """A5: R2/R4 computed both basis flags and then ignored them.

    R1 has demoted to `winner_caveated` on a mixed basis since review F3. R2 and
    R4 emitted `status="decision_grade"` next to their own
    `mixed_throughput_basis: true` — the machine-readable key claimed
    decision-grade while the flag beside it said the comparison was not
    like-for-like. All three rules now agree.
    """
    # R2 builds its basis maps from the PEAK, that config's K=1 baseline and any
    # holder — not from every row — so the mix has to exist across those cells.
    # Peak is C1@8 (decode basis); its K=1 baseline has no decode sample, so
    # throughput_basis falls back to wall-clock.
    mixed = [
        summary_row("C1", 8, 120.0, 20000.0),
        summary_row(
            "C1", 1, 40.0, 8000.0,
            cell_id="testmodel_q8_0-C1-np1-wallclock",
            aggregate_decode_tps=0.0,
            decode_tokens_total=0,
            decode_seconds_total=0.0,
        ),
    ]
    bases = {sns.throughput_basis(row) for row in mixed}
    assert bases == {"decode", "wallclock_fallback"}, (
        f"fixture must actually mix the primary basis, got {bases} — otherwise "
        "this test passes by never exercising the guard"
    )

    r2 = sns.evaluate_r2(mixed)["models"]["testmodel_q8_0+Q8_0"]
    assert r2["mixed_throughput_basis"] is True
    assert r2["status"] == "decision_grade_caveated"
    assert "wallclock_fallback" in r2["basis_note"]
    # decision_grade describes the member CELLS and must NOT be re-graded.
    assert r2["decision_grade"] is True

    r4 = sns.evaluate_r4(mixed)[0]
    assert r4["mixed_throughput_basis"] is True
    assert r4["status"] == "decision_grade_caveated"
    assert r4["decision_grade"] is True

    # The secondary tasks/hour basis demotes too — it never promotes, only demotes.
    metric_mixed = [
        summary_row("C1", 8, 120.0, 20000.0),
        summary_row(
            "C1", 1, 40.0, 8000.0,
            cell_id="testmodel_q8_0-C1-np1-rawonly",
            tasks_per_hour_trimmed=0.0,
        ),
    ]
    assert {sns.aggregate_basis(r) for r in metric_mixed} == {"trimmed", "raw_fallback"}
    r2m = sns.evaluate_r2(metric_mixed)["models"]["testmodel_q8_0+Q8_0"]
    assert r2m["mixed_metric_basis"] is True
    assert r2m["status"] == "decision_grade_caveated"
    assert "ramp+drain" in r2m["basis_note"]


def test_r2_r4_keep_plain_decision_grade_on_a_consistent_basis():
    """The guard must not forbid its own compliant path.

    A caveat that fires on every verdict is indistinguishable from no verdict.
    Same-basis decision-grade rows must still come out plain `decision_grade`
    with no basis_note — this is the case all three banked E5 run dirs are in
    (measured 2026-08-11: 0 of 3 have a mixed basis among decision-grade cells,
    which is why landing this guard changes no banked result).
    """
    clean = [
        summary_row("C1", 8, 120.0, 20000.0),
        summary_row("C1", 1, 40.0, 8000.0),
    ]
    assert len({sns.throughput_basis(r) for r in clean}) == 1
    assert len({sns.aggregate_basis(r) for r in clean}) == 1

    r2 = sns.evaluate_r2(clean)["models"]["testmodel_q8_0+Q8_0"]
    assert r2["status"] == "decision_grade"
    assert r2["mixed_throughput_basis"] is False
    assert "basis_note" not in r2

    r4 = sns.evaluate_r4(clean)[0]
    assert r4["status"] == "decision_grade"
    assert "basis_note" not in r4


def test_basis_caveat_never_upgrades_a_refusal():
    """`insufficient_decision_grade_data` must survive the caveat pass.

    The helper keys on `status == "decision_grade"`; a refusal carries a
    different status and must be returned untouched rather than rewritten into
    a caveated *verdict*, which would read as weaker evidence than it is.
    """
    refusal = {"status": "insufficient_decision_grade_data", "mixed_throughput_basis": True}
    assert sns._apply_basis_caveat(refusal)["status"] == "insufficient_decision_grade_data"
    assert "basis_note" not in refusal


def test_summarizer_r2_r4_refuse_observation_only_cells() -> None:
    rows = [
        summary_row("C1", 1, 40.0, 10000.0, decision_grade=False),
        summary_row("C1", 8, 120.0, 50000.0, decision_grade=False),
        summary_row("C3", 4, 100.0, 20000.0, decision_grade=False),
    ]
    r2 = sns.evaluate_r2(rows)["models"]["testmodel_q8_0+Q8_0"]
    assert r2["status"] == "insufficient_decision_grade_data"
    assert r2["decision_grade"] is False
    assert "peak_cell" not in r2
    r4 = sns.evaluate_r4(rows)[0]
    assert r4["status"] == "insufficient_decision_grade_data"
    assert r4["decision_grade"] is False
    assert r4["recommended"] is None


def test_summarizer_r2_k1_proxy_baseline():
    # Review F5: no C1b@1 in the pre-registered grid — the same-shape C1@1
    # substitutes as the K=1 p95 baseline with the substitution recorded.
    rows = [
        summary_row("C1", 1, 30.0, 10000.0, cell_id="m-C1-np1"),
        summary_row("C1b", 4, 100.0, 25000.0, cell_id="m-C1b-np4"),
        summary_row("C1b", 8, 120.0, 45000.0, cell_id="m-C1b-np8"),
    ]
    rules = sns.evaluate_r2(rows)
    verdict = rules["models"]["testmodel_q8_0+Q8_0"]
    assert verdict["k1_baseline"]["cell_id"] == "m-C1-np1"
    assert verdict["k1_baseline"]["proxy"] is True
    assert "substituted" in verdict["k1_baseline"]["note"]
    assert verdict["sla_violated_at_peak"] is True  # 45000 > 3 x 10000
    assert verdict["holder_cell"] == "m-C1b-np4"  # 100 >= 70% of 120, within SLA
    assert verdict["lanes_real"] is True


def test_summarizer_r3_refuses_without_fresh_baseline():
    rows = r1_fixture_rows()
    refused = sns.evaluate_r3(rows, None)
    assert refused["status"] == "refused"
    assert "current-arm baseline" in refused["reason"]
    with tempfile.TemporaryDirectory() as tmp:
        # baseline without api_worker_count attestation (C10-F1) => refused
        no_workers = Path(tmp) / "baseline_no_workers.json"
        no_workers.write_text(
            json.dumps(
                {"wall_minutes_per_eval": 9.5, "items_per_eval": 50, "attestation": {}}
            )
        )
        refused = sns.evaluate_r3(rows, no_workers)
        assert refused["status"] == "refused"
        assert "api_worker_count" in refused["reason"]
        # baseline without an item count => refused (unit normalization, review
        # F4: core_v2 eval = 50 items vs E5 batch = 43 prompts, ~16% skew)
        no_items = Path(tmp) / "baseline_no_items.json"
        no_items.write_text(
            json.dumps(
                {"wall_minutes_per_eval": 9.5, "attestation": {"api_worker_count": 6}}
            )
        )
        refused = sns.evaluate_r3(rows, no_items)
        assert refused["status"] == "refused"
        assert "items_per_eval" in refused["reason"]
        # fresh, attested, item-counted baseline => priced per ITEM
        good = Path(tmp) / "baseline.json"
        good.write_text(
            json.dumps(
                {
                    "wall_minutes_per_eval": 9.5,
                    "items_per_eval": 50,
                    "attestation": {"api_worker_count": 6},
                    "era": {"cpu_kernel": "E6-cpu-kernel"},
                }
            )
        )
        priced = sns.evaluate_r3(rows, good)
        assert priced["status"] == "priced"
        assert priced["baseline"]["api_worker_count"] == 6
        assert priced["baseline"]["items_per_eval"] == 50
        first = priced["cells"][0]
        assert abs(first["wall_minutes_per_batch"] - 15.0) < 1e-9  # 900s / 60
        assert first["batch_items"] == 43
        # both sides normalized to wall-minutes per ITEM before the ratio
        expected = round((9.5 / 50) / ((900.0 / 60.0) / 43), 3)
        assert first["speedup_vs_current_arm"] == expected


def test_summarizer_r4_model_keyed_capability_rows():
    rows = [
        summary_row("C1", 1, 30.0, 20000.0),
        summary_row("C1", 4, 60.0, 22000.0),
        summary_row("C1b", 4, 120.0, 24000.0),
        summary_row("C3", 4, 115.0, 30000.0),
        summary_row("C3", 8, 120.0, 50000.0),  # peak but slowest
    ]
    r4 = sns.evaluate_r4(rows)
    assert len(r4) == 1
    entry = r4[0]
    assert entry["model_key"] == "testmodel_q8_0" and entry["quant"] == "Q8_0"
    # smallest-latency cell achieving >= 90% of peak (120*0.9=108): C1b@4 (p95 24000)
    assert entry["recommended"]["config_id"] == "C1b"
    assert entry["recommended"]["np"] == 4
    # the pick is made on the PRIMARY metric, from the primary key — if the
    # fixture ever stopped writing aggregate_decode_tps this would read
    # "wallclock_fallback" and fail instead of silently ranking on the fallback
    assert entry["metric"] == "aggregate_decode_tps"
    assert entry["recommended"]["throughput_basis"] == "decode"
    assert entry["recommended"]["aggregate_decode_tps"] == 120.0
    assert entry["recommended"]["peak_decode_tps"] == 120.0
    assert entry["per_shape_np_optimum"]["C3"]["aggregate_decode_tps"] == 120.0
    assert entry["per_shape_np_optimum"]["C3"]["np"] == 8
    assert entry["solo_shape"]["config_id"] == "C1"
    assert entry["numa_splitting_potential"]["C1b_over_C1_at_np4"] == 2.0
    assert entry["kv_unified_attestation"]["testmodel_q8_0-C3-np8"] is False
    # model-keyed, never role-keyed
    assert "role" not in json.dumps(r4)


def test_garbage_gate_demotes_cell():
    rows = [
        summary_row("C3", 4, 115.0, 30000.0),
        summary_row("C3", 8, 120.0, 50000.0),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        scores = Path(tmp) / "scores.jsonl"
        with scores.open("w", encoding="utf-8") as fh:
            for index in range(3):  # 3 parse failures > 2/43 threshold
                fh.write(
                    json.dumps(
                        {
                            "cell_id": "testmodel_q8_0-C3-np8",
                            "qid": f"q{index:03d}",
                            "parse_ok": False,
                        }
                    )
                    + "\n"
                )
            fh.write(
                json.dumps(
                    {"cell_id": "testmodel_q8_0-C3-np4", "qid": "q000", "parse_ok": True}
                )
                + "\n"
            )
        degraded = sns.apply_garbage_gate(rows, scores)
    assert degraded == ["testmodel_q8_0-C3-np8"]
    assert rows[1]["degraded"] is True
    assert "demoted to observation" in rows[1]["degraded_reason"]
    assert "degraded" not in rows[0]
    # degraded cells are excluded from decision picks
    r4 = sns.evaluate_r4(rows)
    assert r4[0]["recommended"]["config_id"] == "C3" and r4[0]["recommended"]["np"] == 4


def test_summarize_run_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        with (run_dir / "cells.jsonl").open("w", encoding="utf-8") as fh:
            for row in r1_fixture_rows():
                fh.write(json.dumps(row) + "\n")
        rc = sns.main(["--summarize-run", str(run_dir)])
        assert rc == 0
        rules = json.loads((run_dir / "rules.json").read_text())
        assert rules["R3"]["status"] == "refused"  # no fresh baseline supplied
        assert rules["R1"]["k_star_roofline_flip"] == 8
        assert sns.E1_COMPARABILITY_CAVEAT in rules["caveats"]
        assert sns.TRIM_BASIS_CAVEAT in rules["caveats"]  # raw-fallback basis caveat
        assert "scout_probes" in rules
        summary = (run_dir / "summary.md").read_text()
        assert "NOT byte-comparable to E1" in summary
        assert "R4 — model-keyed capability rows" in summary


# ---------------------------------------------------------------------------
# Stdlib runner (used when pytest is not installed)
# ---------------------------------------------------------------------------


def _run_all() -> int:
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS {name}")
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0




def test_ensure_clean_runtime_allowing():
    """Coexistence grant (2026-07-23): build-hip GPU bench servers matching the
    pattern do not gate; any non-matching survivor still refuses; no pattern =
    byte-identical passthrough."""
    import server_numa_np_sweep as mod

    calls = {"n": 0}

    def _clean_ok():
        calls["n"] += 1

    def _clean_hip_only():
        raise RuntimeError(
            "existing llama processes would contaminate P-BENCH-3:\n"
            "  111 /mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server -m x --port 18072"
        )

    def _clean_mixed():
        raise RuntimeError(
            "existing llama processes would contaminate P-BENCH-3:\n"
            "  111 /mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server --port 18072\n"
            "  222 /mnt/raid0/llm/llama.cpp/build/bin/llama-server --port 8080"
        )

    _orig = mod.ensure_clean_runtime
    try:
        mod.ensure_clean_runtime = _clean_ok
        assert mod.ensure_clean_runtime_allowing(None) == []
        assert mod.ensure_clean_runtime_allowing("build-hip") == []
        assert calls["n"] == 2

        mod.ensure_clean_runtime = _clean_hip_only
        allowed = mod.ensure_clean_runtime_allowing("build-hip")
        assert len(allowed) == 1 and "build-hip" in allowed[0]
        try:
            mod.ensure_clean_runtime_allowing(None)
            raise AssertionError("no-pattern must re-raise")
        except RuntimeError:
            pass

        mod.ensure_clean_runtime = _clean_mixed
        try:
            mod.ensure_clean_runtime_allowing("build-hip")
            raise AssertionError("non-matching survivor must re-raise")
        except RuntimeError:
            pass
    finally:
        mod.ensure_clean_runtime = _orig


if __name__ == "__main__":
    raise SystemExit(_run_all())


def test_reasoning_flag_emitted_from_manifest():
    # 2026-07-29: the harness had no --reasoning emit at all, so every cell ran
    # at llama-server's `--reasoning auto`. For gemma4 that default is ON and
    # the whole 256-token Stage-B budget went into the reasoning channel:
    # 41/43 response_capture_missing_answer_text on the W2 smoke, and the same
    # signature behind W0's 430/430.
    manifest = make_manifest(reasoning="off")
    with tempfile.TemporaryDirectory() as tmp:
        cell = load_cell(Path(tmp), manifest)
    cmd = sns.build_instance_command(
        binary=Path("/fake/llama-server"), cell=cell, inst=cell.instances[0],
    )
    assert "--reasoning" in cmd
    assert cmd[cmd.index("--reasoning") + 1] == "off"


def test_reasoning_is_stated_on_the_row_and_the_command_is_unchanged_for_auto():
    """A11: absence must stop carrying meaning, WITHOUT changing any launch.

    `reasoning` was present on 19 of 191 manifests. Absence meant the flag was
    never emitted, so the server fell to `--reasoning auto` — which for gemma4 is
    ON, against what both registries record. That gap cost 41/43 captures on the
    2026-07-29 W2 smoke and is the signature behind W0's 430/430.

    Manifests now always state it. The launched command must NOT change: passing
    `--reasoning auto` should equal omitting it, but should-equal is not
    verified-equal and this lane runs zero cells, so `auto` still omits the flag.
    """
    with tempfile.TemporaryDirectory() as tmp:
        auto_cell = load_cell(Path(tmp), make_manifest(reasoning="auto"))
        off_cell = load_cell(Path(tmp), make_manifest(reasoning="off"))
        bare_cell = load_cell(Path(tmp), make_manifest())

    # Resolution: a bare manifest and an explicit "auto" mean the same thing.
    assert sns.effective_reasoning(auto_cell) == "auto"
    assert sns.effective_reasoning(bare_cell) == "auto"
    assert sns.effective_reasoning(off_cell) == "off"

    def cmd_for(cell):
        return sns.build_instance_command(
            binary=Path("/fake/llama-server"), cell=cell, inst=cell.instances[0]
        )

    # Byte-identical for auto vs absent — no banked cell's recipe moves.
    assert cmd_for(auto_cell) == cmd_for(bare_cell)
    assert "--reasoning" not in cmd_for(auto_cell)
    # And a non-default value is still emitted, or the fix that closed the
    # capture failure would be silently reverted.
    off_cmd = cmd_for(off_cell)
    assert "--reasoning" in off_cmd
    assert off_cmd[off_cmd.index("--reasoning") + 1] == "off"


def test_reasoning_flag_absent_when_manifest_omits_it():
    # Back-compat: a manifest without the key keeps the previous command shape
    # exactly, so no already-run cell's recipe silently changes underneath it.
    with tempfile.TemporaryDirectory() as tmp:
        cell = load_cell(Path(tmp))
    cmd = sns.build_instance_command(
        binary=Path("/fake/llama-server"), cell=cell, inst=cell.instances[0],
    )
    assert "--reasoning" not in cmd
