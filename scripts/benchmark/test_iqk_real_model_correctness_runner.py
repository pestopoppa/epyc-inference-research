from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent))
import iqk_real_model_correctness_runner as runner


def test_version_build_commit_accepts_current_nine_character_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, ...]] = []

    def fake_git_value(*args: str) -> str:
        seen.append(args)
        return runner.EXPECTED_HEAD

    monkeypatch.setattr(runner, "git_value", fake_git_value)
    version = {"stdout": "version: 10101 (6c44557bf)\nbuilt with GNU 15.2.0 for Linux x86_64\n", "stderr": ""}
    assert runner.resolve_version_build_commit(version) == {
        "abbreviated": "6c44557bf",
        "resolved": runner.EXPECTED_HEAD,
    }
    assert seen == [("rev-parse", "--verify", "6c44557bf^{commit}")]


@pytest.mark.parametrize(
    ("version", "error"),
    [
        ({"stdout": "", "stderr": ""}, "does not contain a build commit"),
        ({"stdout": "version: 10101 (6C44557BF)\n", "stderr": ""}, "is malformed"),
        ({"stdout": "version: 10101 (123456)\n", "stderr": ""}, "is malformed"),
        ({"stdout": "version: 10101 (6c44557bf)\nversion: 10101 (6c44557bf)\n", "stderr": ""}, "ambiguous"),
        ({"stdout": "version: 10101 (6c44557bf)\nversion: unknown\n", "stderr": ""}, "ambiguous"),
    ],
)
def test_version_build_commit_rejects_missing_malformed_or_ambiguous_witness(
    version: dict[str, str],
    error: str,
) -> None:
    with pytest.raises(runner.GateFailure, match=error):
        runner.resolve_version_build_commit(version)


def test_version_build_commit_rejects_wrong_resolved_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "git_value", lambda *_args: "a" * 40)
    with pytest.raises(runner.GateFailure, match="does not resolve to the expected HEAD"):
        runner.resolve_version_build_commit({"stdout": "version: 10101 (6c44557bf)\n", "stderr": ""})


def test_plan_is_fixed_to_three_cpu_models_and_six_fresh_arms() -> None:
    plan = runner.plan()
    cells = plan["cells"]
    assert len(cells) == 6
    assert {(cell["model"], cell["iqk"]) for cell in cells} == {
        (model.name, iqk) for model in runner.MODELS for iqk in (0, 1)
    }
    assert "qwen3.5-122B" in plan["excluded"]
    assert "Laguna UD-IQ2_M CPU" in plan["excluded"]
    assert plan["fixed_cpu_recipe"]["cpuset"] == "0-95"
    assert plan["fixed_cpu_recipe"]["numa"] == "interleave=all"


def test_child_environment_is_exact_allowlist_and_scrubs_hostile_inheritance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for key in ("LD_PRELOAD", "HSA_VISIBLE_DEVICES", "GGML_IQK", "KMP_BLOCKTIME"):
        monkeypatch.setenv(key, "hostile")
    assert runner.child_env(1) == {**runner.BASE_ENV, "GGML_IQK": "1"}
    assert runner.child_env(0) == {**runner.BASE_ENV, "GGML_IQK": "0"}
    assert runner.child_env(1)["LD_LIBRARY_PATH"] == f"{runner.BINARY.parent}:{runner.LLVM20_LIBDIR}"
    assert runner.child_env(1)["KMP_BLOCKTIME"] == "10"

    proc = tmp_path / "77"
    proc.mkdir()
    exact = b"\0".join(f"{key}={value}".encode() for key, value in runner.child_env(1).items()) + b"\0"
    (proc / "environ").write_bytes(exact)
    assert runner.exact_process_env(77, 1, proc_root=tmp_path) == runner.child_env(1)
    (proc / "environ").write_bytes(exact + b"LD_PRELOAD=/tmp/hostile.so\0")
    with pytest.raises(runner.GateFailure, match="exact sanitized allowlist"):
        runner.exact_process_env(77, 1, proc_root=tmp_path)


def test_run_capture_never_inherits_parent_and_records_exact_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LD_PRELOAD", "/tmp/hostile.so")
    monkeypatch.setenv("GGML_IQK", "99")
    capture = runner.run_capture(["env"])
    assert capture["ok"] is True
    assert capture["environment"] == runner.child_env(0)
    assert "LD_PRELOAD=" not in capture["stdout"]
    assert "GGML_IQK=0" in capture["stdout"]
    with pytest.raises(runner.GateFailure, match="exact allowlist"):
        runner.run_capture(["env"], env={"PATH": "/usr/bin"})


def test_cleanup_pgrep_uses_sanitized_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_capture(argv: list[str], **kwargs: object) -> dict[str, object]:
        seen["argv"] = argv
        seen["env"] = kwargs["env"]
        return {"argv": argv, "environment": kwargs["env"], "returncode": 1, "stdout": "", "stderr": "", "ok": False}

    monkeypatch.setattr(runner, "run_capture", fake_capture)
    result = runner.cleanup_pgrep()
    assert result["returncode"] == 1
    assert seen == {"argv": ["pgrep", "-x", "llama-server"], "env": runner.child_env(0)}


def test_server_recipe_is_fixed_cpu_only_and_reasoning_off() -> None:
    argv = runner.server_argv(runner.MODELS[0], 1, 19000)
    assert argv[:6] == ["taskset", "-c", "0-95", "numactl", "--interleave=all", str(runner.BINARY)]
    for flag, value in (("-t", "96"), ("-tb", "96"), ("-dev", "none"), ("-ngl", "0"), ("--reasoning", "off"), ("--reasoning-budget", "0")):
        assert argv[argv.index(flag) + 1] == value
    assert "--no-op-offload" in argv
    assert "--no-mmap" in argv
    assert "--mmap" not in argv
    with pytest.raises(runner.GateFailure, match="--no-mmap"):
        runner.validate_cpu_argv([*argv, "--mmap"])


def test_local_library_provenance_requires_exact_complete_filename_sha_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    libraries = {}
    for name, body in (("libllama.so.0.0.10101", b"llama"), ("libggml.so.0.16.0", b"ggml")):
        path = tmp_path / name
        path.write_bytes(body)
        libraries[name] = runner.stable_file_identity(path)["sha256"]
    openmp = tmp_path / "llvm-20/lib/libomp.so.5"
    openmp.parent.mkdir(parents=True)
    openmp.write_bytes(b"openmp")
    ldd = "\n".join([*(f"{name} => {tmp_path / name} (0x0)" for name in libraries), f"libomp.so.5 => {openmp} (0x0)"])
    monkeypatch.setattr(runner, "EXPECTED_LOCAL_LIBRARY_SHA256", libraries)
    monkeypatch.setattr(runner, "LLVM20_LIBDIR", openmp.parent)
    monkeypatch.setattr(runner, "OPENMP_RUNTIME", openmp)
    monkeypatch.setattr(runner, "EXPECTED_OPENMP_RUNTIME_SHA256", runner.stable_file_identity(openmp)["sha256"])
    monkeypatch.setattr(runner, "run_capture", lambda *_args, **_kwargs: {"ok": True, "stdout": ldd, "stderr": "", "returncode": 0})
    identity = runner.local_library_identities(tmp_path / "llama-server")
    assert identity["filename_sha256"] == libraries
    monkeypatch.setattr(runner, "EXPECTED_LOCAL_LIBRARY_SHA256", {"unexpected.so": "0"})
    with pytest.raises(runner.GateFailure, match="filename/SHA set"):
        runner.local_library_identities(tmp_path / "llama-server")


def test_ldd_evidence_normalization_removes_only_aslr_addresses() -> None:
    capture = {"argv": ["ldd", "/tmp/server"], "environment": {"PATH": "/usr/bin"}, "returncode": 0,
               "stderr": "", "ok": True,
               "stdout": "\tlinux-vdso.so.1 (0x7abcde1000)\nlibfoo.so.1 => /tmp/libfoo.so.1 (0x7abcde2000)\n/lib64/ld-linux-x86-64.so.2 (0x7abcde3000)"}
    alternate = {**capture, "stdout": capture["stdout"].replace("7abcde", "7fedcb")}
    normalized = runner.normalize_ldd_evidence(capture)
    assert normalized == runner.normalize_ldd_evidence(alternate)
    assert normalized["argv"] == capture["argv"]
    assert normalized["environment"] == capture["environment"]
    assert normalized["stdout"] == "linux-vdso.so.1\nlibfoo.so.1 => /tmp/libfoo.so.1\n/lib64/ld-linux-x86-64.so.2"
    different_path = {**capture, "stdout": capture["stdout"].replace("/tmp/libfoo.so.1", "/tmp/other.so.1")}
    different_soname = {**capture, "stdout": capture["stdout"].replace("libfoo.so.1 =>", "libbar.so.1 =>")}
    assert normalized != runner.normalize_ldd_evidence(different_path)
    assert normalized != runner.normalize_ldd_evidence(different_soname)


@pytest.mark.parametrize("stdout", ["libfoo.so.1 => /tmp/libfoo.so.1", "libfoo.so.1 => not found (0x0)", "noise (0x123)"])
def test_ldd_evidence_normalization_rejects_malformed_or_unrecognized_lines(stdout: str) -> None:
    with pytest.raises(runner.GateFailure, match="malformed or unrecognized"):
        runner.normalize_ldd_evidence({"stdout": stdout})


def test_mapped_openmp_runtime_requires_pinned_runtime(tmp_path: Path) -> None:
    runtime = tmp_path / "llvm-20/lib/libomp.so.5"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"openmp")
    proc = tmp_path / "77"
    proc.mkdir()
    (proc / "maps").write_text(f"1000-2000 r-xp 00000000 00:00 1 {runtime}\n")
    identity = runner.stable_file_identity(runtime)
    assert runner.mapped_openmp_runtime(77, identity, tmp_path)["path"] == str(runtime.resolve())
    (proc / "maps").write_text("")
    with pytest.raises(runner.GateFailure, match="pinned LLVM 20"):
        runner.mapped_openmp_runtime(77, identity, tmp_path)


@pytest.mark.parametrize(
    ("task", "content"),
    [
        ("exact_json", '{"status":"ok","model":"hy3"}'),
        ("math_37_plus_58", "95"),
        ("needle", "IQK-DELTA-9421"),
        ("routing_tradeoffs", "A mixture-of-experts router balances compute cost against bandwidth pressure. Good load balancing avoids concentrating experts, while routing overhead grows when selection, dispatch, and synchronization become complex. Limiting candidate experts can reduce compute and bandwidth use, but risks weaker specialization. The design therefore weighs routing quality against overhead and keeps load balancing visible in admission decisions."),
    ],
)
def test_semantic_contract_accepts_each_deterministic_task(task: str, content: str) -> None:
    result = runner.validate_semantics(task, content, runner.MODELS[0])
    assert result["status"] == "pass"


def test_routing_semantic_contract_counts_hyphenated_compounds_as_words() -> None:
    content = (
        "In a mixture-of-experts inference router, increasing compute cost by deploying more experts "
        "improves accuracy but strains resource budgets. Higher bandwidth demands arise when routing "
        "requires frequent communication between experts and clients, especially with dynamic load "
        "balancing that redistributes queries to prevent bottlenecks. Overly aggressive load balancing "
        "increases routing overhead due to constant re-evaluation of expert suitability, slowing "
        "inference. Conversely, minimizing routing overhead by using static routing reduces adaptability, "
        "leading to underutilized experts and skewed loads. Balancing these four—compute cost, bandwidth, "
        "load balancing, and routing overhead—is critical: optimizing one often degrades another, "
        "demanding a system-wide tradeoff that prioritizes latency, throughput, and efficiency based on "
        "deployment constraints."
    )
    assert len(content.split()) == 105
    assert len(re.findall(r"[A-Za-z]+", content)) == 111
    assert len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", content)) == 107
    assert runner.validate_semantics("routing_tradeoffs", content, runner.MODELS[0])["status"] == "pass"


def test_routing_semantic_contract_rejects_punctuation_padding() -> None:
    content = "compute cost bandwidth load balancing routing overhead " + " ".join(["---"] * 50)
    with pytest.raises(runner.GateFailure, match="out of bounds"):
        runner.validate_semantics("routing_tradeoffs", content, runner.MODELS[0])


def test_routing_semantic_contract_rejects_more_than_110_lexical_words() -> None:
    content = "compute cost bandwidth load balancing routing overhead " + " ".join(["word"] * 104)
    assert len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", content)) == 111
    with pytest.raises(runner.GateFailure, match="out of bounds"):
        runner.validate_semantics("routing_tradeoffs", content, runner.MODELS[0])


@pytest.mark.parametrize(
    ("task", "content"),
    [
        ("exact_json", '{"status":"bad"}'),
        ("math_37_plus_58", "94"),
        ("needle", "IQK-DELTA-9421 IQK-DELTA-9421"),
        ("routing_tradeoffs", "compute bandwidth routing"),
    ],
)
def test_semantic_contract_rejects_wrong_or_incoherent_output(task: str, content: str) -> None:
    with pytest.raises(runner.GateFailure):
        runner.validate_semantics(task, content, runner.MODELS[0])


def test_active_log_proves_new_native_iq_path_and_control_has_none(tmp_path: Path) -> None:
    log = tmp_path / "server.stderr"
    model = runner.MODELS[0]
    log.write_text(
        "[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=21 activation=8 ne00=2048)\n"
        "[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=22 activation=8 ne00=2048)\n"
    )
    evidence = runner.active_log_evidence(log, model, 1)
    assert evidence["native_type_codes"] == [21, 22]
    log.write_text("")
    assert runner.active_log_evidence(log, model, 0)["active_type_codes"] == []
    log.write_text("[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=12 activation=8 ne00=2048)\n")
    with pytest.raises(runner.GateFailure, match="required native IQ types"):
        runner.active_log_evidence(log, model, 1)
    with pytest.raises(runner.GateFailure, match="GGML_IQK=0"):
        runner.active_log_evidence(log, model, 0)


def test_active_log_allows_only_the_exact_benign_eog_logit_bias_inf_line(tmp_path: Path) -> None:
    log = tmp_path / "server.stderr"
    model = runner.MODELS[0]
    log.write_text("0.03.347.059 I cmn  common_init_: added 〈|EOS|〉 logit bias = -inf\n")
    assert runner.active_log_evidence(log, model, 0)["active_type_codes"] == []
    log.write_text("0.03.347.059 I cmn  common_init_: added </assistant> logit bias = -inf\n")
    assert runner.active_log_evidence(log, model, 0)["active_type_codes"] == []
    for invalid in (
        "0.03.347.059 I cmn  common_init_: added 〈|EOS|〉 logit bias = inf",
        "0.03.347.059 I cmn  common_init_: added 〈|EOS|〉 logit bias = -inf extra",
        "0.03.347.059 I cmn  other_init_: added 〈|EOS|〉 logit bias = -inf",
        "0.03.347.059 I other  common_init_: added 〈|EOS|〉 logit bias = -inf",
        "runtime tensor value = -inf",
    ):
        log.write_text(invalid + "\n")
        with pytest.raises(runner.GateFailure, match="fatal/nonfinite"):
            runner.active_log_evidence(log, model, 0)


def valid_meminfo() -> str:
    fields = {
        "MemTotal": "1000 kB", "MemFree": "100 kB", "MemAvailable": "200 kB", "Buffers": "10 kB", "Cached": "20 kB",
        "AnonHugePages": "0 kB", "ShmemHugePages": "0 kB", "ShmemPmdMapped": "0 kB", "FileHugePages": "0 kB", "FilePmdMapped": "0 kB",
        "HugePages_Total": "2", "HugePages_Free": "1", "HugePages_Rsvd": "0", "HugePages_Surp": "0",
        "Hugepagesize": "2048 kB", "Hugetlb": "4096 kB", "DirectMap2M": "0 kB",
    }
    return "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n"


def valid_host_snapshot() -> dict[str, object]:
    memory = runner.parse_memory_state(valid_meminfo())
    return {
        "uptime_seconds": 1.0,
        "governors": {"cpu0": "performance"},
        "thp_enabled": {"raw": "[always] madvise never"},
        "thp_defrag": {"raw": "[always] defer never"},
        "numa_balancing": "0",
        **memory,
        "llama_ownership": [],
        "autopilot_processes": [],
        "kfd_ownership": {"users": [], "unreadable_processes": [], "lsof_fallback": None},
    }


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("uptime_seconds", runner.MAX_UPTIME_SECONDS + 1, "uptime"),
        ("governors", {"cpu0": "powersave"}, "governor"),
        ("thp_enabled", {"raw": "always [madvise] never"}, "THP enabled"),
        ("thp_defrag", {"raw": "always [defer] never"}, "THP defrag"),
        ("numa_balancing", "1", "numa_balancing"),
        ("autopilot_processes", [{"pid": 9}], "autopilot"),
    ],
)
def test_host_state_fails_closed_on_required_cpu_gate_fields(key: str, value: object, error: str) -> None:
    snapshot = valid_host_snapshot()
    snapshot[key] = value
    with pytest.raises(runner.GateFailure, match=error):
        runner.require_host_state(snapshot, expected_llama_pid=None)


def test_memory_contract_requires_all_fields_units_and_consistency() -> None:
    parsed = runner.parse_memory_state(valid_meminfo())
    assert parsed["memory_kib"]["MemTotal"] == 1000
    assert parsed["thp_meminfo"]["HugePages_Total"] == {"value": 2, "unit": "count"}
    assert parsed["thp_meminfo"]["Hugepagesize"] == {"value": 2048, "unit": "kB"}
    with pytest.raises(runner.GateFailure, match="missing required"):
        runner.parse_memory_state(valid_meminfo().replace("FilePmdMapped: 0 kB\n", ""))
    with pytest.raises(runner.GateFailure, match="unexpected meminfo unit"):
        runner.parse_memory_state(valid_meminfo().replace("HugePages_Total: 2", "HugePages_Total: 2 kB"))
    with pytest.raises(runner.GateFailure, match="exceeds"):
        runner.parse_memory_state(valid_meminfo().replace("HugePages_Free: 1", "HugePages_Free: 3"))
    snapshot = valid_host_snapshot()
    snapshot["thp_meminfo"] = {}
    with pytest.raises(runner.GateFailure, match="does not match"):
        runner.require_host_state(snapshot, expected_llama_pid=None)


def test_numastat_requires_positive_total_and_matching_node_sum() -> None:
    runner.validate_numastat_totals([10.0, 20.0, 30.0], [0, 1])
    with pytest.raises(runner.GateFailure, match="sum"):
        runner.validate_numastat_totals([10.0, 20.0, 31.0], [0, 1])
    with pytest.raises(runner.GateFailure, match="positive"):
        runner.validate_numastat_totals([10.0, 0.0, 10.0], [0, 1])


def write_thread_status(proc_root: Path, pid: int, tid: int, cpus_allowed_list: str) -> None:
    status = proc_root / str(pid) / "task" / str(tid) / "status"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(f"Name:\tllama-server\nCpus_allowed_list:\t{cpus_allowed_list}\n")


def test_thread_affinity_accepts_split_team_with_leader_pinned_to_cpu_zero(tmp_path: Path) -> None:
    write_thread_status(tmp_path, 77, 77, "0")
    write_thread_status(tmp_path, 77, 78, "1-95")
    evidence = runner.thread_affinity_evidence(77, proc_root=tmp_path)
    assert evidence["union_cpus"] == list(range(96))
    assert evidence["threads"] == [
        {"tid": 77, "cpus_allowed_list": "0", "cpus": [0]},
        {"tid": 78, "cpus_allowed_list": "1-95", "cpus": list(range(1, 96))},
    ]


@pytest.mark.parametrize(
    ("cpus_allowed_list", "error"),
    [
        ("0,96", "escapes required CPU set"),
        ("0-94", "union does not exactly cover"),
        ("0-foo", "malformed Cpus_allowed_list"),
    ],
)
def test_thread_affinity_rejects_outside_incomplete_or_malformed_masks(
    tmp_path: Path,
    cpus_allowed_list: str,
    error: str,
) -> None:
    write_thread_status(tmp_path, 77, 77, cpus_allowed_list)
    with pytest.raises(runner.GateFailure, match=error):
        runner.thread_affinity_evidence(77, proc_root=tmp_path)


def test_thread_affinity_fails_closed_on_unreadable_thread_status(tmp_path: Path) -> None:
    status = tmp_path / "77/task/77/status"
    status.mkdir(parents=True)
    with pytest.raises(runner.GateFailure, match="cannot read child thread status"):
        runner.thread_affinity_evidence(77, proc_root=tmp_path)


def test_thread_affinity_retries_bounded_thread_list_churn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    write_thread_status(tmp_path, 77, 77, "0")
    write_thread_status(tmp_path, 77, 78, "1-95")
    task_dir = tmp_path / "77/task"
    original_iterdir = Path.iterdir
    calls = 0

    def churn_once(path: Path):
        nonlocal calls
        if path == task_dir:
            calls += 1
            if calls == 2:
                return iter([task_dir / "77"])
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", churn_once)
    evidence = runner.thread_affinity_evidence(77, proc_root=tmp_path, max_attempts=2)
    assert evidence["attempt"] == 2


def test_split_identity_hashes_every_shard_and_rejects_missing_shard(tmp_path: Path) -> None:
    first = tmp_path / "model-00001-of-00003.gguf"
    for index, body in enumerate((b"a", b"bb", b"ccc"), 1):
        (tmp_path / f"model-{index:05d}-of-00003.gguf").write_bytes(body)
    model = runner.Model("test", first)
    identity = runner.model_identity(model)
    assert identity["shard_count"] == 3
    assert identity["total_bytes"] == 6
    (tmp_path / "model-00003-of-00003.gguf").unlink()
    with pytest.raises(runner.GateFailure, match="incomplete"):
        runner.discover_shards(first)


def test_response_validation_rejects_nonfinite_or_reasoning_content() -> None:
    valid = {"choices": [{"message": {"content": "95"}, "logprobs": {"content": [{"token": "95", "bytes": [57, 53], "logprob": -0.1}]}}], "timings": {"prompt_n": 1, "predicted_n": 1, "prompt_ms": 1, "predicted_ms": 1}, "usage": {"prompt_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}, "metadata": {"ratio": 0.5}, "completion_tokens": 1}}
    content, telemetry, _logprobs = runner.content_from_response(valid)
    assert content == "95"
    assert telemetry["counters"] == {"prompt_tokens": 1, "prompt_tokens_details.cached_tokens": 0, "metadata.ratio": 0.5, "completion_tokens": 1}
    round_tripped = json.loads(json.dumps(telemetry))
    assert type(round_tripped["timings"]["prompt_n"]) is int
    assert type(round_tripped["timings"]["predicted_n"]) is int
    assert type(round_tripped["counters"]["completion_tokens"]) is int
    assert type(round_tripped["counters"]["prompt_tokens_details.cached_tokens"]) is int
    assert type(round_tripped["counters"]["metadata.ratio"]) is float
    invalid = json.loads(json.dumps(valid))
    invalid["timings"]["prompt_ms"] = float("inf")
    with pytest.raises(runner.GateFailure, match="nonfinite"):
        runner.content_from_response(invalid)
    invalid = json.loads(json.dumps(valid))
    invalid["choices"][0]["message"]["reasoning_content"] = "hidden chain"
    with pytest.raises(runner.GateFailure, match="reasoning-off"):
        runner.content_from_response(invalid)
    invalid = json.loads(json.dumps(valid))
    invalid["choices"][0]["logprobs"]["content"][0]["logprob"] = float("nan")
    with pytest.raises(runner.GateFailure, match="nonfinite"):
        runner.content_from_response(invalid)
    invalid = json.loads(json.dumps(valid))
    invalid["choices"][0]["logprobs"]["content"] = []
    with pytest.raises(runner.GateFailure, match="nonempty"):
        runner.content_from_response(invalid)
    invalid = json.loads(json.dumps(valid))
    invalid["choices"][0]["logprobs"]["content"].append({"token": "!", "bytes": [33], "logprob": -0.2})
    with pytest.raises(runner.GateFailure, match="token count"):
        runner.content_from_response(invalid)
    invalid = json.loads(json.dumps(valid))
    invalid["usage"]["completion_tokens"] = 2
    with pytest.raises(runner.GateFailure, match="token count"):
        runner.content_from_response(invalid)
    invalid = json.loads(json.dumps(valid))
    invalid["timings"]["predicted_n"] = 1.0
    with pytest.raises(runner.GateFailure, match="native integral"):
        runner.content_from_response(invalid)
    invalid = json.loads(json.dumps(valid))
    invalid["usage"]["completion_tokens"] = True
    with pytest.raises(runner.GateFailure, match="native integral"):
        runner.content_from_response(invalid)


def test_logprob_evidence_accepts_one_final_empty_eog_token() -> None:
    evidence = runner.logprob_evidence({"logprobs": {"content": [
        {"token": "95", "bytes": [57, 53], "logprob": -0.1},
        {"token": "", "bytes": [], "logprob": -0.2},
    ]}})
    assert evidence["token_count"] == 2
    assert evidence["terminal_eog_empty"] is True
    assert evidence["tokens"][-1] == {
        "token_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "bytes": [], "terminal_eog_empty": True, "logprob": -0.2,
    }


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ([{"token": "x", "logprob": 0.0}], "bytes is not a JSON byte list"),
        ([{"token": "x", "bytes": [True], "logprob": 0.0}], "bytes is not a JSON byte list"),
        ([{"token": "x", "bytes": [-1], "logprob": 0.0}], "bytes is not a JSON byte list"),
        ([{"token": "x", "bytes": [121], "logprob": 0.0}], "does not match token UTF-8"),
        ([{"token": "", "bytes": [], "logprob": 0.0}, {"token": "x", "bytes": [120], "logprob": 0.0}], "exactly one terminal empty"),
        ([{"token": "x", "bytes": [120], "logprob": 0.0}, {"token": "", "bytes": [], "logprob": 0.0}, {"token": "", "bytes": [], "logprob": 0.0}], "exactly one terminal empty"),
        ([{"token": "", "bytes": [0], "logprob": 0.0}], "exactly one terminal empty"),
    ],
)
def test_logprob_evidence_rejects_invalid_token_bytes_or_empty_eog(
    content: list[dict[str, object]],
    error: str,
) -> None:
    with pytest.raises(runner.GateFailure, match=error):
        runner.logprob_evidence({"logprobs": {"content": content}})


@pytest.mark.parametrize(
    ("usage", "error"),
    [
        ({"completion_tokens": 1, "nested": {"value": float("inf")}}, "nonfinite"),
        ({"completion_tokens": 1, "nested": {"value": True}}, "strict native JSON number"),
        ({"completion_tokens": 1, "nested": {"value": None}}, "strict native JSON number"),
        ({"completion_tokens": 1, "nested": {"value": "one"}}, "strict native JSON number"),
        ({"completion_tokens": 1, "nested": {"value": []}}, "strict native JSON number"),
        ({"completion_tokens": 1, "nested": {}}, "object is empty"),
        ({"completion_tokens": 1, "nested.value": 0}, "key is ambiguous"),
    ],
)
def test_usage_counter_flattening_fails_closed(usage: dict[str, object], error: str) -> None:
    with pytest.raises(runner.GateFailure, match=error):
        runner.flatten_usage_counters(usage)


def valid_arm(model: runner.Model, iqk: int) -> dict[str, object]:
    evidence: dict[str, object] = {"status": "pass", "active_type_codes": []}
    if iqk == 1:
        required = sorted(runner.EXPECTED_NATIVE_TYPES_BY_MODEL[model.name])
        evidence = {"status": "pass", "active_type_codes": required, "native_type_codes": required}
    return {"status": "pass", "model": model.name, "iqk": iqk, "cleanup": {"status": "pass"}, "iqk_log_evidence": evidence,
        "numerical_safety": {"status": "pass", "scope": "real-model completion token logprobs and server stderr only"}, "rows": [
        {"semantic": {"status": "pass"}, "logprobs": {"status": "pass", "token_count": 1}, "telemetry": {"timings": {"prompt_n": 1.0, "predicted_n": 1.0, "prompt_ms": 1.0, "predicted_ms": 1.0}, "counters": {"prompt_tokens": 1}}}
        for _ in range(4)
    ]}


def test_summary_requires_exact_six_arm_matrix_and_all_attestation_roles() -> None:
    rows = [valid_arm(model, iqk) for model in runner.MODELS for iqk in (0, 1)]
    summary = runner.summarize(rows, {"identity": "bound"})
    assert summary["status"] == "pass"
    assert summary["attestation_roles"] == {"correctness": True, "coherence": True, "numerical_safety": True}
    assert summary["decision_gate"] == {
        "handoff": "iqk-iquant-enablement B2",
        "b2_gate_passed": True,
        "promotion_decision": False,
        "semantic_contract": "IQK arms are not bit-exact; both independently satisfy fixed tasks",
        "timings": "non-decision observational only",
    }
    summary = runner.summarize(rows[:-1], {"identity": "bound"})
    assert summary["status"] == "fail"
    assert summary["decision_gate"]["promotion_decision"] is False


def test_summary_rejects_missing_or_invalid_embedded_iqk_log_evidence() -> None:
    rows = [valid_arm(model, iqk) for model in runner.MODELS for iqk in (0, 1)]
    rows[0]["iqk_log_evidence"] = {"status": "pass", "active_type_codes": [16]}
    assert runner.summarize(rows, {"identity": "bound"})["status"] == "fail"


def test_server_stderr_rejects_fatal_or_nonfinite_diagnostics(tmp_path: Path) -> None:
    log = tmp_path / "server.stderr"
    log.write_text("fatal error: non-finite tensor value\n")
    with pytest.raises(runner.GateFailure, match="fatal/nonfinite"):
        runner.active_log_evidence(log, runner.MODELS[0], 0)
    rows = [valid_arm(model, iqk) for model in runner.MODELS for iqk in (0, 1)]
    rows[1]["iqk_log_evidence"] = {"status": "pass", "active_type_codes": [16], "native_type_codes": []}
    assert runner.summarize(rows, {"identity": "bound"})["status"] == "fail"
