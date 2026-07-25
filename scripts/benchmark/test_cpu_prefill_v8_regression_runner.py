from __future__ import annotations

import errno
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import cpu_prefill_v8_regression_runner as runner


class Result:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def valid_meminfo_raw() -> str:
    return "\n".join(
        (
            "MemTotal: 100 kB",
            "MemFree: 40 kB",
            "MemAvailable: 50 kB",
            "Buffers: 2 kB",
            "Cached: 10 kB",
            "AnonHugePages: 0 kB",
            "ShmemHugePages: 0 kB",
            "ShmemPmdMapped: 0 kB",
            "FileHugePages: 0 kB",
            "FilePmdMapped: 0 kB",
            "HugePages_Total: 0",
            "HugePages_Free: 0",
            "HugePages_Rsvd: 0",
            "HugePages_Surp: 0",
            "Hugepagesize: 2048 kB",
            "Hugetlb: 0 kB",
            "DirectMap2M: 80 kB",
            "DirectMap1G: 0 kB",
        )
    )


def valid_thp_hugepage_state() -> dict[str, object]:
    raw = valid_meminfo_raw()
    return {
        "meminfo_raw": raw,
        "meminfo_fields": runner.parse_meminfo_evidence(raw)["fields"],
        "hpage_pmd_size_bytes": 2 * 1024 * 1024,
        "pools": [
            {
                "path": "/sys/kernel/mm/hugepages/hugepages-1048576kB",
                "page_size_kib": 1048576,
                "nr_hugepages": 0,
                "free_hugepages": 0,
                "resv_hugepages": 0,
                "surplus_hugepages": 0,
            },
            {
                "path": "/sys/kernel/mm/hugepages/hugepages-2048kB",
                "page_size_kib": 2048,
                "nr_hugepages": 0,
                "free_hugepages": 0,
                "resv_hugepages": 0,
                "surplus_hugepages": 0,
            },
        ],
    }


def valid_host() -> dict[str, object]:
    return {
        "uptime_seconds": 10.0,
        "governors": {"cpu0": "performance"},
        "thp_enabled": {"raw": "[always] madvise never"},
        "thp_defrag": {"raw": "[always] defer never"},
        "numa_balancing": "0",
        "memory_kib": {
            "MemTotal": 100,
            "MemFree": 40,
            "MemAvailable": 50,
            "Buffers": 2,
            "Cached": 10,
        },
        "thp_hugepage_state": valid_thp_hugepage_state(),
        "process_ownership": {
            "exact_llama_processes": [],
            "autopilot_processes": [],
            "unreadable_processes": [],
            "unresolved_ownership": [],
            "uncertain_relevant_processes": [],
        },
        "kfd_ownership": {
            "users": [],
            "unreadable_processes": [],
            "lsof_fallback": None,
        },
        "rocm_ownership": {"returncode": 0, "owners": []},
    }


def result_json(
    samples: list[object],
    *,
    metric: str = "pp2048",
    commit: str = "67a433bf4",
) -> str:
    samples_ns = [1_000_000_000] * len(samples)
    return json.dumps(
        [
            {
                "n_prompt": 2048 if metric == "pp2048" else 0,
                "n_gen": 0 if metric == "pp2048" else 128,
                "build_commit": commit,
                "build_number": 10107,
                "samples_ts": samples,
                "samples_ns": samples_ns,
            }
        ]
    )


def wrapper_stderr(
    *,
    library_path: str = "/candidate",
    iqk: int = 1,
    q8_0: bool = False,
    build: str = "build: 67a433bf4 (10107)",
) -> str:
    emitted = {
        "LD_LIBRARY_PATH": f"{library_path}:{runner.LLVM20_LIBDIR}",
        **runner.REQUIRED_WRAPPER_OMP_ENV,
        "GGML_IQK": str(iqk),
    }
    if q8_0:
        emitted["GGML_IQK_Q8_0"] = "1"
    assignments = " ".join(f"{key}={value}" for key, value in emitted.items())
    return f"Env:       {assignments}\n{build}\n"


def make_attestations(tmp_path: Path) -> dict[str, Path]:
    artifacts = {}
    for role in ("host", "correctness", "coherence", "numerical_safety"):
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps({"role": role}))
        artifacts[role] = path
    return artifacts


def synthetic_model_identities() -> dict[str, dict[str, object]]:
    identities: dict[str, dict[str, object]] = {}
    for index, current_name in enumerate(runner.IQK_MODEL_BINDINGS.values(), 1):
        path = f"/models/{current_name}.gguf"
        identities[current_name] = {
            "entry_path": path,
            "shard_count": 1,
            "total_bytes": index,
            "shards": [{
                "path": path,
                "device": 1,
                "inode": index,
                "bytes": index,
                "mtime_ns": 100 + index,
                "sha256": f"{index:064x}",
            }],
        }
    return identities


def valid_iqk_attestation(
    current_models: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    current = synthetic_model_identities() if current_models is None else current_models
    attested_models = {
        attested_name: {
            "name": attested_name,
            **current[current_name],
        }
        for attested_name, current_name in runner.IQK_MODEL_BINDINGS.items()
    }
    identity = {
        "candidate": {
            "branch": runner.CANDIDATE_BRANCH,
            "head": runner.CANDIDATE_HEAD,
            "binary": {"sha256": runner.IQK_SERVER_SHA256},
            "local_libraries": {
                "filename_sha256": runner.IQK_SERVER_LIBRARY_SHA256,
                "openmp_runtime": {
                    "sha256": runner.LLVM20_OPENMP_IDENTITY["sha256"],
                },
            },
        },
        "runner": {"sha256": runner.IQK_RUNNER_SHA256},
        "models": attested_models,
    }
    arms = []
    for name in runner.IQK_ATTESTATION_MODELS:
        for iqk in (0, 1):
            task_rows = [
                {
                    "task": task,
                    "semantic": {"task": task, "status": "pass"},
                    "logprobs": {
                        "status": "pass",
                        "token_count": 1,
                        "tokens": [{
                            "token_sha256": "a" * 64,
                            "logprob": -0.1,
                        }],
                    },
                    "telemetry": {
                        "timings": {
                            "prompt_n": 1,
                            "predicted_n": 1,
                            "prompt_ms": 1.0,
                            "predicted_ms": 1.0,
                        },
                        "counters": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                        },
                    },
                }
                for task in runner.IQK_TASKS
            ]
            log_evidence: dict[str, object] = {
                "status": "pass",
                "iqk": iqk,
                "active_type_codes": [],
            }
            if iqk == 1:
                required = sorted(runner.EXPECTED_NATIVE_TYPES_BY_MODEL[name])
                log_evidence["active_type_codes"] = required
                log_evidence["native_type_codes"] = required
            arms.append({
                "model": name,
                "iqk": iqk,
                "status": "pass",
                "primary_error": None,
                "cleanup": {"status": "pass"},
                "numerical_safety": {
                    "status": "pass",
                    "scope": runner.IQK_NUMERICAL_SCOPE,
                    "logprob_token_count": len(task_rows),
                },
                "iqk_log_evidence": log_evidence,
                "rows": task_rows,
                "runtime_identity": {
                    "before": identity,
                    "after": identity,
                },
            })
    return {
        "schema": runner.IQK_ATTESTATION_SCHEMA, "created_at": "2026-07-24T00:00:00+00:00", "status": "pass",
        "attestation_roles": {"correctness": True, "coherence": True, "numerical_safety": True},
        "decision_gate": {
            "handoff": "iqk-iquant-enablement B2",
            "b2_gate_passed": True,
            "promotion_decision": False,
            "semantic_contract": "IQK arms are not bit-exact; both independently satisfy fixed tasks",
            "timings": "non-decision observational only",
        },
        "identity": identity,
        "arms": arms,
    }


def candidate_identity_for_attestation() -> dict[str, object]:
    return {"shared_library_identity": {"openmp_runtime": {"sha256": runner.LLVM20_OPENMP_IDENTITY["sha256"]}}}


def test_matrix_has_exact_28_runs_and_14_unique_pairs_with_q8_waived() -> None:
    plan = runner.manifest()
    cells = plan["arm_runs"]
    pairs = plan["pairs"]
    assert len(cells) == 28
    assert len({cell["id"] for cell in cells}) == 28
    assert len(pairs) == len({pair["pair_id"] for pair in pairs}) == 14
    assert all("id" not in pair and "kernel_arm" not in pair for pair in pairs)
    assert all(cell["reps"] == 10 and "122" not in cell["model"] for cell in cells)
    assert all("q8" not in cell["model"] for cell in cells)
    assert sum(not cell["iq"] for cell in cells) == 4
    assert sum(cell["iq"] for cell in cells) == 24
    assert plan["cardinality"] == {"arm_runs": 28, "unique_pairs": 14}
    assert plan["q8_waiver"]["source"]["path"] == str(runner.Q8_WAIVER_PATH)
    assert plan["q8_waiver"]["source"]["sha256"] == runner.Q8_WAIVER_SHA256
    assert plan["q8_waiver"]["semantic_binding"]["scope"]["excluded_arm_runs"] == 4
    assert plan["q8_waiver"]["semantic_binding"]["scope"]["remaining_arm_runs"] == 28
    assert plan["promotion_decision"] is False


def test_q8_waiver_fails_closed_on_missing_hash_and_semantic_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(runner, "Q8_WAIVER_PATH", missing)
    with pytest.raises(RuntimeError, match="required file missing"):
        runner.q8_waiver_attestation()
    with pytest.raises(RuntimeError, match="required file missing"):
        runner.manifest()

    waiver = json.loads(Path(runner.__file__).parents[3].joinpath(
        "epyc-root/artifacts/operator/waive_q8_cpu_prefill_v8_20260725.json"
    ).read_text())
    candidate = tmp_path / "waiver.json"
    candidate.write_text(json.dumps(waiver))
    monkeypatch.setattr(runner, "Q8_WAIVER_PATH", candidate)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        runner.q8_waiver_attestation()

    monkeypatch.setattr(
        runner, "Q8_WAIVER_SHA256", runner.hashlib.sha256(candidate.read_bytes()).hexdigest()
    )
    waiver["scope"]["remaining_arm_runs"] = 27
    candidate.write_text(json.dumps(waiver))
    monkeypatch.setattr(
        runner, "Q8_WAIVER_SHA256", runner.hashlib.sha256(candidate.read_bytes()).hexdigest()
    )
    with pytest.raises(RuntimeError, match="semantic binding"):
        runner.q8_waiver_attestation()


def test_build_pairs_rejects_duplicate_or_missing_arm() -> None:
    cells = runner.build_cells()
    cells[1]["kernel_arm"] = "production"
    with pytest.raises(RuntimeError, match="exactly one"):
        runner.build_pairs(cells)


def test_all_bench_argvs_are_canonical_wrapper_cpu_only() -> None:
    plan = runner.manifest()
    for cell in plan["arm_runs"]:
        arm = plan["arms"][cell["kernel_arm"]]
        argv = runner.argv_for_cell(cell, arm, dry_run=False)
        assert argv[:2] == ["bash", str(runner.CANONICAL_WRAPPER)]
        assert "llama-bench" not in argv[2:]
        assert argv[argv.index("-r") + 1] == "10"
        extra = argv[argv.index("--") + 1:]
        assert extra[: len(runner.CPU_EXTRA)] == list(runner.CPU_EXTRA)
        assert extra[-4:] == list(runner.OUTPUT_EXTRA)


def test_iqk0_control_uses_noninferiority_and_iqk1_uses_utility() -> None:
    production = {"median_ts": 100.0}
    candidate = {"median_ts": 96.0}
    iq_pair = next(pair for pair in runner.manifest()["pairs"] if pair["iq"])
    iq_pair["iqk"] = 0
    assert runner.verdict(iq_pair, production, candidate)["state"] == "gray_retry_required"
    iq_pair["iqk"] = 1
    assert runner.verdict(iq_pair, production, candidate)["state"] == "iq_pair_pending"


def test_gray_zone_pools_twenty_samples() -> None:
    initial = {
        "arms": {
            "production": {"samples_ts": [100.0] * 10},
            "candidate": {"samples_ts": [96.0] * 10},
        }
    }
    retry = {
        "arms": {
            "production": {"samples_ts": [100.0] * 10},
            "candidate": {"samples_ts": [101.0] * 10},
        }
    }
    pooled = runner.apply_gray_retry(initial, retry)
    assert pooled["state"] == "pass"
    assert len(pooled["candidate"]["samples_ts"]) == 20


def test_iq_utility_uses_effective_pooled_ratio() -> None:
    def record(metric: str, initial: float, pooled: float | None = None) -> dict[str, object]:
        value: dict[str, object] = {
            "pair": {
                "pair_id": f"iq-{metric}",
                "model": "iq",
                "iq": True,
                "iqk": 1,
                "metric": metric,
            },
            "initial": {"verdict": {"ratio": initial}},
        }
        if pooled is not None:
            value["pooled"] = {"ratio": pooled}
        return value

    records = [record("tg128", 0.50, 1.06), record("pp2048", 1.20, 0.96)]
    assert runner.evaluate_iq_utility(records)[0]["state"] == "pass"
    records[1]["pooled"] = {"ratio": 0.94}
    assert runner.evaluate_iq_utility(records)[0]["state"] == "fail"


def attribution_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for model in (item for item in runner.MODELS if item.iq):
        for metric in ("tg128", "pp2048"):
            for iqk, median in ((0, 100.0), (1, 110.0)):
                records.append(
                    {
                        "pair": {
                            "pair_id": f"{model.name}-{metric}-iqk{iqk}",
                            "model": model.name,
                            "metric": metric,
                            "iq": True,
                            "iqk": iqk,
                        },
                        "initial": {
                            "arms": {
                                "candidate": {
                                    "samples_ts": [median] * runner.REPS,
                                    "median_ts": median,
                                    "mad_ts": 0.0,
                                }
                            }
                        },
                    }
                )
    return records


def test_candidate_iqk_attribution_emits_six_non_gating_ratios() -> None:
    attribution = runner.evaluate_candidate_iqk_attribution(attribution_records())
    assert attribution["status"] == "valid"
    assert attribution["promotion_gate"] is False
    assert attribution["sample_scope"] == "initial_28_arm_matrix_only"
    assert len(attribution["cells"]) == 6
    assert {cell["metric"] for cell in attribution["cells"]} == {
        "pp2048",
        "tg128",
    }
    assert all(
        cell["ratio_iqk1_over_iqk0"] == pytest.approx(1.1)
        for cell in attribution["cells"]
    )
    assert all(
        cell[gate]["n"] == runner.REPS
        for cell in attribution["cells"]
        for gate in ("iqk0", "iqk1")
    )


def test_candidate_iqk_attribution_fails_closed_on_cardinality_and_samples() -> None:
    records = attribution_records()
    with pytest.raises(RuntimeError, match="cardinality"):
        runner.evaluate_candidate_iqk_attribution(records[:-1])
    with pytest.raises(RuntimeError, match="duplicate"):
        runner.evaluate_candidate_iqk_attribution([*records, records[0]])

    records = attribution_records()
    records[0]["initial"]["arms"]["candidate"]["samples_ts"] = [100.0] * 9
    with pytest.raises(RuntimeError, match="exactly 10"):
        runner.evaluate_candidate_iqk_attribution(records)

    records = attribution_records()
    records[0]["initial"]["arms"]["candidate"]["samples_ts"][0] = float("nan")
    with pytest.raises(RuntimeError, match="finite"):
        runner.evaluate_candidate_iqk_attribution(records)

    records = attribution_records()
    records[0]["initial"]["arms"]["candidate"]["median_ts"] = 101.0
    with pytest.raises(RuntimeError, match="disagrees"):
        runner.evaluate_candidate_iqk_attribution(records)


def test_throughput_failure_collection_uses_pooled_control_verdict() -> None:
    record = {
        "pair": {"pair_id": "control", "iq": True, "iqk": 0},
        "initial": {"verdict": {"state": "gray_retry_required"}},
        "pooled": {"state": "pass", "ratio": 0.99},
    }
    assert runner.collect_throughput_failures([record], []) == []
    record["pooled"] = {"state": "fail", "ratio": 0.97}
    assert runner.collect_throughput_failures([record], []) == ["control"]


def test_build_witness_requires_matching_head() -> None:
    raw = "build: 67a433bf4 (10107)\n"
    assert runner.build_witness(raw, runner.CANDIDATE_HEAD)["build_number"] == "10107"
    with pytest.raises(RuntimeError, match="does not match"):
        runner.build_witness(raw, runner.PRODUCTION_HEAD)


@pytest.mark.parametrize("bad", [True, "1.0", 0, -1, float("nan"), float("inf")])
def test_samples_reject_bool_string_nonpositive_and_nonfinite(bad: object) -> None:
    samples: list[object] = [1.0] * 10
    samples[3] = bad
    with pytest.raises(RuntimeError, match="samples_ts"):
        runner.parse_samples(result_json(samples), "pp2048", runner.CANDIDATE_HEAD)


def test_samples_and_json_build_identity_are_strict() -> None:
    samples = list(range(1, 11))
    assert runner.parse_samples(
        result_json(samples), "pp2048", runner.CANDIDATE_HEAD
    ) == [float(value) for value in samples]
    with pytest.raises(RuntimeError, match="build_commit"):
        runner.parse_samples(
            result_json(samples, commit="6ad45fa3f"),
            "pp2048",
            runner.CANDIDATE_HEAD,
        )
    missing_ns = json.loads(result_json(samples))
    del missing_ns[0]["samples_ns"]
    with pytest.raises(RuntimeError, match="samples_ns"):
        runner.parse_result(json.dumps(missing_ns), "pp2048", runner.CANDIDATE_HEAD)
    invalid_ns = json.loads(result_json(samples))
    invalid_ns[0]["samples_ns"][0] = 0
    with pytest.raises(RuntimeError, match="samples_ns"):
        runner.parse_result(json.dumps(invalid_ns), "pp2048", runner.CANDIDATE_HEAD)


def test_split_model_identity_hashes_every_exact_shard(tmp_path: Path) -> None:
    first = tmp_path / "model-00001-of-00003.gguf"
    for number, content in enumerate((b"a", b"bb", b"ccc"), 1):
        (tmp_path / f"model-{number:05d}-of-00003.gguf").write_bytes(content)
    identity = runner.model_identity(first)
    assert identity["shard_count"] == 3
    assert identity["total_bytes"] == 6
    assert len({item["sha256"] for item in identity["shards"]}) == 3
    (tmp_path / "model-00003-of-00003.gguf").unlink()
    with pytest.raises(RuntimeError, match="missing"):
        runner.discover_model_shards(first)


def test_file_identity_rejects_mutation_during_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "trusted.bin"
    target.write_bytes(b"trusted")
    resolved_target = target.resolve()
    real_stat = Path.stat
    real_sha256 = runner.hashlib.sha256
    hash_started = False
    drifted = False

    class TrackingDigest:
        def __init__(self) -> None:
            self.digest = real_sha256()

        def update(self, chunk: bytes) -> None:
            nonlocal hash_started
            self.digest.update(chunk)
            hash_started = True

        def hexdigest(self) -> str:
            return self.digest.hexdigest()

    def drifting_stat(path: Path, *args: object, **kwargs: object) -> object:
        nonlocal drifted
        result = real_stat(path, *args, **kwargs)
        if path == resolved_target and hash_started and not drifted:
            drifted = True
            values = list(result)
            values[8] += 1
            return type(result)(values)
        return result

    monkeypatch.setattr(Path, "stat", drifting_stat)
    monkeypatch.setattr(runner.hashlib, "sha256", TrackingDigest)
    with pytest.raises(RuntimeError, match="changed while hashing"):
        runner.file_identity(target)


def test_prepare_pair_rewarms_every_shard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shards = [tmp_path / "one.gguf", tmp_path / "two.gguf"]
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> Result:
        calls.append(argv)
        return Result()

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    records = runner.prepare_pair(shards)
    rewarm = [item for item in records if item["step"] == "numa_rewarm"]
    assert [item["shard"] for item in rewarm] == [str(path) for path in shards]
    assert all(item["argv"][0:6] == ["taskset", "-c", "0-95", "numactl", "--interleave=all", "dd"] for item in rewarm)


def test_meminfo_evidence_requires_complete_well_formed_consistent_state() -> None:
    parsed = runner.parse_meminfo_evidence(valid_meminfo_raw())
    assert parsed["memory_kib"]["MemAvailable"] == 50
    assert parsed["fields"]["HugePages_Total"] == {"value": 0, "unit": "count"}

    missing = valid_meminfo_raw().replace("AnonHugePages: 0 kB\n", "")
    with pytest.raises(RuntimeError, match="missing required fields"):
        runner.parse_meminfo_evidence(missing)

    malformed = valid_meminfo_raw().replace(
        "AnonHugePages: 0 kB", "AnonHugePages: NaN kB"
    )
    with pytest.raises(RuntimeError, match="malformed value"):
        runner.parse_meminfo_evidence(malformed)

    wrong_unit = valid_meminfo_raw().replace(
        "HugePages_Total: 0", "HugePages_Total: 0 kB"
    )
    with pytest.raises(RuntimeError, match="unit"):
        runner.parse_meminfo_evidence(wrong_unit)

    inconsistent = valid_meminfo_raw().replace(
        "HugePages_Free: 0", "HugePages_Free: 1"
    )
    with pytest.raises(RuntimeError, match="exceeds"):
        runner.parse_meminfo_evidence(inconsistent)


def test_thp_hugepage_state_binds_raw_fields_pools_and_pmd_size() -> None:
    state = valid_thp_hugepage_state()
    assert runner.validate_thp_hugepage_state(state)["memory_kib"]["MemTotal"] == 100

    state = valid_thp_hugepage_state()
    state["meminfo_fields"]["AnonHugePages"]["value"] = 1
    with pytest.raises(RuntimeError, match="disagree"):
        runner.validate_thp_hugepage_state(state)

    state = valid_thp_hugepage_state()
    state["pools"][1]["free_hugepages"] = 1
    with pytest.raises(RuntimeError, match="free count exceeds total"):
        runner.validate_thp_hugepage_state(state)

    state = valid_thp_hugepage_state()
    state["hpage_pmd_size_bytes"] = 4096 * 1024
    with pytest.raises(RuntimeError, match="PMD-size pool"):
        runner.validate_thp_hugepage_state(state)


def test_host_state_fails_closed_on_missing_or_inconsistent_hugepage_evidence() -> None:
    snapshot = valid_host()
    del snapshot["thp_hugepage_state"]
    assert any(
        "THP/hugepage pool state is invalid" in item
        for item in runner.host_state_blockers(snapshot)
    )

    snapshot = valid_host()
    snapshot["memory_kib"]["MemAvailable"] = 49
    assert any(
        "memory_kib disagrees" in item for item in runner.host_state_blockers(snapshot)
    )


def test_shared_library_identity_hashes_only_local_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "llama-bench"
    llama = tmp_path / "libllama.so.0"
    ggml = tmp_path / "libggml.so.0"
    for path in (binary, llama, ggml):
        path.write_bytes(path.name.encode())
    openmp = Path(runner.LLVM20_LIBDIR) / "libgomp.so.1"
    ldd = (
        f"libllama.so.0 => {llama} (0x0)\n"
        f"libggml.so.0 => {ggml} (0x0)\n"
        f"libgomp.so.1 => {openmp} (0x0)\n"
    )
    captured_env: dict[str, str] = {}

    def ldd_run(*_args: object, **kwargs: object) -> Result:
        captured_env.update(kwargs["env"])
        return Result(stdout=ldd)

    monkeypatch.setattr(runner, "run", ldd_run)
    identity = runner.shared_library_identities(binary, tmp_path)
    assert len(identity["libraries"]) == 2
    assert all(item["sha256"] for item in identity["libraries"])
    assert identity["openmp_runtime"]["soname"] == "libgomp.so.1"
    assert identity["openmp_runtime"]["resolved_target"] == str(
        (Path(runner.LLVM20_LIBDIR) / "libomp.so.5").resolve()
    )
    assert identity["openmp_runtime"]["sha256"]
    assert captured_env["LD_LIBRARY_PATH"] == f"{tmp_path}:{runner.LLVM20_LIBDIR}"
    assert "LD_PRELOAD" not in captured_env
    outside = tmp_path.parent / "outside-libllama.so"
    outside.write_bytes(b"x")
    monkeypatch.setattr(
        runner,
        "run",
        lambda *_args, **_kwargs: Result(
            stdout=(
                f"libllama.so.0 => {outside} (0x0)\n"
                f"libggml.so.0 => {ggml} (0x0)\n"
                f"libgomp.so.1 => {openmp} (0x0)\n"
            )
        ),
    )
    with pytest.raises(RuntimeError, match="outside"):
        runner.shared_library_identities(binary, tmp_path)

    monkeypatch.setattr(
        runner,
        "run",
        lambda *_args, **_kwargs: Result(
            stdout=f"libllama.so.0 => {llama} (0x0)\nlibggml.so.0 => {ggml} (0x0)\n"
        ),
    )
    with pytest.raises(RuntimeError, match="OpenMP runtimes"):
        runner.shared_library_identities(binary, tmp_path)


def test_source_status_allows_only_known_production_untracked_entries() -> None:
    allowed = "?? .gitnexusignore\n?? tools/math-tools/\n"
    assert runner.validate_source_status("production", allowed) == allowed.splitlines()
    for dirty in (
        " M src/file.cpp\n",
        "M  src/file.cpp\n",
        "A  src/file.cpp\n",
        "?? unexpected.txt\n",
    ):
        with pytest.raises(RuntimeError):
            runner.validate_source_status("production", dirty)
    assert runner.validate_source_status("candidate", "") == []
    with pytest.raises(RuntimeError, match="completely clean"):
        runner.validate_source_status("candidate", "?? anything\n")


def test_host_state_rejects_uptime_process_kfd_and_rocm_failures() -> None:
    snapshot = valid_host()
    assert runner.host_state_blockers(snapshot) == []
    snapshot["uptime_seconds"] = runner.MAX_UPTIME_SECONDS + 1
    snapshot["process_ownership"]["exact_llama_processes"] = [{"pid": 1}]
    snapshot["kfd_ownership"]["users"] = [{"pid": 2}]
    snapshot["rocm_ownership"]["owners"] = [{"pid": 2}]
    blockers = runner.host_state_blockers(snapshot)
    assert any("uptime" in item for item in blockers)
    assert any("llama" in item for item in blockers)
    assert any("kfd" in item.lower() for item in blockers)
    assert any("ROCm" in item for item in blockers)


def test_host_state_fails_closed_on_unresolved_process_identity() -> None:
    snapshot = valid_host()
    snapshot["process_ownership"]["unresolved_ownership"] = [
        {"pid": 77, "error": "PermissionError"}
    ]
    assert any(
        "comm/cmdline" in item for item in runner.host_state_blockers(snapshot)
    )
    snapshot = valid_host()
    snapshot["process_ownership"]["uncertain_relevant_processes"] = [
        {"pid": 78, "comm": "llama-server"}
    ]
    assert any(
        "unresolved relevant" in item
        for item in runner.host_state_blockers(snapshot)
    )


def test_deleted_llama_executable_is_uncertain_and_blocking(tmp_path: Path) -> None:
    process_dir = tmp_path / "4242"
    process_dir.mkdir()
    (process_dir / "comm").write_text("llama-server\n")
    (process_dir / "cmdline").write_bytes(b"/tmp/llama-server\0-m\0model.gguf\0")
    (process_dir / "exe").symlink_to(tmp_path / "deleted-llama-server")
    ownership = runner.process_ownership(tmp_path)
    assert ownership["exact_llama_processes"] == []
    assert ownership["uncertain_relevant_processes"][0]["pid"] == 4242
    assert "missing/deleted" in ownership["uncertain_relevant_processes"][0]["reason"]
    snapshot = valid_host()
    snapshot["process_ownership"] = ownership
    assert any(
        "unresolved relevant" in item
        for item in runner.host_state_blockers(snapshot)
    )


def test_run_pair_persists_post_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshots = iter([valid_host(), {**valid_host(), "process_ownership": {
        "exact_llama_processes": [{"pid": 44}],
        "autopilot_processes": [],
        "unreadable_processes": [],
        "unresolved_ownership": [],
        "uncertain_relevant_processes": [],
    }}])
    monkeypatch.setattr(runner, "host_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(runner, "prepare_pair", lambda _shards: [])
    monkeypatch.setattr(
        runner,
        "run_arm",
        lambda _pair, arm, _artifact: {
            "arm": arm["name"],
            "median_ts": 100.0,
            "samples_ts": [100.0] * 10,
        },
    )
    pair = {
        "pair_id": "pair",
        "model": "model",
        "model_path": "/tmp/model.gguf",
        "iq": False,
        "iqk": 1,
        "metric": "pp2048",
    }
    arms = {
        "production": {"name": "production"},
        "candidate": {"name": "candidate"},
    }
    model = {"shards": [{"path": "/tmp/model.gguf"}]}
    with pytest.raises(RuntimeError, match="durable evidence"):
        runner.run_pair(pair, arms, model, tmp_path, ("production", "candidate"))
    summary = json.loads(
        (tmp_path / "pair" / "production-then-candidate" / "pair_summary.json").read_text()
    )
    assert "post_host_error" in summary
    assert summary["post_host"]["process_ownership"]["exact_llama_processes"]


def test_execute_requires_existing_file_attestations(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--execute", "--host-attestation-path", str(tmp_path / "missing")])
    directory = tmp_path / "directory"
    directory.mkdir()
    paths = make_attestations(tmp_path)
    paths["host"] = directory
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--execute",
                "--host-attestation-path",
                str(paths["host"]),
                "--correctness-attestation-path",
                str(paths["correctness"]),
                "--coherence-attestation-path",
                str(paths["coherence"]),
                "--numerical-safety-attestation-path",
                str(paths["numerical_safety"]),
            ]
        )
    paths = make_attestations(tmp_path)
    args = runner.parse_args(
        [
            "--execute",
            "--host-attestation-path",
            str(paths["host"]),
            "--correctness-attestation-path",
            str(paths["correctness"]),
            "--coherence-attestation-path",
            str(paths["coherence"]),
            "--numerical-safety-attestation-path",
            str(paths["numerical_safety"]),
        ]
    )
    assert args.host_attestation_path == paths["host"]


def test_attestation_roles_are_hashed_and_path_mutation_is_detected(tmp_path: Path) -> None:
    paths = make_attestations(tmp_path)
    before = {
        "arms": {},
        "models": {},
        "harness": {},
        "attestations": runner.attestation_identities(paths),
        "q8_waiver": runner.q8_waiver_attestation(),
    }
    assert set(before["attestations"]) == {
        "host",
        "correctness",
        "coherence",
        "numerical_safety",
    }
    assert all(item["bytes"] > 0 and item["sha256"] for item in before["attestations"].values())
    paths["host"].write_text('{"mutated": true}')
    after = {
        **before,
        "attestations": runner.attestation_identities(paths),
    }
    with pytest.raises(RuntimeError, match="mutated"):
        runner.require_identical_bound_inputs(before, after)


def test_bound_binary_library_model_and_harness_mutations_fail() -> None:
    openmp_runtime = {
        "soname": "libgomp.so.1",
        "resolved_target": "/usr/lib/llvm-20/lib/libomp.so.5",
        "bytes": 2,
        "sha256": "omp",
    }
    arm = {
        "name": "production",
        "source_root": "/source",
        "actual_head": "head",
        "actual_branch": "branch",
        "source_status": [],
        "binary_identity": {"path": "/binary", "bytes": 1, "sha256": "a"},
        "shared_library_identity": {
            "openmp_runtime": openmp_runtime,
            "libraries": [
                {
                    "soname": "libllama.so",
                    "resolved_target": "/libllama.so",
                    "bytes": 1,
                    "sha256": "b",
                }
            ]
        },
    }
    before = {
        "arms": {"production": arm},
        "models": {"m": {"shards": [{"sha256": "c"}]}},
        "harness": {"runner": {"sha256": "d"}},
        "attestations": {},
        "q8_waiver": runner.q8_waiver_attestation(),
    }
    for section, replacement in (
        ("arms", {**arm, "binary_identity": {"path": "/binary", "bytes": 1, "sha256": "x"}}),
        (
            "arms",
            {
                **arm,
                "shared_library_identity": {
                    "openmp_runtime": openmp_runtime,
                    "libraries": [
                        {
                            "soname": "libllama.so",
                            "resolved_target": "/libllama.so",
                            "bytes": 1,
                            "sha256": "x",
                        }
                    ]
                },
            },
        ),
        ("models", {"shards": [{"sha256": "x"}]}),
        ("harness", {"sha256": "x"}),
    ):
        after = {
            "arms": before["arms"],
            "models": before["models"],
            "harness": before["harness"],
            "attestations": {},
            "q8_waiver": before["q8_waiver"],
        }
        if section == "arms":
            after["arms"] = {"production": replacement}
        elif section == "models":
            after["models"] = {"m": replacement}
        else:
            after["harness"] = {"runner": replacement}
        with pytest.raises(RuntimeError, match="mutated"):
            runner.require_identical_bound_inputs(before, after)

    after = {**before, "q8_waiver": {"source": {"sha256": "mutated"}}}
    with pytest.raises(RuntimeError, match="mutated"):
        runner.require_identical_bound_inputs(before, after)

    after = {
        **before,
        "arms": {
            "production": {
                **arm,
                "shared_library_identity": {
                    **arm["shared_library_identity"],
                    "openmp_runtime": {
                        **openmp_runtime,
                        "sha256": "changed",
                    },
                },
            }
        },
    }
    with pytest.raises(RuntimeError, match="mutated"):
        runner.require_identical_bound_inputs(before, after)


def test_harness_identity_binds_runner_wrapper_and_recipe() -> None:
    identity = runner.harness_identities()
    assert set(identity) == {
        "runner",
        "bench_canonical",
        "canonical_recipe",
        "parent_environment",
        "instrument_eras",
    }
    for role in ("runner", "bench_canonical", "canonical_recipe"):
        assert identity[role]["bytes"] > 0 and identity[role]["sha256"]
    assert identity["parent_environment"] == runner.parent_environment_identity()


def test_hostile_parent_environment_cannot_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poisoned = {
        "LD_PRELOAD": "/tmp/evil.so",
        "LD_LIBRARY_PATH": "/tmp/evil",
        "HSA_OVERRIDE_GFX_VERSION": "poison",
        "HIP_VISIBLE_DEVICES": "1",
        "ROCR_VISIBLE_DEVICES": "1",
        "GGML_IQK": "poison",
        "GGML_OTHER": "poison",
        "OMP_NUM_THREADS": "1",
        "KMP_BLOCKTIME": "999",
        "PYTHONPATH": "/tmp/evil",
        "BASH_ENV": "/tmp/evil.sh",
    }
    for key, value in poisoned.items():
        monkeypatch.setenv(key, value)
    environment = runner.canonical_parent_environment()
    assert environment == runner.CANONICAL_PARENT_ENV
    assert "KMP_BLOCKTIME" not in environment
    assert all(
        key not in environment or environment[key] != value
        for key, value in poisoned.items()
    )
    assert runner.parent_environment_identity()["environment"] == environment
    assert runner.manifest()["parent_environment_identity"] == runner.parent_environment_identity()


def test_canonical_environment_witness_validates_emitted_and_effective_stack() -> None:
    cell = {"iqk": 0}
    arm = {"library_path": "/candidate"}
    witness = runner.canonical_environment_witness(
        wrapper_stderr(iqk=0),
        cell,
        arm,
    )
    assert "KMP_BLOCKTIME" not in witness["wrapper_emitted"]["environment"]
    effective = witness["effective"]["environment"]
    assert "KMP_BLOCKTIME" not in effective
    assert effective["GGML_IQK"] == "0"
    assert {
        key: effective[key] for key in runner.REQUIRED_WRAPPER_OMP_ENV
    } == runner.REQUIRED_WRAPPER_OMP_ENV


def test_canonical_environment_witness_rejects_missing_duplicate_and_drift() -> None:
    with pytest.raises(RuntimeError, match="expected one"):
        runner.parse_wrapper_emitted_environment("build: 67a433bf4 (10107)\n")
    duplicate = wrapper_stderr() + wrapper_stderr()
    with pytest.raises(RuntimeError, match="expected one"):
        runner.parse_wrapper_emitted_environment(duplicate)
    drifted = wrapper_stderr().replace(
        "OMP_WAIT_POLICY=active", "OMP_WAIT_POLICY=passive"
    )
    with pytest.raises(RuntimeError, match="environment drifted"):
        runner.canonical_environment_witness(
            drifted,
            {"iqk": 1},
            {"library_path": "/candidate"},
        )


def test_q8_only_flag_is_absent_from_every_retained_command_and_witness() -> None:
    plan = runner.manifest()
    assert all("qwen36_q8" not in cell["id"] for cell in plan["arm_runs"])
    assert all("--ggml-iqk-q8-0" not in runner.argv_for_cell(
        cell, plan["arms"][cell["kernel_arm"]], dry_run=True
    ) for cell in plan["arm_runs"])
    with pytest.raises(RuntimeError, match="environment drifted"):
        runner.canonical_environment_witness(
            wrapper_stderr(q8_0=True), {"model": "glm_iq2", "iqk": 1},
            {"library_path": "/candidate"},
        )


def test_instrument_era_attestation_fails_closed_on_missing_or_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(runner, "INSTRUMENT_ERAS", missing)
    with pytest.raises(RuntimeError, match="required file missing"):
        runner.instrument_era_attestation()
    registry = tmp_path / "eras.yaml"
    registry.write_text(
        "eras:\n  - id: E6-cpu-kernel\n    from: '2026-07-20T13:30:13Z'\n    scope: cpu_bench\n"
        "  - id: E7-eval-instrument\n    from: '2026-07-21T10:30:00Z'\n    scope: eval_quality\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "INSTRUMENT_ERAS", registry)
    attestation = runner.instrument_era_attestation()
    assert attestation["active"]["cpu_bench"]["id"] == "E6-cpu-kernel"
    assert attestation["active"]["eval_quality"]["row_boundaries"] == {"start_line": 5, "end_line": 7}
    real_file_identity = runner.file_identity
    monkeypatch.setattr(runner, "file_identity", lambda _path: {"sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="changed between identity hashing and parsing"):
        runner.instrument_era_attestation()
    monkeypatch.setattr(runner, "file_identity", real_file_identity)
    registry.write_text(registry.read_text(encoding="utf-8").replace("E6-cpu-kernel", "E9-cpu-kernel"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="drifted"):
        runner.instrument_era_attestation()


def test_measurement_window_observation_preserves_unbound_overlap_facts() -> None:
    monitor = {
        "samples": [{"monotonic": 1.0}, {"monotonic": 11.0}],
        "intervals": [{"index": 0}],
        "sustained_window": {"start_monotonic": 1.0, "end_monotonic": 11.0, "elapsed_s": 10.0},
    }
    observation = runner.measurement_window_observation(monitor, [2_000_000_000] * 2)
    assert observation["total_measured_repetition_duration_s"] == 4.0
    assert observation["minimum_clean_overlap_s"] == 4.0
    assert observation["binding_status"] == "unavailable"
    assert observation["per_repetition_timestamps"] == "unavailable"
    monitor["sustained_window"]["elapsed_s"] = 9.5
    with pytest.raises(RuntimeError, match="duration disagrees with endpoints"):
        runner.measurement_window_observation(monitor, [2_000_000_000] * 2)
    monitor["sustained_window"] = {"start_monotonic": 3.0, "end_monotonic": 9.0, "elapsed_s": 6.0}
    observation = runner.measurement_window_observation(monitor, [2_000_000_000] * 2)
    assert observation["minimum_clean_overlap_s"] == 0.0


def test_measurement_window_observation_accepts_valid_window_with_startup_teardown() -> None:
    samples = monitor_series([(9000, 100)] + [(9000, 8800)] * 10 + [(9000, 100)])
    monitor = runner.validate_monitor_samples(samples)
    assert monitor["sustained_window"]["start_interval_index"] == 1
    assert monitor["sustained_window"]["end_interval_index"] == 10
    for sample in monitor["samples"]:
        sample["monotonic"] += 1.0
    monitor["sustained_window"]["start_monotonic"] += 1.0
    monitor["sustained_window"]["end_monotonic"] += 1.0
    observation = runner.measurement_window_observation(
        monitor, [1_000_000_000] * 10
    )
    assert observation["selected_clean_window_duration_s"] == pytest.approx(10.0)
    assert observation["observed_monitor_duration_s"] == pytest.approx(12.0)
    assert observation["minimum_clean_overlap_s"] == pytest.approx(8.0)
    assert observation["binding_status"] == "unavailable"


def test_run_default_environment_is_exact_under_parent_poison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in {
        "LD_PRELOAD": "/tmp/evil.so",
        "LD_LIBRARY_PATH": "/tmp/evil",
        "HSA_OVERRIDE_GFX_VERSION": "poison",
        "HIP_VISIBLE_DEVICES": "1",
        "ROCR_VISIBLE_DEVICES": "1",
        "GGML_IQK": "poison",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": "/tmp/evil",
        "BASH_ENV": "/tmp/evil.sh",
    }.items():
        monkeypatch.setenv(key, value)
    observed: list[dict[str, str]] = []

    def fake_subprocess_run(_argv: list[str], **kwargs: object) -> Result:
        observed.append(kwargs["env"])
        return Result()

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)
    runner.run(["sync"])
    assert observed == [runner.CANONICAL_PARENT_ENV]


def test_all_helper_commands_receive_the_exact_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_subprocess_run(argv: list[str], **kwargs: object) -> Result:
        calls.append((argv, kwargs["env"]))
        return Result()

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    runner.prepare_pair([tmp_path / "model.gguf"])
    runner.run(["lsof", "-n", "-P", "-Fpcn", "/dev/kfd"])
    runner.run(["rocm-smi", "--showpidgpus"])
    assert [argv[0] for argv, _environment in calls] == [
        "sync",
        "sudo",
        "taskset",
        "lsof",
        "rocm-smi",
    ]
    assert all(
        environment == runner.CANONICAL_PARENT_ENV
        for _argv, environment in calls
    )
    drop_argv = calls[1][0]
    assert drop_argv[:4] == ["sudo", "-n", "/usr/bin/env", "-i"]
    tee_index = drop_argv.index("/usr/bin/tee")
    assert drop_argv[4:tee_index] == runner.environment_assignments()
    assert drop_argv[tee_index + 1 :] == ["/proc/sys/vm/drop_caches"]


def test_subprocess_environment_allowlist_rejects_extras_missing_and_drift() -> None:
    canonical = runner.canonical_parent_environment()
    for environment in (
        {**canonical, "LD_PRELOAD": "/tmp/evil.so"},
        {key: value for key, value in canonical.items() if key != "HOME"},
        {**canonical, "PATH": "/tmp/evil"},
        {**canonical, "LD_LIBRARY_PATH": "relative/path"},
    ):
        with pytest.raises(RuntimeError, match="environment|LD_LIBRARY_PATH"):
            runner.exact_subprocess_environment(environment)
    accepted = {
        **canonical,
        "LD_LIBRARY_PATH": "/candidate/build/bin:/usr/lib/llvm-20/lib",
    }
    assert runner.exact_subprocess_environment(accepted) == accepted


def test_run_arm_passes_only_exact_parent_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, str] = {}

    def fake_run_monitored(_argv: list[str], env: dict[str, str]) -> tuple[Result, dict[str, object]]:
        observed.update(env)
        return Result(
            stdout=result_json([1.0] * 10),
            stderr=wrapper_stderr(),
        ), {
            "status": "pass",
            "samples": [{"monotonic": 1.0}, {"monotonic": 11.0}],
            "intervals": [{}],
            "sustained_window": {
                "start_monotonic": 1.0,
                "end_monotonic": 11.0,
                "elapsed_s": 10.0,
            },
        }

    monkeypatch.setattr(runner, "run_monitored", fake_run_monitored)
    identity = {
        "name": "candidate", "source_root": "/candidate", "actual_head": runner.CANDIDATE_HEAD,
        "actual_branch": runner.CANDIDATE_BRANCH, "source_status": [],
        "binary_identity": {"path": "/candidate/llama-bench", "bytes": 1, "sha256": "x"},
        "shared_library_identity": {"libraries": [], "openmp_runtime": {}},
    }
    monkeypatch.setattr(runner, "collect_arm_identity", lambda _arm: identity)
    monkeypatch.setattr(
        runner,
        "resolve_build_commit",
        lambda *_args: runner.CANDIDATE_HEAD,
    )
    cell = {
        "pair_id": "p",
        "model": "m",
        "model_path": "/tmp/model.gguf",
        "iq": False,
        "iqk": 1,
        "metric": "pp2048",
        "n_prompt": 2048,
        "n_gen": 0,
    }
    arm = {
        "name": "candidate",
        "binary": "/candidate/llama-bench",
        "source_root": "/candidate",
        "library_path": "/candidate",
        "actual_head": runner.CANDIDATE_HEAD,
        "expected_head": runner.CANDIDATE_HEAD,
        "expected_branch": runner.CANDIDATE_BRANCH,
    }
    row = runner.run_arm(cell, arm, tmp_path / "candidate.log")
    assert observed == runner.CANONICAL_PARENT_ENV
    assert row["parent_environment_identity"] == runner.parent_environment_identity()
    assert "KMP_BLOCKTIME" not in row["canonical_environment_witness"]["effective"]["environment"]


def test_run_arm_persists_monitor_before_downstream_parse_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monitor = {
        "status": "pass",
        "samples": [{"monotonic": 1.0}, {"monotonic": 11.0}],
        "intervals": [{"index": 0}],
        "sustained_window": {
            "start_monotonic": 1.0,
            "end_monotonic": 11.0,
            "elapsed_s": 10.0,
        },
    }

    def fake_run_monitored(_argv: list[str], _env: dict[str, str]) -> tuple[Result, dict[str, object]]:
        return Result(stdout="malformed result", stderr=wrapper_stderr()), monitor

    identity = {
        "name": "candidate", "source_root": "/candidate", "actual_head": runner.CANDIDATE_HEAD,
        "actual_branch": runner.CANDIDATE_BRANCH, "source_status": [],
        "binary_identity": {"path": "/candidate/llama-bench", "bytes": 1, "sha256": "x"},
        "shared_library_identity": {"libraries": [], "openmp_runtime": {}},
    }
    monkeypatch.setattr(runner, "run_monitored", fake_run_monitored)
    monkeypatch.setattr(runner, "collect_arm_identity", lambda _arm: identity)
    monkeypatch.setattr(
        runner,
        "parse_result",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("downstream parse failure")),
    )
    cell = {
        "pair_id": "p", "model": "m", "model_path": "/tmp/model.gguf", "iq": False,
        "iqk": 1, "metric": "pp2048", "n_prompt": 2048, "n_gen": 0,
    }
    arm = {
        "name": "candidate", "binary": "/candidate/llama-bench", "source_root": "/candidate",
        "library_path": "/candidate", "actual_head": runner.CANDIDATE_HEAD,
        "expected_head": runner.CANDIDATE_HEAD, "expected_branch": runner.CANDIDATE_BRANCH,
    }
    artifact = tmp_path / "candidate.log"
    with pytest.raises(RuntimeError, match="downstream parse failure"):
        runner.run_arm(cell, arm, artifact)
    assert json.loads(artifact.with_suffix(".contention_monitor.json").read_text()) == monitor


def test_ratios_reject_nonfinite_and_overflow() -> None:
    pair = {"pair_id": "p", "iq": False, "iqk": 1}
    with pytest.raises(RuntimeError, match="overflowed"):
        runner.verdict(pair, {"median_ts": 1e-308}, {"median_ts": 1e308})
    initial = {
        "arms": {
            "production": {"samples_ts": [1e-308] * 10},
            "candidate": {"samples_ts": [1e308] * 10},
        }
    }
    with pytest.raises(RuntimeError, match="overflowed|non-finite"):
        runner.apply_gray_retry(initial, initial)
    record = {
        "pair": {"pair_id": "iq", "model": "iq", "iq": True, "iqk": 1, "metric": "tg128"},
        "initial": {"verdict": {"ratio": float("nan")}},
    }
    with pytest.raises(RuntimeError, match="finite"):
        runner.effective_ratio(record)


def test_json_serialization_rejects_nan(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        runner.write_json(tmp_path / "nan.json", {"ratio": float("nan")})
    with pytest.raises(ValueError):
        runner.json_text({"ratio": float("inf")})


def test_build_commit_must_resolve_exactly_to_pinned_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "git_value", lambda *_args: runner.CANDIDATE_HEAD)
    assert (
        runner.resolve_build_commit(Path("/source"), "67a433bf4", runner.CANDIDATE_HEAD)
        == runner.CANDIDATE_HEAD
    )
    monkeypatch.setattr(runner, "git_value", lambda *_args: runner.PRODUCTION_HEAD)
    with pytest.raises(RuntimeError, match="not pinned HEAD"):
        runner.resolve_build_commit(Path("/source"), "67a433bf4", runner.CANDIDATE_HEAD)


def test_fresh_output_rejects_nonempty_directory(tmp_path: Path) -> None:
    (tmp_path / "existing").write_text("x")
    with pytest.raises(RuntimeError, match="nonempty"):
        runner.fresh_output_dir(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda value: value.clear(), "schema keys"),
        (lambda value: value.__setitem__("status", "fail"), "non-pass"),
        (lambda value: value["identity"]["candidate"].__setitem__("head", "wrong"), "branch/head"),
        (lambda value: value["arms"].pop(), "complete exact"),
        (lambda value: value["identity"]["runner"].__setitem__("sha256", "0" * 64), "runner identity"),
        (lambda value: value["identity"]["models"]["qwen_next_iq2"]["shards"][0].__setitem__("sha256", "f" * 64), "model identity"),
        (lambda value: value["arms"][0]["numerical_safety"].__setitem__("status", "fail"), "numerical evidence"),
        (lambda value: value["arms"][0]["rows"][0]["semantic"].__setitem__("status", "fail"), "task evidence"),
        (lambda value: value["arms"][0]["cleanup"].__setitem__("status", "fail"), "cleanup"),
        (lambda value: value["arms"][0]["runtime_identity"].__setitem__("after", {}), "runtime identity"),
    ],
)
def test_external_iqk_attestation_is_strict_and_fail_closed(
    tmp_path: Path, mutation: object, error: str
) -> None:
    value = valid_iqk_attestation()
    mutation(value)
    path = tmp_path / "attestation.json"
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match=error):
        runner.validate_iqk_attestation(
            "correctness",
            path,
            candidate_identity_for_attestation(),
            synthetic_model_identities(),
        )


def test_external_iqk_attestation_accepts_complete_producer_schema(
    tmp_path: Path,
) -> None:
    current_models = synthetic_model_identities()
    path = tmp_path / "attestation.json"
    path.write_text(json.dumps(valid_iqk_attestation(current_models)))
    validated = runner.validate_iqk_attestation(
        "correctness",
        path,
        candidate_identity_for_attestation(),
        current_models,
    )
    assert validated["verified_by_runner"] is True


def monitor_sample(
    *, monotonic: float, busy: int, total: int, target: int,
    swap_in: int = 0, swap_out: int = 0,
    contaminated: bool = False, ownership_changed: bool = False,
) -> dict[str, object]:
    return {"monotonic": monotonic, "cpu_total_ticks": total, "cpu_busy_ticks": busy,
            "cpu_counter_bracket": {
                "before": {"monotonic": monotonic - 0.1, "total_ticks": total, "busy_ticks": busy},
                "after": {"monotonic": monotonic + 0.1, "total_ticks": total, "busy_ticks": busy},
                "target_scan": {"started_monotonic": monotonic - 0.01, "finished_monotonic": monotonic + 0.01, "elapsed_s": 0.02},
                "target_monotonic": monotonic,
                "interpolation_fraction": 0.5,
            },
            "swap": {"pswpin": swap_in, "pswpout": swap_out},
            "target": {"cpu_ticks": target, "ownership_changed": ownership_changed, "members": [1], "vanished_processes": []},
            "contamination": {"exact_llama": [{"pid": 7}] if contaminated else [], "autopilot": [], "kfd_users": []}}


def monitor_series(
    deltas: list[tuple[int, int]], *, total_delta: int = 9600,
) -> list[dict[str, object]]:
    """Build one-second monitor samples from (busy, target) tick deltas."""
    samples = [monitor_sample(monotonic=0.0, busy=0, total=0, target=0)]
    busy = total = target = 0
    for index, (busy_delta, target_delta) in enumerate(deltas, start=1):
        busy += busy_delta
        total += total_delta
        target += target_delta
        samples.append(monitor_sample(
            monotonic=float(index), busy=busy, total=total, target=target,
        ))
    return samples


def test_contention_monitor_selects_sustained_window_and_retains_raw_intervals() -> None:
    samples = monitor_series([(9000, 8800)] * 10)
    result = runner.validate_monitor_samples(samples)
    assert result["status"] == "pass"
    assert result["accounting"] == "sustained-window-v1"
    assert len(result["samples"]) == 11
    assert len(result["intervals"]) == 10
    assert result["intervals"][0]["signed_external_core_equivalents"] == pytest.approx(2.0)
    assert result["intervals"][0]["exclusion_reasons"] == []
    assert result["sustained_window"]["elapsed_s"] == pytest.approx(10.0)
    assert result["sustained_window"]["direct_endpoint_elapsed_s"] == pytest.approx(10.0)
    assert result["sustained_window"]["aggregate_total_delta_ticks"] == pytest.approx(96000.0)
    assert result["sustained_window"]["target_core_equivalents"] == pytest.approx(88.0)
    assert result["sustained_window"]["signed_external_core_equivalents"] == pytest.approx(2.0)


def test_contention_monitor_accepts_signed_target_over_busy_and_inclusive_bounds() -> None:
    lower = runner.validate_monitor_samples(monitor_series([(7900, 8000)] * 10))
    assert lower["sustained_window"]["signed_external_core_equivalents"] == pytest.approx(-1.0)
    upper = runner.validate_monitor_samples(monitor_series([(8400, 8000)] * 10))
    assert upper["sustained_window"]["signed_external_core_equivalents"] == pytest.approx(4.0)
    with pytest.raises(RuntimeError, match="outside inclusive bounds"):
        runner.validate_monitor_samples(monitor_series([(7899, 8000)] * 10))
    with pytest.raises(RuntimeError, match="outside inclusive bounds"):
        runner.validate_monitor_samples(monitor_series([(8401, 8000)] * 10))


def test_contention_monitor_uses_longest_eligible_run_and_earliest_tie() -> None:
    # Two equally long 10-second windows, split by a retained low-use interval.
    samples = monitor_series([(9000, 8800)] * 10 + [(9000, 100)] + [(9000, 8800)] * 10)
    result = runner.validate_monitor_samples(samples)
    assert result["intervals"][10]["eligible"] is False
    assert result["intervals"][10]["exclusion_reasons"] == ["target_below_minimum_core_equivalents"]
    assert result["sustained_window"]["start_interval_index"] == 0
    assert result["sustained_window"]["end_interval_index"] == 9


def test_contention_monitor_selects_a_later_strictly_longer_run() -> None:
    samples = monitor_series(
        [(9000, 8800)] * 10 + [(9000, 100)] + [(9000, 8800)] * 11,
    )
    result = runner.validate_monitor_samples(samples)
    assert result["sustained_window"]["start_interval_index"] == 11
    assert result["sustained_window"]["end_interval_index"] == 21
    assert result["sustained_window"]["elapsed_s"] == pytest.approx(11.0)


def test_contention_monitor_hard_failures_apply_outside_candidate_window() -> None:
    samples = monitor_series([(9000, 8800)] * 10 + [(9000, 100)])
    samples[-1]["contamination"]["autopilot"].append({"pid": 9})  # type: ignore[index]
    with pytest.raises(RuntimeError, match="transient competing"):
        runner.validate_monitor_samples(samples)
    samples = monitor_series([(9000, 8800)] * 10 + [(9000, 100)])
    samples[-1]["target"]["ownership_changed"] = True  # type: ignore[index]
    with pytest.raises(RuntimeError, match="ownership changed"):
        runner.validate_monitor_samples(samples)
    samples = monitor_series([(9000, 8800)] * 10 + [(9000, 100)])
    samples[-1]["swap"]["pswpout"] = 1  # type: ignore[index]
    with pytest.raises(RuntimeError, match="swap I/O"):
        runner.validate_monitor_samples(samples)


def test_contention_monitor_rejects_short_and_malformed_runs() -> None:
    with pytest.raises(RuntimeError, match="qualifying sustained"):
        runner.validate_monitor_samples(monitor_series([(9000, 8800)] * 9))
    with pytest.raises(RuntimeError, match="sampling failure"):
        runner.validate_monitor_samples([
            monitor_sample(monotonic=0.0, busy=0, total=0, target=0),
            monitor_sample(monotonic=1.0, busy=9601, total=9600, target=0),
        ])
    with pytest.raises(RuntimeError, match="sampling failure"):
        runner.validate_monitor_samples([
            monitor_sample(monotonic=0.0, busy=0, total=0, target=0),
            monitor_sample(monotonic=1.0, busy=100, total=9600, target=-1),
        ])
    with pytest.raises(RuntimeError, match="not finite"):
        runner.validate_monitor_samples([
            monitor_sample(monotonic=0.0, busy=0, total=0, target=0),
            monitor_sample(monotonic=1.0, busy=float("nan"), total=9600, target=0),
        ])
    with pytest.raises(RuntimeError, match="sampling failure"):
        runner.validate_monitor_samples([
            monitor_sample(monotonic=0.0, busy=0, total=0, target=0),
            {"monotonic": 1.0},
        ])
    with pytest.raises(RuntimeError, match="insufficient"):
        runner.validate_monitor_samples([monitor_sample(monotonic=0.0, busy=0, total=0, target=0)])


@pytest.mark.parametrize("mutation", [
    lambda sample: sample.__setitem__("contamination", {}),
    lambda sample: sample["contamination"].__setitem__("exact_llama", {}),
    lambda sample: sample["target"].__setitem__("members", []),
    lambda sample: sample["target"]["vanished_processes"].append({}),
    lambda sample: sample.__setitem__("cpu_counter_bracket", {}),
    lambda sample: sample["cpu_counter_bracket"]["target_scan"].__setitem__("elapsed_s", 3.0),
    lambda sample: sample["cpu_counter_bracket"].__setitem__("interpolation_fraction", 0.25),
    lambda sample: sample["cpu_counter_bracket"]["after"].__setitem__("busy_ticks", 1.5),
    lambda sample: sample["cpu_counter_bracket"]["after"].__setitem__("total_ticks", -1),
])
def test_contention_monitor_fails_closed_for_malformed_witnesses(mutation: object) -> None:
    samples = monitor_series([(9000, 8800)] * 10)
    mutation(samples[0])  # type: ignore[operator]
    with pytest.raises(RuntimeError, match="sampling failure"):
        runner.validate_monitor_samples(samples)


def test_manifest_marks_sustained_accounting_v3_as_prospective() -> None:
    plan = runner.manifest()
    assert plan["schema"] == "cpu-prefill-v8-regression.v3"
    accounting = plan["contention_accounting"]
    assert accounting["id"] == "sustained-window-v1"
    assert accounting["prospective_only"] is True
    assert accounting["minimum_target_core_equivalents"] == pytest.approx(72.0)
    assert plan["q8_waiver"]["semantic_binding"]["protocol_changed"] is False
    assert "72-core eligibility floor remains unchanged" in " ".join(
        plan["q8_waiver"]["semantic_binding"]["consequences"]
    )


def test_monitor_snapshot_excludes_only_verified_target_group_llama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_proc_stat_cpu", lambda: (1000, 900))
    monkeypatch.setattr(
        runner,
        "_target_group_cpu",
        lambda _leader, _pgid: {
            "members": [101, 102],
            "cpu_ticks": 800,
            "ownership_changed": False,
            "vanished_processes": [],
        },
    )
    monkeypatch.setattr(
        runner,
        "process_ownership",
        lambda: {
            "exact_llama_processes": [
                {"pid": 102, "exe": "/target/llama-bench"},
                {"pid": 303, "exe": "/outside/llama-bench"},
            ],
            "autopilot_processes": [],
        },
    )
    monkeypatch.setattr(
        runner,
        "kfd_ownership",
        lambda: {"users": []},
    )
    monkeypatch.setattr(
        runner,
        "_swap_counters",
        lambda: {"pswpin": 0, "pswpout": 0},
    )
    snapshot = runner.monitor_snapshot(101, 77)
    assert snapshot["contamination"]["exact_llama"] == [
        {"pid": 303, "exe": "/outside/llama-bench"},
    ]


@pytest.mark.parametrize(
    ("error_type", "error_number"),
    [(FileNotFoundError, errno.ENOENT), (ProcessLookupError, errno.ESRCH)],
)
def test_target_group_cpu_records_confirmed_vanished_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[OSError],
    error_number: int,
) -> None:
    (tmp_path / "101").mkdir()
    (tmp_path / "202").mkdir()

    def process_stat(pid: int, proc_root: Path) -> tuple[int, int, int]:
        if pid == 202:
            (proc_root / "202").rmdir()
            raise error_type(error_number, "process exited")
        return 1, 77, 123

    monkeypatch.setattr(runner, "_process_stat", process_stat)

    target = runner._target_group_cpu(101, 77, tmp_path)

    assert target == {
        "members": [101],
        "cpu_ticks": 123,
        "ownership_changed": False,
        "vanished_processes": [
            {"pid": 202, "errno": error_number, "error": repr(error_type(error_number, "process exited"))}
        ],
    }


def test_target_group_cpu_rejects_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "101").mkdir()
    (tmp_path / "202").mkdir()

    def process_stat(pid: int, _proc_root: Path) -> tuple[int, int, int]:
        if pid == 202:
            raise PermissionError(errno.EACCES, "permission denied")
        return 1, 77, 123

    monkeypatch.setattr(runner, "_process_stat", process_stat)

    with pytest.raises(RuntimeError, match="unable to sample benchmark process ownership") as exc_info:
        runner._target_group_cpu(101, 77, tmp_path)
    assert isinstance(exc_info.value.__cause__, PermissionError)


@pytest.mark.parametrize(
    ("error_type", "error_number"),
    [(FileNotFoundError, errno.ENOENT), (ProcessLookupError, errno.ESRCH)],
)
def test_target_group_cpu_rejects_unconfirmed_vanished_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[OSError],
    error_number: int,
) -> None:
    (tmp_path / "101").mkdir()
    (tmp_path / "202").mkdir()

    def process_stat(pid: int, _proc_root: Path) -> tuple[int, int, int]:
        if pid == 202:
            raise error_type(error_number, "process still present")
        return 1, 77, 123

    monkeypatch.setattr(runner, "_process_stat", process_stat)

    with pytest.raises(RuntimeError, match="unable to sample benchmark process ownership") as exc_info:
        runner._target_group_cpu(101, 77, tmp_path)
    assert isinstance(exc_info.value.__cause__, error_type)


def test_contention_monitor_accepts_confirmed_vanished_process_witness() -> None:
    samples = monitor_series([(9000, 8800)] * 10)
    witness = {"pid": 202, "errno": errno.ENOENT, "error": "FileNotFoundError(2, 'gone')"}
    samples[0]["target"]["vanished_processes"].append(witness)  # type: ignore[index]

    assert runner.validate_monitor_samples(samples)["status"] == "pass"


@pytest.mark.parametrize("witnesses", [{}, None])
def test_contention_monitor_rejects_non_list_vanished_process_witnesses(witnesses: object) -> None:
    samples = monitor_series([(9000, 8800)] * 10)
    samples[0]["target"]["vanished_processes"] = witnesses  # type: ignore[index]

    with pytest.raises(RuntimeError, match="vanished process witnesses must be a list"):
        runner.validate_monitor_samples(samples)


def test_monitor_snapshot_interpolates_cpu_counters_to_target_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_samples = iter([(1000, 900), (1200, 1100)])
    monotonic = iter([0.0, 0.2, 0.2, 0.8, 0.8, 1.0])
    monkeypatch.setattr(runner, "_proc_stat_cpu", lambda: next(cpu_samples))
    monkeypatch.setattr(
        runner,
        "_target_group_cpu",
        lambda _leader, _pgid: {
            "members": [101],
            "cpu_ticks": 950,
            "ownership_changed": False,
            "vanished_processes": [],
        },
    )
    monkeypatch.setattr(
        runner,
        "process_ownership",
        lambda: {"exact_llama_processes": [], "autopilot_processes": []},
    )
    monkeypatch.setattr(runner, "kfd_ownership", lambda: {"users": []})
    monkeypatch.setattr(runner, "_swap_counters", lambda: {"pswpin": 0, "pswpout": 0})
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic))

    snapshot = runner.monitor_snapshot(101, 77)

    assert snapshot["monotonic"] == pytest.approx(0.5)
    assert snapshot["cpu_total_ticks"] == pytest.approx(1100)
    assert snapshot["cpu_busy_ticks"] == pytest.approx(1000)
    assert snapshot["cpu_counter_bracket"]["interpolation_fraction"] == pytest.approx(0.5)
    assert snapshot["cpu_counter_bracket"]["target_scan"] == {
        "started_monotonic": 0.2,
        "finished_monotonic": 0.8,
        "elapsed_s": pytest.approx(0.6),
    }


def test_contention_monitor_retains_interpolated_target_over_busy_as_telemetry() -> None:
    samples = monitor_series([(7900, 8000)] * 10)
    samples[1]["cpu_busy_ticks"] = 7900.5  # type: ignore[index]
    samples[1]["cpu_total_ticks"] = 9600.5  # type: ignore[index]
    samples[1]["cpu_counter_bracket"]["before"]["busy_ticks"] = 7900  # type: ignore[index]
    samples[1]["cpu_counter_bracket"]["after"]["busy_ticks"] = 7901  # type: ignore[index]
    samples[1]["cpu_counter_bracket"]["before"]["total_ticks"] = 9600  # type: ignore[index]
    samples[1]["cpu_counter_bracket"]["after"]["total_ticks"] = 9601  # type: ignore[index]
    result = runner.validate_monitor_samples(samples)
    assert result["intervals"][0]["signed_external_core_equivalents"] < 0


def test_monitor_snapshot_rejects_invalid_counter_sampling_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_samples = iter([(1000, 900), (1200, 1100)])
    monotonic = iter([0.0, 0.2, 0.9, 1.1, 0.8, 1.0])
    monkeypatch.setattr(runner, "_proc_stat_cpu", lambda: next(cpu_samples))
    monkeypatch.setattr(
        runner,
        "_target_group_cpu",
        lambda _leader, _pgid: {
            "members": [101],
            "cpu_ticks": 950,
            "ownership_changed": False,
            "vanished_processes": [],
        },
    )
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic))

    with pytest.raises(RuntimeError, match="sampling order"):
        runner.monitor_snapshot(101, 77)


def test_run_monitored_accepts_normal_terminal_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = 0

    def lifecycle_snapshot(leader_pid: int, pgid: int) -> dict[str, object]:
        nonlocal sequence
        sequence += 1
        target = runner._target_group_cpu(leader_pid, pgid)
        target["cpu_ticks"] = sequence * 8000
        return {
            "monotonic": float(sequence),
            "cpu_total_ticks": sequence * runner.CLOCK_TICKS * 96,
            "cpu_busy_ticks": target["cpu_ticks"],
            "cpu_counter_bracket": {
                "before": {"monotonic": float(sequence) - 0.1, "total_ticks": sequence * runner.CLOCK_TICKS * 96, "busy_ticks": target["cpu_ticks"]},
                "after": {"monotonic": float(sequence) + 0.1, "total_ticks": sequence * runner.CLOCK_TICKS * 96, "busy_ticks": target["cpu_ticks"]},
                "target_scan": {"started_monotonic": float(sequence) - 0.01, "finished_monotonic": float(sequence) + 0.01, "elapsed_s": 0.02},
                "target_monotonic": float(sequence),
                "interpolation_fraction": 0.5,
            },
            "swap": {"pswpin": 0, "pswpout": 0},
            "target": target,
            "contamination": {
                "exact_llama": [],
                "autopilot": [],
                "kfd_users": [],
            },
        }

    monkeypatch.setattr(runner, "monitor_snapshot", lifecycle_snapshot)
    monkeypatch.setattr(runner, "MONITOR_INTERVAL_S", 0.01)
    monkeypatch.setattr(runner, "MIN_SUSTAINED_WINDOW_SECONDS", 1.0)
    completed, monitor = runner.run_monitored(
        ["bash", "-c", "sleep 0.05; printf done"],
        runner.canonical_parent_environment(),
    )
    assert completed.returncode == 0
    assert completed.stdout == "done"
    assert monitor["status"] == "pass"
    assert len(monitor["samples"]) >= 2
    assert monitor["samples"][-1]["target"]["ownership_changed"] is False
    assert completed.args[0] == "bash"


def test_release_manifest_rejects_same_head_rebuilt_binary_or_library() -> None:
    expected = runner.RELEASE_ARTIFACTS["candidate"]
    identity = {
        "binary_identity": {"sha256": expected["llama_bench_sha256"]},
        "shared_library_identity": {
            "libraries": [{"resolved_target": f"/candidate/{name}", "sha256": digest} for name, digest in expected["libraries"].items()],
            "openmp_runtime": dict(runner.LLVM20_OPENMP_IDENTITY),
        },
    }
    runner.validate_release_artifacts({"name": "candidate"}, identity)
    identity["binary_identity"]["sha256"] = "rebuilt"
    with pytest.raises(RuntimeError, match="llama-bench SHA256"):
        runner.validate_release_artifacts({"name": "candidate"}, identity)
    identity["binary_identity"]["sha256"] = expected["llama_bench_sha256"]
    identity["shared_library_identity"]["libraries"][0]["sha256"] = "substituted"
    with pytest.raises(RuntimeError, match="library SHA256"):
        runner.validate_release_artifacts({"name": "candidate"}, identity)


def test_host_attestation_rejects_malformed_timezone_and_artifact_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arms = {
        name: {"actual_branch": spec["expected_branch"], "actual_head": spec["expected_head"],
               "binary_identity": {"sha256": runner.RELEASE_ARTIFACTS[name]["llama_bench_sha256"]}}
        for name, spec in (("production", runner.arm_spec("production")), ("candidate", runner.arm_spec("candidate")))
    }
    value = {"schema": runner.HOST_ATTESTATION_SCHEMA, "protocol": "P-BENCH-PREFILL-1", "status": "pass",
             "created_at": runner.utc_now(),
             "candidate": {"branch": arms["candidate"]["actual_branch"], "head": arms["candidate"]["actual_head"]},
             "production": {"branch": arms["production"]["actual_branch"], "head": arms["production"]["actual_head"]},
             "artifact_binding": {name: {"binary_sha256": arms[name]["binary_identity"]["sha256"]} for name in arms}}
    path = tmp_path / "host.json"
    monkeypatch.setattr(runner, "host_snapshot", valid_host)
    path.write_text(json.dumps(value))
    assert runner.validate_host_attestation(
        path,
        arms,
        require_fresh=True,
    )["verified_by_runner"] is True
    value["created_at"] = "2026-07-24T00:00:00Z"
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="UTC"):
        runner.validate_host_attestation(path, arms, require_fresh=True)
    value["created_at"] = runner.utc_now()
    value["artifact_binding"]["candidate"]["binary_sha256"] = "wrong"
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="artifact binding"):
        runner.validate_host_attestation(path, arms, require_fresh=True)
    value["artifact_binding"]["candidate"]["binary_sha256"] = arms["candidate"]["binary_identity"]["sha256"]
    value["created_at"] = "2020-01-01T00:00:00+00:00"
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="stale"):
        runner.validate_host_attestation(path, arms, require_fresh=True)
    assert runner.validate_host_attestation(
        path,
        arms,
        require_fresh=False,
    )["verified_by_runner"] is True


def host_attestation_arms() -> dict[str, dict[str, object]]:
    return {
        name: {
            "name": name,
            "actual_branch": spec["expected_branch"],
            "actual_head": spec["expected_head"],
            "binary_identity": {
                "sha256": runner.RELEASE_ARTIFACTS[name]["llama_bench_sha256"],
            },
        }
        for name, spec in (
            ("production", runner.arm_spec("production")),
            ("candidate", runner.arm_spec("candidate")),
        )
    }


def test_write_host_attestation_captures_current_arm_bindings_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arms = host_attestation_arms()
    target = tmp_path / "new" / "host.json"
    monkeypatch.setattr(runner, "collect_arm_identity", lambda spec: arms[spec["name"]])
    monkeypatch.setattr(runner, "host_snapshot", valid_host)
    monkeypatch.setattr(
        runner,
        "fresh_output_dir",
        lambda _requested: (_ for _ in ()).throw(AssertionError("normal output mode ran")),
    )

    assert runner.main(["--write-host-attestation", str(target)]) == 0

    value = json.loads(target.read_text())
    assert set(value) == {
        "schema", "protocol", "status", "created_at", "candidate", "production", "artifact_binding",
    }
    assert value["schema"] == runner.HOST_ATTESTATION_SCHEMA
    assert value["protocol"] == "P-BENCH-PREFILL-1"
    assert value["status"] == "pass"
    assert runner.validate_host_attestation(target, arms, require_fresh=True)["verified_by_runner"] is True
    for name in ("candidate", "production"):
        assert value[name] == {
            "branch": arms[name]["actual_branch"],
            "head": arms[name]["actual_head"],
        }
        assert value["artifact_binding"][name] == {
            "binary_sha256": arms[name]["binary_identity"]["sha256"],
        }
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["mode"] == "host_attestation_written"
    assert emitted["promotion_decision"] is False
    assert not (tmp_path / "plan.json").exists()


def test_write_host_attestation_rejects_dirty_host_and_existing_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arms = host_attestation_arms()
    target = tmp_path / "host.json"
    monkeypatch.setattr(runner, "collect_arm_identity", lambda spec: arms[spec["name"]])
    dirty = valid_host()
    dirty["governors"] = {"cpu0": "powersave"}
    monkeypatch.setattr(runner, "host_snapshot", lambda: dirty)

    with pytest.raises(RuntimeError, match="strict host check failed"):
        runner.write_host_attestation(target)
    assert not target.exists()

    target.write_text("existing artifact\n")
    monkeypatch.setattr(runner, "host_snapshot", valid_host)
    with pytest.raises(RuntimeError, match="already exists; refusing overwrite"):
        runner.write_host_attestation(target)
    assert target.read_text() == "existing artifact\n"


def test_write_host_attestation_mode_is_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args([
            "--execute", "--write-host-attestation", str(tmp_path / "host.json"),
        ])
    with pytest.raises(SystemExit):
        runner.parse_args([
            "--write-host-attestation", str(tmp_path / "host.json"),
            "--output-dir", str(tmp_path / "output"),
        ])
    with pytest.raises(SystemExit):
        runner.parse_args([
            "--write-host-attestation", str(tmp_path / "host.json"),
            "--host-attestation-path", str(tmp_path / "other.json"),
        ])


def test_default_main_writes_preparation_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "fresh"
    assert runner.main(["--output-dir", str(output)]) == 0
    plan = json.loads((output / "plan.json").read_text())
    emitted = json.loads(capsys.readouterr().out)
    assert len(plan["arm_runs"]) == 28
    assert len(plan["pairs"]) == 14
    assert emitted["promotion_decision"] is False
    assert "status" not in emitted


def test_execute_failure_writes_durable_nonpromotion_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        runner,
        "execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    output = tmp_path / "failure"
    attestations = make_attestations(tmp_path)
    argv = [
        "--output-dir",
        str(output),
        "--execute",
        "--host-attestation-path",
        str(attestations["host"]),
        "--correctness-attestation-path",
        str(attestations["correctness"]),
        "--coherence-attestation-path",
        str(attestations["coherence"]),
        "--numerical-safety-attestation-path",
        str(attestations["numerical_safety"]),
    ]
    assert runner.main(argv) == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["throughput_status"] == "invalid"
    assert summary["promotion_decision"] is False
    assert json.loads(capsys.readouterr().out)["promotion_decision"] is False
