"""Synthetic provenance mutation controls; no native run or model claim."""

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.benchmark import autokernel_profile_beliefs as finalizer
from scripts.benchmark import nemo_native_join as join


def raw_fixture(count=32, label="M"):
    run = {
        "pid": 101,
        "pid_birth_ticks": 202,
        "host_boot_id": "MOCK_ONLY_BOOT",
        "started_at": "2026-10-30T01:00:00Z",
        "finished_at": "2026-10-30T01:01:00Z",
        "status": "capture-complete-unvalidated",
        "trip": None,
        "server_core_limit": [0, 0],
        "teardown": {"verified_dead": True, "returncode": 0},
        "marks": [
            {
                "label": label,
                "response_request_id": "MOCK_ONLY_RESPONSE",
                "clock": "clock_gettime(CLOCK_MONOTONIC)",
                "start_monotonic_ns": 500,
                "end_monotonic_ns": 10000,
                "predicted_n": count,
                "err": None,
            }
        ],
        "during_request_gpu_proofs": [
            {
                "pid": 101,
                "kfd_present": True,
                "vram_bytes": 123,
                "hip_library_maps": ["MOCK_ONLY/libggml-hip.so"],
                "sample_begin_monotonic_ns": 501 + i,
                "sample_end_monotonic_ns": 502 + i,
            }
            for i in range(5)
        ],
    }
    native, markers, hip, kernels = [], [], [], []
    for i in range(count):
        call = i + 1
        start = 1000 + i * 100
        base = {
            "schema": "epyc.native_profile.v1",
            "pid": 101,
            "process_start_ticks": 202,
            "tid": 101,
        }
        slot = {
            "slot": 0,
            "task": 7,
            "request_id": "MOCK_ONLY_RESPONSE",
            "server_phase": "prompt" if i == 0 else "generation",
            "generated_before": i,
            "draft_count": 0,
            "target_ctx": "0x11",
            "drafter_ctx": "0x22",
        }
        native.extend(
            [
                {
                    **base,
                    "event": "decode_begin",
                    "api": "decode",
                    "call_id": call,
                    "monotonic_ns": start,
                    "ctx": "0x11",
                    "model": "0x33",
                    "scope_id": call,
                    "slots": [slot],
                    "n_tokens": 1,
                    "rows": [{"index": 0, "pos": i, "token": 100 + i, "seq_ids": [0]}],
                },
                {
                    **base,
                    "event": "decode_end",
                    "api": "decode",
                    "call_id": call,
                    "monotonic_ns": start + 10,
                    "rc": 0,
                },
                {
                    **base,
                    "event": "sampled_token",
                    "monotonic_ns": start + 17,
                    "slot": 0,
                    "task": 7,
                    "request_id": "MOCK_ONLY_RESPONSE",
                    "target_ctx": "0x11",
                    "ordinal": call,
                    "token": 101 + i,
                },
            ]
        )
        markers.append(
            {
                "Domain": "MARKER_CORE_API",
                "Function": f"epyc.decode.{call}",
                "Process_Id": "101",
                "Thread_Id": "101",
                "Correlation_Id": str(999 + call),
                "Start_Timestamp": str(start - 1),
                "End_Timestamp": str(start + 11),
            }
        )
        hip.append(
            {
                "Domain": "HIP_RUNTIME_API",
                "Function": "hipGraphLaunch",
                "Process_Id": "101",
                "Thread_Id": "101",
                "Correlation_Id": str(call),
                "Start_Timestamp": str(start + 2),
                "End_Timestamp": str(start + 3),
            }
        )
        kernels.append(
            {
                "Kind": "KERNEL_DISPATCH",
                "Agent_Id": "4",
                "Queue_Id": "1",
                "Kernel_Id": str(call),
                "Kernel_Name": "MOCK_ONLY_KERNEL",
                "Correlation_Id": str(call),
                "Start_Timestamp": str(start + 5),
                "End_Timestamp": str(start + 15),
            }
        )
    return run, native, markers, hip, kernels


def test_exact_native_graph_join_preserves_async_dispatch_and_real_token_rows():
    report = join.join_raw(*raw_fixture())
    assert report["status"] == "complete"
    assert len(report["requests"][0]["tokens"]) == 32
    assert report["calls"][0]["dispatches"][0]["End_Timestamp"] == "1015"
    assert report["calls"][0]["end"]["monotonic_ns"] == 1010
    assert report["calls"][0]["marker"]["Correlation_Id"] != "1"
    values = join.measurement_values(report)
    assert values[0]["metric"] == "bolt_decode_step_busy_ms"
    assert values[0]["value_ms"] == pytest.approx(0.00001)
    assert values[0]["reps"] == 1 and values[0]["observed_dependent_units"] == 31


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_end",
        "duplicate_end",
        "missing_marker",
        "duplicate_marker",
        "wrong_tid",
        "scope_zero",
        "wrong_context",
        "wrong_request",
        "wrong_slot",
        "wrong_token",
        "null_position",
        "multi_sequence",
        "missing_dispatch",
        "duplicate_dispatch",
        "foreign_correlation",
        "foreign_hip_pid",
        "foreign_token_ordinal",
    ],
)
def test_semantic_ambiguity_is_unknown_not_count_based_success(mutation):
    raw = list(copy.deepcopy(raw_fixture()))
    run, native, markers, hip, kernels = raw
    if mutation == "missing_end":
        native.pop(1)
    elif mutation == "duplicate_end":
        native.append(copy.deepcopy(native[1]))
    elif mutation == "missing_marker":
        markers.pop(0)
    elif mutation == "duplicate_marker":
        markers.append(copy.deepcopy(markers[0]))
    elif mutation == "wrong_tid":
        native[0]["tid"] = 999
    elif mutation == "scope_zero":
        native[0]["scope_id"] = 0
    elif mutation == "wrong_context":
        native[0]["ctx"] = "0xDEAD"
    elif mutation == "wrong_request":
        native[0]["slots"][0]["request_id"] = "other"
    elif mutation == "wrong_slot":
        native[0]["slots"][0]["slot"] = 1
    elif mutation == "wrong_token":
        native[2]["token"] = 999
    elif mutation == "null_position":
        native[3]["rows"][0]["pos"] = None
    elif mutation == "multi_sequence":
        native[3]["rows"][0]["seq_ids"] = [0, 1]
    elif mutation == "missing_dispatch":
        kernels.pop(0)
    elif mutation == "duplicate_dispatch":
        kernels.append(copy.deepcopy(kernels[0]))
    elif mutation == "foreign_correlation":
        kernels[0]["Correlation_Id"] = "9999"
    elif mutation == "foreign_hip_pid":
        hip[0]["Process_Id"] = "999"
    elif mutation == "foreign_token_ordinal":
        native[2]["ordinal"] = 2
    report = join.join_raw(*raw)
    assert report["status"] == "UNKNOWN"
    with pytest.raises(join.JoinError):
        join.measurement_values(report)


@pytest.mark.parametrize(
    "field,value", [("pid", 999), ("process_start_ticks", 999), ("api", "encode")]
)
def test_foreign_native_identity_or_api_is_refused(field, value):
    raw = list(copy.deepcopy(raw_fixture()))
    raw[1][0][field] = value
    with pytest.raises(join.JoinError):
        join.join_raw(*raw)


def test_wrong_clock_never_gets_a_guessed_offset():
    raw = list(copy.deepcopy(raw_fixture()))
    raw[0]["marks"][0]["clock"] = "CLOCK_MONOTONIC_RAW"
    with pytest.raises(join.JoinError, match="clock"):
        join.join_raw(*raw)


def test_overlapping_graph_dispatches_union_without_losing_summed_work():
    raw = list(copy.deepcopy(raw_fixture()))
    raw[-1].append(
        {
            **raw[-1][0],
            "Kernel_Id": "1234",
            "Queue_Id": "2",
            "Start_Timestamp": "1010",
            "End_Timestamp": "1020",
        }
    )
    report = join.join_raw(*raw)
    assert report["status"] == "complete"
    assert report["calls"][0]["busy_ns"] == 15
    assert report["calls"][0]["kernel_sum_ns"] == 20
    assert len(report["calls"][0]["dispatches"]) == 2


@pytest.fixture(autouse=True)
def restore_authority(monkeypatch):
    # External approved artifact authority is replaced only in these hermetic
    # unit controls. Production has no MOCK_ONLY mode or alternate authority.
    monkeypatch.setattr(join, "NATIVE_AUTHORITY", dict(join.NATIVE_AUTHORITY))
    monkeypatch.setattr(join, "BOLT_MODEL", join.BOLT_MODEL)


def add_mock_contract(tmp_path, manifest, run):
    roles = set(join.NATIVE_AUTHORITY) | {
        "producer",
        "claim_module",
        "claim_schemas",
        "worktree_module",
        "contract_module",
        "interpreter",
        "launcher",
        "capture_source_manifest",
        "module_closure",
        "model",
        "template",
    }
    artifacts = {}
    for role in roles:
        path = tmp_path / ("MOCK_ONLY_" + role)
        path.write_text("MOCK_ONLY synthetic " + role)
        artifacts[role] = join.artifact_identity(path)
    join.BOLT_MODEL = artifacts["model"]["path"]
    artifacts["template"] = artifacts["model"]
    modules = {
        artifacts[k]["path"]: artifacts[k]["sha256"]
        for k in ("claim_module", "claim_schemas", "worktree_module")
    }
    for i in range(9):
        path = tmp_path / ("MOCK_ONLY_module_" + str(i))
        path.write_text("MOCK_ONLY_MODULE")
        modules[str(path)] = join.artifact_identity(path)["sha256"]
    Path(artifacts["module_closure"]["path"]).write_text(json.dumps(modules))
    artifacts["module_closure"] = join.artifact_identity(artifacts["module_closure"]["path"])
    for role in join.NATIVE_AUTHORITY:
        join.NATIVE_AUTHORITY[role] = artifacts[role]["sha256"]
    server = [
        artifacts["server"]["path"],
        "-m",
        join.BOLT_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "19111",
        "--batch-size",
        "2048",
        "--ubatch-size",
        "512",
        "--ctx-size",
        "16384",
        "--parallel",
        "1",
        "-ngl",
        "999",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--flash-attn",
        "on",
        "--jinja",
        "-t",
        "8",
    ]
    env = {
        "set": {"LLAMA_NATIVE_ROCTX_LIBRARY": artifacts["roctx"]["path"]},
        "unset": ["GGML_CUDA_DISABLE_GRAPHS"],
    }
    argv = [
        artifacts["profiler"]["path"],
        "--kernel-trace",
        "--memory-copy-trace",
        "--hip-trace",
        "--marker-trace",
        "--stats",
        "--",
        *server,
    ]
    prefix = (
        ["env"]
        + [part for name in env["unset"] for part in ("-u", name)]
        + [key + "=" + value for key, value in env["set"].items()]
    )
    argv = prefix + [
        "numactl",
        "--membind=3",
        "--",
        "taskset",
        "-c",
        "184-191",
        artifacts["interpreter"]["path"],
        artifacts["profiler"]["path"],
        "--kernel-trace",
        "--memory-copy-trace",
        "--hip-trace",
        "--marker-trace",
        "--stats",
        "--output-format",
        "csv",
        "-d",
        str(tmp_path / "trace"),
        "-o",
        "prof",
        "--",
        *server,
    ]
    contract = {
        "schema": "epyc.autokernel.nemo_native_contract.v1",
        "purpose": "prospective_scored_profile",
        "declared_at": manifest["declared_at"],
        "native_commit": "b1e820a0ad8bc2192fc38c3c1beda7abf3a28ecf",
        "native_base": "ffc1bac82eeca6f9099e1ccd9ba49703c460a115",
        "artifacts": artifacts,
        "analysis_sources": manifest["analysis_sources"],
        "pass": "B",
        "campaign_id": manifest["campaign_id"],
        "device_id": "mi210_0",
        "server_argv": server,
        "argv": argv,
        "env": env,
        "phases": join.phase_payloads("B"),
        "template_mode": "embedded_gguf",
        "cpu_list": "184-191",
        "trace_dir": str(tmp_path / "trace"),
    }
    contract["analysis_artifacts"] = {
        name: join.artifact_identity(Path(join.__file__).with_name(name))
        for name in manifest["analysis_sources"]
    }
    artifacts["contract_module"] = contract["analysis_artifacts"]["nemo_native_join.py"]
    manifest["capture_contract"] = contract
    claim = manifest["device_claim_open"]
    claim.update(
        device_id="mi210_0",
        state="held",
        lock_path="/mnt/raid0/llm/tmp/gpu_device.mi210_0.lock",
        holder_pid=303,
        holder_start_ticks=404,
        holder_boot_id=run["host_boot_id"],
        purpose="MOCK_ONLY control",
        acquired_at="2026-10-30T00:58:00Z",
        expires_at="2026-10-30T02:00:00Z",
    )
    manifest["device_claim_released"] = {**claim, "released_at": "2026-10-30T01:02:00Z"}
    manifest["agent_binding"]["device_id"] = "mi210_0"
    run["capture_owner"] = {
        k: claim[k]
        for k in ("holder_pid", "holder_start_ticks", "holder_boot_id", "claim_id", "device_id")
    }
    run.update(
        capture_contract_sha256=join.canonical_hash(contract),
        source_time_artifacts=artifacts,
        server_argv=server,
        argv=argv,
        env=env,
        drained_at=run["finished_at"],
    )
    run["marks"].insert(
        0,
        {
            "label": "warm",
            "response_request_id": "MOCK_ONLY_WARM",
            "clock": "clock_gettime(CLOCK_MONOTONIC)",
            "start_monotonic_ns": 100,
            "end_monotonic_ns": 200,
            "predicted_n": 0,
            "err": None,
        },
    )
    for mark in run["marks"]:
        mark.update(
            **contract["phases"][mark["label"]],
            capture_contract_sha256=run["capture_contract_sha256"],
            runtime_server_argv=server,
            started_at="2026-10-30T01:00:01Z"
            if mark["label"] == "warm"
            else "2026-10-30T01:00:10Z",
            ended_at="2026-10-30T01:00:02Z" if mark["label"] == "warm" else "2026-10-30T01:00:50Z",
        )
    for mark in run["marks"]:
        mark["runtime_artifacts"] = {
            role: join.artifact_identity(str(Path(artifacts[role]["path"]).resolve()))
            for role in (
                "server",
                "libllama",
                "libggml_hip",
                "libggml",
                "libggml_base",
                "libggml_cpu",
                "roctx",
                "profiler_sdk",
            )
        }
    path = tmp_path / manifest["files"]["run"]["path"]
    path.write_text(json.dumps(run))
    manifest["files"]["run"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def write_capture(tmp_path, *, label="M"):
    run, native, markers, hip, kernels = raw_fixture(label=label)
    agents = [
        {
            "Node_Id": "4",
            "Agent_Type": "GPU",
            "Drm_Render_Minor": "128",
            "Location_Id": "17152",
            "Gpu_Id": "57300",
        }
    ]
    paths = {}
    for role, value in zip(join.ROLES, (run, native, markers, hip, kernels, agents)):
        p = tmp_path / (
            role + (".json" if role == "run" else ".jsonl" if role == "native" else ".csv")
        )
        if role == "run":
            p.write_text(json.dumps(value))
        elif role == "native":
            p.write_text("\n".join(json.dumps(x) for x in value) + "\n")
        else:
            with p.open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(value[0]))
                writer.writeheader()
                writer.writerows(value)
        paths[role] = {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    claim = {
        "schema": "epyc.autokernel.device_claim_receipt.v1",
        "campaign_id": "MOCK_ONLY_CAMPAIGN",
        "claim_id": "MOCK_ONLY_CLAIM",
        "device_id": "MOCK_ONLY_DEVICE",
    }
    manifest = {
        "schema": join.CAPTURE_SCHEMA,
        "purpose": "prospective_scored_profile",
        "campaign_id": "MOCK_ONLY_CAMPAIGN",
        "status": "complete",
        "declared_at": "2026-10-30T00:59:00Z",
        "started_at": run["started_at"],
        "ended_at": run["finished_at"],
        "files": paths,
        "device_claim_open": claim,
        "device_claim_released": {**claim, "released_at": run["finished_at"]},
        "agent_binding": {
            "device_id": "MOCK_ONLY_DEVICE",
            "agent_id": 4,
            "drm_render_minor": 128,
            "location_id": 17152,
            "gpu_id": 57300,
        },
        "analysis_sources": {
            "nemo_native_join.py": finalizer.sha256_file(join.__file__),
            "autokernel_profile_beliefs.py": finalizer.sha256_file(finalizer.__file__),
        },
    }
    add_mock_contract(tmp_path, manifest, run)
    return manifest


def test_actual_existing_finalizer_emits_single_request_attribution_and_raw_bindings(tmp_path):
    manifest = write_capture(tmp_path)
    result = finalizer.finalize_nemo_native(
        manifest,
        capture_root=tmp_path,
        source_locator=str(tmp_path / "capture.json"),
        source_sha256=join.canonical_hash(manifest),
    )
    assert result["schema"] == finalizer.SCHEMA
    assert result["receipt_sha256"] == finalizer.receipt_sha256(result)
    assert len(result["belief_measurements"]) == 1
    row = result["belief_measurements"][0]
    assert row["metric"] == "bolt_decode_step_busy_ms" and row["metric_direction"] == "lower_better"
    assert row["reps"] == 1 and "no repeats or noise floor" in row["reps_basis"]
    assert result["native_join"]["raw_readset"] == {
        k: v["sha256"] for k, v in manifest["files"].items()
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "raw_tamper",
        "canary",
        "late_declaration",
        "bad_claim",
        "bad_agent",
        "no_residency",
        "not_drained",
        "unqualified_source",
        "source_pin",
        "path_escape",
    ],
)
def test_finalizer_refuses_incomplete_or_relabelled_capture(tmp_path, mutation):
    manifest = write_capture(tmp_path)
    if mutation == "raw_tamper":
        (tmp_path / "native.jsonl").write_text("{}\n")
    elif mutation == "canary":
        manifest["purpose"] = "r11"
    elif mutation == "late_declaration":
        manifest["declared_at"] = "2026-10-30T02:00:00Z"
    elif mutation == "bad_claim":
        manifest["device_claim_released"]["device_id"] = "other"
    elif mutation == "bad_agent":
        manifest["agent_binding"]["gpu_id"] = 0
    elif mutation == "source_pin":
        manifest["analysis_sources"]["nemo_native_join.py"] = "0" * 64
    elif mutation == "path_escape":
        manifest["files"]["run"]["path"] = "/etc/passwd"
    else:
        p = tmp_path / "run.json"
        run = json.loads(p.read_text())
        if mutation == "no_residency":
            run["during_request_gpu_proofs"] = []
        elif mutation == "not_drained":
            run["teardown"]["verified_dead"] = False
        elif mutation == "unqualified_source":
            run["status"] = "partial"
        p.write_text(json.dumps(run))
        manifest["files"]["run"]["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises((join.JoinError, finalizer.ProfileBeliefError)):
        finalizer.finalize_nemo_native(
            manifest,
            capture_root=tmp_path,
            source_locator=str(tmp_path / "capture.json"),
            source_sha256=join.canonical_hash(manifest),
        )


def speculative_fixture():
    run, native, markers, hip, kernels = raw_fixture(count=20, label="S")
    events = []
    for i in range(10):
        begin, end, token = copy.deepcopy(native[3 * i : 3 * i + 3])
        begin["slots"][0].update(server_phase="generation", generated_before=2 * i)
        begin["n_tokens"] = 2
        begin["rows"] = [
            {"index": j, "pos": 2 * i + j, "token": 100 + 2 * i + j, "seq_ids": [0]}
            for j in range(2)
        ]
        token.update(ordinal=2 * i + 1, token=101 + 2 * i)
        second = {
            **token,
            "ordinal": 2 * i + 2,
            "token": 102 + 2 * i,
            "monotonic_ns": token["monotonic_ns"] + 1,
        }
        accepted = {k: v for k, v in token.items() if k not in ("ordinal", "token")}
        accepted.update(
            event="spec_accept",
            generated_before=2 * i,
            accepted_tokens_including_bonus=[101 + 2 * i, 102 + 2 * i],
            tested_draft_count=1,
            monotonic_ns=end["monotonic_ns"] + 1,
        )
        events.extend([begin, end, accepted, token, second])
    return run, events, markers[:10], hip[:10], kernels[:10]


def test_speculative_identity_is_explicit_and_joint_time_stays_per_round():
    report = join.join_raw(*speculative_fixture())
    assert report["status"] == "complete"
    assert len(report["requests"][0]["rounds"]) == 10
    assert all(r["per_token_busy_ns"] is None for r in report["requests"][0]["rounds"])
    metric = join.measurement_values(report)[0]
    assert metric["metric"] == "dflash_round_busy_ms" and metric["reps"] == 1
    assert metric["observed_dependent_units"] == 10


def test_accepted_but_unemitted_bonus_tail_is_preserved_not_invented():
    raw = list(speculative_fixture())
    raw[0]["marks"][0]["predicted_n"] = 19
    raw[1].pop()
    report = join.join_raw(*raw)
    assert report["status"] == "complete"
    last = report["requests"][0]["rounds"][-1]
    assert len(last["event"]["accepted_tokens_including_bonus"]) == 2
    assert len(last["emitted"]) == 1 and last["per_token_busy_ns"] is None


@pytest.mark.parametrize("mutation", ["token", "missing", "duplicate", "context", "thread"])
def test_speculative_ambiguous_vectors_are_unknown(mutation):
    raw = list(speculative_fixture())
    event = raw[1][2]
    if mutation == "token":
        event["accepted_tokens_including_bonus"][0] = 999
    elif mutation == "missing":
        raw[1].pop(2)
    elif mutation == "duplicate":
        raw[1].append(copy.deepcopy(event))
    elif mutation == "context":
        event["target_ctx"] = "0xDEAD"
    else:
        event["tid"] = 999
    report = join.join_raw(*raw)
    assert report["status"] == "UNKNOWN"
    with pytest.raises(join.JoinError):
        join.measurement_values(report)


def encode_fixture():
    run, native, markers, hip, kernels = raw_fixture()
    for api, call_id, start in (("encode", 100, 1030), ("decode", 101, 1050)):
        base = copy.deepcopy(native[0])
        base.update(
            event=api + "_begin",
            api=api,
            call_id=call_id,
            monotonic_ns=start,
            ctx="0x22",
            scope_id=77,
            n_tokens=2,
        )
        base["slots"][0].update(server_phase="generation", generated_before=1)
        base["rows"] = [
            {
                "index": i,
                "pos": None if api == "encode" else i + 1,
                "token": None,
                "seq_ids": None if api == "encode" else [0],
            }
            for i in range(2)
        ]
        end = {k: v for k, v in native[1].items()}
        end.update(event=api + "_end", api=api, call_id=call_id, monotonic_ns=start + 10)
        native.extend([base, end])
        markers.append(
            {
                **markers[0],
                "Function": f"epyc.{api}.{call_id}",
                "Correlation_Id": str(999 + call_id),
                "Start_Timestamp": str(start - 1),
                "End_Timestamp": str(start + 11),
            }
        )
        hip.append(
            {
                **hip[0],
                "Correlation_Id": str(call_id),
                "Start_Timestamp": str(start + 2),
                "End_Timestamp": str(start + 3),
            }
        )
        kernels.append(
            {
                **kernels[0],
                "Kernel_Id": str(call_id),
                "Correlation_Id": str(call_id),
                "Start_Timestamp": str(start + 5),
                "End_Timestamp": str(start + 15),
            }
        )
    return run, native, markers, hip, kernels


def test_encode_null_positions_are_not_tokens_and_unique_injection_binds():
    report = join.join_raw(*encode_fixture())
    assert report["status"] == "complete"
    encode = next(c for c in report["calls"] if c["key"][3] == "encode")
    assert all(
        r["pos"] is None and r["token"] is None and r["seq_ids"] is None
        for r in encode["begin"]["rows"]
    )
    assert encode["injection_call_key"][-1] == 101 and encode["emitted"] == []
    assert len(report["requests"][0]["tokens"]) == 32


@pytest.mark.parametrize(
    "mutation", ["failed_encode", "failed_inject", "scope", "context", "sequence"]
)
def test_missing_failed_or_cross_bound_encode_injection_is_unknown(mutation):
    raw = list(encode_fixture())
    if mutation == "failed_encode":
        raw[1][-3]["rc"] = -1
    elif mutation == "failed_inject":
        raw[1][-1]["rc"] = -1
    elif mutation == "scope":
        raw[1][-2]["scope_id"] = 999
    elif mutation == "context":
        raw[1][-2]["ctx"] = "0xDEAD"
    else:
        raw[1][-2]["rows"][0]["seq_ids"] = [0, 1]
    assert join.join_raw(*raw)["status"] == "UNKNOWN"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_contract",
        "wrong_model",
        "wrong_build",
        "wrong_template",
        "wrong_argv",
        "wrong_env",
        "wrong_payload",
        "claim_before_request",
        "claim_expired",
        "claim_holder",
        "claim_device",
        "late_source",
        "module_drift",
    ],
)
def test_prospective_identity_mutations_refuse_before_numeric_rows(tmp_path, mutation):
    manifest = write_capture(tmp_path)
    run = json.loads((tmp_path / "run.json").read_text())
    c = manifest["capture_contract"]
    if mutation == "missing_contract":
        manifest.pop("capture_contract")
    elif mutation == "wrong_model":
        c["artifacts"]["model"]["path"] = c["artifacts"]["server"]["path"]
    elif mutation == "wrong_build":
        c["artifacts"]["server"]["sha256"] = "0" * 64
    elif mutation == "wrong_template":
        c["artifacts"]["template"] = c["artifacts"]["producer"]
    elif mutation == "wrong_argv":
        run["argv"] = ["different", *run["argv"]]
    elif mutation == "wrong_env":
        run["env"] = {"set": {"GGML_CUDA_DISABLE_GRAPHS": "1"}, "unset": []}
    elif mutation == "wrong_payload":
        run["marks"][-1]["payload"]["max_tokens"] = 32
    elif mutation == "claim_before_request":
        manifest["device_claim_released"]["released_at"] = "2026-10-30T01:00:20Z"
    elif mutation == "claim_expired":
        for phase in ("device_claim_open", "device_claim_released"):
            manifest[phase]["expires_at"] = "2026-10-30T01:00:20Z"
    elif mutation == "claim_holder":
        manifest["device_claim_released"]["holder_start_ticks"] += 1
    elif mutation == "claim_device":
        manifest["agent_binding"]["device_id"] = "another-device"
    elif mutation == "late_source":
        c["declared_at"] = "2026-10-30T03:00:00Z"
    else:
        Path(c["artifacts"]["claim_module"]["path"]).write_text("mutated module")
    # Rehash the run and source-time contract to prove semantic checks, not only
    # sidecar SHA mismatch, stop wrong-model/build/template/claim rewrites.
    if "capture_contract" in manifest:
        run["capture_contract_sha256"] = join.canonical_hash(c)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))
    manifest["files"]["run"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises((join.JoinError, finalizer.ProfileBeliefError)):
        finalizer.finalize_nemo_native(
            manifest,
            capture_root=tmp_path,
            source_locator=str(tmp_path / "capture.json"),
            source_sha256=join.canonical_hash(manifest),
        )
