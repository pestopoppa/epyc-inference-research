"""Offline tests for affinity_preflight.py cell-manifest mode (E5 batched-decode gate).

All live-system readers (_pid_on_port/_thread_union/_memory_placement/_llama_processes)
are monkeypatched with synthetic fixtures — no real port, process, or /proc is touched.
Also regression-tests that the legacy --roles mode behavior is unchanged.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/server/affinity_preflight.py"
spec = importlib.util.spec_from_file_location("affinity_preflight", MODULE_PATH)
assert spec is not None and spec.loader is not None
affinity_preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(affinity_preflight)

Q0A = "0-23,96-119"
Q0B = "24-47,120-143"
HALF1 = "48-95,144-191"

MANIFEST = {
    "schema_version": "e5-cell-manifest/1",
    "protocol_id": "P-BENCH-3",
    "cell_id": "qwen36_q8_0-C2-np16",
    "model_key": "qwen36_q8_0",
    "instances": [
        {"cpu_list": Q0A, "port": 19080, "threads": 48, "numactl_policy": "none"},
        {"cpu_list": Q0B, "port": 19081, "threads": 48, "numactl_policy": "none"},
    ],
}


def _cpus(s: str) -> set[int]:
    return affinity_preflight._parse_cpulist(s)


def _bind_canonical_lane_fold_without_a_kernel_store(monkeypatch) -> None:
    """Load the real pure fold helper without resolving a production GPU binary.

    The lane module resolves its GPU store at import time. These tests exercise
    only the canonical CPU-set fold, so provide the minimal lane constants and
    disabled-flag function needed to import the real lease module; no launch
    path or fake kernel asset is exercised.
    """
    lane = types.ModuleType("scripts.server.gpu_shadow_lane")
    lane.GpuShadowLaneDisabled = type("GpuShadowLaneDisabled", (RuntimeError,), {})
    lane.LANE_DEVICE = "ROCm0"
    lane.LANE_HOST_CPUSET = "184-191"
    lane.LANE_NAME = "gpu_shadow_lane"
    lane.lane_enabled = lambda *_args, **_kwargs: False
    monkeypatch.setitem(sys.modules, "scripts.server.gpu_shadow_lane", lane)

    name = "_ssu_f13_test_gpu_shadow_lane_lease"
    path = Path(__file__).resolve().parents[2] / "scripts/server/gpu_shadow_lane_lease.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    lease = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, lease)
    spec.loader.exec_module(lease)
    monkeypatch.setattr(
        affinity_preflight,
        "_fold_cpus_to_physical",
        lease.fold_cpus_to_physical,
    )


def _patch_live(monkeypatch, *, pids=None, unions=None, procs=None):
    """Replace every live-system reader with synthetic fixtures.

    pids: {port(int): pid(str)} for _pid_on_port; unions: {pid(str): set[int]}
    for _thread_union; procs: [(pid, args)] for _llama_processes.
    Returns a call-recorder dict.
    """
    pids = pids or {}
    unions = unions or {}
    calls: dict = {"thread_union": []}
    monkeypatch.setattr(affinity_preflight, "_pid_on_port", lambda port: pids.get(int(port)))

    def fake_union(pid):
        calls["thread_union"].append(str(pid))
        return set(unions.get(str(pid), set()))

    monkeypatch.setattr(affinity_preflight, "_thread_union", fake_union)
    monkeypatch.setattr(affinity_preflight, "_llama_processes", lambda: list(procs or []))
    monkeypatch.setattr(
        affinity_preflight, "_memory_placement",
        lambda pid, nodes, thr: {"checked": False, "required": False, "match": None, "note": "mocked"},
    )
    return calls


def _write_manifest(tmp_path: Path, manifest: dict | None = None) -> Path:
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(manifest if manifest is not None else MANIFEST))
    return path


def _run_main(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["affinity_preflight.py", *argv])
    return affinity_preflight.main()


# ---------------------------------------------------------------- cell mode: pass


def test_manifest_exact_match_passes(monkeypatch, tmp_path, capsys) -> None:
    manifest_path = _write_manifest(tmp_path)
    out = tmp_path / "artifact.json"
    _patch_live(
        monkeypatch,
        pids={19080: "101", 19081: "102"},
        unions={"101": _cpus(Q0A), "102": _cpus(Q0B)},
    )
    rc = _run_main(monkeypatch, ["--cell-manifest", str(manifest_path), "--output", str(out)])
    assert rc == 0

    artifact = json.loads(out.read_text())
    assert artifact["live_affinity_verified"] is True
    assert artifact["mode"] == "cell"
    assert artifact["manifest_path"] == str(manifest_path)
    assert artifact["cell_id"] == "qwen36_q8_0-C2-np16"
    assert artifact["foreign_llama_overlaps"] == []
    assert len(artifact["instances"]) == 2
    entry = artifact["instances"][0]
    assert entry["source"] == "cell-manifest"
    assert entry["cell_index"] == 0
    assert entry["port"] == 19080
    assert entry["pid"] == "101"
    assert entry["expected_cpus"] == Q0A
    assert entry["observed_thread_union"] == Q0A
    assert entry["match"] is True
    assert entry["note"] == "ok"
    assert entry["memory_placement"]["note"] == "mocked"
    assert "role" not in entry

    # stdout carries the artifact JSON for the invoking harness
    stdout_artifact = json.loads(capsys.readouterr().out)
    assert stdout_artifact["live_affinity_verified"] is True
    assert [e["note"] for e in stdout_artifact["instances"]] == ["ok", "ok"]


def test_cell_flags_with_inline_pid_pass(monkeypatch, tmp_path) -> None:
    out = tmp_path / "artifact.json"
    calls = _patch_live(
        monkeypatch,
        pids={19011: "555"},
        unions={"555": _cpus(HALF1)},
    )
    rc = _run_main(monkeypatch, [
        "--cell", json.dumps({"cpuset": HALF1, "port": 19011, "pid": 555}),
        "--output", str(out),
    ])
    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["instances"][0]["source"] == "cell"
    assert artifact["instances"][0]["pid"] == "555"
    assert artifact["manifest_path"] is None
    # the supplied (cross-checked) pid is what gets verified
    assert "555" in calls["thread_union"]


def test_pid_map_supplied_and_agreeing_verifies_supplied_pid(monkeypatch, tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path)
    out = tmp_path / "artifact.json"
    calls = _patch_live(
        monkeypatch,
        pids={19080: "101", 19081: "102"},
        unions={"101": _cpus(Q0A), "102": _cpus(Q0B)},
    )
    rc = _run_main(monkeypatch, [
        "--cell-manifest", str(manifest_path),
        "--pid-map", json.dumps({"19080": 101, "19081": 102}),
        "--output", str(out),
    ])
    assert rc == 0
    assert set(calls["thread_union"]) >= {"101", "102"}


# ---------------------------------------------------------------- cell mode: failures


def test_thread_outside_cpuset_fails(monkeypatch, tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path)
    out = tmp_path / "artifact.json"
    _patch_live(
        monkeypatch,
        pids={19080: "101", 19081: "102"},
        # instance 1 has one thread allowed on CPU 50 — outside Q0B
        unions={"101": _cpus(Q0A), "102": _cpus(Q0B) | {50}},
    )
    rc = _run_main(monkeypatch, ["--cell-manifest", str(manifest_path), "--output", str(out)])
    assert rc == 1
    artifact = json.loads(out.read_text())
    assert artifact["live_affinity_verified"] is False
    assert artifact["instances"][0]["match"] is True
    assert artifact["instances"][1]["match"] is False
    assert artifact["instances"][1]["note"] == "AFFINITY MISMATCH"


def test_no_live_process_on_port_fails(monkeypatch, tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path)
    out = tmp_path / "artifact.json"
    _patch_live(
        monkeypatch,
        pids={19080: "101"},  # 19081 has no listener
        unions={"101": _cpus(Q0A)},
    )
    rc = _run_main(monkeypatch, ["--cell-manifest", str(manifest_path), "--output", str(out)])
    assert rc == 1
    artifact = json.loads(out.read_text())
    assert artifact["live_affinity_verified"] is False
    entry = artifact["instances"][1]
    assert entry["match"] is False
    assert entry["pid"] is None
    assert entry["note"] == "no live process on port"


def test_pid_cross_check_mismatch_fails(monkeypatch, tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path)
    out = tmp_path / "artifact.json"
    _patch_live(
        monkeypatch,
        pids={19080: "101", 19081: "102"},
        unions={"101": _cpus(Q0A), "102": _cpus(Q0B), "999": _cpus(Q0B)},
    )
    rc = _run_main(monkeypatch, [
        "--cell-manifest", str(manifest_path),
        "--pid-map", json.dumps({"19081": 999}),  # harness thinks 999; port serves 102
        "--output", str(out),
    ])
    assert rc == 1
    artifact = json.loads(out.read_text())
    entry = artifact["instances"][1]
    assert entry["match"] is False
    assert "PID CROSS-CHECK MISMATCH" in entry["note"]
    assert entry["pid"] == "999"
    assert entry["pid_on_port"] == "102"


def test_foreign_llama_overlap_fails(monkeypatch, tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path)
    out = tmp_path / "artifact.json"
    _patch_live(
        monkeypatch,
        pids={19080: "101", 19081: "102"},
        unions={
            "101": _cpus(Q0A), "102": _cpus(Q0B),
            "777": {24, 25},  # foreign llama-server squatting inside Q0B
        },
        procs=[("101", "llama-server --port 19080"),
               ("777", "llama-server --port 8080")],
    )
    rc = _run_main(monkeypatch, ["--cell-manifest", str(manifest_path), "--output", str(out)])
    assert rc == 1
    artifact = json.loads(out.read_text())
    assert artifact["live_affinity_verified"] is False
    # per-cell checks themselves passed; the foreign overlap is the failure
    assert all(e["match"] for e in artifact["instances"])
    assert artifact["foreign_llama_overlaps"] == [
        {"pid": "777", "args": "llama-server --port 8080", "overlap_cpus": "24-25"},
    ]


def test_llama_proc_pattern_covers_bench_and_cli() -> None:
    """Review F8: the foreign-process scan must cover the same llama family the
    research harness matches — llama-bench/llama-cli squatting on a declared
    cpuset is contention exactly like a foreign llama-server."""
    pattern = re.compile(affinity_preflight.LLAMA_PROC_PATTERN)
    for args in (
        "/mnt/raid0/llm/llama.cpp/build/bin/llama-server -m x.gguf --port 8080",
        "/mnt/raid0/llm/llama.cpp/build/bin/llama-bench -m x.gguf",
        "taskset -c 0-95 llama-cli -m x.gguf -p hi",
        "/mnt/raid0/llm/ik_llama.cpp/build/bin/llama-server --port 8090",
    ):
        assert pattern.search(args), args
    # word/path-boundary anchoring: no substring false positives
    for args in ("python3 not_llama_serverX.py", "vim llama-serverette-notes.md"):
        assert not pattern.search(args), args


def test_foreign_llama_without_overlap_passes(monkeypatch, tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path)
    out = tmp_path / "artifact.json"
    _patch_live(
        monkeypatch,
        pids={19080: "101", 19081: "102"},
        unions={
            "101": _cpus(Q0A), "102": _cpus(Q0B),
            "888": _cpus(HALF1),  # disjoint from the declared cpusets
        },
        procs=[("888", "llama-server --port 8280")],
    )
    rc = _run_main(monkeypatch, ["--cell-manifest", str(manifest_path), "--output", str(out)])
    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["foreign_llama_overlaps"] == []


# ---------------------------------------------------------------- usage errors (exit 2)


def test_port_outside_bench_range_refused(monkeypatch, tmp_path, capsys) -> None:
    _patch_live(monkeypatch)
    rc = _run_main(monkeypatch, ["--cell", json.dumps({"cpuset": Q0A, "port": 8080})])
    assert rc == 2
    assert "outside bench range" in capsys.readouterr().err


def test_allow_any_port_opts_out_of_range_guard(monkeypatch, tmp_path) -> None:
    out = tmp_path / "artifact.json"
    _patch_live(monkeypatch, pids={8080: "42"}, unions={"42": _cpus(Q0A)})
    rc = _run_main(monkeypatch, [
        "--cell", json.dumps({"cpuset": Q0A, "port": 8080}),
        "--allow-any-port", "--output", str(out),
    ])
    assert rc == 0
    assert json.loads(out.read_text())["live_affinity_verified"] is True


def test_unknown_schema_version_refused(monkeypatch, tmp_path, capsys) -> None:
    _patch_live(monkeypatch)
    manifest_path = _write_manifest(tmp_path, {**MANIFEST, "schema_version": "e5-cell-manifest/2"})
    rc = _run_main(monkeypatch, ["--cell-manifest", str(manifest_path)])
    assert rc == 2
    assert "schema_version" in capsys.readouterr().err


def test_bad_cell_json_refused(monkeypatch, capsys) -> None:
    _patch_live(monkeypatch)
    rc = _run_main(monkeypatch, ["--cell", "{not json"])
    assert rc == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_missing_manifest_file_refused(monkeypatch, tmp_path) -> None:
    _patch_live(monkeypatch)
    rc = _run_main(monkeypatch, ["--cell-manifest", str(tmp_path / "does_not_exist.json")])
    assert rc == 2


def test_roles_combined_with_cell_mode_refused(monkeypatch, tmp_path, capsys) -> None:
    _patch_live(monkeypatch)
    manifest_path = _write_manifest(tmp_path)
    rc = _run_main(monkeypatch, ["--roles", "frontdoor", "--cell-manifest", str(manifest_path)])
    assert rc == 2
    assert "--roles" in capsys.readouterr().err


def test_cell_and_manifest_mutually_exclusive(monkeypatch, tmp_path) -> None:
    _patch_live(monkeypatch)
    manifest_path = _write_manifest(tmp_path)
    rc = _run_main(monkeypatch, [
        "--cell-manifest", str(manifest_path),
        "--cell", json.dumps({"cpuset": Q0A, "port": 19080}),
    ])
    assert rc == 2


def test_pid_map_without_cell_mode_refused(monkeypatch, capsys) -> None:
    _patch_live(monkeypatch)
    rc = _run_main(monkeypatch, ["--pid-map", "{}"])
    assert rc == 2
    assert "cell mode" in capsys.readouterr().err


# ---------------------------------------------------------------- legacy --roles regression


def test_legacy_roles_mode_unchanged(monkeypatch, tmp_path, capsys) -> None:
    """Default role-keyed invocation: artifact shape, stdout format, and exit code
    are the pre-extension behavior — no cell-mode fields, no foreign-llama scan."""
    sys.path.insert(0, str(affinity_preflight.ORCH))
    from scripts.server.stack_numa import NUMA_CONFIG

    instances = NUMA_CONFIG["frontdoor"]["instances"]
    pids = {inst[1]: str(1000 + i) for i, inst in enumerate(instances)}
    unions = {str(1000 + i): _cpus(inst[0]) for i, inst in enumerate(instances)}
    calls = _patch_live(monkeypatch, pids=pids, unions=unions)

    def _boom():
        raise AssertionError("_llama_processes must not run in role mode")

    monkeypatch.setattr(affinity_preflight, "_llama_processes", _boom)

    out = tmp_path / "artifact.json"
    rc = _run_main(monkeypatch, ["--roles", "frontdoor", "--output", str(out)])
    assert rc == 0

    artifact = json.loads(out.read_text())
    assert artifact["live_affinity_verified"] is True
    assert artifact["roles_checked"] == ["frontdoor"]
    # role-mode artifact carries none of the cell-mode fields
    for key in ("mode", "manifest_path", "cell_id", "foreign_llama_overlaps"):
        assert key not in artifact
    for entry in artifact["instances"]:
        assert entry["role"] == "frontdoor"
        assert "instance_idx" in entry
        assert "cell_index" not in entry and "source" not in entry
    # stdout is the human-readable role report, not a JSON dump
    stdout = capsys.readouterr().out
    assert stdout.lstrip().startswith("OK")
    assert "live_affinity_verified = True" in stdout
    assert len(calls["thread_union"]) == len(instances)


def test_legacy_roles_mode_mismatch_still_exits_1(monkeypatch, tmp_path) -> None:
    _patch_live(monkeypatch)  # no pids on any port → every instance fails
    out = tmp_path / "artifact.json"
    rc = _run_main(monkeypatch, ["--roles", "frontdoor", "--output", str(out)])
    assert rc == 1
    artifact = json.loads(out.read_text())
    assert artifact["live_affinity_verified"] is False
    assert all(e["note"] == "no live process" for e in artifact["instances"])


def test_foreign_scan_matches_executable_not_arguments(monkeypatch):
    """Live incident 2026-07-23: earlyoom carries llama-server/llama-bench in
    its --ignore/--prefer REGEX ARGUMENTS and spans all CPUs — a full-cmdline
    grep made every E5 cell fail the foreign-overlap gate. The scan must match
    argv[0] basename only."""
    import subprocess as _sp

    from scripts.server import affinity_preflight as ap

    ps_out = (
        "    PID ARGS\n"
        "   1849 /usr/local/bin/earlyoom -M 41943040 --ignore ^(llama-server|sd-server)$ --prefer ^llama-bench$\n"
        "   2001 /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m model.gguf --port 19080\n"
        "   2002 bash -c echo llama-cli is not running here\n"
        "   2003 llama-bench -m model.gguf\n"
    )

    class _R:
        stdout = ps_out
        returncode = 0

    monkeypatch.setattr(ap.subprocess, "run", lambda *a, **k: _R())
    procs = ap._llama_processes()
    pids = {p for p, _ in procs}
    assert "2001" in pids  # real llama-server by argv0
    assert "2003" in pids  # bare llama-bench argv0
    assert "1849" not in pids  # earlyoom: llama names only in ARGUMENTS
    assert "2002" not in pids  # bash with llama text in arguments


def test_foreign_allow_pattern_records_but_does_not_gate(monkeypatch, tmp_path):
    """Operator-sanctioned coexistence (2026-07-23): foreign llama processes
    matching --foreign-allow-pattern are recorded as foreign_allowed_overlaps
    and do not gate; non-matching foreigners still fail the cell."""
    import json as _json

    ap = affinity_preflight

    manifest = tmp_path / "cell.json"
    manifest.write_text(_json.dumps({
        "schema_version": ap.CELL_MANIFEST_SCHEMA_VERSION,
        "cell_id": "t-c1",
        "instances": [{"cpu_list": "0-3", "port": 19990, "threads": 4}],
    }))
    monkeypatch.setattr(ap, "_pid_on_port", lambda port: "500")
    monkeypatch.setattr(ap, "_thread_union", lambda pid: {0, 1, 2, 3} if pid in ("500", "600", "700") else set())
    monkeypatch.setattr(ap, "_cmdline", lambda pid: ["llama-server", "--port", "19990"])
    monkeypatch.setattr(ap, "_llama_processes", lambda: [
        ("500", "llama-server --port 19990"),
        ("600", "/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server -m x --port 18072"),
    ])
    monkeypatch.setattr(ap, "_memory_placement_for_pid", lambda *a, **k: (None, "n/a", None), raising=False)

    out = tmp_path / "artifact.json"
    rc = _run_main(monkeypatch, [
        "--cell-manifest", str(manifest),
        "--pid-map", _json.dumps({"19990": 500}),
        "--output", str(out),
        "--foreign-allow-pattern", "build-hip",
    ])
    art = _json.loads(out.read_text())
    assert art["foreign_llama_overlaps"] == []
    assert len(art["foreign_allowed_overlaps"]) == 1
    assert art["foreign_allowed_overlaps"][0]["pid"] == "600"
    assert rc == 0 and art["live_affinity_verified"] is True

    monkeypatch.setattr(ap, "_llama_processes", lambda: [
        ("500", "llama-server --port 19990"),
        ("700", "llama-bench -m other.gguf"),
    ])
    out2 = tmp_path / "artifact2.json"
    rc2 = _run_main(monkeypatch, [
        "--cell-manifest", str(manifest),
        "--pid-map", _json.dumps({"19990": 500}),
        "--output", str(out2),
        "--foreign-allow-pattern", "build-hip",
    ])
    art2 = _json.loads(out2.read_text())
    assert len(art2["foreign_llama_overlaps"]) == 1
    assert rc2 != 0 and art2["live_affinity_verified"] is False

def test_smt_fold_makes_the_gpu_lane_visible_to_a_full_machine_cell(monkeypatch) -> None:
    """E5 protection defect: the logical-id intersection reported ZERO overlap.

    GPU host threads sit on 184-191, which are the SMT siblings of physical 88-95.
    An E5 cell on 0-95 owns those physical cores, so the two genuinely contend —
    but their LOGICAL id sets are disjoint, so the old gate saw nothing at all.
    """
    _bind_canonical_lane_fold_without_a_kernel_store(monkeypatch)
    lane = set(range(184, 192))
    cell = set(range(0, 96))
    assert not (lane & cell), "premise: the logical sets really are disjoint"
    physical = affinity_preflight._physical_overlap(lane, cell)
    assert physical == set(range(88, 96)), physical


def test_physical_overlap_is_empty_for_a_genuinely_disjoint_half(monkeypatch) -> None:
    """The guard must not forbid its own compliant idiom.

    HALF_A (0-47,96-143) is GPU-DISJOINT by construction — that is why the half
    fleet exists. If folding reported contention here too, every shape would look
    contended and the signal would be worthless.
    """
    _bind_canonical_lane_fold_without_a_kernel_store(monkeypatch)
    lane = set(range(184, 192))
    half_a = set(range(0, 48)) | set(range(96, 144))
    assert affinity_preflight._physical_overlap(lane, half_a) == set()
    # HALF_B genuinely IS a co-tenant, and must still say so.
    half_b = set(range(48, 96)) | set(range(144, 192))
    assert affinity_preflight._physical_overlap(lane, half_b) == set(range(88, 96))


def test_gpu_discovery_uses_an_open_device_not_an_argv_pattern(monkeypatch, tmp_path) -> None:
    """A `python` ROCm trainer was invisible because discovery matched argv[0].

    The definitive test for "is this process on the GPU" is whether it holds an AMD
    device node open, so a trainer named `python` is found and a process merely
    NAMED something gpu-ish is not.
    """
    fake = {"4242": ["/dev/kfd", "/dev/urandom"], "4243": ["/dev/null"]}

    def fake_glob(pattern):
        for pid, targets in fake.items():
            if pattern == f"/proc/{pid}/fd/*":
                return [f"/proc/{pid}/fd/{i}" for i in range(len(targets))]
        return []

    def fake_readlink(path):
        pid = path.split("/")[2]
        return fake[pid][int(path.rsplit("/", 1)[1])]

    monkeypatch.setattr(affinity_preflight.glob, "glob", fake_glob)
    monkeypatch.setattr(affinity_preflight.os, "readlink", fake_readlink)
    assert affinity_preflight._holds_gpu_device("4242") is True
    assert affinity_preflight._holds_gpu_device("4243") is False
    # And the llama pattern would never have found the trainer.
    assert not re.search(affinity_preflight.LLAMA_PROC_PATTERN, "python")


def test_require_memory_locality_refuses_a_vacuous_pass() -> None:
    """T5, partial: a guarantee satisfied by checking nothing is not a guarantee.

    `memory_verified` is `mismatches == 0`, and the locality predicate is
    `no_mmap AND single-node` — false for EVERY instance since the topology change
    made them all multi-node. So the artifact asserted
    `live_memory_placement_verified: true` on a run that examined zero entries, and
    `--require-memory-locality` passed vacuously. Absence read as a pass.

    This does NOT arm the predicate — T5 is blocked on ratifying
    INTERLEAVE_TOLERANCE, and mmap placement is shared across instances so arming
    it could hard-fail a correctly-configured role. It only makes "not checked"
    distinguishable from "checked and passed".
    """
    src = Path(affinity_preflight.__file__).read_text(encoding="utf-8")
    # The vacuous branch exists and fires before the mismatch branch.
    assert "if args.require_memory_locality and not memory_checked:" in src
    vac = src.index("if args.require_memory_locality and not memory_checked:")
    mis = src.index("if args.require_memory_locality and not memory_verified:")
    assert vac < mis, "the vacuity check must precede the mismatch check"
    # And the artifact distinguishes the two states rather than collapsing them.
    assert '"live_memory_placement_checked": memory_checked,' in src
    assert '"memory_locality_vacuous"' in src
    # memory_checked is derived from entries examined, never from mismatches.
    assert "memory_checked = memory_required_entries > 0" in src
