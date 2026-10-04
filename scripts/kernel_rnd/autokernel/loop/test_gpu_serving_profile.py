"""G2: the selected GPU serving target's own rocprofv3 profile as a planner input.

Everything here is hermetic: synthetic traces, synthetic code objects, fake processes.
No GPU, no server, no profiler is started."""
import csv
import json
import os
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import gpu_serving_profile as gsp
from . import hotspots


# ------------------------------------------------------------------ helpers

def _requests(n=1):
    return [(f"p{i}", json.dumps({"prompt": f"hello {i}", "n_predict": 8,
                                  "cache_prompt": True}).encode()) for i in range(n)]


def _mp(obj):
    """Tiny msgpack encoder for the synthetic AMDGPU metadata note."""
    if isinstance(obj, bool):
        return b"\xc3" if obj else b"\xc2"
    if isinstance(obj, int):
        return bytes([obj]) if 0 <= obj <= 0x7F else b"\xcd" + obj.to_bytes(2, "big")
    if isinstance(obj, str):
        raw = obj.encode()
        return (bytes([0xA0 | len(raw)]) if len(raw) < 32 else b"\xd9" + bytes([len(raw)])) + raw
    if isinstance(obj, list):
        return bytes([0x90 | len(obj)]) + b"".join(_mp(x) for x in obj)
    if isinstance(obj, dict):
        return bytes([0x80 | len(obj)]) + b"".join(_mp(k) + _mp(v) for k, v in obj.items())
    raise TypeError(obj)


def _elf(sections):
    """ELF64 with section headers only. `sections`: list of (name, type, bytes)."""
    names = b"\0" + b"".join(n.encode() + b"\0" for n, _t, _b in sections) + b".shstrtab\0"
    blobs, offset, body = [], 64, b""
    for _n, _t, data in sections + [(".shstrtab", 3, names)]:
        body += data + b"\0" * (-len(data) % 8)
        blobs.append((offset, len(data)))
        offset = 64 + len(body)
    shoff = 64 + len(body)
    header = bytearray(64)
    header[:5] = b"\x7fELF\x02"
    struct.pack_into("<Q", header, 0x28, shoff)
    count = len(sections) + 2
    struct.pack_into("<HHH", header, 0x3A, 64, count, count - 1)
    shdrs = bytearray(64)  # null section
    name_off = 1
    for (name, typ, _data), (off, size) in zip(sections + [(".shstrtab", 3, names)], blobs):
        sh = bytearray(64)
        struct.pack_into("<II", sh, 0, name_off, typ)
        struct.pack_into("<QQ", sh, 0x18, off, size)
        shdrs += sh
        name_off += len(name) + 1
    return bytes(header) + body + bytes(shdrs)


def _code_object(kernels):
    desc = _mp({"amdhsa.kernels": kernels})
    note = struct.pack("<III", 7, len(desc), 32) + b"AMDGPU\0\0" + desc + b"\0" * (-len(desc) % 4)
    return _elf([(".note", 7, note)])


def _bundle(code_object):
    triples = [b"host-x86_64-unknown-linux--", b"hipv4-amdgcn-amd-amdhsa--gfx90a"]
    header = gsp._BUNDLE_MAGIC + struct.pack("<Q", 2)
    entries_len = sum(24 + len(t) for t in triples)
    payload_off = len(header) + entries_len
    entries = struct.pack("<QQQ", payload_off, 0, len(triples[0])) + triples[0]
    entries += struct.pack("<QQQ", payload_off, len(code_object), len(triples[1])) + triples[1]
    return header + entries + code_object


KERNEL = {".name": "_Z3fooILi256EEvPKc", ".vgpr_count": 100, ".agpr_count": 0,
          ".sgpr_count": 56, ".vgpr_spill_count": 0, ".sgpr_spill_count": 0,
          ".private_segment_fixed_size": 0, ".group_segment_fixed_size": 8448,
          ".max_flat_workgroup_size": 128}


def _trace_csv(rows):
    out = []
    header = ["Kind", "Agent_Id", "Queue_Id", "Kernel_Id", "Kernel_Name", "Correlation_Id",
              "Start_Timestamp", "End_Timestamp", "Private_Segment_Size", "Group_Segment_Size",
              "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z",
              "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z"]
    out.append(",".join(header))
    for i, (name, start, end) in enumerate(rows):
        out.append(",".join(map(str, ["KERNEL_DISPATCH", 4, 1, 1, f'"{name}"', i, start, end,
                                      0, 0, 128, 1, 1, 128, 1, 1])))
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ plan / key / command

def test_plan_records_skips_and_long_context_hook_seam():
    plans, skipped = gsp.plan_windows(_requests(1), np=1)
    assert [p.name for p in plans] == ["prefill", "short_decode"]
    assert json.loads(plans[0].bodies[0])["n_predict"] == 1
    assert json.loads(plans[0].bodies[0])["cache_prompt"] is False
    assert all(json.loads(b)["stream"] is True for p in plans for b in p.bodies)
    assert set(skipped) == {"concurrent_decode", "long_decode", "prefill_at_depth"}
    assert "LongContextHook" in skipped["long_decode"]

    class Hook:
        identity = "longctx-v0"

        def window_requests(self, window, launch):
            return [_requests(1)[0][1]] if window == "long_decode" else None

        def skip_reason(self, window):
            return "prefill-at-depth surface not restored"

    plans, skipped = gsp.plan_windows(_requests(4), np=4, hook=Hook())
    assert [p.name for p in plans] == ["prefill", "short_decode", "concurrent_decode", "long_decode"]
    assert len(plans[2].bodies) == 4
    assert skipped == {"prefill_at_depth": "prefill-at-depth surface not restored"}


def test_cache_key_moves_with_anchor_launch_and_requests():
    plans, _ = gsp.plan_windows(_requests(1), np=1)
    digest = gsp.request_digest(plans)
    base = dict(anchor_commit="a" * 40, execution_digest="e" * 64, request_digest_=digest)
    key = gsp.cache_key(**base)
    assert key == gsp.cache_key(**base)
    assert key != gsp.cache_key(**{**base, "anchor_commit": "b" * 40})
    assert key != gsp.cache_key(**{**base, "execution_digest": "f" * 64})
    other, _ = gsp.plan_windows([("x", b'{"prompt": "other"}')], np=1)
    assert key != gsp.cache_key(**{**base, "request_digest_": gsp.request_digest(other)})


def test_profile_command_uses_the_gpu_host_lane_and_a_scratch_port(tmp_path):
    argv = ["/b/bin/llama-server", "-m", "m.gguf", "--host", "0.0.0.0", "--port", "8083", "-ngl", "all"]
    env_seen = {}

    def profiler_env(binary, rocprof):
        env_seen["binary"] = binary
        return {"LD_LIBRARY_PATH": f"{binary.parent}:/sdk/lib", "ROCPROFILER_METRICS_PATH": "/sdk/share"}

    command, env = gsp.profile_command(argv, {"LD_LIBRARY_PATH": "/old", "LD_PRELOAD": "x"},
                                       port=18184, trace_dir=tmp_path, rocprof="/sdk/rocprofv3",
                                       profiler_env=profiler_env)
    assert command[:6] == ["numactl", "--membind=3", "--", "taskset", "-c", "184-191"]
    assert command[6] == "/sdk/rocprofv3" and "--kernel-trace" in command
    server = command[command.index("/b/bin/llama-server"):]
    assert server[0] == "/b/bin/llama-server"
    assert server[server.index("--port") + 1] == "18184"
    assert server[server.index("--host") + 1] == "127.0.0.1"
    assert env["LD_LIBRARY_PATH"].startswith("/b/bin") and "LD_PRELOAD" not in env
    with pytest.raises(gsp.ProfileFailed, match="8083"):
        gsp.profile_command(argv, {}, port=8083, trace_dir=tmp_path, rocprof="r",
                            profiler_env=profiler_env)


# ------------------------------------------------------------------ trace / clock / tables

def _marks(base_wall, boot_offset):
    def mark(name, wall):
        return {"name": name, "wall": wall, "boot_ns": int(wall * 1e9 + boot_offset),
                "mono_ns": int(wall * 1e9 + boot_offset + 7e9)}
    return [mark("launch", base_wall), mark("healthy", base_wall + 10),
            mark("fid_start", base_wall + 12), mark("fid_end", base_wall + 18),
            mark("teardown", base_wall + 40), mark("exited", base_wall + 41)]


def test_clock_is_chosen_by_bounds_and_fiducial_and_windows_are_cut():
    wall, offset = 1_700_000_000.0, -1_699_000_000.0 * 1e9
    marks = _marks(wall, offset)

    def ns(t):
        return int((wall + t) * 1e9 + offset)
    rows = [("void warm<1>(int)", ns(10.5), ns(10.6)), ("void warm<1>(int)", ns(11.0), ns(11.2)),
            # fiducial gap 11.2 -> 18.0, then the prefill and decode windows
            ("void mul_mat_q<(ggml_type)8, 16, false>(char const*)", ns(18.01), ns(18.2)),
            ("void mul_mat_vec_q<(ggml_type)8, 1>(void const*)", ns(20.0), ns(20.3)),
            ("void mul_mat_vec_q<(ggml_type)8, 1>(void const*)", ns(20.5), ns(20.7)),
            ("void foo<256>(char const*)", ns(20.8), ns(20.9))]
    dispatches = gsp.parse_trace(_trace_csv(rows))
    clock = gsp.choose_clock(dispatches, marks)
    assert clock["method"] == "clock:boot"
    resources = {"foo<256>": {"vgpr": 100, "agpr": 0, "sgpr": 56, "vgpr_spill": 0,
                              "sgpr_spill": 0, "scratch_bytes": 0, "lds_static_bytes": 8448}}
    result = gsp.analyze(dispatches, marks, {"prefill": {"start": wall + 18, "end": wall + 18.5},
                                             "short_decode": {"start": wall + 19.9, "end": wall + 21}},
                         resources)
    decode = result["windows"]["short_decode"]
    assert decode["status"] == "observed" and decode["dispatches"] == 3
    top = decode["kernels"][0]
    assert top["kernel"] == "mul_mat_vec_q<(ggml_type)8, 1>" and top["calls"] == 2
    assert top["resources"] == "absent" and top["waves_per_simd_est"] is None
    foo = next(r for r in decode["kernels"] if r["kernel"] == "foo<256>")
    # VGPR allows 4 waves/SIMD; 8448 B LDS x 128-thread groups allows 7 groups/CU = 3/SIMD.
    assert (foo["vgpr"], foo["lds_bytes"], foo["waves_per_simd_est"]) == (100, 8448, 3)
    assert foo["occupancy_limiter"] == "lds"
    assert result["windows"]["prefill"]["kernels"][0]["kernel"].startswith("mul_mat_q<")


def test_unaligned_trace_refuses_rather_than_guessing():
    with pytest.raises(gsp.ProfileFailed, match="aligned"):
        gsp.analyze([], [{"name": "launch", "wall": 1.0}], {"short_decode": {"start": 1, "end": 2}}, {})


def test_occupancy_estimate_names_its_limiter():
    assert gsp.occupancy(vgpr=56, agpr=0, sgpr=32)["waves_per_simd"] == 8
    assert gsp.occupancy(vgpr=124, agpr=4)["waves_per_simd"] == 4
    lds = gsp.occupancy(vgpr=16, sgpr=16, lds_bytes=32768, workgroup=256)
    assert (lds["waves_per_simd"], lds["limiter"]) == (2, "lds")


def test_code_object_metadata_gives_registers_spills_and_lds(tmp_path):
    co = _code_object([KERNEL, {**KERNEL, ".name": "_Z3barv", ".vgpr_spill_count": 3,
                                ".private_segment_fixed_size": 64}])
    blob = _bundle(co) + b"\0" * 16 + _bundle(_code_object([{**KERNEL, ".name": "_Z3bazv"}]))
    kernels = gsp.code_object_kernels(blob)
    assert [k[".name"] for k in kernels] == ["_Z3fooILi256EEvPKc", "_Z3barv", "_Z3bazv"]
    library = tmp_path / "libggml-hip.so"
    library.write_bytes(_elf([(".text", 1, b"\x90" * 8), (".hip_fatbin", 1, blob)]))
    table = gsp.kernel_resources(library, demangle=lambda names: [
        {"_Z3fooILi256EEvPKc": "void foo<256>(char const*)", "_Z3barv": "bar()",
         "_Z3bazv": "baz()"}[n] for n in names])
    assert table["foo<256>"]["vgpr"] == 100 and table["foo<256>"]["lds_static_bytes"] == 8448
    assert table["bar"]["vgpr_spill"] == 3 and table["bar"]["scratch_bytes"] == 64


# ------------------------------------------------------------------ lifecycle (fake process)

def test_run_profiles_one_launch_and_cuts_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(gsp, "FIDUCIAL_S", 1.2)
    build = tmp_path / "bin"
    build.mkdir()
    (build / "llama-server").write_bytes(b"\x7fELF fake")
    (build / "libggml-hip.so").write_bytes(b"")
    dispatches = []

    def now_ns():
        return time.clock_gettime_ns(time.CLOCK_BOOTTIME)

    class Proc:
        pid = 99_999_999  # not a live process: maps unreadable, proof legs unproven
        returncode = None

        def __init__(self, command, **_kw):
            self.trace_dir = Path(command[command.index("-d") + 1])
            self.signals = []

        def poll(self):
            return self.returncode

        def send_signal(self, sig):
            self.signals.append(sig)
            (self.trace_dir / "serving_kernel_trace.csv").write_text(_trace_csv(dispatches))
            self.returncode = 0

        def wait(self, _timeout=None):
            return self.returncode

    procs = []

    def popen(command, **kwargs):
        procs.append(Proc(command, **kwargs))
        return procs[-1]

    def stream(url, body, _timeout):
        sent = time.time()
        start = now_ns()
        kernel = ("void mul_mat_q<(ggml_type)8, 16, false>(char const*)"
                  if json.loads(body).get("n_predict") == 1 else
                  "void mul_mat_vec_q<(ggml_type)8, 1>(void const*)")
        dispatches.append((kernel, start, start + 2_000_000))
        time.sleep(0.02)
        first = time.time()
        time.sleep(0.005)
        s2 = now_ns()
        dispatches.append((kernel, s2, s2 + 1_000_000))
        time.sleep(0.02)
        return {"sent": sent, "first": first, "done": time.time(), "error": None}

    class Sampler:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def watch_pid(self, pid):
            self.pid = pid

        proof = {"own_pid_peak_vram_bytes": 0, "own_pid_kfd_reads": 0, "peak_kfd_processes": 1}

    plans, _skipped = gsp.plan_windows(_requests(1), np=1)
    body = gsp.run(command_argv=[str(build / "llama-server"), "--port", "8099"], launch_env={},
                   port=18184, plans=plans, out_dir=tmp_path / "out", rocprof="/sdk/rocprofv3",
                   profiler_env=lambda b, r: {"LD_LIBRARY_PATH": str(b.parent)},
                   resources={}, popen=popen, stream=stream,
                   wait_healthy=lambda port, proc, timeout: 0.1, sampler_factory=Sampler)
    assert procs[0].signals == [15]  # SIGTERM to our own child only
    assert body["teardown"] == "terminated"
    assert body["clock"]["method"] == "clock:boot", body["clock"]
    assert body["windows"]["short_decode"]["status"] == "observed"
    assert body["windows"]["short_decode"]["kernels"][0]["kernel"] == "mul_mat_vec_q<(ggml_type)8, 1>"
    assert body["windows"]["prefill"]["kernels"][0]["kernel"].startswith("mul_mat_q<")
    assert body["hip_proof"]["status"] in {"unproven", "refuted", "proven"}


def test_run_refuses_a_non_hip_build_before_launching(tmp_path):
    (tmp_path / "llama-server").write_text("#!/bin/sh\n")
    with pytest.raises(gsp.ProfileFailed, match="not a HIP llama-server build"):
        gsp.run(command_argv=[str(tmp_path / "llama-server")], launch_env={}, port=1,
                plans=[], out_dir=tmp_path / "o", rocprof="r", profiler_env=lambda b, r: {},
                resources={}, popen=lambda *a, **k: pytest.fail("launched"))


# ------------------------------------------------------------------ hotspots entry + cache

def test_serving_profile_is_cached_per_anchor_and_feeds_the_planner(tmp_path, monkeypatch):
    monkeypatch.setattr(hotspots, "_resolve_rocprof", lambda: "/sdk/rocprofv3")
    launch = SimpleNamespace(template=SimpleNamespace(np=1), execution_digest="e" * 64,
                             command_argv=["/b/bin/llama-server"], launch_env=(),
                             build_dir=str(tmp_path / "build"))
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        table = [{"kernel": "mul_mat_vec_q<(ggml_type)8, 1>", "calls": 4, "total_ns": 4000,
                  "share": 0.8}]
        return {"windows": {"short_decode": {"status": "observed", "kernels": table},
                            "prefill": {"status": "observed", "kernels": []}},
                "clock": {"method": "clock:boot"}, "hip_proof": {"status": "proven"}}

    view, rows = hotspots.serving_profile(launch, _requests(1), store=tmp_path,
                                          anchor_commit="a" * 40, runner=runner, resources={})
    assert len(calls) == 1 and view["cached"] is False
    assert view["primary_window"] == "short_decode" and view["hip_proof"] == "proven"
    assert view["windows"]["long_decode"]["status"] == "skipped"
    assert view["windows"]["concurrent_decode"]["status"] == "skipped"
    assert rows[0].signature == "mul_mat_vec_q<(ggml_type)8, 1>" and rows[0].calls == 4
    view2, _ = hotspots.serving_profile(launch, _requests(1), store=tmp_path,
                                        anchor_commit="a" * 40, runner=runner, resources={})
    assert len(calls) == 1 and view2["cached"] is True and view2["key"] == view["key"]
    hotspots.serving_profile(launch, _requests(1), store=tmp_path, anchor_commit="b" * 40,
                             runner=runner, resources={})
    assert len(calls) == 2  # a new anchor is a new profile


def test_serving_profile_without_a_profiler_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(hotspots, "_resolve_rocprof", lambda: None)
    launch = SimpleNamespace(template=SimpleNamespace(np=1), execution_digest="e" * 64,
                             command_argv=["/b/bin/llama-server"], launch_env=(),
                             build_dir=str(tmp_path))
    with pytest.raises(hotspots.ProfileFailed, match="rocprofv3"):
        hotspots.serving_profile(launch, _requests(1), store=tmp_path, anchor_commit="a" * 40,
                                 resources={})


def test_run_py_no_longer_prints_the_unavailable_placeholder():
    source = Path(__file__).with_name("run.py").read_text(encoding="utf-8")
    assert "selected GPU serving profile unavailable" not in source
    assert "hotspots.serving_profile(" in source


def test_planner_prompt_renders_windows_registers_and_skips():
    from . import actors
    view = gsp.observation({"key": "k", "windows": {"short_decode": {
        "status": "observed", "window_s": 2.0, "busy_fraction": 0.9, "dispatches": 10,
        "kernels": [{"kernel": "foo<256>", "calls": 3, "total_ns": 9000, "share": 0.5,
                     "mean_us": 3.0, "vgpr": 100, "agpr": 0, "sgpr": 56, "vgpr_spill": 0,
                     "sgpr_spill": 0, "scratch_bytes": 0, "lds_bytes": 8448,
                     "waves_per_simd_est": 3, "occupancy_limiter": "lds"}]}},
        "clock": {"method": "clock:boot"}, "hip_proof": {"status": "proven"}},
        record="/store/p.json", np=1, skipped={"long_decode": "no long-context surface"})
    text = "\n".join(actors._render_gpu_serving_profile(view))
    assert "100/0" in text and "3 (lds)" in text and "`foo<256>`" in text
    assert "`long_decode`: skipped — no long-context surface" in text
    absent = "\n".join(actors._render_gpu_serving_profile({"status": "unavailable",
                                                            "reason": "no rocprofv3"}))
    assert "unavailable: no rocprofv3" in absent
