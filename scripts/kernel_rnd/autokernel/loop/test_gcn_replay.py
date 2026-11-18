"""Pure-Python tests for gcn_replay (C5-ISA-2 GPU half, host side). The GPU driver is replaced by a fake runner."""
import json
import re
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gcn_replay as gr  # noqa: E402

K = "_ZL9mul_mat_qIL9ggml_type8ELi64ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiii"
FA = "_ZL18flash_attn_ext_f16ILi256ELi256ELi1ELi8ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiiiiiiiiiiiliiliiiiil"

NOTES = f"""
---
amdhsa.kernels:
  - .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .offset: 24
        .size: 4
        .value_kind: by_value
      - .offset: 32
        .size: 8
        .value_kind: hidden_block_count_x
    .kernarg_segment_size: 32
    .name: {K}
amdhsa.target: amdgcn-amd-amdhsa--gfx90a
...
"""


def _rec(tmp, ptr2=0, dumped=True):
    ka = bytearray(32)
    ka[0:8] = (0x7f0000001000).to_bytes(8, "little")
    ka[8:16] = (0x7f0000002010).to_bytes(8, "little")
    ka[16:24] = ptr2.to_bytes(8, "little")
    ka[24:28] = (77).to_bytes(4, "little")
    allocs = [{"size": 16, "pre": "L0.a0.pre", "post": "L0.a0.post"}, {"size": 32, "pre": "L0.a1.pre", "post": "L0.a1.post"}]
    for j, a in enumerate(allocs):
        (tmp / a["pre"]).write_bytes(bytes([j]) * a["size"])
        (tmp / a["post"]).write_bytes(struct.pack(f"<{a['size'] // 4}f", *([1.5 + j] * (a["size"] // 4))))
    return {"id": 0, "k": K, "grid": [4, 2, 1], "block": [256, 1, 1], "shmem": 0, "kernarg": ka.hex(),
            "dumped": dumped, "dump_skipped": not dumped, "launch_rc": 0, "allocs": allocs, "ptrs": [[0, 0, 0], [8, 1, 16]]}


def fake_runner(times, post_from=None, mutate=False, rc=0, calls=None):
    """Emulates gcn_replay_driver: reads the plan, writes a<j>.out (copy of the given .post or the .pre), prints JSON."""
    def run(argv, timeout):
        plan, outdir, _w, iters = argv[1], Path(argv[2]), argv[3], int(argv[4])
        lines = Path(plan).read_text().splitlines()
        if calls is not None:
            calls.append(argv)
        co = next(l.split()[1] for l in lines if l.startswith("co "))
        for l in lines:
            if l.startswith("alloc "):
                _, j, n, pre, _s = l.split()
                src = Path(pre.replace(".pre", ".post")) if post_from == "post" else Path(pre)
                data = bytearray(src.read_bytes()) if src.exists() else bytearray(int(n))
                if mutate and "edited" in co:
                    data[0] ^= 1
                (outdir / f"a{j}.out").write_bytes(bytes(data))
        t = times(co) if callable(times) else times
        return rc, json.dumps({"ok": rc == 0, "times_ms": [t] * iters}), ""
    return run


def test_target_regexes_match_mangled_names():
    assert gr.target_of(K) == "MMQ Q8_0 J=64 prefill tile"
    assert gr.target_of(FA) == "MMA-FA DKQ>=192"
    assert gr.target_of(K.replace("ELi64E", "ELi32E")) is None
    assert gr.target_of(FA.replace("ILi256E", "ILi128E")) is None
    assert re.search(gr.CAPTURE_MATCH, K) and re.search(gr.CAPTURE_MATCH, FA)


def test_kernel_meta_skips_hidden():
    m = gr.kernel_meta(NOTES)[K]
    assert m["kernarg_segment_size"] == 32
    assert [a["kind"] for a in m["args"]] == ["global_buffer"] * 3 + ["by_value"]


def test_read_records_splits_dumped_and_timed(tmp_path):
    rec = _rec(tmp_path)
    p = tmp_path / "cap.jsonl"
    timed = [{"id": i, "k": K, "grid": [4, 2, 1], "block": [256, 1, 1], "shmem": 0, "timed": True, "ms": 0.1 * i,
              "launch_rc": 0} for i in range(1, 4)]
    bad = dict(timed[0], ms=-1.0)
    skip = [{"skipped": "_Z6quantize", "why": "no_match"}, {"skipped": K, "why": "no_argtable"}]
    p.write_text("\n".join(json.dumps(x) for x in [rec] + timed + [bad] + skip) + "\n{\"torn")
    r = gr.read_records(p)
    key = (K, (4, 2, 1), (256, 1, 1), 0)
    assert len(r[key]["dumped"]) == 1 and r[key]["ms"] == pytest.approx([0.1, 0.2, 0.3]) and len(r) == 1
    assert gr.read_skipped(p) == {"no_match": ["_Z6quantize"], "no_argtable": [K]}


def test_build_plan_patches_pointers_and_renders(tmp_path):
    plan = gr.build_plan(_rec(tmp_path), gr.kernel_meta(NOTES), "/x/orig.co", tmp_path)
    assert plan["ptrs"] == [(0, 0, 0), (8, 1, 16)]
    assert plan["args"] == [(0, 8), (8, 8), (16, 8), (24, 4)]
    txt = gr.render_plan(plan)
    assert "co /x/orig.co" in txt and "ptr 8 1 16" in txt and f"alloc 1 32 {tmp_path}/L0.a1.pre 2" in txt
    assert "co /y/edited.hsaco" in gr.render_plan(plan, "/y/edited.hsaco")
    fill = gr.build_plan(_rec(tmp_path), gr.kernel_meta(NOTES), "/x/orig.co", None)
    assert all(a["pre"] == "-" for a in fill["allocs"])


def test_build_plan_refuses(tmp_path):
    meta = gr.kernel_meta(NOTES)
    with pytest.raises(gr.ReplayRefused, match="did not resolve"):
        gr.build_plan(_rec(tmp_path, ptr2=0xdead0000), meta, "/x.co", tmp_path)
    with pytest.raises(gr.ReplayRefused, match="not dumped"):
        gr.build_plan(_rec(tmp_path, dumped=False), meta, "/x.co", tmp_path)
    r = _rec(tmp_path); r["kernarg"] = r["kernarg"][:-2]
    with pytest.raises(gr.ReplayRefused, match="segment"):
        gr.build_plan(r, meta, "/x.co", tmp_path)
    r = _rec(tmp_path); r["ptrs"] = [[0, 0, 99], [8, 1, 16]]
    with pytest.raises(gr.ReplayRefused, match="delta"):
        gr.build_plan(r, meta, "/x.co", tmp_path)
    with pytest.raises(gr.ReplayRefused, match="metadata"):
        gr.build_plan(dict(_rec(tmp_path), k="other"), meta, "/x.co", tmp_path)
    with pytest.raises(gr.ReplayRefused, match="whitespace"):
        gr.render_plan(gr.build_plan(_rec(tmp_path), meta, "/x y.co", tmp_path))


def test_compare_bytes():
    a = struct.pack("<4f", 1, 2, 3, float("nan"))
    assert gr.compare_bytes(a, a)["equal"]
    b = struct.pack("<4f", 1, 2, 3.5, 0.0)
    c = gr.compare_bytes(a, b)
    assert not c["equal"] and c["max_abs_f32"] == pytest.approx(0.5) and c["nan_mismatch"] == 1
    assert gr.compare_bytes(a, a[:8])["size_mismatch"] == [16, 8]


def test_identity_gate_rules():
    eq, ne = {"equal": True}, {"equal": False}
    assert gr.identity_gate([1.0] * 10, [1.05] * 10, eq)["verdict"] == "PASS"
    g = gr.identity_gate([1.0] * 10, [1.2] * 10, eq)
    assert g["verdict"] == "FAIL" and g["reasons"] == ["timing_outside_tolerance"]
    assert gr.identity_gate([1.0] * 10, [1.0] * 10, ne)["reasons"] == ["output_differs"]
    assert gr.identity_gate([1.0] * 3, [1.0] * 10, eq)["verdict"] == "INSUFFICIENT"
    assert gr.identity_gate([1.0] * 10, [0.89] * 10, eq)["verdict"] == "FAIL"


def test_identity_replay_with_fake_runner(tmp_path):
    rec, meta = _rec(tmp_path), gr.kernel_meta(NOTES)
    ok = gr.identity_replay(fake_runner(0.102, post_from="post"), "/d", rec, meta, "/x/orig.co", tmp_path,
                            [0.1] * 20, tmp_path / "w", iters=10)
    assert ok["verdict"] == "PASS" and ok["target"] == "MMQ Q8_0 J=64 prefill tile"
    bad = gr.identity_replay(fake_runner(0.1), "/d", rec, meta, "/x/orig.co", tmp_path, [0.1] * 20, tmp_path / "w2",
                             iters=10)   # outputs = .pre, not .post
    assert bad["verdict"] == "FAIL" and "output_differs" in bad["reasons"]
    err = gr.identity_replay(fake_runner(0.1, rc=3), "/d", rec, meta, "/x/orig.co", tmp_path, [0.1] * 20,
                             tmp_path / "w3", iters=10)
    assert err["verdict"] == "DRIVER_ERROR"
    ref = gr.identity_replay(fake_runner(0.1), "/d", _rec(tmp_path, ptr2=5), meta, "/x/orig.co", tmp_path, [0.1] * 20,
                             tmp_path / "w4")
    assert ref["verdict"] == "REFUSED"


def test_abba_runresult_and_decide(tmp_path):
    import gcn_gym
    plan = gr.build_plan(_rec(tmp_path), gr.kernel_meta(NOTES), "/x/orig.co", tmp_path)
    calls = []
    rr, raw = gr.abba(fake_runner(lambda co: 0.08 if "edited" in co else 0.1, calls=calls), "/d", plan, tmp_path / "ab",
                      incumbent_co="/x/orig.co", candidate_co="/y/edited.hsaco", pairs=2, iters=5)
    order = ["edited" in Path(c[1]).read_text() for c in calls]
    assert order == [False, True, True, False] * 2
    assert rr.oracle_pass and rr.candidate_ms == pytest.approx((0.08, 0.08)) and rr.incumbent_ms == pytest.approx((0.1, 0.1))
    assert gcn_gym.run_qualifies(rr)["qualifies"]
    rr2, raw2 = gr.abba(fake_runner(0.08, mutate=True), "/d", plan, tmp_path / "ab2", incumbent_co="/x/orig.co",
                        candidate_co="/y/edited.hsaco", pairs=2, iters=5)
    assert not rr2.oracle_pass and gcn_gym.run_qualifies(rr2)["reason"] == "oracle_fail"
    rr3, _ = gr.abba(fake_runner(0.08, mutate=True), "/d", plan, tmp_path / "ab3", incumbent_co="/x/orig.co",
                     candidate_co="/y/edited.hsaco", iters=5, oracle=lambda c: all(p.get("n_diff_bytes", 0) <= 2 for p in c["per_alloc"]))
    assert rr3.oracle_pass


def test_build_argvs_never_run_here():
    assert gr.driver_build_argv("/o/drv")[:2] == ["/opt/rocm/bin/hipcc", "-O2"]
    a = gr.capture_build_argv("/o/lib.so")
    assert "-shared" in a and "-lamdhip64" in a and a[-1] == "-lpthread"
    assert gr.DRIVER_SOURCE.exists() and gr.CAPTURE_SOURCE.exists()
