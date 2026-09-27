"""Tests for gfx90a_isa_audit (INF03-REGAUDIT-1).

Fixtures are synthetic llvm-readelf / llvm-objdump text and packed bytes, so the
suite needs neither ROCm nor a GPU. One integration test runs against the frozen
production library when it and ROCm's llvm-objdump are present (read-only).
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx90a_isa_audit as ga  # noqa: E402

SYM = "_ZL9mul_mat_qIL9ggml_type8ELi64ELb0EEvPKcPKi"
STUB = "_ZL9mul_mat_qIL9ggml_type8ELi48ELb1EEvPKcPKi"

NOTES = f"""Displaying notes found in: .note
  Owner                Data size 	Description
  AMDGPU               0x0000067c	NT_AMDGPU_METADATA (AMDGPU Metadata)
    AMDGPU Metadata:
        ---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .max_flat_workgroup_size: 512
    .name:           {SYM}
    .private_segment_fixed_size: 100
    .sgpr_count:     72
    .sgpr_spill_count: 0
    .symbol:         {SYM}.kd
    .vgpr_count:     256
    .vgpr_spill_count: 37
    .wavefront_size: 64
  - .agpr_count:     4
    .group_segment_fixed_size: 1024
    .max_flat_workgroup_size: 256
    .name:           {STUB}
    .private_segment_fixed_size: 16
    .reqd_workgroup_size:
      - 128
      - 2
      - 1
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         {STUB}.kd
    .vgpr_count:     44
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx90a
amdhsa.version:
  - 1
  - 2
...
"""


def _ins(addr, text, target=None):
    tail = f" <{SYM}+0x{target:x}>" if target is not None else ""
    return f"\t{text:<58}// {addr:012X}: DEADBEEF{tail}"


def _disasm():
    """A kernel with a prologue, an outer loop and an inner hot loop with two latches."""
    base = 0x1000
    lines = [f"{base:016x} <{SYM}>:"]
    body = [
        ("s_mov_b64 s[66:67], s[2:3]", None),                     # 0 private resource copy (hi)
        ("s_mov_b64 s[64:65], s[0:1]", None),                     # 1 private resource copy (lo)
        ("s_add_u32 s64, s64, s7", None),                         # 2
        ("buffer_store_dword v1, off, s[64:67], 0 offset:64", None),  # 3 spill store (prologue)
        ("v_mov_b32_e32 v2, 0", None),                            # 4 outer header
        ("s_waitcnt vmcnt(0)", None),                             # 5 hot header
        ("buffer_load_dword v3, off, s[64:67], 0 offset:64", None),   # 6 spill reload
        ("buffer_load_dwordx4 v[8:11], v4, s[8:11], 0 offen", None),  # 7 real global buffer load
        ("v_mfma_i32_16x16x16i8 v[64:67], v56, v60, 0", None),    # 8
        ("s_barrier", None),                                      # 9
        ("ds_read_b128 v[12:15], v5", None),                      # 10
        ("s_waitcnt lgkmcnt(0)", None),                           # 11
        ("v_mfma_i32_16x16x16i8 v[68:71], v56, v60, 0", None),    # 12
        ("v_mfma_i32_16x16x16i8 v[72:75], v56, v60, 0", None),    # 13
        ("s_cbranch_execz", "hot"),                               # 14 early latch -> hot header
        ("buffer_load_dword v6, off, s[64:67], 0 offset:68", None),   # 15 spill reload (tail)
        ("v_accvgpr_read_b32 v7, a0", None),                      # 16
        ("v_accvgpr_write_b32 a1, v7", None),                     # 17
        ("s_barrier", None),                                      # 18
        ("v_mfma_i32_16x16x16i8 v[76:79], v56, v60, 0", None),    # 19
        ("s_waitcnt vmcnt(0) lgkmcnt(0)", None),                  # 20
        ("s_branch", "hot"),                                      # 21 far latch -> hot header
        ("v_add_u32_e32 v2, 1, v2", None),                        # 22
        ("s_cbranch_scc1", "outer"),                              # 23 outer latch
        ("global_store_dword v[0:1], v2, off", None),             # 24
        ("s_endpgm", None),                                       # 25
    ]
    addrs = [base + 4 * i for i in range(len(body))]
    hot, outer = addrs[5] - base, addrs[4] - base
    for addr, (text, tgt) in zip(addrs, body):
        if tgt == "hot":
            lines.append(_ins(addr, f"{text} 1", hot))
        elif tgt == "outer":
            lines.append(_ins(addr, f"{text} 1", outer))
        else:
            lines.append(_ins(addr, text))
    lines.append("")
    lines.append(f"{0x2000:016x} <{STUB}>:")
    for i, text in enumerate(["s_add_u32 s0, s0, s9", "s_getpc_b64 s[4:5]",
                              "s_add_u32 s4, s4, 0xfffffffc", "s_addc_u32 s5, s5, -1",
                              "s_swappc_b64 s[30:31], s[4:5]"]):
        lines.append(f"\t{text:<58}// {0x2000 + 4 * i:012X}: DEADBEEF")
    return "\n".join(lines) + "\n"


def _kd(accum_field=15, vgpr_gran=31, tg_split=False):
    kd = bytearray(64)
    struct.pack_into("<III", kd, 0, 0, 100, 168)
    rsrc3 = accum_field | ((1 << 16) if tg_split else 0)
    rsrc1 = vgpr_gran | (8 << 6)
    struct.pack_into("<III", kd, 44, rsrc3, rsrc1, 1)
    return bytes(kd)


# --------------------------------------------------------------------- parsing


def test_parse_notes_text_extracts_resource_fields():
    meta = ga.parse_notes_text(NOTES)
    assert set(meta) == {SYM, STUB}
    m = meta[SYM]
    assert (m[".vgpr_count"], m[".agpr_count"], m[".vgpr_spill_count"]) == (256, 0, 37)
    assert m[".private_segment_fixed_size"] == 100 and m[".max_flat_workgroup_size"] == 512
    assert meta[STUB][".reqd_workgroup_size"] == [128, 2, 1]


def test_kernel_descriptor_decodes_accum_offset_and_tg_split():
    kd = ga.parse_kernel_descriptor(_kd(accum_field=15, vgpr_gran=31, tg_split=True))
    assert kd["accum_offset"] == 64
    assert kd["tg_split"] is True
    assert kd["vgpr_alloc_granulated"] == 256
    assert kd["kd_private_segment_bytes"] == 100 and kd["private_segment_enabled"]
    with pytest.raises(ga.AuditError):
        ga.parse_kernel_descriptor(b"\0" * 10)


def test_toolchain_regime_from_comment():
    rocm = (b"Linker: AMD LLD 18.0.0\0AMD clang version 18.0.0git (https://github.com/"
            b"RadeonOpenCompute/llvm-project roc-6.2.0 24292 26466ce)\0")
    t = ga.toolchain_from_comment(rocm)
    assert t["llvm_major"] == 18 and t["rocm_version"] == "6.2.0"
    assert t["mfma_form_regime"] == "mayneedagprs_rule_pre_159493"
    assert ga.toolchain_from_comment(b"clang version 22.1.0\0")["mfma_form_regime"] == "llvm_159493_default"
    assert ga.toolchain_from_comment(None)["mfma_form_regime"] == "unknown"


def test_split_bundles_keeps_only_gfx90a_entries():
    host, dev = b"HOSTCODE", b"\x7fELFdevice-object"
    triples = [b"host-x86_64-unknown-linux-gnu-", b"hipv4-amdgcn-amd-amdhsa--gfx90a"]
    header_len = len(ga.BUNDLE_MAGIC) + 8 + sum(24 + len(t) for t in triples)
    blob = bytearray(ga.BUNDLE_MAGIC + struct.pack("<Q", 2))
    off = header_len
    for t, payload in zip(triples, (host, dev)):
        blob += struct.pack("<QQQ", off, len(payload), len(t)) + t
        off += len(payload)
    blob += host + dev
    assert ga.split_bundles(bytes(blob)) == [dev]
    assert ga.split_bundles(bytes(blob) + bytes(blob)) == [dev, dev]  # concatenated TUs
    with pytest.raises(ga.AuditError):
        ga.split_bundles(b"CCOB" + b"\0" * 64)


# ------------------------------------------------------------------ ISA / loops


def test_private_resource_copy_is_followed():
    insts = ga.split_disassembly(_disasm())[SYM]
    assert ga.private_resource_quads(insts) == {0, 64}


def test_find_loops_merges_back_edges_to_one_header():
    insts = ga.split_disassembly(_disasm())[SYM]
    loops = ga.find_loops(insts, SYM)
    assert loops == [(4, 23), (5, 21)]  # outer loop; hot loop spans to the FAR latch


def test_analyze_kernel_counts_hot_loop():
    text = _disasm()
    dis = ga.split_disassembly(text)
    meta = ga.parse_notes_text(NOTES)
    row = ga.analyze_kernel(SYM, dis[SYM], meta[SYM], ga.parse_kernel_descriptor(_kd()))
    assert row["family"] == "mmq" and row["params"]["ggml_type_name"] == "Q8_0"
    assert row["params"]["mmq_x"] == 64 and row["params"]["need_check"] is False
    assert row["n_loops"] == 2 and row["n_hot_loops"] == 1 and row["hot_loop_key"] == "mfma"
    hot = row["hot"]
    assert hot["mfma"] == 4 and hot["s_barrier"] == 2
    assert hot["priv_ld_off"] == 2 and hot["global_ld"] == 1  # s[8:11] is not the private quad
    assert row["hot_spill_reloads"] == 2 and row["hot_accvgpr_copies"] == 2
    assert hot["waitcnt_vmcnt0"] == 2 and hot["waitcnt_lgkmcnt0"] == 2 and hot["waitcnt_full"] == 1
    # phases: [header..1st barrier]=1 (+ tail after 2nd barrier = 1) , [1st..2nd barrier]=2
    assert row["hot_mfma_barrier_segments"] == [[2, 2]] and row["hot_mfma_phases"] == 2
    assert row["kern_spill_reloads"] == 2 and row["kern"]["priv_st_off"] == 1
    assert row["scratch_kind"] == "spill"
    assert row["accum_offset"] == 64 and row["arch_vgpr"] == 256  # agpr=0 -> arch == total
    assert row["occupancy_waves_per_simd"] == 2.0 and row["occupancy_limiter"] == "vgpr"
    assert row["stub"] is False
    detail = row["hot_loops_detail"][0]
    assert detail["spill_reloads"] == 2 and detail["accvgpr_copies"] == 2


def test_stub_kernel_and_stack_only_scratch():
    dis = ga.split_disassembly(_disasm())
    meta = ga.parse_notes_text(NOTES)
    # getpc at 0x2004 -> pc 0x2008, imm -4 -> callee at 0x2004
    row = ga.analyze_kernel(STUB, dis[STUB], meta[STUB], None,
                            {0x2004: "_ZL14no_device_codePKciS0_iS0_"})
    assert row["callees"] == ["_ZL14no_device_codePKciS0_iS0_"]
    assert row["stub"] is True and row["n_hot_loops"] == 0
    real_call = ga.analyze_kernel(STUB, dis[STUB], meta[STUB], None, {0x2004: "_Z6helperv"})
    assert real_call["stub"] is True  # still <= STUB_MAX_INSTS with no loop and no MFMA
    padded = dis[STUB] + [ga.Inst(0x3000 + 4 * i, "v_add_u32_e32", "v1, v1, v2") for i in range(40)]
    assert ga.analyze_kernel(STUB, padded, meta[STUB], None, {0x2004: "_Z6helperv"})["stub"] is False
    assert ga.analyze_kernel(STUB, padded, meta[STUB], None,
                             {0x2004: "_ZL14no_device_codePKciS0_iS0_"})["stub"] is True
    assert row["scratch_kind"] == "stack"  # private bytes but zero spills: never called spills
    assert row["hot_spill_reloads"] == 0
    assert row["arch_vgpr"] == 40  # no descriptor: total - agpr
    assert row["waves_per_wg"] == 4  # reqd_workgroup_size 128x2x1 = 256 threads


def test_no_spill_kernel_reports_zero_reloads_even_with_private_traffic():
    insts = ga.split_disassembly(_disasm())[SYM]
    meta = dict(ga.parse_notes_text(NOTES)[SYM], **{".vgpr_spill_count": 0})
    row = ga.analyze_kernel(SYM, insts, meta, None)
    assert row["hot"]["priv_ld_off"] == 2 and row["hot_spill_reloads"] == 0
    assert row["scratch_kind"] == "stack"


@pytest.mark.parametrize("vgpr,sgpr,lds,wg,waves,limiter", [
    (256, 72, 0, 512, 2.0, "vgpr"),
    (512, 104, 0, 256, 1.0, "vgpr"),
    (128, 80, 0, 256, 4.0, "vgpr"),
    (64, 40, 40000, 256, 1.0, "lds"),
    (24, 16, 0, 64, 8.0, "vgpr"),
])
def test_occupancy_model(vgpr, sgpr, lds, wg, waves, limiter):
    o = ga.occupancy(vgpr, sgpr, lds, wg)
    assert o["occupancy_waves_per_simd"] == waves
    assert o["occupancy_limiter"] == limiter


@pytest.mark.parametrize("name,family,key,value", [
    ("_ZL9mul_mat_qIL9ggml_type10ELi64ELb1EEvPKc", "mmq", "ggml_type_name", "Q2_K"),
    ("_ZL24mul_mat_q_stream_k_fixupIL9ggml_type8ELi64ELb0EEvPKi", "mmq_fixup", None, None),
    ("_ZL18flash_attn_ext_f16ILi256ELi256ELi8ELi4ELb1ELb0EEvPKcS1_", "fattn_mma", "logit_softcap", True),
    ("_ZL18flash_attn_ext_f16ILi128ELi16ELi4ELi16E6__halfLb0EEvPKc", "fattn_wmma", "kq_acc", "half"),
    ("_ZL18flash_attn_ext_f16ILi64ELi16ELi4ELi8EfLb0EEvPKc", "fattn_wmma", "kq_acc", "float"),
    ("_ZL9mul_mat_fIfLi64ELi9ELi8ELb1EEvPKT_PKf", "mmf", "ids", False),
    ("_ZL13mul_mat_f_idsI15__hip_bfloat162Li32ELi10ELi1EEvPKT_", "mmf", "ids", True),
    ("_ZL18flash_attn_ext_vecILi128ELi1EL9ggml_type1ELS0_1ELb0ELb1EEvPKc", "fattn_vec", None, None),
    ("_ZL7acc_f32PKfS0_Pflllllllll", "other", None, None),
])
def test_classify_family(name, family, key, value):
    fam, params = ga.classify_family(name)
    assert fam == family
    if key:
        assert params[key] == value


# ------------------------------------------------------------------------ diff


def _row(kernel, **kw):
    base = {"kernel": kernel, "family": "mmq", "code_object": "co_000", "stub": False,
            "params": {"ggml_type": 8, "ggml_type_name": "Q8_0", "mmq_x": 64, "need_check": False},
            "vgpr_total": 200, "agpr": 0, "sgpr": 70, "vgpr_spill": 0, "sgpr_spill": 0,
            "private_bytes": 0, "lds_bytes": 0, "occupancy_waves_per_simd": 2.0,
            "hot_spill_reloads": 0, "hot_accvgpr_copies": 0, "kern_accvgpr_copies": 0,
            "n_hot_loops": 1, "hot": {"waitcnt_vmcnt0": 2, "waitcnt_lgkmcnt0": 4}}
    base.update(kw)
    return base


def _doc(rows, path="x.so"):
    return {"source": {"path": path, "sha256": "0" * 64}, "rows": rows}


def test_diff_flags_new_spill_and_occupancy_drop():
    base = _doc([_row(SYM), _row("k2")])
    cand = _doc([_row(SYM, vgpr_spill=5, hot_spill_reloads=3, vgpr_total=256),
                 _row("k2", occupancy_waves_per_simd=1.0, agpr=8)])
    res = ga.run_diff(base, cand, ("all",), None)
    checks = {(f["kernel"], f["check"]): f["severity"] for f in res["findings"]}
    assert checks[(SYM, "vgpr_spill")] == "FAIL"
    assert checks[(SYM, "hot_spill_reloads")] == "FAIL"
    assert checks[(SYM, "vgpr_total")] == "WARN"
    assert checks[("k2", "occupancy_waves_per_simd")] == "FAIL"
    assert checks[("k2", "agpr_appears")] == "FAIL"
    assert res["counts"]["FAIL"] == 4


def test_diff_improvement_is_info_and_passes(tmp_path):
    base = _doc([_row(SYM, vgpr_spill=37, hot_spill_reloads=12, vgpr_total=256)])
    cand = _doc([_row(SYM, vgpr_total=210)])
    b, c = tmp_path / "b.json", tmp_path / "c.json"
    b.write_text(json.dumps(base))
    c.write_text(json.dumps(cand))
    assert ga.main(["diff", str(b), str(c)]) == 0
    res = ga.run_diff(base, cand, ("all",), None)
    assert [f["severity"] for f in res["findings"]] == ["INFO"]
    assert "hot_spill_reloads 12 -> 0" in res["findings"][0]["detail"]


def test_diff_exit_codes_and_fail_on_warn(tmp_path):
    base = _doc([_row(SYM)])
    fail = _doc([_row(SYM, hot_accvgpr_copies=6)])
    warn = _doc([_row(SYM, vgpr_total=210)])
    paths = {}
    for name, doc in (("b", base), ("f", fail), ("w", warn)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(doc))
    assert ga.main(["diff", str(paths["b"]), str(paths["f"])]) == 1
    assert ga.main(["diff", str(paths["b"]), str(paths["w"])]) == 0
    assert ga.main(["diff", str(paths["b"]), str(paths["w"]), "--fail-on", "warn"]) == 1


def test_diff_ignores_stubs_and_reports_added_removed():
    base = _doc([_row(SYM), _row("gone"), _row("stubbed", stub=True)])
    cand = _doc([_row(SYM), _row("new", vgpr_spill=2), _row("stubbed", stub=True, vgpr_spill=9)])
    res = ga.run_diff(base, cand, ("all",), None)
    checks = {(f["kernel"], f["check"]): f["severity"] for f in res["findings"]}
    assert checks == {("gone", "kernel_removed"): "WARN", ("new", "kernel_added"): "WARN"}


# ------------------------------------------------------- belief-kernel feed


def test_claim_projection_is_observation_grade_and_skips_absent():
    row = _row(SYM, vgpr_spill=37, hot_spill_reloads=12, arch_vgpr=256, accum_offset=None,
               binary_sha256="a" * 64, row_id="r1", self_sha256="b" * 64, wg_max=512,
               hot_mfma_barrier_segments=[[0, 25, 7, 32]])
    del row["arch_vgpr"]
    doc = {"tool_id": ga.TOOL_ID, "tool_sha256": "c" * 64, "toolchain": ["AMD clang 18"],
           "mfma_form_regime": ["mayneedagprs_rule_pre_159493"]}
    claims = ga.claim_projection(row, doc, category="BASELINE", attestation_path="a.json",
                                 attestation_sha256="d" * 64, date="2026-09-26")
    metrics = [c["metric"] for c in claims]
    assert "gfx90a_static.arch_vgpr" not in metrics  # absent is skipped, never defaulted
    assert "gfx90a_static.hot_spill_reloads" in metrics
    for c in claims:
        assert c["protocol_id"] == "" and c["metric_direction"] == "lower_better"
        assert c["reps"] == 1 and c["category"] == "BASELINE"
        assert json.loads(c["attestation_locator"])[-1] == c["metric"].split(".", 1)[1]
    root_ct = Path("/workspace/scripts/vidya/claim_tuple.py")
    if root_ct.exists():  # the root grader accepts the projection as-is
        sys.path.insert(0, str(root_ct.parent))
        try:
            from claim_tuple import ClaimTuple
        except Exception:  # pragma: no cover - root tree mid-edit
            return
        for c in claims:
            ClaimTuple(**c)


# ------------------------------------------------------------------ integration

PROD = Path("/mnt/raid0/llm/kernels/production/gpu/libggml-hip.so.0.16.0")


@pytest.mark.skipif(not (PROD.exists() and (ga.LLVM_BIN / "llvm-objdump").exists()),
                    reason="frozen production library or ROCm llvm tools not present")
def test_integration_production_q8_0_j64(tmp_path):
    """Read-only: re-derive the intake-1823 Q8_0 J=64 datapoint from the frozen v10 library."""
    doc = ga.run_audit(PROD, ("mmq",), r"type8ELi64ELb0E", jobs=1)
    rows = [r for r in doc["rows"] if not r["stub"]]
    assert rows, "Q8_0 J=64 instance not found"
    r = rows[0]
    assert (r["vgpr_total"], r["agpr"], r["vgpr_spill"], r["wg_max"]) == (256, 0, 37, 512)
    assert r["n_hot_loops"] == 2
    assert sorted(h["spill_reloads"] for h in r["hot_loops_detail"]) == [10, 12]
    assert doc["mfma_form_regime"] == ["mayneedagprs_rule_pre_159493"]
