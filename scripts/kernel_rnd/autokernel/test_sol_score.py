import json
import math
import sqlite3
import struct

import pytest

import sol_score as ss


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, kv: dict, tensors: list):
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), len(kv))
    for k, v in kv.items():
        out += _s(k)
        if isinstance(v, str):
            out += struct.pack("<I", 8) + _s(v)
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", v)
    for name, dims, ttype in tensors:
        out += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<I", ttype) + struct.pack("<Q", 0)
    path.write_bytes(out)
    return path


def test_gap_closed_formula_and_bounds():
    assert ss.gap_closed(tk=2.0, tb=2.0, tsol=1.0)["S"] == pytest.approx(0.5)      # baseline
    assert ss.gap_closed(tk=1.0, tb=2.0, tsol=1.0)["S"] == pytest.approx(1.0)      # at SOL
    assert ss.gap_closed(tk=1.5, tb=2.0, tsol=1.0)["S"] == pytest.approx(1 / 1.5)
    worse = ss.gap_closed(tk=4.0, tb=2.0, tsol=1.0)["S"]
    assert 0 < worse < 0.5                                                         # bounded below by 0


def test_s_monotone_in_candidate_speed():
    vals = [ss.gap_closed(tk, 2.0, 1.0)["S"] for tk in (2.0, 1.8, 1.5, 1.2, 1.0)]
    assert vals == sorted(vals) and all(0 < v <= 1 for v in vals)


@pytest.mark.parametrize("tk,tb,tsol", [(0.9, 2.0, 1.0), (1.0, 0.9, 1.0), (1.0, 1.0, 1.0)])
def test_bound_audit_when_below_bound_or_baseline_at_bound(tk, tb, tsol):
    r = ss.gap_closed(tk, tb, tsol)
    assert r["status"] == ss.BOUND_AUDIT and r["S"] is None


@pytest.mark.parametrize("bad", [(0, 1, 1), (1, -1, 1), (float("nan"), 1, 1), (1, 1, math.inf), (None, 1, 1)])
def test_indeterminate_on_nonsense(bad):
    assert ss.gap_closed(*bad)["status"] == ss.INDETERMINATE


def test_gguf_bytes_and_active_expert_scaling(tmp_path):
    p = write_gguf(tmp_path / "m.gguf",
                   {"x.n_routed_experts": 8, "x.num_experts_per_tok": 2},
                   [("token_embd.weight", (64, 100), 8),            # 1 row: 64/32*34 bytes
                    ("blk.0.attn_q.weight", (64, 64), 8),           # full
                    ("blk.0.ffn_gate_exps.weight", (64, 64, 8), 8)])  # x 2/8
    h = ss.read_gguf_header(p)
    r = ss.active_bytes_per_token(h)
    q8 = lambda n: n // 32 * 34
    assert r["total_bytes"] == q8(6400) + q8(4096) + q8(32768)
    assert r["active_bytes"] == pytest.approx(q8(6400) / 100 + q8(4096) + q8(32768) * 2 / 8)
    assert r["moe_tensors_scaled"] == 1 and r["expert_count"] == 8 and r["expert_used_count"] == 2


def test_dense_model_counts_everything_and_bad_files_refused(tmp_path):
    p = write_gguf(tmp_path / "d.gguf", {}, [("blk.0.w.weight", (64, 64), 1)])
    assert ss.active_bytes_per_token(ss.read_gguf_header(p))["active_bytes"] == 64 * 64 * 2
    junk = tmp_path / "j.gguf"
    junk.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(ss.GgufError):
        ss.read_gguf_header(junk)
    with pytest.raises(ss.GgufError):
        ss.tensor_bytes((33,), 8)      # not a multiple of the block
    with pytest.raises(ss.GgufError):
        ss.tensor_bytes((32,), 999)


def test_sharded_headers_merge(tmp_path):
    write_gguf(tmp_path / "m-00001-of-00002.gguf", {"x.n_routed_experts": 4}, [("a.weight", (32,), 8)])
    write_gguf(tmp_path / "m-00002-of-00002.gguf", {}, [("b.weight", (32,), 8)])
    h = ss.read_gguf_header_sharded(tmp_path / "m-00001-of-00002.gguf")
    assert [t[0] for t in h["tensors"]] == ["a.weight", "b.weight"] and h["metadata"]["x.n_routed_experts"] == 4


def test_tsol_seconds():
    assert ss.tsol_seconds(260e9, 260.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ss.tsol_seconds(0, 1)


def test_residency_tags():
    assert ss.residency_tag(ngl=99, device_none=False, vram_bytes=50 << 30) == "GPU-RESIDENT"
    assert ss.residency_tag(ngl=0, device_none=True, vram_bytes=0) == "CPU-RESIDENT"
    assert ss.residency_tag(ngl=0, device_none=True, vram_bytes=60 << 30) == "HYBRID"
    assert ss.residency_tag(ngl=None, device_none=False, vram_bytes=None) == "UNKNOWN"


def test_with_sol_column_is_descriptive_and_non_mutating():
    row = {"id": "k1", "anchor_tok_s": 10.0, "candidate_tok_s": 12.0, "effect_pct": 20.0}
    out = ss.with_sol_column(row, tsol=0.05, residency="CPU-RESIDENT")
    assert "sol_gap_closed" not in row and out["effect_pct"] == 20.0
    col = out["sol_gap_closed"]
    assert col["authority"] == "descriptive_only" and col["promotion_input"] is False
    assert col["S"] == pytest.approx(1 / (1 + (1 / 12 - 0.05) / (1 / 10 - 0.05)))
    assert ss.with_sol_column({"anchor_tok_s": 10.0}, tsol=0.05, residency="x")["sol_gap_closed"]["status"] \
        == ss.INDETERMINATE


def test_spearman_and_ties():
    assert ss.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert ss.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert ss.spearman([1, 1, 1], [1, 2, 3]) is None and ss.spearman([1, 2], [1, 2]) is None
    assert ss.rank([5, 5, 1]) == [2.5, 2.5, 1.0]


def test_retro_over_synthetic_campaign(tmp_path):
    model = write_gguf(tmp_path / "m.gguf", {}, [("blk.0.w.weight", (4096, 4096), 8)])   # ~17.8 MB
    census = tmp_path / "census.json"
    census.write_text(json.dumps({"ceiling_gbps": 1500.0, "cpu": {"ceiling_gbps": {
        "half-48t": 100.0, "quarter-24t": 50.0}}}))
    db_dir = tmp_path / "camp" / "store"
    db_dir.mkdir(parents=True)
    con = sqlite3.connect(db_dir / "experiments.db")
    con.execute("create table experiments (attempt_id, effect_fraction, status, payload)")
    tsol = ss.active_bytes_per_token(ss.read_gguf_header(model))["active_bytes"] / 100e9
    for i, (a, c) in enumerate([(1 / (tsol * 4), 1 / (tsol * 3)), (1 / (tsol * 4), 1 / (tsol * 2)),
                                (1 / (tsol * 4), 1 / (tsol * 1.5)), (1 / (tsol * 4), 1 / (tsol * 0.5))]):
        payload = {"comparison": {"anchor_tok_s": a, "candidate_tok_s": c, "noise_floor_pct": 1.0,
                                  "residency": {"median_vram_bytes": 0}},
                   "research_scope": {"recipe": {"original_serving_arms": {"anchor": {
                       "backend": "cpu", "argv": ["x", "-m", str(model), "-t", "48", "-ngl", "0",
                                                  "--device", "none"]}}}}}
        con.execute("insert into experiments values (?,?,?,?)", (f"a{i}", c / a - 1, "kept", json.dumps(payload)))
    con.execute("insert into experiments values ('x', 0.1, 'measured_null', '{}')")
    con.commit()
    con.close()
    doc = ss.retro(tmp_path, census)
    assert doc["keep_rows"] == 4 and doc["by_device"] == {"cpu": 4}      # only status=kept
    assert doc["status_counts"][ss.OK] == 3 and doc["status_counts"][ss.BOUND_AUDIT] == 1
    assert doc["promotion_input"] is False
    assert doc["rank_correlation"]["spearman_S_vs_effect_pct"] == pytest.approx(1.0)
    assert all(r["residency"] == "CPU-RESIDENT" for r in doc["rows"])


def test_missing_model_is_indeterminate_unless_proxy_tagged(tmp_path):
    proxy = write_gguf(tmp_path / "p.gguf", {}, [("w.weight", (4096, 4096), 8)])
    row = {"attempt_id": "a", "effect_fraction": 0.1, "payload": {
        "comparison": {"anchor_tok_s": 10.0, "candidate_tok_s": 11.0},
        "research_scope": {"recipe": {"original_serving_arms": {"anchor": {
            "backend": "cpu", "argv": ["x", "-m", "/nonexistent/old.gguf", "-t", "48", "-ngl", "0"]}}}}}}
    bw = {"cpu-48t": 1.0, "cpu-24t": 1.0, "gpu": 1.0}
    assert ss.score_kept_row(row, bw, {})["sol_gap_closed"]["status"] == ss.INDETERMINATE
    tagged = ss.score_kept_row(row, bw, {}, {"/nonexistent/old.gguf": str(proxy)})
    assert tagged["bytes_source"].startswith("proxy_gguf:")
