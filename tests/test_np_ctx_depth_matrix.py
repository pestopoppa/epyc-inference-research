"""Unit tests for scripts/benchmark/np_ctx_depth_matrix.py (no server, no compute)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("ndm", HERE.parent / "scripts/benchmark/np_ctx_depth_matrix.py")
ndm = importlib.util.module_from_spec(SPEC)
sys.modules["ndm"] = ndm
SPEC.loader.exec_module(ndm)

SERVER = {"id": "t", "device": "gpu", "argv_file": "", "n_ctx_train": 262144, "kv_bytes_per_token": 1024 * 1024,
          "non_kv_gib": 10.0, "capacity_gib": 110.0, "arms": {"INCUMBENT": {"store": "/s/a"}},
          "kv_mode_by_np": {"8": {"mode": "unified", "evidence": "e8"}, "default": {"mode": "split", "evidence": "d"}},
          "prefill_tok_s_prior": 1000, "decode_tok_s_prior": 50}


def test_capacity_skip_and_c_scaling():
    sp = ndm.plan_server(SERVER, ["1k", "64k", "100k"], [1, 8], 64, 256)
    # 1 MiB/token: np1 x 100k ~ 100 GiB > budget 100 GiB? per_slot 102720 tok -> 100.3 GiB -> skipped
    by = {(c["np"], c["depth"]): c["status"] for c in sp["cells"]}
    assert by[(1, "1k")] == "PLANNED" and by[(1, "64k")] == "PLANNED"
    assert by[(1, "100k")] == "SKIPPED_CAPACITY"
    assert by[(8, "64k")] == "SKIPPED_CAPACITY"
    l1 = next(L for L in sp["launches"] if L["np"] == 1)
    assert l1["c"] % 256 == 0 and l1["c"] >= 65536 + 64 + 256
    assert l1["kv_mode"] == "split" and l1["n_ctx_seq"] == l1["c"]


def test_kv_mode_per_np_and_n_ctx_seq():
    s = dict(SERVER, kv_bytes_per_token=1.0)
    sp = ndm.plan_server(s, ["1k", "100k"], [2, 8], 64, 256)
    l2 = next(L for L in sp["launches"] if L["np"] == 2)
    l8 = next(L for L in sp["launches"] if L["np"] == 8)
    assert l2["kv_mode"] == "split" and l2["n_ctx_seq"] == l2["c"] // 2
    assert l8["kv_mode"] == "unified" and l8["n_ctx_seq"] == l8["c"] and l8["n_ctx_seq_over_train"]
    assert l8["max_position"] <= s["n_ctx_train"]


def test_requires_ctx_extension_flagged_not_run():
    s = dict(SERVER, kv_bytes_per_token=1.0, n_ctx_train=32768)
    sp = ndm.plan_server(s, ["1k", "64k"], [1], 64, 256)
    by = {c["depth"]: c["status"] for c in sp["cells"]}
    assert by["64k"] == "REQUIRES_CTX_EXTENSION"
    assert sp["launches"][0]["depths"] == ["1k"]


def test_documents_distinct_and_deterministic(tmp_path):
    for i in range(6):
        (tmp_path / f"f{i}.md").write_text(f"doc {i} " * 500)
    src = [str(p) for p in tmp_path.glob("*.md")]
    a = ndm.build_documents(src, 4, 20000)
    b = ndm.build_documents(list(reversed(src)), 4, 20000)
    assert a == b
    assert len({d[:4096] for d in a}) == 4 and all(len(d) >= 20000 for d in a)
    m = ndm.doc_manifest(a)
    assert len({x["sha256"] for x in m}) == 4


def _fake(delays=None):
    calls = []

    def req(stream, prompt, n):
        calls.append((stream, len(prompt)))
        prev = max([L for s, L in calls[:-1] if s == stream] or [0])
        return {"timings": {"prompt_n": len(prompt) - prev, "prompt_ms": (len(prompt) - prev) / 2.0,
                            "predicted_n": n, "predicted_ms": n * 10.0, "draft_n": 10, "draft_n_accepted": 4,
                            "cache_n": prev}}
    return req, calls


def test_run_steps_incremental_prefill_and_aggregate():
    req, calls = _fake()
    toks = [list(range(200000)) for _ in range(2)]
    L = {"np": 2, "depths": ["1k", "8k"]}
    rows = ndm.run_steps(req, toks, L, 64, 1e9, {"arm": "A", "server": "t"})
    assert [r["depth"] for r in rows] == ["1k", "8k"]
    r8 = rows[1]
    assert all(s["prompt_n"] == 8192 - 1024 for s in r8["streams"])      # only the delta is prefilled
    assert r8["prefill_incremental_tok_s_med"] == pytest.approx(2000.0)
    assert r8["decode_tok_s_per_request_med"] == pytest.approx(100.0)
    assert r8["decode_tok_s_aggregate"] == pytest.approx(200.0)
    assert r8["draft_acceptance"] == pytest.approx(0.4)
    assert all(ndm.verify_row(r) for r in rows)


def test_run_steps_step_cap_stops_deeper():
    req, _ = _fake()
    toks = [list(range(200000))]
    rows = ndm.run_steps(req, toks, {"np": 1, "depths": ["1k", "8k", "32k"]}, 8, -1.0, {"arm": "A"})
    assert rows[0]["status"] == "MEASURED"
    assert [r["status"] for r in rows[1:]] == ["STOPPED_STEP_CAP", "STOPPED_STEP_CAP"]


def test_export_catalog_shape_and_hash_check():
    req, _ = _fake()
    rows = []
    for arm in ("INCUMBENT", "C22"):
        rows += ndm.run_steps(req, [list(range(10000))], {"np": 1, "depths": ["1k"]}, 8, 1e9,
                              {"arm": arm, "server": "t", "kv_mode": "split", "c": 2048, "n_ctx_seq": 2048})
    rec = ndm.export_catalog(rows, "x", "T", "H", "src", ["INCUMBENT", "C22"])
    assert rec["matrix_columns"][:2] == ["context depth", "np"] and len(rec["matrix_columns"]) == 8
    assert rec["matrix"][0][:2] == ["1k", "1"] and rec["matrix"][0][2] != ""
    assert {s["arm"] for s in rec["study"]} == {"INCUMBENT", "C22"}
    bad = dict(rows[0], decode_tok_s_aggregate=1.0)
    with pytest.raises(SystemExit):
        ndm.export_catalog([bad], "x", "T", "H", "src", ["INCUMBENT"])


def test_launch_argv_rewrites(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(["@@BIN@@", "-m", "m.gguf", "-np", "4", "-c", "999", "--kv-unified", "--port", "1",
                             "--slot-save-path", "/x"]))
    s = dict(SERVER, argv_file=str(p))
    a = ndm.launch_argv(s, {"np": 8, "c": 1024, "kv_mode": "split"}, "/store/bin", 5555)
    assert a[0] == "/store/bin/llama-server"
    assert a[a.index("-np") + 1] == "8" and a[a.index("-c") + 1] == "1024" and a[a.index("--port") + 1] == "5555"
    assert "--no-kv-unified" in a and "--kv-unified" not in a and "--slot-save-path" not in a


def test_block_order_balances_first_position():
    arms = ["INCUMBENT", "INCUMBENT2", "C21", "C22"]
    firsts = [ndm.block_order(arms, b)[0] for b in range(8)]
    assert set(firsts) == set(arms) and all(sorted(ndm.block_order(arms, b)) == sorted(arms) for b in range(8))


def test_resolve_paired_ci_and_aa_guard():
    rows = []
    for b in range(6):
        for arm, v in (("INCUMBENT", 100.0), ("INCUMBENT2", 100.0 + (0.5 if b % 2 else -0.5)), ("C22", 96.0 + (b % 2) * 0.4)):
            rows.append({"kind": "repeat", "np": 4, "depth": "1k", "status": "MEASURED", "block": b, "arm": arm,
                         "decode_tok_s_aggregate": v})
    r = ndm.resolve(rows, 4, "1k", "decode_tok_s_aggregate", "INCUMBENT", "INCUMBENT2", 5, 0.02)
    assert r["blocks"] == 6 and r["aa_ok"]
    assert r["arms"]["C22"]["ci95"][1] < 1.0 and r["futility_look"]["C22"] == "decided"
