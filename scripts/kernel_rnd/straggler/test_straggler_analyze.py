"""Unit test for straggler_analyze on a synthetic GTT2 trace."""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import straggler_analyze as S  # noqa: E402


def write_eval(f, nth, nodes, ends, cpus, batch=4, eidx=0):
    """nodes: [(opname, name, kind, ne0)], ends[k][t] = compute end ns; start = 1000*k, leave = max end + 50."""
    f.write(S.HDR.pack(S.MAGIC, nth, len(nodes), batch, eidx, 0))
    f.write(struct.pack(f"<{nth}i", *cpus))
    for k, (opn, nm, kind, ne0) in enumerate(nodes):
        f.write(S.NODE.pack(k, 0, kind, 0, ne0, 4, 1, 1, opn.encode(), nm.encode()))
    for t in range(nth):
        for k in range(len(nodes)):
            st = 10_000 * k
            e = st + ends[k][t]
            f.write(struct.pack("<4I", st, e, e, st + max(ends[k]) + 50))


def make(tmp, slow_tid=3, nth=4, truncate=False):
    p = tmp / "t.bin"
    nodes = [("MUL_MAT_ID", "ffn_moe_down-5", 0, 2048), ("GATED_DELTA_NET", "gdn_out-5", 0, 128),
             ("RMS_NORM", "attn_norm-5", 0, 2048), ("ARGSORT", "ffn_moe_argsort-5", 0, 128),
             ("ADD", "l_out-5", 0, 2048)]
    with open(p, "wb") as f:
        for ev in range(3):
            ends = []
            for k, _ in enumerate(nodes):
                base = [1000] * nth
                if k == 0:
                    base[slow_tid] = 5000      # big straggler on MUL_MAT_ID: wait 4000 - 1000 = 3000
                elif k == 1:
                    base[slow_tid] = 3000      # GDN: wait 1500
                elif k == 2:
                    base[slow_tid] = 1400      # norm: wait 300
                elif k == 3:
                    base[0] = 1200             # argsort: wait 150, last = tid 0
                ends.append(base)
            write_eval(f, nth, nodes, ends, cpus=[0, 1, 2, 97], eidx=ev)
        if truncate:
            f.write(S.HDR.pack(S.MAGIC, nth, 5, 4, 9, 0) + b"\0" * 10)
    return p


def test_attribution_and_located(tmp_path):
    topo = tmp_path / "lscpu.csv"
    topo.write_text("# CPU,Core,Socket,Node,L1d:L1i:L2:L3\n0,0,0,0,0:0:0:0\n1,1,0,0,1:1:1:0\n2,2,0,1,2:2:2:1\n"
                    "96,0,0,0,0:0:0:0\n97,1,0,0,1:1:1:0\n")
    r = S.analyse([make(tmp_path, truncate=True)], str(topo))
    assert r["n_evals"] == 3                          # truncated tail ignored
    by = {o["key"]: o for o in r["by_op"]}
    assert abs(by["MUL_MAT_ID"]["wait_ns"] - 3 * 3000) < 1e-6
    assert abs(by["GATED_DELTA_NET"]["wait_ns"] - 3 * 1500) < 1e-6
    assert abs(by["MUL_MAT_ID"]["thread_wait_ns"] - 3 * 4 * 3000) < 1e-6
    assert by["ADD"]["wait_ns"] == 0
    assert r["top3_ops"] == ["MUL_MAT_ID", "GATED_DELTA_NET", "RMS_NORM"]
    assert r["verdict"] == "LOCATED" and r["top3_share"] > 0.95
    cats = {c["key"]: c["wait_ns"] for c in r["by_category"]}
    assert cats["gdn_ssm"] == 3 * 1500 and cats["moe_routing"] == 3 * 150 and cats["norm"] == 3 * 300
    assert cats["mul_mat_id"] == 3 * 3000             # 2048 rows / 4 threads is not few-rows
    names = {n["key"] for n in r["by_name_top"]}
    assert "ffn_moe_down-N" in names
    lt = r["last_thread"]
    assert lt["top_tids"][0]["tid"] == 3
    assert lt["top_cpus"][0]["cpu"] == 97 and lt["top_cpus"][0]["smt"] == 1
    assert lt["by_smt"]["1"]["last"] > lt["by_smt"]["0"]["last"]


def test_diffuse(tmp_path):
    p = tmp_path / "d.bin"
    nodes = [(f"OP{k}", f"n{k}", 0, 4096) for k in range(6)]
    with open(p, "wb") as f:
        write_eval(f, 2, nodes, [[1000, 2000]] * 6, cpus=[0, 1])
    r = S.analyse([p])
    assert r["verdict"] == "DIFFUSE" and abs(r["top3_share"] - 0.5) < 1e-9
