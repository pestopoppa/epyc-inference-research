"""AKX-ALL-LB3.2a unit tests: offline, no perf, no model, no process.

Fixtures (fixtures/): 62 perf-script lines recorded by the 2026-10-22 premeasure (champion store image dd3d9721c, test
9 = pp3 @ d65536, t48) around the end of timed repetition 1, the restore that follows, repetition 2 and the next
restore; the matching llama-bench progress lines. Times in the stderr tables below are placed from the recorded
llama-bench samples_ns of that test (rep 1 = 106.754 ms, rep 2 = 93.272 ms), which the recorded main-thread samples
corroborate: graph work up to 861107.948, restore (libc memcpy, state_read_meta, apply_ubatch) from 861107.951.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from akx_lb3_attribution import evalprof, perfagg, phases, rule  # noqa: E402
from akx_lb3_attribution import perfscript as P  # noqa: E402

FX = HERE / "fixtures"
PERF_LINES = (FX / "perfscript_lb3_test9.txt").read_text().splitlines()
REP1_NS, REP2_NS = 106754260, 93271513
SAMPLES_T9 = {9: [REP1_NS, REP2_NS, 93078292, 166286572, 94237442, 91847301, 93324704, 95395782, 92813989, 93041094]}
MAIN = 652031


def _stderr(t_prompt1: float):
    return [(861107.700000, "llama-bench: benchmark 9/12: starting"),
            (861107.700100, "llama-bench: benchmark 9/12: warmup prompt run"),
            (861107.800000, "llama-bench: benchmark 9/12: depth run 1/10 (cached)"),
            (t_prompt1, "llama-bench: benchmark 9/12: prompt run 1/10"),
            (861108.075500, "llama-bench: benchmark 9/12: depth run 2/10 (cached)"),
            (861108.075600, "llama-bench: benchmark 9/12: prompt run 2/10"),
            (861108.300000, "llama-bench: benchmark 9/12: depth run 3/10 (cached)")]


def _samples():
    return [s for s in (P.parse_line(l) for l in PERF_LINES) if s is not None]


# ------------------------------------------------------------------------------------------- (b) line parsing
OLD_LINE_RE = re.compile(r"^\s*(\d+)\s+(\d+\.\d+):\s+([0-9a-f]+)\s+(.*?)\s+\((.*)\)\s*$")   # the 2026-10-22 regex


def test_every_fixture_line_parses():
    assert len(_samples()) == len(PERF_LINES) == 62


def test_templated_symbol_keeps_its_full_name_and_dso():
    line = next(l for l in PERF_LINES if "tinyBLAS" in l)
    s = P.parse_line(line)
    assert s.sym == ("void (anonymous namespace)::tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm4xN<3>"
                     "(long, long, long, long)")
    assert s.dso.endswith("/bin/libggml-cpu.so.0.16.0")
    assert P.classify(s.sym, s.dso) == "matmul"
    # the old non-greedy regex (tid-only field order) cut the symbol at its first space
    old = OLD_LINE_RE.match(line.split("/", 1)[1])
    assert old.group(4) == "void"


def test_templated_heap_and_dequantizer_symbols_classify():
    heap = P.parse_line(next(l for l in PERF_LINES if "cmp_top_k" in l))
    assert heap.sym.startswith("void std::__adjust_heap<int*, long, int, __gnu_cxx::__ops::_Iter_comp_iter<cmp_top_k> >")
    assert P.classify(heap.sym, heap.dso) == "indexer"
    q6 = P.parse_line(next(l for l in PERF_LINES if "DequantizerQ6K" in l))
    assert "(anonymous namespace)::DequantizerQ6K_AVX2, 1>" in q6.sym
    assert P.classify(q6.sym, q6.dso) == "matmul"
    fa = P.parse_line(next(l for l in PERF_LINES if "flash_attn_ext_f16_one_chunk" in l))
    assert fa.sym.startswith("ggml_compute_forward_flash_attn_ext_f16_one_chunk(ggml_compute_params const*")
    assert P.classify(fa.sym, fa.dso) == "fa"


def test_pid_tid_field_and_tid_only_field_both_parse():
    a = P.parse_line(" 652031/652031  861107.951008:      73a0def9a9f5 [unknown] (/usr/lib/x86_64-linux-gnu/libc.so.6)")
    b = P.parse_line(" 652031  861107.951008:      73a0def9a9f5 [unknown] (/usr/lib/x86_64-linux-gnu/libc.so.6)")
    assert a.main_thread and a.pid == a.tid == MAIN and a.t == pytest.approx(861107.951008)
    assert b.pid is None and b.tid == MAIN and b.sym == "[unknown]" and b.dso == "/usr/lib/x86_64-linux-gnu/libc.so.6"
    assert P.parse_line("") is None and P.parse_line("not a sample line") is None


def test_symbol_with_parenthesis_and_no_dso():
    s = P.parse_line(" 1/2  10.000001:  abc foo(int (*)(void)) [clone .isra.0]")
    assert s.sym == "foo(int (*)(void)) [clone .isra.0]" and s.dso == ""


# ------------------------------------------------------------------------------------------- (a) DSO classification
@pytest.mark.parametrize("sym,dso,cls", [
    ("[unknown]", "/usr/lib/x86_64-linux-gnu/libc.so.6", "libc"),
    ("__memmove_avx512_unaligned_erms", "/usr/lib/x86_64-linux-gnu/libc.so.6", "libc"),
    ("[unknown]", "/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0", "omp_spin"),
    ("llama_kv_cache::state_read_meta(llama_io_read_i&, unsigned int)", "/x/bin/libllama.so.0.0.10345", "state_restore"),
    ("llama_io_read_host::read(void*, unsigned long)", "/x/bin/libllama.so.0.0.10345", "state_restore"),
    ("llama_kv_cache::get_prev_tokens(llama_ubatch const&, unsigned int)", "/x/bin/libllama.so.0.0.10413", "host_prev_tokens"),
    ("llama_kv_cache::apply_ubatch(llama_kv_cache::slot_info const&)", "/x/bin/libllama.so.0.0.10413", "host_apply_ubatch"),
    ("llama_kv_cache::set_input_kq_mask(ggml_tensor*, llama_ubatch const*, bool) const", "/x/bin/libllama.so", "host_mask_input"),
    ("std::_Rb_tree_increment(std::_Rb_tree_node_base*)", "/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.34", "host_cxx"),
    ("ggml_vec_dot_f16", "/x/bin/libggml-cpu.so.0.16.0", "fa"),
    ("ggml_compute_forward_gated_delta_net", "/x/bin/libggml-cpu.so.0.16.0", "ssm_gdn"),
    ("ggml_compute_forward_mean_d1", "/x/bin/libggml-cpu.so.0.16.0", "ggml_other"),
])
def test_classify_dso_first(sym, dso, cls):
    assert P.classify(sym, dso) == cls


def test_memcpy_symbol_in_libc_is_libc_not_restore_by_name_alone():
    # a worker-thread libc copy inside a timed step is timed work (ggml cpy / get_rows call memcpy): class libc,
    # kept in the denominator; what keeps a RESTORE memcpy out is the interval, not the name
    s = P.parse_line(next(l for l in PERF_LINES if "652031/652090" in l))
    assert not s.main_thread and P.classify(s.sym, s.dso) == "libc"


# ------------------------------------------------------------------------------------------- (a) phase gating
def test_control_policy_on_recorded_progress_lines():
    acts = []
    for text in (FX / "llama_bench_progress_test1.err").read_text().splitlines():
        p = phases.parse_progress(text)
        if p:
            acts.append((p[2], phases.control_action(p[2])))
    assert acts[:4] == [("starting", "enable"), ("warmup", None), ("depth", "disable"), ("prompt", "enable")]
    kinds = [phases.parse_progress(t)[2] for t in (FX / "llama_bench_progress_test9.err").read_text().splitlines()
             if phases.parse_progress(t)]
    assert kinds.count("depth_cached") == 10 and kinds.count("prompt") == 10
    assert all(phases.control_action(k) is None for k in kinds if k == "depth_cached")


def test_restore_memcpy_after_the_timed_interval_never_enters_the_denominator():
    segs = phases.segments(_stderr(861107.8435))
    reps = phases.timed_reps(segs, SAMPLES_T9)
    assert [(r.rep, r.excluded) for r in reps] == [(1, None), (2, None)]
    r1 = reps[0]
    assert r1.interval[1] == pytest.approx(861107.8435 + REP1_NS / 1e9 - 0.002)
    agg = perfagg.aggregate(lambda: iter(_samples()), reps)[9]
    assert agg["reps_used"] == 2 and agg["counts"].get("state_restore", 0) == 0
    # every main-thread restore sample (libc memcpy at .951-.984 and .070-.073, state_read, apply_ubatch) is outside
    starts = [r.interval[0] for r in reps]
    for s in _samples():
        if s.main_thread and 861107.950 < s.t < 861108.075:
            assert phases.assign(s.t, starts, reps) is None
    # the 2026-10-22 window (prompt run 1 -> depth run 2) held them all
    old = [s for s in _samples() if s.main_thread and 861107.8435 <= s.t < 861108.0755 and P.classify(s.sym, s.dso) == "libc"]
    assert len(old) >= 8


def test_restore_run_straddling_a_late_interval_is_trimmed():
    # the wrapper received "prompt run 1" 5 ms late: the interval end now overlaps the restore's first memcpy
    reps = phases.timed_reps(phases.segments(_stderr(861107.8485)), SAMPLES_T9)
    assert reps[0].interval[1] > 861107.951008            # the libc memcpy at .951 would be inside
    agg = perfagg.aggregate(lambda: iter(_samples()), reps)[9]
    assert reps[0].excluded is None
    assert reps[0].interval[1] == pytest.approx(861107.948008)   # cut back to the last graph sample (top_k heap)
    assert agg["restore_trim_ms"][0] > 0
    main_libc_in = [s for s in _samples() if s.main_thread and P.classify(s.sym, s.dso) == "libc"
                    and reps[0].interval[0] <= s.t <= reps[0].interval[1]]
    assert main_libc_in == []


def test_long_straddle_excludes_the_repetition():
    lines = _stderr(861107.8435)
    reps = phases.timed_reps(phases.segments(lines), {9: [REP1_NS + 22_000_000] + SAMPLES_T9[9][1:]})
    perfagg.aggregate(lambda: iter(_samples()), reps)
    assert reps[0].excluded and "straddles" in reps[0].excluded


def test_state_restore_sample_inside_an_interval_drops_the_repetition():
    mk = lambda t, tid, sym, dso: P.Sample(pid=MAIN, tid=tid, t=t, ip="0", sym=sym, dso=dso)
    gg = "/x/bin/libggml-cpu.so.0.16.0"
    ss = [mk(10.001, MAIN, "ggml_vec_dot_f16", gg),
          mk(10.004, MAIN, "llama_kv_cache::state_read_data(llama_io_read_i&)", "/x/bin/libllama.so"),
          mk(10.007, MAIN, "ggml_compute_forward_top_k", gg), mk(10.007, MAIN + 1, "[unknown]", "/x/libgomp.so.1")]
    segs = phases.segments([(10.0, "llama-bench: benchmark 1/1: prompt run 1/1"),
                            (10.5, "llama-bench: benchmark 1/1: depth run 2/2 (cached)")])
    reps = phases.timed_reps(segs, {1: [20_000_000]})
    out = perfagg.aggregate(lambda: iter(ss), reps)[1]
    assert out["reps_used"] == 0 and "state_restore" in out["reps_excluded"][0]["why"]


def test_worker_libc_copy_inside_the_step_stays_in_the_denominator():
    reps = phases.timed_reps(phases.segments(_stderr(861107.8435)), SAMPLES_T9)
    agg = perfagg.aggregate(lambda: iter(_samples()), reps)[9]
    assert agg["counts"]["libc"] >= 2 and agg["share"]["libc"] > 0
    assert agg["counts"]["fa"] >= 4 and agg["active_tids"]["fa"] >= 3
    assert agg["samples"] == sum(agg["counts"].values())


def test_first_rep_after_a_noncached_prefill_is_perf_uncovered():
    lines = [(1.0, "llama-bench: benchmark 1/2: starting"), (1.1, "llama-bench: benchmark 1/2: warmup prompt run"),
             (1.2, "llama-bench: benchmark 1/2: depth run 1/2"), (90.0, "llama-bench: benchmark 1/2: prompt run 1/2"),
             (90.2, "llama-bench: benchmark 1/2: depth run 2/2 (cached)"),
             (90.3, "llama-bench: benchmark 1/2: prompt run 2/2"), (90.5, "llama-bench: benchmark 2/2: starting")]
    log = [(1.0, "enable", 1.03), (1.2, "disable", 1.21), (90.0, "enable", 90.04)]
    reps = phases.timed_reps(phases.segments(lines), {1: [50_000_000, 50_000_000]}, control_log=log)
    assert [r.perf_covered for r in reps] == [False, True]
    out = perfagg.aggregate(lambda: iter([]), reps)[1]
    assert out["reps_used"] == 1 and "not enabled" in out["reps_excluded"][0]["why"]


# ------------------------------------------------------------------------------------------- (c) per-eval attribution
EVAL_FILE = """# ggml-cpu-prof-eval/1 units=us clock=CLOCK_MONOTONIC
S\t00000000000000aa\t12
N\t0\tRMS_NORM\tattn_norm-3\tinp\t-\t2560\t2\t1\t1
N\t1\tMUL_MAT\t\tblk.3.index_k_proj.weight\tattn_norm-3\t128\t2\t1\t1
N\t2\tMUL_MAT\tindexer_score-3\tindexer_k-3\t\t512\t2\t1\t1
N\t3\tADD\t\tindexer_score-3\tbias\t512\t2\t1\t1
N\t4\tTOP_K\t\t\t-\t2048\t2\t1\t1
N\t5\tFILL\tkq_mask (filled)\tkq_mask\t-\t65536\t2\t1\t1
N\t6\tSET_ROWS\tkq_mask_top_k\tkq_mask (filled) (view)\tzeros\t65536\t2\t1\t1
N\t7\tADD\t\tkq_mask_top_k (view)\tkq_mask\t65536\t2\t1\t1
N\t8\tFLASH_ATTN_EXT\t__fattn__-3\tq-3\tcache_k_l3 (view)\t256\t2\t1\t1
N\t9\tSET_ROWS\tcache_k_l3 (view)\tcache_k_l3\tk-3\t256\t65536\t1\t1
N\t10\tMUL_MAT_ID\tffn_moe_up-3\tblk.3.ffn_up_exps.weight\t-\t512\t8\t2\t1
N\t11\tGATED_DELTA_NET\t\tq-2\tk-2\t128\t2\t1\t1
"""


def _eval(gi, t_open, walls, tot_extra=0):
    w = "\t".join(f"{i}:{v}:{v // 2}" for i, v in enumerate(walls) if v)
    return (f"E\t{gi}\t1\t00000000000000aa\t12\t48\t{t_open}\t{t_open + sum(walls) + tot_extra}\t{sum(walls) + tot_extra}"
            f"\t{sum(walls) // 2}\nW\t{w}\n")


LO = [10, 20, 30, 5, 5, 10, 5, 5, 100, 5, 400, 50]          # d512: 645 us of node wall
HI = [10, 20, 150, 40, 60, 200, 80, 90, 900, 5, 400, 50]    # d64k


def _m_arm():
    ev = EVAL_FILE
    ev += _eval(0, 1_000_000, LO)                               # test 1 warmup (excluded)
    ev += _eval(1, 1_100_000, LO)                               # test 1 depth prefill (excluded)
    ev += _eval(2, 2_000_000, LO, tot_extra=15)                 # test 1 rep 1 (15 us in solo runs)
    ev += _eval(3, 3_000_000, LO, tot_extra=15)                 # test 1 rep 2
    ev += _eval(4, 4_000_000, HI)                               # test 2 warmup
    ev += _eval(5, 5_000_000, HI, tot_extra=15)                 # test 2 rep 1
    ev += _eval(6, 6_000_000, HI, tot_extra=15)                 # test 2 rep 2
    ev += "T\t0\t7\n"
    ef = evalprof.parse_eval_file(ev.splitlines(keepends=True))

    def ph(t, gi, n, ms_apply, ms_set, ms_gc):
        return [(t + 0.0003, f"[input_prof] n_tokens={n} us=400 class=24llm_graph_input_attn_kv"),
                (t + 0.0004, f"[input_prof] n_tokens={n} us=150 class=N14llama_model_qwen4exp9ple_inputE"),
                (t + 0.0011, f"[cpu_prof] graph_eval idx={gi} n_nodes=12 nth=48 acc=1"),
                (t + 0.0020, f"[phase_prof] n_tokens={n} mctx_apply={ms_apply} ms set_inputs={ms_set} ms "
                             f"graph_compute={ms_gc} ms")]
    L = [(0.9, "llama-bench: benchmark 1/2: starting"), (0.95, "llama-bench: benchmark 1/2: warmup prompt run"),
         (1.0005, "[cpu_prof] graph_eval idx=0 n_nodes=12 nth=48 acc=1"),
         (1.05, "llama-bench: benchmark 1/2: depth run 1/2"), (1.1005, "[cpu_prof] graph_eval idx=1 n_nodes=12 nth=48 acc=1"),
         (1.99, "llama-bench: benchmark 1/2: prompt run 1/2")] + ph(1.999, 2, 2, "0.100", "0.700", "0.700") + \
        [(2.9, "llama-bench: benchmark 1/2: depth run 2/2 (cached)"), (2.99, "llama-bench: benchmark 1/2: prompt run 2/2")] + \
        ph(2.999, 3, 2, "0.100", "0.700", "0.700") + \
        [(3.9, "llama-bench: benchmark 2/2: starting"), (3.95, "llama-bench: benchmark 2/2: warmup prompt run"),
         (4.0005, "[cpu_prof] graph_eval idx=4 n_nodes=12 nth=48 acc=1"),
         (4.5, "llama-bench: benchmark 2/2: depth run 1/2 (cached)"),
         (4.99, "llama-bench: benchmark 2/2: prompt run 1/2")] + ph(4.999, 5, 2, "0.300", "1.200", "2.100") + \
        [(5.9, "llama-bench: benchmark 2/2: depth run 2/2 (cached)"), (5.99, "llama-bench: benchmark 2/2: prompt run 2/2")] + \
        ph(5.999, 6, 2, "0.300", "1.200", "2.100") + [(6.5, "llama-bench: benchmark 2/2: done")]
    samples = {1: [1_600_000, 1_600_000], 2: [3_700_000, 3_700_000]}
    return ef, phases.segments(L), samples


def test_eval_file_parses_and_classifies_nodes():
    ef, _segs, _s = _m_arm()
    nodes = ef.shapes["00000000000000aa"]
    assert [n.cls for n in nodes] == ["norm", "matmul", "indexer", "indexer", "topk", "mask_build", "mask_build",
                                      "mask_build", "fa", "kv_store", "matmul", "ssm_gdn"]
    assert nodes[3].label == "after:indexer_score-3+1" and nodes[1].label == "after:attn_norm-3+1"
    assert len(ef.evals) == 7 and not ef.truncated and ef.evals[5].nodes[8] == (900, 450)


def test_only_timed_evaluations_are_attributed_and_classes_sum():
    ef, segs, samples = _m_arm()
    reps = evalprof.attribute(segs, ef, samples)
    assert [(r.test, r.rep, r.evals, r.excluded) for r in reps] == [(1, 1, [2], None), (1, 2, [3], None),
                                                                     (2, 1, [5], None), (2, 2, [6], None)]
    roll = evalprof.rollup(reps)
    hi = roll[2]
    assert hi["node_wall_us"]["fa"] == 900 and hi["node_wall_us"]["mask_build"] == 370
    assert hi["node_wall_us"]["indexer"] == 190 and hi["node_wall_us"]["topk"] == 60
    assert hi["solo_wall_us"] == 15 and hi["t_step_us"] == 3700
    assert hi["share"]["fa"] == pytest.approx(900 / 3700, abs=1e-5)
    assert hi["host_us"]["mctx_apply"] == pytest.approx(300) and hi["host_us"]["input:24llm_graph_input_attn_kv"] == 400
    assert hi["unattributed_us"] == pytest.approx(3700 - (300 + 1200 + 2100))
    assert hi["receipt_lag_ms"]["max"] == pytest.approx(0.1, abs=1e-3)       # line read 0.1 ms after t_open
    assert hi["evals_per_rep"] == [1]


def test_rep_with_unaccumulated_or_missing_eval_is_excluded():
    ef, segs, samples = _m_arm()
    ef.evals[6].acc = 0
    del ef.evals[3]
    reps = {(r.test, r.rep): r for r in evalprof.attribute(segs, ef, samples)}
    assert "missing" in reps[(1, 2)].excluded and "acc=0" in reps[(2, 2)].excluded
    assert evalprof.rollup(list(reps.values()))[2]["reps_used"] == 1


def test_top_nodes_name_the_depth_growing_nodes():
    ef, segs, samples = _m_arm()
    reps = evalprof.attribute(segs, ef, samples)
    top = evalprof.top_nodes(ef, [r for r in reps if r.test == 2], [r for r in reps if r.test == 1], k=3)
    assert [(t["idx"], t["cls"], t["delta_us"]) for t in top] == [(8, "fa", 800.0), (5, "mask_build", 190.0),
                                                                  (2, "indexer", 120.0)]


# ------------------------------------------------------------------------------------------- the predeclared rule
def _grid(fa_share_hi, s_hi_ms=100.0, s_lo_ms=50.0, overhead=0.01, cov=0.02, reps=10):
    g_rows, m_rows, roll = [], [], {}
    i = 0
    for d in (512, 65536):
        for N in rule.NS:
            i += 1
            tg = (s_lo_ms if d == 512 else s_hi_ms) * 1e6
            g_rows.append(dict(n_prompt=N, n_depth=d, avg_ns=tg))
            m_rows.append(dict(n_prompt=N, n_depth=d, avg_ns=tg * (1 + overhead)))
            T = tg * (1 + overhead) / 1e3
            fa = (fa_share_hi if d == 65536 else 0.02) * T
            mask = (0.20 if d == 65536 else 0.01) * T
            roll[i] = dict(reps_used=reps, t_step_us=T, coverage_defect=cov, share=dict(fa=fa / T),
                           node_wall_us=dict(fa=fa, mask_build=mask, matmul=0.3 * T), solo_wall_us=0.0,
                           eval_span_us=0.9 * T, host_us=dict(mctx_apply=10.0, set_inputs=20.0, graph_compute=0.9 * T),
                           unattributed_us=0.0)
    return g_rows, m_rows, roll


def test_rule_lane_large():
    out = rule.decide(*_grid(0.12))
    assert out["verdict"] == "LANE_LARGE" and out["max_F_N2to5"] == pytest.approx(0.12)


def test_rule_lane_small():
    out = rule.decide(*_grid(0.05, s_hi_ms=52.0, s_lo_ms=50.0))
    assert out["verdict"] == "LANE_SMALL" and out["max_S_N2to5"] < 0.10


def test_rule_split_names_the_seed_class():
    out = rule.decide(*_grid(0.06))
    assert out["verdict"] == "SPLIT_UNRESOLVED" and out["seed_class"] == "node:mask_build"
    assert out["mean_depth_delta_share_N2to5"]["node:mask_build"] == pytest.approx(0.20 - 0.01 * 0.5, abs=1e-4)


@pytest.mark.parametrize("kw,needle", [(dict(overhead=0.08), "overhead"), (dict(cov=0.2), "coverage"),
                                       (dict(reps=6), "usable M repetitions")])
def test_rule_invalid(kw, needle):
    out = rule.decide(*_grid(0.12, **kw))
    assert out["verdict"] == "INVALID" and any(needle in w for w in out["why"])


def test_rule_invalid_on_missing_cell_and_failed_step():
    g, m, roll = _grid(0.12)
    out = rule.decide(g[:-1], m, roll)
    assert out["verdict"] == "INVALID" and any("missing" in w for w in out["why"])
    assert rule.decide(*_grid(0.12), steps_ok=False)["verdict"] == "INVALID"


# ------------------------------------------------------------------------------------------- file-level round trip
def test_analyze_cli_round_trip(tmp_path):
    import json
    from akx_lb3_attribution import analyze
    # G arm: the recorded perf lines + a 9-row jsonl whose 9th row is the recorded test
    g = tmp_path / "G-t48"
    (g.parent / "G-t48.lines.tsv").write_text("".join(f"{t:.6f}\t{x}\n" for t, x in _stderr(861107.8435)))
    (g.parent / "G-t48.perfscript.txt").write_text("\n".join(PERF_LINES) + "\n")
    rows = [dict(n_prompt=1, n_depth=512, avg_ns=1, samples_ns=[1])] * 8 + \
           [dict(n_prompt=3, n_depth=65536, avg_ns=102005094, samples_ns=SAMPLES_T9[9])]
    (g.parent / "G-t48.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert analyze.main(["--arm", "G", "--prefix", str(g)]) == 0
    prof = json.loads((g.parent / "G-t48.profile.json").read_text())
    r9 = prof["rows"][8]
    assert r9["reps_used"] == 2 and "state_restore" not in r9["share"] and r9["share"]["fa"] > 0
    # M arm
    ef_text = EVAL_FILE + _eval(2, 2_000_000, LO, 15) + _eval(5, 5_000_000, HI, 15) + "T\t0\t2\n"
    m = tmp_path / "M-t48"
    (tmp_path / "M-t48.evals.tsv").write_text(ef_text)
    L = [(1.99, "llama-bench: benchmark 1/2: prompt run 1/1"), (2.0011, "[cpu_prof] graph_eval idx=2 n_nodes=12 nth=48 acc=1"),
         (2.002, "[phase_prof] n_tokens=2 mctx_apply=0.100 ms set_inputs=0.700 ms graph_compute=0.700 ms"),
         (4.99, "llama-bench: benchmark 2/2: prompt run 1/1"), (5.0011, "[cpu_prof] graph_eval idx=5 n_nodes=12 nth=48 acc=1"),
         (5.002, "[phase_prof] n_tokens=2 mctx_apply=0.300 ms set_inputs=1.200 ms graph_compute=2.100 ms"),
         (6.0, "llama-bench: benchmark 2/2: done")]
    (tmp_path / "M-t48.lines.tsv").write_text("".join(f"{t:.6f}\t{x}\n" for t, x in L))
    (tmp_path / "M-t48.jsonl").write_text(json.dumps(dict(n_prompt=2, n_depth=512, avg_ns=1_600_000, samples_ns=[1_600_000]))
                                          + "\n" + json.dumps(dict(n_prompt=2, n_depth=65536, avg_ns=3_700_000,
                                                                   samples_ns=[3_700_000])) + "\n")
    assert analyze.main(["--arm", "M", "--prefix", str(m)]) == 0
    pm = json.loads((tmp_path / "M-t48.profile.json").read_text())
    assert pm["rows"][1]["share"]["fa"] == pytest.approx(900 / 3700, abs=1e-5)
    assert pm["top_depth_growing_nodes"]["2"][0]["cls"] == "fa"


# ------------------------------------------------------------------------------------------- belief sidecar (VB-AKX-LB32-WRITE)
def _verdict_file(tmp_path, fa_hi, t96_fa=None):
    import json
    gating = rule.decide(*_grid(fa_hi))
    res = dict(item="akx-lb3-2-attribution", gating=gating, verdict=gating["verdict"])
    if t96_fa is not None:
        res["reported_t96"] = rule.decide(*_grid(t96_fa))
    vp = tmp_path / "VERDICT.json"
    vp.write_text(json.dumps(res, indent=1))           # JSON round trip: cell keys become strings, as on disk
    pp = tmp_path / "PREDECLARATION.md"
    pp.write_text("# frozen rule\n")
    return vp, pp


def test_sidecar_rows_from_a_fixture_verdict(tmp_path):
    import hashlib
    import json
    from akx_lb3_attribution import beliefs
    vp, pp = _verdict_file(tmp_path, 0.06, t96_fa=0.07)
    side = beliefs.write_sidecar(vp, pp, run_id="t", emitted_at="2026-10-24T12:00:00Z")
    rows = [json.loads(l) for l in side.read_text().splitlines()]
    cells = [r for r in rows if r["extra"]["kind"] == "cell"]
    verd = [r for r in rows if r["extra"]["kind"] == "verdict"]
    assert len(cells) == 12 and len(verd) == 1                     # 6 N x (t48 gating + t96 reported)
    assert {(r["extra"]["threads"], r["extra"]["n_prompt"]) for r in cells} == {(t, n) for t in (48, 96) for n in rule.NS}
    v = verd[0]
    assert v["extra"]["verdict"] == "SPLIT_UNRESOLVED" and v["extra"]["seed_class"] == "node:mask_build"
    assert v["value"] == pytest.approx(0.06) and v["reps"] == 10
    pre = hashlib.sha256(pp.read_bytes()).hexdigest()
    for r in rows:
        assert r["extra"]["predeclaration_sha256"] == pre
        assert r["results_sha256"] == hashlib.sha256(vp.read_bytes()).hexdigest()
        assert r["protocol_id"] == "" and r["schema"] == beliefs.CAPTURE_SCHEMA
        assert r["row_sha256"] == beliefs.content_hash({k: x for k, x in r.items() if k != "row_sha256"})
    assert len({r["measurement_id"] for r in rows}) == len(rows)
    c = next(r for r in cells if r["extra"]["threads"] == 48 and r["extra"]["n_prompt"] == 3)
    assert c["value"] == pytest.approx(0.06) and c["extra"]["gating"] is True and c["extra"]["S_depth_share"] == 0.5


def test_invalid_run_emits_zero_rows_and_removes_a_stale_sidecar(tmp_path):
    from akx_lb3_attribution import beliefs
    vp, pp = _verdict_file(tmp_path, 0.12)
    assert beliefs.write_sidecar(vp, pp, run_id="t") is not None
    import json
    g, m, roll = _grid(0.12, overhead=0.08)
    vp.write_text(json.dumps(dict(gating=rule.decide(g, m, roll), verdict="INVALID")))
    assert beliefs.write_sidecar(vp, pp, run_id="t") is None
    assert not (tmp_path / beliefs.SIDECAR_NAME).exists()


def test_invalid_t96_report_keeps_only_t48_cells(tmp_path):
    import json
    from akx_lb3_attribution import beliefs
    vp, pp = _verdict_file(tmp_path, 0.12)
    d = json.loads(vp.read_text())
    g, m, roll = _grid(0.12, reps=6)
    d["reported_t96"] = rule.decide(g, m, roll)
    vp.write_text(json.dumps(d))
    rows = [json.loads(l) for l in beliefs.write_sidecar(vp, pp, run_id="t").read_text().splitlines()]
    assert {r["extra"]["threads"] for r in rows} == {48} and len(rows) == 7
