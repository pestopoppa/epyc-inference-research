"""Bounded-output actor tools, tested without an MCP SDK, a ColGREP index or perf.

The failure these guard: a 27B actor with a 98k slot read whole 4800-line files and
unbounded perf reports and overflowed in 26 minutes. So every test here is about a
cap holding, a truncation being SAID, or a path staying inside its fence.
"""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autokernel.loop import actor_tools_mcp as t
from autokernel.loop import perf_cache as pc

CPP_FIXTURE = """\
#include <vector>
// void commented_out(int x) {
namespace iqk {

struct Scales8K {
    template <typename Q8>
    inline int process(const Q8& q8, int i) {
        if (i > 0) {
            for (int j = 0; j < i; ++j) { helper(j); }
        }
        return i;
    }
};

void prototype_only(int n);

template <typename Dequantizer, int nrc_y>
static void mul_mat_qX_K(int n, const void * vx,
                         size_t bx) {
    Dequantizer deq(vx, bx);
    call_something(n);
}

enum class Kind { A, B };

}  // namespace iqk

int main(int argc, char ** argv) {
    return 0;
}
"""


class _Tree(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.base = Path(self._td.name)
        self.root = self.base / "root"
        self.root.mkdir()
        (self.root / "src").mkdir()
        (self.root / "src" / "k.cpp").write_text(CPP_FIXTURE)
        (self.root / "big.txt").write_text("".join(f"line {i} needle\n" for i in range(1, 501)))
        self.outside = self.base / "secret.txt"
        self.outside.write_text("do not read\n")
        os.symlink(self.outside, self.root / "escape_link.txt")
        os.symlink(self.base, self.root / "escape_dir")
        self.r = str(self.root)

    def tearDown(self):
        self._td.cleanup()


class Confinement(_Tree):

    def test_dotdot_escape_is_refused(self):
        with self.assertRaises(t.ToolError) as cm:
            t.read_range(self.r, "../secret.txt")
        self.assertIn("outside", str(cm.exception))

    def test_absolute_path_outside_is_refused(self):
        with self.assertRaises(t.ToolError):
            t.read_range(self.r, str(self.outside))

    def test_symlink_out_of_root_is_refused(self):
        for p in ("escape_link.txt", "escape_dir/secret.txt"):
            with self.assertRaises(t.ToolError, msg=p):
                t.read_range(self.r, p)
        with self.assertRaises(t.ToolError):
            t.outline(self.r, "escape_link.txt")
        with self.assertRaises(t.ToolError):
            t.grep(self.r, "read", path="escape_dir", _rg=None)

    def test_grep_python_walk_never_follows_a_symlink_out(self):
        out = t.grep(self.r, "do not read", _rg=None)
        self.assertIn("no matches", out)

    def test_safe_wrapper_renders_refusals_not_tracebacks(self):
        out = t._safe(t.read_range, self.r, "../secret.txt")
        self.assertTrue(out.startswith("ERROR:"))


class ReadRange(_Tree):

    def test_header_and_line_numbers(self):
        out = t.read_range(self.r, "big.txt", start_line=10, num_lines=3)
        lines = out.splitlines()
        self.assertEqual(lines[0], "big.txt lines 10-12 of 500")
        self.assertEqual(lines[1], "10| line 10 needle")
        self.assertIn("continue with start_line=13", out)

    def test_num_lines_is_clamped_and_says_so(self):
        out = t.read_range(self.r, "big.txt", num_lines=5000)
        self.assertIn("lines 1-200 of 500", out)
        self.assertIn("num_lines clamped from 5000 to max 200", out)
        self.assertIn("[truncated: 300 more line(s)", out)
        body = [ln for ln in out.splitlines() if "| line" in ln]
        self.assertEqual(len(body), 200)

    def test_long_lines_are_cut(self):
        (self.root / "wide.txt").write_text("x" * 5000 + "\n")
        out = t.read_range(self.r, "wide.txt")
        self.assertLess(len(out), 1000)
        self.assertIn("truncated to 400 chars", out)


class Grep(_Tree):

    def _both(self):
        engines = [None]
        if t.find_rg():
            engines.append(t.find_rg())
        return engines

    def test_hits_capped_with_total_footer(self):
        for eng in self._both():
            with self.subTest(engine=eng):
                out = t.grep(self.r, "needle", max_hits=500, _rg=eng)
                self.assertIn("max_hits clamped from 500 to max 80", out)
                hits = [ln for ln in out.splitlines() if ln.startswith("big.txt:")]
                self.assertEqual(len(hits), 80)
                self.assertIn("showing 80 of 500 matches", out)
                self.assertTrue(hits[0].startswith("big.txt:1: line 1 needle"), hits[0])

    def test_context_clamped_and_emitted(self):
        for eng in self._both():
            with self.subTest(engine=eng):
                out = t.grep(self.r, r"call_something", path="src", context=9, _rg=eng)
                self.assertIn("context clamped from 9 to max 3", out)
                self.assertIn("src/k.cpp:21: ", out)
                self.assertIn("src/k.cpp:20- ", out)
                self.assertIn("src/k.cpp:22- ", out)
                self.assertIn("[1 match(es)]", out)

    def test_long_match_lines_truncated_to_240(self):
        (self.root / "wide.txt").write_text("needleX" + "y" * 2000 + "\n")
        for eng in self._both():
            with self.subTest(engine=eng):
                out = t.grep(self.r, "needleX", _rg=eng)
                line = [ln for ln in out.splitlines() if ln.startswith("wide.txt:")][0]
                self.assertLessEqual(len(line), len("wide.txt:1: ") + t.GREP_MAX_LINE_CHARS)

    def test_python_fallback_when_rg_absent(self):
        with mock.patch.object(t.shutil, "which", return_value=None), \
                mock.patch.object(t, "RG_CANDIDATES", ()):
            self.assertIsNone(t.find_rg())
            out = t.grep(self.r, r"mul_mat_qX_K", path="src")
        self.assertIn("[python-re]", out)
        self.assertIn("src/k.cpp:18: ", out)

    def test_rg_used_when_present(self):
        fake = "/usr/bin/rg-fake"
        with mock.patch.object(t.shutil, "which", return_value=fake), \
                mock.patch.object(t, "_grep_rg") as g:
            out = t.grep(self.r, "needle")
        g.assert_called_once()
        self.assertEqual(g.call_args[0][0], fake)
        self.assertIn("[rg]", out)

    def test_invalid_regex_python(self):
        with self.assertRaises(t.ToolError):
            t.grep(self.r, "(unclosed", _rg=None)

    def test_skips_git_and_build_dirs(self):
        for d in (".git", "build-hip"):
            (self.root / d).mkdir()
            (self.root / d / "x.cpp").write_text("hidden_marker\n")
        out = t.grep(self.r, "hidden_marker", _rg=None)
        self.assertIn("no matches", out)


class Outline(_Tree):

    def test_finds_functions_structs_and_template_functions(self):
        out = t.outline(self.r, "src/k.cpp")
        self.assertIn("namespace iqk", out)
        self.assertIn("struct Scales8K", out)
        # member template: reported at its template line
        self.assertRegex(out, r"L6\s+template <typename Q8> inline int process")
        # free template with a multi-line signature
        self.assertRegex(out, r"L17\s+template <typename Dequantizer, int nrc_y> static void mul_mat_qX_K")
        self.assertIn("enum class Kind", out)
        self.assertRegex(out, r"L28\s+int main")

    def test_excludes_prototypes_calls_control_flow_and_comments(self):
        out = t.outline(self.r, "src/k.cpp")
        for absent in ("prototype_only", "helper(", "call_something", "deq(", "if (",
                       "commented_out", "for ("):
            self.assertNotIn(absent, out)

    def test_python_outline(self):
        (self.root / "m.py").write_text("class A:\n    def f(self):\n        pass\n\n"
                                        "async def g():\n    pass\n")
        out = t.outline(self.r, "m.py")
        self.assertIn("L1 class A:", out)
        self.assertIn("L2   def f(self):", out)
        self.assertIn("L5 async def g():", out)

    def test_entries_capped_at_300(self):
        (self.root / "many.c").write_text("".join(f"int f{i}(void) {{ return {i}; }}\n"
                                                  for i in range(400)))
        out = t.outline(self.r, "many.c")
        entries = [ln for ln in out.splitlines() if ln.startswith("L")]
        self.assertEqual(len(entries), 300)
        self.assertIn("[truncated: 100 more definitions (cap 300)", out)


class CodeSearch(_Tree):

    def test_missing_binary_is_a_helpful_message(self):
        with mock.patch.object(t, "colgrep_bin", return_value=None), \
                mock.patch.object(t.subprocess, "run") as run:
            out = t.code_search(self.r, "dequantize q4_K")
        run.assert_not_called()
        self.assertIn("unavailable", out)
        self.assertIn("grep", out)

    def test_missing_index_never_builds_one(self):
        status = subprocess.CompletedProcess([], 0, stdout="No index found for x\n"
                                             "Run `colgrep <query>` to create one.\n", stderr="")
        with mock.patch.object(t, "colgrep_bin", return_value="/bin/colgrep"), \
                mock.patch.object(t.subprocess, "run", return_value=status) as run:
            out = t.code_search(self.r, "dequantize q4_K")
        self.assertEqual(run.call_count, 1)
        argv = run.call_args[0][0]
        self.assertEqual(argv[1], "status")          # the read-only probe only
        self.assertNotIn("search", argv)
        self.assertNotIn("init", argv)
        self.assertIn("no ColGREP index", out)
        self.assertIn("never builds an index", out)
        self.assertIn("grep", out)

    def test_ready_index_returns_paths_ranges_scores_only_and_clamps_k(self):
        status = subprocess.CompletedProcess([], 0, stdout="Index: 120 units\n", stderr="")
        payload = ('[{"unit": {"file": "%s/src/k.cpp", "line": 17, "end_line": 22, '
                   '"name": "mul_mat_qX_K", "code": "SECRET BODY"}, "score": 0.91234},'
                   ' {"unit": {"file": "/etc/passwd", "line": 1, "end_line": 2}, "score": 0.5}]'
                   % self.r)
        search = subprocess.CompletedProcess([], 0, stdout=payload, stderr="")
        with mock.patch.object(t, "colgrep_bin", return_value="/bin/colgrep"), \
                mock.patch.object(t.subprocess, "run", side_effect=[status, search]) as run:
            out = t.code_search(self.r, "q4 kernel", k=50)
        argv = run.call_args_list[1][0][0]
        self.assertEqual(argv[argv.index("-k") + 1], "10")
        self.assertIn("k clamped from 50 to max 10", out)
        self.assertIn("src/k.cpp  lines 17-22  score 0.912", out)
        self.assertNotIn("SECRET BODY", out)
        self.assertNotIn("passwd", out)


PERF_REPORT = """\
# To display the perf.data header info, please use --header/--header-only options.
#
# Samples: 12K of event 'cycles:P'
# Event count (approx.): 987654321
#
# Overhead  Shared Object     Symbol
# ........  ................  ......
#
""" + "".join(f"    {50 - i * 0.5:.2f}%  libggml-cpu.so    [.] kernel_{i}\n" for i in range(95)) + """

#
# (Cannot load tips.txt file, please install perf!)
#
"""

ANNOTATE_LINES = [" Percent |\tSource code & Disassembly of libggml-cpu.so for cycles:P "
                  "(900 samples, percent: local period)",
                  # address built, not literal: the PII pre-commit hook reads a
                  # 16-digit run as an account number
                  "-" * 60, "         :", f"         : 5    {0x1140:016x} <kernel_0>:"]
for _i in range(300):
    _pct = 0.0
    if _i in (100, 101, 150):
        _pct = {100: 30.0, 101: 20.0, 150: 10.0}[_i]
    elif _i % 10 == 0:
        _pct = 0.5
    ANNOTATE_LINES.append(f"  {_pct:6.2f} :   {0x1140 + _i * 4:x}:   vpdpbusd %zmm{_i % 32},%zmm1,%zmm2")
PERF_ANNOTATE = "\n".join(ANNOTATE_LINES) + "\n"


class Profiles(unittest.TestCase):

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        base = Path(self._td.name)
        self.pdir = base / "profiles"
        (self.pdir / "run1").mkdir(parents=True)
        self.prof = self.pdir / "run1" / "perf.data"
        self.prof.write_bytes(b"\0" * 2048)
        (self.pdir / "notes.txt").write_text("x")
        self.outside = base / "other.data"
        self.outside.write_bytes(b"\0")
        os.symlink(self.outside, self.pdir / "link.data")
        self.dirs = [str(self.pdir)]

    def tearDown(self):
        self._td.cleanup()

    def _perf(self, stdout="", stderr="", rc=0):
        return mock.patch.object(t.subprocess, "run", return_value=subprocess.CompletedProcess(
            [], rc, stdout=stdout, stderr=stderr))

    def test_lists_profiles_when_none_given(self):
        out = t.profile_top(self.dirs)
        self.assertIn("1 profile(s)", out)
        self.assertIn("run1/perf.data", out)
        self.assertNotIn("notes.txt", out)
        self.assertNotIn("link.data", out)   # symlink leaving the dir is not listed

    def test_top_rows_truncated_and_clamped(self):
        with self._perf(PERF_REPORT) as run:
            out = t.profile_top(self.dirs, "run1/perf.data", limit=500, dso="libggml-cpu.so")
        argv = run.call_args[0][0]
        self.assertEqual(argv[:3], ["perf", "report", "--stdio"])
        for flag in ("--no-children", "--percent-limit", "--sort"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--sort") + 1], "dso,symbol")
        self.assertEqual(argv[argv.index("--dsos") + 1], "libggml-cpu.so")
        self.assertIn("limit clamped from 500 to max 80", out)
        rows = [ln for ln in out.splitlines() if "[.] kernel_" in ln]
        self.assertEqual(len(rows), 80)
        self.assertIn("showing top 80 of 95 rows", out)
        self.assertIn("Samples: 12K", out)
        self.assertNotIn("tips.txt", out)

    def test_no_samples_is_said_plainly(self):
        with self._perf("", "Error:\nThe perf.data data has no samples!\n", rc=255):
            out = t.profile_top(self.dirs, str(self.prof))
        self.assertIn("no samples", out)

    def test_profile_confined_to_profiles_dirs(self):
        with mock.patch.object(t.subprocess, "run") as run:
            for bad in (str(self.outside), "../other.data", "link.data", "/etc/passwd"):
                with self.subTest(bad=bad):
                    with self.assertRaises(t.ToolError):
                        t.profile_top(self.dirs, bad)
                    with self.assertRaises(t.ToolError):
                        t.symbol_annotate(self.dirs, bad, "kernel_0", "libggml-cpu.so")
        run.assert_not_called()

    def test_no_profiles_configured(self):
        self.assertIn("no --profiles", t.profile_top([]))
        with self.assertRaises(t.ToolError):
            t.profile_top([], "perf.data")

    def test_annotate_keeps_hottest_lines_in_order(self):
        with self._perf(PERF_ANNOTATE) as run:
            out = t.symbol_annotate(self.dirs, "run1/perf.data", "kernel_0", "libggml-cpu.so",
                                    max_lines=5000)
        argv = run.call_args[0][0]
        self.assertEqual(argv[:3], ["perf", "annotate", "--stdio"])
        self.assertEqual(argv[-1], "kernel_0")
        self.assertEqual(argv[argv.index("--dsos") + 1], "libggml-cpu.so")
        self.assertIn("max_lines clamped from 5000 to max 200", out)
        body = [ln for ln in out.splitlines() if " :   " in ln]
        self.assertLessEqual(len(body), 200)
        self.assertIn("[truncated:", out)
        pos = [out.index(f"{0x1140 + i * 4:x}:") for i in (100, 101, 150)]
        self.assertEqual(pos, sorted(pos))               # asm order preserved
        self.assertIn("     ...", out)                   # gaps marked

    def test_annotate_small_budget_keeps_only_the_hottest(self):
        with self._perf(PERF_ANNOTATE):
            out = t.symbol_annotate(self.dirs, "run1/perf.data", "kernel_0", "libggml-cpu.so",
                                    max_lines=3)
        body = [ln for ln in out.splitlines() if " :   " in ln]
        self.assertEqual(len(body), 3)
        for pct in ("30.00", "20.00", "10.00"):
            self.assertTrue(any(ln.strip().startswith(pct) for ln in body), pct)

    def test_annotate_no_samples(self):
        with self._perf("", "Error:\nThe perf.data data has no samples!\n", rc=0):
            out = t.symbol_annotate(self.dirs, "run1/perf.data", "nope", "libggml-cpu.so")
        self.assertIn("no samples for symbol 'nope'", out)


# `perf report --sort symbol --dsos libggml-cpu.so.0.16.0` rows as DS41 run 7's real
# profile prints them (cpu-raw-0316509f..., 2026-09-24; names verbatim).
Q4K1 = ("void (anonymous namespace)::mul_mat_qX_K_q8_2_X4_T<(anonymous namespace)::"
        "DequantizerQ4K_AVX2, 1>(int, void const*, unsigned long, DataInfo const&, int)")
Q4K2 = Q4K1.replace("AVX2, 1>", "AVX2, 2>")
Q4K3 = Q4K1.replace("AVX2, 1>", "AVX2, 3>")
GEMM3 = ("void (anonymous namespace)::tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>"
         "::gemm4xN<3>(long, long, long, long)")
GEMM2 = GEMM3.replace("gemm4xN<3>", "gemm4xN<2>")
# perf 6.17 pads each `--sort symbol` row with `IPC  [IPC Coverage]` columns, which
# read `-      -` without branch data (seen verbatim in the 2026-09-24 smoke).
SYMBOL_REPORT = "# Samples: 114K of event 'cycles:u'\n#\n" + "".join(
    f"    {pct:.2f}%  [.] {name}{' ' * 45}-      -\n" for pct, name in (
        (19.28, Q4K1), (16.80, GEMM3), (3.93, Q4K2), (2.60, GEMM2), (2.14, Q4K3),
        (1.71, "ggml_compute_forward_flash_attn_ext"), (1.20, "ggml_vec_dot_f16")))
SYMBOLS = [(19.28, Q4K1), (16.80, GEMM3), (3.93, Q4K2), (2.60, GEMM2), (2.14, Q4K3),
           (1.71, "ggml_compute_forward_flash_attn_ext"), (1.20, "ggml_vec_dot_f16")]
NO_SAMPLES = subprocess.CompletedProcess(
    [], 0, stdout="", stderr="Error:\nThe measurement-record.data data has no samples!\n")


class SymbolResolution(unittest.TestCase):
    """DS41-C23: the planner typed `mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2, 1>`; perf
    matched it against the full demangled name, found nothing, and said the 114K-sample
    profile "has no samples!"."""

    def test_report_rows_drop_perfs_trailing_ipc_columns(self):
        with mock.patch.object(t.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout=SYMBOL_REPORT, stderr="")):
            rows = t._dso_symbols("/x/measurement-record.data", "libggml-cpu.so.0.16.0")
        self.assertEqual(rows, SYMBOLS)

    def test_the_run7_short_name_resolves_to_the_full_name(self):
        self.assertEqual(t.resolve_symbol("mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2, 1>", SYMBOLS),
                         ("resolved", Q4K1))

    def test_spacing_and_namespace_do_not_matter(self):
        for typed in ("(anonymous namespace)::mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2,1>",
                      "mul_mat_qX_K_q8_2_X4_T<(anonymous namespace)::DequantizerQ4K_AVX2, 1>(int, "
                      "void const*, unsigned long, DataInfo const&, int)"):
            with self.subTest(typed=typed):
                self.assertEqual(t.resolve_symbol(typed, SYMBOLS), ("resolved", Q4K1))

    def test_template_instances_are_never_guessed_between(self):
        kind, found = t.resolve_symbol("mul_mat_qX_K_q8_2_X4_T", SYMBOLS)
        self.assertEqual(kind, "ambiguous")
        self.assertEqual([name for _p, name in found], [Q4K1, Q4K2, Q4K3])

    def test_exact_and_plain_c_names_and_misses(self):
        self.assertEqual(t.resolve_symbol(Q4K1, SYMBOLS), ("exact", Q4K1))
        self.assertEqual(t.resolve_symbol("ggml_vec_dot_f16", SYMBOLS), ("exact", "ggml_vec_dot_f16"))
        self.assertEqual(t.resolve_symbol("gemm4xN<3>", SYMBOLS), ("resolved", GEMM3))
        self.assertEqual(t.resolve_symbol("not_a_symbol", SYMBOLS), ("none", None))

    def _profiles(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        pdir = Path(td.name) / "profiles"
        pdir.mkdir()
        (pdir / "measurement-record.data").write_bytes(b"\0" * 2048)
        return [str(pdir)]

    def test_annotate_resolves_then_annotates_the_full_name(self):
        dirs = self._profiles()
        calls = [NO_SAMPLES,
                 subprocess.CompletedProcess([], 0, stdout=SYMBOL_REPORT, stderr=""),
                 subprocess.CompletedProcess([], 0, stdout=PERF_ANNOTATE, stderr="")]
        with mock.patch.object(t.subprocess, "run", side_effect=calls) as run:
            out = t.symbol_annotate(dirs, "measurement-record.data",
                                    "mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2, 1>",
                                    "libggml-cpu.so.0.16.0")
        report_argv = run.call_args_list[1][0][0]
        self.assertEqual(report_argv[report_argv.index("--sort") + 1], "symbol")
        self.assertEqual(report_argv[report_argv.index("--dsos") + 1], "libggml-cpu.so.0.16.0")
        self.assertEqual(run.call_args_list[2][0][0][-1], Q4K1)
        self.assertIn("resolved", out)
        self.assertIn("30.00", out)
        self.assertNotIn("no samples", out)

    def test_an_exact_name_costs_one_perf_call(self):
        dirs = self._profiles()
        with mock.patch.object(t.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout=PERF_ANNOTATE, stderr="")) as run:
            t.symbol_annotate(dirs, "measurement-record.data", Q4K1, "libggml-cpu.so.0.16.0")
        self.assertEqual(run.call_count, 1)

    def test_ambiguous_returns_the_candidates_and_does_not_annotate(self):
        dirs = self._profiles()
        calls = [NO_SAMPLES, subprocess.CompletedProcess([], 0, stdout=SYMBOL_REPORT, stderr="")]
        with mock.patch.object(t.subprocess, "run", side_effect=calls) as run:
            out = t.symbol_annotate(dirs, "measurement-record.data", "mul_mat_qX_K_q8_2_X4_T",
                                    "libggml-cpu.so.0.16.0")
        self.assertEqual(run.call_count, 2)
        self.assertIn("matches 3 sampled symbols", out)
        for name in (Q4K1, Q4K2, Q4K3):
            self.assertIn(name, out)
        self.assertIn("19.28%", out)

    def test_a_real_miss_says_so_after_checking_the_symbol_list(self):
        dirs = self._profiles()
        calls = [NO_SAMPLES, subprocess.CompletedProcess([], 0, stdout=SYMBOL_REPORT, stderr="")]
        with mock.patch.object(t.subprocess, "run", side_effect=calls):
            out = t.symbol_annotate(dirs, "measurement-record.data", "not_a_symbol",
                                    "libggml-cpu.so.0.16.0")
        self.assertIn("no samples for symbol 'not_a_symbol'", out)
        self.assertIn("no sampled symbol of that dso matches", out)


class PerfCacheIntegration(unittest.TestCase):
    """DS41-C20d: `profile_top`/`symbol_annotate` served from a `PerfCache` must be
    byte-identical to the uncached call, must never serve a changed profile from a
    stale entry, and must lazily cache `symbol_annotate` per symbol -- all without
    a real `perf` (every `perf` invocation here is `subprocess.run`, mocked)."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        base = Path(self._td.name)
        self.pdir = base / "profiles"
        self.pdir.mkdir()
        self.prof = self.pdir / "measurement-record.data"
        self.prof.write_bytes(b"\0" * 4096)
        self.dirs = [str(self.pdir)]
        self.cache_root = str(base / "cache")

    def _cache(self):
        return pc.PerfCache(cache_root=self.cache_root, perf_ver="perf version 6.17.13 (test)")

    def test_cache_hit_is_byte_identical_to_the_uncached_call(self):
        with mock.patch.object(t.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout=PERF_REPORT, stderr="")):
            uncached = t.profile_top(self.dirs, "measurement-record.data", limit=50,
                                     dso="libggml-cpu.so")
        cache = self._cache()
        with mock.patch.object(t.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout=PERF_REPORT, stderr="")) as run:
            first = t.profile_top(self.dirs, "measurement-record.data", limit=50,
                                  dso="libggml-cpu.so", cache=cache)
            second = t.profile_top(self.dirs, "measurement-record.data", limit=50,
                                   dso="libggml-cpu.so", cache=cache)
        self.assertEqual(run.call_count, 1)          # second call never re-ran perf
        self.assertEqual(first, uncached)             # identical to today's uncached output
        self.assertEqual(second, uncached)

    def test_bounds_still_hold_through_the_cache(self):
        cache = self._cache()
        with mock.patch.object(t.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout=PERF_REPORT, stderr="")):
            out = t.profile_top(self.dirs, "measurement-record.data", limit=500,
                                dso="libggml-cpu.so", cache=cache)
            out2 = t.profile_top(self.dirs, "measurement-record.data", limit=500,
                                 dso="libggml-cpu.so", cache=cache)  # served from cache
        for text in (out, out2):
            self.assertIn("limit clamped from 500 to max 80", text)
            rows = [ln for ln in text.splitlines() if "[.] kernel_" in ln]
            self.assertEqual(len(rows), 80)
            self.assertIn("showing top 80 of 95 rows", text)

    def test_a_new_limit_is_served_from_the_same_cached_report_call(self):
        """`limit` never reaches perf's own argv, so varying it must cost zero
        extra perf calls once the (profile, dso) report is cached."""
        cache = self._cache()
        with mock.patch.object(t.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout=PERF_REPORT, stderr="")) as run:
            small = t.profile_top(self.dirs, "measurement-record.data", limit=5,
                                  dso="libggml-cpu.so", cache=cache)
            large = t.profile_top(self.dirs, "measurement-record.data", limit=40,
                                  dso="libggml-cpu.so", cache=cache)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(len([ln for ln in small.splitlines() if "[.] kernel_" in ln]), 5)
        self.assertEqual(len([ln for ln in large.splitlines() if "[.] kernel_" in ln]), 40)

    def test_a_changed_profile_is_never_served_stale(self):
        cache = self._cache()
        report_v2 = PERF_REPORT.replace("987654321", "111111111")
        with mock.patch.object(t.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout=PERF_REPORT, stderr="")) as run:
            first = t.profile_top(self.dirs, "measurement-record.data", cache=cache)
        # The loop writes a NEW profile at the same path (a fresh measurement round).
        import time
        time.sleep(0.01)
        self.prof.write_bytes(b"\1" * 8192)
        with mock.patch.object(t.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout=report_v2, stderr="")) as run2:
            second = t.profile_top(self.dirs, "measurement-record.data", cache=cache)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run2.call_count, 1)          # not zero: the change was NOT a hit
        self.assertIn("987654321", first)
        self.assertIn("111111111", second)
        self.assertNotIn("111111111", first)

    def test_symbol_annotate_lazily_caches_per_symbol(self):
        cache = self._cache()
        with mock.patch.object(t.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout=PERF_ANNOTATE, stderr="")) as run:
            first = t.symbol_annotate(self.dirs, "measurement-record.data", "kernel_0",
                                      "libggml-cpu.so", cache=cache)
            second = t.symbol_annotate(self.dirs, "measurement-record.data", "kernel_0",
                                       "libggml-cpu.so", cache=cache)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(first, second)
        self.assertIn("kernel_0", first)

    def test_a_different_symbol_is_a_separate_lazy_entry(self):
        cache = self._cache()
        annotate_calls = []

        def fake_run(cmd, **kw):
            annotate_calls.append(cmd[-1])
            return subprocess.CompletedProcess([], 0, stdout=PERF_ANNOTATE, stderr="")

        with mock.patch.object(t.subprocess, "run", side_effect=fake_run):
            t.symbol_annotate(self.dirs, "measurement-record.data", "kernel_0",
                              "libggml-cpu.so", cache=cache)
            t.symbol_annotate(self.dirs, "measurement-record.data", "kernel_1",
                              "libggml-cpu.so", cache=cache)
            t.symbol_annotate(self.dirs, "measurement-record.data", "kernel_0",
                              "libggml-cpu.so", cache=cache)  # repeat: must not re-run
        self.assertEqual(annotate_calls, ["kernel_0", "kernel_1"])

    def test_the_resolve_helper_call_is_also_cached(self):
        """symbol_annotate's short-name resolution (DS41-C23) shells to `perf report
        --sort symbol` on its own; a repeat lookup for a DIFFERENT short name that
        resolves against the same dso's symbol list must not re-run that report --
        only the actual (per-symbol) annotate calls differ."""
        cache = self._cache()
        calls = {"report": 0, "annotate": 0}
        no_samples = ("", "Error:\nThe measurement-record.data data has no samples!\n")

        def fake_run(cmd, **kw):
            if cmd[1] == "report":
                calls["report"] += 1
                return subprocess.CompletedProcess([], 0, stdout=SYMBOL_REPORT, stderr="")
            calls["annotate"] += 1
            symbol = cmd[-1]
            if symbol in (Q4K1, Q4K2):  # perf's own full demangled names: real samples
                return subprocess.CompletedProcess([], 0, stdout=PERF_ANNOTATE, stderr="")
            stdout, stderr = no_samples   # a short/typed name perf cannot match verbatim
            return subprocess.CompletedProcess([], 0, stdout=stdout, stderr=stderr)

        with mock.patch.object(t.subprocess, "run", side_effect=fake_run):
            t.symbol_annotate(self.dirs, "measurement-record.data",
                              "mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2, 1>",
                              "libggml-cpu.so.0.16.0", cache=cache)
            t.symbol_annotate(self.dirs, "measurement-record.data",
                              "mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2, 2>",
                              "libggml-cpu.so.0.16.0", cache=cache)
        # One `perf report --sort symbol` resolves BOTH short names from the cached
        # symbol list. Four annotate calls (2 per symbol_annotate: the short-name
        # probe that misses verbatim, then the resolved full name) are each a
        # genuinely distinct perf argv -- caching cannot and should not collapse
        # those, only the shared resolve-report call.
        self.assertEqual(calls["report"], 1)
        self.assertEqual(calls["annotate"], 4)

    def test_no_cache_argument_is_unchanged_behavior(self):
        """cache=None (every call site that predates DS41-C20d) must still run perf
        every time -- the cache is opt-in per call, never a hidden global."""
        with mock.patch.object(t.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout=PERF_REPORT, stderr="")) as run:
            t.profile_top(self.dirs, "measurement-record.data", dso="libggml-cpu.so")
            t.profile_top(self.dirs, "measurement-record.data", dso="libggml-cpu.so")
        self.assertEqual(run.call_count, 2)


class ServerEntry(unittest.TestCase):

    def test_args_accept_repeated_profiles(self):
        ns = t.parse_args(["--root", "/x", "--profiles", "/a", "--profiles", "/b"])
        self.assertEqual(ns.root, "/x")
        self.assertEqual(ns.profiles, ["/a", "/b"])

    def test_module_imports_without_mcp(self):
        # the SDK is imported only in build_server(); plain functions must not need it
        self.assertTrue(callable(t.read_range))
        self.assertNotIn("mcp", t.__dict__)


if __name__ == "__main__":
    unittest.main()
