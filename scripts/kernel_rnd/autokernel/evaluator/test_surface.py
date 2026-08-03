#!/usr/bin/env python3
"""test_surface.py — the regression barrier for affected-surface derivation (§6.4)
and the two-stage backend-unchanged test (§3.2).

WHY THIS FILE EXISTS
--------------------
The affected-surface manifest decides freeze scope, lineage composition and sentinel
selection. The cheap exploit is not a faked score — it is a candidate that touches
shared ggml core and declares `backends: [llama_cpu]`, so its GPU cells are never
measured and the freeze ships an unmeasured HIP binary. Invariant 18 ("declared equals
traced") is only real if `traced ⊄ derived` actually FAILS and if a declaration
actually cannot reach the derivation. Both are asserted here.

The second half asserts §3.2's shape: naive whole-binary byte comparison must be
REFUSED (ROCm builds embed build IDs, timestamps and paths, so such a test never
fires), stage 2 must be run against a base rebuilt in the candidate's environment, and
a disagreement between the two stages must be a hard finding rather than a silent
preference for the cheaper stage. The ELF fixtures below are synthesised in-process so
the "differ only in `.comment` and `.note.gnu.build-id`" case is exact: whole-file
SHA-256 differs, the normalized §3.2 digest does not.

NO inference, NO benchmark, NO build, NO kernel tree is touched. The only processes are
none; the only writes are into a `tempfile.TemporaryDirectory` this suite creates and
removes.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_surface.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/evaluator/test_surface.py
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

# Import through the PACKAGE so `surface.schemas` is the same module object the
# validators use (README, "Import convention").
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import schemas as S  # noqa: E402
from autokernel.evaluator import api  # noqa: E402
from autokernel.evaluator import surface as SU  # noqa: E402

NOW = "2026-08-03T12:00:00+00:00"
BASE_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
CAND_COMMIT = "aa11bb22cc33dd44ee55ff6677889900aabbccdd"
SHA0 = "0" * 64
SHA1 = "1" * 64


# =============================================================================
# Fixtures — a miniature llama.cpp-shaped CMake/Ninja build
# =============================================================================

SOURCE_ROOT = "/repo/llama.cpp"
BUILD_DIR = "build-hip"

#: Exactly what `gcc -MD -MP` writes next to each object, relative to the build dir.
DEPFILE_GGML = r"""
CMakeFiles/ggml.dir/ggml.c.o: ../ggml/src/ggml.c ../ggml/include/ggml.h \
  ../ggml/include/ggml-alloc.h /usr/include/stdio.h

../ggml/include/ggml.h:
../ggml/include/ggml-alloc.h:
"""

DEPFILE_CPU = r"""
CMakeFiles/ggml.dir/ggml-cpu.c.o: ../ggml/src/ggml-cpu.c ../ggml/include/ggml.h \
  ../ggml/src/ggml-cpu-quants.c
"""

DEPFILE_HIP = r"""
CMakeFiles/ggml-hip.dir/ggml-cuda.cu.o: ../ggml/src/ggml-cuda/ggml-cuda.cu \
  ../ggml/include/ggml.h ../ggml/src/ggml-cuda/mmq.cuh
"""

LINK_BENCH = ("/usr/bin/c++ -O3 -march=native CMakeFiles/ggml.dir/ggml.c.o "
              "CMakeFiles/ggml.dir/ggml-cpu.c.o -o bin/llama-bench -lm")
LINK_SERVER = ("/usr/bin/c++ -O3 CMakeFiles/ggml.dir/ggml.c.o "
               "CMakeFiles/ggml-hip.dir/ggml-cuda.cu.o -o bin/llama-server -lamdhip64")


def make_index(label="candidate", *, depfiles=None, links=None, backends=None):
    dep_edges = []
    for name, text in (depfiles or {
        "ggml.d": DEPFILE_GGML, "cpu.d": DEPFILE_CPU, "hip.d": DEPFILE_HIP,
    }).items():
        dep_edges.extend(SU.parse_make_depfile(text, origin_ref=name))
    link_edges = []
    for name, text in (links or {
        "bench/link.txt": LINK_BENCH, "server/link.txt": LINK_SERVER,
    }).items():
        link_edges.append(SU.parse_cmake_link_txt(text, origin_ref=name))
    return SU.build_dependency_index(
        label=label, build_dir=BUILD_DIR, source_root=SOURCE_ROOT,
        dep_edges=dep_edges, link_edges=link_edges,
        backend_link_targets=backends or {
            "llama_cpu": ["bin/llama-bench"],
            "llama_gpu": ["bin/llama-server"],
        },
    )


def make_diff(paths, *, kinds=None, base=BASE_COMMIT, candidate=CAND_COMMIT):
    kinds = kinds or {}
    entries = tuple(SU.DiffEntry(path=p, change_kind=kinds.get(p, "modified")) for p in paths)
    return SU.SourceDiff(base_commit=base, candidate_commit=candidate, entries=entries,
                         origin_ref="git diff --name-status")


def make_registrations(provenance=SU.PROVENANCE_TOOL_EXTRACTED):
    return SU.SymbolRegistrationIndex(
        label="nm+clang-index",
        symbols_by_source={
            "ggml/src/ggml-cpu.c": ("ggml_compute_forward_mul_mat",),
            "ggml/src/ggml-cuda/ggml-cuda.cu": ("ggml_cuda_op_mul_mat",),
        },
        registrations_by_symbol={
            "ggml_compute_forward_mul_mat": (
                SU.OpRegistration("MUL_MAT", "llama_cpu", "ggml_cpu_extra_supported"),),
            "ggml_cuda_op_mul_mat": (
                SU.OpRegistration("MUL_MAT", "llama_gpu", "ggml_cuda_supports_op"),),
        },
        provenance=provenance,
    )


def trace_line(**kwargs):
    return json.dumps(kwargs, sort_keys=True)


def make_trace(candidate_id="akc-001", *, events=(), truncated=False, header=True):
    lines = []
    if header:
        lines.append(json.dumps({"schema": SU.DISPATCH_TRACE_SCHEMA,
                                 "candidate_id": candidate_id,
                                 "truncated": truncated}, sort_keys=True))
    lines.extend(trace_line(**e) for e in events)
    return "\n".join(lines) + "\n"


CPU_EVENT = {"op_name": "MUL_MAT", "backend": "llama_cpu",
             "kernel_symbol": "ggml_compute_forward_mul_mat",
             "link_target": "build-hip/bin/llama-bench",
             "dispatch_predicate": "ggml_cpu_extra_supported"}
GPU_EVENT = {"op_name": "MUL_MAT", "backend": "llama_gpu",
             "kernel_symbol": "ggml_cuda_op_mul_mat",
             "link_target": "build-hip/bin/llama-server",
             "dispatch_predicate": "ggml_cuda_supports_op"}


def toolchain(**overrides):
    base = dict(
        compiler_id="clang", compiler_version="17.0.0-rocm6.2",
        linker_id="ld.lld", linker_version="17.0.0",
        flags=("-O3", "-march=native"), defines=("GGML_USE_HIP=1",),
        environment=(("HIP_PATH", "/opt/rocm"),), sysroot=None,
    )
    base.update(overrides)
    return SU.ToolchainIdentity(**base)


# =============================================================================
# Synthetic ELF64 builder — so the §3.2 normalization can be tested exactly
# =============================================================================

def build_elf(section_data, *, dynsyms=None, dynsym_values=None) -> bytes:
    """Build a minimal ELF64 LSB file containing exactly `section_data`."""
    secs = [[name, blob, 1, 0] for name, blob in section_data.items()]
    if dynsyms is not None:
        strtab = b"\x00"
        offsets = {}
        for row in dynsyms:
            offsets[row[0]] = len(strtab)
            strtab += row[0].encode() + b"\x00"
        symbytes = b"\x00" * 24
        for i, (nm, bind, typ, defined) in enumerate(dynsyms):
            value = 0 if dynsym_values is None else dynsym_values[i]
            symbytes += struct.pack("<IBBHQQ", offsets[nm], (bind << 4) | typ, 0,
                                    1 if defined else 0, value, 0)
        secs.append([".dynsym", symbytes, 11, 24])
        secs.append([".dynstr", strtab, 3, 0])

    shstr = b"\x00"
    name_off = {}
    for sec in secs:
        name_off[sec[0]] = len(shstr)
        shstr += sec[0].encode() + b"\x00"
    name_off[".shstrtab"] = len(shstr)
    shstr += b".shstrtab\x00"
    secs.append([".shstrtab", shstr, 3, 0])

    body = bytearray()
    offset = 64
    offsets_out = []
    for sec in secs:
        pad = (-offset) % 8
        body += b"\x00" * pad
        offset += pad
        offsets_out.append(offset)
        body += sec[1]
        offset += len(sec[1])
    pad = (-offset) % 8
    body += b"\x00" * pad
    offset += pad
    shoff = offset

    shdrs = bytearray(struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    for i, sec in enumerate(secs):
        shdrs += struct.pack("<IIQQQQIIQQ", name_off[sec[0]], sec[2], 0, 0,
                             offsets_out[i], len(sec[1]), 0, 0, 1, sec[3])

    e_ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\x00" * 8
    ehdr = struct.pack("<16sHHIQQQIHHHHHH", e_ident, 2, 62, 1, 0, 0, shoff, 0,
                       64, 0, 0, 64, len(secs) + 1, len(secs))
    return bytes(ehdr) + bytes(body) + bytes(shdrs)


class ElfDir:
    """Small helper owning a TemporaryDirectory for ELF and depfile fixtures."""

    def __init__(self, testcase):
        self._tmp = tempfile.TemporaryDirectory()
        testcase.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write(self, name, data):
        path = self.root / name
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            path.write_bytes(data)
        return path


# =============================================================================
# Depfile parsing
# =============================================================================

class MakeDepfileTest(unittest.TestCase):

    def test_continuations_and_targets(self):
        edges = SU.parse_make_depfile(DEPFILE_GGML, origin_ref="ggml.d")
        objects = [e for e in edges if e.valid]
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].target, "CMakeFiles/ggml.dir/ggml.c.o")
        self.assertIn("../ggml/src/ggml.c", objects[0].prerequisites)
        self.assertIn("/usr/include/stdio.h", objects[0].prerequisites)

    def test_gcc_MP_phony_targets_are_marked_invalid_not_treated_as_objects(self):
        edges = SU.parse_make_depfile(DEPFILE_GGML, origin_ref="ggml.d")
        phony = [e for e in edges if not e.valid]
        self.assertEqual({e.target for e in phony},
                         {"../ggml/include/ggml.h", "../ggml/include/ggml-alloc.h"})
        self.assertTrue(all(e.invalidity_reason == "PHONY_TARGET" for e in phony))

    def test_escaped_spaces_and_dollars(self):
        edges = SU.parse_make_depfile("a.o: some\\ path/x.c $$HOME/y.h", origin_ref="d")
        self.assertEqual(edges[0].prerequisites, ("some path/x.c", "$HOME/y.h"))

    def test_escaped_colon_is_not_a_separator(self):
        edges = SU.parse_make_depfile("a.o: weird\\:name.c", origin_ref="d")
        self.assertEqual(edges[0].prerequisites, ("weird:name.c",))

    def test_comment_lines_are_skipped(self):
        edges = SU.parse_make_depfile("# CMake depend.make\na.o: b.c\n", origin_ref="d")
        self.assertEqual(len(edges), 1)

    def test_two_unescaped_colons_raises_rather_than_guessing(self):
        with self.assertRaises(SU.DepfileParseError):
            SU.parse_make_depfile("a.o: b.c: c.h", origin_ref="d")

    def test_line_without_a_colon_raises(self):
        with self.assertRaises(SU.DepfileParseError):
            SU.parse_make_depfile("this is not a rule", origin_ref="d")

    def test_escaped_trailing_backslash_does_not_continue(self):
        edges = SU.parse_make_depfile("a.o: dir\\\\\nb.o: c.c\n", origin_ref="d")
        self.assertEqual(len(edges), 2)

    def test_non_string_input_raises(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.parse_make_depfile(b"a.o: b.c", origin_ref="d")


class NinjaDepsTest(unittest.TestCase):

    VALID = ("build-hip/CMakeFiles/ggml.dir/ggml.c.o: #deps 2, deps mtime 1754 (VALID)\n"
             "    ../ggml/src/ggml.c\n"
             "    ../ggml/include/ggml.h\n")
    STALE = ("build-hip/CMakeFiles/ggml.dir/ggml-cpu.c.o: #deps 1, deps mtime 0 (STALE)\n"
             "    ../ggml/src/ggml-cpu.c\n")

    def test_valid_entry(self):
        edges = SU.parse_ninja_deps(self.VALID, origin_ref="ninja")
        self.assertEqual(len(edges), 1)
        self.assertTrue(edges[0].valid)
        self.assertEqual(len(edges[0].prerequisites), 2)

    def test_stale_entry_is_returned_but_invalid(self):
        edges = SU.parse_ninja_deps(self.STALE, origin_ref="ninja")
        self.assertFalse(edges[0].valid)
        self.assertEqual(edges[0].invalidity_reason, "NINJA_DEPS_STALE")

    def test_declared_count_mismatch_raises(self):
        bad = "x.o: #deps 3, deps mtime 1 (VALID)\n    a.c\n"
        with self.assertRaises(SU.DepfileParseError):
            SU.parse_ninja_deps(bad, origin_ref="ninja")

    def test_indented_line_without_header_raises(self):
        with self.assertRaises(SU.DepfileParseError):
            SU.parse_ninja_deps("    orphan.c\n", origin_ref="ninja")

    def test_unparseable_header_raises(self):
        with self.assertRaises(SU.DepfileParseError):
            SU.parse_ninja_deps("x.o: something else\n", origin_ref="ninja")

    def test_two_entries_separated_by_blank_line(self):
        edges = SU.parse_ninja_deps(self.VALID + "\n" + self.STALE, origin_ref="ninja")
        self.assertEqual(len(edges), 2)


class LinkManifestTest(unittest.TestCase):

    def test_cmake_link_txt_objects_and_output(self):
        edge = SU.parse_cmake_link_txt(LINK_SERVER, origin_ref="server/link.txt")
        self.assertEqual(edge.link_target, "bin/llama-server")
        self.assertEqual(edge.objects, ("CMakeFiles/ggml.dir/ggml.c.o",
                                        "CMakeFiles/ggml-hip.dir/ggml-cuda.cu.o"))

    def test_response_file_is_recorded_not_ignored(self):
        edge = SU.parse_cmake_link_txt("/usr/bin/c++ @objects.rsp -o bin/x",
                                       origin_ref="x/link.txt")
        self.assertEqual(edge.unresolved_inputs, ("objects.rsp",))
        self.assertEqual(edge.objects, ())

    def test_no_output_and_no_link_target_raises(self):
        with self.assertRaises(SU.DepfileParseError):
            SU.parse_cmake_link_txt("/usr/bin/c++ a.o", origin_ref="x")

    def test_link_target_disagreeing_with_output_raises(self):
        with self.assertRaises(SU.DepfileParseError):
            SU.parse_cmake_link_txt("/usr/bin/c++ a.o -o bin/x", origin_ref="x",
                                    link_target="bin/y")

    def test_unterminated_quote_raises(self):
        with self.assertRaises(SU.DepfileParseError):
            SU.parse_cmake_link_txt('/usr/bin/c++ "a.o -o bin/x', origin_ref="x")

    def test_empty_link_command_raises(self):
        with self.assertRaises(SU.DepfileParseError):
            SU.parse_cmake_link_txt("   ", origin_ref="x")

    def test_json_link_manifest(self):
        text = json.dumps({"links": [{"link_target": "bin/x", "objects": ["a.o", "b.o"]}]})
        edges = SU.parse_link_manifest_json(text, origin_ref="targets.json")
        self.assertEqual(edges[0].objects, ("a.o", "b.o"))

    def test_json_link_manifest_rejects_wrong_shape(self):
        for bad in ('{"links": {}}', '{"nope": []}', "[]", "not json",
                    '{"links": [{"objects": ["a.o"]}]}',
                    '{"links": [{"link_target": "x", "objects": "a.o"}]}'):
            with self.subTest(bad=bad):
                with self.assertRaises(SU.DepfileParseError):
                    SU.parse_link_manifest_json(bad, origin_ref="t")

    def test_load_from_disk_and_missing_file_raises(self):
        tmp = ElfDir(self)
        path = tmp.write("ggml.d", DEPFILE_CPU)
        edges = SU.load_make_depfile(path)
        self.assertTrue(any(e.valid for e in edges))
        with self.assertRaises(SU.SurfaceInputError):
            SU.load_make_depfile(tmp.root / "does-not-exist.d")

    def test_load_cmake_link_txt_from_disk(self):
        tmp = ElfDir(self)
        path = tmp.write("link.txt", LINK_BENCH)
        self.assertEqual(SU.load_cmake_link_txt(path).link_target, "bin/llama-bench")


# =============================================================================
# The dependency index — the trust boundary of stage 1
# =============================================================================

class BuildDependencyIndexTest(unittest.TestCase):

    def test_actor_declared_closure_is_refused(self):
        with self.assertRaises(SU.UntrustedProvenance):
            make_index()
            SU.build_dependency_index(
                label="actor", build_dir=BUILD_DIR, source_root=SOURCE_ROOT,
                dep_edges=SU.parse_make_depfile(DEPFILE_GGML, origin_ref="d"),
                link_edges=[SU.parse_cmake_link_txt(LINK_BENCH, origin_ref="l")],
                backend_link_targets={"llama_cpu": ["bin/llama-bench"]},
                provenance=SU.PROVENANCE_ACTOR_DECLARED)

    def test_directory_prefix_guess_is_refused_by_name(self):
        with self.assertRaises(SU.UntrustedProvenance):
            SU.build_dependency_index(
                label="guess", build_dir=BUILD_DIR, source_root=SOURCE_ROOT,
                dep_edges=SU.parse_make_depfile(DEPFILE_GGML, origin_ref="d"),
                link_edges=[SU.parse_cmake_link_txt(LINK_BENCH, origin_ref="l")],
                backend_link_targets={"llama_cpu": ["bin/llama-bench"]},
                provenance=SU.PROVENANCE_DIRECTORY_PREFIX_GUESS)

    def test_actor_declared_backend_map_is_refused(self):
        with self.assertRaises(SU.UntrustedProvenance):
            SU.build_dependency_index(
                label="x", build_dir=BUILD_DIR, source_root=SOURCE_ROOT,
                dep_edges=SU.parse_make_depfile(DEPFILE_GGML, origin_ref="d"),
                link_edges=[SU.parse_cmake_link_txt(LINK_BENCH, origin_ref="l")],
                backend_link_targets={"llama_cpu": ["bin/llama-bench"]},
                backend_map_provenance=SU.PROVENANCE_ACTOR_DECLARED)

    def test_resolution_is_repo_relative(self):
        index = make_index()
        self.assertIn("ggml/src/ggml.c", index.known_sources)
        self.assertEqual(index.objects_for_source("ggml/src/ggml.c"),
                         ("build-hip/CMakeFiles/ggml.dir/ggml.c.o",))

    def test_system_headers_are_external_and_out_of_the_closure(self):
        index = make_index()
        self.assertIn("/usr/include/stdio.h", index.external_prerequisites)
        self.assertNotIn("/usr/include/stdio.h", index.known_sources)

    def test_shared_header_reaches_both_link_targets(self):
        index = make_index()
        objs = index.objects_for_source("ggml/include/ggml.h")
        targets = set()
        for obj in objs:
            targets.update(index.link_targets_for_object(obj))
        self.assertEqual(targets, {"build-hip/bin/llama-bench", "build-hip/bin/llama-server"})

    def test_backend_attribution(self):
        index = make_index()
        self.assertEqual(index.backends_for_link_target("build-hip/bin/llama-server"),
                         ("llama_gpu",))
        self.assertEqual(index.link_targets_for_backend("llama_cpu"),
                         ("build-hip/bin/llama-bench",))

    def test_source_closure_for_backend(self):
        index = make_index()
        closure = index.source_closure_for_backend("llama_cpu")
        self.assertIn("ggml/src/ggml-cpu.c", closure)
        self.assertNotIn("ggml/src/ggml-cuda/ggml-cuda.cu", closure)

    def test_unknown_backend_name_raises(self):
        with self.assertRaises(SU.SurfaceInputError):
            make_index(backends={"llama_tpu": ["bin/llama-bench"]})

    def test_declared_link_target_the_build_never_emitted_raises(self):
        with self.assertRaises(SU.SurfaceInputError):
            make_index(backends={"llama_cpu": ["bin/does-not-exist"]})

    def test_backend_with_no_targets_raises(self):
        with self.assertRaises(SU.SurfaceInputError):
            make_index(backends={"llama_cpu": []})

    def test_empty_backend_map_raises(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.build_dependency_index(
                label="no-backends", build_dir=BUILD_DIR, source_root=SOURCE_ROOT,
                dep_edges=SU.parse_make_depfile(DEPFILE_GGML, origin_ref="d"),
                link_edges=[SU.parse_cmake_link_txt(LINK_BENCH, origin_ref="l")],
                backend_link_targets={})

    def test_coverage_pass_on_clean_index(self):
        self.assertEqual(make_index().coverage_check().outcome, S.PASS)

    def test_coverage_could_not_check_on_stale_entry(self):
        edges = list(SU.parse_make_depfile(DEPFILE_GGML, origin_ref="d"))
        edges += list(SU.parse_ninja_deps(NinjaDepsTest.STALE, origin_ref="ninja"))
        index = SU.build_dependency_index(
            label="stale", build_dir=BUILD_DIR, source_root=SOURCE_ROOT,
            dep_edges=edges,
            link_edges=[SU.parse_cmake_link_txt(LINK_BENCH, origin_ref="l")],
            backend_link_targets={"llama_cpu": ["bin/llama-bench"]})
        self.assertEqual(index.coverage_check().outcome, S.COULD_NOT_CHECK)
        self.assertTrue(index.stale_targets)

    def test_coverage_could_not_check_on_unexpanded_response_file(self):
        link = SU.parse_cmake_link_txt(
            "/usr/bin/c++ CMakeFiles/ggml.dir/ggml.c.o @more.rsp -o bin/llama-bench",
            origin_ref="l")
        index = SU.build_dependency_index(
            label="rsp", build_dir=BUILD_DIR, source_root=SOURCE_ROOT,
            dep_edges=SU.parse_make_depfile(DEPFILE_GGML, origin_ref="d"),
            link_edges=[link], backend_link_targets={"llama_cpu": ["bin/llama-bench"]})
        self.assertEqual(index.coverage_check().outcome, S.COULD_NOT_CHECK)

    def test_wrong_edge_types_raise(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.build_dependency_index(
                label="x", build_dir=BUILD_DIR, source_root=SOURCE_ROOT,
                dep_edges=["not an edge"],
                link_edges=[SU.parse_cmake_link_txt(LINK_BENCH, origin_ref="l")],
                backend_link_targets={"llama_cpu": ["bin/llama-bench"]})

    def test_to_dict_is_content_hashable(self):
        S.content_hash(make_index().to_dict())


# =============================================================================
# The diff
# =============================================================================

class DiffTest(unittest.TestCase):

    def test_name_status_parsing(self):
        text = ("M\tggml/src/ggml.c\n"
                "A\tggml/src/new.c\n"
                "D\tggml/src/old.c\n"
                "R100\tggml/src/a.c\tggml/src/b.c\n")
        diff = SU.parse_git_name_status(text, base_commit=BASE_COMMIT,
                                        candidate_commit=CAND_COMMIT, origin_ref="git")
        self.assertEqual(len(diff.entries), 4)
        self.assertIn("ggml/src/a.c", diff.touched_paths)
        self.assertIn("ggml/src/b.c", diff.touched_paths)

    def test_unknown_status_letter_raises(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.parse_git_name_status("X\tfoo.c\n", base_commit=BASE_COMMIT,
                                     candidate_commit=CAND_COMMIT, origin_ref="git")

    def test_malformed_line_raises(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.parse_git_name_status("M\ta\tb\n", base_commit=BASE_COMMIT,
                                     candidate_commit=CAND_COMMIT, origin_ref="git")
        with self.assertRaises(SU.SurfaceInputError):
            SU.parse_git_name_status("R100\tonly-one\n", base_commit=BASE_COMMIT,
                                     candidate_commit=CAND_COMMIT, origin_ref="git")

    def test_rename_without_old_path_raises(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.DiffEntry(path="b.c", change_kind="renamed")

    def test_actor_declared_diff_is_refused(self):
        with self.assertRaises(SU.UntrustedProvenance):
            SU.SourceDiff(base_commit=BASE_COMMIT, candidate_commit=CAND_COMMIT,
                          entries=(SU.DiffEntry(path="a.c", change_kind="modified"),),
                          origin_ref="actor", provenance=SU.PROVENANCE_ACTOR_DECLARED)

    def test_bad_change_kind_raises(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.DiffEntry(path="a.c", change_kind="fiddled")


# =============================================================================
# STAGE 1 — static derivation
# =============================================================================

class DerivationTest(unittest.TestCase):

    def test_cpu_only_change_derives_cpu_only(self):
        derived = SU.derive_affected_surface(
            candidate_id="akc-001", diff=make_diff(["ggml/src/ggml-cpu.c"]),
            indexes=[make_index()], registrations=make_registrations())
        self.assertEqual(derived.backends, ("llama_cpu",))
        self.assertFalse(derived.full_tree)
        self.assertEqual(derived.coverage.outcome, S.PASS)

    def test_shared_header_change_reaches_both_backends(self):
        derived = SU.derive_affected_surface(
            candidate_id="akc-001", diff=make_diff(["ggml/include/ggml.h"]),
            indexes=[make_index()], registrations=make_registrations())
        self.assertEqual(derived.backends, ("llama_cpu", "llama_gpu"))
        fanout = [o for o in derived.over_approximations
                  if o.reason == SU.OA_SHARED_HEADER_FANOUT]
        self.assertEqual(len(fanout), 1)
        self.assertEqual(fanout[0].kind, SU.OA_KIND_MECHANICAL)

    def test_fanout_comes_from_the_depfiles_not_from_a_path_prefix(self):
        # Same file, same directory prefix, two build graphs. Only the depfiles can
        # tell these apart; a `ggml/**` prefix rule would answer identically for both.
        wide = SU.derive_affected_surface(
            candidate_id="akc-001", diff=make_diff(["ggml/include/ggml.h"]),
            indexes=[make_index()], registrations=make_registrations())
        self.assertEqual(wide.backends, ("llama_cpu", "llama_gpu"))

        narrow_index = make_index(depfiles={
            "ggml.d": "CMakeFiles/ggml.dir/ggml.c.o: ../ggml/src/ggml.c\n",
            "cpu.d": ("CMakeFiles/ggml.dir/ggml-cpu.c.o: ../ggml/src/ggml-cpu.c "
                      "../ggml/include/ggml.h\n"),
            "hip.d": ("CMakeFiles/ggml-hip.dir/ggml-cuda.cu.o: "
                      "../ggml/src/ggml-cuda/ggml-cuda.cu\n"),
        })
        narrow = SU.derive_affected_surface(
            candidate_id="akc-001", diff=make_diff(["ggml/include/ggml.h"]),
            indexes=[narrow_index], registrations=make_registrations())
        self.assertEqual(narrow.backends, ("llama_cpu",))
        self.assertEqual(narrow.coverage.outcome, S.PASS)

    def test_unmapped_touched_file_widens_fail_closed(self):
        derived = SU.derive_affected_surface(
            candidate_id="akc-001", diff=make_diff(["docs/README.md"]),
            indexes=[make_index()], registrations=make_registrations())
        self.assertTrue(derived.full_tree)
        self.assertEqual(derived.backends, ("llama_cpu", "llama_gpu"))
        self.assertEqual(derived.coverage.outcome, S.COULD_NOT_CHECK)
        reasons = {o.reason for o in derived.fail_closed_widenings}
        self.assertIn(SU.OA_UNMAPPED_TOUCHED_FILE, reasons)

    def test_widening_is_never_silent(self):
        derived = SU.derive_affected_surface(
            candidate_id="akc-001", diff=make_diff(["not/in/build.c"]),
            indexes=[make_index()])
        self.assertTrue(derived.fail_closed_widenings)
        for widening in derived.fail_closed_widenings:
            self.assertTrue(widening.trigger)
            self.assertTrue(widening.widened_to)

    def test_core_header_change_class_forces_full_tree(self):
        derived = SU.derive_affected_surface(
            candidate_id="akc-001", diff=make_diff(["ggml/src/ggml-cpu.c"]),
            indexes=[make_index()], registrations=make_registrations(),
            change_class="core_header")
        self.assertTrue(derived.full_tree)
        self.assertEqual(derived.backends, ("llama_cpu", "llama_gpu"))
        self.assertIn(SU.OA_CORE_HEADER_CHANGE_CLASS,
                      {o.reason for o in derived.over_approximations})

    def test_stale_dependency_information_widens(self):
        edges = list(SU.parse_make_depfile(DEPFILE_GGML, origin_ref="d"))
        edges += list(SU.parse_make_depfile(DEPFILE_CPU, origin_ref="d2"))
        edges += list(SU.parse_make_depfile(DEPFILE_HIP, origin_ref="d3"))
        edges += list(SU.parse_ninja_deps(NinjaDepsTest.STALE, origin_ref="ninja"))
        index = SU.build_dependency_index(
            label="stale", build_dir=BUILD_DIR, source_root=SOURCE_ROOT, dep_edges=edges,
            link_edges=[SU.parse_cmake_link_txt(LINK_BENCH, origin_ref="l"),
                        SU.parse_cmake_link_txt(LINK_SERVER, origin_ref="l2")],
            backend_link_targets={"llama_cpu": ["bin/llama-bench"],
                                  "llama_gpu": ["bin/llama-server"]})
        derived = SU.derive_affected_surface(
            candidate_id="akc-001", diff=make_diff(["ggml/src/ggml-cpu.c"]),
            indexes=[index])
        self.assertTrue(derived.full_tree)
        self.assertIn(SU.OA_STALE_DEPENDENCY_ENTRY,
                      {o.reason for o in derived.fail_closed_widenings})

    def test_deleted_file_resolves_through_the_base_index(self):
        # The candidate's build no longer knows the deleted file; the base's does.
        cpu_gone = "CMakeFiles/ggml.dir/ggml-cpu.c.o: ../ggml/src/ggml-cpu.c\n"
        candidate_index = make_index("candidate",
                                     depfiles={"ggml.d": DEPFILE_GGML, "cpu.d": cpu_gone,
                                               "hip.d": DEPFILE_HIP})
        base_index = make_index("production_base")
        diff = make_diff(["ggml/src/ggml-cpu-quants.c"],
                         kinds={"ggml/src/ggml-cpu-quants.c": "deleted"})
        only_candidate = SU.derive_affected_surface(
            candidate_id="akc-001", diff=diff, indexes=[candidate_index],
            registrations=make_registrations())
        self.assertTrue(only_candidate.full_tree)
        both = SU.derive_affected_surface(
            candidate_id="akc-001", diff=diff, indexes=[candidate_index, base_index],
            registrations=make_registrations())
        self.assertFalse(both.full_tree)
        self.assertEqual(both.coverage.outcome, S.PASS)

    def test_no_index_raises_rather_than_deriving_nothing(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.derive_affected_surface(candidate_id="akc-001",
                                       diff=make_diff(["ggml/src/ggml.c"]), indexes=[])

    def test_a_declaration_cannot_be_passed_as_an_index(self):
        declaration = SU.ActorDeclaration(candidate_id="akc-001", backends=("llama_cpu",))
        with self.assertRaises(SU.UntrustedProvenance):
            SU.derive_affected_surface(candidate_id="akc-001",
                                       diff=make_diff(["ggml/src/ggml.c"]),
                                       indexes=[declaration])

    def test_actor_provenance_symbol_index_refused(self):
        with self.assertRaises(SU.UntrustedProvenance):
            make_registrations(provenance=SU.PROVENANCE_ACTOR_DECLARED)

    def test_registrations_produce_ops_and_predicates(self):
        derived = SU.derive_affected_surface(
            candidate_id="akc-001", diff=make_diff(["ggml/src/ggml-cuda/ggml-cuda.cu"]),
            indexes=[make_index()], registrations=make_registrations())
        self.assertEqual(derived.op_names, ("MUL_MAT",))
        self.assertEqual(derived.dispatch_predicates, ("ggml_cuda_supports_op",))
        self.assertIn("ggml_cuda_op_mul_mat", derived.symbols)
        self.assertIn(SU.AXIS_OP_NAMES, derived.axes_derived)

    def test_missing_symbol_index_records_the_gap_and_narrows_the_axes(self):
        derived = SU.derive_affected_surface(
            candidate_id="akc-001", diff=make_diff(["ggml/src/ggml-cpu.c"]),
            indexes=[make_index()])
        self.assertNotIn(SU.AXIS_OP_NAMES, derived.axes_derived)
        self.assertIn(SU.OA_NO_SYMBOL_INDEX, {o.reason for o in derived.over_approximations})
        self.assertEqual(derived.coverage.outcome, S.COULD_NOT_CHECK)

    def test_op_registration_backend_widens_the_backend_set(self):
        # A symbol in a CPU file that registers a GPU op still reaches llama_gpu.
        registrations = SU.SymbolRegistrationIndex(
            label="x",
            symbols_by_source={"ggml/src/ggml-cpu.c": ("sneaky_register",)},
            registrations_by_symbol={
                "sneaky_register": (SU.OpRegistration("MUL_MAT", "llama_gpu"),)},
        )
        derived = SU.derive_affected_surface(
            candidate_id="akc-001", diff=make_diff(["ggml/src/ggml-cpu.c"]),
            indexes=[make_index()], registrations=registrations)
        self.assertIn("llama_gpu", derived.backends)

    def test_bad_change_class_raises(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.derive_affected_surface(candidate_id="akc-001",
                                       diff=make_diff(["ggml/src/ggml.c"]),
                                       indexes=[make_index()], change_class="whatever")

    def test_sha256_is_stable_and_content_sensitive(self):
        a = SU.derive_affected_surface(candidate_id="akc-001",
                                       diff=make_diff(["ggml/src/ggml-cpu.c"]),
                                       indexes=[make_index()])
        b = SU.derive_affected_surface(candidate_id="akc-001",
                                       diff=make_diff(["ggml/src/ggml-cpu.c"]),
                                       indexes=[make_index()])
        c = SU.derive_affected_surface(candidate_id="akc-001",
                                       diff=make_diff(["ggml/src/ggml-cuda/ggml-cuda.cu"]),
                                       indexes=[make_index()])
        self.assertEqual(a.sha256(), b.sha256())
        self.assertNotEqual(a.sha256(), c.sha256())


# =============================================================================
# STAGE 2 — the dispatch trace
# =============================================================================

class DispatchTraceTest(unittest.TestCase):

    def test_header_and_events(self):
        traced = SU.parse_dispatch_trace(make_trace(events=[CPU_EVENT, GPU_EVENT]),
                                         trace_ref="t0.jsonl")
        self.assertEqual(traced.candidate_id, "akc-001")
        self.assertEqual(traced.backends, ("llama_cpu", "llama_gpu"))
        self.assertEqual(traced.op_names, ("MUL_MAT",))
        self.assertEqual(traced.completeness.outcome, S.PASS)
        self.assertEqual(traced.no_fallback.outcome, S.PASS)

    def test_unknown_backend_is_never_defaulted(self):
        bad = make_trace(events=[dict(CPU_EVENT, backend="llama_tpu")])
        with self.assertRaises(SU.TraceParseError):
            SU.parse_dispatch_trace(bad, trace_ref="t")

    def test_unknown_field_is_refused(self):
        bad = make_trace(events=[dict(CPU_EVENT, sneaky="x")])
        with self.assertRaises(SU.TraceParseError):
            SU.parse_dispatch_trace(bad, trace_ref="t")

    def test_missing_required_field_is_refused(self):
        event = dict(CPU_EVENT)
        del event["kernel_symbol"]
        with self.assertRaises(SU.TraceParseError):
            SU.parse_dispatch_trace(make_trace(events=[event]), trace_ref="t")

    def test_malformed_json_line_raises_rather_than_skipping(self):
        text = make_trace(events=[CPU_EVENT]) + "{not json}\n"
        with self.assertRaises(SU.TraceParseError):
            SU.parse_dispatch_trace(text, trace_ref="t")

    def test_non_object_line_raises(self):
        with self.assertRaises(SU.TraceParseError):
            SU.parse_dispatch_trace('{"schema": "%s", "candidate_id": "akc-001"}\n[1,2]\n'
                                    % SU.DISPATCH_TRACE_SCHEMA, trace_ref="t")

    def test_fallback_without_a_reason_is_refused(self):
        with self.assertRaises(SU.TraceParseError):
            SU.parse_dispatch_trace(make_trace(events=[dict(CPU_EVENT, fallback=True)]),
                                    trace_ref="t")

    def test_fallback_fails_the_no_fallback_proof(self):
        event = dict(CPU_EVENT, fallback=True, fallback_reason="iqk path unsupported")
        traced = SU.parse_dispatch_trace(make_trace(events=[event]), trace_ref="t")
        self.assertEqual(traced.no_fallback.outcome, S.FAIL)
        self.assertEqual(len(traced.fallback_events), 1)

    def test_empty_trace_is_could_not_check_not_pass(self):
        traced = SU.parse_dispatch_trace(make_trace(events=[]), trace_ref="t")
        self.assertEqual(traced.completeness.outcome, S.COULD_NOT_CHECK)
        self.assertEqual(traced.no_fallback.outcome, S.COULD_NOT_CHECK)

    def test_truncated_trace_is_could_not_check(self):
        traced = SU.parse_dispatch_trace(make_trace(events=[CPU_EVENT], truncated=True),
                                         trace_ref="t")
        self.assertEqual(traced.completeness.outcome, S.COULD_NOT_CHECK)
        self.assertTrue(traced.truncated)

    def test_unknown_schema_raises(self):
        text = json.dumps({"schema": "epyc.autokernel.dispatch_trace.v99",
                           "candidate_id": "akc-001"}) + "\n"
        with self.assertRaises(SU.TraceParseError):
            SU.parse_dispatch_trace(text, trace_ref="t")

    def test_second_header_raises(self):
        text = make_trace(events=[]) + make_trace(events=[])
        with self.assertRaises(SU.TraceParseError):
            SU.parse_dispatch_trace(text, trace_ref="t")

    def test_unattributed_trace_raises(self):
        text = trace_line(**CPU_EVENT) + "\n"
        with self.assertRaises(SU.TraceParseError):
            SU.parse_dispatch_trace(text, trace_ref="t")

    def test_candidate_id_disagreement_raises(self):
        with self.assertRaises(SU.TraceParseError):
            SU.parse_dispatch_trace(make_trace("akc-001", events=[CPU_EVENT]),
                                    trace_ref="t", candidate_id="akc-999")

    def test_non_string_trace_raises(self):
        with self.assertRaises(SU.TraceParseError):
            SU.parse_dispatch_trace(b"{}", trace_ref="t")

    def test_traced_surface_sha256_is_content_hashable(self):
        traced = SU.parse_dispatch_trace(make_trace(events=[CPU_EVENT]), trace_ref="t")
        self.assertEqual(len(traced.sha256()), 64)


# =============================================================================
# STAGE 3 — reconciliation. Invariant 18.
# =============================================================================

def derived_for(paths, *, candidate_id="akc-001", registrations=None, **kwargs):
    return SU.derive_affected_surface(candidate_id=candidate_id, diff=make_diff(paths),
                                      indexes=[make_index()], registrations=registrations,
                                      **kwargs)


class ReconciliationTest(unittest.TestCase):

    def test_containment_passes(self):
        derived = derived_for(["ggml/include/ggml.h"], registrations=make_registrations())
        traced = SU.parse_dispatch_trace(make_trace(events=[CPU_EVENT, GPU_EVENT]),
                                         trace_ref="t")
        rec = SU.reconcile_surface(derived, traced)
        self.assertEqual(rec.check.outcome, S.PASS)
        self.assertFalse(rec.hard_failure)

    def test_traced_backend_outside_derived_is_a_hard_failure(self):
        # THE exploit: a change the actor would like scoped to the CPU, whose kernel
        # actually executes on the GPU.
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        self.assertEqual(derived.backends, ("llama_cpu",))
        traced = SU.parse_dispatch_trace(make_trace(events=[CPU_EVENT, GPU_EVENT]),
                                         trace_ref="t")
        rec = SU.reconcile_surface(derived, traced)
        self.assertEqual(rec.check.outcome, S.FAIL)
        self.assertTrue(rec.hard_failure)
        self.assertIn((SU.AXIS_BACKENDS, "llama_gpu"), rec.escaped)

    def test_traced_link_target_outside_derived_is_a_hard_failure(self):
        derived = derived_for(["ggml/src/ggml-cuda/ggml-cuda.cu"],
                              registrations=make_registrations())
        event = dict(GPU_EVENT, link_target="build-hip/bin/llama-bench",
                     backend="llama_gpu")
        traced = SU.parse_dispatch_trace(make_trace(events=[event]), trace_ref="t")
        rec = SU.reconcile_surface(derived, traced)
        self.assertEqual(rec.check.outcome, S.FAIL)
        self.assertIn((SU.AXIS_LINK_TARGETS, "build-hip/bin/llama-bench"), rec.escaped)

    def test_no_trace_is_could_not_check_never_pass(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        rec = SU.reconcile_surface(derived, None)
        self.assertEqual(rec.check.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(rec.hard_failure)

    def test_empty_trace_is_could_not_check(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        traced = SU.parse_dispatch_trace(make_trace(events=[]), trace_ref="t")
        self.assertEqual(SU.reconcile_surface(derived, traced).check.outcome,
                         S.COULD_NOT_CHECK)

    def test_truncated_trace_cannot_confirm_containment(self):
        derived = derived_for(["ggml/include/ggml.h"], registrations=make_registrations())
        traced = SU.parse_dispatch_trace(
            make_trace(events=[CPU_EVENT, GPU_EVENT], truncated=True), trace_ref="t")
        self.assertEqual(SU.reconcile_surface(derived, traced).check.outcome,
                         S.COULD_NOT_CHECK)

    def test_op_escape_without_a_symbol_index_is_a_derivation_gap_not_a_finding(self):
        derived = derived_for(["ggml/include/ggml.h"])  # no registrations
        traced = SU.parse_dispatch_trace(make_trace(events=[CPU_EVENT, GPU_EVENT]),
                                         trace_ref="t")
        rec = SU.reconcile_surface(derived, traced)
        self.assertEqual(rec.check.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(rec.hard_failure)
        ops = [a for a in rec.axes if a.axis == SU.AXIS_OP_NAMES][0]
        self.assertEqual(ops.check.outcome, S.COULD_NOT_CHECK)

    def test_op_escape_with_a_symbol_index_is_a_hard_failure(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        event = dict(CPU_EVENT, op_name="FLASH_ATTN_EXT",
                     kernel_symbol="ggml_compute_forward_flash_attn")
        traced = SU.parse_dispatch_trace(make_trace(events=[event]), trace_ref="t")
        rec = SU.reconcile_surface(derived, traced)
        self.assertEqual(rec.check.outcome, S.FAIL)
        self.assertIn((SU.AXIS_OP_NAMES, "FLASH_ATTN_EXT"), rec.escaped)

    def test_reconciling_across_candidates_raises(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"])
        traced = SU.parse_dispatch_trace(make_trace("akc-999", events=[CPU_EVENT]),
                                         trace_ref="t")
        with self.assertRaises(SU.SurfaceInputError):
            SU.reconcile_surface(derived, traced)

    def test_wrong_types_raise(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.reconcile_surface("nope", None)
        with self.assertRaises(SU.SurfaceInputError):
            SU.reconcile_surface(derived_for(["ggml/src/ggml.c"]), "nope")

    def test_gate_results_are_integrity_class(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        traced = SU.parse_dispatch_trace(make_trace(events=[CPU_EVENT]), trace_ref="t")
        gates = SU.reconcile_surface(derived, traced).gate_results()
        self.assertEqual({g.gate_class for g in gates}, {api.GATE_INTEGRITY})
        self.assertIn(api.GATE_INTEGRITY, api.LEXICOGRAPHICALLY_PRIOR_GATE_CLASSES)
        self.assertEqual({g.gate_id for g in gates},
                         {"surface.derived_coverage", "surface.reconciliation",
                          "surface.trace_completeness", "surface.no_fallback"})

    def test_gates_without_a_trace_are_could_not_check(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        gates = {g.gate_id: g for g in SU.reconcile_surface(derived, None).gate_results()}
        self.assertEqual(gates["surface.trace_completeness"].check.outcome, S.COULD_NOT_CHECK)
        self.assertEqual(gates["surface.no_fallback"].check.outcome, S.COULD_NOT_CHECK)

    def test_to_dict_is_content_hashable(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        traced = SU.parse_dispatch_trace(make_trace(events=[CPU_EVENT]), trace_ref="t")
        S.content_hash(SU.reconcile_surface(derived, traced).to_dict())


class CandidateRecordBlockTest(unittest.TestCase):

    def _candidate(self, block):
        return {
            "schema": S.SCHEMA_CANDIDATE,
            "candidate_id": "akc-001", "campaign_id": "ak-001", "proposal_id": "akp-001",
            "parent_candidate_id": None,
            "worktree": {"path": "/mnt/raid0/llm/llama.cpp-ak-001",
                         "branch": "ak/ak-001/mul-mat", "source_commit": CAND_COMMIT,
                         "clean": True},
            "source_snapshot": {"snapshot_sha256": SHA0, "patch_bundle_sha256": SHA1},
            "ancestry": {"production_base_commit": BASE_COMMIT,
                         "is_descendant_of_production_base": True,
                         "proof": "git merge-base --is-ancestor"},
            "build": {"toolchain": "rocm-6.2", "compiler": "clang-17",
                      "command": "cmake --build build-hip", "build_dir": "build-hip",
                      "log_path": "logs/build.log", "log_sha256": SHA0},
            "artifacts": {"binary_sha256": SHA0, "linkage_sha256": SHA1,
                          "library_sha256s": {"libggml.so": SHA0}},
            "dispatch": {"feature_flags": ["GGML_IQK"], "dispatch_predicate": "iqk_supported"},
            "affected_surface": block,
            "determinism": {"class": "bitwise_stable", "same_seed_repeat_runs": 3},
            "evaluator": {"id": "P-AK-SEARCH-1/v1", "bundle_sha256": SHA0},
            "receipts": {"host_receipt": "host-1", "resource_claim_receipt": "claim-1"},
            "storage": {"footprint_gb": 1.5, "durability_class": "carried_in_git"},
            "evaluation_event_ids": ["ake-001"], "derived_verdicts": {},
            "controller": {"provider": "anthropic", "model_id": "m", "effort": "high",
                           "prompt_bundle_sha256": SHA0},
            "champion_status": "none", "status": "evaluating",
            "supersession_reason": None, "created_at": NOW,
        }

    def test_block_validates_inside_a_candidate_record(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        traced = SU.parse_dispatch_trace(make_trace(events=[CPU_EVENT]), trace_ref="t")
        block = SU.candidate_affected_surface_block(SU.reconcile_surface(derived, traced))
        self.assertTrue(block["reconciled"])
        self.assertEqual(S.validate_candidate(self._candidate(block)), [])

    def test_untraced_block_is_null_and_not_reconciled(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        block = SU.candidate_affected_surface_block(SU.reconcile_surface(derived, None))
        self.assertIsNone(block["traced_sha256"])
        self.assertFalse(block["reconciled"])
        self.assertEqual(S.validate_candidate(self._candidate(block)), [])

    def test_could_not_check_is_not_reconciled(self):
        derived = derived_for(["ggml/include/ggml.h"])  # no symbol index
        traced = SU.parse_dispatch_trace(make_trace(events=[CPU_EVENT, GPU_EVENT]),
                                         trace_ref="t")
        rec = SU.reconcile_surface(derived, traced)
        self.assertEqual(rec.check.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(SU.candidate_affected_surface_block(rec)["reconciled"])

    def test_wrong_type_raises(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.candidate_affected_surface_block("nope")


# =============================================================================
# The actor's declaration — scored, never consumed
# =============================================================================

class DeclarationTest(unittest.TestCase):

    def test_declaration_cannot_relabel_its_own_provenance(self):
        with self.assertRaises(SU.UntrustedProvenance):
            SU.ActorDeclaration(candidate_id="akc-001", backends=("llama_cpu",),
                                provenance=SU.PROVENANCE_BUILD_SYSTEM)

    def test_reconcile_surface_has_no_declaration_parameter(self):
        # Structural, not conventional: there is no argument through which a
        # declaration could become a scope input (invariant 18).
        names = SU.reconcile_surface.__code__.co_varnames[
            :SU.reconcile_surface.__code__.co_argcount]
        self.assertEqual(names, ("derived", "traced"))

    def test_derive_affected_surface_takes_no_declaration(self):
        params = SU.derive_affected_surface.__code__.co_varnames[
            :SU.derive_affected_surface.__code__.co_argcount
            + SU.derive_affected_surface.__code__.co_kwonlyargcount]
        for name in params:
            self.assertNotIn("declar", name)

    def test_under_declaration_is_detected(self):
        derived = derived_for(["ggml/include/ggml.h"], registrations=make_registrations())
        declaration = SU.ActorDeclaration(candidate_id="akc-001", backends=("llama_cpu",))
        score = SU.score_actor_declaration(declaration, derived)
        self.assertTrue(score.under_declared_any)
        self.assertIn((SU.AXIS_BACKENDS, "llama_gpu"), score.under_declared)

    def test_over_declaration_is_detected(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        declaration = SU.ActorDeclaration(
            candidate_id="akc-001", backends=("llama_cpu", "llama_gpu"),
            link_targets=derived.link_targets, op_names=derived.op_names)
        score = SU.score_actor_declaration(declaration, derived)
        self.assertIn((SU.AXIS_BACKENDS, "llama_gpu"), score.over_declared)
        self.assertFalse(score.under_declared_any)
        stats = dict(score.per_axis)[SU.AXIS_BACKENDS]
        self.assertEqual(stats["recall"], 1.0)
        self.assertEqual(stats["precision"], 0.5)

    def test_payload_says_it_is_not_a_scope_input(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        payload = SU.score_actor_declaration(
            SU.ActorDeclaration(candidate_id="akc-001", backends=("llama_cpu",)),
            derived).to_dict()
        self.assertIs(payload["is_scope_input"], False)
        self.assertEqual(payload["consumer"], "critic")
        S.content_hash(payload)

    def test_scoring_does_not_change_the_derived_surface(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        before = derived.sha256()
        SU.score_actor_declaration(
            SU.ActorDeclaration(candidate_id="akc-001", backends=("llama_gpu",)), derived)
        self.assertEqual(derived.sha256(), before)

    def test_candidate_mismatch_raises(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"])
        with self.assertRaises(SU.SurfaceInputError):
            SU.score_actor_declaration(
                SU.ActorDeclaration(candidate_id="akc-999", backends=("llama_cpu",)), derived)

    def test_unknown_declared_backend_raises(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.ActorDeclaration(candidate_id="akc-001", backends=("llama_tpu",))


# =============================================================================
# §3.2 — normalized binary digests
# =============================================================================

TEXT = b"\x55\x48\x89\xe5" * 64
RODATA = b"hello world\x00" * 8
DATA_REL_RO = b"\x08\x00\x00\x00\x00\x00\x00\x00" * 4
DYNSYMS = [("ggml_init", 1, 2, True), ("malloc", 1, 2, False), ("ggml_free", 1, 2, True)]


def elf_bytes(*, text=TEXT, rodata=RODATA, comment=b"clang 17", build_id=b"\x01" * 20,
              dynsyms=DYNSYMS, dynsym_values=None):
    sections = {
        ".text": text,
        ".rodata": rodata,
        ".data.rel.ro": DATA_REL_RO,
        ".comment": comment,
        ".note.gnu.build-id": build_id,
        ".debug_info": b"\xde\xad\xbe\xef",
    }
    return build_elf(sections, dynsyms=dynsyms, dynsym_values=dynsym_values)


class NormalizedDigestTest(unittest.TestCase):

    def test_naive_byte_identity_is_refused_by_name(self):
        with self.assertRaises(SU.NaiveByteIdentityRefused):
            SU.compare_binaries_byte_identical("a", "b")

    def test_whole_file_digest_cannot_be_stored(self):
        with self.assertRaises(SU.NaiveByteIdentityRefused):
            SU.NormalizedBinaryDigest(
                ref="x",
                section_digests={".text": SHA0, ".rodata": SHA0, ".data.rel.ro": SHA0,
                                 "binary_sha256": SHA1},
                dynsym_digest=SHA0)

    def test_excluded_section_cannot_be_stored(self):
        for excluded in (".comment", ".note.gnu.build-id", ".debug_info"):
            with self.subTest(section=excluded):
                with self.assertRaises(SU.NormalizationViolation):
                    SU.NormalizedBinaryDigest(
                        ref="x",
                        section_digests={".text": SHA0, ".rodata": SHA0,
                                         ".data.rel.ro": SHA0, excluded: SHA1},
                        dynsym_digest=SHA0)

    def test_missing_compared_section_is_refused(self):
        with self.assertRaises(SU.NormalizationViolation):
            SU.NormalizedBinaryDigest(ref="x", section_digests={".text": SHA0},
                                      dynsym_digest=SHA0)

    def test_absent_section_must_be_spelled_not_omitted(self):
        digest = SU.normalized_binary_digest_from_sections(
            ref="x", section_digests={".text": SHA0, ".rodata": SU.SECTION_ABSENT,
                                      ".data.rel.ro": SHA0},
            dynsym_digest=SHA0)
        self.assertEqual(digest.absent_sections, (".rodata",))

    def test_residual_risks_are_declared_not_hidden(self):
        digest = SU.normalized_binary_digest_from_sections(
            ref="x", section_digests={s: SHA0 for s in SU.COMPARED_SECTIONS},
            dynsym_digest=SHA0)
        self.assertTrue(digest.residual_risks)
        self.assertIn("__FILE__", digest.residual_risks[0])


class ElfReaderTest(unittest.TestCase):

    def setUp(self):
        self.tmp = ElfDir(self)

    def test_build_id_and_comment_do_not_change_the_normalized_digest(self):
        a = self.tmp.write("a.so", elf_bytes(comment=b"clang 17.0.0 (build 1)",
                                             build_id=b"\x01" * 20))
        b = self.tmp.write("b.so", elf_bytes(comment=b"clang 17.0.0 (build 2)",
                                             build_id=b"\x02" * 20))
        self.assertNotEqual(hashlib.sha256(a.read_bytes()).hexdigest(),
                            hashlib.sha256(b.read_bytes()).hexdigest(),
                            "the fixture must differ, or the test proves nothing")
        da = SU.read_normalized_binary_digest(a)
        db = SU.read_normalized_binary_digest(b)
        self.assertEqual(da.differences(db), ())
        self.assertEqual(da.section_digests, db.section_digests)

    def test_text_difference_is_detected(self):
        a = self.tmp.write("a.so", elf_bytes())
        b = self.tmp.write("b.so", elf_bytes(text=TEXT[:-4] + b"\x90\x90\x90\x90"))
        diffs = SU.read_normalized_binary_digest(a).differences(
            SU.read_normalized_binary_digest(b))
        self.assertEqual(len(diffs), 1)
        self.assertTrue(diffs[0].startswith(".text:"))

    def test_rodata_difference_is_detected(self):
        a = self.tmp.write("a.so", elf_bytes())
        b = self.tmp.write("b.so", elf_bytes(rodata=b"different\x00"))
        self.assertTrue(any(d.startswith(".rodata:") for d in
                            SU.read_normalized_binary_digest(a).differences(
                                SU.read_normalized_binary_digest(b))))

    def test_dynsym_is_compared_by_symbol_set_not_by_address(self):
        a = self.tmp.write("a.so", elf_bytes(dynsym_values=[0x1000, 0x2000, 0x3000]))
        b = self.tmp.write("b.so", elf_bytes(dynsym_values=[0x9000, 0xA000, 0xB000]))
        self.assertEqual(SU.read_normalized_binary_digest(a).differences(
            SU.read_normalized_binary_digest(b)), ())

    def test_removed_exported_symbol_is_detected(self):
        a = self.tmp.write("a.so", elf_bytes())
        b = self.tmp.write("b.so", elf_bytes(dynsyms=DYNSYMS[:-1]))
        diffs = SU.read_normalized_binary_digest(a).differences(
            SU.read_normalized_binary_digest(b))
        self.assertTrue(any("dynamic symbol table" in d for d in diffs))

    def test_static_binary_has_an_absent_dynsym(self):
        path = self.tmp.write("static.so", elf_bytes(dynsyms=None))
        digest = SU.read_normalized_binary_digest(path)
        self.assertEqual(digest.dynsym_digest, SU.SECTION_ABSENT)
        self.assertIn(".dynsym", digest.absent_sections)

    def test_not_an_elf_raises(self):
        path = self.tmp.write("junk.bin", b"MZ\x00\x00" + b"\x00" * 200)
        with self.assertRaises(SU.ElfFormatError):
            SU.read_normalized_binary_digest(path)

    def test_elf32_raises(self):
        data = bytearray(elf_bytes())
        data[4] = 1  # ELFCLASS32
        path = self.tmp.write("elf32.so", bytes(data))
        with self.assertRaises(SU.ElfFormatError):
            SU.read_normalized_binary_digest(path)

    def test_big_endian_raises(self):
        data = bytearray(elf_bytes())
        data[5] = 2  # ELFDATA2MSB
        path = self.tmp.write("be.so", bytes(data))
        with self.assertRaises(SU.ElfFormatError):
            SU.read_normalized_binary_digest(path)

    def test_binary_without_text_raises(self):
        path = self.tmp.write("no-text.so", build_elf({".rodata": RODATA}))
        with self.assertRaises(SU.ElfFormatError):
            SU.read_normalized_binary_digest(path)

    def test_unreadable_path_raises(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.read_normalized_binary_digest(self.tmp.root / "nope.so")

    def test_absent_data_rel_ro_is_spelled_absent(self):
        path = self.tmp.write("min.so", build_elf({".text": TEXT, ".rodata": RODATA}))
        digest = SU.read_normalized_binary_digest(path)
        self.assertEqual(digest.section_digests[".data.rel.ro"], SU.SECTION_ABSENT)

    def test_real_system_binary_parses(self):
        candidate = Path(sys.executable)
        if not candidate.exists():
            self.skipTest("no interpreter path")
        with open(candidate, "rb") as handle:
            if handle.read(4) != b"\x7fELF":
                self.skipTest(f"{candidate} is not an ELF file on this host")
        digest = SU.read_normalized_binary_digest(candidate)
        self.assertNotEqual(digest.section_digests[".text"], SU.SECTION_ABSENT)


# =============================================================================
# §3.2 — the two stages
# =============================================================================

def digest_from(text=TEXT, ref="x"):
    return SU.normalized_binary_digest_from_sections(
        ref=ref,
        section_digests={".text": hashlib.sha256(text).hexdigest(),
                         ".rodata": hashlib.sha256(RODATA).hexdigest(),
                         ".data.rel.ro": hashlib.sha256(DATA_REL_RO).hexdigest()},
        dynsym_digest=SHA0)


def rebuild_of(commit=BASE_COMMIT, tc=None):
    return SU.RebuildAttestation(rebuilt_commit=commit, build_dir="build-hip-rebase",
                                 toolchain=tc or toolchain(), build_log_sha256=SHA0)


class Stage1Test(unittest.TestCase):

    def test_change_outside_the_closure_passes(self):
        result = SU.backend_unchanged_stage1_source_closure(
            backend="llama_gpu", diff=make_diff(["ggml/src/ggml-cpu-quants.c"]),
            indexes=[make_index()], candidate_toolchain=toolchain(),
            base_toolchain=toolchain())
        self.assertEqual(result.check.outcome, S.PASS)
        self.assertEqual(result.changed_in_closure, ())

    def test_change_inside_the_closure_fails(self):
        result = SU.backend_unchanged_stage1_source_closure(
            backend="llama_gpu", diff=make_diff(["ggml/src/ggml-cuda/mmq.cuh"]),
            indexes=[make_index()], candidate_toolchain=toolchain(),
            base_toolchain=toolchain())
        self.assertEqual(result.check.outcome, S.FAIL)
        self.assertIn("ggml/src/ggml-cuda/mmq.cuh", result.changed_in_closure)

    def test_shared_header_is_inside_both_closures(self):
        for backend in ("llama_cpu", "llama_gpu"):
            with self.subTest(backend=backend):
                result = SU.backend_unchanged_stage1_source_closure(
                    backend=backend, diff=make_diff(["ggml/include/ggml.h"]),
                    indexes=[make_index()], candidate_toolchain=toolchain(),
                    base_toolchain=toolchain())
                self.assertEqual(result.check.outcome, S.FAIL)

    def test_toolchain_difference_fails_even_with_an_empty_diff(self):
        result = SU.backend_unchanged_stage1_source_closure(
            backend="llama_gpu", diff=make_diff(["ggml/src/ggml-cpu-quants.c"]),
            indexes=[make_index()], candidate_toolchain=toolchain(flags=("-O2",)),
            base_toolchain=toolchain())
        self.assertEqual(result.check.outcome, S.FAIL)
        self.assertTrue(result.toolchain_differences)

    def test_flag_order_is_part_of_identity(self):
        a = toolchain(flags=("-O3", "-march=native"))
        b = toolchain(flags=("-march=native", "-O3"))
        self.assertTrue(a.differences(b))

    def test_unmapped_diff_path_is_could_not_check(self):
        result = SU.backend_unchanged_stage1_source_closure(
            backend="llama_gpu", diff=make_diff(["scripts/build.sh"]),
            indexes=[make_index()], candidate_toolchain=toolchain(),
            base_toolchain=toolchain())
        self.assertEqual(result.check.outcome, S.COULD_NOT_CHECK)
        self.assertEqual(result.unmapped_diff_paths, ("scripts/build.sh",))

    def test_backend_absent_from_the_index_is_could_not_check(self):
        index = make_index(backends={"llama_cpu": ["bin/llama-bench"]})
        result = SU.backend_unchanged_stage1_source_closure(
            backend="whisper_stt", diff=make_diff(["ggml/src/ggml.c"]), indexes=[index],
            candidate_toolchain=toolchain(), base_toolchain=toolchain())
        self.assertEqual(result.check.outcome, S.COULD_NOT_CHECK)
        self.assertEqual(result.closure_size, 0)

    def test_unknown_backend_and_bad_types_raise(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.backend_unchanged_stage1_source_closure(
                backend="nope", diff=make_diff(["a.c"]), indexes=[make_index()],
                candidate_toolchain=toolchain(), base_toolchain=toolchain())
        with self.assertRaises(SU.SurfaceInputError):
            SU.backend_unchanged_stage1_source_closure(
                backend="llama_cpu", diff=make_diff(["a.c"]), indexes=[],
                candidate_toolchain=toolchain(), base_toolchain=toolchain())
        with self.assertRaises(SU.UntrustedProvenance):
            SU.backend_unchanged_stage1_source_closure(
                backend="llama_cpu", diff=make_diff(["a.c"]), indexes=["x"],
                candidate_toolchain=toolchain(), base_toolchain=toolchain())
        with self.assertRaises(SU.SurfaceInputError):
            SU.backend_unchanged_stage1_source_closure(
                backend="llama_cpu", diff=make_diff(["a.c"]), indexes=[make_index()],
                candidate_toolchain="clang", base_toolchain=toolchain())


class Stage2Test(unittest.TestCase):

    def test_without_a_rebuild_attestation_it_is_could_not_check(self):
        result = SU.backend_unchanged_stage2_normalized_binary(
            backend="llama_gpu", candidate_digest=digest_from(ref="cand"),
            base_digest=digest_from(ref="base"), candidate_toolchain=toolchain(),
            base_commit=BASE_COMMIT, rebuild=None)
        self.assertEqual(result.check.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(result.rebuild_verified)
        self.assertIn("non-determinism regime", result.check.reasons[0])

    def test_rebuild_of_the_wrong_commit_is_could_not_check(self):
        result = SU.backend_unchanged_stage2_normalized_binary(
            backend="llama_gpu", candidate_digest=digest_from(ref="cand"),
            base_digest=digest_from(ref="base"), candidate_toolchain=toolchain(),
            base_commit=BASE_COMMIT, rebuild=rebuild_of(commit=CAND_COMMIT))
        self.assertEqual(result.check.outcome, S.COULD_NOT_CHECK)

    def test_rebuild_in_a_different_environment_is_could_not_check(self):
        result = SU.backend_unchanged_stage2_normalized_binary(
            backend="llama_gpu", candidate_digest=digest_from(ref="cand"),
            base_digest=digest_from(ref="base"), candidate_toolchain=toolchain(),
            base_commit=BASE_COMMIT,
            rebuild=rebuild_of(tc=toolchain(compiler_version="16.0.0")))
        self.assertEqual(result.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("candidate's build environment", result.check.reasons[0])

    def test_identical_normalized_sections_pass(self):
        result = SU.backend_unchanged_stage2_normalized_binary(
            backend="llama_gpu", candidate_digest=digest_from(ref="cand"),
            base_digest=digest_from(ref="base"), candidate_toolchain=toolchain(),
            base_commit=BASE_COMMIT, rebuild=rebuild_of())
        self.assertEqual(result.check.outcome, S.PASS)
        self.assertTrue(result.rebuild_verified)

    def test_differing_normalized_sections_fail(self):
        result = SU.backend_unchanged_stage2_normalized_binary(
            backend="llama_gpu", candidate_digest=digest_from(text=b"different", ref="cand"),
            base_digest=digest_from(ref="base"), candidate_toolchain=toolchain(),
            base_commit=BASE_COMMIT, rebuild=rebuild_of())
        self.assertEqual(result.check.outcome, S.FAIL)
        self.assertTrue(result.differing)

    def test_bad_types_raise(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.backend_unchanged_stage2_normalized_binary(
                backend="llama_gpu", candidate_digest="x", base_digest=digest_from(),
                candidate_toolchain=toolchain(), base_commit=BASE_COMMIT, rebuild=None)
        with self.assertRaises(SU.SurfaceInputError):
            SU.backend_unchanged_stage2_normalized_binary(
                backend="llama_gpu", candidate_digest=digest_from(),
                base_digest=digest_from(), candidate_toolchain=toolchain(),
                base_commit=BASE_COMMIT, rebuild="not an attestation")


def in_scope(**overrides):
    base = dict(same_models=True, same_recipes=True,
                candidate_topology_hash="topo-1", incumbent_topology_hash="topo-1",
                era_boundary_crossed=False)
    base.update(overrides)
    return SU.EvidenceTransferScope(**base)


class BackendUnchangedTest(unittest.TestCase):

    def _stage1(self, outcome):
        if outcome == S.PASS:
            return SU.backend_unchanged_stage1_source_closure(
                backend="llama_gpu", diff=make_diff(["ggml/src/ggml-cpu-quants.c"]),
                indexes=[make_index()], candidate_toolchain=toolchain(),
                base_toolchain=toolchain())
        if outcome == S.FAIL:
            return SU.backend_unchanged_stage1_source_closure(
                backend="llama_gpu", diff=make_diff(["ggml/src/ggml-cuda/mmq.cuh"]),
                indexes=[make_index()], candidate_toolchain=toolchain(),
                base_toolchain=toolchain())
        return SU.backend_unchanged_stage1_source_closure(
            backend="llama_gpu", diff=make_diff(["scripts/build.sh"]),
            indexes=[make_index()], candidate_toolchain=toolchain(),
            base_toolchain=toolchain())

    def _stage2(self, outcome):
        candidate = digest_from(ref="cand") if outcome != S.FAIL else \
            digest_from(text=b"different", ref="cand")
        return SU.backend_unchanged_stage2_normalized_binary(
            backend="llama_gpu", candidate_digest=candidate,
            base_digest=digest_from(ref="base"), candidate_toolchain=toolchain(),
            base_commit=BASE_COMMIT,
            rebuild=None if outcome == S.COULD_NOT_CHECK else rebuild_of())

    def test_both_stages_pass_and_evidence_in_scope_permits_dropping_cells(self):
        result = SU.backend_unchanged(stage1=self._stage1(S.PASS),
                                      stage2=self._stage2(S.PASS),
                                      transfer_scope=in_scope())
        self.assertEqual(result.unchanged.outcome, S.PASS)
        self.assertEqual(result.agreement.outcome, S.PASS)
        self.assertTrue(result.may_drop_cells)
        self.assertEqual(result.findings, ())

    def test_stage1_pass_stage2_fail_is_a_hard_disagreement(self):
        result = SU.backend_unchanged(stage1=self._stage1(S.PASS),
                                      stage2=self._stage2(S.FAIL),
                                      transfer_scope=in_scope())
        self.assertEqual(result.agreement.outcome, S.FAIL)
        self.assertEqual(result.unchanged.outcome, S.FAIL)
        self.assertFalse(result.may_drop_cells)
        self.assertEqual([f.code for f in result.findings],
                         [SU.FINDING_STAGE_DISAGREEMENT_SOURCE_CLEAN])
        self.assertEqual(result.findings[0].filed_against, "build_identity")
        self.assertEqual(result.findings[0].severity, "hard")

    def test_stage1_fail_stage2_pass_is_also_a_hard_disagreement(self):
        result = SU.backend_unchanged(stage1=self._stage1(S.FAIL),
                                      stage2=self._stage2(S.PASS),
                                      transfer_scope=in_scope())
        self.assertEqual(result.agreement.outcome, S.FAIL)
        self.assertFalse(result.may_drop_cells)
        self.assertEqual([f.code for f in result.findings],
                         [SU.FINDING_STAGE_DISAGREEMENT_SOURCE_DIRTY])

    def test_the_cheaper_stage_is_never_silently_preferred(self):
        result = SU.backend_unchanged(stage1=self._stage1(S.PASS),
                                      stage2=self._stage2(S.FAIL),
                                      transfer_scope=in_scope())
        self.assertNotEqual(result.unchanged.outcome, S.PASS)
        self.assertTrue(result.blocking_reasons)

    def test_stage2_missing_blocks_the_drop(self):
        result = SU.backend_unchanged(stage1=self._stage1(S.PASS), stage2=None,
                                      transfer_scope=in_scope())
        self.assertEqual(result.unchanged.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(result.may_drop_cells)
        self.assertIn(SU.FINDING_STAGE2_NOT_RUN, [f.code for f in result.findings])

    def test_core_header_forces_the_binary_stage(self):
        result = SU.backend_unchanged(stage1=self._stage1(S.PASS), stage2=None,
                                      transfer_scope=in_scope(),
                                      change_class="core_header")
        self.assertIn(SU.FINDING_CORE_HEADER_REQUIRES_STAGE2,
                      [f.code for f in result.findings])
        self.assertFalse(result.may_drop_cells)

    def test_stage2_could_not_check_blocks_the_drop(self):
        result = SU.backend_unchanged(stage1=self._stage1(S.PASS),
                                      stage2=self._stage2(S.COULD_NOT_CHECK),
                                      transfer_scope=in_scope())
        self.assertEqual(result.unchanged.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(result.may_drop_cells)

    def test_stage1_could_not_check_blocks_the_drop(self):
        result = SU.backend_unchanged(stage1=self._stage1(S.COULD_NOT_CHECK),
                                      stage2=self._stage2(S.PASS),
                                      transfer_scope=in_scope())
        self.assertEqual(result.unchanged.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(result.may_drop_cells)

    def test_unknown_transfer_scope_blocks_the_drop(self):
        result = SU.backend_unchanged(stage1=self._stage1(S.PASS),
                                      stage2=self._stage2(S.PASS))
        self.assertEqual(result.unchanged.outcome, S.PASS)
        self.assertFalse(result.may_drop_cells)
        self.assertTrue(result.blocking_reasons)

    def test_era_boundary_blocks_the_drop(self):
        result = SU.backend_unchanged(stage1=self._stage1(S.PASS),
                                      stage2=self._stage2(S.PASS),
                                      transfer_scope=in_scope(era_boundary_crossed=True))
        self.assertFalse(result.may_drop_cells)
        self.assertEqual(result.transfer_scope.check().outcome, S.FAIL)

    def test_topology_change_blocks_the_drop(self):
        result = SU.backend_unchanged(
            stage1=self._stage1(S.PASS), stage2=self._stage2(S.PASS),
            transfer_scope=in_scope(candidate_topology_hash="topo-2"))
        self.assertFalse(result.may_drop_cells)

    def test_transfer_scope_defaults_to_could_not_check_never_true(self):
        self.assertEqual(SU.EvidenceTransferScope().check().outcome, S.COULD_NOT_CHECK)

    def test_gate_result_requires_an_anchor(self):
        result = SU.backend_unchanged(stage1=self._stage1(S.PASS),
                                      stage2=self._stage2(S.PASS),
                                      transfer_scope=in_scope())
        gate = result.gate_result()
        self.assertTrue(gate.requires_anchor)
        self.assertEqual(gate.gate_class, api.GATE_INTEGRITY)
        self.assertEqual(gate.check.outcome, S.PASS)

    def test_mismatched_backends_raise(self):
        stage2 = SU.backend_unchanged_stage2_normalized_binary(
            backend="llama_cpu", candidate_digest=digest_from(), base_digest=digest_from(),
            candidate_toolchain=toolchain(), base_commit=BASE_COMMIT, rebuild=rebuild_of())
        with self.assertRaises(SU.SurfaceInputError):
            SU.backend_unchanged(stage1=self._stage1(S.PASS), stage2=stage2)

    def test_bad_types_and_change_class_raise(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.backend_unchanged(stage1="nope")
        with self.assertRaises(SU.SurfaceInputError):
            SU.backend_unchanged(stage1=self._stage1(S.PASS), stage2="nope")
        with self.assertRaises(SU.SurfaceInputError):
            SU.backend_unchanged(stage1=self._stage1(S.PASS), transfer_scope="nope")
        with self.assertRaises(SU.SurfaceInputError):
            SU.backend_unchanged(stage1=self._stage1(S.PASS), change_class="nope")

    def test_to_dict_is_content_hashable(self):
        result = SU.backend_unchanged(stage1=self._stage1(S.PASS),
                                      stage2=self._stage2(S.PASS),
                                      transfer_scope=in_scope())
        S.content_hash(result.to_dict())

    def test_finding_code_vocabulary_is_closed(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.BuildIdentityFinding(code="MADE_UP", severity="hard", detail="x")
        with self.assertRaises(SU.SurfaceInputError):
            SU.BuildIdentityFinding(code=SU.FINDING_STAGE2_NOT_RUN, severity="soft",
                                    detail="x")


# =============================================================================
# The seam into api.TierDispatcher
# =============================================================================

# NOT SHA0/SHA1: a single-repeated-character digest is a PLACEHOLDER, and
# `schemas.is_placeholder_digest` refuses one in an anchor — an anchor that says
# `0`*64 claims an identity nothing measured. The artifact digests below may stay
# synthetic; the anchor may not.
ANCHOR = api.AnchorIdentity(
    source_commit=BASE_COMMIT,
    binary_sha256=hashlib.sha256(b"surface-anchor-binary").hexdigest(),
    linkage_sha256=hashlib.sha256(b"surface-anchor-linkage").hexdigest())


def make_request(candidate_id="akc-001", tier="T0", backend="llama_cpu", anchor=ANCHOR):
    return api.EvaluationRequest(
        event_id="ake-001", campaign_id="ak-001", candidate_id=candidate_id, tier=tier,
        backend=backend, phase="decode", cell_class="instrument",
        protocol_id=api.PROTOCOL_VERSIONED_ID,
        artifact=api.ArtifactIdentity(source_sha256=SHA0, binary_sha256=SHA0,
                                      linkage_sha256=SHA1),
        anchor=anchor,
        evaluator=api.EvaluatorIdentity(id="P-AK-SEARCH-1/v1", bundle_sha256=SHA0,
                                        runtime_source_label_ref="srclabel-1"),
        scope_denominator=api.ScopeDenominator(machine_subset="full", numa_nodes=(0, 1, 2, 3),
                                               devices=(), cores=96),
        scope_manifest_sha256=SHA0, co_residency="single",
        determinism=api.DeterminismReport(determinism_class="bitwise_stable",
                                          same_seed_repeat_runs=3),
        metric="decode_tokens_per_s", metric_direction="higher_better", reps=5,
        created_at=NOW,
        campaign_controls=api.CampaignControls(
            calibration_block_count=30, contribution_floor=0.02, max_candidates=100,
            confirmation_admission_count=5, max_blocks_per_candidate=40,
            storage_floor_bytes_free=200 * 1024 ** 3),
        calibration=api.CalibrationOutputs(
            backend=backend, phase="decode", cell_class="instrument",
            noise_floor_phi=0.009, b_min_blocks=10, alpha_sel=0.01, alpha_conf=0.002,
            anchor_gate_band=(0.97, 1.03), accepted=True,
            solve_order_recorded=api.CALIBRATION_SOLVE_ORDER,
            samples_ref="data/ak-001/calibration/aa-blocks.jsonl",
            e_process_construction_id="sign_martingale_predictable_lambda/v1"),
    )


class SurfaceGateRunnerTest(unittest.TestCase):

    def _runner(self, *, escape=False, tier="T0"):
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        events = [CPU_EVENT, GPU_EVENT] if escape else [CPU_EVENT]
        traced = SU.parse_dispatch_trace(make_trace(events=events), trace_ref="t")
        rec = SU.reconcile_surface(derived, traced)
        score = SU.score_actor_declaration(
            SU.ActorDeclaration(candidate_id="akc-001", backends=("llama_cpu",)), derived)
        return SU.SurfaceGateRunner(tier=tier, reconciliation=rec, declaration_score=score)

    def test_runner_returns_gate_results(self):
        gates = self._runner().run_gates(make_request())
        self.assertTrue(all(isinstance(g, api.GateResult) for g in gates))

    def test_runner_refuses_the_wrong_candidate(self):
        with self.assertRaises(SU.SurfaceInputError):
            self._runner().run_gates(make_request(candidate_id="akc-999"))

    def test_runner_refuses_a_non_request(self):
        with self.assertRaises(SU.SurfaceInputError):
            self._runner().run_gates("nope")

    def test_release_tier_is_refused_at_wiring(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"])
        traced = SU.parse_dispatch_trace(make_trace(events=[CPU_EVENT]), trace_ref="t")
        with self.assertRaises(api.TierNotOwned):
            SU.SurfaceGateRunner(tier="T3",
                                 reconciliation=SU.reconcile_surface(derived, traced))

    def test_bad_wiring_raises(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"])
        rec = SU.reconcile_surface(derived, None)
        with self.assertRaises(SU.SurfaceInputError):
            SU.SurfaceGateRunner(tier="T0", reconciliation="nope")
        with self.assertRaises(SU.SurfaceInputError):
            SU.SurfaceGateRunner(tier="T0", reconciliation=rec,
                                 backend_unchanged_results=["nope"])
        with self.assertRaises(SU.SurfaceInputError):
            SU.SurfaceGateRunner(tier="T0", reconciliation=rec, declaration_score="nope")

    def test_the_declaration_never_becomes_a_gate(self):
        runner = self._runner()
        gate_ids = {g.gate_id for g in runner.run_gates(make_request())}
        self.assertFalse(any("declar" in gid for gid in gate_ids))
        self.assertIsNotNone(runner.critic_payload()["declaration_score"])
        self.assertIs(runner.critic_payload()["declaration_score"]["is_scope_input"], False)

    def test_critic_payload_is_content_hashable(self):
        S.content_hash(self._runner().critic_payload())

    def test_dispatch_end_to_end_with_an_escape_yields_no_speed_rank(self):
        # The ONLY difference from the green run below is the trace: one GPU dispatch
        # outside the derived surface. The verdict flips pass -> fail and the speed rank
        # becomes unobtainable, not penalised.
        dispatcher = api.TierDispatcher(gate_runners={"T0": self._runner(escape=True)})
        outcome = dispatcher.dispatch(make_request(), self._window())
        self.assertEqual(outcome.verdict.status, api.STATUS_FAIL)
        self.assertIn("INTEGRITY:surface.reconciliation:FAIL", outcome.verdict.integrity_flags)
        with self.assertRaises(api.SpeedRankUnavailable):
            outcome.verdict.rank_key()

    def test_dispatch_end_to_end_without_an_escape_passes(self):
        dispatcher = api.TierDispatcher(gate_runners={"T0": self._runner(escape=False)})
        outcome = dispatcher.dispatch(make_request(), self._window())
        self.assertEqual(outcome.verdict.status, api.STATUS_PASS)
        gates = {g["gate_id"]: g for g in outcome.verdict.to_dict()["gates"]}
        self.assertEqual(gates["surface.reconciliation"]["outcome"], S.PASS)
        self.assertEqual(gates["surface.no_fallback"]["outcome"], S.PASS)
        self.assertEqual(gates["surface.derived_coverage"]["outcome"], S.PASS)

    def test_a_backend_unchanged_gate_reaches_the_dispatcher(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        traced = SU.parse_dispatch_trace(make_trace(events=[CPU_EVENT]), trace_ref="t")
        stage1 = SU.backend_unchanged_stage1_source_closure(
            backend="llama_gpu", diff=make_diff(["ggml/src/ggml-cuda/mmq.cuh"]),
            indexes=[make_index()], candidate_toolchain=toolchain(),
            base_toolchain=toolchain())
        result = SU.backend_unchanged(stage1=stage1, transfer_scope=in_scope())
        runner = SU.SurfaceGateRunner(
            tier="T0", reconciliation=SU.reconcile_surface(derived, traced),
            backend_unchanged_results=[result])
        dispatcher = api.TierDispatcher(gate_runners={"T0": runner})
        outcome = dispatcher.dispatch(make_request(), self._window())
        gates = {g["gate_id"]: g for g in outcome.verdict.to_dict()["gates"]}
        self.assertIn("build_identity.backend_unchanged.llama_gpu", gates)
        self.assertTrue(gates["build_identity.backend_unchanged.llama_gpu"]["requires_anchor"])

    def test_a_fallback_dispatch_fails_the_run(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        event = dict(CPU_EVENT, fallback=True, fallback_reason="iqk path unsupported")
        traced = SU.parse_dispatch_trace(make_trace(events=[event]), trace_ref="t")
        runner = SU.SurfaceGateRunner(
            tier="T0", reconciliation=SU.reconcile_surface(derived, traced))
        outcome = api.TierDispatcher(gate_runners={"T0": runner}).dispatch(
            make_request(), self._window())
        self.assertEqual(outcome.verdict.status, api.STATUS_FAIL)
        self.assertIn("INTEGRITY:surface.no_fallback:FAIL", outcome.verdict.integrity_flags)

    def test_backend_unchanged_pass_is_demoted_without_an_anchor(self):
        # Precondition 4: a byte-identity label produced without a named anchor
        # comparison is not a verdict. The gate carries requires_anchor=True, so api.py
        # demotes its PASS rather than letting "unchanged" stand unanchored.
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        traced = SU.parse_dispatch_trace(make_trace(events=[CPU_EVENT]), trace_ref="t")
        stage1 = SU.backend_unchanged_stage1_source_closure(
            backend="llama_gpu", diff=make_diff(["ggml/src/ggml-cpu-quants.c"]),
            indexes=[make_index()], candidate_toolchain=toolchain(),
            base_toolchain=toolchain())
        stage2 = SU.backend_unchanged_stage2_normalized_binary(
            backend="llama_gpu", candidate_digest=digest_from(ref="cand"),
            base_digest=digest_from(ref="base"), candidate_toolchain=toolchain(),
            base_commit=BASE_COMMIT, rebuild=rebuild_of())
        result = SU.backend_unchanged(stage1=stage1, stage2=stage2,
                                      transfer_scope=in_scope())
        self.assertEqual(result.gate_result().check.outcome, S.PASS)
        runner = SU.SurfaceGateRunner(
            tier="T0", reconciliation=SU.reconcile_surface(derived, traced),
            backend_unchanged_results=[result])
        anchorless = api.WindowAttestations(
            **{**{f.name: getattr(self._window(), f.name)
                  for f in api.WindowAttestations.__dataclass_fields__.values()},
               "anchor_at_open": None, "anchor_at_close": None})
        outcome = api.TierDispatcher(gate_runners={"T0": runner}).dispatch(
            make_request(anchor=None), anchorless)
        gates = {g["gate_id"]: g for g in outcome.verdict.to_dict()["gates"]}
        self.assertEqual(gates["build_identity.backend_unchanged.llama_gpu"]["outcome"],
                         S.COULD_NOT_CHECK)
        self.assertEqual(outcome.verdict.status, api.STATUS_INVALID)

    def _window(self):
        # Built field by field: `api` deliberately has no all_clear() helper, because a
        # fixture that fabricates PASS is the fixture that removes the signal under test.
        ok = S.Check(S.PASS)
        return api.WindowAttestations(
            resource_claim_receipt="gpu_device.mi210_0:claim-0001",
            resource_claim_open=ok, resource_claim_close=ok,
            resource_claim_same_holder=ok, no_concurrent_inference=ok,
            preflight_attestation_ref="ake-preflight-0001",
            host_receipt="host-health-0001", host_health=ok,
            anchor_at_open=ANCHOR, anchor_at_close=ANCHOR, anchor_gate=ok,
            evaluator_bundle=ok, runtime_source_label=ok,
            recipe=api.RecipeReceipt(constructor_id="ak.microbench/v1",
                                     constructor_sha256=SHA0, argv_sha256=SHA1),
            storage_open=ok, storage_close=ok, strata=ok,
            stopping_rule_id="ak.stopping.bounded_extension/v1", rule_immutability=ok,
            order_randomized=ok, order_seed="campaign-seed-4711", aa_cadence=ok,
            controls=api.ControlPanel(positive=ok, neutral=ok, degraded_negative=ok,
                                      aa=ok, historical_replay=ok),
            calibration=ok, control_definitions_immutable=ok,
            raw_evidence_ref="data/ak-001/raw/akc-001/",
        )


# =============================================================================
# Self-audit
# =============================================================================

class AuditTest(unittest.TestCase):

    def test_module_is_read_only(self):
        result = SU.audit_surface_module_is_read_only()
        self.assertEqual(result.outcome, S.PASS, result.reasons)

    def test_audit_detects_a_write_call(self):
        result = SU.audit_surface_module_is_read_only(
            "def f(p):\n    p.write_text('x')\n")
        self.assertEqual(result.outcome, S.FAIL)

    def test_audit_detects_a_process_launch(self):
        result = SU.audit_surface_module_is_read_only(
            "import subprocess\ndef f():\n    subprocess.run(['ls'])\n")
        self.assertEqual(result.outcome, S.FAIL)

    def test_audit_detects_an_os_import(self):
        self.assertEqual(
            SU.audit_surface_module_is_read_only("from os import remove\n").outcome, S.FAIL)

    def test_audit_permits_a_read_only_open(self):
        for mode in ('"r"', '"rb"', '"rt"'):
            with self.subTest(mode=mode):
                self.assertEqual(
                    SU.audit_surface_module_is_read_only(
                        f"def f(p):\n    return open(p, {mode}).read()\n").outcome, S.PASS)
        self.assertEqual(
            SU.audit_surface_module_is_read_only("def f(p):\n    return open(p)\n").outcome,
            S.PASS)

    def test_audit_rejects_a_write_mode_open(self):
        for mode in ('"w"', '"a"', '"r+"', '"xb"'):
            with self.subTest(mode=mode):
                self.assertEqual(
                    SU.audit_surface_module_is_read_only(
                        f"def f(p):\n    return open(p, {mode})\n").outcome, S.FAIL)

    def test_audit_rejects_a_non_literal_mode(self):
        self.assertEqual(
            SU.audit_surface_module_is_read_only(
                "def f(p, m):\n    return open(p, m)\n").outcome, S.FAIL)
        self.assertEqual(
            SU.audit_surface_module_is_read_only(
                "def f(p, m):\n    return open(p, mode=m)\n").outcome, S.FAIL)

    def test_audit_reports_could_not_check_on_unparseable_source(self):
        self.assertEqual(SU.audit_surface_module_is_read_only("def (:").outcome,
                         S.COULD_NOT_CHECK)

    def test_api_audit_still_passes(self):
        self.assertEqual(api.audit_no_write_or_process_paths().outcome, S.PASS)


# =============================================================================
# ADVERSARIAL REGRESSION BARRIER
#
# Every test below reproduces a defect found by red-teaming the first version of
# surface.py. Each one FAILED before its fix. They are grouped by the property the
# defect broke, not by the function that held it, because in every case the module
# claimed the property in prose while the code delivered it only on the happy path.
# =============================================================================

class LinkClosureIsTransitiveTest(unittest.TestCase):
    """DEFECT 1 — the closure stopped at the first link edge.

    A real llama.cpp `link.txt` for `llama-bench` names `ggml/src/libggml.so`, not
    `ggml.c.o`: the ggml objects live inside the shared library. `LinkEdge` recorded
    only `.o` tokens and dropped every library input, so
    `source_closure_for_backend("llama_cpu")` returned "llama-bench.cpp only" and
    `backend_unchanged_stage1_source_closure()` answered PASS ("unchanged") for a
    `ggml/src/ggml-cpu.c` edit — with `coverage_check()` PASS, no widening and no
    reason. A silent under-approximation, in the one place the module's docstring
    promises there is never one.
    """

    LINK_LIBGGML = ("/usr/bin/c++ -fPIC -O3 -shared CMakeFiles/ggml.dir/ggml.c.o "
                    "CMakeFiles/ggml.dir/ggml-cpu.c.o -o ggml/src/libggml.so -lm")
    LINK_BENCH_LIB = ("/usr/bin/c++ -O3 CMakeFiles/llama-bench.dir/llama-bench.cpp.o "
                      "-o bin/llama-bench -Wl,-rpath,/x ggml/src/libggml.so -lm")
    DEPFILE_BENCH = ("CMakeFiles/llama-bench.dir/llama-bench.cpp.o: "
                     "../tools/llama-bench/llama-bench.cpp ../ggml/include/ggml.h\n")

    def _index(self, *, with_libggml_link=True):
        dep_edges = []
        for name, text in (("ggml.d", DEPFILE_GGML), ("cpu.d", DEPFILE_CPU),
                           ("bench.d", self.DEPFILE_BENCH)):
            dep_edges.extend(SU.parse_make_depfile(text, origin_ref=name))
        links = [SU.parse_cmake_link_txt(self.LINK_BENCH_LIB, origin_ref="bench/link.txt")]
        if with_libggml_link:
            links.insert(0, SU.parse_cmake_link_txt(self.LINK_LIBGGML,
                                                    origin_ref="libggml/link.txt"))
        return SU.build_dependency_index(
            label="candidate", build_dir=BUILD_DIR, source_root=SOURCE_ROOT,
            dep_edges=dep_edges, link_edges=links,
            backend_link_targets={"llama_cpu": ["bin/llama-bench"]})

    def test_library_input_is_recorded_not_dropped(self):
        edge = SU.parse_cmake_link_txt(self.LINK_BENCH_LIB, origin_ref="x")
        self.assertEqual(edge.library_inputs, ("ggml/src/libggml.so",))
        self.assertEqual(edge.objects, ("CMakeFiles/llama-bench.dir/llama-bench.cpp.o",))

    def test_flags_and_rpaths_are_not_mistaken_for_libraries(self):
        edge = SU.parse_cmake_link_txt(LINK_SERVER, origin_ref="x")
        self.assertEqual(edge.library_inputs, ())

    def test_closure_reaches_through_the_shared_library(self):
        closure = self._index().source_closure_for_backend("llama_cpu")
        self.assertIn("ggml/src/ggml-cpu.c", closure)
        self.assertIn("ggml/src/ggml.c", closure)

    def test_a_ggml_edit_is_not_unchanged_for_the_backend_that_links_ggml(self):
        stage1 = SU.backend_unchanged_stage1_source_closure(
            backend="llama_cpu", diff=make_diff(["ggml/src/ggml-cpu.c"]),
            indexes=[self._index()], candidate_toolchain=toolchain(),
            base_toolchain=toolchain())
        self.assertEqual(stage1.check.outcome, S.FAIL)
        self.assertIn("ggml/src/ggml-cpu.c", stage1.changed_in_closure)

    def test_the_library_target_is_attributed_to_the_backend_that_links_it(self):
        index = self._index()
        self.assertIn("llama_cpu",
                      index.backends_for_link_target("build-hip/ggml/src/libggml.so"))
        derived = SU.derive_affected_surface(
            candidate_id="akc-001", diff=make_diff(["ggml/src/ggml-cpu.c"]),
            indexes=[index], registrations=make_registrations())
        self.assertEqual(derived.backends, ("llama_cpu",))
        self.assertFalse(derived.full_tree)

    def test_an_unresolvable_library_input_fails_closed(self):
        index = self._index(with_libggml_link=False)
        self.assertTrue(index.unresolved_link_inputs)
        self.assertEqual(index.coverage_check().outcome, S.COULD_NOT_CHECK)
        stage1 = SU.backend_unchanged_stage1_source_closure(
            backend="llama_cpu", diff=make_diff(["ggml/src/ggml-cpu.c"]),
            indexes=[index], candidate_toolchain=toolchain(), base_toolchain=toolchain())
        self.assertNotEqual(stage1.check.outcome, S.PASS)
        derived = SU.derive_affected_surface(
            candidate_id="akc-001", diff=make_diff(["ggml/src/ggml-cpu.c"]),
            indexes=[index])
        self.assertTrue(derived.full_tree)

    def test_json_manifest_carries_library_inputs(self):
        edges = SU.parse_link_manifest_json(
            json.dumps({"links": [{"link_target": "bin/x", "objects": ["a.o"],
                                   "library_inputs": ["lib/y.so"]}]}),
            origin_ref="m.json")
        self.assertEqual(edges[0].library_inputs, ("lib/y.so",))
        with self.assertRaises(SU.DepfileParseError):
            SU.parse_link_manifest_json(
                json.dumps({"links": [{"link_target": "bin/x", "objects": [],
                                       "library_inputs": [1]}]}), origin_ref="m.json")

    def test_an_empty_closure_is_never_unchanged(self):
        # A link target the build emitted with no objects at all: every diff path is
        # outside an empty closure, so the previous code answered PASS for any change.
        index = SU.build_dependency_index(
            label="candidate", build_dir=BUILD_DIR, source_root=SOURCE_ROOT,
            dep_edges=list(SU.parse_make_depfile(DEPFILE_CPU, origin_ref="cpu.d")),
            link_edges=[SU.parse_cmake_link_txt(LINK_BENCH, origin_ref="b"),
                        SU.LinkEdge(link_target="bin/llama-server", objects=(),
                                    origin_ref="s")],
            backend_link_targets={"llama_cpu": ["bin/llama-bench"],
                                  "llama_gpu": ["bin/llama-server"]})
        stage1 = SU.backend_unchanged_stage1_source_closure(
            backend="llama_gpu", diff=make_diff(["ggml/include/ggml.h"]),
            indexes=[index], candidate_toolchain=toolchain(), base_toolchain=toolchain())
        self.assertEqual(stage1.closure_size, 0)
        self.assertEqual(stage1.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("EMPTY", stage1.check.reasons[0])

    def test_lname_library_references_remain_a_declared_residual(self):
        # `-L<dir> -lggml` cannot be resolved to a build target by this parser. The
        # limitation is real and is asserted here so it is tracked rather than
        # rediscovered: a build that emits `-l` for an IN-TREE library still
        # under-approximates. CMake emits full paths for in-tree libraries.
        edge = SU.parse_cmake_link_txt(
            "/usr/bin/c++ x.o -o bin/b -L ggml/src -lggml -lm", origin_ref="x")
        self.assertEqual(edge.library_inputs, ())


class AbsenceIsNeverContainmentTest(unittest.TestCase):
    """DEFECT 2 — a gating axis PASSed against an empty observation.

    `link_target` is an OPTIONAL dispatch-trace field. A trace that omitted it
    produced `traced.link_targets == ()`, which contains no escape, so the
    `link_targets` axis — one of the two GATING axes freeze scope is computed from —
    returned PASS. Deleting one field from every event flipped a real link-target
    escape from FAIL to PASS and stamped `reconciled: true` on the candidate record.
    """

    def _derived(self, paths=("ggml/src/ggml-cpu.c",)):
        return derived_for(list(paths), registrations=make_registrations())

    def test_gating_axis_with_no_traced_values_is_could_not_check(self):
        event = {k: v for k, v in CPU_EVENT.items() if k != "link_target"}
        traced = SU.parse_dispatch_trace(make_trace(events=[event]), trace_ref="t")
        self.assertEqual(traced.link_targets, ())
        rec = SU.reconcile_surface(self._derived(), traced)
        axis = [a for a in rec.axes if a.axis == SU.AXIS_LINK_TARGETS][0]
        self.assertEqual(axis.check.outcome, S.COULD_NOT_CHECK)
        self.assertEqual(rec.check.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(SU.candidate_affected_surface_block(rec)["reconciled"])

    def test_dropping_the_field_cannot_convert_an_escape_into_a_pass(self):
        derived = derived_for(["ggml/src/ggml-cuda/ggml-cuda.cu"],
                              registrations=make_registrations())
        escaping = dict(GPU_EVENT, link_target="build-hip/bin/llama-bench")
        with_field = SU.reconcile_surface(derived, SU.parse_dispatch_trace(
            make_trace(events=[escaping]), trace_ref="t"))
        self.assertEqual(with_field.check.outcome, S.FAIL)
        without = {k: v for k, v in GPU_EVENT.items() if k != "link_target"}
        without_field = SU.reconcile_surface(derived, SU.parse_dispatch_trace(
            make_trace(events=[without]), trace_ref="t"))
        self.assertNotEqual(without_field.check.outcome, S.PASS)

    def test_a_populated_gating_axis_still_passes(self):
        # The guard must not forbid the compliant path: a trace that DOES carry link
        # targets, all inside the derived surface, is still a PASS.
        traced = SU.parse_dispatch_trace(make_trace(events=[CPU_EVENT]), trace_ref="t")
        self.assertEqual(SU.reconcile_surface(self._derived(), traced).check.outcome,
                         S.PASS)


class TracedSurfaceCarriesItsInvariantInTheTypeTest(unittest.TestCase):
    """DEFECT 3 — `TracedSurface` validated nothing it asserted.

    `TracedSurface` is public API, and its `__post_init__` checked only that the two
    id strings were non-empty. A caller could therefore construct a zero-event trace
    stamped `completeness=PASS`, hand it to `reconcile_surface()`, and obtain
    `reconciled: true` — the "absence is containment" answer the module exists to
    refuse — or stamp `no_fallback=PASS` on a trace holding a fallback dispatch and
    clear the §8.6 gate. Every other trust-carrying dataclass here enforces its
    invariant in `__post_init__`; this one did not.
    """

    def test_an_empty_trace_cannot_be_declared_complete(self):
        with self.assertRaises(SU.TraceParseError):
            SU.TracedSurface(candidate_id="akc-001", trace_ref="t", events=(),
                             truncated=False, completeness=S.Check(S.PASS),
                             no_fallback=S.Check(S.COULD_NOT_CHECK))

    def test_a_truncated_trace_cannot_be_declared_complete(self):
        event = SU.DispatchEvent(op_name="MUL_MAT", backend="llama_cpu", kernel_symbol="k")
        with self.assertRaises(SU.TraceParseError):
            SU.TracedSurface(candidate_id="akc-001", trace_ref="t", events=(event,),
                             truncated=True, completeness=S.Check(S.PASS),
                             no_fallback=S.Check(S.PASS))

    def test_no_fallback_cannot_be_declared_over_a_fallback_dispatch(self):
        event = SU.DispatchEvent(op_name="MUL_MAT", backend="llama_cpu",
                                 kernel_symbol="generic", fallback=True,
                                 fallback_reason="iqk unsupported")
        with self.assertRaises(SU.TraceParseError):
            SU.TracedSurface(candidate_id="akc-001", trace_ref="t", events=(event,),
                             truncated=False, completeness=S.Check(S.PASS),
                             no_fallback=S.Check(S.PASS))

    def test_no_fallback_cannot_be_proved_by_zero_observations(self):
        with self.assertRaises(SU.TraceParseError):
            SU.TracedSurface(candidate_id="akc-001", trace_ref="t", events=(),
                             truncated=False, completeness=S.Check(S.COULD_NOT_CHECK),
                             no_fallback=S.Check(S.PASS))

    def test_events_and_checks_are_type_checked(self):
        with self.assertRaises(SU.TraceParseError):
            SU.TracedSurface(candidate_id="akc-001", trace_ref="t", events=("nope",),
                             truncated=False, completeness=S.Check(S.COULD_NOT_CHECK),
                             no_fallback=S.Check(S.COULD_NOT_CHECK))
        with self.assertRaises(SU.TraceParseError):
            SU.TracedSurface(candidate_id="akc-001", trace_ref="t", events=(),
                             truncated=False, completeness="PASS",
                             no_fallback=S.Check(S.COULD_NOT_CHECK))

    def test_the_factory_still_produces_a_constructible_surface(self):
        # The guard must not forbid the module's own idiom.
        traced = SU.parse_dispatch_trace(make_trace(events=[CPU_EVENT]), trace_ref="t")
        self.assertEqual(traced.completeness.outcome, S.PASS)
        for events, truncated in (([], False), ([CPU_EVENT], True)):
            surface = SU.parse_dispatch_trace(
                make_trace(events=events, truncated=truncated), trace_ref="t")
            self.assertEqual(surface.completeness.outcome, S.COULD_NOT_CHECK)


class DerivationGapIsNeverACandidateFindingTest(unittest.TestCase):
    """DEFECT 4 — an extractor that produced nothing became a hard candidate FAIL.

    A `SymbolRegistrationIndex` naming no symbol at all — an `nm`/clang-index run
    that produced nothing — was treated as a successful derivation. The three symbol
    axes were marked DERIVED with an empty derived set, so reconciliation reported
    every traced op, symbol and predicate as `traced ⊄ derived` and returned a hard
    candidate failure. That is a gap in the instrument filed as a finding about the
    candidate, which the reconciliation contract forbids in the opposite direction
    (no index at all was already COULD_NOT_CHECK).
    """

    def test_an_index_naming_no_symbol_is_a_gap_not_a_finding(self):
        empty = SU.SymbolRegistrationIndex(label="nm (produced nothing)",
                                           symbols_by_source={},
                                           registrations_by_symbol={})
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=empty)
        self.assertNotIn(SU.AXIS_OP_NAMES, derived.axes_derived)
        self.assertIn(SU.OA_NO_SYMBOL_INDEX,
                      {o.reason for o in derived.over_approximations})
        traced = SU.parse_dispatch_trace(make_trace(events=[CPU_EVENT]), trace_ref="t")
        rec = SU.reconcile_surface(derived, traced)
        self.assertFalse(rec.hard_failure)
        self.assertEqual(rec.check.outcome, S.COULD_NOT_CHECK)

    def test_the_gap_names_the_index_that_produced_nothing(self):
        empty = SU.SymbolRegistrationIndex(label="nm-run-42", symbols_by_source={},
                                           registrations_by_symbol={})
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=empty)
        gap = [o for o in derived.over_approximations
               if o.reason == SU.OA_NO_SYMBOL_INDEX][0]
        self.assertIn("nm-run-42", gap.trigger)
        self.assertIsNotNone(derived.inputs["registrations"])

    def test_a_populated_index_still_derives_and_still_fails_on_an_escape(self):
        derived = derived_for(["ggml/src/ggml-cpu.c"], registrations=make_registrations())
        self.assertIn(SU.AXIS_OP_NAMES, derived.axes_derived)
        event = dict(CPU_EVENT, op_name="FLASH_ATTN_EXT", kernel_symbol="ggml_fa")
        rec = SU.reconcile_surface(derived, SU.parse_dispatch_trace(
            make_trace(events=[event]), trace_ref="t"))
        self.assertEqual(rec.check.outcome, S.FAIL)


class NormalizedDigestCannotBeEmptyTest(unittest.TestCase):
    """DEFECT 5 — a digest of nothing compared identical to a digest of nothing.

    `NormalizedBinaryDigest` required every name in `COMPARED_SECTIONS` to be
    PRESENT, and the sentinel `"ABSENT"` satisfied that. Two all-absent digests
    therefore produced `differences() == ()`, stage 2 returned PASS, and
    `backend_unchanged()` returned `may_drop_cells=True`: a failed or stubbed
    extractor read as "the binary is unchanged, drop the backend's cells".
    """

    def test_an_absent_text_section_is_refused(self):
        with self.assertRaises(SU.NormalizationViolation):
            SU.normalized_binary_digest_from_sections(
                ref="candidate",
                section_digests={".text": SU.SECTION_ABSENT,
                                 ".rodata": SU.SECTION_ABSENT,
                                 ".data.rel.ro": SU.SECTION_ABSENT},
                dynsym_digest=SU.SECTION_ABSENT)

    def test_a_real_binary_with_absent_optional_sections_is_still_accepted(self):
        # The guard must not forbid the compliant path: a stripped, statically linked
        # binary genuinely has no .data.rel.ro and no .dynsym.
        digest = SU.normalized_binary_digest_from_sections(
            ref="static", section_digests={".text": "a" * 64,
                                           ".rodata": SU.SECTION_ABSENT,
                                           ".data.rel.ro": SU.SECTION_ABSENT},
            dynsym_digest=SU.SECTION_ABSENT)
        self.assertEqual(digest.absent_sections, (".rodata", ".data.rel.ro"))


class AuditCoversAttributeOpenTest(unittest.TestCase):
    """DEFECT 6 — the read-only audit inspected only the bare `open` name.

    `audit_surface_module_is_read_only()` routed `open(...)` through the mode check
    but treated `Path(p).open("w")` as an ordinary attribute call whose name was not
    in the forbidden set — so it PASSed. `io.open` and `codecs.open` passed for the
    same reason, their modules not being in `_FORBIDDEN_IMPORTS`. The audit's whole
    claim is that a write-capable `open` cannot clear it.
    """

    def test_a_write_mode_path_open_is_detected(self):
        for mode in ('"w"', '"a"', '"r+"', '"xb"'):
            with self.subTest(mode=mode):
                result = SU.audit_surface_module_is_read_only(
                    f"from pathlib import Path\ndef f(p):\n    Path(p).open({mode})\n")
                self.assertEqual(result.outcome, S.FAIL)

    def test_a_non_literal_path_open_mode_is_detected(self):
        result = SU.audit_surface_module_is_read_only(
            "from pathlib import Path\ndef f(p, m):\n    Path(p).open(m)\n")
        self.assertEqual(result.outcome, S.FAIL)
        result = SU.audit_surface_module_is_read_only(
            "from pathlib import Path\ndef f(p, m):\n    Path(p).open(mode=m)\n")
        self.assertEqual(result.outcome, S.FAIL)

    def test_a_read_only_path_open_is_still_permitted(self):
        # The guard must not forbid its own idiom: this module reads files.
        for source in ("from pathlib import Path\ndef f(p):\n    Path(p).open('rb')\n",
                       "from pathlib import Path\ndef f(p):\n    Path(p).open()\n"):
            with self.subTest(source=source):
                self.assertEqual(
                    SU.audit_surface_module_is_read_only(source).outcome, S.PASS)

    def test_alternative_open_providers_are_forbidden_imports(self):
        for module in ("io", "codecs", "gzip", "mmap", "pickle"):
            with self.subTest(module=module):
                self.assertEqual(
                    SU.audit_surface_module_is_read_only(
                        f"import {module}\ndef f(p):\n    {module}.open(p, 'w')\n").outcome,
                    S.FAIL)

    def test_the_module_still_audits_clean(self):
        self.assertEqual(SU.audit_surface_module_is_read_only().outcome, S.PASS)


class TraceParserHasNoTolerantModeTest(unittest.TestCase):
    """DEFECT 7 — the parser coerced where it documents that it refuses.

    `fallback=bool(obj.get("fallback", False))` coerced asymmetrically: `"false"`
    and `"no"` became True, while `[]`, `{}` and `null` became False — so a producer
    emitting `"fallback": null` for "unknown" obtained a no-fallback PASS. The
    coercion also made `DispatchEvent`'s own `isinstance(..., bool)` check
    unreachable from the parser. `link_target` was not type-checked at all: a
    non-string value either vanished from the traced set (feeding the gating-axis
    defect above) or raised an unhandled `TypeError` from a property, long after the
    parser had accepted the line.
    """

    def _trace(self, event):
        return (json.dumps({"schema": SU.DISPATCH_TRACE_SCHEMA,
                            "candidate_id": "akc-001"}) + "\n"
                + json.dumps(event) + "\n")

    def test_fallback_must_be_a_json_boolean(self):
        for value in (0, 1, "false", "true", "no", [], {}, None):
            with self.subTest(value=value):
                with self.assertRaises(SU.TraceParseError):
                    SU.parse_dispatch_trace(self._trace(
                        {"op_name": "MUL_MAT", "backend": "llama_cpu",
                         "kernel_symbol": "k", "fallback": value}), trace_ref="t")

    def test_real_booleans_still_parse(self):
        clean = SU.parse_dispatch_trace(self._trace(
            {"op_name": "MUL_MAT", "backend": "llama_cpu", "kernel_symbol": "k",
             "fallback": False}), trace_ref="t")
        self.assertEqual(clean.no_fallback.outcome, S.PASS)
        fell_back = SU.parse_dispatch_trace(self._trace(
            {"op_name": "MUL_MAT", "backend": "llama_cpu", "kernel_symbol": "k",
             "fallback": True, "fallback_reason": "iqk unsupported"}), trace_ref="t")
        self.assertEqual(fell_back.no_fallback.outcome, S.FAIL)

    def test_optional_string_fields_are_type_checked(self):
        for key in ("link_target", "dispatch_predicate", "fallback_reason"):
            for value in ([], 123, {"a": 1}, "", "   "):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(SU.TraceParseError):
                        SU.parse_dispatch_trace(self._trace(
                            {"op_name": "MUL_MAT", "backend": "llama_cpu",
                             "kernel_symbol": "k", key: value}), trace_ref="t")

    def test_absent_optional_fields_are_still_fine(self):
        traced = SU.parse_dispatch_trace(self._trace(
            {"op_name": "MUL_MAT", "backend": "llama_cpu", "kernel_symbol": "k",
             "link_target": None}), trace_ref="t")
        self.assertEqual(traced.link_targets, ())


class TwoStagesMustBeAboutOneTreeTest(unittest.TestCase):
    """DEFECT 9 — `backend_unchanged()` combined two stages about different bases.

    Stage 1 is a diff over `production_base..candidate`; stage 2 is a normalized
    comparison against a named `base_commit`. `backend_unchanged()` checked only that
    the two stages named the same BACKEND, so a stage 1 taken over one base and a
    stage 2 taken against another combined into one "unchanged" verdict from two
    unrelated facts — and their "agreement" was the thing that authorised dropping
    the backend's cells.
    """

    OTHER_BASE = "b" * 40

    def _stage1(self, base=BASE_COMMIT):
        return SU.backend_unchanged_stage1_source_closure(
            backend="llama_gpu",
            diff=make_diff(["ggml/src/ggml-cpu-quants.c"], base=base),
            indexes=[make_index()], candidate_toolchain=toolchain(),
            base_toolchain=toolchain())

    def _stage2(self, base=BASE_COMMIT):
        return SU.backend_unchanged_stage2_normalized_binary(
            backend="llama_gpu", candidate_digest=digest_from(ref="cand"),
            base_digest=digest_from(ref="base"), candidate_toolchain=toolchain(),
            base_commit=base, rebuild=rebuild_of(commit=base))

    def test_the_stages_record_the_base_they_are_about(self):
        self.assertEqual(self._stage1().base_commit, BASE_COMMIT)
        self.assertEqual(self._stage1().candidate_commit, CAND_COMMIT)
        self.assertEqual(self._stage2().base_commit, BASE_COMMIT)

    def test_disagreeing_bases_are_a_wiring_defect_not_a_verdict(self):
        with self.assertRaises(SU.SurfaceInputError):
            SU.backend_unchanged(stage1=self._stage1(),
                                 stage2=self._stage2(base=self.OTHER_BASE))

    def test_matching_bases_still_combine(self):
        result = SU.backend_unchanged(stage1=self._stage1(), stage2=self._stage2(),
                                      transfer_scope=in_scope())
        self.assertTrue(result.may_drop_cells)

    def test_a_hand_built_stage_without_commits_still_combines(self):
        # `None` means "unknown", and an unknown cross-check must not become a hard
        # refusal of a caller that predates the field.
        stage1 = SU.SourceClosureIdentity(
            backend="llama_gpu", closure_size=3, changed_in_closure=(),
            unmapped_diff_paths=(), toolchain_differences=(), check=S.Check(S.PASS))
        result = SU.backend_unchanged(stage1=stage1, stage2=self._stage2(),
                                      transfer_scope=in_scope())
        self.assertTrue(result.may_drop_cells)


class TruncationIsDisclosedTest(unittest.TestCase):
    """DEFECT 8 — reason strings truncated their evidence lists without saying so."""

    def test_the_unmapped_reason_names_how_many_it_omitted(self):
        paths = [f"docs/f{i:02d}.md" for i in range(25)]
        stage1 = SU.backend_unchanged_stage1_source_closure(
            backend="llama_cpu", diff=make_diff(paths), indexes=[make_index()],
            candidate_toolchain=toolchain(), base_toolchain=toolchain())
        self.assertEqual(len(stage1.unmapped_diff_paths), 25)
        self.assertIn("15 more", stage1.check.reasons[0])

    def test_the_stage_disagreement_detail_names_how_many_it_omitted(self):
        changed = tuple(f"ggml/src/f{i:02d}.c" for i in range(9))
        stage1 = SU.SourceClosureIdentity(
            backend="llama_cpu", closure_size=99, changed_in_closure=changed,
            unmapped_diff_paths=(), toolchain_differences=(),
            check=S.Check(S.FAIL, ("changed",)))
        stage2 = SU.NormalizedBinaryIdentity(
            backend="llama_cpu", candidate_ref="c", base_ref="b", differing=(),
            rebuild_verified=True, check=S.Check(S.PASS))
        result = SU.backend_unchanged(stage1=stage1, stage2=stage2)
        detail = result.findings[0].detail
        self.assertIn("9 changes", detail)
        self.assertIn("4 more", detail)
        self.assertFalse(result.may_drop_cells)


if __name__ == "__main__":
    unittest.main(verbosity=2)
