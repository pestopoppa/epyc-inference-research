#!/usr/bin/env python3
"""test_integrity.py — the regression barrier for the §8.5.1 source-integrity gates.

WHY THIS FILE EXISTS
--------------------
AutoPilot destroyed `src/escalation.py` (454 lines -> 3) with an edit that passed
`ast.parse()`. Its four Python defenses do not transfer to compiled C++/HIP, and
the failure mode this suite guards is not "the gate is missing" but "the gate is
present and answers PASS anyway":

  * an empty anchor symbol table diffs clean against everything;
  * a registration extractor with no patterns finds no removals;
  * a shrinkage check with no file length has nothing to divide by;
  * a behavioural gate that still runs after an integrity failure produces
    passing correctness evidence for a binary whose ABI was never verified.

Each of those is asserted here as a `COULD_NOT_CHECK` or a refusal, never as a
PASS. AK3's red-team list is covered by name in `TestRedTeamCandidates`: a
candidate that deletes a template specialization, one that drops a dispatch case,
one that removes an op registration, and one whose incremental tree compiles
while its snapshot does not — each must FAIL, and each must fail BEFORE any
behavioural check runs.

The ELF reader is checked against BOTH a synthetic fixture this file builds and a
REAL shared library on the host. The synthetic fixture alone would be circular:
the writer and the reader could share the same misunderstanding of the format and
agree perfectly.

NO inference, NO benchmark, NO kernel build, NO process. The suite writes only
into its own `tempfile` directories, and it asserts that the module under test
cannot write at all by running that module's own AST self-audit — plus three
negative tests proving the audit detects a write path, a process path and a
write-mode `open`.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_integrity.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/evaluator/test_integrity.py
"""
from __future__ import annotations

import hashlib
import struct
import sys
import tempfile
import unittest
from pathlib import Path

# Import through the PACKAGE so `integrity.schemas` is the same module object the
# journal validates with (README, "Import convention").
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import schemas as S  # noqa: E402
from autokernel.evaluator import api  # noqa: E402
from autokernel.evaluator import integrity as I  # noqa: E402
from autokernel.evaluator import devices as D  # noqa: E402

PASS = S.Check(S.PASS)
NOW = "2026-08-03T12:00:00+00:00"
V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"


def sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def outcomes(gates) -> dict:
    return {g.gate_id: g.check.outcome for g in gates}


def reasons_of(gates, gate_id: str) -> str:
    for g in gates:
        if g.gate_id == gate_id:
            return " | ".join(g.check.reasons)
    raise AssertionError(f"no gate {gate_id!r} in {[g.gate_id for g in gates]}")


# ---------------------------------------------------------------------------
# A minimal ELF64 writer, so the reader can be exercised on controlled input.
# It is NOT the only check on the reader: TestElfReaderAgainstRealBinary reads a
# shared library this file did not write, precisely so a shared misunderstanding
# between writer and reader cannot pass.
# ---------------------------------------------------------------------------

_BIND = {"LOCAL": 0, "GLOBAL": 1, "WEAK": 2, "GNU_UNIQUE": 10}
_TYPE = {"NOTYPE": 0, "OBJECT": 1, "FUNC": 2, "SECTION": 3, "FILE": 4, "TLS": 6,
         "GNU_IFUNC": 10}
_VIS = {"DEFAULT": 0, "INTERNAL": 1, "HIDDEN": 2, "PROTECTED": 3}


def build_elf64(symbols, *, table=".dynsym") -> bytes:
    """Build a valid little-endian ELF64 with one symbol table.

    `symbols` is a sequence of `(name, bind, type, visibility, defined)`.
    """
    str_table = ".dynstr" if table == ".dynsym" else ".strtab"
    sym_type = 11 if table == ".dynsym" else 2

    strtab = bytearray(b"\x00")
    offsets = {}
    for name, *_rest in symbols:
        offsets[name] = len(strtab)
        strtab += name.encode("utf-8") + b"\x00"

    syms = bytearray(struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0))  # index 0 is null
    for name, bind, styp, vis, defined in symbols:
        info = (_BIND[bind] << 4) | _TYPE[styp]
        shndx = 1 if defined else 0
        syms += struct.pack("<IBBHQQ", offsets[name], info, _VIS[vis], shndx, 0x1000, 8)

    shstr = bytearray(b"\x00")
    sh_off = {}
    for name in (str_table, table, ".shstrtab"):
        sh_off[name] = len(shstr)
        shstr += name.encode("ascii") + b"\x00"

    ehdr_size = 64
    o_strtab = ehdr_size
    o_syms = o_strtab + len(strtab)
    o_shstr = o_syms + len(syms)
    o_shdrs = o_shstr + len(shstr)

    # e_ident is exactly 16 bytes: magic(4) class data version osabi abiversion pad(7)
    ident = b"\x7fELF" + bytes([2, 1, 1, 0, 0]) + b"\x00" * 7
    assert len(ident) == 16
    ehdr = ident + struct.pack(
        "<HHIQQQIHHHHHH",
        3,        # e_type ET_DYN
        62,       # e_machine x86-64
        1,        # e_version
        0,        # e_entry
        0,        # e_phoff
        o_shdrs,  # e_shoff
        0,        # e_flags
        64,       # e_ehsize
        0, 0,     # e_phentsize, e_phnum
        64, 4,    # e_shentsize, e_shnum
        3,        # e_shstrndx
    )

    def shdr(name_off, sh_type, offset, size, link=0, info=0, entsize=0):
        return struct.pack("<IIQQQQIIQQ", name_off, sh_type, 0, 0, offset, size,
                           link, info, 1, entsize)

    shdrs = b"".join([
        shdr(0, 0, 0, 0),
        shdr(sh_off[str_table], 3, o_strtab, len(strtab)),
        shdr(sh_off[table], sym_type, o_syms, len(syms), link=1, info=1, entsize=24),
        shdr(sh_off[".shstrtab"], 3, o_shstr, len(shstr)),
    ])
    return bytes(ehdr) + bytes(strtab) + bytes(syms) + bytes(shstr) + shdrs


def write_elf(directory: Path, name: str, symbols, **kw) -> Path:
    path = directory / name
    path.write_bytes(build_elf64(symbols, **kw))
    return path


def fn(name: str, *, defined=True, bind="GLOBAL", vis="DEFAULT"):
    return (name, bind, "FUNC", vis, defined)


# Anchor ABI used throughout: two template specializations, a C entry point and a
# C++ member. The candidate variants below each break exactly one thing.
ANCHOR_SYMS = [
    fn("_ZN4ggml6detail15kernel_dispatchILi4EEEvPKfPfi"),
    fn("_ZN4ggml6detail15kernel_dispatchILi8EEEvPKfPfi"),
    fn("_ZN4ggml6TensorD1Ev"),
    fn("ggml_mul_mat"),
    fn("ggml_backend_hip_supports_op"),
    fn("_ZN4ggml3addEPKfS1_Pfi"),
]


def elf_table(tmp: Path, name: str, symbols, label: str) -> I.ElfSymbolTable:
    return I.extract_elf_symbols(write_elf(tmp, name, symbols), label=label)


def deltas(added=(), removed=(), arity_changed=()) -> I.DeclaredSymbolDeltas:
    return I.DeclaredSymbolDeltas(added=frozenset(added), removed=frozenset(removed),
                                  arity_changed=frozenset(arity_changed))


# ---------------------------------------------------------------------------
# ELF reader
# ---------------------------------------------------------------------------

class TestElfReader(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_reads_dynsym_and_classifies_bindings(self):
        table = elf_table(self.tmp, "cand.so", [
            fn("exported_fn"),
            fn("hidden_fn", vis="HIDDEN"),
            fn("undefined_fn", defined=False),
            ("local_fn", "LOCAL", "FUNC", "DEFAULT", True),
            fn("weak_fn", bind="WEAK"),
        ], "candidate")
        self.assertEqual(table.elf_class, 64)
        self.assertEqual(table.preferred, "dynsym")
        self.assertEqual(table.exported_names(), frozenset({"exported_fn", "weak_fn"}))

    def test_symtab_only_binary_is_usable_and_says_so(self):
        path = self.tmp / "static.o"
        path.write_bytes(build_elf64([fn("only_static")], table=".symtab"))
        table = I.extract_elf_symbols(path, label="candidate")
        self.assertEqual(table.preferred, "symtab")
        self.assertEqual(table.exported_names(), frozenset({"only_static"}))
        self.assertTrue(any("no .dynsym" in n for n in table.coverage_notes))

    def test_file_sha256_is_the_real_digest(self):
        path = write_elf(self.tmp, "x.so", [fn("a")])
        table = I.extract_elf_symbols(path, label="anchor")
        self.assertEqual(table.file_sha256,
                         hashlib.sha256(path.read_bytes()).hexdigest())

    def test_non_elf_raises_rather_than_returning_empty(self):
        path = self.tmp / "not-elf"
        path.write_bytes(b"#!/bin/sh\necho hi\n" + b"\x00" * 200)
        with self.assertRaises(I.ElfFormatError):
            I.extract_elf_symbols(path, label="candidate")

    def test_truncated_elf_raises(self):
        path = self.tmp / "trunc.so"
        path.write_bytes(build_elf64([fn("a")])[:100])
        with self.assertRaises(I.ElfFormatError):
            I.extract_elf_symbols(path, label="candidate")

    def test_stripped_section_headers_raise_not_empty(self):
        blob = bytearray(build_elf64([fn("a")]))
        blob[16 + 24:16 + 32] = struct.pack("<Q", 0)   # e_shoff = 0
        path = self.tmp / "stripped.so"
        path.write_bytes(bytes(blob))
        with self.assertRaises(I.ElfFormatError) as ctx:
            I.extract_elf_symbols(path, label="candidate")
        self.assertIn("stripped", str(ctx.exception))

    def test_missing_file_raises_tree_read_error(self):
        with self.assertRaises(I.TreeReadError):
            I.extract_elf_symbols(self.tmp / "nope.so", label="anchor")

    def test_symbol_versions_are_declared_as_not_extracted(self):
        table = elf_table(self.tmp, "v.so", [fn("a")], "anchor")
        self.assertTrue(any(I.F_SYMBOL_VERSIONS_NOT_EXTRACTED in n
                            for n in table.coverage_notes))


class TestElfReaderAgainstRealBinary(unittest.TestCase):
    """Non-circular check: parse a shared library this suite did not write.

    A fixture-only test would let the writer and the reader share one wrong idea
    of ELF and agree with each other forever.
    """

    CANDIDATES = (
        "/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.34",
        "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        "/usr/lib/x86_64-linux-gnu/libc.so.6",
    )

    def real_library(self) -> Path:
        for candidate in self.CANDIDATES:
            path = Path(candidate)
            if path.is_file():
                return path
        raise AssertionError(
            "no real shared library found among "
            f"{list(self.CANDIDATES)}; this test refuses to skip, because skipping "
            "would leave the ELF reader checked only against its own fixture writer")

    def test_parses_a_real_library_and_finds_a_known_symbol(self):
        path = self.real_library()
        table = I.extract_elf_symbols(path, label="anchor")
        names = table.exported_names()
        self.assertGreater(len(names), 100, "a real libc/libstdc++ exports many symbols")
        self.assertTrue(any(n.startswith("_Z") or n in ("malloc", "memcpy")
                            for n in names))
        self.assertEqual(table.elf_class, 64)

    def test_mangled_names_from_a_real_library_parse_at_a_useful_rate(self):
        path = self.real_library()
        table = I.extract_elf_symbols(path, label="anchor")
        mangled = [n for n in table.exported_names() if n.startswith("_Z")]
        if not mangled:
            self.skipTest("selected library exports no C++ mangled names")
        parsed = [n for n in mangled if I.parse_mangled_name(n) is not None]
        # The rate is asserted, not assumed: a regression that silently stopped
        # parsing would otherwise turn every arity finding into "signature_changed".
        self.assertGreater(len(parsed) / len(mangled), 0.80)


# ---------------------------------------------------------------------------
# Itanium name parsing
# ---------------------------------------------------------------------------

class TestMangledNameParsing(unittest.TestCase):

    def test_known_manglings(self):
        cases = [
            ("_Z3fooi", "foo", 1),
            ("_Z3fooii", "foo", 2),
            ("_Z3foov", "foo", 0),
            ("_ZN3foo3barEv", "foo::bar", 0),
            ("_ZN4ggml3addEPKfS1_Pfi", "ggml::add", 4),
            ("_Z3addRKiS0_", "add", 2),
            ("_ZN4ggml6detail15kernel_dispatchILi4EEEvPKfPfi",
             "ggml::detail::kernel_dispatch", 3),
        ]
        for mangled, qualified, arity in cases:
            with self.subTest(mangled=mangled):
                parsed = I.parse_mangled_name(mangled)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.qualified, qualified)
                self.assertEqual(parsed.param_count, arity)

    def test_template_function_return_type_is_not_counted_as_a_parameter(self):
        parsed = I.parse_mangled_name("_Z3maxIiET_S0_S0_")
        self.assertTrue(parsed.templated)
        self.assertEqual(parsed.param_count, 2)

    def test_unmangled_name_is_none_not_a_guess(self):
        self.assertIsNone(I.parse_mangled_name("ggml_mul_mat"))
        self.assertIsNone(I.parse_mangled_name(""))
        self.assertIsNone(I.parse_mangled_name("_Z"))

    def test_unsupported_construct_yields_none_arity_not_zero(self):
        # A construct the type parser does not handle must leave param_count None.
        # Zero would make an arity change from 0 to 1 invisible.
        parsed = I.parse_mangled_name("_Z3fooDv4_f")   # vector type, unsupported
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.qualified, "foo")
        self.assertIsNone(parsed.param_count)

    def test_clone_suffix_is_stripped_for_parsing_but_name_identity_is_raw(self):
        parsed = I.parse_mangled_name("_Z3fooi.constprop.0")
        self.assertEqual(parsed.qualified, "foo")
        self.assertEqual(parsed.mangled, "_Z3fooi.constprop.0")


# ---------------------------------------------------------------------------
# §8.5.1 (1) — symbol preservation
# ---------------------------------------------------------------------------

class TestSymbolPreservation(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.anchor = elf_table(self.tmp, "anchor.so", ANCHOR_SYMS, "anchor")

    def cand(self, symbols, name="cand.so"):
        return elf_table(self.tmp, name, symbols, "candidate")

    def test_identical_surface_passes(self):
        gate = I.check_symbol_preservation(self.anchor, self.cand(ANCHOR_SYMS), deltas())
        self.assertEqual(gate.check.outcome, S.PASS)
        self.assertEqual(gate.gate_class, api.GATE_INTEGRITY)
        self.assertTrue(gate.requires_anchor)

    def test_undeclared_removal_fails(self):
        cand = self.cand([s for s in ANCHOR_SYMS if s[0] != "ggml_mul_mat"])
        gate = I.check_symbol_preservation(self.anchor, cand, deltas())
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_UNDECLARED_SYMBOL_REMOVAL, " ".join(gate.check.reasons))
        self.assertIn("ggml_mul_mat", " ".join(gate.check.reasons))

    def test_declared_removal_passes(self):
        cand = self.cand([s for s in ANCHOR_SYMS if s[0] != "ggml_mul_mat"])
        gate = I.check_symbol_preservation(self.anchor, cand,
                                           deltas(removed=("ggml_mul_mat",)))
        self.assertEqual(gate.check.outcome, S.PASS)

    def test_arity_change_is_labelled_and_fails_when_undeclared(self):
        cand = self.cand(
            [s for s in ANCHOR_SYMS if s[0] != "_ZN4ggml3addEPKfS1_Pfi"]
            + [fn("_ZN4ggml3addEPKfS1_Pfii")])   # one extra int parameter
        gate = I.check_symbol_preservation(self.anchor, cand, deltas())
        self.assertEqual(gate.check.outcome, S.FAIL)
        blob = " ".join(gate.check.reasons)
        self.assertIn(I.F_UNDECLARED_ARITY_CHANGE, blob)
        self.assertIn("ggml::add", blob)
        self.assertIn("4 -> 5", blob)

    def test_arity_change_declared_by_qualified_name_passes(self):
        cand = self.cand(
            [s for s in ANCHOR_SYMS if s[0] != "_ZN4ggml3addEPKfS1_Pfi"]
            + [fn("_ZN4ggml3addEPKfS1_Pfii")])
        gate = I.check_symbol_preservation(
            self.anchor, cand, deltas(arity_changed=("ggml::add",)))
        self.assertEqual(gate.check.outcome, S.PASS)

    def test_undeclared_addition_is_a_note_not_a_failure(self):
        gate = I.check_symbol_preservation(
            self.anchor, self.cand(ANCHOR_SYMS + [fn("ggml_brand_new_op")]), deltas())
        self.assertEqual(gate.check.outcome, S.PASS)
        self.assertIn(I.F_UNDECLARED_SYMBOL_ADDITION, " ".join(gate.notes))

    def test_empty_anchor_surface_is_could_not_check_never_pass(self):
        empty = elf_table(self.tmp, "empty.so", [fn("hidden", vis="HIDDEN")], "anchor")
        gate = I.check_symbol_preservation(empty, self.cand(ANCHOR_SYMS), deltas())
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn(I.F_EMPTY_ANCHOR_SYMBOL_TABLE, " ".join(gate.check.reasons))

    def test_missing_table_is_could_not_check(self):
        self.assertEqual(
            I.check_symbol_preservation(None, self.cand(ANCHOR_SYMS), deltas()).check.outcome,
            S.COULD_NOT_CHECK)
        self.assertEqual(
            I.check_symbol_preservation(self.anchor, None, deltas()).check.outcome,
            S.COULD_NOT_CHECK)

    def test_signature_change_without_derivable_arity_is_still_hard(self):
        # Both sides carry an unsupported type, so the arity is not derivable; the
        # finding degrades to the SUPERSET label and stays a failure.
        anchor = elf_table(self.tmp, "a2.so", [fn("_Z3fooDv4_f")], "anchor")
        cand = self.cand([fn("_Z3fooDv8_f")], name="c2.so")
        gate = I.check_symbol_preservation(anchor, cand, deltas())
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_UNDECLARED_SIGNATURE_CHANGE, " ".join(gate.check.reasons))


class TestSymbolArityCoverage(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_c_linkage_arity_gap_is_could_not_check_with_the_missing_input_named(self):
        anchor = elf_table(self.tmp, "a.so", [fn("ggml_mul_mat")], "anchor")
        cand = elf_table(self.tmp, "c.so", [fn("ggml_mul_mat")], "candidate")
        gate = I.check_symbol_arity_coverage(anchor, cand, None)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        blob = " ".join(gate.check.reasons)
        self.assertIn(I.F_UNMANGLED_ARITY_NOT_DERIVABLE, blob)
        self.assertIn("signature_index", blob)

    def test_no_unmangled_symbols_passes(self):
        syms = [fn("_ZN4ggml3addEPKfS1_Pfi")]
        anchor = elf_table(self.tmp, "a.so", syms, "anchor")
        cand = elf_table(self.tmp, "c.so", syms, "candidate")
        self.assertEqual(
            I.check_symbol_arity_coverage(anchor, cand, None).check.outcome, S.PASS)

    def test_signature_index_closes_the_gap_and_can_fail(self):
        anchor = elf_table(self.tmp, "a.so", [fn("ggml_mul_mat")], "anchor")
        cand = elf_table(self.tmp, "c.so", [fn("ggml_mul_mat")], "candidate")
        ok = I.check_symbol_arity_coverage(
            anchor, cand, {"ggml_mul_mat": {"anchor": 3, "candidate": 3}})
        self.assertEqual(ok.check.outcome, S.PASS)
        bad = I.check_symbol_arity_coverage(
            anchor, cand, {"ggml_mul_mat": {"anchor": 3, "candidate": 4}})
        self.assertEqual(bad.check.outcome, S.FAIL)
        self.assertIn(I.F_UNDECLARED_ARITY_CHANGE, " ".join(bad.check.reasons))

    def test_partial_signature_index_is_could_not_check(self):
        anchor = elf_table(self.tmp, "a.so", [fn("a_op"), fn("b_op")], "anchor")
        cand = elf_table(self.tmp, "c.so", [fn("a_op"), fn("b_op")], "candidate")
        gate = I.check_symbol_arity_coverage(
            anchor, cand, {"a_op": {"anchor": 1, "candidate": 1}})
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("b_op", " ".join(gate.check.reasons))


# ---------------------------------------------------------------------------
# §8.5.1 (1) — op registrations and dispatch predicates
# ---------------------------------------------------------------------------

OPS_SOURCE_ANCHOR = {
    "ggml/src/ggml-hip/ops.cpp": (
        "GGML_OP_REGISTER(GGML_OP_MUL_MAT, 2);\n"
        "GGML_OP_REGISTER(GGML_OP_MUL_MAT_ID, 3);\n"
        "GGML_OP_REGISTER(GGML_OP_SOFT_MAX, 1);\n"
    ),
}
OPS_SOURCE_CANDIDATE_DROPPED = {
    "ggml/src/ggml-hip/ops.cpp": (
        "GGML_OP_REGISTER(GGML_OP_MUL_MAT, 2);\n"
        "GGML_OP_REGISTER(GGML_OP_SOFT_MAX, 1);\n"
    ),
}
DISPATCH_SOURCE_ANCHOR = {
    "ggml/src/ggml-hip/supports.cpp": (
        "bool ggml_backend_hip_supports_op(...) { switch (op) {\n"
        "  case GGML_OP_MUL_MAT: return true;\n"
        "  case GGML_OP_MUL_MAT_ID: return true;\n"
        "  case GGML_OP_ROPE: return true;\n"
        "} }\n"
    ),
}
DISPATCH_SOURCE_CANDIDATE_DROPPED = {
    "ggml/src/ggml-hip/supports.cpp": (
        "bool ggml_backend_hip_supports_op(...) { switch (op) {\n"
        "  case GGML_OP_MUL_MAT: return true;\n"
        "  case GGML_OP_ROPE: return true;\n"
        "} }\n"
    ),
}

OP_EXTRACTOR = I.PatternRegistrationExtractor(
    kind=I.KIND_OP_REGISTRATION,
    patterns={"ggml_hip_op_table":
              r"GGML_OP_REGISTER\((?P<key>GGML_OP_[A-Z_0-9]+),\s*(?P<arity>\d+)\)"},
    declared_by="adapter:llama_gpu/v1")

DISPATCH_EXTRACTOR = I.PatternRegistrationExtractor(
    kind=I.KIND_DISPATCH_PREDICATE,
    patterns={"ggml_backend_hip_supports_op":
              r"case\s+(?P<key>GGML_OP_[A-Z_0-9]+):"},
    declared_by="adapter:llama_gpu/v1")


class TestRegistrationPreservation(unittest.TestCase):

    def test_extractor_without_patterns_is_refused(self):
        with self.assertRaises(I.EnvelopeNotDeclared):
            I.PatternRegistrationExtractor(kind=I.KIND_OP_REGISTRATION, patterns={},
                                           declared_by="adapter")

    def test_extractor_requires_a_named_key_group(self):
        with self.assertRaises(ValueError):
            I.PatternRegistrationExtractor(
                kind=I.KIND_OP_REGISTRATION, patterns={"r": r"GGML_OP_\w+"},
                declared_by="adapter")

    def test_dropped_op_registration_fails(self):
        anchor = OP_EXTRACTOR.extract_text("anchor", OPS_SOURCE_ANCHOR)
        cand = OP_EXTRACTOR.extract_text("candidate", OPS_SOURCE_CANDIDATE_DROPPED)
        gate = I.check_registration_preservation(anchor, cand, deltas())
        self.assertEqual(gate.gate_id, I.GATE_OP_REGISTRATION)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_UNDECLARED_REGISTRATION_REMOVAL, " ".join(gate.check.reasons))
        self.assertIn("GGML_OP_MUL_MAT_ID", " ".join(gate.check.reasons))

    def test_dropped_dispatch_case_fails(self):
        anchor = DISPATCH_EXTRACTOR.extract_text("anchor", DISPATCH_SOURCE_ANCHOR)
        cand = DISPATCH_EXTRACTOR.extract_text(
            "candidate", DISPATCH_SOURCE_CANDIDATE_DROPPED)
        gate = I.check_registration_preservation(anchor, cand, deltas())
        self.assertEqual(gate.gate_id, I.GATE_DISPATCH_PREDICATE)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("GGML_OP_MUL_MAT_ID", " ".join(gate.check.reasons))

    def test_declared_removal_passes(self):
        anchor = OP_EXTRACTOR.extract_text("anchor", OPS_SOURCE_ANCHOR)
        cand = OP_EXTRACTOR.extract_text("candidate", OPS_SOURCE_CANDIDATE_DROPPED)
        gate = I.check_registration_preservation(
            anchor, cand, deltas(removed=("GGML_OP_MUL_MAT_ID",)))
        self.assertEqual(gate.check.outcome, S.PASS)

    def test_registration_arity_change_fails(self):
        cand_src = {"ggml/src/ggml-hip/ops.cpp":
                    OPS_SOURCE_ANCHOR["ggml/src/ggml-hip/ops.cpp"].replace(
                        "GGML_OP_MUL_MAT, 2", "GGML_OP_MUL_MAT, 4")}
        anchor = OP_EXTRACTOR.extract_text("anchor", OPS_SOURCE_ANCHOR)
        cand = OP_EXTRACTOR.extract_text("candidate", cand_src)
        gate = I.check_registration_preservation(anchor, cand, deltas())
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_UNDECLARED_REGISTRATION_ARITY_CHANGE,
                      " ".join(gate.check.reasons))

    def test_empty_anchor_table_is_could_not_check_never_pass(self):
        anchor = OP_EXTRACTOR.extract_text("anchor", {"x.cpp": "// nothing here\n"})
        cand = OP_EXTRACTOR.extract_text("candidate", OPS_SOURCE_ANCHOR)
        gate = I.check_registration_preservation(anchor, cand, deltas())
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn(I.F_EMPTY_ANCHOR_REGISTRATION_TABLE, " ".join(gate.check.reasons))

    def test_both_tables_absent_needs_an_expected_kind_to_name_the_gate(self):
        with self.assertRaises(ValueError):
            I.check_registration_preservation(None, None, deltas())
        gate = I.check_registration_preservation(
            None, None, deltas(), expected_kind=I.KIND_DISPATCH_PREDICATE)
        self.assertEqual(gate.gate_id, I.GATE_DISPATCH_PREDICATE)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)

    def test_expected_kind_must_match_the_supplied_table(self):
        anchor = OP_EXTRACTOR.extract_text("anchor", OPS_SOURCE_ANCHOR)
        with self.assertRaises(ValueError):
            I.check_registration_preservation(
                anchor, None, deltas(), expected_kind=I.KIND_DISPATCH_PREDICATE)

    def test_diffing_two_different_kinds_raises(self):
        a = OP_EXTRACTOR.extract_text("anchor", OPS_SOURCE_ANCHOR)
        d = DISPATCH_EXTRACTOR.extract_text("candidate", DISPATCH_SOURCE_ANCHOR)
        with self.assertRaises(ValueError):
            I.diff_registration_tables(a, d)

    def test_extract_tree_reads_real_files(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "ggml").mkdir()
            (root / "ggml" / "ops.cpp").write_text(
                OPS_SOURCE_ANCHOR["ggml/src/ggml-hip/ops.cpp"], encoding="utf-8")
            (root / "notes.txt").write_text("GGML_OP_REGISTER(GGML_OP_FAKE, 9)",
                                            encoding="utf-8")
            table = OP_EXTRACTOR.extract_tree("anchor", root, suffixes=(".cpp",))
        self.assertEqual({e.key for e in table.entries},
                         {"GGML_OP_MUL_MAT", "GGML_OP_MUL_MAT_ID", "GGML_OP_SOFT_MAX"})

    def test_extract_tree_refuses_an_empty_suffix_list(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(ValueError):
                OP_EXTRACTOR.extract_tree("anchor", raw, suffixes=())


# ---------------------------------------------------------------------------
# Declared deltas / declared surface
# ---------------------------------------------------------------------------

class TestDeclarations(unittest.TestCase):

    def test_absent_declared_symbol_deltas_raises_rather_than_defaulting_empty(self):
        with self.assertRaises(I.DeclarationMissing):
            I.DeclaredSymbolDeltas.from_proposal({"change_class": "dispatcher"})

    def test_partially_declared_deltas_raise(self):
        with self.assertRaises(I.DeclarationMissing):
            I.DeclaredSymbolDeltas.from_proposal(
                {"declared_symbol_deltas": {"added": [], "removed": []}})

    def test_full_declaration_parses(self):
        d = I.DeclaredSymbolDeltas.from_proposal({"declared_symbol_deltas": {
            "added": ["a"], "removed": ["b"], "arity_changed": ["c"]}})
        self.assertEqual(d.removed, frozenset({"b"}))

    def test_declared_surface_split_rule(self):
        surface = I.DeclaredSurface.from_proposal({"change": {"files_and_symbols": [
            "ggml/src/ggml-hip/mmq.cu::launch_mmq",
            "ggml/src/ggml-hip/mmq.cuh",
            "ggml_mul_mat",
        ]}})
        self.assertEqual(surface.files, frozenset({
            "ggml/src/ggml-hip/mmq.cu", "ggml/src/ggml-hip/mmq.cuh"}))
        self.assertEqual(surface.symbols, frozenset({"launch_mmq", "ggml_mul_mat"}))

    def test_absent_declared_surface_raises(self):
        with self.assertRaises(I.DeclarationMissing):
            I.DeclaredSurface.from_proposal({"change": {}})

    def test_empty_declared_surface_is_legal_and_self_punishing(self):
        surface = I.DeclaredSurface.from_proposal({"change": {"files_and_symbols": []}})
        diff = I.parse_unified_diff(SMALL_DIFF)
        gate = I.check_semantic_diff_conformance(
            diff, surface, envelope(), original_line_counts={"a.cpp": 100})
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_UNDECLARED_FILE_TOUCHED, " ".join(gate.check.reasons))


# ---------------------------------------------------------------------------
# §8.5.1 (2) — clean build from the recorded snapshot
# ---------------------------------------------------------------------------

PROD_TREES = ("/mnt/raid0/llm/llama.cpp", "/mnt/raid0/llm/whisper.cpp")


def provenance(**overrides) -> I.BuildProvenance:
    kwargs = dict(
        candidate_id="akc-0001",
        snapshot_sha256=sha("snapshot"),
        source_root="/mnt/raid0/llm/ak/campaigns/ak-1/snapshots/akc-0001",
        build_dir="/mnt/raid0/llm/ak/campaigns/ak-1/build/akc-0001",
        build_dir_created_for_this_build=True,
        build_dir_pre_build_digest=I.EMPTY_TREE_SHA256,
        actor_worktree="/mnt/raid0/llm/ak/worktrees/akp-0001",
        production_tree_paths=PROD_TREES,
        toolchain="rocm-6.2",
        compiler="hipcc 6.2.0",
        command="cmake --build . -j 32",
        build_log_path="/mnt/raid0/llm/ak/campaigns/ak-1/build/akc-0001/build.log",
        build_log_sha256=sha("build-log"),
        output_binary_sha256=sha("clean-binary"),
        incremental_output_binary_sha256=None,
    )
    kwargs.update(overrides)
    return I.BuildProvenance(**kwargs)


class TestCleanBuildFromSnapshot(unittest.TestCase):

    def test_clean_build_passes_with_a_recomputed_snapshot(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "src").mkdir()
            (root / "src" / "a.cpp").write_text("int main(){}\n", encoding="utf-8")
            digest = I.hash_source_tree(root).sha256
            gate, receipt = I.check_clean_build_from_snapshot(
                provenance(snapshot_sha256=digest), sha("clean-binary"),
                recompute_root=root, snapshot_attested_by=None)
        self.assertEqual(gate.check.outcome, S.PASS)
        self.assertEqual(receipt.snapshot_verification, "recomputed")
        self.assertTrue(receipt.content_hash)

    def test_snapshot_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "a.cpp").write_text("int main(){}\n", encoding="utf-8")
            gate, receipt = I.check_clean_build_from_snapshot(
                provenance(), sha("clean-binary"),
                recompute_root=root, snapshot_attested_by=None)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_SNAPSHOT_DIGEST_MISMATCH, " ".join(gate.check.reasons))

    def test_unverified_snapshot_is_could_not_check_not_pass(self):
        gate, receipt = I.check_clean_build_from_snapshot(
            provenance(), sha("clean-binary"),
            recompute_root=None, snapshot_attested_by=None)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn(I.F_SNAPSHOT_NOT_VERIFIED, " ".join(gate.check.reasons))
        self.assertEqual(receipt.snapshot_verification, "unverified")

    def test_attested_snapshot_passes_and_says_it_was_attested(self):
        gate, receipt = I.check_clean_build_from_snapshot(
            provenance(), sha("clean-binary"),
            recompute_root=None, snapshot_attested_by="storage.verify_durability:ak-1")
        self.assertEqual(gate.check.outcome, S.PASS)
        self.assertEqual(receipt.snapshot_verification, "attested")

    def test_artifact_that_is_not_the_clean_output_fails(self):
        gate, _ = I.check_clean_build_from_snapshot(
            provenance(), sha("some-other-binary"),
            recompute_root=None, snapshot_attested_by="attester")
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_ARTIFACT_NOT_FROM_CLEAN_BUILD, " ".join(gate.check.reasons))

    def test_artifact_taken_from_the_actors_incremental_tree_fails(self):
        """The red-team case: the incremental tree compiles, the snapshot does not."""
        gate, _ = I.check_clean_build_from_snapshot(
            provenance(incremental_output_binary_sha256=sha("incremental-binary")),
            sha("incremental-binary"),
            recompute_root=None, snapshot_attested_by="attester")
        self.assertEqual(gate.check.outcome, S.FAIL)
        blob = " ".join(gate.check.reasons)
        self.assertIn(I.F_ARTIFACT_FROM_INCREMENTAL_TREE, blob)

    def test_non_fresh_build_dir_fails(self):
        gate, _ = I.check_clean_build_from_snapshot(
            provenance(build_dir_created_for_this_build=False,
                       build_dir_pre_build_digest=sha("stale-objects")),
            sha("clean-binary"), recompute_root=None, snapshot_attested_by="attester")
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_BUILD_DIR_NOT_FRESH, " ".join(gate.check.reasons))

    def test_build_dir_inside_the_actor_worktree_fails(self):
        gate, _ = I.check_clean_build_from_snapshot(
            provenance(build_dir="/mnt/raid0/llm/ak/worktrees/akp-0001/build"),
            sha("clean-binary"), recompute_root=None, snapshot_attested_by="attester")
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_BUILD_DIR_INSIDE_ACTOR_WORKTREE, " ".join(gate.check.reasons))

    def test_building_in_a_production_tree_fails(self):
        gate, _ = I.check_clean_build_from_snapshot(
            provenance(build_dir="/mnt/raid0/llm/llama.cpp/build-hip"),
            sha("clean-binary"), recompute_root=None, snapshot_attested_by="attester")
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_BUILD_IN_PRODUCTION_TREE, " ".join(gate.check.reasons))

    def test_source_root_in_a_production_tree_also_fails(self):
        gate, _ = I.check_clean_build_from_snapshot(
            provenance(source_root="/mnt/raid0/llm/llama.cpp"),
            sha("clean-binary"), recompute_root=None, snapshot_attested_by="attester")
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_BUILD_IN_PRODUCTION_TREE, " ".join(gate.check.reasons))

    def test_provenance_refuses_a_relative_path(self):
        with self.assertRaises(ValueError):
            provenance(build_dir="build/akc-0001")

    def test_receipt_is_canonical_json_able(self):
        _, receipt = I.check_clean_build_from_snapshot(
            provenance(), sha("clean-binary"),
            recompute_root=None, snapshot_attested_by="attester")
        S.canonical_json(receipt.to_dict())
        self.assertEqual(receipt.content_hash, S.content_hash(receipt.to_dict()))


class TestTreeHashing(unittest.TestCase):

    def test_empty_tree_digest_constant(self):
        with tempfile.TemporaryDirectory() as raw:
            self.assertEqual(I.hash_source_tree(raw).sha256, I.EMPTY_TREE_SHA256)

    def test_digest_is_content_addressed_and_path_sensitive(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            for root in (Path(a), Path(b)):
                (root / "s").mkdir()
                (root / "s" / "x.cpp").write_text("int x;\n", encoding="utf-8")
            self.assertEqual(I.hash_source_tree(a).sha256, I.hash_source_tree(b).sha256)
            (Path(b) / "s" / "x.cpp").write_text("int y;\n", encoding="utf-8")
            self.assertNotEqual(I.hash_source_tree(a).sha256,
                                I.hash_source_tree(b).sha256)

    def test_executable_bit_changes_the_digest(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            script = root / "build.sh"
            script.write_text("#!/bin/bash\n", encoding="utf-8")
            before = I.hash_source_tree(root).sha256
            script.chmod(0o755)
            self.assertNotEqual(before, I.hash_source_tree(root).sha256)

    def test_symlink_is_recorded_not_followed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "real.h").write_text("#pragma once\n", encoding="utf-8")
            (root / "link.h").symlink_to("real.h")
            digest = I.hash_source_tree(root)
            modes = {e[2]: e[0] for e in digest.entries}
            self.assertEqual(modes["link.h"], "120000")

    def test_unreadable_file_raises_rather_than_being_skipped(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            secret = root / "secret.h"
            secret.write_text("x\n", encoding="utf-8")
            secret.chmod(0o000)
            try:
                readable = True
                try:
                    with secret.open("rb") as handle:
                        handle.read(1)
                except PermissionError:
                    readable = False
                # Asserted, not assumed: if this process could read a mode-000 file
                # (running as root) the test below would prove nothing, so say so
                # rather than passing vacuously.
                self.assertFalse(
                    readable,
                    "this process can read a mode-000 file, so the unreadable-input "
                    "path cannot be exercised here")
                with self.assertRaises(I.TreeReadError):
                    I.hash_source_tree(root)
            finally:
                secret.chmod(0o644)

    def test_exclusions_default_to_excluding_nothing(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / ".git").mkdir()
            (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
            everything = I.hash_source_tree(root)
            without = I.hash_source_tree(root, exclude_dir_names=(".git",))
            self.assertEqual(everything.file_count, 1)
            self.assertEqual(without.file_count, 0)

    def test_hashing_a_non_directory_raises(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "f.txt"
            path.write_text("x", encoding="utf-8")
            with self.assertRaises(I.TreeReadError):
                I.hash_source_tree(path)

    def test_sha256_file_max_bytes_raises_instead_of_truncating(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "big.bin"
            path.write_bytes(b"x" * 4096)
            with self.assertRaises(I.TreeReadError):
                I.sha256_file(path, max_bytes=10)


# ---------------------------------------------------------------------------
# §8.5.1 (3) — diff parsing and semantic conformance
# ---------------------------------------------------------------------------

SMALL_DIFF = """diff --git a/a.cpp b/a.cpp
index 1111111..2222222 100644
--- a/a.cpp
+++ b/a.cpp
@@ -10,4 +10,5 @@ void f() {
     int a = 1;
-    int b = 2;
+    int b = 3;
+    int c = 4;
     return;
 }
"""

# The escalation.py shape: a file gutted from 454 lines to 3, in one hunk.
GUTTING_DIFF_LINES = (
    ["diff --git a/src/escalation.cpp b/src/escalation.cpp",
     "index aaaa..bbbb 100644",
     "--- a/src/escalation.cpp",
     "+++ b/src/escalation.cpp",
     "@@ -1,454 +1,3 @@"]
    + [f"-line {i}" for i in range(1, 455)]
    + ["+int main(){}", "+// gutted", "+// by an edit that compiled"]
)
GUTTING_DIFF = "\n".join(GUTTING_DIFF_LINES) + "\n"

NEW_FILE_DIFF = """diff --git a/new.cu b/new.cu
new file mode 100644
--- /dev/null
+++ b/new.cu
@@ -0,0 +1,2 @@
+__global__ void k() {}
+// new
"""

DELETE_FILE_DIFF = """diff --git a/old.cu b/old.cu
deleted file mode 100644
--- a/old.cu
+++ /dev/null
@@ -1,2 +0,0 @@
-__global__ void k() {}
-// old
"""

BINARY_DIFF = """diff --git a/blob.bin b/blob.bin
index 1111111..2222222 100644
Binary files a/blob.bin and b/blob.bin differ
"""

CORE_HEADER_DIFF = """diff --git a/ggml/include/ggml.h b/ggml/include/ggml.h
index 1111111..2222222 100644
--- a/ggml/include/ggml.h
+++ b/ggml/include/ggml.h
@@ -100,3 +100,4 @@ struct ggml_tensor {
     int ne[4];
+    int pad;
     int nb[4];
 };
"""


def envelope(**overrides) -> I.ChangeClassEnvelope:
    kwargs = dict(change_class="dispatcher", max_files_touched=3,
                  max_changed_lines=200, max_hunks=10,
                  max_file_shrinkage_ratio=0.60,
                  allows_file_creation=False, allows_file_deletion=False,
                  allows_pure_deletion_hunks=False,
                  declared_by="adapter:llama_gpu/v1")
    kwargs.update(overrides)
    return I.ChangeClassEnvelope(**kwargs)


def ceiling(**overrides) -> I.ComplexityCeiling:
    kwargs = dict(backend="llama_gpu", max_diff_lines=150, max_files_touched=4,
                  shared_core_modification_requires_review=True,
                  declared_by="adapter:llama_gpu/v1")
    kwargs.update(overrides)
    return I.ComplexityCeiling(**kwargs)


def surface(files=("a.cpp",), symbols=()) -> I.DeclaredSurface:
    return I.DeclaredSurface(files=frozenset(files), symbols=frozenset(symbols))


class TestDiffParsing(unittest.TestCase):

    def test_parses_a_git_diff(self):
        diff = I.parse_unified_diff(SMALL_DIFF)
        self.assertEqual(diff.files_touched, 1)
        f = diff.files[0]
        self.assertEqual(f.path, "a.cpp")
        self.assertEqual((f.added_lines, f.removed_lines, f.hunks), (2, 1, 1))
        self.assertEqual(f.observed_old_extent, 13)

    def test_new_and_deleted_files(self):
        new = I.parse_unified_diff(NEW_FILE_DIFF).files[0]
        self.assertTrue(new.is_new_file)
        self.assertEqual(new.path, "new.cu")
        gone = I.parse_unified_diff(DELETE_FILE_DIFF).files[0]
        self.assertTrue(gone.is_deleted_file)
        self.assertEqual(gone.path, "old.cu")

    def test_binary_file_is_recorded_as_binary(self):
        f = I.parse_unified_diff(BINARY_DIFF).files[0]
        self.assertTrue(f.is_binary)
        self.assertEqual(f.path, "blob.bin")

    def test_hunk_body_that_contradicts_its_header_raises(self):
        bad = SMALL_DIFF.replace("@@ -10,4 +10,5 @@", "@@ -10,9 +10,5 @@")
        with self.assertRaises(I.DiffParseError):
            I.parse_unified_diff(bad)

    def test_garbage_inside_a_hunk_raises(self):
        bad = SMALL_DIFF.replace("     return;", "!!! corrupted")
        with self.assertRaises(I.DiffParseError):
            I.parse_unified_diff(bad)

    def test_empty_diff_is_an_empty_source_diff(self):
        diff = I.parse_unified_diff("")
        self.assertEqual(diff.files_touched, 0)
        self.assertEqual(diff.total_changed, 0)

    def test_gutting_diff_counts(self):
        diff = I.parse_unified_diff(GUTTING_DIFF)
        f = diff.files[0]
        self.assertEqual(f.removed_lines, 454)
        self.assertEqual(f.added_lines, 3)
        self.assertEqual(f.observed_old_extent, 454)


class TestSemanticDiffConformance(unittest.TestCase):

    def test_declared_small_change_passes(self):
        gate = I.check_semantic_diff_conformance(
            I.parse_unified_diff(SMALL_DIFF), surface(), envelope(),
            original_line_counts={"a.cpp": 400})
        self.assertEqual(gate.check.outcome, S.PASS)

    def test_undeclared_file_fails(self):
        gate = I.check_semantic_diff_conformance(
            I.parse_unified_diff(SMALL_DIFF), surface(files=("b.cpp",)), envelope(),
            original_line_counts={"a.cpp": 400})
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_UNDECLARED_FILE_TOUCHED, " ".join(gate.check.reasons))

    def test_escalation_shape_fails_when_the_file_length_is_known(self):
        gate = I.check_semantic_diff_conformance(
            I.parse_unified_diff(GUTTING_DIFF),
            surface(files=("src/escalation.cpp",)),
            envelope(max_changed_lines=1000, max_hunks=20),
            original_line_counts={"src/escalation.cpp": 454})
        self.assertEqual(gate.check.outcome, S.FAIL)
        blob = " ".join(gate.check.reasons)
        self.assertIn(I.F_EXCESSIVE_SHRINKAGE, blob)
        self.assertIn("454", blob)

    def test_escalation_shape_is_could_not_check_without_the_file_length(self):
        gate = I.check_semantic_diff_conformance(
            I.parse_unified_diff(GUTTING_DIFF),
            surface(files=("src/escalation.cpp",)),
            envelope(max_changed_lines=1000, max_hunks=20),
            original_line_counts=None)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        blob = " ".join(gate.check.reasons)
        self.assertIn(I.F_SHRINKAGE_NOT_DERIVABLE, blob)
        self.assertIn("original_line_counts", blob)

    def test_small_shrinkage_is_conclusive_from_the_hunk_bound_alone(self):
        gate = I.check_semantic_diff_conformance(
            I.parse_unified_diff(SMALL_DIFF), surface(), envelope(),
            original_line_counts=None)
        # removed=1, extent=13 -> upper bound 7.7% < 60%: conclusive without counts.
        self.assertEqual(gate.check.outcome, S.PASS)

    def test_file_creation_and_deletion_respect_the_envelope(self):
        created = I.check_semantic_diff_conformance(
            I.parse_unified_diff(NEW_FILE_DIFF), surface(files=("new.cu",)), envelope(),
            original_line_counts=None)
        self.assertEqual(created.check.outcome, S.FAIL)
        self.assertIn(I.F_UNDECLARED_FILE_CREATED, " ".join(created.check.reasons))

        allowed = I.check_semantic_diff_conformance(
            I.parse_unified_diff(NEW_FILE_DIFF), surface(files=("new.cu",)),
            envelope(allows_file_creation=True), original_line_counts=None)
        self.assertEqual(allowed.check.outcome, S.PASS)

        deleted = I.check_semantic_diff_conformance(
            I.parse_unified_diff(DELETE_FILE_DIFF), surface(files=("old.cu",)),
            envelope(), original_line_counts=None)
        self.assertEqual(deleted.check.outcome, S.FAIL)
        self.assertIn(I.F_UNDECLARED_FILE_DELETED, " ".join(deleted.check.reasons))

    def test_pure_deletion_hunk_fails_by_default(self):
        pure = """diff --git a/a.cpp b/a.cpp
--- a/a.cpp
+++ b/a.cpp
@@ -5,3 +5,1 @@
 keep
-drop one
-drop two
"""
        gate = I.check_semantic_diff_conformance(
            I.parse_unified_diff(pure), surface(), envelope(),
            original_line_counts={"a.cpp": 500})
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_PURE_DELETION_HUNK, " ".join(gate.check.reasons))

    def test_binary_file_is_could_not_check(self):
        gate = I.check_semantic_diff_conformance(
            I.parse_unified_diff(BINARY_DIFF), surface(files=("blob.bin",)), envelope(),
            original_line_counts=None)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn(I.F_BINARY_FILE_IN_DIFF, " ".join(gate.check.reasons))

    def test_envelope_ceilings_fail(self):
        gate = I.check_semantic_diff_conformance(
            I.parse_unified_diff(GUTTING_DIFF),
            surface(files=("src/escalation.cpp",)),
            envelope(max_changed_lines=10, max_hunks=1),
            original_line_counts={"src/escalation.cpp": 100000})
        blob = " ".join(gate.check.reasons)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_ENVELOPE_LINES_EXCEEDED, blob)

    def test_envelope_is_never_defaulted(self):
        with self.assertRaises(I.EnvelopeNotDeclared):
            I.envelope_for({"dispatcher": envelope()}, "core_header")
        self.assertIs(I.envelope_for({"dispatcher": envelope()}, "dispatcher").__class__,
                      I.ChangeClassEnvelope)

    def test_envelope_rejects_an_impossible_shrinkage_ratio(self):
        for bad in (0.0, -0.1, 1.5):
            with self.subTest(ratio=bad):
                with self.assertRaises(ValueError):
                    envelope(max_file_shrinkage_ratio=bad)


class TestComplexityCeiling(unittest.TestCase):

    def test_within_ceiling_needs_no_review(self):
        assessment = I.assess_complexity_ceiling(
            I.parse_unified_diff(SMALL_DIFF), ceiling(),
            touches_shared_core=False, change_class="dispatcher")
        self.assertFalse(assessment.requires_human_code_review)
        self.assertIsNone(assessment.first_page_notice)

    def test_large_diff_is_marked_for_review_not_failed(self):
        assessment = I.assess_complexity_ceiling(
            I.parse_unified_diff(GUTTING_DIFF), ceiling(),
            touches_shared_core=False, change_class="dispatcher")
        self.assertTrue(assessment.requires_human_code_review)
        self.assertTrue(assessment.first_page_notice.startswith(
            I.REQUIRES_HUMAN_CODE_REVIEW))

    def test_shared_core_forces_review_regardless_of_size(self):
        assessment = I.assess_complexity_ceiling(
            I.parse_unified_diff(SMALL_DIFF), ceiling(),
            touches_shared_core=True, change_class="dispatcher")
        self.assertTrue(assessment.requires_human_code_review)

    def test_core_header_change_class_forces_review(self):
        assessment = I.assess_complexity_ceiling(
            I.parse_unified_diff(SMALL_DIFF), ceiling(),
            touches_shared_core=False, change_class="core_header")
        self.assertTrue(assessment.requires_human_code_review)


# ---------------------------------------------------------------------------
# §8.5.1 — core_header risk tier
# ---------------------------------------------------------------------------

def core_policy(**overrides) -> I.CoreHeaderPolicy:
    kwargs = dict(core_path_prefixes=("ggml/include", "ggml/src/ggml.c"),
                  core_path_globs=("ggml/src/*.h",),
                  backends_served=("llama_cpu", "llama_gpu"),
                  declared_by="adapter:llama/v1")
    kwargs.update(overrides)
    return I.CoreHeaderPolicy(**kwargs)


class TestCoreHeaderRiskTier(unittest.TestCase):

    def test_standard_change_stays_standard(self):
        decision, gate = I.assess_risk_tier(
            "dispatcher", I.parse_unified_diff(SMALL_DIFF), core_policy(),
            declared_surface_scope=I.SURFACE_PARTIAL)
        self.assertEqual(decision.tier, "standard")
        self.assertFalse(decision.requires_human_code_review)
        self.assertEqual(gate.check.outcome, S.PASS)

    def test_core_path_forces_the_tier_even_when_declared_otherwise(self):
        decision, gate = I.assess_risk_tier(
            "dispatcher", I.parse_unified_diff(CORE_HEADER_DIFF), core_policy(),
            declared_surface_scope=I.SURFACE_FULL_TREE)
        self.assertEqual(decision.tier, "core_header")
        self.assertTrue(decision.misdeclared)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_MISDECLARED_CORE_HEADER_CHANGE, " ".join(gate.check.reasons))

    def test_core_header_forces_full_tree_surface_and_per_backend_comparison(self):
        decision, gate = I.assess_risk_tier(
            "core_header", I.parse_unified_diff(CORE_HEADER_DIFF), core_policy(),
            declared_surface_scope=I.SURFACE_FULL_TREE)
        self.assertTrue(decision.full_tree_surface_required)
        self.assertEqual(decision.per_backend_binary_comparison_required,
                         ("llama_cpu", "llama_gpu"))
        self.assertTrue(decision.requires_human_code_review)
        self.assertEqual(gate.check.outcome, S.PASS)

    def test_core_header_with_a_partial_declared_surface_fails(self):
        _, gate = I.assess_risk_tier(
            "core_header", I.parse_unified_diff(CORE_HEADER_DIFF), core_policy(),
            declared_surface_scope=I.SURFACE_PARTIAL)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_CORE_HEADER_SURFACE_UNDER_DECLARED,
                      " ".join(gate.check.reasons))

    def test_core_header_class_alone_forces_the_tier_with_no_core_path_touched(self):
        decision, _ = I.assess_risk_tier(
            "core_header", I.parse_unified_diff(SMALL_DIFF), core_policy(),
            declared_surface_scope=I.SURFACE_FULL_TREE)
        self.assertEqual(decision.tier, "core_header")
        self.assertTrue(decision.requires_human_code_review)

    def test_policy_matching_nothing_is_refused(self):
        with self.assertRaises(I.EnvelopeNotDeclared):
            core_policy(core_path_prefixes=(), core_path_globs=())

    def test_policy_needs_the_backends_the_tree_serves(self):
        with self.assertRaises(ValueError):
            core_policy(backends_served=())

    def test_glob_matching(self):
        policy = core_policy()
        self.assertTrue(policy.matches("ggml/include/ggml.h"))
        self.assertTrue(policy.matches("ggml/src/ggml-impl.h"))
        self.assertFalse(policy.matches("ggml/src/ggml-hip/mmq.cu"))


# ---------------------------------------------------------------------------
# §8.5.1 (4) — repair from a clean parent, capped
# ---------------------------------------------------------------------------

def attempt(**overrides) -> I.RepairAttempt:
    kwargs = dict(proposal_id="akp-0001", attempt_index=1,
                  parent_candidate_id="akc-parent",
                  parent_snapshot_sha256=sha("parent-snapshot"),
                  base_tree_sha256=sha("parent-snapshot"),
                  failed_attempt_tree_sha256=sha("failed-tree"),
                  checked_out_fresh=True,
                  worktree_path="/mnt/raid0/llm/ak/worktrees/akp-0001-r1")
    kwargs.update(overrides)
    return I.RepairAttempt(**kwargs)


class TestRepairFromCleanParent(unittest.TestCase):

    def test_non_repair_candidate_passes(self):
        gate = I.check_repair_from_clean_parent(None)
        self.assertEqual(gate.check.outcome, S.PASS)

    def test_clean_reapply_passes(self):
        self.assertEqual(
            I.check_repair_from_clean_parent(attempt()).check.outcome, S.PASS)

    def test_continuing_on_the_failed_tree_fails(self):
        gate = I.check_repair_from_clean_parent(
            attempt(base_tree_sha256=sha("failed-tree")))
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_REPAIR_CONTINUED_ON_FAILED_TREE, " ".join(gate.check.reasons))

    def test_base_that_is_neither_parent_nor_failed_tree_fails(self):
        gate = I.check_repair_from_clean_parent(
            attempt(base_tree_sha256=sha("some-third-tree")))
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_REPAIR_BASE_NOT_PARENT_SNAPSHOT, " ".join(gate.check.reasons))

    def test_not_rechecked_out_fails(self):
        gate = I.check_repair_from_clean_parent(attempt(checked_out_fresh=False))
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_REPAIR_NOT_RECHECKED_OUT, " ".join(gate.check.reasons))

    def test_ledger_grants_up_to_the_cap_then_signals_planner_degraded(self):
        ledger = I.RepairLedger(
            proposal_id="akp-0001",
            policy=I.RepairPolicy(max_repairs_per_proposal=2, declared_by="campaign"),
            parent_candidate_id="akc-parent",
            parent_snapshot_sha256=sha("parent-snapshot"), used=0)
        d1, ledger = ledger.request()
        d2, ledger = ledger.request()
        d3, ledger_after = ledger.request()
        self.assertTrue(d1.granted and d2.granted)
        self.assertEqual((d1.attempt_index, d2.attempt_index), (1, 2))
        self.assertFalse(d3.granted)
        self.assertEqual(d3.signal, I.PLANNER_DEGRADED)
        self.assertIn(I.F_REPAIR_CAP_EXCEEDED, d3.reason)
        self.assertIs(ledger_after, ledger)   # the ledger does not advance on refusal

    def test_zero_cap_refuses_immediately(self):
        ledger = I.RepairLedger(
            proposal_id="akp-0001",
            policy=I.RepairPolicy(max_repairs_per_proposal=0, declared_by="campaign"),
            parent_candidate_id="akc-parent",
            parent_snapshot_sha256=sha("parent-snapshot"), used=0)
        decision, _ = ledger.request()
        self.assertFalse(decision.granted)
        self.assertEqual(decision.signal, I.PLANNER_DEGRADED)

    def test_grant_names_the_parent_to_recheck_out(self):
        ledger = I.RepairLedger(
            proposal_id="akp-0001",
            policy=I.RepairPolicy(max_repairs_per_proposal=1, declared_by="campaign"),
            parent_candidate_id="akc-parent",
            parent_snapshot_sha256=sha("parent-snapshot"), used=0)
        decision, _ = ledger.request()
        self.assertIn("akc-parent", decision.reason)
        self.assertIn("never continue on the failed attempt's tree", decision.reason)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

class _Behavioural:
    """A behavioural runner stub. It runs nothing and counts its invocations."""

    def __init__(self, tier="T0"):
        self.tier = tier
        self.calls = 0

    def run_gates(self, request):
        self.calls += 1
        return (
            api.GateResult("mul_mat_exact_shapes", api.GATE_CORRECTNESS, PASS,
                           requires_anchor=True),
            api.GateResult("bitwise_same_seed", api.GATE_DETERMINISM, PASS,
                           requires_anchor=True),
        )


class IntegrityFixture:
    """Builds a complete, PASSING `SourceIntegrityInputs`, then lets one thing break.

    Every field is spelled out. There is deliberately no `all_clear()` helper in
    `integrity` for the same reason `api` has none: a fixture that fabricates PASS
    is the fixture that removes the signal under test.
    """

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.anchor_syms = elf_table(tmp, "anchor.so", ANCHOR_SYMS, "anchor")
        self.cand_syms = elf_table(tmp, "cand.so", ANCHOR_SYMS, "candidate")
        self.anchor_ops = OP_EXTRACTOR.extract_text("anchor", OPS_SOURCE_ANCHOR)
        self.cand_ops = OP_EXTRACTOR.extract_text("candidate", OPS_SOURCE_ANCHOR)
        self.anchor_disp = DISPATCH_EXTRACTOR.extract_text(
            "anchor", DISPATCH_SOURCE_ANCHOR)
        self.cand_disp = DISPATCH_EXTRACTOR.extract_text(
            "candidate", DISPATCH_SOURCE_ANCHOR)

    def inputs(self, **overrides) -> I.SourceIntegrityInputs:
        kwargs = dict(
            candidate_id="akc-0001",
            backend="llama_gpu",
            change_class="dispatcher",
            # The artifact under test IS the binary the candidate symbol table
            # was read out of. Anything else and `check_evidence_binding` has
            # nothing to bind the ABI diff to.
            artifact_binary_sha256=self.cand_syms.file_sha256,
            anchor_symbols=self.anchor_syms,
            candidate_symbols=self.cand_syms,
            signature_index={
                "ggml_mul_mat": {"anchor": 3, "candidate": 3},
                "ggml_backend_hip_supports_op": {"anchor": 2, "candidate": 2},
            },
            anchor_registrations=(self.anchor_ops, self.anchor_disp),
            candidate_registrations=(self.cand_ops, self.cand_disp),
            declared_symbol_deltas=deltas(),
            declared_surface=surface(),
            declared_surface_scope=I.SURFACE_PARTIAL,
            diff=I.parse_unified_diff(SMALL_DIFF),
            envelope=envelope(),
            complexity_ceiling=ceiling(),
            core_header_policy=core_policy(),
            original_line_counts={"a.cpp": 400},
            build=provenance(output_binary_sha256=self.cand_syms.file_sha256),
            snapshot_recompute_root=None,
            snapshot_attested_by="storage.verify_durability:ak-1",
            repair=None,
        )
        kwargs.update(overrides)
        return I.SourceIntegrityInputs(**kwargs)

    # -- the request/window pair these inputs are actually EVIDENCE FOR ------
    #
    # Before `integrity.check_evidence_binding` existed, every runner test used
    # the module-level `request()` whose anchor binary SHA-256 is `sha(
    # "anchor-binary")` — a hash belonging to no file the fixture ever wrote,
    # while `anchor_syms` was extracted from `anchor.so`. The gates compared a
    # symbol table against a binary the request had never named, and the
    # dispatcher answered PASS. These helpers make the fixture state what it is
    # evidence for, so a red-team test isolates the defect it means to inject.

    def anchor_id(self, inputs=None) -> api.AnchorIdentity:
        table = (inputs or self.inputs()).anchor_symbols
        return anchor_identity(binary_sha256=table.file_sha256)

    def bound_request(self, inputs=None, **overrides) -> api.EvaluationRequest:
        inputs = inputs or self.inputs()
        kwargs = dict(
            artifact=api.ArtifactIdentity(
                source_sha256=sha("cand-source"),
                binary_sha256=inputs.artifact_binary_sha256,
                linkage_sha256=sha("cand-linkage")),
            anchor=self.anchor_id(inputs))
        kwargs.update(overrides)
        return request(**kwargs)

    def bound_window(self, inputs=None, **overrides) -> api.WindowAttestations:
        identity = self.anchor_id(inputs)
        kwargs = dict(anchor_at_open=identity, anchor_at_close=identity)
        kwargs.update(overrides)
        return window(**kwargs)


class TestOrchestration(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.fx = IntegrityFixture(self.tmp)

    def test_clean_candidate_passes_every_gate(self):
        report = I.run_source_integrity_gates(self.fx.inputs())
        self.assertEqual(set(outcomes(report.gates).values()), {S.PASS})
        self.assertFalse(report.blocking)
        self.assertFalse(report.requires_human_code_review)
        self.assertIsNone(report.first_page_notice)

    def test_all_eight_gates_are_emitted_in_order(self):
        report = I.run_source_integrity_gates(self.fx.inputs())
        self.assertEqual([g.gate_id for g in report.gates], list(I.GATE_IDS))

    def test_every_gate_is_the_integrity_class(self):
        report = I.run_source_integrity_gates(self.fx.inputs())
        self.assertEqual({g.gate_class for g in report.gates}, {api.GATE_INTEGRITY})
        self.assertIn(api.GATE_INTEGRITY, api.LEXICOGRAPHICALLY_PRIOR_GATE_CLASSES)

    def test_anchor_comparisons_declare_requires_anchor(self):
        report = I.run_source_integrity_gates(self.fx.inputs())
        needs = {g.gate_id for g in report.gates if g.requires_anchor}
        self.assertEqual(needs, {I.GATE_SYMBOL_PRESERVATION,
                                 I.GATE_SYMBOL_ARITY_COVERAGE,
                                 I.GATE_OP_REGISTRATION,
                                 I.GATE_DISPATCH_PREDICATE})

    def test_receipt_is_canonical_json_able_and_content_hashed(self):
        report = I.run_source_integrity_gates(self.fx.inputs())
        S.canonical_json(report.receipt)
        self.assertEqual(report.content_hash, S.content_hash(report.receipt))
        self.assertEqual(report.receipt["protocol_id"], api.PROTOCOL_VERSIONED_ID)

    def test_receipt_carries_no_authority_flavoured_key(self):
        report = I.run_source_integrity_gates(self.fx.inputs())
        self.assertEqual(S.find_authority_flavoured_keys(report.receipt), [])

    def test_a_missing_registration_kind_keeps_its_own_gate_id(self):
        """Both dispatch tables absent must NOT collapse onto the op-registration gate."""
        report = I.run_source_integrity_gates(self.fx.inputs(
            anchor_registrations=(self.fx.anchor_ops,),
            candidate_registrations=(self.fx.cand_ops,)))
        ids = [g.gate_id for g in report.gates]
        self.assertEqual(ids, list(I.GATE_IDS))
        by_id = outcomes(report.gates)
        self.assertEqual(by_id[I.GATE_OP_REGISTRATION], S.PASS)
        self.assertEqual(by_id[I.GATE_DISPATCH_PREDICATE], S.COULD_NOT_CHECK)
        self.assertTrue(report.blocking)

    def test_duplicate_registration_kinds_are_refused(self):
        with self.assertRaises(ValueError):
            self.fx.inputs(anchor_registrations=(self.fx.anchor_ops,
                                                 self.fx.anchor_ops))

    def test_could_not_check_alone_is_blocking(self):
        report = I.run_source_integrity_gates(
            self.fx.inputs(signature_index=None))
        self.assertEqual(outcomes(report.gates)[I.GATE_SYMBOL_ARITY_COVERAGE],
                         S.COULD_NOT_CHECK)
        self.assertTrue(report.blocking)

    def test_envelope_class_must_match_the_proposal_class(self):
        with self.assertRaises(ValueError):
            self.fx.inputs(change_class="arithmetic")

    def test_ceiling_backend_must_match_the_cell(self):
        with self.assertRaises(ValueError):
            self.fx.inputs(backend="llama_cpu")

    def test_build_provenance_must_be_for_this_candidate(self):
        with self.assertRaises(ValueError):
            self.fx.inputs(build=provenance(candidate_id="akc-9999"))

    def test_core_header_report_surfaces_the_first_page_notice(self):
        report = I.run_source_integrity_gates(self.fx.inputs(
            change_class="core_header",
            envelope=envelope(change_class="core_header"),
            declared_surface_scope=I.SURFACE_FULL_TREE))
        self.assertTrue(report.requires_human_code_review)
        self.assertTrue(report.first_page_notice.startswith(
            I.REQUIRES_HUMAN_CODE_REVIEW))


class TestRunnerWiring(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.fx = IntegrityFixture(self.tmp)

    def runner(self, **overrides) -> I.SourceIntegrityGateRunner:
        return I.SourceIntegrityGateRunner(
            tier="T0", inputs_by_candidate={"akc-0001": self.fx.inputs(**overrides)})

    def test_release_tier_is_refused_at_wiring_time(self):
        with self.assertRaises(api.TierNotOwned):
            I.SourceIntegrityGateRunner(tier="T3", inputs_by_candidate={})

    def test_unregistered_candidate_raises_rather_than_returning_no_gates(self):
        runner = self.runner()
        req = request(candidate_id="akc-unknown")
        with self.assertRaises(I.IntegrityInputsMissing):
            runner.run_gates(req)

    def test_runner_returns_the_integrity_gates(self):
        gates = self.runner().run_gates(self.fx.bound_request())
        self.assertEqual([g.gate_id for g in gates], list(I.RUNNER_GATE_IDS))
        self.assertEqual(list(I.RUNNER_GATE_IDS),
                         list(I.GATE_IDS) + [I.GATE_EVIDENCE_BINDING])

    def test_last_report_before_a_run_raises(self):
        with self.assertRaises(I.IntegrityInputsMissing):
            self.runner().last_report("akc-0001")

    def test_behavioural_runner_runs_only_when_integrity_is_clean(self):
        behavioural = _Behavioural()
        composed = I.SourceIntegrityFirstRunner(
            integrity=self.runner(), behavioural=behavioural)
        gates = composed.run_gates(self.fx.bound_request())
        self.assertEqual(behavioural.calls, 1)
        self.assertIn("mul_mat_exact_shapes", {g.gate_id for g in gates})

    def test_tier_mismatch_between_the_two_runners_is_refused(self):
        with self.assertRaises(ValueError):
            I.SourceIntegrityFirstRunner(integrity=self.runner(),
                                         behavioural=_Behavioural(tier="T2"))

    def test_composed_runner_plugs_into_the_api_dispatcher(self):
        composed = I.SourceIntegrityFirstRunner(
            integrity=self.runner(), behavioural=_Behavioural())
        dispatcher = api.TierDispatcher(gate_runners={"T0": composed})
        outcome = dispatcher.dispatch(self.fx.bound_request(tier="T0"),
                                      self.fx.bound_window())
        self.assertEqual(outcome.verdict.status, api.STATUS_PASS)


# ---------------------------------------------------------------------------
# The AK3 red-team list, by name
# ---------------------------------------------------------------------------

class TestRedTeamCandidates(unittest.TestCase):
    """*"a candidate that deletes a template specialization, one that drops a
    dispatch case, one that removes an op registration, and one whose incremental
    tree compiles while its snapshot does not. Each must fail before any
    behavioural check runs."* (AK3 checklist)"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.fx = IntegrityFixture(self.tmp)

    def assert_blocked_before_behaviour(self, inputs, expected_gate, expected_code):
        behavioural = _Behavioural()
        runner = I.SourceIntegrityGateRunner(
            tier="T0", inputs_by_candidate={"akc-0001": inputs})
        composed = I.SourceIntegrityFirstRunner(
            integrity=runner, behavioural=behavioural)
        req = self.fx.bound_request(inputs, tier="T0")
        gates = composed.run_gates(req)

        # 1. the named gate failed
        self.assertEqual(outcomes(gates)[expected_gate], S.FAIL)
        self.assertIn(expected_code, reasons_of(gates, expected_gate))
        # 2. the behavioural runner was never invoked
        self.assertEqual(behavioural.calls, 0)
        self.assertNotIn("mul_mat_exact_shapes", {g.gate_id for g in gates})
        self.assertEqual(outcomes(gates)[I.GATE_BEHAVIOURAL_NOT_RUN],
                         S.COULD_NOT_CHECK)
        # 3. the computed verdict is FAIL and the speed rank is UNOBTAINABLE
        dispatcher = api.TierDispatcher(gate_runners={"T0": composed})
        outcome = dispatcher.dispatch(self.fx.bound_request(inputs, tier="T0"),
                                      self.fx.bound_window(inputs))
        self.assertEqual(outcome.verdict.status, api.STATUS_FAIL)
        with self.assertRaises(api.SpeedRankUnavailable):
            outcome.verdict.rank_key()
        return outcome

    def test_deleted_template_specialization(self):
        cand = elf_table(
            self.tmp, "no-template.so",
            [s for s in ANCHOR_SYMS
             if s[0] != "_ZN4ggml6detail15kernel_dispatchILi8EEEvPKfPfi"],
            "candidate")
        outcome = self.assert_blocked_before_behaviour(
            self.fx.inputs(candidate_symbols=cand),
            I.GATE_SYMBOL_PRESERVATION, I.F_UNDECLARED_SYMBOL_REMOVAL)
        self.assertIn("kernel_dispatch",
                      " ".join(outcome.verdict.derivation) + str(outcome.verdict.to_dict()))

    def test_dropped_dispatch_case(self):
        cand_disp = DISPATCH_EXTRACTOR.extract_text(
            "candidate", DISPATCH_SOURCE_CANDIDATE_DROPPED)
        self.assert_blocked_before_behaviour(
            self.fx.inputs(candidate_registrations=(self.fx.cand_ops, cand_disp)),
            I.GATE_DISPATCH_PREDICATE, I.F_UNDECLARED_REGISTRATION_REMOVAL)

    def test_removed_op_registration(self):
        cand_ops = OP_EXTRACTOR.extract_text(
            "candidate", OPS_SOURCE_CANDIDATE_DROPPED)
        self.assert_blocked_before_behaviour(
            self.fx.inputs(candidate_registrations=(cand_ops, self.fx.cand_disp)),
            I.GATE_OP_REGISTRATION, I.F_UNDECLARED_REGISTRATION_REMOVAL)

    def test_incremental_tree_compiles_while_the_snapshot_does_not(self):
        self.assert_blocked_before_behaviour(
            self.fx.inputs(
                artifact_binary_sha256=sha("incremental-binary"),
                build=provenance(
                    incremental_output_binary_sha256=sha("incremental-binary"))),
            I.GATE_CLEAN_BUILD, I.F_ARTIFACT_FROM_INCREMENTAL_TREE)

    def test_gutted_file_that_still_compiles(self):
        """The escalation.py shape, ported: 454 lines to 3, and it compiles."""
        self.assert_blocked_before_behaviour(
            self.fx.inputs(
                declared_surface=surface(files=("src/escalation.cpp",)),
                diff=I.parse_unified_diff(GUTTING_DIFF),
                envelope=envelope(max_changed_lines=1000, max_hunks=20),
                original_line_counts={"src/escalation.cpp": 454}),
            I.GATE_SEMANTIC_DIFF, I.F_EXCESSIVE_SHRINKAGE)

    def test_core_header_edit_declared_as_a_small_dispatcher_change(self):
        self.assert_blocked_before_behaviour(
            self.fx.inputs(
                declared_surface=surface(files=("ggml/include/ggml.h",)),
                diff=I.parse_unified_diff(CORE_HEADER_DIFF)),
            I.GATE_CORE_HEADER, I.F_MISDECLARED_CORE_HEADER_CHANGE)

    def test_repair_compounded_onto_the_failed_tree(self):
        self.assert_blocked_before_behaviour(
            self.fx.inputs(repair=attempt(base_tree_sha256=sha("failed-tree"))),
            I.GATE_REPAIR_CLEAN_PARENT, I.F_REPAIR_CONTINUED_ON_FAILED_TREE)


# ---------------------------------------------------------------------------
# The module cannot write or run anything
# ---------------------------------------------------------------------------

class TestSelfAudit(unittest.TestCase):

    def test_module_passes_its_own_audit(self):
        self.assertEqual(I.audit_no_write_or_process_paths().outcome, S.PASS)

    def test_audit_detects_a_write_path(self):
        result = I.audit_no_write_or_process_paths(
            "from pathlib import Path\n"
            "def f(p):\n"
            "    Path(p).write_text('x')\n")
        self.assertEqual(result.outcome, S.FAIL)
        self.assertIn("write_text", " ".join(result.reasons))

    def test_audit_detects_a_process_import(self):
        result = I.audit_no_write_or_process_paths("import subprocess\n")
        self.assertEqual(result.outcome, S.FAIL)
        self.assertIn("subprocess", " ".join(result.reasons))

    def test_audit_detects_a_bare_open(self):
        result = I.audit_no_write_or_process_paths("def f(p):\n    return open(p)\n")
        self.assertEqual(result.outcome, S.FAIL)

    def test_audit_detects_a_write_mode_open(self):
        result = I.audit_no_write_or_process_paths(
            "from pathlib import Path\n"
            "def f(p):\n"
            "    return Path(p).open('wb')\n")
        self.assertEqual(result.outcome, S.FAIL)
        self.assertIn("'wb'", " ".join(result.reasons))

    def test_audit_detects_a_computed_open_mode(self):
        result = I.audit_no_write_or_process_paths(
            "from pathlib import Path\n"
            "def f(p, m):\n"
            "    return Path(p).open(m)\n")
        self.assertEqual(result.outcome, S.FAIL)

    def test_audit_permits_a_literal_read_open(self):
        """The compliant-path control: this module opens ELF binaries and source
        trees, so the rule is "no opener that could WRITE", not "no opener".

        The snippet declares `MODULE_ID` because the audit gained an identity
        binding on 2026-08-04 — a clean result is a statement about THIS module,
        and before the binding `audit_no_write_or_process_paths("")` returned PASS.
        Declaring it keeps the assertion at PASS rather than weakening it to
        `!= FAIL`, which would stop distinguishing "permitted" from "could not
        tell" — and that distinction is the entire point of the control.
        """
        result = I.audit_no_write_or_process_paths(
            "from pathlib import Path\n"
            f"MODULE_ID = {I.MODULE_ID!r}\n"
            "def f(p):\n"
            "    with Path(p).open('rb') as fh:\n"
            "        return fh.read()\n")
        self.assertEqual(result.outcome, S.PASS)

    def test_a_clean_foreign_module_is_not_certified(self):
        """The binding itself: clean text that is not this module cannot PASS.

        Bites the hole the 2026-08-03 pass recorded as closed at five sites and
        left live here, because the plan that scheduled the fix skipped this
        module as condemned — and it then survived the deletion, since
        `worktree.py` and `microbench.py` both need it.
        """
        clean = "import re\nPAT = re.compile('x')\ndef f():\n    return PAT\n"
        self.assertEqual(
            I.audit_no_write_or_process_paths(clean).outcome, S.COULD_NOT_CHECK)
        self.assertEqual(I.audit_no_write_or_process_paths("").outcome,
                         S.COULD_NOT_CHECK)
        # A finding is about the TEXT, so FAIL still outranks the identity question.
        self.assertEqual(
            I.audit_no_write_or_process_paths(
                "def f():\n    open('/x', 'w')\n").outcome, S.FAIL)

    def test_audit_of_unparseable_source_is_could_not_check(self):
        self.assertEqual(
            I.audit_no_write_or_process_paths("def (:\n").outcome, S.COULD_NOT_CHECK)

    def test_audit_detects_a_path_replace(self):
        """`Path.replace` overwrites a file and the audit used to permit it."""
        result = I.audit_no_write_or_process_paths(
            "from pathlib import Path\n"
            "def f(p, q):\n"
            "    Path(p).replace(q)\n")
        self.assertEqual(result.outcome, S.FAIL)
        self.assertIn("replace", " ".join(result.reasons))

    def test_audit_detects_an_io_import(self):
        """`io.open` is `open` under another name; `io` used to be permitted."""
        result = I.audit_no_write_or_process_paths("import io\n")
        self.assertEqual(result.outcome, S.FAIL)
        self.assertIn("io", " ".join(result.reasons))

    def test_audit_detects_a_symlink_to(self):
        result = I.audit_no_write_or_process_paths(
            "from pathlib import Path\n"
            "def f(p, q):\n"
            "    Path(p).symlink_to(q)\n")
        self.assertEqual(result.outcome, S.FAIL)

    def test_finding_codes_are_unique(self):
        self.assertEqual(len(I.FINDING_CODES), len(set(I.FINDING_CODES)))

    def test_gate_ids_are_unique_and_namespaced(self):
        self.assertEqual(len(I.GATE_IDS), len(set(I.GATE_IDS)))
        for gate_id in I.GATE_IDS:
            self.assertTrue(gate_id.startswith("integrity."))


# ---------------------------------------------------------------------------
# api fixtures reused for the dispatcher integration
# ---------------------------------------------------------------------------

def anchor_identity(**overrides) -> api.AnchorIdentity:
    kwargs = dict(
        source_commit=V8_COMMIT,
        binary_sha256=sha("anchor-binary"),
        linkage_sha256=sha("anchor-linkage"),
        measurement_event_ids=("ake-anchor-0001",),
    )
    kwargs.update(overrides)
    return api.AnchorIdentity(**kwargs)


def window(**overrides) -> api.WindowAttestations:
    kwargs = dict(
        resource_claim_receipt="gpu_device.mi210_0:claim-20260803T1200Z-8801",
        resource_claim_open=PASS,
        resource_claim_close=PASS,
        resource_claim_same_holder=PASS,
        no_concurrent_inference=PASS,
        preflight_attestation_ref="ake-preflight-0007",
        host_receipt="host-health-20260803T1159Z",
        host_health=PASS,
        anchor_at_open=anchor_identity(),
        anchor_at_close=anchor_identity(),
        anchor_gate=PASS,
        evaluator_bundle=PASS,
        runtime_source_label=PASS,
        recipe=api.RecipeReceipt(
            constructor_id="ak.microbench.llama_gpu.decode/v1",
            constructor_sha256=sha("recipe-constructor"),
            argv_sha256=sha("argv")),
        storage_open=PASS,
        storage_close=PASS,
        strata=PASS,
        stopping_rule_id="ak.stopping.bounded_extension/v1",
        rule_immutability=PASS,
        order_randomized=PASS,
        order_seed="campaign-seed-4711",
        aa_cadence=PASS,
        controls=api.ControlPanel(positive=PASS, neutral=PASS, degraded_negative=PASS,
                                  aa=PASS, historical_replay=PASS),
        calibration=PASS,
        control_definitions_immutable=PASS,
        raw_evidence_ref="data/ak-gpu-1/raw/akc-0001/",
    )
    kwargs.update(overrides)
    return api.WindowAttestations(**kwargs)


def request(**overrides) -> api.EvaluationRequest:
    kwargs = dict(
        event_id="ake-0001",
        campaign_id="ak-llama_gpu-decode-20260803",
        candidate_id="akc-0001",
        tier="T0",
        backend="llama_gpu",
        phase="decode",
        cell_class="instrument_tokens_per_s",
        protocol_id=api.PROTOCOL_VERSIONED_ID,
        artifact=api.ArtifactIdentity(
            source_sha256=sha("cand-source"),
            binary_sha256=sha("clean-binary"),
            linkage_sha256=sha("cand-linkage")),
        anchor=anchor_identity(),
        evaluator=api.EvaluatorIdentity(
            id="P-AK-SEARCH-1/v1",
            bundle_sha256=sha("evaluator-bundle"),
            runtime_source_label_ref="ake-srclabel-0003"),
        scope_denominator=api.ScopeDenominator(
            machine_subset="partial", numa_nodes=(), devices=("mi210_0",), cores=8),
        scope_manifest_sha256=sha("scope-manifest"),
        co_residency="single",
        determinism=api.DeterminismReport(
            determinism_class="bitwise_stable", same_seed_repeat_runs=3),
        metric="decode_tokens_per_s",
        metric_direction="higher_better",
        reps=10,
        change_class="parameter", anchor_tier="T1", transfer_ratio_to=(),
        created_at=NOW,
        campaign_controls=api.CampaignControls(
            calibration_block_count=30, contribution_floor=0.02, max_candidates=100,
            confirmation_admission_count=5, max_blocks_per_candidate=40,
            storage_floor_bytes_free=200 * 1024 ** 3),
        calibration=api.CalibrationOutputs(
            backend="llama_gpu", phase="decode", cell_class="instrument_tokens_per_s",
            noise_floor_phi=0.009, b_min_blocks=10, alpha_sel=0.01, alpha_conf=0.002,
            anchor_gate_band=(0.97, 1.03), accepted=True,
            solve_order_recorded=api.CALIBRATION_SOLVE_ORDER,
            samples_ref="data/ak-gpu-1/calibration/aa-blocks.jsonl",
            e_process_construction_id="sign_martingale_predictable_lambda/v1"),
        device_state=D.DeviceState(
            device_id="mi210_0", source="fixture/rocm-smi",
            nominal_sclk_mhz=1700, min_sclk_ratio=0.9,
            samples=(D.DeviceStateSample(1700, 1600, 180, 55, True),),
            receipt_ref="fixture://device-state/integrity"),
    )
    kwargs.update(overrides)
    return api.EvaluationRequest(**kwargs)


# ---------------------------------------------------------------------------
# Red-team regressions (2026-08-03 adversarial review of integrity.py)
#
# Every test below reproduces a way the module answered PASS, or answered
# nothing, on evidence it had not actually checked. Each one FAILED against the
# module as first written; the accompanying fix is named in the docstring.
# ---------------------------------------------------------------------------

class TestArityCoverageFailOpen(unittest.TestCase):
    """`signature_index` entries that declare no arity used to answer PASS.

    `entry.get("anchor") != entry.get("candidate")` compares None against None
    and finds no mismatch, so `{name: {}}` — an entry with the arities deleted —
    produced *"signature_index covers all N unmangled exported symbols and their
    arities agree"*. That is the gate passing because the thing it inspects was
    removed, which is the exact fail-open the module's own docstring names.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.anchor = elf_table(self.tmp, "a.so", ANCHOR_SYMS, "anchor")
        self.cand = elf_table(self.tmp, "c.so", ANCHOR_SYMS, "candidate")

    def gate(self, index):
        return I.check_symbol_arity_coverage(self.anchor, self.cand, index)

    def test_entries_with_both_arities_deleted_are_could_not_check_never_pass(self):
        gate = self.gate({"ggml_mul_mat": {},
                          "ggml_backend_hip_supports_op": {}})
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn(I.F_UNMANGLED_ARITY_NOT_DERIVABLE, " ".join(gate.check.reasons))
        self.assertIn("declares no anchor/candidate arity", " ".join(gate.check.reasons))

    def test_an_entry_missing_one_side_covers_nothing(self):
        gate = self.gate({"ggml_mul_mat": {"anchor": 3},
                          "ggml_backend_hip_supports_op": {"anchor": 2,
                                                           "candidate": 2}})
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("ggml_mul_mat", " ".join(gate.check.reasons))
        self.assertIn("declares no candidate arity", " ".join(gate.check.reasons))

    def test_a_non_int_arity_raises_rather_than_comparing(self):
        with self.assertRaises(TypeError):
            self.gate({"ggml_mul_mat": {"anchor": "3", "candidate": "3"},
                       "ggml_backend_hip_supports_op": {"anchor": 2, "candidate": 2}})

    def test_fully_declared_agreeing_arities_still_pass(self):
        gate = self.gate({"ggml_mul_mat": {"anchor": 3, "candidate": 3},
                          "ggml_backend_hip_supports_op": {"anchor": 2, "candidate": 2}})
        self.assertEqual(gate.check.outcome, S.PASS)

    def test_a_real_arity_change_still_fails(self):
        gate = self.gate({"ggml_mul_mat": {"anchor": 3, "candidate": 4},
                          "ggml_backend_hip_supports_op": {"anchor": 2, "candidate": 2}})
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_UNDECLARED_ARITY_CHANGE, " ".join(gate.check.reasons))

    def test_the_uncovered_branch_states_its_own_cap(self):
        """The 20-symbol display cap must say so, as the other branch already did."""
        many = [fn(f"c_sym_{i:03d}") for i in range(40)]
        anchor = elf_table(self.tmp, "a40.so", many, "anchor")
        cand = elf_table(self.tmp, "c40.so", many, "candidate")
        gate = I.check_symbol_arity_coverage(anchor, cand, {})
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertTrue(gate.notes, "a truncated listing with no note is a silent cap")
        self.assertIn("at most 20 of 40", " ".join(gate.notes))


class TestRegistrationArityNotDerivable(unittest.TestCase):
    """An arity present on one side and absent on the other used to read clean.

    `diff_registration_tables` compared arities only when BOTH were non-None, so
    an adapter pattern whose `arity` group is optional turned `MUL_MAT, 2` ->
    `MUL_MAT` into PASS with no finding and no note — the same "None means
    unchanged" conflation `ParsedName.param_count` documents as forbidden.
    """

    EXTRACTOR = I.PatternRegistrationExtractor(
        kind=I.KIND_OP_REGISTRATION,
        patterns={"ops": r"GGML_OP_REGISTER\((?P<key>GGML_OP_\w+)(?:,\s*(?P<arity>\d+))?\)"},
        declared_by="adapter:optional-arity/v1")

    def tables(self, anchor_src, cand_src):
        return (self.EXTRACTOR.extract_text("anchor", {"o.cpp": anchor_src}),
                self.EXTRACTOR.extract_text("candidate", {"o.cpp": cand_src}))

    def test_arity_dropped_from_the_candidate_is_could_not_check(self):
        anchor, cand = self.tables("GGML_OP_REGISTER(GGML_OP_MUL_MAT, 2)\n",
                                   "GGML_OP_REGISTER(GGML_OP_MUL_MAT)\n")
        self.assertEqual(anchor.entries[0].arity, 2)
        self.assertIsNone(cand.entries[0].arity)
        gate = I.check_registration_preservation(anchor, cand, deltas())
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn(I.F_REGISTRATION_ARITY_NOT_DERIVABLE, " ".join(gate.check.reasons))
        self.assertIn("GGML_OP_MUL_MAT", " ".join(gate.check.reasons))

    def test_arity_appearing_only_on_the_candidate_is_also_could_not_check(self):
        anchor, cand = self.tables("GGML_OP_REGISTER(GGML_OP_MUL_MAT)\n",
                                   "GGML_OP_REGISTER(GGML_OP_MUL_MAT, 4)\n")
        gate = I.check_registration_preservation(anchor, cand, deltas())
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)

    def test_declaring_the_arity_change_settles_it(self):
        anchor, cand = self.tables("GGML_OP_REGISTER(GGML_OP_MUL_MAT, 2)\n",
                                   "GGML_OP_REGISTER(GGML_OP_MUL_MAT)\n")
        gate = I.check_registration_preservation(
            anchor, cand, deltas(arity_changed=("GGML_OP_MUL_MAT",)))
        self.assertEqual(gate.check.outcome, S.PASS)

    def test_both_arities_absent_is_still_a_clean_pass(self):
        anchor, cand = self.tables("GGML_OP_REGISTER(GGML_OP_MUL_MAT)\n",
                                   "GGML_OP_REGISTER(GGML_OP_MUL_MAT)\n")
        gate = I.check_registration_preservation(anchor, cand, deltas())
        self.assertEqual(gate.check.outcome, S.PASS)

    def test_the_diff_records_the_uncomparable_pair(self):
        anchor, cand = self.tables("GGML_OP_REGISTER(GGML_OP_MUL_MAT, 2)\n",
                                   "GGML_OP_REGISTER(GGML_OP_MUL_MAT)\n")
        diff = I.diff_registration_tables(anchor, cand)
        self.assertEqual(diff.arity_changed, ())
        self.assertEqual(diff.arity_not_comparable,
                         (("ops", "GGML_OP_MUL_MAT", 2, None),))
        self.assertIn("arity_not_comparable", diff.to_dict())


class TestPathContainmentEvasion(unittest.TestCase):
    """`_is_within` compared raw `Path.parts`, so a `..` segment escaped it.

    `/mnt/raid0/llm/x/../llama.cpp/build` IS inside the frozen production tree
    `/mnt/raid0/llm/llama.cpp`, but its parts do not have the production tree's
    parts as a prefix, so `check_clean_build_from_snapshot` answered PASS on a
    build in a production tree — the invariant-3 check the gate exists for.
    """

    PROD = "/mnt/raid0/llm/llama.cpp"

    def test_dotdot_no_longer_escapes_containment(self):
        self.assertTrue(I._is_within(f"{self.PROD}/build", self.PROD))
        self.assertTrue(I._is_within("/mnt/raid0/llm/x/../llama.cpp/build", self.PROD))
        self.assertTrue(I._is_within("/mnt/raid0/llm/./llama.cpp/build", self.PROD))
        self.assertFalse(I._is_within("/mnt/raid0/llm/llama.cpp/../ak/build", self.PROD))
        self.assertFalse(I._is_within("/mnt/raid0/llm/llama.cpp2/build", self.PROD))

    def test_relative_paths_are_still_not_within_anything(self):
        self.assertFalse(I._is_within("llama.cpp/build", self.PROD))
        self.assertFalse(I._is_within(f"{self.PROD}/build", "llama.cpp"))

    def test_an_unnormalized_attested_build_dir_is_refused_at_construction(self):
        for field in ("build_dir", "source_root", "actor_worktree"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    provenance(**{field: "/mnt/raid0/llm/x/../llama.cpp/build"})

    def test_an_unnormalized_production_tree_path_is_refused(self):
        # `Path` already collapses a bare '.', so '..' is what the guard must
        # catch — and '..' is the segment that made containment escapable.
        with self.assertRaises(ValueError):
            provenance(production_tree_paths=("/mnt/raid0/llm/x/../llama.cpp",))

    def test_a_build_inside_a_production_tree_fails_however_it_is_spelled(self):
        gate, _receipt = I.check_clean_build_from_snapshot(
            provenance(build_dir="/mnt/raid0/llm/llama.cpp/build",
                       source_root="/mnt/raid0/llm/llama.cpp"),
            sha("clean-binary"), recompute_root=None,
            snapshot_attested_by="storage.verify_durability:ak-1")
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_BUILD_IN_PRODUCTION_TREE, " ".join(gate.check.reasons))


class TestUnparseableAndEmptyDiffs(unittest.TestCase):
    """A diff the parser could not account for became a zero-line diff.

    `parse_unified_diff` documents *"Raises DiffParseError on anything it cannot
    account for"*, but text with no file section fell through the commentary
    branch and returned `SourceDiff(())`. The consequence is not cosmetic: an
    empty diff touches no undeclared file, deletes nothing, shrinks nothing and
    matches no core path, so the semantic gate PASSed and the mechanically
    derived core-header tier came back `standard`.
    """

    def test_text_that_is_not_a_diff_raises(self):
        with self.assertRaises(I.DiffParseError) as ctx:
            I.parse_unified_diff("this is not a diff at all\njust prose\n")
        self.assertIn(I.F_UNPARSEABLE_DIFF, str(ctx.exception))

    def test_a_truncated_diff_whose_headers_were_lost_raises(self):
        with self.assertRaises(I.DiffParseError):
            I.parse_unified_diff("index 1111111..2222222 100644\nsimilarity index 90%\n")

    def test_a_genuinely_empty_diff_is_still_an_empty_source_diff(self):
        self.assertEqual(I.parse_unified_diff("").files_touched, 0)
        self.assertEqual(I.parse_unified_diff("   \n\n").files_touched, 0)

    def test_an_empty_diff_cannot_evidence_semantic_conformance(self):
        gate = I.check_semantic_diff_conformance(
            I.SourceDiff(files=()), surface(), envelope(), original_line_counts=None)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn(I.F_EMPTY_DIFF, " ".join(gate.check.reasons))

    def test_an_empty_diff_cannot_evidence_a_risk_tier(self):
        decision, gate = I.assess_risk_tier(
            "dispatcher", I.SourceDiff(files=()), core_policy(),
            declared_surface_scope=I.SURFACE_PARTIAL)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn(I.F_EMPTY_DIFF, " ".join(gate.check.reasons))
        self.assertEqual(decision.tier, "standard")

    def test_an_empty_diff_blocks_the_whole_report(self):
        with tempfile.TemporaryDirectory() as raw:
            fx = IntegrityFixture(Path(raw))
            report = I.run_source_integrity_gates(
                fx.inputs(diff=I.SourceDiff(files=()), original_line_counts=None))
        self.assertTrue(report.blocking)
        self.assertEqual(outcomes(report.gates)[I.GATE_SEMANTIC_DIFF],
                         S.COULD_NOT_CHECK)


class TestFirstPageNoticeOnTheReceipt(unittest.TestCase):
    """§10.6's marker was computed on the report and absent from the receipt.

    *"Above it, the package is marked `REQUIRES_HUMAN_CODE_REVIEW` and says so on
    its first page."* The receipt is the journaled page, and it carried only the
    `complexity` block — which reads `requires_human_code_review: false` for a
    misdeclared core-header edit whenever the adapter declares
    `shared_core_modification_requires_review=False`, while the report itself
    says true.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.fx = IntegrityFixture(self.tmp)

    def test_the_receipt_carries_the_marker_at_its_top_level(self):
        report = I.run_source_integrity_gates(self.fx.inputs(
            change_class="core_header",
            envelope=envelope(change_class="core_header"),
            declared_surface_scope=I.SURFACE_FULL_TREE))
        self.assertTrue(report.receipt["requires_human_code_review"])
        self.assertEqual(report.receipt["first_page_notice"],
                         report.first_page_notice)
        self.assertTrue(report.receipt["first_page_notice"].startswith(
            I.REQUIRES_HUMAN_CODE_REVIEW))

    def test_a_clean_candidate_carries_the_marker_as_false_not_as_silence(self):
        report = I.run_source_integrity_gates(self.fx.inputs())
        self.assertIn("requires_human_code_review", report.receipt)
        self.assertFalse(report.receipt["requires_human_code_review"])
        self.assertIsNone(report.receipt["first_page_notice"])

    def test_an_adapter_waiving_shared_core_review_cannot_zero_the_receipt(self):
        report = I.run_source_integrity_gates(self.fx.inputs(
            declared_surface=surface(files=("ggml/include/ggml.h",)),
            diff=I.parse_unified_diff(CORE_HEADER_DIFF),
            declared_surface_scope=I.SURFACE_FULL_TREE,
            original_line_counts={"ggml/include/ggml.h": 200},
            complexity_ceiling=ceiling(
                shared_core_modification_requires_review=False)))
        self.assertEqual(report.risk_tier.tier, "core_header")
        self.assertFalse(report.complexity.requires_human_code_review)
        self.assertTrue(report.requires_human_code_review)
        self.assertTrue(report.receipt["requires_human_code_review"])
        self.assertIn(I.REQUIRES_HUMAN_CODE_REVIEW,
                      report.receipt["first_page_notice"])

    def test_the_receipt_is_still_canonical_json_able(self):
        report = I.run_source_integrity_gates(self.fx.inputs())
        S.canonical_json(report.receipt)
        self.assertEqual(report.content_hash, S.content_hash(report.receipt))
        self.assertEqual(S.find_authority_flavoured_keys(report.receipt), [])


class TestEvidenceBinding(unittest.TestCase):
    """The gates never checked WHICH binaries their symbol tables came from.

    P-AK-SEARCH-1 precondition 4: *"Every … comparison names its anchor by source
    commit, binary SHA-256, and linkage SHA-256 … A rebuilt anchor is a different
    anchor."* `requires_anchor=True` only proves that SOME anchor is bound to the
    window; it never compared `ElfSymbolTable.file_sha256` against
    `request.anchor.binary_sha256`. A table extracted from any file at all diffed
    clean and the dispatcher returned `pass`.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.fx = IntegrityFixture(self.tmp)

    def runner(self, **overrides):
        inputs = self.fx.inputs(**overrides)
        return inputs, I.SourceIntegrityGateRunner(
            tier="T0", inputs_by_candidate={inputs.candidate_id: inputs})

    def test_a_bound_pair_passes(self):
        inputs, runner = self.runner()
        gates = runner.run_gates(self.fx.bound_request(inputs))
        self.assertEqual(outcomes(gates)[I.GATE_EVIDENCE_BINDING], S.PASS)

    def test_a_symbol_table_from_a_binary_that_is_not_the_named_anchor(self):
        inputs, runner = self.runner()
        req = self.fx.bound_request(
            inputs, anchor=anchor_identity(binary_sha256=sha("some-other-build")))
        gates = runner.run_gates(req)
        self.assertEqual(outcomes(gates)[I.GATE_EVIDENCE_BINDING], S.COULD_NOT_CHECK)
        self.assertIn(I.F_SYMBOL_TABLE_NOT_BOUND_TO_ANCHOR,
                      reasons_of(gates, I.GATE_EVIDENCE_BINDING))

    def test_a_candidate_table_that_is_not_the_artifact_under_test(self):
        other = elf_table(self.tmp, "other.so", ANCHOR_SYMS + [fn("pad")], "candidate")
        inputs, runner = self.runner(candidate_symbols=other)
        gates = runner.run_gates(self.fx.bound_request(inputs))
        self.assertEqual(outcomes(gates)[I.GATE_EVIDENCE_BINDING], S.COULD_NOT_CHECK)
        self.assertIn(I.F_SYMBOL_TABLE_NOT_BOUND_TO_ARTIFACT,
                      reasons_of(gates, I.GATE_EVIDENCE_BINDING))

    def test_an_artifact_hash_the_request_and_the_inputs_disagree_on_fails(self):
        inputs, runner = self.runner()
        req = self.fx.bound_request(inputs, artifact=api.ArtifactIdentity(
            source_sha256=sha("cand-source"),
            binary_sha256=sha("a-different-binary"),
            linkage_sha256=sha("cand-linkage")))
        gates = runner.run_gates(req)
        self.assertEqual(outcomes(gates)[I.GATE_EVIDENCE_BINDING], S.FAIL)
        self.assertIn(I.F_ARTIFACT_SHA256_MISMATCH,
                      reasons_of(gates, I.GATE_EVIDENCE_BINDING))

    def test_an_anchorless_request_cannot_bind_anything(self):
        inputs, runner = self.runner()
        gates = runner.run_gates(self.fx.bound_request(inputs, anchor=None))
        self.assertEqual(outcomes(gates)[I.GATE_EVIDENCE_BINDING], S.COULD_NOT_CHECK)
        self.assertIn(I.F_NO_ANCHOR_BOUND, reasons_of(gates, I.GATE_EVIDENCE_BINDING))

    def test_the_binding_gate_declares_requires_anchor(self):
        inputs, runner = self.runner()
        gates = runner.run_gates(self.fx.bound_request(inputs))
        binding = [g for g in gates if g.gate_id == I.GATE_EVIDENCE_BINDING][0]
        self.assertTrue(binding.requires_anchor)
        self.assertEqual(binding.gate_class, api.GATE_INTEGRITY)

    def test_behavioural_gates_do_not_run_on_unbound_evidence(self):
        inputs, runner = self.runner()
        behavioural = _Behavioural()
        composed = I.SourceIntegrityFirstRunner(
            integrity=runner, behavioural=behavioural)
        req = self.fx.bound_request(
            inputs, tier="T0",
            anchor=anchor_identity(binary_sha256=sha("some-other-build")))
        gates = composed.run_gates(req)
        self.assertEqual(behavioural.calls, 0)
        self.assertEqual(outcomes(gates)[I.GATE_BEHAVIOURAL_NOT_RUN],
                         S.COULD_NOT_CHECK)
        self.assertNotIn("mul_mat_exact_shapes", {g.gate_id for g in gates})

    def test_an_unbound_anchor_denies_the_verdict_and_the_speed_rank(self):
        inputs, runner = self.runner()
        composed = I.SourceIntegrityFirstRunner(
            integrity=runner, behavioural=_Behavioural())
        dispatcher = api.TierDispatcher(gate_runners={"T0": composed})
        stranger = anchor_identity(binary_sha256=sha("some-other-build"))
        outcome = dispatcher.dispatch(
            self.fx.bound_request(inputs, tier="T0", anchor=stranger),
            window(anchor_at_open=stranger, anchor_at_close=stranger))
        self.assertNotEqual(outcome.verdict.status, api.STATUS_PASS)
        with self.assertRaises(api.SpeedRankUnavailable):
            outcome.verdict.rank_key()

    def test_a_mis_keyed_inputs_mapping_is_refused_at_wiring_time(self):
        with self.assertRaises(ValueError):
            I.SourceIntegrityGateRunner(
                tier="T0", inputs_by_candidate={"akc-9999": self.fx.inputs()})

    def test_a_request_for_another_candidate_fails_the_binding(self):
        inputs = self.fx.inputs(candidate_id="akc-0002",
                                build=provenance(
                                    candidate_id="akc-0002",
                                    output_binary_sha256=self.fx.cand_syms.file_sha256))
        gate = I.check_evidence_binding(self.fx.bound_request(inputs), inputs)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn(I.F_CANDIDATE_ID_MISMATCH, " ".join(gate.check.reasons))


if __name__ == "__main__":
    unittest.main(verbosity=2)
