"""One shared ANSI strip, the vacuous-parse rule, and the lint that keeps them single."""
from __future__ import annotations

from pathlib import Path
import re

from . import fold2_gates
from .tool_output import UNKNOWN, strip_ansi, vacuous_parse

PACKAGE = Path(__file__).resolve().parent.parent
GREEN, RED, RESET = "\x1b[32m", "\x1b[31m", "\x1b[0m"
#: The shapes of a parser reading test-backend-ops / llama-bench verdict text.
PARSER_MARKER = re.compile(r'tests passed|Backend CPU: |: OK\b|: FAIL\b')
HOME = "loop/tool_output.py"


def _sources():
    # loop/ is the scope: execution/microbench.py and asm_retarget strip a wider CSI set.
    for path in sorted((PACKAGE / "loop").rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        if not path.name.startswith("test_") and rel != HOME:
            yield rel, path.read_text(encoding="utf-8")


def test_coloured_real_fail_still_fails():
    text = (f"  ADD(a): {RED}FAIL{RESET}\n  ADD(b): {GREEN}OK{RESET}\n  1/2 tests passed\n")
    assert fold2_gates.backend_op_verdict(0, fold2_gates.tally(text)) == "FAIL"


def test_coloured_all_ok_passes():
    text = "".join(f"  ADD({i}): {GREEN}OK{RESET}\n" for i in range(167)) + "  167/167 tests passed\n"
    counts = fold2_gates.tally(text)
    assert counts["ok"] == 167
    assert fold2_gates.backend_op_verdict(0, counts) == "PASS"


def test_nonempty_output_with_no_verdict_line_is_unknown():
    text = "main: loading model\n" * 167
    assert fold2_gates.backend_op_verdict(0, fold2_gates.tally(text)) == UNKNOWN
    assert fold2_gates.backend_op_verdict(1, fold2_gates.tally(text)) == UNKNOWN


def test_empty_output_is_unknown():
    assert fold2_gates.backend_op_verdict(0, fold2_gates.tally("")) == UNKNOWN


def test_rule_needs_all_three_counts_empty():
    assert vacuous_parse(0, 0, 0)
    assert not vacuous_parse(1, 0, 0)
    assert not vacuous_parse(0, 1, 0)
    assert not vacuous_parse(0, 0, 2)


def test_strip_is_exact():
    assert strip_ansi(f"a{GREEN}b{RESET}c") == "abc"


def test_no_private_ansi_copy():
    copies = [rel for rel, text in _sources() if "x1b" in text and "[0-9;]" in text]
    assert copies == []


def test_every_verdict_parser_imports_the_shared_strip():
    bad = [rel for rel, text in _sources()
           if PARSER_MARKER.search(re.sub(r'"""[\s\S]*?"""', "", text))
           and "strip_ansi" not in text]
    assert bad == []
