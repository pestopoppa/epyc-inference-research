"""One reading of llama.cpp tool stdout: colour stripping and the vacuous-parse rule.

test-backend-ops colours its verdicts, so a parser that forgets to strip counts zero
`OK` lines. Private copies of the regex let that recur (false REJECT on C8-1-LCATTN:
0 of 167 OK lines). Every parser of tool output imports `strip_ansi` from here;
`test_tool_output.py` fails on a new copy or an unmigrated parser.

Vacuous-parse rule: zero verdict lines read means the parser saw no result, which is
a harness fault and not evidence about a candidate. It yields UNKNOWN, never a reject.
"""
from __future__ import annotations

import re

UNKNOWN = "UNKNOWN"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def vacuous_parse(ok: int, fail: int, total: int = 0) -> bool:
    """True when no verdict line and no harness tally was read from the output."""
    return ok == 0 and fail == 0 and total == 0
