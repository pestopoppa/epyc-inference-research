"""Canonical test-backend-ops output parser for row harnesses (import, never copy).

Never use ``line.endswith('OK')``: colour output ends lines with ``OK\\x1b[0m``.
"""
import re

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# A passing case line ends in ``: OK`` (default) or ``... OK`` with no colon (--autokernel-properties:
# ``... oracle=ggml_cpu_reference/v1 OK``). A bare ``OK`` summary line never counts (needs a non-space before it).
_OKLINE = re.compile(r"(?::\s*|\S\s+)OK\s*$")
_NM = re.compile(r"^\s*(\d+)/(\d+) tests passed\s*$")


def strip_ansi(text):
    return _ANSI.sub("", text)


def parse_tbo(text, rc):
    """Return {n_ok, n_fail, fail_names, passed_line, parsed}.

    Fails closed (parsed=False): the 'N/M tests passed' line is missing, rc != 0
    with 0 failures found, or the counts disagree with that line.
    """
    clean = strip_ansi(text)
    lines = clean.splitlines()
    n_ok = sum(1 for l in lines if _OKLINE.search(l) and "Backend" not in l)
    names, in_block = [], False
    for l in lines:
        if "Failing tests:" in l:
            in_block = True
            continue
        if in_block:
            if re.match(r"^\s*Backend [^:]*:", l) or not l.strip():
                in_block = False
            elif re.match(r"^\s+\S+\(", l):
                names.append(l.strip())
    names = sorted(set(names))
    passed_line = None
    n_pass = n_tot = None
    for l in lines:
        m = _NM.match(l)
        if m:
            passed_line = l.strip()
            n_pass, n_tot = int(m.group(1)), int(m.group(2))
    n_fail = len(names)
    parsed = passed_line is not None
    if parsed and n_tot - n_pass != n_fail:
        parsed = False
    if rc != 0 and n_fail == 0:
        parsed = False
    return {"n_ok": n_ok, "n_fail": n_fail, "fail_names": names,
            "passed_line": passed_line, "parsed": parsed}
