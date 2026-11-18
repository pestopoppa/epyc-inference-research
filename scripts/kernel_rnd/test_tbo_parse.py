import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from rowlib.tbo_parse import parse_tbo, strip_ansi

FX = pathlib.Path(__file__).parent / "rowlib" / "fixtures"


def test_ansi_ok_line_counted():
    t = (FX / "pass_trim.log").read_text()
    assert "\x1b[" in t
    r = parse_tbo(t, 0)
    assert r["parsed"] and r["n_fail"] == 0 and r["n_ok"] >= 2
    assert r["passed_line"] == "817/817 tests passed"
    assert not any(l.endswith("OK") for l in t.splitlines() if "MUL_MAT_ID(" in l)


def test_fail_block():
    r = parse_tbo((FX / "fail_trim.log").read_text(), 1)
    assert r["parsed"] and r["n_fail"] == 1
    assert r["fail_names"][0].startswith("MUL_MAT(type_a=q6_K")
    assert r["passed_line"] == "1211/1212 tests passed"


def test_fail_closed():
    ok = (FX / "pass_trim.log").read_text()
    assert not parse_tbo(ok, 1)["parsed"]          # rc!=0, 0 failures
    assert not parse_tbo("garbage\n", 0)["parsed"]  # no N/M line
    assert not parse_tbo("", 0)["parsed"]
    assert strip_ansi("\x1b[1;32mOK\x1b[0m") == "OK"
