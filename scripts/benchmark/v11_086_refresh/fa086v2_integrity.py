#!/usr/bin/env python3
"""fa086v2_integrity.py -- V11-086 v2 integrity verdict from the correctness half run by jobs/v11_086_fa_vkq_build_job.sh.

numpy-free on purpose (it runs under /usr/bin/python3 -I, which has no numpy; the v1 driver's run_fa_harness imported numpy and
used cmp.py, so it would have died at the harness step under the filler's interpreter).

Inputs (all under --corr): tbo_<arm>.log/.rc for arms C Bunset B1 Bauto long_B1 long_Bauto, fa_<h>_<arm>/out_*.bin + .rc for
h in {h1,h2} x arm in {C,Bunset,B1,Bauto}.  Output: INTEGRITY.json with the three predeclared gates
  tbo_fa_100            every tbo run rc==0 AND 'N/N tests passed' with N>0 AND no FAIL line
  fa_unset_bit_identical  for h1 and h2: Bunset dumps byte-identical to C dumps, same file set, >0 files, both runs rc==0
  finite_all_dumps      every float32 in every dump of every arm finite (exponent not all-ones), >0 files per arm, rc==0
None = unmeasured (a step missing).  Exit 0 whenever the file is written.
"""
import argparse, json, re, subprocess, sys, time
from array import array
from pathlib import Path

TBO_ARMS = ("C", "Bunset", "B1", "Bauto", "long_B1", "long_Bauto")
H_ARMS = ("C", "Bunset", "B1", "Bauto")


def rc_of(p: Path):
    try: return int(p.read_text().strip())
    except Exception: return None


# 2026-10-22 parser fix (V11-086 integrity check): test-backend-ops colours its verdicts ("\x1b[1;31mFAIL\x1b[0m"), so the
# 'm' of the escape sequence sits directly before "FAIL" and r"\bFAIL\b" never matched -> n_fail_lines 0 for 12 failures.
# Strip ANSI first; capture each per-case error line ("[OP] ERR|NMSE = x > thr   CASE(vars): test failed FAIL") as a record,
# and the "Failing tests:" block.  A pass count below total with no captured case is reported as parser_consistent=False.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ERR_LINE_RE = re.compile(r"^\s*\[(?P<op>[A-Z_0-9]+)\]\s+(?P<metric>[A-Z_]+)\s*=\s*(?P<err>[0-9.eE+-]+|nan|inf)\s*>\s*(?P<thr>[0-9.eE+-]+)\s+"
                         r"(?P<case>[A-Z_0-9]+\(.*\)):\s*test failed")
FAIL_RE = re.compile(r"\bFAIL\b")


def parse_tbo_text(t: str) -> dict:
    t = ANSI_RE.sub("", t)
    lines = t.splitlines()
    m = re.findall(r"(\d+)/(\d+) tests passed", t)
    passed, total = (int(m[-1][0]), int(m[-1][1])) if m else (None, None)
    fails = [l.strip() for l in lines if FAIL_RE.search(l)]
    cases = []
    for l in lines:
        e = ERR_LINE_RE.match(l)
        if e:
            cases.append({"op": e["op"], "metric": e["metric"], "err": float(e["err"]), "threshold": float(e["thr"]), "case": e["case"]})
    listed = []
    if "Failing tests:" in t:
        for l in t.split("Failing tests:", 1)[1].splitlines()[1:]:
            s = l.strip()
            if not s or s.startswith("Backend ") or not re.match(r"^[A-Z_0-9]+\(", s):
                break
            listed.append(s)
    n_failed = (total - passed) if m else None
    consistent = None if n_failed is None else (n_failed == 0 and not cases) or (len(cases) == n_failed == (len(listed) or n_failed))
    return {"present": True, "passed": passed, "total": total, "n_fail_lines": len(fails), "first_fails": fails[:5],
            "n_failed": n_failed, "failed_cases": cases, "failing_tests_listed": listed, "parser_consistent": consistent,
            "max_err_on_fail": max((c["err"] for c in cases), default=None),
            "ok100": bool(m) and total > 0 and passed == total and not fails}


def parse_tbo(log: Path) -> dict:
    if not log.exists(): return {"present": False, "ok100": None}
    return parse_tbo_text(log.read_text(errors="replace"))


def _selftest() -> None:
    red = "\x1b[1;31mFAIL\x1b[0m"; grn = "\x1b[1;32mOK\x1b[0m"
    txt = ("  FLASH_ATTN_EXT(hsk=40,kv=113): " + grn + "\n"
           "[FLASH_ATTN_EXT] ERR = 0.001371142 > 0.000500000   FLASH_ATTN_EXT(hsk=256,hsv=256,kv=65536,nb=1,permute=[0,1,2,3]): test failed " + red + "\n"
           "  1/2 tests passed\n\n  Failing tests:\n  FLASH_ATTN_EXT(hsk=256,hsv=256,kv=65536,nb=1,permute=[0,1,2,3])\n  Backend CPU: " + red + "\n0/1 backends passed\n" + red + "\n")
    r = parse_tbo_text(txt)
    assert r["passed"] == 1 and r["total"] == 2 and not r["ok100"], r
    assert len(r["failed_cases"]) == 1 and r["failed_cases"][0]["err"] == 0.001371142 and r["failed_cases"][0]["threshold"] == 5e-4, r
    assert r["failing_tests_listed"] == ["FLASH_ATTN_EXT(hsk=256,hsv=256,kv=65536,nb=1,permute=[0,1,2,3])"], r
    assert r["n_fail_lines"] == 3 and r["parser_consistent"] is True, r
    ok = parse_tbo_text("  X(a=1): " + grn + "\n  5/5 tests passed\n  Backend CPU: \x1b[1;32mOK\x1b[0m\n")
    assert ok["ok100"] and ok["parser_consistent"] is True and ok["n_fail_lines"] == 0, ok
    blind = parse_tbo_text("  3/5 tests passed\n")
    assert blind["parser_consistent"] is False and not blind["ok100"], blind


def nonfinite_count(f: Path) -> int:
    a = array("I"); n = f.stat().st_size // 4
    with open(f, "rb") as fh: a.fromfile(fh, n)
    return sum(1 for x in a if (x & 0x7F800000) == 0x7F800000)


def main() -> int:
    if sys.argv[1:] == ["--selftest"]:
        _selftest(); print("selftest ok"); return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr", type=Path, required=True)
    ap.add_argument("--b-build", type=Path, required=True)
    ap.add_argument("--c-build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    R = {"schema": "epyc.v11-086v2.integrity.v1", "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "b_build": str(a.b_build), "c_build": str(a.c_build)}
    libs = [p for p in sorted((a.b_build / "bin").glob("libggml-cpu.so*")) if p.is_file() and not p.is_symlink()]
    R["knob_strings_proof"] = bool(libs) and any(b"GGML_FA_VKQ_REG" in subprocess.run(["strings", "-a", str(l)], capture_output=True).stdout for l in libs)
    tbo = {}
    for arm in TBO_ARMS:
        r = parse_tbo(a.corr / f"tbo_{arm}.log") | {"rc": rc_of(a.corr / f"tbo_{arm}.rc")}
        r["ok"] = None if not r["present"] or r["rc"] is None else (r["rc"] == 0 and bool(r["ok100"]))
        tbo[arm] = r
    R["tbo"] = tbo
    oks = [r["ok"] for r in tbo.values()]
    R["tbo_fa_100"] = None if any(o is None for o in oks) else all(oks)
    R["tbo_parser_consistent"] = all(r.get("parser_consistent") is not False for r in tbo.values())

    harness, ident, finite = {}, [], []
    for hn in ("h1", "h2"):
        hh = {}
        for arm in H_ARMS:
            d = a.corr / f"fa_{hn}_{arm}"; rc = rc_of(a.corr / f"fa_{hn}_{arm}.rc")
            files = sorted(d.glob("out_*.bin")) if d.is_dir() else []
            nf = sum(nonfinite_count(f) for f in files) if files else None
            hh[arm] = {"rc": rc, "n_dumps": len(files), "nonfinite_values": nf}
            finite.append(None if rc is None or not files else (rc == 0 and nf == 0))
        c, u = a.corr / f"fa_{hn}_C", a.corr / f"fa_{hn}_Bunset"
        cf = {f.name for f in c.glob("out_*.bin")} if c.is_dir() else set()
        uf = {f.name for f in u.glob("out_*.bin")} if u.is_dir() else set()
        if hh["C"]["rc"] is None or hh["Bunset"]["rc"] is None or not cf:
            same = None; diff = None
        else:
            diff = sorted(n for n in cf | uf if n not in cf or n not in uf or (c / n).read_bytes() != (u / n).read_bytes())
            same = hh["C"]["rc"] == 0 and hh["Bunset"]["rc"] == 0 and not diff
        hh["unset_vs_C"] = {"bit_identical": same, "n_compared": len(cf), "not_identical": diff[:20] if diff else diff}
        ident.append(same)
        harness[hn] = hh
    R["harness"] = harness
    R["fa_unset_bit_identical"] = None if any(x is None for x in ident) else all(ident)
    R["finite_all_dumps"] = None if any(x is None for x in finite) else all(finite)
    R["gap"] = "the 84-case overflow harness has no path on disk; 'finite across all harness dumps' stands in for its 10 NaN cases"
    a.out.write_text(json.dumps(R, indent=1))
    print(json.dumps({k: R[k] for k in ("knob_strings_proof", "tbo_fa_100", "fa_unset_bit_identical", "finite_all_dumps")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
