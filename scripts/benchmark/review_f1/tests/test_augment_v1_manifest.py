"""Offline invariants for the EV-13b Augment-v1 pin manifest (NO network, NO inference).

The golden comments are unlicensed upstream and this repo is public, so the
committed artifact is ``data/review_f1/augment_v1_manifest.json`` (pins +
counts), not the data. The manifest invariants always run; the fetched
``golden_set.json`` is additionally checked against the manifest when present
(``fetch_augment_v1.py`` has been run on this host).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import assemble_golden_set  # noqa: E402
import conftest  # noqa: E402
import fetch_augment_v1  # noqa: E402

MANIFEST = conftest.RESEARCH_ROOT / "data" / "review_f1" / "augment_v1_manifest.json"
FETCHED = conftest.RESEARCH_ROOT / "data" / "external" / "review_f1" / "augment_v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _m() -> dict:
    return json.loads(MANIFEST.read_text())


def test_upstream_pinned_by_commit_and_digest():
    m = _m()
    assert m["upstream"]["commit"] == fetch_augment_v1.UPSTREAM_COMMIT
    assert _HEX40.match(m["upstream"]["commit"])
    assert sorted(m["upstream"]["files"]) == sorted(fetch_augment_v1.BENCHES)
    for f in m["upstream"]["files"].values():
        assert _HEX64.match(f["sha256"])
        assert fetch_augment_v1.UPSTREAM_COMMIT in f["url"]


def test_fifty_prs_ten_per_bench_unique():
    prs = _m()["prs"]
    assert len(prs) == 50
    assert Counter(p["bench"] for p in prs) == {b: 10 for b in fetch_augment_v1.BENCHES}
    assert len({(p["bench"], p["number"]) for p in prs}) == 50
    for p in prs:
        assert _HEX40.match(p["head_sha"]) and _HEX40.match(p["base_sha"])
        assert _HEX64.match(p["diff_sha256"])
        assert p["diff_path"] == f"diffs/{p['bench']}/pr-{p['number']}.diff"


def test_comment_counts_consistent():
    """Upstream data holds 137 comments (97 scored) — NOT the 145 its README
    table claims (the table was never updated; blobs identical since the first
    data commit). Pinned here so a silent upstream change is caught."""
    m = _m()
    prs = m["prs"]
    for bench, f in m["upstream"]["files"].items():
        assert f["n_prs"] == 10
        assert f["n_comments"] == sum(p["n_comments"] for p in prs if p["bench"] == bench)
    total = sum(p["n_comments"] for p in prs)
    low = sum(p["n_low"] for p in prs)
    g = m["golden_set"]
    assert (g["n_cases"], g["n_golden_total"], g["n_golden_scored"]) == (50, 137, 97)
    assert total == g["n_golden_total"] and total - low == g["n_golden_scored"]
    assert g["schema_version"] == assemble_golden_set.SCHEMA_VERSION
    assert sum(m["anchor_check"].values()) == total


def test_identifier_tokens_extracts_code_names():
    toks = fetch_augment_v1.identifier_tokens(
        "get_item_key assumes a numeric key; calling math.floor on `order_by` or getFoo() fails"
    )
    for t in ("get_item_key", "math.floor", "order_by", "getFoo"):
        assert t in toks, (t, toks)
    assert fetch_augment_v1.identifier_tokens("Case sensitivity bypass in email blacklist") == []


def test_fetched_golden_set_matches_manifest():
    gs = FETCHED / "golden_set.json"
    if not gs.exists():
        print("SKIP fetched golden_set.json absent (run fetch_augment_v1.py)")
        return
    m = _m()
    g = json.loads(gs.read_text())
    for k, v in m["golden_set"].items():
        assert g[k] == v, k
    recomputed = hashlib.sha256(
        json.dumps(g["cases"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recomputed == g["checksum"]
    pins = {f"{p['bench']}__pr-{p['number']}": p for p in m["prs"]}
    assert set(pins) == {c["case_id"] for c in g["cases"]}
    for c in g["cases"]:
        p = pins[c["case_id"]]
        assert c["pr_ref"]["diff_path"] == p["diff_path"]
        diff = (FETCHED / p["diff_path"]).read_bytes()
        assert hashlib.sha256(diff).hexdigest() == p["diff_sha256"], c["case_id"]
        assert len(c["golden_findings"]) == p["n_comments"]
        assert {f["severity"] for f in c["golden_findings"]} <= {"low", "medium", "high", "critical"}


# --------------------------------------------------------------------------- #
def _run_standalone() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
