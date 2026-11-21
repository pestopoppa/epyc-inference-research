"""Lineage/fold manager tests on a synthetic llama.cpp-shaped repo (no real tree touched)."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts.kernel_rnd.autokernel.lineage import ledger as L
from scripts.kernel_rnd.autokernel.lineage.__main__ import main as cli


def _g(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True,
                          capture_output=True,
                          env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                               "GIT_AUTHOR_DATE": "2026-11-01T00:00:00",
                               "GIT_COMMITTER_DATE": "2026-11-01T00:00:00",
                               "PATH": "/usr/bin:/bin"}).stdout.strip()


def _commit(repo: Path, files: dict[str, str], msg: str) -> str:
    for name, text in files.items():
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        _g(repo, "add", name)
    _g(repo, "commit", "-q", "-m", msg)
    return _g(repo, "rev-parse", "HEAD")


@pytest.fixture()
def repo(tmp_path: Path):
    r = tmp_path / "llama"
    r.mkdir()
    _g(r, "init", "-q", "-b", "prod")
    prod = _commit(r, {"src/a.c": "a\n"}, "production v10")
    _g(r, "branch", "champ")
    _g(r, "checkout", "-q", "champ")
    keep = _commit(r, {"src/k.c": "k\n", "tests/test-k.cpp": "t\n"}, "ggml: keep one (AK-KEEP-1)")
    # side branch c16: tested feature + untested feature + DNF commit
    _g(r, "checkout", "-q", "-b", "llama.cpp-experimental-c16-side-20261118")
    tested = _commit(r, {"src/f.c": "f\n"}, "ggml-cuda: GGML_HIP_FOO knob (AKX-ALL-24, default off)")
    _commit(r, {"tests/test-backend-ops.cpp": "foo\n"}, "tests: foo cases (AKX-ALL-24)")
    untested = _commit(r, {"src/u.c": "u\n"}, "ggml-cpu: untested lever (C16-NOTEST)")
    dnf = _commit(r, {"src/d.c": "d\n", "tests/test-d.cpp": "d\n"}, "sync17 fix2 (INF70-SYNC17)")
    # composite c17 carries the tested feature (cherry-picked: new sha, same patch-id)
    _g(r, "checkout", "-q", "champ")
    _g(r, "checkout", "-q", "-b", "llama.cpp-experimental-c17-20261121")
    _g(r, "cherry-pick", tested)
    _commit(r, {"tests/test-backend-ops.cpp": "foo\n"}, "tests: foo cases (AKX-ALL-24)")
    _g(r, "checkout", "-q", "champ")
    return {"repo": r, "prod": prod, "keep": keep, "tested": tested,
            "untested": untested, "dnf": dnf}


def _ledger(repo, **kw):
    dnf_text = {"kcp.md": f"| `inf70/sync17-fix2` @ `{repo['dnf'][:9]}` | **DO NOT FOLD** | x |\n"
                          "| lower case do not fold `deadbeef1` hold |\n"}
    manifest = ("| ID | what | kind | source | status | gain | presence gate | owner |\n"
                f"| V11-001 | keep one | perf | `champ` @ `{repo['keep'][:9]}` | **IN** in champion |"
                " x | `tests/test-k` passes | o |\n"
                "| V11-002 | gateless | perf | `champ` | **IN** in champion | x |  | o |\n")
    return L.build_ledger(L.git_runner(repo["repo"]), champion="champ", production=repo["prod"],
                          dnf_texts=dnf_text, manifests={"m.md": manifest}, **kw)


def test_ledger_fields_and_identity(repo):
    led = _ledger(repo)
    assert led["current_fold"] == "llama.cpp-experimental-c17-20261121"
    assert led["next_c"] == 18
    f = {x["tag"]: x for x in led["features"]}
    assert f["AK-KEEP-1"]["in_champion"] and f["AK-KEEP-1"]["manifest_ids"] == ["V11-001"]
    a = f["AKX-ALL-24"]
    assert a["in_fold"] and not a["in_champion"]
    assert len(a["commits"]) == 2 and a["has_test"]          # cherry-pick deduped by patch-id
    assert a["knobs"] == ["GGML_HIP_FOO"]
    assert not f["C16-NOTEST"]["has_test"] and not f["C16-NOTEST"]["in_fold"]
    assert f["INF70-SYNC17"]["do_not_fold"] == ["kcp.md:1"]
    assert [d["sha"] for d in led["do_not_fold"]] == [repo["dnf"][:9]]  # lower case ignored


def test_fold_list_refuses_untested_and_dnf(repo):
    folds = L.fold_list(_ledger(repo))
    assert folds["candidate"] == "C18"
    tags = [e["tag"] for e in folds["entries"]]
    assert tags == ["AKX-ALL-24"]
    entry = folds["entries"][0]
    assert entry["carried_from_current_fold"] and entry["squash_into_one_commit"]
    assert entry["tests"] == ["tests/test-backend-ops.cpp"]
    refused = {r["tag"]: r["reason"] for r in folds["refused"]}
    assert "DS41-C116" in refused["C16-NOTEST"]
    assert "DO-NOT-FOLD" in refused["INF70-SYNC17"]


def test_fold_entry_raises_without_test():
    f = L.Feature(tag="X-1", commits=[{"sha": "a" * 40, "subject": "s", "in_champion": False}])
    with pytest.raises(L.FoldRefused, match="DS41-C116"):
        L.fold_entry(f)
    f.test_files = ["tests/test-x.cpp"]
    assert L.fold_entry(f)["tests"] == ["tests/test-x.cpp"]


def test_drop_verdict_refuses(repo):
    led = _ledger(repo)
    for x in led["features"]:
        if x["tag"] == "AKX-ALL-24":
            x["gate_results"] = [{"row": "c15-x-20261118-r1", "verdict": "DROP"}]
    assert [e["tag"] for e in L.fold_list(led)["entries"]] == []


def test_presence_manifest_and_check(repo):
    led = _ledger(repo)
    folds = L.fold_list(led)
    pres = L.presence_manifest(led, folds)
    ids = [e["id"] for e in pres["entries"]]
    assert "AK-KEEP-1" in ids and "AKX-ALL-24" in ids and "V11-001" in ids
    assert any("V11-002" in d for d in pres["defects"])     # manifest row without a gate
    git = L.git_runner(repo["repo"])
    ok = L.check_presence(git, pres, "llama.cpp-experimental-c17-20261121", production=repo["prod"])
    assert ok["passed"], ok
    bad = L.check_presence(git, pres, "champ", production=repo["prod"])
    assert not bad["passed"] and bad["missing"][0]["id"] == "AKX-ALL-24"


def test_specd_rows_staged_not_published(repo, tmp_path):
    folds = L.fold_list(_ledger(repo))
    rows = L.specd_rows(folds, date="20261121", dnf_shas=["2516c9807"])
    assert all(r["schema"] == "epyc.row.v3" and r["staged_only"] and r["launcher"] is None
               for r in rows)
    assert {r["id"] for r in rows} >= {"specd-c18-census-footprint-20261121-r1",
                                       "specd-c18-loo-akx-all-24-20261121-r1"}
    paths = L.write_staged(rows, tmp_path / "staging")
    assert len(paths) == len(rows)


def test_read_only_git():
    with pytest.raises(PermissionError):
        L.git_runner("/nonexistent")("checkout", "x")


def test_deterministic_and_cli(repo, tmp_path, monkeypatch):
    assert json.dumps(_ledger(repo), sort_keys=True) == json.dumps(_ledger(repo), sort_keys=True)
    root = tmp_path / "root"
    (root / "docs/reference/kernel-manifests").mkdir(parents=True)
    out = tmp_path / "out"
    rc = cli(["generate", "--repo", str(repo["repo"]), "--root", str(root), "--out", str(out),
              "--champion", "champ", "--production", repo["prod"], "--staging", str(tmp_path / "st"), "--date", "20261121"])
    assert rc == 3                       # C16-NOTEST has no test -> refused, exit 3
    assert (out / "fold-list.md").read_text().startswith("# C18 fold list")
    assert cli(["check-fold", "--ledger", str(out / "ledger.json"), "C16-NOTEST"]) == 3
    assert cli(["check-fold", "--ledger", str(out / "ledger.json"), "AKX-ALL-24"]) == 0
