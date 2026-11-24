"""RSL-R2: census_inbox installs/removes BWS-*.md inbox stubs, idempotently, with dedupe."""
from __future__ import annotations

import copy
import json

import pytest

from ..controller import inbox as ctl_inbox
from . import census_inbox as ci
from . import census_update as cu
from . import journal_keys as jk


def surf(path, op, status="open", covered="", measured=True, saving=1.5):
    return {"path": path, "op": op, "device": "mi210", "status": status, "covered_by": covered, "ours_measured": measured,
            "ours_pct": 40.0, "share_of_step": 0.1, "step_ms": 45.7, "saving_ms_per_step": saving, "ref": None,
            "census_rows": [], "source": "fixture", "step_source": "fixture"}


@pytest.fixture
def census():
    return {"target_pct": 88.0, "surfaces": [
        surf("27b-df2-mi210", "lm_head"), surf("a3b-mtp-mi210", "gdn_proj"), surf("a3b-mtp-mi210", "fullattn_proj"),
        surf("a3b-mtp-mi210", "owned", covered="AK-X", status="covered"),   # has an owner: never a stub
        surf("a3b-mtp-mi210", "placeholder", measured=False, status="needs_measurement", saving=None)]}


@pytest.fixture
def box(tmp_path):
    (tmp_path / "inbox").mkdir()
    (tmp_path / "spool").mkdir()
    return tmp_path / "inbox", tmp_path / "spool"


def run(census, inbox, spool):
    return ci.apply(ci.plan(census, inbox, spool), inbox)


def names(inbox):
    return sorted(p.name for p in inbox.iterdir())


def test_fixture_run_writes_n_stubs_and_rerun_writes_zero(census, box):
    inbox, spool = box
    first = run(census, inbox, spool)
    want = sorted(cu.surface_seed_id(s) + ".md" for s in cu.seedable_surfaces(census))
    assert first["written"] == len(want) == 3 and names(inbox) == want
    assert all(n.startswith("BWS-") for n in names(inbox))
    assert run(census, inbox, spool) == {"written": 0, "removed": 0, "skipped": {}}


def test_stub_body_is_what_write_stubs_writes(census, box, tmp_path):
    inbox, spool = box
    run(census, inbox, spool)
    cu.write_stubs(str(tmp_path / "preview"), census)
    for n in names(inbox):
        body = (inbox / n).read_text()
        assert jk.KEY_RE.sub("", body).replace("\n\n\n", "\n\n") == (tmp_path / "preview" / n).read_text()


def test_stub_carries_a_valid_journal_key_and_shape_class(census, box):
    inbox, spool = box
    census["surfaces"][0]["op"] = "lm_head 248320x5120"
    census["surfaces"][0]["path"] = "27b-df2-mi210"
    run(census, inbox, spool)
    sid = cu.surface_seed_id(census["surfaces"][0])
    key = jk.parse_key((inbox / (sid + ".md")).read_text())
    assert jk.problems(key) == [] and key["op_family"] == "lm-head-gemv" and key["device"] == "gpu"
    assert key["shape_class"] == {"m": 248320, "k": 5120}
    assert all(jk.problems(jk.parse_key(p.read_text())) == [] for p in inbox.glob("BWS-*.md"))
    assert ci.seed_shape_class(inbox, sid) == \
        {"op_family": "MUL_MAT", "dims": {"k": [5120], "m": [248320]}}


def test_surface_that_turns_covered_or_ok_loses_its_stub(census, box):
    inbox, spool = box
    run(census, inbox, spool)
    c2 = copy.deepcopy(census)
    c2["surfaces"][0].update(status="covered", covered_by="AK-LM-HEAD")
    c2["surfaces"][1].update(status="ok", ours_pct=90.0, saving_ms_per_step=0.0)
    assert run(c2, inbox, spool)["removed"] == 2
    assert names(inbox) == [cu.surface_seed_id(c2["surfaces"][2]) + ".md"]
    assert run(c2, inbox, spool) == {"written": 0, "removed": 0, "skipped": {}}


def test_dedupes_on_target_surface_against_existing_inbox(census, box):
    inbox, spool = box
    (inbox / "26-bw-surfaces-20261124.md").write_text(
        "## seed\nEffort: M. target_surface: 27b-df2-mi210/lm_head\n")
    (inbox / "operator-note.md").write_text("see target_surface: a3b-mtp-mi210/gdn_proj and more\n")  # prose: not a match
    rep = run(census, inbox, spool)
    assert rep["skipped"] == {"BWS-27B-DF2-GPU-LM-HEAD": "inbox"} and rep["written"] == 2
    assert "BWS-27B-DF2-GPU-LM-HEAD.md" not in names(inbox)


def test_stub_made_redundant_by_a_later_inbox_seed_is_removed(census, box):
    inbox, spool = box
    run(census, inbox, spool)
    (inbox / "26-seed.md").write_text("target_surface: 27b-df2-mi210/lm_head\n")
    assert run(census, inbox, spool)["removed"] == 1
    assert "BWS-27B-DF2-GPU-LM-HEAD.md" not in names(inbox)


def test_dedupes_against_candidate_spool_hypothesis_id(census, box):
    inbox, spool = box
    (spool / "cand-x-abc.json").write_text(json.dumps({"hypothesis_id": "BWS-A3B-MTP-GPU-GDN-PROJ-tiling", "id": "cand-x"}))
    (spool / "cand-y.json").write_text(json.dumps({"hypothesis_id": "a3b-mtp-mi210/fullattn_proj"}))
    (spool / "broken.json").write_text("{not json")
    rep = run(census, inbox, spool)
    assert rep["skipped"] == {"BWS-A3B-MTP-GPU-GDN-PROJ": "spool", "BWS-A3B-MTP-GPU-FULLATTN-PROJ": "spool"}
    assert names(inbox) == ["BWS-27B-DF2-GPU-LM-HEAD.md"]


def test_only_bws_files_are_ever_removed_and_unknown_surfaces_are_left(census, box):
    inbox, spool = box
    (inbox / "01-keep.md").write_text("x")
    (inbox / "BWS-NOT-IN-CENSUS.md").write_text("hand-made")
    run(census, inbox, spool)
    c2 = copy.deepcopy(census)
    for s in c2["surfaces"]:
        s.update(status="ok", ours_pct=95.0)
    run(c2, inbox, spool)
    assert names(inbox) == ["01-keep.md", "BWS-NOT-IN-CENSUS.md"]


def test_missing_inbox_is_an_error_not_a_mkdir(census, tmp_path):
    with pytest.raises(FileNotFoundError):
        ci.apply({"install": {"BWS-X.md": "x"}, "remove": [], "skipped": {}}, tmp_path / "nope")
    assert not (tmp_path / "nope").exists()


def test_cli_sync_dry_run_and_the_planner_reader_sees_the_stub(census, tmp_path, capsys):
    store = tmp_path / "store"
    (store / "inbox").mkdir(parents=True)
    cp = tmp_path / "census.json"
    cp.write_text(json.dumps(census))
    assert ci.main(["--census", str(cp), "--store", str(store), "--spool", str(tmp_path / "none"), "--dry-run"]) == 0
    assert list((store / "inbox").iterdir()) == []
    assert ci.main(["--census", str(cp), "--store", str(store), "--spool", str(tmp_path / "none")]) == 0
    assert '"written": 3' in capsys.readouterr().out.splitlines()[-1]
    seen = ctl_inbox.read_inbox(store / "inbox", note=lambda m: pytest.fail(m))
    assert len(seen) == 3 and all("BWS-" in t and "target_surface:" in t for t in seen)
    assert ci.main(["--census", str(tmp_path / "absent.json"), "--store", str(store)]) == 1
