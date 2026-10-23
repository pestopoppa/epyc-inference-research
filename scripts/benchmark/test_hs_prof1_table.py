import importlib.util
import json
import sys
from pathlib import Path

import pytest

_S = importlib.util.spec_from_file_location("hs_prof1_profile_arms_t", Path(__file__).parent / "hs_prof1_profile_arms.py")
M = importlib.util.module_from_spec(_S); sys.modules["hs_prof1_profile_arms_t"] = M; _S.loader.exec_module(M)


def _w(d, arm, rows):
    (d / f"scored_{arm}.json").write_text(json.dumps({"summary": {"judge_model": "j"}, "per_question": rows}))


def test_table_paired_deltas(tmp_path):
    _w(tmp_path, "pref_current", [{"question_id": "q1", "ability": "preference_following", "score": 0.5},
                                  {"question_id": "q2", "ability": "preference_following", "score": 1.0},
                                  {"question_id": "q3", "ability": "instruction_following", "score": 0.0}])
    _w(tmp_path, "admission", [{"question_id": "q1", "ability": "preference_following", "score": 1.0},
                               {"question_id": "q2", "ability": "preference_following", "score": 1.0},
                               {"question_id": "q4", "ability": "instruction_following", "score": 1.0}])
    t = M.table(tmp_path)
    pf = t["per_ability"]["preference_following"]
    assert t["n_shared_questions"] == 2 and t["only_in_one_arm"] == 2 and "instruction_following" not in t["per_ability"]
    assert pf["mean_delta"] == 0.25 and pf["admission_wins"] == 1 and pf["ties"] == 1 and pf["n"] == 2


def test_table_refuses_missing_arm_or_no_overlap(tmp_path):
    with pytest.raises(SystemExit):
        M.table(tmp_path)
    _w(tmp_path, "pref_current", [{"question_id": "a", "ability": "x", "score": 1}])
    _w(tmp_path, "admission", [{"question_id": "b", "ability": "x", "score": 1}])
    with pytest.raises(SystemExit, match="no question ids"):
        M.table(tmp_path)


def _pair(tmp_path):
    _w(tmp_path, "pref_current", [{"question_id": "q1", "ability": "preference_following", "score": 0.5}])
    _w(tmp_path, "admission", [{"question_id": "q1", "ability": "preference_following", "score": 1.0}])


def test_table_belief_flag_refuses_loudly_without_epyc_root(tmp_path, monkeypatch, capsys):
    """VB-HSPROF1-WRITE: the table is still written, the capture failure is not silent (exit 3)."""
    monkeypatch.delenv("EPYC_ROOT", raising=False)
    _pair(tmp_path)
    out = tmp_path / "table.json"
    rc = M.main(["table", "--out-dir", str(tmp_path), "--out", str(out), "--belief-measurements"])
    assert rc == 3 and out.is_file()
    assert "BELIEF CAPTURE FAILED" in capsys.readouterr().err


def test_table_belief_flag_calls_the_root_writer_with_the_table(tmp_path, monkeypatch):
    adapters = tmp_path / "root" / "scripts" / "vidya" / "adapters"
    adapters.mkdir(parents=True)
    calls = tmp_path / "calls.json"
    (adapters / "hs_prof1_profile_arms.py").write_text(
        "import json\n"
        "def write_belief_measurements(out_dir, **kw):\n"
        f"    open({str(calls)!r}, 'w').write(json.dumps({{'out_dir': str(out_dir), **{{k: str(v) if k == 'table_path' else v for k, v in kw.items()}}}}))\n"
        "    return 'sidecar'\n")
    monkeypatch.setenv("EPYC_ROOT", str(tmp_path / "root"))
    run = tmp_path / "run"
    run.mkdir()
    _pair(run)
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}")
    rc = M.main(["table", "--out-dir", str(run), "--out", str(run / "table.json"),
                 "--belief-measurements", "--reader-recipe", str(recipe)])
    assert rc == 0
    got = json.loads(calls.read_text())
    assert got["out_dir"] == str(run) and got["table_path"] == str(run / "table.json")
    assert got["producer"] == "hs_prof1_profile_arms.py table"
    assert got["reader_pins"]["recipe_path"] == str(recipe.resolve())
    assert len(got["reader_pins"]["recipe_sha256"]) == 64
