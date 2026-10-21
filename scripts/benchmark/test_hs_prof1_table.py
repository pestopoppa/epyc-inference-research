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
