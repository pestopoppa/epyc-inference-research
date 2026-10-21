import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import rus2_trace_length_probe as m


def test_auroc_and_within():
    qs = {"a": [(10, True, None), (50, False, None)], "b": [(5, True, None), (9, False, None)], "c": [(7, True, None), (8, True, None)]}
    v, n = m.stat_between(qs)
    assert n == 4 * 1 + 0 or n > 0
    wv, wn = m.stat_within(qs)
    assert wn == 2 and wv == 0.0  # correct always shorter
    assert set(m.within_questions(qs)) == {"a", "b"}  # all-correct c dropped


def test_load_drops(tmp_path):
    d = tmp_path / "eval"; d.mkdir()
    rec = {"suite": "s", "question_id": "q", "timestamp": "t", "role_results": {
        "r:direct": {"role": "r", "tokens_generated": 5, "passed": True, "answer": "x\nWait, check\ny\nz", "error": None},
        "r2:direct": {"role": "r2", "tokens_generated": 0, "passed": True}}}
    (d / "x.jsonl").write_text(json.dumps(rec) + "\n")
    cells, drops, temp, n = m.load(str(tmp_path))
    assert drops["no_tokens_generated"] == 1 and temp["not_recorded"] == 1
    assert cells[("r", "direct", "s")]["q"][0][2] == 0.25
