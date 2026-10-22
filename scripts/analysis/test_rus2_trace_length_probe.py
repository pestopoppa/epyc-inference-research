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


def _fixture_tree(tmp_path):
    d = tmp_path / "eval"; d.mkdir()
    lines = []
    for i in range(24):  # 24 questions, 2 samples each: a cell with >= 20 questions
        for k, (tok, ok) in enumerate(((10 + i, True), (50 + i, i % 2 == 0))):
            lines.append(json.dumps({"suite": "s", "question_id": f"q{i}", "timestamp": f"t{k}", "role_results": {
                "r:direct": {"role": "r", "tokens_generated": tok, "passed": ok, "answer": "a", "error": None}}}))
    (d / "x.jsonl").write_text("\n".join(lines) + "\n")
    (d / "seen_questions.jsonl").write_text("{}\n")
    return tmp_path


def test_result_carries_timestamp_input_digests_and_script_hash(tmp_path, monkeypatch):
    import hashlib
    root = _fixture_tree(tmp_path)
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", ["x", "--results-dir", str(root), "--out", str(out)])
    assert m.main() == 0
    res = json.loads(out.read_text())
    assert res["generated_at"].endswith("Z")
    assert [d["file"] for d in res["input_digests"]] == ["eval/x.jsonl"]  # seen_questions excluded, as in load()
    assert res["input_digests"][0]["sha256"] == hashlib.sha256((root / "eval/x.jsonl").read_bytes()).hexdigest()
    assert res["script_sha256"] == hashlib.sha256(pathlib.Path(m.__file__).read_bytes()).hexdigest()
    assert res["rows"] and "files_scanned" in res  # legacy keys intact


def test_belief_out_writes_sidecar_through_root_adapter(tmp_path, monkeypatch):
    import pytest
    root = pathlib.Path("/mnt/raid0/llm/worktrees/root-pbe-3c4ca599")
    if not (root / "scripts/vidya/adapters/rus2_probe.py").is_file():
        pytest.skip("no root checkout with the rus2 writer")
    tree = _fixture_tree(tmp_path)
    out, bdir = tmp_path / "res.json", tmp_path / "b"
    monkeypatch.setattr(sys, "argv", ["x", "--results-dir", str(tree), "--out", str(out), "--belief-out", str(bdir),
                                      "--epyc-root", str(root)])
    assert m.main() == 0
    rows = [json.loads(l) for l in (bdir / "rus2_probe.beliefs.jsonl").read_text().splitlines()]
    assert rows and all(r["protocol_id"] == "" for r in rows)
    assert rows[0]["emitted_at"] == json.loads(out.read_text())["generated_at"]


def test_belief_refusal_is_loud(tmp_path, monkeypatch, capsys):
    tree = _fixture_tree(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--results-dir", str(tree), "--out", str(tmp_path / "r.json"),
                                      "--belief-out", str(tmp_path / "b"), "--epyc-root", str(tmp_path / "nope")])
    assert m.main() == 3
    assert "BELIEF CAPTURE FAILED [rus2]" in capsys.readouterr().err
