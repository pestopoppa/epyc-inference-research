"""No-inference tests for ruler_local (fake whitespace tokenizer, tiny fake essay dir)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.benchmark.yarn_gate import cells, ruler_local as rl  # noqa: E402

TARGET = 2000


def count(t: str) -> int:
    return len(t.split())


@pytest.fixture()
def essays(tmp_path, monkeypatch):
    d = tmp_path / "essays"
    d.mkdir()
    sents = " ".join(f"Sentence number {i} talks about startups and founders building things people want."
                     for i in range(300))
    (d / "b.txt").write_text(sents + "\n")
    (d / "a.txt").write_text("Apples grow slowly. Great companies begin small. Writing clarifies thinking.\n")
    monkeypatch.setattr(rl, "ESSAY_DIR", str(d))
    return d


def build(task, n=3, **kw):
    return rl.build_cell(task, TARGET, n, count=count, **kw)


@pytest.mark.parametrize("task", rl.TASKS)
def test_deterministic_and_valid(essays, task):
    a, b = build(task), build(task)
    assert a == b
    for it in a:
        cells.validate_item(it)
        assert it["suite"] == "ruler" and it["scorer"] == "string_match_all" and it["thinking"] is False
        assert it["context_id"] == it["item_id"] and it["nominal_len"] == TARGET
        assert it["pins"]["deviations"] and it["pins"]["ruler_sha"].startswith("ab17b78")
    assert len({i["messages"][0]["content"] for i in a}) == len(a)
    assert build(task, seed=1) != a


@pytest.mark.parametrize("task", rl.TASKS)
def test_gold_in_context_query_and_sizing(essays, task):
    for it in build(task):
        text = it["messages"][0]["content"]
        key = it["pins"]["query_key"]
        assert text.rstrip().endswith("provided text is") or text.rstrip().endswith("provided text are")
        assert f"for {key} mentioned in the provided text" in text
        for g in it["gold"]:
            assert text.count(f"for {key} is: {g}.") == 1
        n = count(text)
        assert 0.95 * TARGET <= n <= TARGET, n
        assert it["real_n_offline"] == n and it["n_source"] == "model_tokenizer_json"
        # context first, question last
        assert text.index("hidden within") < text.index(it["gold"][0]) < text.rindex("What ")


def test_ids_and_values(essays):
    mk2 = build("niah_multikey_2")
    assert mk2[1]["item_id"] == "ruler_mk_2@1k#0001" and mk2[1]["cell"] == "ruler_mk_2@1k"
    assert all(len(i["gold"]) == 1 for i in mk2)
    assert "-" in mk2[0]["pins"]["query_key"] and mk2[0]["gold"][0].isdigit()
    mk3 = build("niah_multikey_3")
    assert len(mk3[0]["gold"][0]) == 36 and len(mk3[0]["pins"]["query_key"]) == 36
    mv = build("niah_multivalue")
    assert all(len(i["gold"]) == 4 and len(set(i["gold"])) == 4 for i in mv)
    assert "essay_sha256" in mv[0]["pins"] and "essay_sha256" not in mk2[0]["pins"]
    assert "What are all the special magic numbers for" in mv[0]["messages"][0]["content"]
    assert "What is the special magic number for" in mk2[0]["messages"][0]["content"]
    assert 64 == len(mv[0]["pins"]["essay_sha256"])


@pytest.mark.parametrize("task", rl.TASKS)
def test_scorer(essays, task):
    it = build(task, 1)[0]
    full = "The values are " + ", ".join(it["gold"])
    assert cells.score(it, full, model="m")["score"] == 1.0
    assert cells.score(it, "no idea", model="m")["score"] == 0.0
    if len(it["gold"]) > 1:
        assert 0.0 < cells.score(it, it["gold"][0], model="m")["score"] < 1.0


def test_unknown_task(essays):
    with pytest.raises(ValueError):
        rl.build_cell("niah_single_1", TARGET, 1, count=count)


def test_available_names_missing_path(monkeypatch, tmp_path):
    monkeypatch.setattr(rl, "ESSAY_DIR", str(tmp_path / "nope"))
    ok, why = rl.available()
    assert not ok and str(tmp_path / "nope") in why
    monkeypatch.setattr(rl, "RULER_REPO", str(tmp_path / "norepo"))
    ok, why = rl.available()
    assert not ok and str(tmp_path / "norepo") in why


def test_real_tokenizer_smoke():
    ok, why = rl.available()
    if not ok:
        pytest.skip(why)
    t0 = time.time()
    it = rl.build_cell("niah_multikey_2", 32768, 1)[0]
    dt = time.time() - t0
    cells.validate_item(it)
    assert 0.95 * 32768 <= it["real_n_offline"] <= 32768
    assert it["pins"]["tokenizer_sha256"] == cells.DEFAULT_TOKENIZER_SHA256
    assert dt < 60, dt
