#!/usr/bin/env python3
"""Offline tests for CJ-1d GPQA sampling and its belief-kernel projection."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import cj_gpqa_sample as cj
import v7_quality_gate_beliefs as beliefs
import v7_quality_gate_runner as runner

COT = ("\n\nReason step by step, then give your final answer on its own last "
       "line in the form: ANSWER: <letter>")


def _item(i: int) -> dict:
    return {
        "id": f"gpqa_diamond_cot_{hashlib.sha256(str(i).encode()).hexdigest()[:12]}",
        "suite": "gpqa_diamond_cot",
        "prompt": f"synthetic question {i}\n\nA) a\nB) b\nC) c\nD) d{COT}",
        "context": "",
        "expected": "ABCD"[i % 4],
        "scoring": [],
        "image_path": "",
        "tier": 1 + i % 3,
        "scoring_method": "multiple_choice",
        "scoring_config": {},
    }


@pytest.fixture
def population() -> list[dict]:
    return [_item(i) for i in range(cj.POPULATION_SIZE)]


def _write(path: Path, items: list[dict]) -> Path:
    path.write_text(json.dumps({"suites": {cj.SUITE: items}}))
    return path


def test_full_population_is_default_and_order_independent(tmp_path, population):
    a = cj.build_manifest(_write(tmp_path / "a.json", population), expected_ids_sha256=None)
    b = cj.build_manifest(_write(tmp_path / "b.json", list(reversed(population))),
                          expected_ids_sha256=None)
    assert cj.DEFAULT_N == 198
    assert a["suites"][cj.SUITE] == b["suites"][cj.SUITE]
    ident = a["sample"]
    assert ident["n"] == ident["n_population"] == 198
    assert ident["full_population"] is True and ident["seed_effective"] is False
    assert ident["sample_ids_sha256"] == ident["population_ids_sha256"]
    # file-level digest differs (different byte order), id-set digest does not
    assert a["sample"]["population_sha256"] != b["sample"]["population_sha256"]
    assert a["sample"]["population_ids_sha256"] == b["sample"]["population_ids_sha256"]


def test_seeded_subset_is_deterministic_and_seed_sensitive(population):
    s1 = cj.select_sample(population, 100, 42)
    s2 = cj.select_sample(list(reversed(population)), 100, 42)
    s3 = cj.select_sample(population, 100, 7)
    assert [x["id"] for x in s1] == [x["id"] for x in s2]
    assert len(s1) == 100 and len({x["id"] for x in s1}) == 100
    assert {x["id"] for x in s1} != {x["id"] for x in s3}
    assert [x["id"] for x in s1] == sorted(x["id"] for x in s1)


@pytest.mark.parametrize("n", [0, 199, -1, True, 1.5])
def test_bad_n_refused(population, n):
    with pytest.raises(cj.SampleError):
        cj.select_sample(population, n, 42)


def test_letter_only_framing_refused(tmp_path, population):
    population[5]["prompt"] = "q\n\nA) a\n\nAnswer with the letter only (A, B, C, or D)."
    with pytest.raises(cj.SampleError, match="CoT framing"):
        cj.load_population(_write(tmp_path / "p.json", population), expected_ids_sha256=None)


@pytest.mark.parametrize("mutate, msg", [
    (lambda p: p.pop(), "198"),
    (lambda p: p.__setitem__(1, dict(p[0])), "unique"),
    (lambda p: p[3].__setitem__("expected", "E"), "A-D"),
    (lambda p: p[3].__setitem__("suite", "gpqa_diamond"), "suite"),
    (lambda p: p[3].__setitem__("scoring_method", "judge"), "multiple_choice"),
])
def test_malformed_population_refused(tmp_path, population, mutate, msg):
    mutate(population)
    with pytest.raises(cj.SampleError, match=msg):
        cj.load_population(_write(tmp_path / "p.json", population), expected_ids_sha256=None)


def test_unknown_population_id_set_refused_by_default(tmp_path, population):
    with pytest.raises(cj.SampleError, match="population id set"):
        cj.load_population(_write(tmp_path / "p.json", population))


def test_manifest_replays_through_runner_and_cli_refuses_unknown_population(
        tmp_path, population):
    src = _write(tmp_path / "pop.json", population)
    out = tmp_path / "manifest.json"
    out.write_text(json.dumps(cj.build_manifest(src, n=50, seed=42,
                                                expected_ids_sha256=None)))
    replay = runner.load_questions(cj.SUITE, n=0, seed=0, questions_in=out)
    assert len(replay) == 50 and all(q["suite"] == cj.SUITE for q in replay)
    # The CLI always enforces the known Diamond id set: a synthetic one is refused.
    assert cj.main(["--population", str(src), "--out", str(tmp_path / "x.json")]) == 2
    assert not (tmp_path / "x.json").exists()


def _result(pinned: Path, n: int = 198, correct: int = 161) -> dict:
    return {
        "meta": {"arm": "qwen3.8-27b", "timestamp": "2026-09-16T00:00:00+00:00",
                 "seed": 42, "repeats": 1, "endpoint": "chat",
                 "questions_pinned": str(pinned)},
        "suites": [{"suite": cj.SUITE, "n": n, "correct": correct,
                    "truncated": 0, "errors": 0}],
    }


def test_belief_row_carries_sample_identity(tmp_path, population):
    manifest = cj.build_manifest(_write(tmp_path / "pop.json", population),
                                 n=120, seed=42, expected_ids_sha256=None)
    pinned = tmp_path / "manifest.json"
    pinned.write_text(json.dumps(manifest))
    rows = beliefs.attach_accuracy_beliefs(
        _result(pinned, n=120, correct=90), output_path=tmp_path / "result.json",
        category="CANDIDATE", runner_source_sha256="0" * 64,
        host="127.0.0.1", port=8080)
    sample = rows[0]["extra"]["prompt_set"]["sample"]
    assert sample["schema"] == cj.SAMPLE_SCHEMA
    assert sample["n"] == 120 and sample["n_population"] == 198
    assert sample["seed"] == 42 and sample["seed_effective"] is True
    assert sample["sample_ids_sha256"] == manifest["sample"]["sample_ids_sha256"]
    assert sample["manifest_item_count"] == 120
    assert rows[0]["reps"] == 120  # scored n still comes from the summary


def test_belief_row_without_sample_block_says_none(tmp_path, population):
    pinned = _write(tmp_path / "legacy_pin.json", population)
    rows = beliefs.attach_accuracy_beliefs(
        _result(pinned), output_path=tmp_path / "result.json",
        category="BASELINE", runner_source_sha256="0" * 64,
        host="127.0.0.1", port=8080)
    assert rows[0]["extra"]["prompt_set"]["sample"] is None
    assert rows[0]["extra"]["prompt_set"]["sha256"]


def test_sample_block_for_other_suite_not_attributed(tmp_path, population):
    manifest = cj.build_manifest(_write(tmp_path / "pop.json", population),
                                 expected_ids_sha256=None)
    manifest["sample"]["suite"] = "mmlu_pro"
    pinned = tmp_path / "manifest.json"
    pinned.write_text(json.dumps(manifest))
    rows = beliefs.attach_accuracy_beliefs(
        _result(pinned), output_path=tmp_path / "result.json",
        category="BASELINE", runner_source_sha256="0" * 64,
        host="127.0.0.1", port=8080)
    assert rows[0]["extra"]["prompt_set"]["sample"] is None


REAL_POPULATION = cj.DEFAULT_POPULATION


@pytest.mark.skipif(not REAL_POPULATION.is_file(), reason="host population pin absent")
def test_real_population_pin_matches_known_digest():
    items, _ = cj.load_population(REAL_POPULATION)
    assert len(items) == 198
    manifest = cj.build_manifest(REAL_POPULATION)
    assert manifest["sample"]["gold_counts"] == {"A": 38, "B": 61, "C": 62, "D": 37}
    assert manifest["sample"]["tier_counts"] == {"1": 15, "2": 43, "3": 140}
