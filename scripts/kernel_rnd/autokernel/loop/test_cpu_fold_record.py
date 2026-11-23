import json
import sqlite3

import pytest

from . import cpu_fold_record as fr
from . import cpu_keep_bundle as ckb

CMP = {"effect_pct": 4.5, "noise_floor_pct": 5.5, "floor_unit": "process", "recipe": "r1",
       "recipe_hash": "h1", "residency": {"contention": "unproven"},
       "measurement_plan": {"estimator": "median-ratio;paired-bootstrap-p95.v1"}}


def mkdb(tmp_path, rows):
    p = tmp_path / "experiments.db"
    c = sqlite3.connect(p)
    c.execute("create table experiments (attempt_id text, mechanism_id text, status text, "
              "recorded_at text, payload text)")
    for i, (aid, mech, status, cmp) in enumerate(rows):
        c.execute("insert into experiments values (?,?,?,?,?)",
                  (aid, mech, status, f"2026-01-0{i + 1}", json.dumps({"comparison": cmp})))
    c.commit()
    c.close()
    return p


def test_manifest_prefixes():
    text = "x attempt_id `e4ab8a122c40...` y attempt_id `86a2875125f5...` e4ab8a122c40..."
    assert fr.attempt_prefixes_in_manifest(text) == ["86a2875125f5", "e4ab8a122c40"]


def test_derive_groups_by_recipe_and_stays_provisional(tmp_path):
    db = mkdb(tmp_path, [("a" * 64, "m1", "kept", CMP), ("b" * 64, "m2", "kept", CMP),
                         ("c" * 64, "m3", "kept", {**CMP, "recipe": "r2", "recipe_hash": "h2"}),
                         ("d" * 64, "m4", "rejected", CMP)])
    out = fr.derive(db, ["aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc"],
                    champion_of_record="c" * 40, tip="t" * 40)
    assert [b["recipe"] for b in out] == ["r1.h1", "r2.h2"]
    assert len(out[0]["keeps"]) == 2 and out[0]["compounded_bench_pct"] == pytest.approx(
        (1.045 ** 2 - 1) * 100)
    k = out[0]["keeps"][0]
    assert (k["magnitude"], k["contention_label"], k["floor_pct"]) == ("provisional", "unproven", 5.5)
    assert k["statistic"].startswith("median-ratio")


def test_unmatched_nonkept_and_missing_fields_refuse(tmp_path):
    db = mkdb(tmp_path, [("a" * 64, "m1", "kept", CMP), ("d" * 64, "m4", "rejected", CMP),
                         ("e" * 64, "m5", "kept", {**CMP, "noise_floor_pct": None}),
                         ("f" * 64, "m6", "kept", {**CMP, "residency": {}})])
    for prefix in ("zzzzzzzzzzzz", "dddddddddddd", "eeeeeeeeeeee", "ffffffffffff"):
        with pytest.raises(ckb.CpuBundleError):
            fr.derive(db, [prefix], champion_of_record="c" * 40, tip="t" * 40)


def test_record_writes_readable_per_recipe_files(tmp_path):
    db = mkdb(tmp_path, [("a" * 64, "m1", "kept", CMP)])
    (path,) = fr.record(db, ["aaaaaaaaaaaa"], tmp_path / "out", champion_of_record="c" * 40,
                        tip="t" * 40)
    assert path.name == "accumulator-bundle.cpu.r1.h1.json" and ckb.read(path)["recipe"] == "r1.h1"


def test_unproven_contention_never_settles():
    with pytest.raises(ckb.CpuBundleError):
        ckb.validate_keep({"keep_id": "k", "effect_pct": 9.0, "magnitude": "settled",
                           "contention_label": "unproven", "statistic": "arm_level_permutation",
                           "floor_pct": 1.0, "floor_unit": "arm"})


def test_skipped_collects_foreign_prefixes(tmp_path):
    db = mkdb(tmp_path, [("a" * 64, "m1", "kept", CMP)])
    skipped = []
    out = fr.derive(db, ["aaaaaaaaaaaa", "fedcba987654"], champion_of_record="c" * 40,
                    tip="t" * 40, skipped=skipped)
    assert skipped == ["fedcba987654"] and len(out[0]["keeps"]) == 1


def test_same_recipe_name_with_two_hashes_does_not_collide(tmp_path):
    db = mkdb(tmp_path, [("a" * 64, "m1", "kept", CMP),
                         ("b" * 64, "m2", "kept", {**CMP, "recipe_hash": "h2xxxxxxxx"})])
    paths = fr.record(db, ["aaaaaaaaaaaa", "bbbbbbbbbbbb"], tmp_path / "o",
                      champion_of_record="c" * 40, tip="t" * 40)
    assert len({p.name for p in paths}) == 2 and len(list((tmp_path / "o").glob("*.json"))) == 2
