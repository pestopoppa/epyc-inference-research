"""RT-2.1 schema tests. Synthetic data only (no tap exists; RT-1 is compute-gated and file-only)."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import trace_shape as ts  # noqa: E402

HEX = "a" * 64


def manifest():
    # deepseek41-like: layer 0 main (256 experts, top-6), layer 1 DSpark (128 experts, top-6)
    return {
        "schema": ts.SCHEMA_ID, "model": "fixture", "arch": "deepseek41", "batch_size": 512,
        "layers": [{"layer": 0, "n_experts": 256, "top_k": 6, "id_dtype": "int16", "group": "main"},
                   {"layer": 1, "n_experts": 128, "top_k": 6, "id_dtype": "int16", "group": "dspark"}],
        "samples": [{"sample_id": 0, "domain": "A", "start_row": 0, "n_rows": 5},
                    {"sample_id": 1, "domain": "B", "start_row": 5, "n_rows": 2}],
        "columns": ts.COLUMNS,
        "ids": {"dtype": "int16", "shape_note": "[rows, n_layers, top_k] row-major; layers with smaller top_k pad with -1",
                "sha256": HEX},
        "rows_sha256": HEX,
    }


def data():
    rows = [  # sample 0: 2-token prefill chunk, plain decode, then a 2-token speculative block
        [0, 0, "prefill", 0, 2, None], [0, 1, "prefill", 0, 2, None],
        [0, 2, "decode", 1, 1, None],
        [0, 3, "decode", 2, 2, True], [0, 4, "decode", 2, 2, False],
        [1, 0, "prefill", 0, 2, None], [1, 1, "prefill", 0, 2, None]]
    ids = [[[200 + i, 1, 2, 3, 4, 5], [100 + i, 1, 2, 3, 4, 5]] for i in range(len(rows))]  # 200+ needs int16-safe >127
    return rows, ids


def test_good_trace_validates():
    rows, ids = data()
    assert ts.validate_manifest(manifest()) == []
    assert ts.validate_rows(manifest(), rows, ids) == []


def test_schema_file_matches_module_constants():
    sch = ts.load_schema()
    assert sch["properties"]["schema"]["const"] == ts.SCHEMA_ID
    assert sch["properties"]["columns"]["const"] == ts.COLUMNS
    assert sch["properties"]["layers"]["items"]["properties"]["id_dtype"]["const"] == "int16"
    json.dumps(sch)


@pytest.mark.parametrize("mutate,needle", [
    (lambda m: m["layers"][0].update(id_dtype="int8"), "int16"),
    (lambda m: m.update(columns=ts.COLUMNS[:4]), "columns"),
    (lambda m: m["samples"][1].update(start_row=6), "contiguous"),
])
def test_manifest_mutations_fail(mutate, needle):
    m = copy.deepcopy(manifest())
    mutate(m)
    assert any(needle in x for x in ts.validate_manifest(m))


@pytest.mark.parametrize("edit,needle", [
    (lambda rows, ids: rows[1].__setitem__(1, 0), "token order"),            # aggregate/shuffled order
    (lambda rows, ids: rows[3].__setitem__(2, "prefill"), "prefill after decode"),
    (lambda rows, ids: rows[2].__setitem__(2, "both"), "phase"),
    (lambda rows, ids: rows[3].__setitem__(4, 1), "spec_block"),             # block size must equal rows in forward
    (lambda rows, ids: rows[0].__setitem__(5, True), "spec_accepted"),       # prefill cannot carry acceptance
    (lambda rows, ids: rows[5].__setitem__(0, 0), "foreign row"),            # sample boundary violated
    (lambda rows, ids: ids[0][1].__setitem__(0, 128), "outside"),            # 128-expert DSpark layer, id 128 invalid
    (lambda rows, ids: ids[0][0].__setitem__(1, 200), "distinct"),
])
def test_row_mutations_fail(edit, needle):
    rows, ids = data()
    edit(rows, ids)
    assert any(needle in x for x in ts.validate_rows(manifest(), rows, ids)), ts.validate_rows(manifest(), rows, ids)


def test_ids_above_int8_range_are_representable():
    rows, ids = data()
    assert max(e for r in ids for layer in r for e in layer) > 127  # the case int8 would silently wrap
    assert ts.validate_rows(manifest(), rows, ids) == []


def test_derive_counts_steps_layers_experts_and_phase_split():
    rows, ids = data()
    m = manifest()
    allc = ts.derive_counts(m, rows, ids)
    assert set(allc) == {(0, 0), (0, 1), (0, 2), (1, 0)}           # steps keep sample identity
    assert len(allc[(0, 0)]) == 2 and len(allc[(0, 0)][0]) == 256 and len(allc[(0, 0)][1]) == 128
    assert sum(allc[(0, 2)][0]) == 2 * 6                              # speculative block: 2 tokens x top-6
    dec = ts.derive_counts(m, rows, ids, phase="decode")
    assert set(dec) == {(0, 1), (0, 2)}
    pre = ts.derive_counts(m, rows, ids, phase="prefill")
    assert set(pre) == {(0, 0), (1, 0)}


def test_aggregate_only_dump_is_rejected():
    rows, ids = data()
    assert ts.validate_rows(manifest(), rows[:1], ids[:1])  # fewer rows than the sample boundaries declare
