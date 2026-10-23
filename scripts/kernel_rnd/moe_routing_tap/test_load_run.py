"""RT-1 reader tests: trace_shape.load_run on a run directory written exactly as llama-moe-trace writes it. Synthetic data only."""
import hashlib
import json
import struct
import sys
import tempfile
from array import array
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import trace_shape as ts  # noqa: E402
from test_trace_shape import data, manifest  # noqa: E402


def write_run(d: Path, m: dict, rows, ids):
    mk = max(l["top_k"] for l in m["layers"])
    rb = b"".join(ts.ROW_STRUCT.pack(r[0], r[1], r[3], r[4], ts.PHASES.index(r[2]), ts.ACC_NULL if r[5] is None else int(r[5])) for r in rows)
    flat = array("h")
    for row in ids:
        for sel in row:
            flat.extend(list(sel) + [-1] * (mk - len(sel)))
    ib = flat.tobytes()
    d.mkdir(parents=True, exist_ok=True)
    (d / "routing_trace.rows.bin").write_bytes(rb)
    (d / "routing_trace.ids.i16").write_bytes(ib)
    mm = json.loads(json.dumps(m))
    mm["rows_sha256"] = hashlib.sha256(rb).hexdigest()
    mm["ids"]["sha256"] = hashlib.sha256(ib).hexdigest()
    (d / "routing_trace.json").write_text(json.dumps(mm))
    return mm


def test_round_trip_validates_and_derives(tmp_path):
    rows, ids = data()
    write_run(tmp_path / "run", manifest(), rows, ids)
    m, r2, i2 = ts.load_run(tmp_path / "run")
    assert r2 == rows and i2 == ids
    assert ts.validate_rows(m, r2, i2) == []
    counts = ts.derive_counts(m, r2, i2, phase="decode")
    assert set(counts) == {(0, 1), (0, 2)}
    assert sum(counts[(0, 2)][0]) == 2 * 6


def test_short_top_k_layer_is_padded_on_disk(tmp_path):
    rows, ids = data()
    m = manifest()
    m["layers"][1]["top_k"] = 4                       # DSpark-like layer with a smaller top_k
    ids = [[row[0], row[1][:4]] for row in ids]
    write_run(tmp_path / "run", m, rows, ids)
    mm, r2, i2 = ts.load_run(tmp_path / "run")
    assert i2[0][1] == ids[0][1] + [-1, -1]
    assert ts.validate_rows(mm, r2, i2) == []


def test_tampered_ids_are_refused(tmp_path):
    rows, ids = data()
    write_run(tmp_path / "run", manifest(), rows, ids)
    p = tmp_path / "run" / "routing_trace.ids.i16"
    b = bytearray(p.read_bytes()); b[0] ^= 1; p.write_bytes(bytes(b))
    with pytest.raises(ValueError, match="ids.i16 sha256"):
        ts.load_run(tmp_path / "run")
    assert ts.load_run(tmp_path / "run", verify=False)[0]["arch"] == "deepseek41"   # explicit opt-out only


def test_truncated_rows_and_missing_manifest(tmp_path):
    rows, ids = data()
    write_run(tmp_path / "run", manifest(), rows, ids)
    p = tmp_path / "run" / "routing_trace.rows.bin"
    p.write_bytes(p.read_bytes()[:-16])
    with pytest.raises(ValueError):
        ts.load_run(tmp_path / "run", verify=False)
    (tmp_path / "run" / "routing_trace.json").unlink()
    with pytest.raises(FileNotFoundError):                    # an interrupted capture has no manifest: not a trace
        ts.load_run(tmp_path / "run")


def test_int8_manifest_is_refused_before_any_data_is_read(tmp_path):
    rows, ids = data()
    m = manifest(); m["layers"][0]["id_dtype"] = "int8"
    write_run(tmp_path / "run", m, rows, ids)
    with pytest.raises(ValueError, match="int16"):
        ts.load_run(tmp_path / "run")
