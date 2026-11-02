import concurrent.futures
import ast
import json
from pathlib import Path

import pytest

from scripts.benchmark.hpr_no_redraw_guard import reserve, sha


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "artifacts/hpr-r5-no-redraw-20261102/exposure-ledger.json"


def call(ledger, claims, **changes):
    doc = json.loads(ledger.read_bytes())
    values = dict(**doc["cohort"], arm="coarse", wave=2, attempt_id="test")
    values.update(changes)
    return reserve(ledger, sha(ledger.read_bytes()), claims, **values)


def test_original_inventory_refuses_every_exposed_cell(tmp_path):
    for arm in ("dense", "coarse"):
        for wave in range(3):
            with pytest.raises(ValueError, match="already exposed or unknown"):
                call(ORIGINAL, tmp_path, arm=arm, wave=wave)
    with pytest.raises(ValueError, match="disjointness is unknown"):
        call(ORIGINAL, tmp_path, corpus_sha256="0" * 64)
    assert not list(tmp_path.iterdir())


def test_unexposed_reservation_is_exclusive_across_concurrent_attempts(tmp_path):
    doc = json.loads(ORIGINAL.read_bytes())
    doc["cells"][-1]["state"] = "unexposed"
    doc["cells"][-1]["evidence"] = "synthetic fixture only"
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(doc))
    claims = tmp_path / "claims"
    claims.mkdir()

    def attempt(i):
        try:
            return call(ledger, claims, attempt_id=f"try-{i}")
        except FileExistsError:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        result = list(pool.map(attempt, range(8)))
    assert sum(path is not None for path in result) == 1
    assert len(list(claims.iterdir())) == 1
    with pytest.raises(FileExistsError):
        call(ledger, claims, attempt_id="retry-after-crash")


def test_incomplete_ledger_and_wrong_pin_refuse(tmp_path):
    doc = json.loads(ORIGINAL.read_bytes())
    doc["cells"].pop()
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(doc))
    claims = tmp_path / "claims"
    claims.mkdir()
    with pytest.raises(ValueError, match="complete six-cell"):
        call(ledger, claims)
    with pytest.raises(ValueError, match="digest mismatch"):
        reserve(ledger, "0" * 64, claims, **doc["cohort"], arm="coarse", wave=2, attempt_id="bad")


def test_successor_calls_exposure_gate_before_native_launch():
    source = ROOT / "artifacts/hpr-r5-no-redraw-20261102/full_run_guarded.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "checked_launch")
    calls = [(node.lineno, node.func.id) for node in ast.walk(fn) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id in {"reserve_exposure", "orig_launch"}]
    assert len(calls) == 2
    assert dict((name, line) for line, name in calls)["reserve_exposure"] < dict(
        (name, line) for line, name in calls)["orig_launch"]
