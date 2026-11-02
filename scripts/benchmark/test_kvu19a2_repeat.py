import importlib
import sys
import types
from pathlib import Path


def load(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts/benchmark"))
    monkeypatch.setenv("EPYC_VIDYA_ROOT", "/tmp")
    package = types.ModuleType("adapters")
    package.__path__ = []
    adapter = types.ModuleType("adapters.kvu19a2_native")
    adapter.record_run = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "adapters", package)
    monkeypatch.setitem(sys.modules, "adapters.kvu19a2_native", adapter)
    monkeypatch.setenv("KVU19A2_EXE", "/tmp/test-exe")
    monkeypatch.setenv("KVU19A2_REPEAT_OUT", "/tmp/test-out")
    monkeypatch.setenv("KVU19A2_SOURCE_ARCHIVE", "/tmp/test-source")
    sys.modules.pop("kvu19a2_repeat", None)
    return importlib.import_module("kvu19a2_repeat")


def test_comparison_refuses_missing_samples_input_drift_and_duplicates(monkeypatch):
    m = load(monkeypatch)
    rows = [{"case": 78, "arm": arm, "repeat": repeat, "rc": 0,
             "input_hashes": {"a": "1", "b": "2", "c": "3"}, "output_sha256": "same",
             "mapping": {"pid": 1}, "samples": [{"kfd": True, "vram_bytes": 1}]}
            for arm in ("base", "candidate") for repeat in range(1, 6)]
    assert m.compare(rows)["78"]["arms"]["base"]["verdict"] == "BIT_IDENTICAL"
    rows[0]["output_sha256"] = "different"
    assert m.compare(rows)["78"]["arms"]["base"]["verdict"] == "VARIATION"
    rows[0]["samples"] = []
    assert m.compare(rows)["78"]["arms"]["base"]["verdict"] == "UNKNOWN"
    rows[0]["input_hashes"] = {"a": "wrong", "b": "2", "c": "3"}
    assert m.compare(rows)["78"]["arms"]["candidate"]["verdict"] == "UNKNOWN"
    rows[0]["input_hashes"] = rows[1]["input_hashes"]
    rows[0]["repeat"] = 2
    assert m.compare(rows)["78"]["arms"]["base"]["verdict"] == "UNKNOWN"


def test_rootbuild_comparison_requires_predeclared_canonical_input(monkeypatch):
    load(monkeypatch)
    monkeypatch.setenv("KVU19A2_BUILD_RECEIPT_SHA256", "f" * 64)
    monkeypatch.setenv("KVU19A2_EXE_SHA256", "e" * 64)
    m = importlib.import_module("kvu19a2_repeat_rootbuild")
    hashes = {"a": "1", "b": "2", "c": "3"}
    rows = [{"case": 78, "arm": arm, "repeat": repeat, "rc": 0,
             "input_hashes": hashes, "output_sha256": "same", "mapping": {"pid": 1},
             "samples": [{"kfd": True, "vram_bytes": 1}]}
            for arm in ("base", "candidate") for repeat in range(1, 6)]
    canonical = {"78": hashes, "80": {"a": "1", "b": "2", "c": "3", "d": "4"}}
    assert m.compare(rows, canonical)["78"]["arms"]["base"]["verdict"] == "BIT_IDENTICAL"
    canonical["78"] = {"a": "other", "b": "2", "c": "3"}
    assert m.compare(rows, canonical)["78"]["arms"]["base"]["verdict"] == "UNKNOWN"
