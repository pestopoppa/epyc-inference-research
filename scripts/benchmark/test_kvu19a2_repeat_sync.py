import importlib
import sys


def test_sync_comparison_fails_closed(monkeypatch):
    from test_kvu19a2_repeat import load
    load(monkeypatch)
    monkeypatch.setenv("KVU19A2_BUILD_RECEIPT_SHA256", "f" * 64)
    monkeypatch.setenv("KVU19A2_EXE_SHA256", "e" * 64)
    monkeypatch.setenv("KVU19A2_HARNESS_RECEIPT_SHA256", "d" * 64)
    sys.modules.pop("kvu19a2_repeat_sync", None)
    m = importlib.import_module("kvu19a2_repeat_sync")
    hashes = {"a": "1", "b": "2", "c": "3"}
    rows = [{"case": 78, "arm": arm, "repeat": i, "rc": 0, "input_hashes": hashes,
             "output_sha256": "x", "mapping": {"pid": 1},
             "samples": [{"kfd": True, "vram_bytes": 1}]}
            for arm in ("base", "candidate") for i in range(1, 6)]
    canonical = {"78": hashes, "80": {"a": "1", "b": "2", "c": "3", "d": "4"}}
    assert m.compare(rows, canonical)["78"]["arms"]["candidate"]["verdict"] == "BIT_IDENTICAL"
    rows[-1]["output_sha256"] = "y"
    assert m.compare(rows, canonical)["78"]["arms"]["candidate"]["verdict"] == "VARIATION"
    rows[-1]["mapping"] = None
    assert m.compare(rows, canonical)["78"]["arms"]["candidate"]["verdict"] == "UNKNOWN"
    rows[-1]["mapping"] = {"pid": 1}
    rows[-1]["samples"] = []
    assert m.compare(rows, canonical)["78"]["arms"]["candidate"]["verdict"] == "UNKNOWN"
    rows[-1]["input_hashes"] = {"a": "changed", "b": "2", "c": "3"}
    assert m.compare(rows, canonical)["78"]["arms"]["base"]["verdict"] == "UNKNOWN"
