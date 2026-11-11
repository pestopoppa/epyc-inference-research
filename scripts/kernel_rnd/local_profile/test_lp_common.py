"""Pure-Python unit tests for lp_common (no server, no model)."""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lp_common as L  # noqa: E402


def test_compress_and_fold():
    assert L.compress_cpus([76, 77, 78, 79, 5]) == "5,76-79"
    assert L.expand_cpus("0-2,5") == [0, 1, 2, 5]
    assert L.fold_physical(list(range(192))) == list(range(96))
    assert len(L.granted_physical({"FILLER_CPUS": "0-95,96-191"})) == 96
    assert L.granted_physical({"FILLER_CPUS": "184-191"}) == list(range(88, 96))
    assert L.granted_physical({}, affinity={3, 4, 99}) == [3, 4]


def test_seeded_orders_balanced_deterministic():
    o = L.seeded_orders(20261106, 24)
    assert o.count("AB") == 12 and o.count("BA") == 12
    assert o == L.seeded_orders(20261106, 24)
    assert o != L.seeded_orders(1, 24)
    with pytest.raises(ValueError):
        L.seeded_orders(1, 5)


def test_tci():
    assert L.tci([]) == {"n": 0}
    assert L.tci([3.0]) == {"n": 1, "mean": 3.0}
    c = L.tci([1.0, 2.0, 3.0])
    assert c["mean"] == 2.0 and math.isclose(c["sd"], 1.0)
    assert math.isclose(c["hi"] - c["mean"], 4.303 / math.sqrt(3), rel_tol=1e-9)
    c24 = L.tci([1.0, 3.0] * 12)
    assert c24["n"] == 24 and c24["lo"] < 2.0 < c24["hi"]


def test_valid():
    assert L.valid({"predicted_n": 256, "prompt_n": 10}, None)
    assert L.valid({"predicted_n": 256, "prompt_n": 10}, 10)
    assert not L.valid({"predicted_n": 256, "prompt_n": 11}, 10)
    assert not L.valid({"predicted_n": 1, "prompt_n": 10}, 10)


def test_ledger_resume_and_torn_line(tmp_path):
    p = tmp_path / "r.jsonl"
    led = L.Ledger(p)
    led.add({"unit_id": "a"})
    p.write_text(p.read_text() + '{"unit_id": "torn')
    led2 = L.Ledger(p)
    assert led2.done("a") and not led2.done("torn") and len(led2.rows) == 1


def test_should_stop(tmp_path, monkeypatch):
    monkeypatch.delenv("FILLER_STOPFILE", raising=False)
    monkeypatch.delenv("FILLER_DEADLINE", raising=False)
    assert not L.should_stop()
    s = tmp_path / "STOP"
    monkeypatch.setenv("FILLER_STOPFILE", str(s))
    assert not L.should_stop()
    s.write_text("")
    assert L.should_stop()
    s.unlink()
    monkeypatch.setenv("FILLER_DEADLINE", "1")
    assert L.should_stop()


def test_server_argv_shape():
    a = L.server_argv("/s", 18312, 48, True, [0, 1, 2, 3], fence_dir="/o")
    assert a[:6] == ["/usr/bin/python3", "-I", L.FENCE, "fence", "/o", "--"]
    assert a[a.index("-t") + 1] == "48" and a[a.index("-tb") + 1] == "48"
    assert "/s/bin/llama-server" in a and "--no-kv-unified" in a and "draft-mtp" in a
    assert a[a.index("-c", a.index("/s/bin/llama-server")) + 1] == "262144" and a[a.index("-np") + 1] == "4"
    n = L.server_argv("/s", 1, 96, False, range(96))
    assert "draft-mtp" not in n and "-md" not in n and "fence" not in n


class FakeServer:
    """Token = one whitespace-separated word; completion stops on 'limit' unless the prompt ends with 'EOSBAD'."""

    def tokenize(self, text, special=True):
        return text.replace("<|endoftext|><|im_end|>", " EOSA EOSB").split()

    def complete(self, text, n):
        return {"stop_type": "eos" if text.rstrip().endswith("EOSBAD") else "limit"}


def _src(tmp_path, texts):
    p = tmp_path / "src.jsonl"
    p.write_text("\n".join(json.dumps({"path": f"src/f{i:02d}.py", "text": t}) for i, t in enumerate(texts)))
    return p


def test_build_code_prompt_budget_and_cue(tmp_path):
    src = _src(tmp_path, ["a " * 10] * 5)
    meta = L.build_code_prompt(FakeServer(), tmp_path / "p.txt", tmp_path / "m.json", src, target=30)
    text = (tmp_path / "p.txt").read_text()
    assert meta["files"] == 2 and text.endswith("# File: src/f02.py\n")
    assert json.loads((tmp_path / "m.json").read_text())["sha256"] == meta["sha256"]


def test_build_code_prompt_backs_off_on_eos(tmp_path):
    src = _src(tmp_path, ["a " * 10, "a " * 10, "a " * 10])
    class Bad(FakeServer):
        def complete(self, text, n):  # 3 files -> eos, 2 files -> limit
            return {"stop_type": "eos" if text.count("# File:") == 4 else "limit"}
    meta = L.build_code_prompt(Bad(), tmp_path / "p.txt", tmp_path / "m.json", src, target=1000)
    assert meta["files"] == 2 and [t["bad"] for t in meta["tries"]] == [True, False]


def test_build_code_prompt_fails(tmp_path):
    src = _src(tmp_path, ["a " * 10] * 3)
    class AlwaysEos(FakeServer):
        def complete(self, text, n):
            return {"stop_type": "eos"}
    with pytest.raises(RuntimeError, match="no non-EOS"):
        L.build_code_prompt(AlwaysEos(), tmp_path / "p.txt", tmp_path / "m.json", src, target=1000)
    with pytest.raises(RuntimeError, match="no non-EOS"):
        L.build_code_prompt(FakeServer(), tmp_path / "p.txt", tmp_path / "m.json", _src(tmp_path, ["a " * 99]), target=5)


def test_precond(tmp_path, monkeypatch):
    f = tmp_path / "x"
    f.write_text("hi")
    monkeypatch.setattr(L, "STORE", str(tmp_path))
    monkeypatch.setattr(L, "MODEL", str(f))
    monkeypatch.setattr(L, "MTP", str(f))
    assert L.precond([f]) == 5  # no bin/llama-server
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "llama-server").write_text("x")
    assert L.precond([f]) == 0
    assert L.precond([tmp_path / "missing"]) == 5
