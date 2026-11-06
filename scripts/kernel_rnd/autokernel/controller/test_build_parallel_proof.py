from pathlib import Path
import tempfile

from autokernel.controller import build_parallel_proof as p


def tree(body=b"x", names=("llama-bench", "test-backend-ops")):
    r = Path(tempfile.mkdtemp())
    (r / "bin").mkdir()
    for n in names:
        (r / "bin" / n).write_bytes(body + n.encode())
    (r / "bin" / "libggml-base.so.0.1").write_bytes(body)
    return r


OK = lambda d: (0, "1139/1139 tests passed\nOK")


def kw(**o):
    d = dict(serial_dir=tree(), serial_walls=[2000, 2100, 1900], serial_jobs=1,
             parallel_dir=tree(), parallel_walls=[600, 700, 650], parallel_jobs=64, smoke=OK)
    d.update(o)
    return d


def test_accept():
    r = p.prove(**kw())
    assert r["accepted"] and r["digests_identical"] and r["smoke_clean"] and r["median_wall_drop"] > 0.6


def test_digest_change_reverts():
    r = p.prove(**kw(parallel_dir=tree(b"y")))
    assert not r["accepted"] and any("digest changed" in x for x in r["reasons"])


def test_insufficient_drop_and_bad_smoke_and_jobs():
    assert any("drop" in x for x in p.prove(**kw(parallel_walls=[1500, 1400]))["reasons"])
    assert any("smoke" in x for x in p.prove(**kw(smoke=lambda d: (1, "FAIL")))["reasons"])
    assert any("-j" in x for x in p.prove(**kw(parallel_jobs=1))["reasons"])


def test_missing_artifact_never_accepts():
    r = p.prove(**kw(parallel_dir=tree(names=("llama-bench",))))
    assert not r["accepted"]
