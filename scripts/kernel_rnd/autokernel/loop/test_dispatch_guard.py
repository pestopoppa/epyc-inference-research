from pathlib import Path
import os
import subprocess
import sys
import tempfile

import pytest

from autokernel.loop import dispatch_guard as D
from autokernel.loop.loop import Hypothesis


def identity(**overrides):
    args = dict(diff="+ int  x = 1;\n", champion="c0", cmake_defines=("-DX=1",),
                bench_recipe={"pairs": 5, "pp": 512}, model="m.gguf", surface="pp512")
    args.update(overrides)
    return D.attempt_identity(**args)


def test_identity_normalizes_whitespace_and_reopens_on_recipe_or_champion():
    assert identity(diff="+ int x = 1;\n") == identity(diff="+   int   x = 1; \n")
    assert identity(champion="c1") != identity()
    assert identity(bench_recipe={"pairs": 6, "pp": 512}) != identity()


def test_answered_identity_refuses_after_fresh_process_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        code = (
            "from pathlib import Path; from autokernel.loop.dispatch_guard import Registry; "
            f"r=Registry(Path({tmp!r})); r.reserve({identity()!r}); "
            f"r.finish({identity()!r},status='measured_null',effect=-.01,epoch='e0'); r.close()")
        package_root = str(Path(__file__).resolve().parents[2])
        child_env = dict(os.environ)
        child_env["PYTHONPATH"] = package_root + (
            os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else "")
        subprocess.run([sys.executable, "-c", code], check=True, env=child_env)
        with pytest.raises(D.DispatchRefused) as caught:
            D.Registry(Path(tmp)).reserve(identity())
        assert caught.value.duplicate_of == identity()
        assert caught.value.prior_effect == -.01
        assert caught.value.prior_epoch == "e0"


def test_non_answer_gets_one_identical_retry_then_closes_infeasible():
    with tempfile.TemporaryDirectory() as tmp:
        registry = D.Registry(Path(tmp))
        assert registry.reserve(identity()).dispatch_count == 1
        registry.finish(identity(), status="bench_failed", effect=None, epoch="e0")
        assert registry.reserve(identity()).dispatch_count == 2
        registry.finish(identity(), status="planner_transient", effect=None, epoch="e0")
        with pytest.raises(D.DispatchRefused, match="closed infeasible"):
            registry.reserve(identity())


def test_corrupt_registry_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "dispatch-identity.sqlite3").write_bytes(b"not sqlite")
        with pytest.raises(D.DispatchRefused, match="corrupt"):
            D.Registry(Path(tmp))


def test_characterised_gate_uses_facets_and_epoch_not_statement():
    h = Hypothesis("akm-x", "new prose", "f", "a.cu", "sym")
    rows = [{"mechanism_id": "akm-x", "target_surface": "a.cu",
             "target_symbol": "sym", "statement": f"different {n}",
             "epoch_sha256": "e0", "status": "measured_null"} for n in range(3)]
    assert "characterised" in D.characterised_reason(
        h, {"epoch_sha256": "e0", "prior_experiments": rows})
    assert D.characterised_reason(h, {"epoch_sha256": "e1", "prior_experiments": rows}) is None
