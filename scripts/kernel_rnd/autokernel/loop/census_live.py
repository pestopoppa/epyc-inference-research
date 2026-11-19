"""AKX-P0d/P1d live census plumbing: run a real dispatch probe pinned to a cpu set.

The row jobs (`akx-p1d-*`, `akx-p3a-*`) use this so the probe command is built in ONE place:
the child inherits only an explicit environment (loader path = the probed build's own bin,
FILLER_ID passed through), runs under ``taskset``, and every census is stamped with the
`census_store.census_identity` that the keep hook re-derives.
"""
from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import census as CE
from . import census_store as CS


def child_env(binary: Path | str, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/home/node"),
           "LD_LIBRARY_PATH": str(Path(binary).resolve().parent),
           "FILLER_ID": os.environ.get("FILLER_ID", ""), "OMP_NUM_THREADS": "1"}
    env.update(extra or {})
    return env


def taskset_runner(cpus: str, *, run=subprocess.run):
    """A `census.run_process`-shaped runner that pins the child to ``cpus``."""
    def runner(argv, *, env, timeout):
        # -v output can carry non-UTF-8 bytes (tensor names/vocab); a strict decode raised
        # UnicodeDecodeError mid-census, so decode leniently.
        return run(["/usr/bin/taskset", "-c", cpus, *argv], capture_output=True, text=True,
                   errors="replace", timeout=timeout, env=dict(env))
    return runner


def bench_argv(binary: Path | str, model: Path | str, shape: CE.Shape,
               recipe_argv: Iterable[str] = ()) -> list[str]:
    """llama-bench argv for a shape. Production and C16 `llama-bench` reject `-np`
    (`census.shape_argv` emits it), so parallel sequences are refused here and a verify
    width is the equivalent n-token batch forward (`-p n -n 0`)."""
    if shape.n_seq != 1:
        raise ValueError("llama-bench cannot express parallel sequences; use n_seq=1")
    work = (["-p", "0", "-n", str(shape.n_tokens)] if shape.phase == "decode"
            else ["-p", str(shape.n_tokens), "-n", "0"])
    return [str(binary), "-m", str(model), *work, *list(recipe_argv), "-r", "1", "-o", "json", "-v"]


def produce(*, binary: Path | str, model: Path | str, workload_id: str, signature: str,
            champion_commit: str, probe_name: str, shapes: Iterable[CE.Shape], cpus: str,
            threads: int, expected_ops: Sequence[str] = ("MUL_MAT",), timeout: int = 1800,
            runner=None) -> dict[str, Any]:
    """One real ``workload_census.v1`` for ``model`` via ``binary`` (a llama-bench build)."""
    probe = functools.partial(CE.run_dispatch_probe, runner=runner or taskset_runner(cpus),
                              argv_fn=bench_argv)
    return CE.produce_census(
        workload_id=workload_id, workload_signature=signature, model=Path(model),
        identity=CS.census_identity(champion_commit=champion_commit, probe=probe_name),
        shapes=list(shapes), binary=Path(binary), recipe_argv=("-t", str(threads)),
        env=child_env(binary), expected_ops=expected_ops, timeout=timeout, probe=probe)


__all__ = ["bench_argv", "child_env", "produce", "taskset_runner"]
