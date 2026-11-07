#!/usr/bin/env python3
"""Model-free LB3.2b fixture: an 8-thread OpenMP program under the recipe OMP env
(OMP_PROC_BIND=spread, OMP_PLACES=cores, OMP_WAIT_POLICY=active, OMP_DYNAMIC=false,
as recorded in the cv5 G-t48.environ), taskset to a small CPU set. The legacy
main-TID readback must refuse (RC 68); the v3 all-task readback must pass.
Usage: test_omp_fixture.py [CPUS]   (default 92-95; seconds, no model, no GPU)."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
from affinity_readback import affinity, legacy_affinity  # noqa: E402

RECIPE_OMP = {"OMP_PROC_BIND": "spread", "OMP_PLACES": "cores",
              "OMP_WAIT_POLICY": "active", "OMP_DYNAMIC": "false"}


def run(cpus):
    with tempfile.TemporaryDirectory(prefix="lb32b-fixture-") as t:
        exe = Path(t) / "fixture"
        subprocess.run(["gcc", "-O0", "-fopenmp", "-o", str(exe), str(D / "omp_fixture.c")], check=True)
        p = subprocess.Popen(["taskset", "-c", cpus, str(exe)], env=dict(os.environ, **RECIPE_OMP),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            line = p.stdout.readline().strip()
            assert line == "ready 8", line
            ntask = len(os.listdir(f"/proc/{p.pid}/task"))
            try:
                legacy = legacy_affinity(p.pid, cpus)
            except AssertionError as e:
                legacy = {"RC": 68, "error": str(e)}
            try:
                new = affinity(p.pid, cpus, bench=True)
            except (ValueError, OSError) as e:
                new = {"RC": 68, "error": str(e)}
        finally:
            p.stdin.write("\n")
            p.stdin.flush()
            p.wait(10)
    return ntask, legacy, new


def main(cpus):
    ntask, legacy, new = run(cpus)
    print("env", RECIPE_OMP, "cpus", cpus, "tasks", ntask)
    print("LEGACY", json.dumps(legacy))
    shown = {k: new[k] for k in ("RC", "scope", "union", "tasks")} if new["RC"] == 0 else new
    print("V3    ", json.dumps(shown))
    ok = legacy["RC"] == 68 and new["RC"] == 0 and ntask >= 8
    print("FIXTURE", "PASS (legacy refuses, v3 accepts)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "92-95"))
