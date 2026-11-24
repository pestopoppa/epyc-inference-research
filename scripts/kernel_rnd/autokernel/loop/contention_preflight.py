"""AK-EVAL-QUARTERS: may quarter-scoped A/B streams run concurrently?

Rule (C21's, `c21-fold-cpu-20261124-r2/job.py:bw_preflight`): measure each quarter's
read bandwidth alone, then the candidate quarter sets concurrently. Admit
q0+q1+q2 (3 streams) when every member keeps >= 95 % of its solo bandwidth, else
q0+q2, else q0 alone. q3 stays with the GPU host threads/actors.

The probe is injected (`probe(quarter) -> GB/s | None`); `readbw_probe` builds the
real one around C21's `bench_readbw` under the SAME node-bound policy the quarter
screen now uses (`cpu_screen.numa_policy`). Running it is compute: a granted row
does that, never a unit test. The decision and the receipt are pure.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
from typing import Callable

SCHEMA = "epyc.autokernel.contention_preflight.v1"
SOLO_FRACTION_MIN = 0.95
CANDIDATE_SETS = (("q0", "q1", "q2"), ("q0", "q2"))
FALLBACK = ("q0",)
QUARTER_CPUS = {f"q{i}": f"{24 * i}-{24 * i + 23}" for i in range(4)}


def admit(solo: dict, concurrent: dict) -> tuple[tuple[str, ...], dict]:
    """Pure: the admitted quarter set and every tried ratio. `concurrent` maps a
    '+'-joined set to {quarter: GB/s} measured while the whole set ran together."""
    tried = {}
    for quarters in CANDIDATE_SETS:
        key = "+".join(quarters)
        measured = concurrent.get(key) or {}
        ratio = {q: (measured[q] / solo[q]) if measured.get(q) and solo.get(q) else None
                 for q in quarters}
        tried[key] = {"concurrent": dict(measured), "ratio": ratio}
        if all(r is not None and r >= SOLO_FRACTION_MIN for r in ratio.values()):
            return quarters, tried
    return FALLBACK, tried


def run_preflight(probe: Callable[[str], float | None]) -> dict:
    """Solo per quarter, then each candidate set concurrently, stopping at the first
    admitted set (stop-on-answer: the cheapest set that answers is the receipt)."""
    solo = {q: probe(q) for q in ("q0", "q1", "q2")}
    concurrent: dict = {}
    for quarters in CANDIDATE_SETS:
        out: dict = {}
        threads = [threading.Thread(target=lambda q=q: out.__setitem__(q, probe(q)))
                   for q in quarters]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        concurrent["+".join(quarters)] = out
        admitted, _ = admit(solo, concurrent)
        if admitted == quarters:
            break
    admitted, tried = admit(solo, concurrent)
    return {"schema": SCHEMA, "solo": solo, "tried": tried, "admitted": list(admitted),
            "streams": len(admitted), "solo_fraction_min": SOLO_FRACTION_MIN}


def write_receipt(path: Path, receipt: dict) -> Path:
    body = json.dumps(receipt, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**receipt, "content_sha256": hashlib.sha256(body.encode()).hexdigest()},
                               sort_keys=True, indent=1))
    return path


def admitted_region(receipt_path: Path, region: str) -> str:
    """The run's gate: refuse a concurrent stream on a quarter the receipt did not admit."""
    receipt = json.loads(Path(receipt_path).read_text())
    body = {k: v for k, v in receipt.items() if k != "content_sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if receipt.get("schema") != SCHEMA or receipt.get("content_sha256") != digest:
        raise ValueError("contention preflight receipt is malformed or altered")
    if region not in receipt.get("admitted", ()):
        raise ValueError(f"quarter {region} is not admitted for concurrent streams "
                         f"(admitted: {receipt.get('admitted')})")
    return region


def readbw_probe(readbw: Path, *, gib: str = "1", reps: str = "2", timeout_s: int = 900):
    """The real probe (C21's bench_readbw, node-bound like the screen). Compute."""
    def probe(quarter: str) -> float | None:
        node = int(quarter[1:])
        argv = ["/usr/bin/taskset", "-c", QUARTER_CPUS[quarter], "/usr/bin/numactl",
                f"--membind={node}", str(readbw), gib, reps, "0"]
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s,
                              env={"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "24",
                                   "HIP_VISIBLE_DEVICES": "-1"})
        found = re.search(r"^read-sum\s+([\d.]+)", done.stdout, re.M)
        return float(found.group(1)) if done.returncode == 0 and found else None
    return probe
