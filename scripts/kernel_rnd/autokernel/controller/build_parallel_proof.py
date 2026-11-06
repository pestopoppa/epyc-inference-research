"""AK-SETUP-2 acceptance proof: a pinned parallel `-j` build vs the serial build of one source.

Accept only when ALL hold (the handoff's own rule):
  * both builds name the SAME declared `-j` pin discipline (parallel jobs > serial jobs, fixed),
  * every compared binary digest is identical (a changed digest or a stale object -> revert),
  * the parallel tree passes a `test-backend-ops` smoke (exit 0 and no failing backend),
  * median build wall drops by >= 50% across the supplied samples.

Pure functions plus an injectable smoke runner: this module never launches a build itself.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import statistics
import subprocess
from typing import Callable, Mapping, Sequence

COMPARED = ("bin/llama-bench", "bin/test-backend-ops")
MIN_DROP = 0.50


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def digest_manifest(build_dir: Path, extra_globs: Sequence[str] = ("bin/libggml*.so*",)) -> dict[str, str]:
    root = Path(build_dir)
    names = list(COMPARED)
    for g in extra_globs:
        names += [str(p.relative_to(root)) for p in sorted(root.glob(g)) if p.is_file() and not p.is_symlink()]
    out = {}
    for n in dict.fromkeys(names):
        p = root / n
        if not p.is_file():
            raise FileNotFoundError(str(p))
        out[n] = _sha(p)
    return out


def default_smoke(build_dir: Path, timeout: float = 600.0) -> tuple[int, str]:
    """CPU-backend ADD/MUL smoke; never touches a GPU."""
    exe = Path(build_dir) / "bin" / "test-backend-ops"
    r = subprocess.run([str(exe), "test", "-b", "CPU", "-o", "ADD"], capture_output=True, text=True,
                       timeout=timeout, check=False)
    return r.returncode, r.stdout[-2000:]


def smoke_clean(rc: int, out: str) -> bool:
    return rc == 0 and "FAIL" not in out and ("passed" in out or "OK" in out)


def prove(*, serial_dir: Path, serial_walls: Sequence[float], serial_jobs: int,
          parallel_dir: Path, parallel_walls: Sequence[float], parallel_jobs: int,
          smoke: Callable[[Path], tuple[int, str]] = default_smoke) -> Mapping[str, object]:
    reasons = []
    if not serial_walls or not parallel_walls:
        reasons.append("missing wall samples")
    if parallel_jobs <= serial_jobs:
        reasons.append("parallel -j is not above the serial -j")
    try:
        sd, pd = digest_manifest(serial_dir), digest_manifest(parallel_dir)
    except FileNotFoundError as exc:
        sd = pd = {}
        reasons.append(f"missing artifact {exc}")
    differing = sorted(k for k in set(sd) | set(pd) if sd.get(k) != pd.get(k))
    if differing:
        reasons.append(f"digest changed (revert): {differing}")
    rc, out = smoke(Path(parallel_dir)) if not reasons or differing else (None, "")
    smoke_ok = rc is not None and smoke_clean(rc, out)
    if rc is not None and not smoke_ok:
        reasons.append(f"smoke not clean (rc={rc})")
    drop = None
    if serial_walls and parallel_walls:
        drop = 1.0 - statistics.median(parallel_walls) / statistics.median(serial_walls)
        if drop < MIN_DROP:
            reasons.append(f"median wall drop {drop:.1%} < {MIN_DROP:.0%}")
    return {"schema": "epyc.autokernel.parallel_build_proof.v1", "accepted": not reasons,
            "reasons": reasons, "serial_jobs": serial_jobs, "parallel_jobs": parallel_jobs,
            "median_wall_drop": drop, "digests_identical": not differing and bool(sd),
            "smoke_clean": smoke_ok, "compared": sorted(sd)}
