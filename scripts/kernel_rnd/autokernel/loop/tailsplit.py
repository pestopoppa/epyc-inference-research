#!/usr/bin/env python3
"""AK parallel loop P3 (AK-TAIL-SPLIT via rowflow): lanes emit CANDIDATE RECORDS instead of entering the tail.

Design: epyc-root docs/design/autokernel-parallel-loop.md §2 ("Parallel loop") and §4 P3.

Before: planner -> critic -> author -> critic (concurrent across lanes) -> ONE `pipeline.SerializedTail` that builds,
oracles, A/B-times and commits in-process. After: at the point a candidate would enter that tail, the lane

  1. snapshots the lane's validated diff as a commit whose parent is the lane's base (a private index, never the
     lane's own index or any shared clone's; ref `refs/ak-candidates/<id>` keeps it from gc);
  2. writes a candidate record (base sha, patch sha, hypothesis id, predicted effect, touched paths) to the spool;
  3. hands it to epyc-root `akloop/chain.py scaffold` (dry-run scaffold + lint, NEVER publishes: main sessions are the
     only row publishers), and ends the iteration as status `handed_off`.

The keeper then packs the chain rows (build on a CPU quarter, oracle-kld / ab-timing / fold-check on the GPU), so two
candidates can build/gate while a third times. Only the fold-check CAS (`stage_bodies.stage_fold_check`) and the quiet
timing row are exclusive.

Stale base: when the champion advanced past a candidate's base, `rebase_recheck()` produces a NEW candidate record on
the current head (the same hypothesis — never redrawn) whose chain is build + oracle-kld, plus ab-timing only when the
diff touches the measured path, then fold-check.

Rollback: `--legacy-tail` (or `--workers 1`) keeps the in-process SerializedTail path byte for byte.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import loop as loop_mod

CANDIDATE_SCHEMA = "epyc.akloop.candidate.v1"   # == epyc-root akloop/chain.py CANDIDATE_SCHEMA
#: The LIVE candidate spool the row chain and flow_status.py read. `AKCHAIN_SPOOL` overrides it (the test
#: conftest points it at a per-test temp dir, so no fixture candidate ever lands in the live spool).
SPOOL = Path(os.environ.get("AKCHAIN_SPOOL") or "/mnt/raid0/llm/tmp/akchain-spool")
EPYC_ROOT = Path(os.environ.get("EPYC_ROOT", "/workspace"))
CHAIN_PY = "scripts/coordination/akloop/chain.py"
HANDED_OFF = loop_mod.HANDED_OFF
#: Paths whose change moves what the A/B timing row measures (served decode/prefill kernels and their dispatch).
#: A rebase whose diff vs the old base touches none of these re-runs the oracle only.
MEASURED_PATH_RES = tuple(re.compile(p) for p in (
    r"^ggml/src/ggml-cuda/", r"^ggml/src/ggml-hip/", r"^ggml/src/ggml-cpu/", r"^ggml/src/iqk/",
    r"^ggml/src/ggml-backend", r"^ggml/src/ggml\.c$", r"^src/llama-graph", r"^src/llama-context",
    r"^src/llama-model\.cpp$", r"^src/models/", r"^tools/llama-bench/", r"^ggml/CMakeLists\.txt$",
    r"^ggml/src/CMakeLists\.txt$",
))


#: Defined in loop.py (iterate converts it to a `handed_off` Outcome); re-exported here.
HandedOff = loop_mod.HandedOff


def _git(repo: Path | str, *args: str, env: dict | None = None, input_: str | None = None) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
                          env=env, input=input_).stdout.strip()


def snapshot_commit(worktree: Path, base: str, paths: Iterable[str], message: str) -> str:
    """Commit the lane's declared paths on top of `base` WITHOUT touching the lane's index, HEAD or any branch.

    Private GIT_INDEX_FILE seeded from `base`, only the literal declared paths staged (deletions included)."""
    with tempfile.TemporaryDirectory(prefix="ak-cand-idx-") as td:
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(td, "index"))
        _git(worktree, "read-tree", base, env=env)
        for p in paths:
            p = str(p)
            if (Path(worktree) / p).exists():
                _git(worktree, "add", "--", p, env=env)
            else:
                _git(worktree, "rm", "--cached", "--ignore-unmatch", "-q", "--", p, env=env)
        tree = _git(worktree, "write-tree", env=env)
        return _git(worktree, "commit-tree", tree, "-p", base, "-m", message, env=env)


def candidate_record(*, base: str, patch: str, hypothesis: Any, paths: Sequence[str], lane: str,
                     predicted_effect: str | None = None, **extra) -> dict[str, Any]:
    hid = (getattr(hypothesis, "mechanism_id", None) or getattr(hypothesis, "hypothesis_id", None)
           or (hypothesis.get("mechanism_id") if isinstance(hypothesis, Mapping) else None) or "unnamed")
    eff = predicted_effect
    if eff is None:
        for attr in ("predicted_effect", "expected_effect", "expected_gain_pct"):
            v = getattr(hypothesis, attr, None) if not isinstance(hypothesis, Mapping) else hypothesis.get(attr)
            if v not in (None, ""):
                eff = f"{v}%" if isinstance(v, (int, float)) else str(v)
                break
    slug = re.sub(r"[^a-z0-9]+", "-", str(hid).lower()).strip("-")[:40] or "cand"
    rec = {"schema": CANDIDATE_SCHEMA, "id": f"cand-{slug}-{patch[:9]}", "base_sha": base, "patch_sha": patch,
           "hypothesis_id": str(hid), "predicted_effect": str(eff or "unstated"),
           "paths": [str(p) for p in paths], "lane": lane,
           "target_symbol": getattr(hypothesis, "target_symbol", None) if not isinstance(hypothesis, Mapping)
           else hypothesis.get("target_symbol"),
           "statement": (getattr(hypothesis, "statement", None) if not isinstance(hypothesis, Mapping)
                         else hypothesis.get("statement")),
           "emitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rec.update({k: v for k, v in extra.items() if v is not None})
    return rec


def touches_measured_path(paths: Iterable[str]) -> bool:
    return any(rx.search(str(p)) for p in paths for rx in MEASURED_PATH_RES)


def rebase_recheck(rec: Mapping[str, Any], new_head: str, *, rebase: Callable[[str, str, str], str],
                   changed_since_base: Callable[[str, str], list[str]]) -> dict[str, Any]:
    """A stale candidate -> a rebase-and-recheck candidate on `new_head` (same hypothesis; no redraw).

    `rebase(patch, old_base, new_head)` returns the rebased patch sha (cherry-pick onto new_head; a conflict raises).
    `changed_since_base(old_base, new_head)` lists the paths the champion moved under the candidate.
    Stages: build + oracle-kld always (an oracle needs a binary of the rebased tree); ab-timing only when the
    candidate's own paths or the champion's movement touch the measured path; fold-check always (it is the CAS)."""
    old_base, patch = str(rec["base_sha"]), str(rec["patch_sha"])
    new_patch = rebase(patch, old_base, new_head)
    moved = list(changed_since_base(old_base, new_head))
    retime = touches_measured_path(rec.get("paths") or []) and touches_measured_path(moved)
    stages = ["build", "oracle-kld"] + (["ab-timing"] if retime else []) + ["fold-check"]
    out = dict(rec)
    out.update({"base_sha": new_head, "patch_sha": new_patch,
                "id": f"{rec['id'].rsplit('-', 1)[0]}-{new_patch[:9]}",
                "rebase_of": {"candidate": rec["id"], "base_sha": old_base, "patch_sha": patch,
                              "champion_moved_paths": moved[:200], "retime": retime},
                "stages": stages, "skip_stages": [s for s in ("ab-timing",) if s not in stages],
                "emitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    out.pop("stage_results", None)
    return out


def git_rebase_fns(repo: Path) -> tuple[Callable[[str, str, str], str], Callable[[str, str], list[str]]]:
    """Live rebase helpers over a private index (no checkout, no branch move): replay patch's diff onto new_head."""
    def rebase(patch: str, old_base: str, new_head: str) -> str:
        diff = _git(repo, "diff", "--binary", old_base, patch) + "\n"
        msg = _git(repo, "log", "-1", "--format=%B", patch)
        with tempfile.TemporaryDirectory(prefix="ak-rebase-idx-") as td:
            env = dict(os.environ, GIT_INDEX_FILE=os.path.join(td, "index"))
            _git(repo, "read-tree", new_head, env=env)
            _git(repo, "apply", "--cached", "--3way", "-", env=env, input_=diff)
            tree = _git(repo, "write-tree", env=env)
            return _git(repo, "commit-tree", tree, "-p", new_head, "-m", msg + f"\n\nRebased-from: {patch}", env=env)

    def changed(old_base: str, new_head: str) -> list[str]:
        return [p for p in _git(repo, "diff", "--name-only", old_base, new_head).splitlines() if p]
    return rebase, changed


@dataclass
class ChainHandoff:
    """Writes the record to the spool and scaffolds its chain via epyc-root chain.py (dry run; never publishes).

    `scaffold(record_path)` is injectable; the default runs `python3 <EPYC_ROOT>/scripts/coordination/akloop/chain.py
    scaffold --candidate <path> --date <today> --root <spool>/rows`. A scaffold failure is recorded on the record
    (`scaffold_error`), never raised into the lane: the record on disk is the hand-off, the scaffold is a convenience
    the owner can re-run."""
    spool: Path = SPOOL
    scaffold: Callable[[Path], dict] | None = None
    epyc_root: Path = EPYC_ROOT

    def __call__(self, rec: Mapping[str, Any]) -> dict:
        self.spool.mkdir(parents=True, exist_ok=True)
        p = self.spool / f"{rec['id']}.json"
        if p.exists():
            try:
                prior = json.loads(p.read_text())
            except ValueError:
                prior = {}
            if record_digest(prior) == record_digest(rec):
                return dict(prior, record_path=str(p), duplicate_of=prior.get("id"))
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(rec), indent=1, sort_keys=True) + "\n")
        os.replace(tmp, p)
        out = dict(rec, record_path=str(p))
        try:
            out["scaffold"] = (self.scaffold or self._default_scaffold)(p)
        except Exception as e:  # noqa: BLE001 - the record is the hand-off; a scaffold failure is reported, not fatal
            out["scaffold_error"] = f"{type(e).__name__}: {e}"
        return out

    def _default_scaffold(self, p: Path) -> dict:
        r = subprocess.run(["python3", str(self.epyc_root / CHAIN_PY), "scaffold", "--candidate", str(p),
                            "--date", time.strftime("%Y%m%d", time.gmtime()), "--root", str(self.spool / "rows")],
                           capture_output=True, text=True, timeout=600)
        try:
            rep = json.loads(r.stdout)
        except ValueError:
            rep = {"error": (r.stderr or r.stdout)[-400:]}
        rep["rc"] = r.returncode
        return rep


class EmittingTail:
    """Drop-in for `pipeline._TailSession`: entering it emits the candidate record and raises HandedOff.

    It is entered exactly where the legacy tail session would be (after critic 2 and the post-critic integrity
    re-validation), so a candidate that reaches here is fully formed. Nothing is built or measured in-process."""
    accepts_hypothesis = True
    accepts_paths = True

    def __init__(self, *, worker, base: str, handoff: Callable[[Mapping[str, Any]], dict],
                 snapshot: Callable[[Path, str, Sequence[str], str], str] = snapshot_commit,
                 log: list | None = None) -> None:
        self.worker, self.base = worker, base
        self._handoff, self._snapshot = handoff, snapshot
        self.log = log if log is not None else []

    def __call__(self, hypothesis: Any = None, paths: Sequence[str] = ()):
        return self._Session(self, hypothesis, paths)

    class _Session:
        def __init__(self, outer: "EmittingTail", hypothesis: Any, paths: Sequence[str]) -> None:
            self.o, self.h, self.paths = outer, hypothesis, list(paths or [])

        def __enter__(self):
            o = self.o
            paths = [str(p) for p in self.paths]
            hid = getattr(self.h, "mechanism_id", None) or "candidate"
            patch = o._snapshot(Path(o.worker.worktree), o.base, paths,
                                f"akloop candidate {hid} (lane {o.worker.name}, base {o.base[:12]})")
            rec = candidate_record(base=o.base, patch=patch, hypothesis=self.h, paths=paths, lane=o.worker.name)
            done = o._handoff(rec)
            o.log.append(done)
            exc = HandedOff(f"handed off as candidate {rec['id']} (patch {patch[:12]} on base {o.base[:12]}); "
                            f"measured by its row chain, not in-process")
            exc.record = done
            raise exc

        def __exit__(self, *a):
            return False


def record_digest(rec: Mapping[str, Any]) -> str:
    keys = ("base_sha", "patch_sha", "hypothesis_id")
    return hashlib.sha256(json.dumps({k: rec.get(k) for k in keys}, sort_keys=True).encode()).hexdigest()[:16]
