#!/usr/bin/env python3
"""Per-target champion lineage: a shared trunk plus one champion branch per target.

WHY (operator 2026-10-04). DS41's held keeps (b3e0b0902) cut Qwen3.8-Flash-Next prefill
by 29-41% (`/mnt/raid0/llm/tmp/q38fn-transfer-20261003/run2.log`). A shared change can
help one model and hurt another. Vetoing it throws the gain away, so the lineage is
split and every decision is deterministic:

* Each lane-bound instance keeps on ITS target's champion branch
  (`lane_targets` `champion_branch`; that branch is the instance's `--champion-branch`,
  its anchor and its accumulator, as in 741c85d8). A keep that clears its own target's
  gates is KEPT there. It is never vetoed because another target regressed, and the
  patch does not have to be gated by model.
* `decide`: the SHARED TRUNK (`lanes.trunk.branch`) advances only with a keep that
  regresses no target. That means every peer's A/B is within its floor (or bar), and
  every peer's kernel coverage is unchanged (`kernel_coverage.peer_coverage`:
  observed, nothing lost, nothing changed). A change is allowed only with a decisive
  peer gain. A keep that touches only the lane's exclusive paths cannot affect a peer,
  so it is shared too. Anything else is TARGET-ONLY.
* `record_keep` cherry-picks a shared keep onto the trunk, using plumbing only
  (`git merge-tree --write-tree` + `commit-tree` + a compare-and-swap `update-ref`),
  so no other instance's worktree is touched. `sync_from_trunk` then brings trunk
  commits that are missing (by patch id, `git cherry`) onto the receiving lane's own
  branch at ITS next keep boundary, before its anchor is rebuilt. They were already
  A/B'd on that target as the peer check.
* `pending` + `recheck_one`: at every champion advancement a lane re-checks ONE
  target-only keep from another lane (oldest first) on its own target. The keep is
  applied onto this lane's champion, built, then MUL_MAT oracle, kernel coverage and
  an A/B against this target's matched floor. If it HELPS (decisive gain), or is
  neutral and SIMPLIFIES (the patch deletes more lines than it adds), it propagates
  onto this lane's branch. Once every lane carries it, it goes onto the trunk. If it
  regresses or does not simplify, it stays target-only and is re-checked at this
  lane's next advancement (one row per keep per champion head, so no repeats). The
  worst case is a missed propagation, which the next advancement catches.

Every decision is one JSON line in `<binding dir>/cross-target-ledger.jsonl`, appended
under flock (both instances share it). The same decision also rides into the keep's
experiments-DB row (`integrity_screen.cross_target`).
"""
from __future__ import annotations

import fcntl
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping

LEDGER = "cross-target-ledger.jsonl"
LEDGER_SCHEMA = "epyc.autokernel.cross_target_ledger.v1"
SHARED, TARGET_ONLY = "shared", "target_only"


# ------------------------------------------------------------------ ledger

def ledger_path(bound) -> Path:
    return Path(bound.path).parent / LEDGER


def append(bound, entry: Mapping[str, Any], *, now: Callable[[], float] = time.time) -> dict:
    row = {"schema": LEDGER_SCHEMA, "at": now(), **entry}
    path = ledger_path(bound)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
    return row


def read(bound) -> list[dict]:
    path = ledger_path(bound)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("schema") == LEDGER_SCHEMA:
            rows.append(row)
    return rows


# ------------------------------------------------------------------ decisions

def decide(cross: Mapping[str, Any], coverage: Mapping[str, Mapping[str, Any]]) -> dict:
    """shared (the trunk may take it) or target_only, for one keep. Pure."""
    if not cross.get("required"):
        return {"decision": SHARED, "reason": "every changed path is exclusive to this "
                                              "lane's target; no peer can be affected"}
    reasons = []
    for row in cross.get("peers") or ():
        target = row.get("target_id")
        if not row.get("passed"):
            reasons.append(f"{target}: {row.get('reason')}")
            continue
        cov = coverage.get(target)
        if not cov or not cov.get("observed"):
            reasons.append(f"{target}: peer kernel coverage not observed on both builds")
        elif cov.get("losses"):
            reasons.append(f"{target}: peer kernel path lost: "
                           + ", ".join(loss["key"] for loss in cov["losses"][:4]))
        elif cov.get("changed") and not (row.get("decisive") is True
                                         and _finite(row.get("effect")) and row["effect"] > 0):
            reasons.append(f"{target}: peer executed kernel path changed without a "
                           "decisive peer gain: " + ", ".join(cov["changed"][:4]))
    if not cross.get("peers"):
        reasons.append("no peer target was checked")
    return ({"decision": TARGET_ONLY, "reason": "; ".join(reasons)} if reasons else
            {"decision": SHARED, "reason": "no peer regressed and every peer's executed "
                                           "kernel path is unchanged"})


def recheck_decision(row: Mapping[str, Any] | None, *, simplifies: bool) -> dict:
    """propagate or stay, for one target-only keep re-measured on another target."""
    if row is None or not _finite(row.get("effect")):
        return {"result": "stay", "why": "failed", "reason": "no finite effect"}
    pct = row["effect"] * 100.0
    if row.get("decisive") is True and pct > 0:
        return {"result": "propagate", "why": "helps", "reason": f"decisive {pct:+.3f}%"}
    if row.get("decisive") is True and pct < 0:
        return {"result": "stay", "why": "regressed", "reason": f"decisive {pct:+.3f}%"}
    if simplifies:
        return {"result": "propagate", "why": "neutral_simplifies",
                "reason": f"{pct:+.3f}% within the floor and the patch removes code"}
    return {"result": "stay", "why": "neutral_not_simpler",
            "reason": f"{pct:+.3f}% within the floor; the patch adds code"}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


# ------------------------------------------------------------------ git plumbing

def _git(repo: Path, *args: str, input: str | None = None,
         env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess:
    from .archive import _git_env
    return subprocess.run(["git", "-C", str(repo), *args], input=input, text=True,
                          capture_output=True, env=_git_env(env), check=False)


def _out(repo: Path, *args: str, **kwargs) -> str:
    done = _git(repo, *args, **kwargs)
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {done.stderr.strip()[:300]}")
    return done.stdout.strip()


def patch_id(repo: Path, commit: str) -> str:
    diff = _out(repo, "show", "--format=", "--no-color", "--no-ext-diff", commit)
    return _out(repo, "patch-id", "--stable", input=diff + "\n").split()[0]


def simplifies(repo: Path, commit: str) -> bool:
    added = deleted = 0
    for line in _out(repo, "show", "--format=", "--numstat", commit).splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            added, deleted = added + int(parts[0]), deleted + int(parts[1])
    return deleted > added


def cherry_pick_ref(repo: Path, branch: str, commit: str, *, note: str) -> tuple[str | None, str]:
    """Apply `commit` onto `refs/heads/<branch>` without a worktree (CAS ref move).

    Returns (new head, "applied") or (None, "conflict" | "already present")."""
    from .kernel_mutation_guard import ensure_kernel_mutation_allowed
    ensure_kernel_mutation_allowed(repo, branch)   # never a frozen production ref
    ref = f"refs/heads/{branch}"
    onto = _out(repo, "rev-parse", "--verify", ref)
    parent = _out(repo, "rev-parse", f"{commit}^")
    merged = _git(repo, "merge-tree", "--write-tree", f"--merge-base={parent}", onto, commit)
    if merged.returncode == 1:
        return None, "conflict"
    if merged.returncode != 0:
        raise RuntimeError(f"git merge-tree: {merged.stderr.strip()[:300]}")
    tree = merged.stdout.split()[0]
    if tree == _out(repo, "rev-parse", f"{onto}^{{tree}}"):
        return None, "already present"
    name, email, date = _out(repo, "log", "-1", "--format=%an%x00%ae%x00%ad", commit).split("\0")
    message = (_out(repo, "log", "-1", "--format=%B", commit)
               + f"\n\n(cherry picked from commit {commit})\nCross-target: {note}\n")
    new = _out(repo, "commit-tree", tree, "-p", onto, "-F", "-", input=message,
               env={"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email,
                    "GIT_AUTHOR_DATE": date})
    _out(repo, "update-ref", ref, new, onto)
    return new, "applied"


def apply_in_worktree(worktree: Path, commit: str) -> str | None:
    """Cherry-pick `commit` onto the worktree's detached HEAD; None on conflict."""
    done = _git(worktree, "cherry-pick", "-x", "--allow-empty", commit)
    if done.returncode != 0:
        _git(worktree, "cherry-pick", "--abort")
        return None
    return _out(worktree, "rev-parse", "HEAD")


def _reset_tree(champion_tree: Path, head: str) -> None:
    _out(champion_tree, "reset", "--hard", head)


# ------------------------------------------------------------------ lineage moves

def _lanes(bound) -> list[str]:
    return sorted([bound.lane.name, *(peer.entry.name for peer in bound.peers)])


def on_lane(rows: Iterable[Mapping], patch: str, lane: str) -> bool:
    for row in rows:
        if row.get("patch_id") != patch:
            continue
        if row.get("event") == "keep_decision" and row.get("lane") == lane:
            return True
        if row.get("event") in {"recheck", "trunk_sync_commit"} and row.get("lane") == lane \
                and row.get("applied"):
            return True
    return False


def record_keep(bound, *, repo: Path, keep_commit: str, decision: Mapping[str, Any],
                cross: Mapping[str, Any], coverage: Mapping[str, Any],
                mechanism_id: str) -> dict:
    """Ledger the keep's decision; a shared keep is cherry-picked onto the trunk."""
    entry: dict[str, Any] = {
        "event": "keep_decision", "lane": bound.lane.name, "target_id": bound.lane.target_id,
        "branch": bound.lane.champion_branch, "keep_commit": keep_commit,
        "patch_id": patch_id(repo, keep_commit), "mechanism_id": mechanism_id,
        "decision": decision["decision"], "reason": decision["reason"],
        "peers": [{k: row.get(k) for k in ("target_id", "lane", "effect", "decisive",
                                           "passed", "reason")}
                  for row in cross.get("peers") or ()],
        "peer_coverage": {target: {k: cov.get(k) for k in ("observed", "unchanged",
                                                           "losses", "changed")}
                          for target, cov in coverage.items()}}
    if decision["decision"] == SHARED and bound.trunk_branch:
        try:
            new, result = cherry_pick_ref(repo, bound.trunk_branch, keep_commit,
                                          note=f"shared keep from {bound.lane.target_id}")
        except Exception as exc:  # noqa: BLE001 -- a missed propagation, recorded
            new, result = None, f"error: {type(exc).__name__}: {exc}"
        entry["trunk"] = {"branch": bound.trunk_branch, "commit": new, "result": result}
    return append(bound, entry)


def sync_from_trunk(bound, *, repo: Path, champion_tree: Path, branch: str) -> list[dict]:
    """Trunk commits missing from this lane's branch (by patch id), applied in order."""
    if not bound.trunk_branch:
        return []
    listing = _out(repo, "cherry", f"refs/heads/{branch}", f"refs/heads/{bound.trunk_branch}")
    rows = []
    for line in listing.splitlines():
        sign, _, commit = line.partition(" ")
        if sign != "+" or not commit:
            continue
        try:
            new, result = cherry_pick_ref(repo, branch, commit,
                                          note=f"trunk sync into {bound.lane.target_id}")
        except Exception as exc:  # noqa: BLE001
            new, result = None, f"error: {type(exc).__name__}: {exc}"
        if new is not None:
            _reset_tree(champion_tree, new)
        rows.append(append(bound, {
            "event": "trunk_sync_commit", "lane": bound.lane.name,
            "target_id": bound.lane.target_id, "branch": branch, "trunk_commit": commit,
            "patch_id": patch_id(repo, commit),
            "applied": new is not None or result == "already present",
            "new_head": new, "result": result}))
    return rows


def pending(bound, *, repo: Path, head: str) -> list[dict]:
    """Target-only keeps of OTHER lanes not on this lane and not yet re-checked at
    this lane's current champion head, oldest first."""
    rows = read(bound)
    lane = bound.lane.name
    seen = {(row.get("patch_id"), row.get("checker_head")) for row in rows
            if row.get("event") == "recheck" and row.get("lane") == lane}
    out, picked = [], set()
    for row in rows:
        if row.get("event") != "keep_decision" or row.get("decision") != TARGET_ONLY \
                or row.get("lane") == lane:
            continue
        patch = row.get("patch_id")
        if patch in picked or (patch, head) in seen or on_lane(rows, patch, lane):
            continue
        picked.add(patch)
        out.append(row)
    return out


def recheck_one(bound, *, repo: Path, champion_tree: Path, branch: str,
                measure: Callable[[Mapping[str, Any]], Mapping[str, Any]],
                advance: Callable[[str], None]) -> dict | None:
    """Re-check the oldest pending target-only keep on this lane's target.

    `measure(entry)` applies the keep onto this lane's champion in a lane worktree,
    builds and gates it and A/Bs it against this target's floor. It returns
    {"row": serving.compare row, "candidate_commit": sha} or {"error": text}.
    `advance(commit)` is called only on propagate; the branch has then already moved."""
    head = _out(repo, "rev-parse", "--verify", f"refs/heads/{branch}")
    queue = pending(bound, repo=repo, head=head)
    if not queue:
        return None
    entry = queue[0]
    try:
        measured = dict(measure(entry))
    except Exception as exc:  # noqa: BLE001 -- recorded, retried next advancement
        measured = {"error": f"{type(exc).__name__}: {exc}"}
    event: dict[str, Any] = {
        "event": "recheck", "lane": bound.lane.name, "target_id": bound.lane.target_id,
        "branch": branch, "checker_head": head, "origin_lane": entry.get("lane"),
        "origin_target": entry.get("target_id"), "keep_commit": entry["keep_commit"],
        "patch_id": entry["patch_id"], "mechanism_id": entry.get("mechanism_id"),
        "applied": False}
    if measured.get("error"):
        event.update(result="stay", why="failed", reason=measured["error"])
        return append(bound, event)
    row = measured.get("row")
    event["effect"] = None if row is None else row.get("effect")
    event["decisive"] = None if row is None else row.get("decisive")
    verdict = recheck_decision(row, simplifies=simplifies(repo, entry["keep_commit"]))
    event.update(verdict)
    candidate = measured.get("candidate_commit")
    if verdict["result"] == "propagate" and candidate:
        from .kernel_mutation_guard import ensure_kernel_mutation_allowed
        ensure_kernel_mutation_allowed(repo, branch)
        _out(repo, "update-ref", f"refs/heads/{branch}", candidate, head)
        _reset_tree(champion_tree, candidate)
        event.update(applied=True, new_head=candidate)
        advance(candidate)
        rows = read(bound) + [event]
        if bound.trunk_branch and all(on_lane(rows, entry["patch_id"], lane)
                                      for lane in _lanes(bound)):
            try:
                new, result = cherry_pick_ref(repo, bound.trunk_branch, entry["keep_commit"],
                                              note="every target now carries it")
            except Exception as exc:  # noqa: BLE001
                new, result = None, f"error: {type(exc).__name__}: {exc}"
            event["trunk"] = {"branch": bound.trunk_branch, "commit": new, "result": result}
    return append(bound, event)


__all__ = ["LEDGER", "LEDGER_SCHEMA", "SHARED", "TARGET_ONLY", "append", "apply_in_worktree",
           "cherry_pick_ref", "decide", "ledger_path", "on_lane", "patch_id", "pending",
           "read", "recheck_decision", "recheck_one", "record_keep", "simplifies",
           "sync_from_trunk"]
