#!/usr/bin/env python3
"""Cross-target lineage under the ONE-champion invariant: lane working branches fold
into the single global champion.

WHY (operator 2026-10-04). DS41's held keeps (b3e0b0902) cut Qwen3.8-Flash-Next prefill
by 29-41% (`/mnt/raid0/llm/tmp/q38fn-transfer-20261003/run2.log`). A shared change can
help one model and hurt another, and vetoing it throws the gain away. The first version
of this module (4f8f11c7) gave each lane its own champion branch plus a shared `trunk`.
That is a second lineage for one production tree, which the 2026-08-31 single-champion
incident forbids (`champion.INCIDENT`, `handoffs/active/autokernel-champion-aggregate.md`).
Production is ONE frozen CPU build plus ONE GPU build of ONE commit. Both are served from
the kernel store, and the compiled registry names them for every role. A per-model
lineage would imply per-model production builds, and nothing downstream supports those.
So:

* The TRUNK IS the global champion (`lanes.champion.branch`, which must equal
  `champion.CANONICAL_BRANCH`). There is no `ak-trunk`.
* Each lane keeps on its own WORKING branch (`working_branch` = the instance's
  `--experimental-branch`, like DS41's `experimental/fastload-ds41-*`). The anchor and
  the accumulator follow that branch, as before. A working branch is never a champion
  and never ships. Only what folds into the champion reaches production.
* `decide`: a keep FOLDS into the champion immediately if it regresses no peer target.
  That means every peer's A/B is within its floor (or bar), and every peer's
  executed-kernel manifest is unchanged (`kernel_coverage.peer_coverage`). A change is
  allowed only with a decisive peer gain. A keep that touches only the lane's exclusive
  paths cannot affect a peer, so it folds too.
  The fold uses plumbing: `merge-tree --write-tree` + `commit-tree` + a compare-and-swap
  `update-ref`. The source coverage gate G0 (`kernel_coverage.fold_check`, champion tip
  vs the fold result) runs before the ref moves. A checked-out champion tree is
  fast-forwarded only if it is clean and at the old tip.
* Every other keep is `target_only_pending_gate`. It stays on the lane's working branch,
  and the loop ENQUEUES a deterministic "gate this keep by model / arch / shape"
  hypothesis for that lane. That is one file in the lane store's `inbox/` (the channel
  the planner reads every iteration, sorted first) plus one `gate_enqueued` ledger row.
  The gating keep declares `GATES-KEEP: <patch id>` in its hypothesis. It is HELD with
  the keep it gates and never folds alone.
* A GATE MUST BE VISIBLE TO runtime_attestation (stack owner, 2026-10-04). It must key
  on something in argv or GGUF metadata: `general.architecture` / `LLM_ARCH_*`, tensor
  shapes, quant types, hparams. An environment-variable gate is invisible to
  attestation, because it is outside the declared env contract. `gate_mechanism` refuses
  a gating diff that adds a `getenv`, or that adds no model/arch/shape/type key.
  `run.py` vetoes such a keep before it commits. If one is recorded anyway, it is
  `gate_refused`, and its series cannot fold.
* When every pending keep in a lane's held series has a gating keep, the series is
  ELIGIBLE. Each OTHER lane fold-checks it at its own advancement (`check_one`), once per
  origin head. The check applies the whole held series onto that lane's working branch,
  builds it, runs the MUL_MAT oracle and kernel coverage, and A/Bs it against that
  lane's floor. Approval uses the same `decide` rule, so the checking lane's executed
  path must be unchanged. When every other lane approves at the same origin head, the
  series folds into the champion in order, after G0.
  A rejection enqueues the next gate attempt for the origin lane, up to
  `MAX_GATE_ATTEMPTS`. After that the keep is `gate_exhausted`: still held on the
  working branch, so OUT of production, and the ledger says so.
* `sync_from_champion`: at each keep boundary, before the anchor is rebuilt, champion
  commits the lane lacks (by patch id, `git cherry`) are brought onto the working branch.
  These are other lanes' folds and manual champion work. The champion aggregates
  everything, so every lane explores on top of it.

So model-specific behaviour reaches production only as runtime dispatch inside the one
binary. Production stays one binary per device.

Every decision is one JSON line in `<binding dir>/cross-target-ledger.jsonl`, appended
under flock (both instances share it). The same decision also rides into the keep's
experiments-DB row (`integrity_screen.cross_target`).
"""
from __future__ import annotations

import fcntl
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

LEDGER = "cross-target-ledger.jsonl"
LEDGER_SCHEMA = "epyc.autokernel.cross_target_ledger.v2"
FOLD, PENDING_GATE = "fold", "target_only_pending_gate"
GATE_COMMIT, GATE_REFUSED = "gate_commit", "gate_refused"
MAX_GATE_ATTEMPTS = 3
INBOX_PREFIX = "00-gate-"
_GATES = re.compile(r"GATES-KEEP:\s*([0-9a-f]{12,40})")
#: Any added reference to the process environment, called or not (a function
#: pointer, `getenv_s`, `environ`): all of it is outside the declared env contract.
_ENV_GATE = re.compile(r"\b(?:secure_)?getenv(?:_s)?\b|\benviron\b")
#: Keys runtime_attestation can see: GGUF metadata (architecture, hparams, tensor
#: shapes and types) or llama-server argv (cparams).
_MODEL_KEY = re.compile(
    r"LLM_ARCH_\w+|general\.architecture|\bhparams\b|\bcparams\b|\barch\s*[!=]=|"
    r"->ne\[\d\]|\.ne\[\d\]|GGML_TYPE_\w+|gguf_get_\w+|n_expert\w*|"
    r"n_embd\w*|n_head\w*|n_ff\w*")


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
    """fold (into the champion now) or target_only_pending_gate, for one keep. Pure."""
    if not cross.get("required"):
        return {"decision": FOLD, "reason": "every changed path is exclusive to this "
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
    return ({"decision": PENDING_GATE, "reason": "; ".join(reasons)} if reasons else
            {"decision": FOLD, "reason": "no peer regressed and every peer's executed "
                                         "kernel path is unchanged"})


def gate_declarations(*texts: str | None) -> list[str]:
    """Patch ids (12-40 hex) a keep declares it gates: `GATES-KEEP: <patch id>`."""
    return sorted({m.group(1) for text in texts if text for m in _GATES.finditer(text)})


def gate_mechanism(diff: str) -> tuple[bool, str]:
    """Is a gating diff keyed on something runtime_attestation can see? Pure.

    Refused: any added `getenv` (env gates sit outside the declared env contract), or
    no added reference to a GGUF-metadata / argv key (architecture, hparams, tensor
    shape or type, cparams)."""
    added = "\n".join(line[1:] for line in diff.splitlines()
                      if line.startswith("+") and not line.startswith("+++"))
    if _ENV_GATE.search(added):
        return False, ("the gate adds a getenv: an env-var gate is invisible to "
                       "runtime_attestation; key on GGUF metadata or argv instead")
    if not _MODEL_KEY.search(added):
        return False, ("the gate keys on no GGUF-metadata or argv field (architecture, "
                       "hparams, tensor shape/type, cparams)")
    return True, "gate keyed on GGUF metadata / argv"


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


def _pick_onto(repo: Path, onto: str, commit: str, *, note: str) -> tuple[str | None, str]:
    """A commit object for `commit` applied onto `onto`. No ref moves."""
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
    return new, "applied"


def cherry_pick_ref(repo: Path, branch: str, commit: str, *, note: str) -> tuple[str | None, str]:
    """Apply `commit` onto `refs/heads/<branch>` without a worktree (CAS ref move).

    Returns (new head, "applied") or (None, "conflict" | "already present")."""
    from .kernel_mutation_guard import ensure_kernel_mutation_allowed
    ensure_kernel_mutation_allowed(repo, branch)   # never a frozen production ref
    ref = f"refs/heads/{branch}"
    onto = _out(repo, "rev-parse", "--verify", ref)
    new, result = _pick_onto(repo, onto, commit, note=note)
    if new is not None:
        _out(repo, "update-ref", ref, new, onto)
    return new, result


def checkouts(repo: Path, branch: str) -> list[Path]:
    """Worktrees that have `branch` checked out."""
    trees, current = [], None
    for line in _out(repo, "worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            current = Path(line.split(" ", 1)[1])
        elif line == f"branch refs/heads/{branch}" and current is not None:
            trees.append(current)
    return trees


def fold_onto_champion(repo: Path, champion_branch: str, commits: Sequence[str], *,
                       note: str, fold_check: Callable[..., Mapping[str, Any]] | None = None
                       ) -> dict:
    """Fold `commits` (in order) into the ONE champion: pick, G0, CAS, fast-forward.

    Returns {"result": folded | conflict | already present | g0_refused |
    checkout_dirty, "commit": new tip or None, "picked": [...]}. Nothing moves unless
    every commit applies and G0 passes."""
    from .kernel_mutation_guard import ensure_kernel_mutation_allowed
    ensure_kernel_mutation_allowed(repo, champion_branch)
    ref = f"refs/heads/{champion_branch}"
    old = _out(repo, "rev-parse", "--verify", ref)
    tip, picked = old, []
    for commit in commits:
        new, result = _pick_onto(repo, tip, commit, note=note)
        if result == "conflict":
            return {"result": "conflict", "commit": None, "picked": picked,
                    "conflict_at": commit}
        if new is not None:
            tip = new
            picked.append(commit)
    if tip == old:
        return {"result": "already present", "commit": None, "picked": []}
    if fold_check is None:
        from .kernel_coverage import fold_check
    g0 = fold_check(repo=repo, base=old, candidate=tip)
    if not g0.get("passed"):
        return {"result": "g0_refused", "commit": None, "picked": picked,
                "g0": g0.get("reason")}
    trees = checkouts(repo, champion_branch)
    for tree in trees:
        dirty = _git(tree, "status", "--porcelain", "--untracked-files=no").stdout.strip()
        if dirty or _out(tree, "rev-parse", "HEAD") != old:
            return {"result": "checkout_dirty", "commit": None, "picked": picked,
                    "checkout": str(tree)}
    _out(repo, "update-ref", ref, tip, old)
    # The ref has moved: the fold IS done. What follows only refreshes checkouts. A
    # refusal there (an untracked file in the way of `-u`, a vanished tree) is not a
    # failed fold: reporting it as one would show the keep held while the champion
    # carries it, and the retry would find it "already present". The checkout is left
    # with HEAD at the new tip and index/worktree at the old one, which `status` shows
    # as dirty, so the next fold defers until a human reconciles that tree.
    stale = []
    for tree in trees:
        # Two-tree fast-forward of the clean checkout from the old tip to the new one.
        done = _git(tree, "read-tree", "-m", "-u", old, tip)
        if done.returncode != 0:
            stale.append({"checkout": str(tree), "error": done.stderr.strip()[:300]})
    out = {"result": "folded", "commit": tip, "picked": picked, "g0": g0.get("reason"),
           "checkouts": [str(t) for t in trees]}
    if stale:
        out["checkouts_stale"] = stale
    return out


def apply_in_worktree(worktree: Path, commit: str) -> str | None:
    """Cherry-pick `commit` onto the worktree's detached HEAD; None on conflict."""
    done = _git(worktree, "cherry-pick", "-x", "--allow-empty", commit)
    if done.returncode != 0:
        _git(worktree, "cherry-pick", "--abort")
        return None
    return _out(worktree, "rev-parse", "HEAD")


def _reset_tree(champion_tree: Path, head: str) -> None:
    _out(champion_tree, "reset", "--hard", head)


def held(repo: Path, champion_branch: str, branch: str) -> list[str]:
    """Commits on a working branch that the champion lacks (by patch id), oldest first."""
    listing = _out(repo, "cherry", f"refs/heads/{champion_branch}", f"refs/heads/{branch}")
    return [commit for sign, _, commit in (line.partition(" ") for line in listing.splitlines())
            if sign == "+" and commit]


# ------------------------------------------------------------------ lineage moves

def _keep_rows(rows: Iterable[Mapping], lane: str) -> dict[str, dict]:
    return {row["patch_id"]: dict(row) for row in rows
            if row.get("event") == "keep_decision" and row.get("lane") == lane}


def record_keep(bound, *, repo: Path, keep_commit: str, decision: Mapping[str, Any],
                cross: Mapping[str, Any], coverage: Mapping[str, Any], mechanism_id: str,
                store: Path | None = None, declaration_texts: Sequence[str | None] = (),
                fold_check: Callable[..., Mapping[str, Any]] | None = None) -> dict:
    """Ledger the keep. FOLD -> into the champion now; PENDING_GATE -> held, with a gate
    hypothesis enqueued; a keep declaring `GATES-KEEP:` for a held keep -> held with it
    (GATE_COMMIT), or GATE_REFUSED when its mechanism is invisible to attestation."""
    rows = read(bound)
    lane = bound.lane.name
    pid = patch_id(repo, keep_commit)
    declared = gate_declarations(*declaration_texts)
    gates = sorted(p for p, row in _keep_rows(rows, lane).items()
                   if row.get("decision") == PENDING_GATE
                   and any(p.startswith(g) for g in declared))
    final = dict(decision)
    if gates:
        ok, why = gate_mechanism(_out(repo, "show", "--format=", "--no-color", keep_commit))
        final = ({"decision": GATE_COMMIT, "reason": "gates held keep(s) "
                  + ", ".join(g[:12] for g in gates) + "; folds with them as one series "
                  "once every other lane approves it (" + decision["reason"] + ")"}
                 if ok else {"decision": GATE_REFUSED, "reason": why})
    entry: dict[str, Any] = {
        "event": "keep_decision", "lane": lane, "target_id": bound.lane.target_id,
        "branch": bound.lane.working_branch, "keep_commit": keep_commit, "patch_id": pid,
        "mechanism_id": mechanism_id, "decision": final["decision"],
        "reason": final["reason"], "gates": gates if final["decision"] == GATE_COMMIT else [],
        **({"refused_gates": gates} if final["decision"] == GATE_REFUSED else {}),
        "peers": [{k: row.get(k) for k in ("target_id", "lane", "effect", "decisive",
                                           "passed", "reason")}
                  for row in cross.get("peers") or ()],
        "peer_coverage": {target: {k: cov.get(k) for k in ("observed", "unchanged",
                                                           "losses", "changed")}
                          for target, cov in coverage.items()}}
    if final["decision"] == FOLD and bound.champion_branch:
        try:
            entry["fold"] = {"branch": bound.champion_branch, **fold_onto_champion(
                repo, bound.champion_branch, [keep_commit], fold_check=fold_check,
                note=f"fold from {bound.lane.target_id} ({lane})")}
        except Exception as exc:  # noqa: BLE001 -- held on the working branch, retried
            entry["fold"] = {"branch": bound.champion_branch, "commit": None,
                             "result": f"error: {type(exc).__name__}: {exc}"}
    row = append(bound, entry)
    if final["decision"] == PENDING_GATE and store is not None:
        enqueue_gate(bound, store=store, keep=row, attempt=1)
    return row


def _inbox_name(patch: str, attempt: int) -> str:
    return f"{INBOX_PREFIX}{patch[:12]}-a{attempt}.md"


def enqueue_gate(bound, *, store: Path, keep: Mapping[str, Any], attempt: int,
                 rejection: str | None = None) -> dict:
    """Queue ONE deterministic gating hypothesis for a target-only keep (lane inbox)."""
    inbox = Path(store) / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    for stale in inbox.glob(f"{INBOX_PREFIX}{keep['patch_id'][:12]}-a*.md"):
        stale.unlink()
    peers = "; ".join(f"{p.get('target_id')}: {p.get('reason')}" for p in keep.get("peers") or ())
    text = (
        f"# GATE HELD KEEP {keep['keep_commit'][:12]} (attempt {attempt}/{MAX_GATE_ATTEMPTS})\n\n"
        f"Keep `{keep['keep_commit']}` (`{keep.get('mechanism_id')}`) cleared "
        f"{keep.get('target_id')}'s gates but regressed or changed a peer target: {peers}"
        + (f". Last rejection: {rejection}" if rejection else "") + ".\n\n"
        "It is HELD on this lane's working branch. Until it is gated it cannot fold into "
        "the single champion, so it cannot reach production. Propose a patch that keeps "
        "the mechanism for this target ONLY, through runtime dispatch inside the one "
        "binary. The executed kernel path of every other model must stay byte-identical "
        "to the champion's. Do not tune the mechanism itself.\n\n"
        "The gate MUST key on something runtime_attestation can see: GGUF metadata "
        "(`general.architecture` / `LLM_ARCH_*`, hparams, tensor shapes, quant types) or "
        "llama-server argv (cparams). An environment-variable gate (`getenv`) is REFUSED, "
        "because it is invisible to attestation and outside the declared env contract.\n\n"
        f"Put this line in the hypothesis statement: `GATES-KEEP: {keep['patch_id'][:12]}`\n")
    path = inbox / _inbox_name(keep["patch_id"], attempt)
    path.write_text(text, encoding="utf-8")
    return append(bound, {"event": "gate_enqueued", "lane": bound.lane.name,
                          "target_id": bound.lane.target_id, "patch_id": keep["patch_id"],
                          "keep_commit": keep["keep_commit"], "attempt": attempt,
                          "inbox_file": str(path), "rejection": rejection})


def refresh_gates(bound, *, repo: Path, store: Path, branch: str,
                  fold_check: Callable[..., Mapping[str, Any]] | None = None) -> list[dict]:
    """This lane's held-series housekeeping at a keep boundary. All of it is
    deterministic:

    * retry the fold of any FOLD-decided keep that is still held (an earlier conflict
      or dirty checkout);
    * retire the inbox entries of keeps the champion now carries;
    * after a fold-check rejection or a refused gate, enqueue the next gate attempt, or
      mark the keep gate_exhausted."""
    if not bound.champion_branch:
        return []
    lane, out = bound.lane.name, []
    rows = read(bound)
    keeps = _keep_rows(rows, lane)
    held_ids = {patch_id(repo, c): c for c in held(repo, bound.champion_branch, branch)}
    for pid, commit in held_ids.items():
        if keeps.get(pid, {}).get("decision") == FOLD:
            try:
                result = fold_onto_champion(repo, bound.champion_branch, [commit],
                                            fold_check=fold_check,
                                            note=f"fold retry from {bound.lane.target_id}")
            except Exception as exc:  # noqa: BLE001
                result = {"result": f"error: {type(exc).__name__}: {exc}", "commit": None}
            out.append(append(bound, {"event": "fold_retry", "lane": lane, "patch_id": pid,
                                      "keep_commit": commit, "fold": result}))
    inbox = Path(store) / "inbox"
    for path in sorted(inbox.glob(f"{INBOX_PREFIX}*.md")) if inbox.is_dir() else ():
        prefix = path.name[len(INBOX_PREFIX):].split("-a", 1)[0]
        if not any(pid.startswith(prefix) for pid in held_ids):
            path.unlink()
            out.append(append(bound, {"event": "gate_resolved", "lane": lane,
                                      "patch_prefix": prefix, "inbox_file": str(path),
                                      "why": "the champion carries the keep"}))
    for pid in held_ids:
        row = keeps.get(pid)
        if row is None or row.get("decision") != PENDING_GATE:
            continue
        mine = [r for r in rows if r.get("patch_id") == pid and r.get("lane") == lane]
        if any(r.get("event") == "gate_exhausted" for r in mine):
            continue
        enqueued = [r for r in mine if r.get("event") == "gate_enqueued"]
        last_at = max((r["at"] for r in enqueued), default=0.0)
        rejected = [r for r in rows if r["at"] > last_at and (
            (r.get("event") == "fold_check" and r.get("origin_lane") == lane
             and r.get("result") == "reject" and pid in (r.get("series_patches") or ()))
            or (r.get("event") == "keep_decision" and r.get("lane") == lane
                and r.get("decision") == GATE_REFUSED and pid in (r.get("refused_gates") or ())))]
        if not rejected:
            continue
        attempt = len(enqueued) + 1
        if attempt > MAX_GATE_ATTEMPTS:
            for path in inbox.glob(f"{INBOX_PREFIX}{pid[:12]}-a*.md"):
                path.unlink()
            out.append(append(bound, {
                "event": "gate_exhausted", "lane": lane, "patch_id": pid,
                "keep_commit": row["keep_commit"], "attempts": len(enqueued),
                "why": "held on the working branch only: NOT in the champion, NOT in "
                       "production"}))
        else:
            out.append(enqueue_gate(bound, store=store, keep=row, attempt=attempt,
                                    rejection=rejected[-1].get("reason")))
    return out


def sync_from_champion(bound, *, repo: Path, champion_tree: Path, branch: str) -> list[dict]:
    """Champion commits missing from this working branch (by patch id), applied in order."""
    if not bound.champion_branch:
        return []
    listing = _out(repo, "cherry", f"refs/heads/{branch}", f"refs/heads/{bound.champion_branch}")
    rows = []
    for line in listing.splitlines():
        sign, _, commit = line.partition(" ")
        if sign != "+" or not commit:
            continue
        try:
            new, result = cherry_pick_ref(repo, branch, commit,
                                          note=f"champion sync into {bound.lane.target_id}")
        except Exception as exc:  # noqa: BLE001
            new, result = None, f"error: {type(exc).__name__}: {exc}"
        if new is not None:
            _reset_tree(champion_tree, new)
        rows.append(append(bound, {
            "event": "champion_sync_commit", "lane": bound.lane.name,
            "target_id": bound.lane.target_id, "branch": branch, "champion_commit": commit,
            "patch_id": patch_id(repo, commit),
            "applied": new is not None or result == "already present",
            "new_head": new, "result": result}))
    return rows


def eligible(bound, *, repo: Path) -> list[dict]:
    """Other lanes' held series ready for THIS lane's fold-check, in peer order.

    A series is a lane's held commits: its working branch minus the champion, by patch
    id. It is eligible when every target-only keep in it has a gating keep in it, no
    keep in it is a refused gate, and this lane has not yet checked it at the origin's
    current head."""
    if not bound.champion_branch:
        return []
    rows = read(bound)
    lane, out = bound.lane.name, []
    for peer in bound.peers:
        origin, branch = peer.entry.name, peer.entry.working_branch
        if not branch:
            continue
        head = _out(repo, "rev-parse", "--verify", f"refs/heads/{branch}")
        if any(r.get("event") == "fold_check" and r.get("lane") == lane
               and r.get("origin_lane") == origin and r.get("origin_head") == head
               for r in rows):
            continue
        series = held(repo, bound.champion_branch, branch)
        if not series:
            continue
        keeps = _keep_rows(rows, origin)
        patches = [patch_id(repo, c) for c in series]
        if any(keeps.get(p, {}).get("decision") == GATE_REFUSED for p in patches):
            continue
        pending = [p for p in patches if keeps.get(p, {}).get("decision") == PENDING_GATE]
        gated = {g for p in patches for g in keeps.get(p, {}).get("gates") or ()}
        if not pending or not set(pending) <= gated:
            continue
        out.append({"origin_lane": origin, "origin_target": peer.entry.target_id,
                    "origin_branch": branch, "origin_head": head, "commits": series,
                    "series_patches": patches, "pending": pending})
    return out


def check_one(bound, *, repo: Path,
              measure: Callable[[Mapping[str, Any]], Mapping[str, Any]],
              fold_check: Callable[..., Mapping[str, Any]] | None = None) -> dict | None:
    """Fold-check the first eligible held series of another lane on THIS lane's target.

    `measure(series)` applies `series["commits"]` onto this lane's working branch in a
    lane worktree, builds and gates it, and A/Bs it against this target's floor.
    It returns {"row": serving.compare row, "passed": bool, "reason": str, "coverage":
    peer_coverage of this target} or {"error": text}. It never moves this lane's branch:
    the series reaches every lane through the champion."""
    queue = eligible(bound, repo=repo)
    if not queue:
        return None
    series = queue[0]
    try:
        measured = dict(measure(series))
    except Exception as exc:  # noqa: BLE001 -- recorded as a rejection
        measured = {"error": f"{type(exc).__name__}: {exc}"}
    own = bound.lane.target_id
    event: dict[str, Any] = {"event": "fold_check", "lane": bound.lane.name, "target_id": own,
                             **{k: series[k] for k in ("origin_lane", "origin_target",
                                                       "origin_head", "series_patches")}}
    if measured.get("error"):
        event.update(result="reject", reason=measured["error"])
        return append(bound, event)
    row = measured.get("row") or {}
    verdict = decide({"required": True, "peers": [{
        "target_id": own, "passed": bool(measured.get("passed")),
        "reason": measured.get("reason"), "effect": row.get("effect"),
        "decisive": row.get("decisive")}]}, {own: measured.get("coverage") or {}})
    event.update(result="approve" if verdict["decision"] == FOLD else "reject",
                 reason=verdict["reason"], effect=row.get("effect"),
                 decisive=row.get("decisive"))
    if event["result"] == "approve":
        checkers = {p.entry.name for p in bound.peers if p.entry.name != series["origin_lane"]}
        checkers.add(bound.lane.name)
        approved = {r.get("lane") for r in read(bound) + [event]
                    if r.get("event") == "fold_check"
                    and r.get("origin_lane") == series["origin_lane"]
                    and r.get("origin_head") == series["origin_head"]
                    and r.get("result") == "approve"}
        if checkers <= approved:
            try:
                event["fold"] = {"branch": bound.champion_branch, **fold_onto_champion(
                    repo, bound.champion_branch, series["commits"], fold_check=fold_check,
                    note=f"gated series from {series['origin_target']}, approved by "
                         + ", ".join(sorted(checkers)))}
            except Exception as exc:  # noqa: BLE001
                event["fold"] = {"branch": bound.champion_branch, "commit": None,
                                 "result": f"error: {type(exc).__name__}: {exc}"}
    return append(bound, event)


__all__ = ["FOLD", "GATE_COMMIT", "GATE_REFUSED", "LEDGER", "LEDGER_SCHEMA",
           "MAX_GATE_ATTEMPTS", "PENDING_GATE", "append", "apply_in_worktree", "check_one",
           "checkouts", "cherry_pick_ref", "decide", "eligible", "enqueue_gate",
           "fold_onto_champion", "gate_declarations", "gate_mechanism", "held",
           "ledger_path", "patch_id", "read", "record_keep", "refresh_gates",
           "sync_from_champion"]
