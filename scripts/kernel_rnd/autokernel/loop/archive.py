#!/usr/bin/env python3
"""Champion source identity and immutable measurement history.

A champion commit identifies re-executable source; it is not evidence that a build,
benchmark, or scientific result occurred. Those records live separately in the
immutable experiment journal, with negatives retained alongside accepted attempts.
Advancing source never creates or upgrades an artifact/evidence claim.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from ..controller import experiments
from . import kernel_mutation_guard


class RatchetRefused(RuntimeError):
    """The champion branch could not be advanced."""


def _retain_bytes(path: Path, raw: bytes) -> None:
    """Publish immutable archive bytes; an interrupted publication can be retried."""
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".patch-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode) \
                        or stream.read(len(raw) + 1) != raw:
                    raise RatchetRefused(f"existing immutable patch artifact differs: {path}")
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def retain_patch(store_root: Path, repo: Path, *, lane: str,
                 mechanism_id: str = "interrupted") -> Path | None:
    """Keep original HEAD and working source before an owned lane is reset.

    All tracked changes remain covered, including build recipes and documentation.
    Newly included untracked files are limited to literal, regular UTF-8 kernel
    source under ggml/src or src. This is a source archive, not build/measurement
    evidence or permission to resume execution. The real Git index is untouched.
    """
    repo = _verified_repo(Path(repo))
    head = _git(repo, "rev-parse", "HEAD")

    def raw_git(*args):
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              env=_git_env(), timeout=600)
        if done.returncode:
            raise RatchetRefused(f"patch capture git {args[0]} failed")
        return done.stdout

    patch = raw_git("diff", "--binary", "--full-index", "--no-ext-diff",
                    "--no-textconv", head, "--")
    additions = []
    untracked = raw_git("ls-files", "--others", "--exclude-standard", "-z",
                        "--", "ggml/src/", "src/")
    source_suffixes = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
                       ".cu", ".cuh", ".inc", ".inl", ".s", ".S"}
    for entry in sorted(item for item in untracked.split(b"\0") if item):
        name = os.fsdecode(entry)
        if Path(name).suffix not in source_suffixes:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_./+-]+", name) or any(
                part.startswith(".") for part in Path(name).parts):
            raise RatchetRefused("untracked kernel source has an unsafe path")
        path = repo / name
        if any(parent.is_symlink() for parent in (path, *path.parents) if parent != repo):
            raise RatchetRefused(f"untracked kernel source is a symlink: {name}")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > 4 * 1024 * 1024:
                raise RatchetRefused(f"untracked kernel source is not bounded text: {name}")
            raw = stream.read(4 * 1024 * 1024 + 1)
            after = os.fstat(stream.fileno())
        if (before.st_size != len(raw) or before.st_mtime_ns != after.st_mtime_ns
                or before.st_size != after.st_size or b"\0" in raw):
            raise RatchetRefused(f"untracked kernel source changed or is binary: {name}")
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RatchetRefused(f"untracked kernel source is not UTF-8 text: {name}") from exc
        mode = "100755" if before.st_mode & 0o111 else "100644"
        new = f"diff --git a/{name} b/{name}\nnew file mode {mode}\n".encode()
        if raw:
            lines = raw.split(b"\n")
            if raw.endswith(b"\n"):
                lines.pop()
            new += f"--- /dev/null\n+++ b/{name}\n@@ -0,0 +1,{len(lines)} @@\n".encode()
            for line in lines:
                new += b"+" + line + b"\n"
            if not raw.endswith(b"\n"):
                new += b"\\ No newline at end of file\n"
        patch += new
        additions.append(name)
    if not patch:
        return None
    if _git(repo, "rev-parse", "HEAD") != head:
        raise RatchetRefused("lane HEAD moved during patch capture; no reset")
    digest = hashlib.sha256(head.encode() + b"\n" + patch).hexdigest()
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", mechanism_id)[:80] or "unnamed"
    lane_label = re.sub(r"[^A-Za-z0-9_.-]", "_", lane)[:40] or "lane"
    directory = Path(store_root) / "patches"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{label}.{lane_label}.{digest}.patch"
    _retain_bytes(path, patch)
    metadata = {"schema": "epyc.autokernel.source_patch_archive.v1", "original_head": head,
                "worktree": str(repo), "lane": lane, "mechanism_id": mechanism_id,
                "patch_file": path.name, "patch_sha256": hashlib.sha256(patch).hexdigest(),
                "untracked_source_paths": additions, "scope": "source_only_not_execution_evidence"}
    _retain_bytes(path.with_suffix(".json"),
                  (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode())
    return path


_AMBIENT_REPO_ENV = ("GIT_DIR", "GIT_WORK_TREE")


def _git_env(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an isolated Git environment without mutating the caller's mapping."""
    command_env = os.environ.copy()
    for name in (*_AMBIENT_REPO_ENV, "GIT_INDEX_FILE"):
        command_env.pop(name, None)
    if overrides:
        command_env.update(overrides)
    return command_env


def _git(repo: Path, *args: str, check: bool = True,
         env: Mapping[str, str] | None = None, input_text: str | None = None,
         raw_output: bool = False) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, timeout=600,
                          env=_git_env(env), input=input_text)
    if check and done.returncode != 0:
        raise RatchetRefused(f"git {' '.join(args)}: {done.stderr.strip()[:400]}")
    return done.stdout if raw_output else done.stdout.strip()


def _verified_repo(repo: Path) -> Path:
    redirected = [name for name in _AMBIENT_REPO_ENV if name in os.environ]
    if redirected:
        raise RatchetRefused(
            f"ambient Git repository redirection is not allowed: {', '.join(redirected)}")
    requested = repo.resolve()
    actual = Path(_git(requested, "rev-parse", "--show-toplevel")).resolve()
    if actual != requested:
        raise RatchetRefused(
            f"repository path must be its actual root: requested {requested}, root {actual}")
    return actual


def _validated_paths(repo: Path, paths: tuple[str, ...]) -> tuple[str, ...]:
    if not paths:
        raise RatchetRefused("a champion advance must name the files it changes")
    validated = []
    for raw in paths:
        if not isinstance(raw, str) or not raw:
            raise RatchetRefused("accepted paths must be non-empty strings")
        parts = raw.split("/")
        if (Path(raw).is_absolute() or raw != raw.strip() or raw.startswith(":")
                or any(part in {"", ".", ".."} for part in parts)
                or any(char in raw for char in "*?[")
                or any(ord(char) < 32 or ord(char) == 127 for char in raw)):
            raise RatchetRefused(f"unsafe accepted path {raw!r}; literal relative files only")
        candidate = repo / raw
        object_type = _git(repo, "cat-file", "-t", f"HEAD:{raw}", check=False)
        if candidate.is_dir() or object_type == "tree":
            raise RatchetRefused(f"accepted path is a directory, not a file: {raw!r}")
        if raw in validated:
            raise RatchetRefused(f"duplicate accepted path {raw!r}")
        validated.append(raw)
    return tuple(validated)


def _hook(repo: Path, name: str, env: Mapping[str, str], *args: str,
          check: bool = True) -> None:
    hook_name = _git(repo, "rev-parse", "--git-path", f"hooks/{name}")
    hook = Path(hook_name)
    if not hook.is_absolute():
        hook = repo / hook
    if not hook.is_file() or not os.access(hook, os.X_OK):
        return
    command_env = _git_env(env)
    try:
        done = subprocess.run([str(hook), *args], cwd=repo, env=command_env,
                              capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        if check:
            raise RatchetRefused(f"{name} hook could not run: {exc}") from exc
        print(f"warning: {name} hook could not report after committed ref update: {exc}",
              file=sys.stderr)
        return
    if check and done.returncode != 0:
        detail = (done.stderr or done.stdout).strip()[:400]
        raise RatchetRefused(f"{name} hook refused champion commit: {detail}")
    if not check and done.returncode != 0:
        print(f"warning: {name} hook reported after committed ref update: "
              f"{(done.stderr or done.stdout).strip()[:400]}", file=sys.stderr)


def _changed_paths(repo: Path, expected: str, env: Mapping[str, str]) -> set[str]:
    raw = _git(repo, "diff", "--cached", "--name-only", "-z", expected, "--",
               env=env, raw_output=True)
    return {path for path in raw.split("\0") if path}


def _require_private_diff(repo: Path, expected: str, paths: tuple[str, ...],
                          env: Mapping[str, str]) -> None:
    changed = _changed_paths(repo, expected, env)
    if not changed:
        raise RatchetRefused("accepted paths produced no committable change")
    unexpected = changed - set(paths)
    if unexpected:
        raise RatchetRefused(
            f"commit hook staged paths outside the accepted patch: {sorted(unexpected)}")
    # A hook may inspect or even format the accepted working files, but the private
    # index must still name their explicit current bytes. It may never substitute a
    # peer's differently staged version.
    _git(repo, "diff", "--quiet", "--", *paths, env=env)


def _verify_head_binding(repo: Path, symbolic: str, expected: str) -> None:
    current_root = Path(_git(repo, "rev-parse", "--show-toplevel")).resolve()
    if current_root != repo:
        raise RatchetRefused(
            f"repository root changed before commit: expected {repo}, found {current_root}")
    current_symbolic = _git(repo, "symbolic-ref", "-q", "HEAD", check=False)
    _guard_kernel_mutation(repo, current_symbolic)
    if current_symbolic != symbolic:
        before = symbolic or "detached HEAD"
        after = current_symbolic or "detached HEAD"
        raise RatchetRefused(f"HEAD binding changed before commit: {before} -> {after}")
    current_head = _git(repo, "rev-parse", "HEAD")
    if current_head != expected:
        raise RatchetRefused(
            f"HEAD moved before commit: expected {expected}, found {current_head}")


def _guard_kernel_mutation(repo: Path, branch: str | None) -> None:
    try:
        kernel_mutation_guard.ensure_kernel_mutation_allowed(repo, branch)
    except kernel_mutation_guard.FrozenKernelMutationRefused as exc:
        raise RatchetRefused(str(exc)) from exc


def keep(repo: Path, *, branch: str, message: str, paths: tuple[str, ...]) -> str:
    """CAS an accepted source patch onto the current branch and return its commit.

    A private index is seeded from the exact captured HEAD and stages only validated
    literal files from the working tree, including tracked deletions. ``keep`` itself
    never resets or rewrites the caller's shared index or peer working files; normal
    hooks retain their ordinary ability to edit the working tree. After the ref
    advances that untouched index can legitimately appear stale relative to the new
    HEAD; do not "repair" it, even for selected paths, because those entries may
    contain distinct peer staging.

    Normal pre-commit, prepare-commit-msg, and commit-msg constraints run against the
    private index. ``commit-tree`` preserves Git author/committer identity and optional
    configured signing; atomic ``update-ref <new> <expected-old>`` prevents overwriting
    a concurrent advance. Post-commit is notification-only and runs after the CAS; as
    with normal Git it cannot veto or roll back an already-created commit.

    This commits source identity only. It infers no build, artifact, measurement, or
    evidence claim.
    """
    repo = _verified_repo(Path(repo))
    accepted = _validated_paths(repo, paths)
    symbolic = _git(repo, "symbolic-ref", "-q", "HEAD", check=False)
    _guard_kernel_mutation(repo, symbolic)
    if branch == "HEAD":
        target = symbolic or "HEAD"
        no_deref = not symbolic
    else:
        short = branch.removeprefix("refs/heads/")
        _git(repo, "check-ref-format", "--branch", short)
        target = f"refs/heads/{short}"
        if symbolic != target:
            current = symbolic.removeprefix("refs/heads/") if symbolic else "detached HEAD"
            raise RatchetRefused(
                f"requested branch {short!r} is not checked out (current: {current})")
        no_deref = False
    expected = _git(repo, "rev-parse", "HEAD")

    with tempfile.TemporaryDirectory(prefix="autokernel-private-index-") as tmp:
        index = Path(tmp) / "index"
        env = {"GIT_INDEX_FILE": str(index), "GIT_LITERAL_PATHSPECS": "1"}
        _git(repo, "read-tree", expected, env=env)
        _git(repo, "add", "-A", "--", *accepted, env=env)
        _require_private_diff(repo, expected, accepted, env)
        _hook(repo, "pre-commit", env)

        message_path = Path(tmp) / "COMMIT_EDITMSG"
        cleaned = _git(repo, "stripspace", input_text=message)
        if not cleaned:
            raise RatchetRefused("champion commit message is empty after cleanup")
        message_path.write_text(cleaned + "\n", encoding="utf-8")
        _hook(repo, "prepare-commit-msg", env, str(message_path), "message")
        _hook(repo, "commit-msg", env, str(message_path))
        if not message_path.read_text(encoding="utf-8").strip():
            raise RatchetRefused("commit-msg hook left an empty champion message")

        _require_private_diff(repo, expected, accepted, env)
        tree = _git(repo, "write-tree", env=env)
        commit_args = ["commit-tree", tree, "-p", expected, "-F", str(message_path)]
        signing = _git(repo, "config", "--type=bool", "--default=false", "--get",
                       "commit.gpgSign")
        if signing == "true":
            commit_args.insert(1, "-S")
        new_head = _git(repo, *commit_args, env=env)
        # update-ref rejects a symref verification and an update of its referent as
        # duplicate updates in one transaction. Recheck both immediately before the
        # CAS; expected-old still atomically protects the captured parent.
        _verify_head_binding(repo, symbolic, expected)
        update_args = ["update-ref"]
        if no_deref:
            update_args.append("--no-deref")
        update_args.extend([target, new_head, expected])
        _git(repo, *update_args, env=env)
        try:
            _hook(repo, "post-commit", env, check=False)
        except Exception as exc:
            # The ref has landed. A notification failure cannot turn success into an
            # ambiguous caller-visible exception claiming no commit was made.
            print(f"warning: post-commit notification failed after committed ref "
                  f"update: {exc}", file=sys.stderr)
        return new_head


def record(store_root: Path, attempt: Mapping[str, Any], *, epoch: str,
           recorded_at: str, campaign_id: str, on_serving_export=None) -> bool:
    """Append one attempt to durable memory and refresh `experiments.md`.

    Idempotent on attempt identity, so a resumed loop re-recording its own rows
    cannot inflate the history it will later read back.
    """
    with experiments.ExperimentStore(store_root) as store:
        added = store.record(attempt, epoch=epoch, recorded_at=recorded_at,
                             campaign_id=campaign_id)
        store.write_markdown(epoch=epoch)
        if added:
            try:
                from . import serving_beliefs
                receipt = serving_beliefs.export(store_root, attempt, campaign_id=campaign_id,
                                                epoch=epoch, recorded_at=recorded_at)
                if receipt is not None and on_serving_export is not None:
                    on_serving_export(receipt)
            except Exception as exc:
                # The experiment is already durable. Auxiliary export cannot change
                # its outcome, trigger a relaunch or claim that settlement failed.
                print(f"warning: serving belief export failed after durable archive: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return added


def recall(store_root: Path, *, epoch: str, limit: int = 40,
           ranking_authorized: bool = False) -> list[dict]:
    """What has been tried, cross-epoch records marked stale.

    Recency order by default. `ranking_authorized` is `P-AK-SEARCH-1-A3`'s epoch-scoped
    ranking, off unless the caller asks (`run.py --rank-prior-experiments`): it returns
    an order of merit instead, with cross-epoch magnitudes redacted. The default is
    unchanged, and passing the flag is a decision with a name on it.
    """
    with experiments.ExperimentStore(store_root) as store:
        return store.recall(epoch=epoch, limit=limit,
                            ranking_authorized=ranking_authorized)


CANONICAL_HISTORY_ROOT = Path("/mnt/raid0/llm/autokernel/loop-memory")


def _shared_rows(source, scope):
    """Bounded formation lookup: useful outcomes cannot be buried by transients.

    Relevance orders suggestions, not evidence applicability. No magnitude enters
    this selection. Keep qualitative negative/invalid outcomes and their caveats.
    """
    kept = ("kept", "runtime_kept")
    measured = ("measured_null", "runtime_observed", "measurement_invalid",
                "confirm_vetoed", "screened_out")
    pools = [source.recall(epoch="", limit=16, include_source_scope=True, statuses=kept),
             source.recall(epoch="", limit=16, include_source_scope=True, statuses=measured),
             source.recall(epoch="", limit=16, include_source_scope=True,
                           statuses=kept + measured, exclude_statuses=True)]

    def relevance(row):
        original = row["research_scope"] or {}
        model = original.get("model")
        model = model.get("path") if isinstance(model, dict) else model
        return tuple(bool(value is not None and value == scope.get(key)) for key, value in (
            ("model", model), ("quant", original.get("quant")),
            ("backend", original.get("backend")),
            ("measurement_surface", original.get("measurement_surface"))))

    pools = [sorted(rows, key=relevance, reverse=True) for rows in pools]
    rows, seen = [], set()

    def take(pool, count):
        while pool and count and len(rows) < 5:
            row = pool.pop(0)
            key = (row["mechanism_id"] or row["attempt_id"], row["status"])
            if key not in seen:
                rows.append(row)
                seen.add(key)
                count -= 1

    for pool, count in zip(pools, (1, 3, 1)):
        take(pool, count)
    for pool in pools:
        take(pool, 5)
    return rows


class SharedHistory:
    """Read-only formation suggestions, separate from current-target evidence."""

    def __init__(self, roots, *, current_store: Path, batch_directory=None):
        current = Path(current_store).resolve()
        self.roots = tuple(dict.fromkeys(Path(root).resolve() for root in roots
                                        if Path(root).resolve() != current))
        # The original serial owner supplies --out .../batch-NNNNNN. Use only
        # that declared identity, never scan directories or infer process state.
        # The sweep term prevents a roster-sized batch stride aliasing forever
        # to the same roots when every child performs only one iteration.
        batch = re.fullmatch(r"batch-(\d+)", Path(batch_directory).name) if batch_directory else None
        number = int(batch[1]) if batch else 0
        self._cursor = 8 * (number + number // max(1, len(self.roots)))
        self._lock = threading.Lock()

    def recall(self, *, scope=None) -> dict:
        with self._lock:
            # Rotate the bounded read budget, including for a 17-target roster.
            # Omitted roots are explicit; no claim of complete memory coverage.
            count = min(8, len(self.roots))
            selected = [self.roots[(self._cursor + i) % len(self.roots)] for i in range(count)]
            self._cursor = (self._cursor + count) % len(self.roots) if self.roots else 0
        result = {"status": "shared_history_nontransfer", "rows": [], "errors": [],
                  "queried_roots": [str(root) for root in selected],
                  "omitted_roots": [str(root) for root in self.roots if root not in selected],
                  "omitted_rows": 0, "comparable_measurement": False,
                  "selection": "bounded recent pools: 16 keeps, 16 measured/invalid, 16 other per root; "
                               "up to 5 distinct mechanism/status suggestions, qualitative scope relevance only",
                  "use": "historical mechanism suggestions only; not local gain or refutation"}
        for root in selected:
            try:
                with experiments.ExperimentStore(root, read_only=True) as source:
                    rows = _shared_rows(source, scope or {})
                for row in rows:
                    row.update(source_store=str(root), comparable_measurement=False,
                               same_epoch=False, stale_epoch=True, magnitude_redacted=True,
                               transfer="unproven_not_local_gain_or_refutation")
                    for key in experiments._MAGNITUDE_FIELDS:
                        row[key] = None
                    if row["research_scope"] is None:
                        row["research_scope"] = {"model": None, "quant": None, "recipe": None,
                            "measurement_surface": None,
                            "unknown_reason": "not captured or outside bounded source-scope projection"}
                    candidate = {**result, "rows": [*result["rows"], row]}
                    if len(json.dumps(candidate).encode()) > 128 * 1024:
                        result["omitted_rows"] += 1
                    else:
                        result["rows"].append(row)
            except Exception as exc:
                # Advisory read failure never prevents fresh local research.
                result["errors"].append({"source_store": str(root),
                                         "reason": f"{type(exc).__name__}: {exc}"[:512]})
        return result


def original_research_scope(attempt, *, model, quant, backend, build_recipe, surface) -> dict:
    """Original owner metadata; arm recipes come from the recorded comparison.

    In particular, never read the current post-keep anchor to label an older arm.
    Missing historical facts remain missing; this function is write-side only.
    """
    comparison = attempt.get("comparison") or {}
    inputs = (comparison.get("belief_capture") or {}).get("inputs") or {}
    arms = inputs.get("resolved_arms") or {}
    return {"model": {"path": str(model), "sha256":
                      ((arms.get("anchor") or {}).get("model") or {}).get("sha256")},
            "quant": quant, "backend": backend, "measurement_surface": surface,
            "recipe": {"build_recipe": build_recipe, "original_serving_arms": arms or None},
            "request_digest": comparison.get("request_digest")}


def epoch_for(*, anchor_commit: str, build_recipe: Mapping[str, Any],
              host_state: Mapping[str, Any] | None = None) -> str:
    return experiments.epoch_sha256(anchor_commit=anchor_commit,
                                    build_recipe=build_recipe,
                                    host_state=host_state)


__all__ = ["RatchetRefused", "epoch_for", "keep", "recall", "record"]
