"""Serial finite batches of the existing loop, not another measurement owner."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time

from . import campaign_cli, champion, legacy_targets, status

CONTINUATION_SCHEMA = "epyc.autokernel.loop_continuation.v1"
SERIAL_SCHEMA = "epyc.autokernel.serial_run.v1"
_DOCUMENT_FLAGS = ("--resolved-campaign", "--cpu-serving-launch", "--frozen-prompts", "--serving-recipe")
_CHANGING_FLAGS = frozenset({"--out", "--iterations", "--resume-run", "--anchor-build",
                            "--cor-build", "--cpu-calibrate-serving", "--dry-run"})
_CONTINUATION_FIELDS = {"schema", "terminal", "input_argv", "input_argv_sha256", "binding",
                        "worktree", "branch", "model", "selected_target", "current_anchor", "cor_anchor",
                        "iterations_requested", "iterations_completed", "outcome_counts", "result_file"}


class SerialRefused(ValueError):
    pass


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _read(path: Path, *, limit: int = 2 * 1024 * 1024) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise SerialRefused(f"not a bounded regular file: {path}")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(raw) > limit or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise SerialRefused(f"file changed while read: {path}")
    return raw


def _json(path: Path, *, limit: int = 2 * 1024 * 1024):
    raw = _read(path, limit=limit)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def option(argv, name, default=None):
    values = []
    for index, arg in enumerate(argv):
        if arg == name:
            if index + 1 == len(argv) or argv[index + 1].startswith("--"):
                raise SerialRefused(f"{name} has no value")
            values.append(argv[index + 1])
        elif arg.startswith(name + "="):
            values.append(arg[len(name) + 1:])
    # Match the existing argparse last-value behavior, including old launchers
    # which append a finite iteration override to common arguments.
    return values[-1] if values else default


def _without(argv, names):
    out = []
    iterator = iter(argv)
    for arg in iterator:
        flag, equals, _value = arg.partition("=")
        if flag in names:
            if not equals and flag != "--dry-run":
                next(iterator, None)
        else:
            out.append(arg)
    return out


def input_binding(argv) -> dict:
    """Original argument/workload bytes; never hash a model or derive a grant."""
    documents = {}
    for flag in _DOCUMENT_FLAGS:
        value = option(argv, flag)
        if value is not None:
            documents[flag] = hashlib.sha256(_read(Path(value))).hexdigest()
    return {"argv": _without(argv, _CHANGING_FLAGS), "documents": documents}


def continuation(*, argv, binding, terminal, worktree, branch, model, selected_target,
                 anchor_build, anchor_commit, iterations_requested, outcomes,
                 cor_build=None, cor_commit=None) -> dict:
    counts = {}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    return {"schema": CONTINUATION_SCHEMA, "terminal": terminal,
            "input_argv": list(argv), "input_argv_sha256": _digest(list(argv)), "binding": binding,
            "worktree": str(Path(worktree).resolve()), "branch": branch,
            "model": str(Path(model).resolve()), "selected_target": selected_target,
            "iterations_requested": iterations_requested, "iterations_completed": len(outcomes),
            "outcome_counts": counts, "result_file": "loop-run.json",
            "current_anchor": {"path": str(Path(anchor_build).resolve()), "commit": anchor_commit},
            "cor_anchor": ({"path": str(Path(cor_build).resolve()), "commit": cor_commit}
                           if cor_build is not None else None)}


def load_completed(path: Path, *, expected_argv=None, expected_binding=None):
    # The full result can legitimately contain hundreds of MiB of original
    # lifecycle observations. This original-owner routing receipt does not parse
    # or grade them, and never pretends that its file stat is a content hash.
    row, sha = _json(path)
    if not isinstance(row, dict) or set(row) != _CONTINUATION_FIELDS \
            or row["schema"] != CONTINUATION_SCHEMA or row["terminal"] not in {"complete", "stopped"}:
        raise SerialRefused("missing or malformed terminal continuation")
    argv = row["input_argv"]
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv) \
            or row["input_argv_sha256"] != _digest(argv):
        raise SerialRefused("continuation input arguments differ")
    if expected_argv is not None and argv != list(expected_argv):
        raise SerialRefused("result is for different actual child arguments")
    if not isinstance(row["binding"], dict) or set(row["binding"]) != {"argv", "documents"}:
        raise SerialRefused("invalid original input binding")
    if expected_binding is not None and row["binding"] != expected_binding:
        raise SerialRefused("result workload/input binding differs")
    for key in ("worktree", "model"):
        if not isinstance(row[key], str) or not Path(row[key]).is_absolute():
            raise SerialRefused(f"invalid continuation {key}")
    if not isinstance(row["branch"], str) or not row["branch"]:
        raise SerialRefused("invalid continuation branch")
    if Path(row["worktree"]).resolve() != Path(option(argv, "--worktree", "")).resolve():
        raise SerialRefused("recorded worktree differs from actual input arguments")
    cpu = option(argv, "--cpu-serving-launch") is not None
    branch = option(argv, "--experimental-branch") if cpu else option(
        argv, "--champion-branch", champion.CANONICAL_BRANCH)
    if branch != row["branch"] or cpu != (row["cor_anchor"] is None):
        raise SerialRefused("recorded branch/backend differs from actual input arguments")
    if option(argv, "--resolved-campaign") is not None:
        resolved = campaign_cli.load_previous(Path(option(argv, "--resolved-campaign")))
        selected = legacy_targets.select_target(resolved, option(argv, "--target-id"), cpu_serving=cpu)
        expected_target = {"campaign_id": resolved.campaign_id, "request_id": resolved.request_id,
                           "manifest_digest": resolved.manifest_digest,
                           "selected_id": option(argv, "--target-id"),
                           "scope": "cpu_serving_selected_workload" if cpu else "legacy_gpu_screen",
                           "original_target": selected.to_dict()}
        if row["selected_target"] != expected_target \
                or Path(row["model"]).resolve() != Path(selected.execution.model.path).resolve():
            raise SerialRefused("recorded selected target/model differs from original inputs")
    elif row["selected_target"] is not None \
            or Path(row["model"]).resolve() != Path(option(argv, "--model", "")).resolve():
        raise SerialRefused("recorded model differs from actual input arguments")
    for key in ("current_anchor", "cor_anchor"):
        anchor = row[key]
        if key == "cor_anchor" and anchor is None:
            continue
        if not isinstance(anchor, dict) or set(anchor) != {"path", "commit"} \
                or not isinstance(anchor["path"], str) or not Path(anchor["path"]).is_absolute() \
                or not isinstance(anchor["commit"], str) or len(anchor["commit"]) != 40 \
                or any(c not in "0123456789abcdef" for c in anchor["commit"]):
            raise SerialRefused(f"invalid {key}")
    if row["result_file"] != "loop-run.json" or Path(option(argv, "--out", "")).resolve() != path.parent.resolve():
        raise SerialRefused("continuation is not at its original output path")
    result = path.parent / row["result_file"]
    info = result.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        raise SerialRefused("original full result is missing or not a regular nonempty file")
    requested, count = row["iterations_requested"], row["iterations_completed"]
    if type(requested) is not int or type(count) is not int or count < 0 \
            or requested != int(option(argv, "--iterations", "10")):
        raise SerialRefused("invalid original iteration counts")
    counts = row["outcome_counts"]
    if not isinstance(counts, dict) or any(not isinstance(k, str) or type(v) is not int or v < 1
                                           for k, v in counts.items()) or sum(counts.values()) != count:
        raise SerialRefused("outcome counts differ from completed iterations")
    if requested <= 0 or count > requested or (row["terminal"] == "complete" and count != requested):
        raise SerialRefused("result does not cover its finite batch")
    return row, sha


def full_commit(worktree: Path, commit: str) -> str:
    done = champion._git(worktree, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if done.returncode != 0:
        raise SerialRefused("original commit cannot be resolved unambiguously")
    return done.stdout.strip()


def verify_exact_anchor(path: Path, worktree: Path, commit: str, *, experimental=False):
    """Reuse the original verifier, then require the named exact arm, not an ancestor."""
    commit = full_commit(worktree, commit)
    champion.verify_anchor(path, worktree, commit, experimental_identity=experimental)
    prov = path / "provenance.json"
    if prov.exists():
        body, _sha = _json(prov)
        named = body.get("champion_commit") if isinstance(body, dict) else None
        if not isinstance(named, str) or not named:
            raise SerialRefused("anchor provenance has no original commit")
        if full_commit(worktree, named) != commit:
            raise SerialRefused("anchor provenance is not the exact recorded arm")
    # Without provenance the existing experimental-identity verifier already
    # required exact head/source and original binary inventory; no new receipt.


def _target_args(path: Path) -> list[str]:
    argv, _sha = _json(path, limit=64 * 1024)
    if not isinstance(argv, list) or not argv or len(argv) > 512 \
            or not all(isinstance(item, str) and "\0" not in item for item in argv):
        raise SerialRefused("target args must be a bounded JSON string array")
    reserved = {"--out", "--iterations", "--resume-run", "--dry-run", "--calibrate-surface",
                "--allow-unverified-anchor", "--help", "-h"}
    if any(item.partition("=")[0] in reserved for item in argv):
        raise SerialRefused("target args contain a serial-owned or non-execution option")
    for flag in ("--worktree", "--anchor-build", "--store", "--worker-root", "--worker-build-root",
                 "--resolved-campaign", "--target-id"):
        value = option(argv, flag)
        if not value or (flag != "--target-id" and not Path(value).is_absolute()):
            raise SerialRefused(f"serial target requires explicit absolute {flag}" if flag != "--target-id"
                                else "serial target requires --target-id")
    resolved = campaign_cli.load_previous(Path(option(argv, "--resolved-campaign")))
    legacy_targets.select_target(resolved, option(argv, "--target-id"),
                                 cpu_serving=option(argv, "--cpu-serving-launch") is not None,
                                 model=Path(option(argv, "--model")) if option(argv, "--model") else None)
    return argv


def _child_command(argv):
    return [sys.executable, "-m", "scripts.kernel_rnd.autokernel.loop.run", *argv]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-args", type=Path, action="append", required=True)
    parser.add_argument("--batch-iterations", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=1, help="0 rotates continuously until STOP")
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch_iterations <= 0 or args.rounds < 0:
        parser.error("batch iterations must be positive and rounds nonnegative")
    try:
        targets = [_target_args(path) for path in args.target_args]
        # No two configured targets may overwrite an active target's source,
        # build lanes or history. The owner never creates/repoints those roots.
        for flag in ("--worktree", "--store", "--worker-root", "--worker-build-root"):
            paths = [Path(option(row, flag)).resolve() for row in targets]
            if any(a == b or a in b.parents or b in a.parents
                   for i, a in enumerate(paths) for b in paths[i + 1:]):
                raise SerialRefused(f"target {flag} roots overlap")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    root = args.state_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "serial.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SerialRefused("serial session already has an owner") from exc
        return _drive(root, targets, args.batch_iterations, args.rounds)


def _drive(root, targets, batch_iterations, rounds):
    config = _digest({"targets": targets, "batch_iterations": batch_iterations, "rounds": rounds})
    state_path = root / "serial-state.json"
    if state_path.exists():
        state, _sha = _json(state_path)
        if state.get("schema") != SERIAL_SCHEMA or state.get("config_digest") != config:
            raise SerialRefused("serial state belongs to different inputs")
        if state.get("active") is not None:
            raise SerialRefused("previous owned batch has no reconciled terminal result; no relaunch")
    else:
        state = {"schema": SERIAL_SCHEMA, "config_digest": config, "next_batch": 0,
                 "active": None, "last_results": {}, "failed_targets": {}}
    stop = threading.Event()
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def request_stop(_sig, _frame):
        stop.set()
        (root / "STOP").touch(exist_ok=True)

    def stopped():
        return stop.is_set() or (root / "STOP").exists()

    def save():
        status.write_json(root, state_path.name, state)

    def publish(phase, active=None):
        status.write(root, state=phase, epoch=config, campaign_id="legacy-serial",
                     anchor_commit="", surface="serial_targets", pairs=0, noise_floor_pct=None,
                     target=active, step="serial routing only; detailed original status stays in target store",
                     stale_after_s=180)

    for sig in handlers:
        signal.signal(sig, request_stop)
    try:
        while not stopped() and (rounds == 0 or state["next_batch"] < rounds * len(targets)):
            if len(state["failed_targets"]) == len(targets):
                publish("failed")
                return 1  # Continuous mode must not spin over a failed roster.
            number = state["next_batch"]
            index = number % len(targets)
            key = str(index)
            if key in state["failed_targets"]:
                state["next_batch"] += 1
                save()
                continue
            original = targets[index]
            directory = root / "batches" / f"batch-{number:06d}"
            directory.mkdir(parents=True, exist_ok=False)
            child_argv = list(original)
            prior = state["last_results"].get(key)
            if prior is not None:
                _body, sha = load_completed(Path(prior["path"]), expected_binding=input_binding(original))
                if sha != prior["sha256"]:
                    raise SerialRefused("retained child result changed")
                child_argv = _without(child_argv, {"--cpu-calibrate-serving"})
                child_argv += ["--resume-run", prior["path"]]
            child_argv += ["--iterations", str(batch_iterations), "--out", str(directory)]
            expected_binding = input_binding(child_argv)
            active = {"target_index": index, "selected_id": option(original, "--target-id"),
                      "store": option(original, "--store"), "batch_dir": str(directory),
                      "input_argv_sha256": _digest(child_argv), "pid": None}
            state["active"] = active
            save()
            publish("starting", active)
            process = None
            try:
                with (directory / "stdout.log").open("xb") as stdout, \
                        (directory / "stderr.log").open("xb") as stderr:
                    process = subprocess.Popen(_child_command(child_argv), stdout=stdout, stderr=stderr,
                                               cwd=Path(__file__).resolve().parents[4])
                    active["pid"] = process.pid
                    save()
                    publish("running", active)
                    sent = False
                    heartbeat_at = time.monotonic() + 30
                    while process.poll() is None:
                        if stopped() and not sent:
                            process.send_signal(signal.SIGTERM)  # This captured child only; it drains its tail.
                            sent = True
                        try:
                            process.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            pass
                        if time.monotonic() >= heartbeat_at:
                            publish("running", active)
                            heartbeat_at = time.monotonic() + 30
                    if process.returncode != 0:
                        raise SerialRefused(f"child exited {process.returncode}; see retained logs")
                result_path = directory / "loop-continuation.json"
                body, sha = load_completed(result_path, expected_argv=child_argv,
                                           expected_binding=expected_binding)
                identity = body["selected_target"]
                if not isinstance(identity, dict) or identity.get("selected_id") != active["selected_id"]:
                    raise SerialRefused("child terminal belongs to another selected target")
                state["last_results"][key] = {"path": str(result_path), "sha256": sha}
                if body["terminal"] == "stopped":
                    request_stop(None, None)
            except (OSError, ValueError) as exc:
                state["failed_targets"][key] = f"{type(exc).__name__}: {exc}"
            finally:
                # Enrolled immediately after Popen: even publication/reader failure
                # cannot abandon a child still holding the legacy loop's claims.
                if process is not None and process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    process.wait()  # The existing owner drains; never kill by name/group.
            state["active"] = None
            state["next_batch"] += 1
            save()
        publish("complete", {"stop_requested": stopped(), "failed_targets": state["failed_targets"]})
        return 1 if state["failed_targets"] else 0
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
