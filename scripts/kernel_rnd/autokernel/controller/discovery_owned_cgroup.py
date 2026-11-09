#!/usr/bin/env python3
"""Run one command inside a job-owned cgroup (AK-DISCOVERY-OWNED-CGROUP).

Design: docs/design/ak-discovery-owned-cgroup-20261109.md in epyc-root.

The wrapper creates <root>/akd-<run_id> exclusively, the child enters it before
exec, and cleanup touches only that path after re-checking the receipt identity.
Removal is rmdir only, deepest first; the kernel refuses to rmdir a populated
cgroup, which is the safety property. Stdlib only.

Exit codes: 0 PASS, 1 test FAIL, 2 refused before any cgroup write, 3 identity
mismatch (nothing touched), 4 still populated or not removable (left in place),
75 deadline, STOP or SIGTERM (always a FAIL).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

CGROUP_ROOT = "/sys/fs/cgroup"
NAME_RE = re.compile(r"^akd-[A-Za-z0-9._-]{8,96}$")
LEAF_RE = re.compile(r"^autokernel-")
SWEEP_RE = re.compile(r"^(autokernel-|akd-)")
ENTER_SCRIPT = 'echo $$ > "$1/cgroup.procs" && cat /proc/self/cgroup > "$2" && shift 2 && exec "$@"'
RC_PASS, RC_FAIL, RC_REFUSED, RC_IDENTITY, RC_POPULATED, RC_DEADLINE = 0, 1, 2, 3, 4, 75
PASS_LABEL = "native toy-compiler fixture PASS"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json_atomic(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as handle:
        json.dump(data, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def owned_name(row_id: str, token: str | None = None) -> str:
    name = f"akd-{row_id}-{token or secrets.token_hex(8)}"
    if not NAME_RE.match(name):
        raise ValueError(f"owned cgroup name refused: {name!r}")
    return name


def list_swept(root: Path) -> list[str]:
    return sorted(e.name for e in os.scandir(root)
                  if SWEEP_RE.match(e.name) and e.is_dir(follow_symlinks=False))


def create_owned(root: Path, name: str, receipt_path: Path, run_id: str) -> dict:
    """Create the owned cgroup exclusively (FileExistsError if present) and record its identity."""
    if not NAME_RE.match(name):
        raise ValueError(f"owned cgroup name refused: {name!r}")
    path = root / name
    os.mkdir(path, 0o700)
    info = os.lstat(path)
    receipt = {"path": str(path), "dev": info.st_dev, "ino": info.st_ino, "uid": info.st_uid,
               "nlink": info.st_nlink, "mode": stat.S_IMODE(info.st_mode), "created_utc": utc_now(),
               "run_id": run_id, "wrapper_pid": os.getpid()}
    write_json_atomic(receipt_path, receipt)
    return receipt


def validate_owned(root: Path, receipt: dict) -> str | None:
    """Return None when the path is still the one this run created, else the mismatch."""
    path = Path(receipt["path"])
    if not NAME_RE.match(path.name):
        return f"name does not match: {path.name!r}"
    if path.parent != root:
        return f"not a direct child of {root}: {path}"
    try:
        info = os.lstat(path)
    except OSError as exc:
        return f"lstat failed: {exc}"
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return "not a real directory"
    for key, actual in (("dev", info.st_dev), ("ino", info.st_ino), ("uid", info.st_uid)):
        if receipt.get(key) != actual:
            return f"{key} changed: receipt {receipt.get(key)} != {actual}"
    return None


def descendants(path: Path) -> list[Path]:
    """Every descendant directory, deepest first, never following symlinks."""
    found: list[Path] = []
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            child = Path(entry.path)
            found.extend(descendants(child))
            found.append(child)
    return found


def _populated(owned: Path) -> bool:
    """Fail closed: an unreadable or absent control file counts as populated."""
    for group in [owned, *descendants(owned)]:
        try:
            if (group / "cgroup.procs").read_text().strip():
                return True
        except OSError:
            return True
    try:
        fields = dict(line.split() for line in (owned / "cgroup.events").read_text().splitlines())
    except (OSError, ValueError):
        return True
    return fields.get("populated") != "0"


def cleanup_owned(root: Path, receipt_path: Path, *, kill_wait_s: float = 5.0, rmdir=os.rmdir) -> tuple[int, dict]:
    """Design section 3: re-validate, cgroup.kill on the owned path only, rmdir deepest first."""
    outcome: dict = {"removed": False, "leftover_leaves": [], "killed": False}
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError) as exc:
        outcome["identity_mismatch"] = f"receipt unreadable: {exc}"
        return RC_IDENTITY, outcome
    mismatch = validate_owned(root, receipt)
    if mismatch:
        outcome["identity_mismatch"] = mismatch
        return RC_IDENTITY, outcome
    owned = Path(receipt["path"])
    if _populated(owned):
        fd = os.open(owned / "cgroup.kill", os.O_WRONLY | os.O_NOFOLLOW)
        try:
            os.write(fd, b"1")
        finally:
            os.close(fd)
        outcome["killed"] = True
        limit = time.monotonic() + kill_wait_s
        while _populated(owned):
            if time.monotonic() >= limit:
                outcome["still_populated"] = True
                return RC_POPULATED, outcome
            time.sleep(0.05)
    unexpected = [str(p) for p in descendants(owned)
                  if not LEAF_RE.match(p.relative_to(owned).parts[0])]
    if unexpected:
        outcome["unexpected_children"] = unexpected
        return RC_POPULATED, outcome
    mismatch = validate_owned(root, receipt)
    if mismatch:
        outcome["identity_mismatch"] = mismatch
        return RC_IDENTITY, outcome
    leftovers = descendants(owned)
    outcome["leftover_leaves"] = [str(p) for p in leftovers]
    try:
        for leaf in leftovers:
            rmdir(leaf)
        rmdir(owned)
    except OSError as exc:
        outcome["rmdir_error"] = str(exc)
        return RC_POPULATED, outcome
    outcome["removed"] = True
    return RC_PASS, outcome


def run_child(owned: Path, marker: Path, argv: list[str], *, cwd: str, log: Path, deadline_s: float,
              stop_file: str | None, env: dict | None = None) -> tuple[int | None, str | None]:
    """Run argv in its own session inside owned; return (rc, stop_reason)."""
    terminated: list[str] = []
    previous = signal.signal(signal.SIGTERM, lambda *_: terminated.append("sigterm"))
    try:
        with open(log, "wb") as output:
            child = subprocess.Popen(
                ["/bin/sh", "-c", ENTER_SCRIPT, "sh", str(owned), str(marker), *argv],
                stdout=output, stderr=subprocess.STDOUT, cwd=cwd, env=env, start_new_session=True)
        limit = time.monotonic() + deadline_s
        reason = None
        while child.poll() is None:
            if terminated:
                reason = "sigterm"
            elif stop_file and os.path.exists(stop_file):
                reason = "stop-file"
            elif time.monotonic() >= limit:
                reason = "deadline"
            if reason:
                for sig, grace in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
                    try:
                        os.killpg(child.pid, sig)
                    except ProcessLookupError:
                        break
                    try:
                        child.wait(grace)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                break
            time.sleep(0.2)
        return child.wait(), reason
    finally:
        signal.signal(signal.SIGTERM, previous)


def unittest_passed(log: Path) -> bool:
    """Exactly one test ran and passed: a bare 'OK' line, so 'OK (skipped=1)' cannot pass."""
    lines = log.read_text(errors="replace").splitlines()
    return "OK" in lines and any(re.match(r"^Ran 1 test in ", line) for line in lines)


def run(args: argparse.Namespace, root: Path = Path(CGROUP_ROOT), *, rmdir=os.rmdir, kill_wait_s: float = 5.0) -> int:
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.unlink(missing_ok=True)
    receipt_path = scratch / "cgroup-receipt.json"
    marker = scratch / "child-cgroup.txt"
    log = scratch / "child.log"
    for stale in (receipt_path, marker, log):
        stale.unlink(missing_ok=True)
    started = utc_now()
    result: dict = {"schema": "epyc.ak.discovery_owned_cgroup.v1", "row_id": args.row_id, "started_utc": started,
                    "started_epoch": int(time.time()), "pass": False, "label": None}
    try:
        name = owned_name(args.row_id)
    except ValueError as exc:
        result["refused"] = str(exc)
        write_json_atomic(result_path, result)
        return RC_REFUSED
    if args.dry_run:
        print(json.dumps({"plan": str(root / name), "argv": args.argv, "cwd": args.cwd, "result": str(result_path)}))
        return RC_PASS
    before = list_swept(root)
    try:
        receipt = create_owned(root, name, receipt_path, args.row_id)
    except OSError as exc:
        result["refused"] = f"create failed, nothing to clean: {exc}"
        write_json_atomic(result_path, result)
        return RC_REFUSED
    owned = Path(receipt["path"])
    child_rc, stop_reason, child_error = None, None, None
    try:
        child_rc, stop_reason = run_child(owned, marker, args.argv, cwd=args.cwd, log=log,
                                          deadline_s=args.deadline_s, stop_file=args.stop)
    except OSError as exc:
        child_error = str(exc)
    cleanup_rc, cleanup = cleanup_owned(root, receipt_path, kill_wait_s=kill_wait_s, rmdir=rmdir)
    receipt = json.loads(receipt_path.read_text())
    receipt["cleanup"] = cleanup
    write_json_atomic(receipt_path, receipt)
    new_entries = sorted(set(list_swept(root)) - set(before))
    membership = marker.read_text().strip() if marker.exists() else ""
    checks = {"child_rc_zero": child_rc == 0,
              "unittest_one_ok": log.exists() and unittest_passed(log),
              "child_in_owned_cgroup": membership == f"0::/{name}",
              "owned_removed": cleanup["removed"] and not os.path.lexists(owned),
              "no_new_root_entries": not new_entries}
    passed = stop_reason is None and child_error is None and all(checks.values())
    result.update(child_rc=child_rc, stop_reason=stop_reason, child_error=child_error, owned_path=str(owned),
                  child_cgroup=membership, checks=checks, cleanup=cleanup, new_root_entries=new_entries,
                  ended_utc=utc_now(), **{"pass": passed, "label": PASS_LABEL if passed else None})
    write_json_atomic(result_path, result)
    if cleanup_rc:
        return cleanup_rc
    if stop_reason:
        return RC_DEADLINE
    return RC_PASS if passed else RC_FAIL


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--deadline-s", type=float, default=600.0)
    parser.add_argument("--stop")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; no cgroup or child")
    parser.add_argument("argv", nargs="+", help="command to run, after --")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
