"""region-lock — hold CPU-region locks for the lifetime of a child command.

WHY: the orchestrator's dispatch path serializes inference through per-region
`flock`s (`cpu_region_lock`), but standalone benchmarks (`bench_canonical.sh` /
llama-bench) historically took NO lock at all, and `run_benchmark.py` took its
own `fcntl.flock` in a *different* namespace. A canonical bench and an
orchestrator placement could therefore occupy the same physical cores with
nothing preventing it — the per-run operator-approval clause was the only
serializer between them (a human being used as a mutex).

This wrapper closes that gap without a second lock implementation: it acquires
the SAME locks the dispatch path uses, holds them for exactly as long as the
child process lives, and releases them when the child exits (the kernel
releases the `flock` even on SIGKILL, because release is fd-close).

Usage:
    region-lock run --cpu-list 0-95 -- llama-bench -m model.gguf ...
    region-lock run --regions q0,q1 --role bench-cpu -- ./bench_canonical.sh
    region-lock run --gpu-quiet exclusive --role bench-gpu -- ./gpu_bench.sh
    region-lock run --cpu-list 0-47 --gpu-quiet shared -- ./cpu_measure.sh
    region-lock run --cpu-list 160-183 -- ./gpu_slot.sh   # SMT siblings of 64-87 -> q2,q3
    region-lock status

gpu-quiet (2026-10-04, `src/runtime/gpu_quiet_lock.py`): a separate host-wide
flock beside the region locks. GPU benches take it EXCLUSIVE and NO region
claim (pinned GPU host threads degrade the CPU A/A floor ~9x, but a region
claim also 503s every serving role placed there); CPU measurements take it
SHARED alongside their regions. Orchestrator per-call serving claims never
look at it. Order: gpu-quiet first, then regions (with back-off, see module).

`--cpu-list` folds SMT sibling CPUs onto their physical cores' regions by
default (`--fold-siblings`, REGION-SIBLING-1); `--no-fold-siblings` restores the
legacy mapping that ignores CPUs 96-191. The library default in
`instance_topology` is unchanged ("drop"); only this CLI opts in.

Exit codes:
    0..255  the child's exit status (or 128+N when it died on signal N)
    75      EX_TEMPFAIL — could not acquire the regions within the budget
    64      EX_USAGE    — bad arguments / failed preflight

CRITICAL — the cross-role layer. `cpu_region_lock` takes two layers: a
role-AGNOSTIC `GLOBAL` mutex per region (true cross-role exclusion) and a
per-role lock (attribution). The GLOBAL layer is gated behind
ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT. Without it, this wrapper would take
only `cpu_region.bench.*` locks that no orchestrator path ever contends —
providing false confidence rather than exclusion. So we force the flag on for
ourselves AND preflight that the live dispatch process has it too, failing
closed when it does not (fabric axiom 3).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Must be set BEFORE cpu_region_lock evaluates _cross_role_mutex_enabled(),
# which reads os.environ at call time. Setting it at import is the safest point.
os.environ.setdefault("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")

from src.runtime.cpu_region_lock import (  # noqa: E402
    CpuRegionLockTimeout,
    cpu_region_lock,
    global_region_lock_path,
    read_region_lock_payload,
    region_lock_path,
)
from src.runtime.gpu_quiet_lock import (  # noqa: E402
    GPU_QUIET_MODES,
    gpu_quiet_then_regions,
)
from src.runtime.gpu_quiet_lock import holders as gpu_quiet_holders  # noqa: E402
from src.runtime.instance_topology import ATOMIC_REGIONS, cpu_list_to_regions  # noqa: E402
from src.runtime.lock_queue import queue_dir as _fifo_queue_dir  # noqa: E402
from src.runtime.lock_queue import queue_status as _fifo_queue_status  # noqa: E402

EX_TEMPFAIL = 75
EX_USAGE = 64

_TRUTHY = {"1", "true", "yes", "on"}
# Matches how the stack launches the API; used only for the preflight probe.
_DISPATCH_PROC_HINT = "uvicorn src.api:app"


def _lock_dir() -> Path:
    """Directory holding every region lock file (derived, not a private import)."""
    return region_lock_path("x", "y").parent


def _flock_held(path: Path) -> bool:
    """True if some process currently holds an exclusive flock on `path`.

    Probes with LOCK_EX|LOCK_NB and immediately releases on success, so this
    never disturbs a real holder and never blocks. Treats an unreadable file as
    held (fail-closed, fabric axiom 3).
    """
    import fcntl

    if not path.exists():
        return False
    try:
        with open(path, "a+b") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            return False
    except OSError:
        return True


def _dispatch_pids() -> list[int]:
    """Best-effort: pids of live orchestrator API processes."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", _DISPATCH_PROC_HINT],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(line) for line in out.stdout.split() if line.strip().isdigit()]


def _proc_has_cross_role_flag(pid: int) -> bool | None:
    """True/False if readable, None if the environ could not be read."""
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except (OSError, PermissionError):
        return None
    for entry in raw.split(b"\0"):
        if entry.startswith(b"ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT="):
            return entry.split(b"=", 1)[1].decode(errors="replace").strip().lower() in _TRUTHY
    return False


def _preflight(strict: bool) -> str | None:
    """Return an error string when the cross-role layer would be one-sided.

    A dispatch process without the flag takes only per-role locks, so it would
    never contend our GLOBAL locks — we would appear protected while sharing
    cores. Fail closed on that; a merely-unreadable environ is a warning.
    """
    pids = _dispatch_pids()
    if not pids:
        return None  # nothing dispatching — nothing to be one-sided against
    unflagged = [p for p in pids if _proc_has_cross_role_flag(p) is False]
    if unflagged:
        msg = (
            f"orchestrator dispatch pid(s) {unflagged} do NOT have "
            "ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT enabled — they take per-role "
            "locks only and will not contend this wrapper's GLOBAL locks. Exclusion "
            "would be one-sided (false confidence)."
        )
        return msg if strict else None
    unreadable = [p for p in pids if _proc_has_cross_role_flag(p) is None]
    if unreadable:
        print(
            f"[region-lock] WARNING: could not read environ of pid(s) {unreadable}; "
            "cross-role flag unverified.",
            file=sys.stderr,
        )
    return None


def _resolve_regions(args: argparse.Namespace) -> frozenset[str]:
    if args.cpu_list:
        # REGION-SIBLING-1: with --fold-siblings (the CLI default) SMT siblings
        # claim their physical core's region, so `--cpu-list 160-183` takes the
        # same regions as 64-87. The LIBRARY default stays "drop" for every
        # other caller; only this path opts in. --no-fold-siblings restores the
        # legacy CLI meaning exactly.
        fold = getattr(args, "fold_siblings", True)
        mode = "fold" if fold else "drop"
        try:
            regions = cpu_list_to_regions(args.cpu_list, smt_siblings=mode)
        except ValueError as e:  # incl. instance_topology.CpuTopologyUnavailable
            raise SystemExit(
                f"[region-lock] --cpu-list {args.cpu_list!r}: cannot resolve the physical "
                f"cores it occupies ({e}); refusing rather than under-claiming "
                f"(--no-fold-siblings restores the legacy primary-only mapping)"
            ) from e
        if not regions:
            raise SystemExit(f"[region-lock] --cpu-list {args.cpu_list!r} maps to no regions")
        if fold:
            legacy = cpu_list_to_regions(args.cpu_list, smt_siblings="drop")
            if legacy != regions:
                print(
                    f"[region-lock] --cpu-list {args.cpu_list!r} includes SMT siblings: "
                    f"claiming {sorted(regions)} (their physical cores) instead of the "
                    f"primary-only {sorted(legacy)}; --no-fold-siblings for the legacy mapping",
                    file=sys.stderr,
                )
        return regions
    if not args.regions:
        return frozenset()  # --gpu-quiet alone (validated in _run)
    names = [r.strip() for r in args.regions.split(",") if r.strip()]
    unknown = sorted(set(names) - set(ATOMIC_REGIONS))
    if unknown:
        raise SystemExit(
            f"[region-lock] unknown region(s) {unknown}; valid: {list(ATOMIC_REGIONS)}"
        )
    return frozenset(names)


def _run(args: argparse.Namespace) -> int:
    if not args.command:
        raise SystemExit("[region-lock] no command given after `--`")
    if not (args.regions or args.cpu_list or args.gpu_quiet):
        print(
            "[region-lock] run needs --regions, --cpu-list and/or --gpu-quiet",
            file=sys.stderr,
        )
        return EX_USAGE

    regions = _resolve_regions(args)
    if regions:
        # The cross-role preflight is about the region GLOBAL layer; a gpu-quiet-only
        # hold takes no region lock and has nothing to be one-sided against.
        err = _preflight(strict=not args.no_preflight)
        if err:
            print(f"[region-lock] REFUSING: {err}", file=sys.stderr)
            print(
                "[region-lock] re-run with --no-preflight to override deliberately.",
                file=sys.stderr,
            )
            return EX_USAGE

    tag = args.tag or f"region-lock:{Path(args.command[0]).name}"
    waited_from = time.time()
    what = f"regions {sorted(regions)}"
    if args.gpu_quiet:
        what = f"gpu-quiet {args.gpu_quiet}" + (f" + {what}" if regions else "")

    print(
        f"[region-lock] acquiring {what} as role={args.role!r} (timeout_s={args.timeout_s}) …",
        file=sys.stderr,
    )
    try:
        if args.gpu_quiet is None:
            # Byte-for-byte the pre-gpu-quiet path.
            with cpu_region_lock(
                args.role,
                regions,
                timeout_s=args.timeout_s,
                request_tag=tag,
            ) as held:
                waited = time.time() - waited_from
                print(
                    f"[region-lock] held {sorted(held)} after {waited:.1f}s wait; "
                    f"running: {' '.join(args.command)}",
                    file=sys.stderr,
                )
                return _spawn(args.command)
        with gpu_quiet_then_regions(
            args.role,
            regions,
            gpu_quiet=args.gpu_quiet,
            timeout_s=args.timeout_s,
            request_tag=tag,
        ) as grant:
            waited = time.time() - waited_from
            print(
                f"[region-lock] held gpu-quiet {args.gpu_quiet} + regions "
                f"{sorted(grant['regions'])} after {waited:.1f}s wait; "
                f"running: {' '.join(args.command)}",
                file=sys.stderr,
            )
            return _spawn(args.command)
    except CpuRegionLockTimeout as e:
        print(f"[region-lock] TIMEOUT acquiring {what}: {e}", file=sys.stderr)
        return EX_TEMPFAIL


def _spawn(command: list[str]) -> int:
    """Run the child, forwarding termination signals so drain works.

    Signals are forwarded rather than swallowed: a quiesce/drain SIGTERM must
    reach the benchmark so it can stop at its own boundary (fabric axiom 4 —
    no forcible mid-flight preemption).
    """
    proc = subprocess.Popen(command)
    forwarded: list[int] = []

    def _forward(signum, _frame):
        forwarded.append(signum)
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            pass

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            previous[sig] = signal.signal(sig, _forward)
        except (OSError, ValueError):
            pass
    try:
        rc = proc.wait()
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass
    if rc < 0:  # died on signal N
        rc = 128 + (-rc)
    if forwarded:
        print(f"[region-lock] forwarded signal(s) {forwarded} to child", file=sys.stderr)
    return rc


def _status(args: argparse.Namespace) -> int:
    """Report, per region, whether the cross-role GLOBAL mutex is held and by whom.

    `active_region_holders()` deliberately excludes the GLOBAL pseudo-role and
    only knows roles present in the instance topology, so a bench holding a
    region would not appear there. We probe the GLOBAL lock directly and
    attribute via whatever per-role payloads are present.
    """
    rows = []
    for region in ATOMIC_REGIONS:
        held = _flock_held(global_region_lock_path(region))
        holders = []
        for lock_file in sorted(_lock_dir().glob(f"cpu_region.*.{region}.lock")):
            if lock_file.name.startswith("cpu_region.GLOBAL."):
                continue
            # Realized-first truth (fabric axiom 2): a payload is a stored
            # record, the flock is the fact. A holder killed with SIGKILL never
            # runs its cleanup, so a stale payload outlives the lock — trust it
            # only for a lock that is *currently* held.
            if not _flock_held(lock_file):
                continue
            payload = read_region_lock_payload(lock_file)
            if payload:
                holders.append(payload)
        rows.append({"region": region, "global_held": held, "holders": holders})

    quiet = gpu_quiet_holders()
    queue_rows = _fifo_queue_status(_fifo_queue_dir(_lock_dir()))
    if args.json:
        # The bare list stays the default shape: consumers validate it as exactly
        # q0..q3 (e.g. the research reanchor runner). gpu-quiet is opt-in.
        if args.gpu_quiet:
            print(json.dumps(
                {"regions": rows, "gpu_quiet": quiet, "queue": queue_rows},
                indent=2, default=str,
            ))
        else:
            print(json.dumps(rows, indent=2, default=str))
        return 0
    for row in rows:
        state = "HELD" if row["global_held"] else "free"
        who = ", ".join(
            f"{h.get('role', '?')}[{h.get('instance_idx', '-')}] {h.get('request_tag', '')}".strip()
            for h in row["holders"]
        )
        print(f"{row['region']:>4}  {state:<5}  {who}")
    if quiet["held"]:
        state = "EXCL" if quiet["mode"] == "exclusive" else "SHRD" if quiet["mode"] else "HELD"
    else:
        state = "free"
    who = ", ".join(
        f"{h.get('role', '?')}[pid {h.get('pid', '-')}] {h.get('request_tag') or ''}".strip()
        for h in quiet["holders"]
    )
    print(f"gpu-quiet  {state:<5}  {who}")
    if queue_rows:
        print("queue (FIFO, EPYC_LOCK_FIFO=1 clients only):")
        for row in queue_rows:
            print(
                f"  #{row['position']:<3} age={row['age_s']:>6.1f}s "
                f"pid={row['pid']:<8} mode={row['mode']:<9} "
                f"regions={','.join(row['regions']) or '-':<12} tag={row['tag']}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="region-lock",
        description="Hold orchestrator CPU-region locks for the lifetime of a child command.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="acquire regions, run a command, release on exit")
    target = run.add_mutually_exclusive_group()
    target.add_argument("--regions", help="comma-separated region ids, e.g. q0,q1,q2,q3")
    target.add_argument(
        "--cpu-list",
        help="taskset-style cpu list, e.g. 0-95 (regions derived; SMT siblings such as "
        "160-183 claim the regions of their physical cores, here 64-87; see --fold-siblings)",
    )
    run.add_argument(
        "--fold-siblings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with --cpu-list, map SMT sibling CPUs (96-191 on this host) onto their "
        "physical cores' regions, read from the kernel's thread-sibling topology "
        "(default: on). --no-fold-siblings restores the legacy mapping, which ignores "
        "sibling CPUs entirely. No effect with --regions.",
    )
    run.add_argument(
        "--gpu-quiet",
        choices=GPU_QUIET_MODES,
        help="also hold the host-wide gpu-quiet lock: 'exclusive' for a GPU bench (valid "
        "with no regions — GPU benches should take NO region claim), 'shared' for a CPU "
        "measurement (waits for any exclusive holder). Taken BEFORE the regions.",
    )
    run.add_argument(
        "--role",
        default="bench",
        help="attribution label for the per-role lock (default: bench). Not a real stack role.",
    )
    run.add_argument(
        "--timeout-s",
        type=float,
        default=0.0,
        help="per-region acquire timeout; 0 = block indefinitely (default), with periodic "
        "waiting logs from the lock module. With --gpu-quiet it bounds the whole "
        "gpu-quiet+regions acquisition. Exit 75 on expiry.",
    )
    run.add_argument("--tag", help="opaque tag recorded in the lock payload for diagnostics")
    run.add_argument(
        "--no-preflight",
        action="store_true",
        help="skip the cross-role-flag preflight (deliberate override; exclusion may be one-sided)",
    )
    run.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND [ARGS...]")
    run.set_defaults(func=_run)

    st = sub.add_parser("status", help="show which regions are currently held")
    st.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    st.add_argument(
        "--gpu-quiet",
        action="store_true",
        help="with --json: emit {regions: [...], gpu_quiet: {...}} instead of the bare "
        "region list (the table always shows the gpu-quiet row)",
    )
    st.set_defaults(func=_status)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "command", None) and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
