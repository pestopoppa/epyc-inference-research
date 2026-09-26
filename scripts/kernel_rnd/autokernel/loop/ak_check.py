#!/usr/bin/env python3
"""`ak-check`: the AutoKernel author's scratch sanity-check sandbox (operator 2026-09-26).

WHY. The local 27B author produced five AVX-512 patches for
`mul_mat_qX_K_q8_2_X4_T` that the critic had to reject: intrinsics GCC does not have,
wrong arity (compile errors), block-layout offsets, a missing permute, an epilogue that
summed half the lanes (numeric errors). The lane guard (OAB-11) denies every build
command so an actor cannot contaminate a CPU measurement, which also left the author
unable to find out that its patch did not compile. `ak-check` is the one command the
author may run: it compiles and op-tests the lane's patch OUTSIDE the lane and the
anchor, bounded in cores, time and priority, and refuses while the campaign measures.

MODES
* default -- COMPILE CHECK. Every changed .c/.cpp in the lane (`git diff --name-only`
  against the lane base, untracked files included; a changed header selects the TUs
  whose anchor depfile lists it) is compiled with the anchor build's exact command from
  its `compile_commands.json`, rewritten to the lane's source and include dirs, into the
  scratch dir (`-c`, not `-fsyntax-only`: GCC diagnoses bad intrinsic immediates and
  most template-instantiation errors only while generating code). `nice -n 19`, 4
  cores, 120 s per TU. Diagnostics are printed with lane-relative paths, truncated.
* `--op-test` -- the compile above (objects reused when the source did not change),
  a relink of each library the recompiled TUs belong to (`libggml-cpu.so`, and
  `test-backend-ops` itself if its source changed) from the anchor's own link line into
  the scratch dir, then `test-backend-ops test -b CPU -o MUL_MAT[,MUL_MAT_ID] -p
  type_a=(<touched quants>)` with `GGML_IQK=1` against those libraries. This fork's
  CPU test mode compares the optimized CPU backend with the SAME backend in reference
  mode (`ggml_backend_cpu_set_use_ref`, which bypasses iqk), so it is a real CPU-vs-
  reference check. 8 cores, 300 s. The anchor build is only ever READ.
* SAFETY. Refuses while a tail measurement or calibration of this campaign is active.
  The check the loop cannot race is a lock, not an observation: the loop's serialized
  tail takes `tail_fence()` (an exclusive flock pair in the campaign's worker root)
  for every build/oracle/measure/commit session, and ak-check holds a SHARED slot for
  its whole run, so a tail waits for running checks and a check started during a
  tail refuses -- for every lane of the pool, since all lanes share the worker root.
  As a second line (and the only one for a loop running older code), it also refuses
  when the process tree of the loop that launched it holds a measuring binary.

SCRATCH. ak-check never creates a build dir of its own: the loop allocates one per
lane per iteration and passes it as `AK_CHECK_SCRATCH` (or `--scratch`); without it
ak-check refuses. The loop's scratch registry owns its lifetime and disk budget; when
it cannot guarantee free space it sets `AK_CHECK_OP_TEST=off:<reason>` and `--op-test`
degrades to the compile check, saying so.

Exit status: 0 pass (or nothing to check), 1 the patch failed, 2 refused or the sandbox
could not run (NOT evidence about the patch), 64 usage.

Standard library only: the author's shim runs this file by path.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Iterable, Iterator, Mapping, Sequence

SCHEMA = "epyc.autokernel.ak_check_call.v1"
COMMAND = "ak-check"
#: The author's permission allows exactly these command texts (see
#: `actor_opencode_config.AK_CHECK_ALLOW`).
ALLOWED_COMMANDS = (COMMAND, f"{COMMAND} --op-test")

ENV_SCRATCH = "AK_CHECK_SCRATCH"
ENV_OP_TEST = "AK_CHECK_OP_TEST"
ENV_CALL_ID = "AK_CHECK_CALL_ID"
ENV_LOG = "AK_CHECK_LOG"
ENV_CPUS = "AK_CHECK_CPUS"

#: The fence lives in the campaign's worker root (every lane's parent), so every lane of
#: a pool shares it.
FENCE_DIR_NAME = ".ak-check-fence"
GATE_LOCK, SLOT_LOCK = "tail.gate.lock", "tail.lock"
SHIM_DIR_NAME = "ak-check-bin"
LANE_LOCK = ".ak-check.lock"
#: The ownership marker the loop's scratch registry (`scratch.py`, `MARKER`) writes into
#: every directory it allocates: ak-check builds only inside a dir that carries it.
SCRATCH_MARKER = ".ak-scratch-owner"
SCRATCH_KIND = "ak-check-build"
#: What one lane's op test may need on disk (measured: 3.4 MB for one iqk TU and the
#: relinked libggml-cpu.so; headroom for a header edit that rebuilds many TUs).
OP_TEST_SCRATCH_BYTES = 512 * 10 ** 6

COMPILE_CPUS, OP_TEST_CPUS = 4, 8
COMPILE_TIMEOUT_S, OP_TEST_TIMEOUT_S = 120, 300
LINK_TIMEOUT_S = 120
#: How long a tail waits for running checks before it gives up (a check's own ceiling
#: is OP_TEST_TIMEOUT_S plus process start; this is well past it).
TAIL_WAIT_S = 900
LANE_WAIT_S = 60
MAX_HEADER_TUS_COMPILE = 3
MAX_OP_TEST_TUS = 24
MAX_DIAG_LINES, MAX_DIAG_CHARS = 60, 6000

SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx")
HEADER_SUFFIXES = (".h", ".hh", ".hpp", ".hxx", ".inc", ".inl")
#: A process under the launching loop with one of these argv[0] basenames is a
#: measurement, a calibration or an oracle run in progress.
MEASURING = frozenset({"llama-bench", "llama-server", "llama-cli", "llama-perplexity",
                       "llama-batched-bench", "test-backend-ops", "perf"})
LOOP_MARKERS = ("autokernel.loop.run", "autokernel/loop/run.py",
                "autokernel.loop.serial_run", "autokernel/loop/serial_run.py")
SELF_MARKER = "ak_check"

EXIT_PASS, EXIT_FAIL, EXIT_REFUSED, EXIT_USAGE = 0, 1, 2, 64
TYPE_TOKEN = re.compile(r"(?<![A-Za-z0-9])(?:block_|Dequantizer)?[Qq]([2-8])_?([Kk01])(?![A-Za-z0-9])")
#: Quant families by the iqk file that implements them, when the diff names none.
FILE_TYPES = {"iqk_gemm_kquants": ("q2_K", "q3_K", "q4_K", "q5_K", "q6_K"),
              "iqk_gemm_legacy_quants": ("q4_0", "q4_1", "q5_0", "q5_1", "q8_0")}
DEFAULT_TYPES = ("q4_K", "q5_K", "q8_0")


class Refused(RuntimeError):
    """The sandbox will not run now (tail active, no scratch, ...): not about the patch."""


class TailFenceTimeout(RuntimeError):
    """The tail could not take the fence within TAIL_WAIT_S."""


# --------------------------------------------------------------------------------------
# The fence.

def fence_dir(lane: Path) -> Path:
    return Path(lane).parent / FENCE_DIR_NAME


def _open_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a+")


@contextmanager
def sandbox_slot(fence: Path) -> Iterator[None]:
    """ak-check's side: a SHARED slot held for the whole check, or `Refused`.

    The gate is taken shared only long enough to take the slot: a tail holds the gate
    exclusively from before it waits until its session ends, so a check that starts
    while a tail is waiting or running is refused, and a waiting tail cannot be starved
    by a stream of overlapping checks from other lanes."""
    gate = _open_lock(fence / GATE_LOCK)
    slot = None
    try:
        try:
            fcntl.flock(gate, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EACCES):
                raise
            raise Refused("a tail measurement/calibration of this campaign is active "
                          "(its fence is held)") from None
        slot = _open_lock(fence / SLOT_LOCK)
        try:
            fcntl.flock(slot, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EACCES):
                raise
            raise Refused("a tail measurement/calibration of this campaign is active") from None
        fcntl.flock(gate, fcntl.LOCK_UN)
        yield
    finally:
        gate.close()
        if slot is not None:
            slot.close()


@contextmanager
def tail_fence(fences: Iterable[Path], *, wait_s: float = TAIL_WAIT_S,
               poll_s: float = 0.25, clock=time.monotonic, sleep=time.sleep) -> Iterator[None]:
    """The loop's side, around ONE tail session: exclusive gate then exclusive slot on
    every fence dir. Waits for running checks (each is bounded by OP_TEST_TIMEOUT_S);
    `TailFenceTimeout` past `wait_s`, never a measurement beside a compile."""
    handles = []
    deadline = clock() + wait_s
    try:
        for fence in sorted({Path(f) for f in fences}):
            for name in (GATE_LOCK, SLOT_LOCK):
                handle = _open_lock(Path(fence) / name)
                handles.append(handle)
                while True:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError as exc:
                        if exc.errno not in (errno.EAGAIN, errno.EACCES):
                            raise
                        if clock() >= deadline:
                            raise TailFenceTimeout(
                                f"an ak-check still holds {fence / name} after {wait_s:.0f} s; "
                                "the tail does not measure beside a compile") from None
                        sleep(poll_s)
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def _proc_table() -> dict[int, tuple[int, list[str]]]:
    table: dict[int, tuple[int, list[str]]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text()
            ppid = int(stat[stat.rfind(")") + 2:].split()[1])
            argv = Path(f"/proc/{entry}/cmdline").read_bytes().split(b"\0")
        except (OSError, ValueError, IndexError):
            continue
        table[int(entry)] = (ppid, [a.decode(errors="replace") for a in argv if a])
    return table


def measuring_processes(pid: int | None = None,
                        table: Mapping[int, tuple[int, list[str]]] | None = None) -> list[str]:
    """Measuring binaries under the topmost autokernel loop that launched `pid`, outside
    any ak-check's own subtree. Empty when no loop launched us (a manual run)."""
    table = _proc_table() if table is None else table
    pid = os.getpid() if pid is None else pid
    root, seen, cursor = None, set(), pid
    while cursor in table and cursor not in seen and cursor > 1:
        seen.add(cursor)
        if any(marker in " ".join(table[cursor][1]) for marker in LOOP_MARKERS):
            root = cursor
        cursor = table[cursor][0]
    if root is None:
        return []
    children: dict[int, list[int]] = {}
    for child, (parent, _argv) in table.items():
        children.setdefault(parent, []).append(child)
    found, stack = [], list(children.get(root, ()))
    while stack:
        current = stack.pop()
        argv = table[current][1]
        if any(SELF_MARKER in arg for arg in argv[:3]):
            continue      # a check's own compiler / test-backend-ops
        if argv and Path(argv[0]).name in MEASURING:
            found.append(f"pid {current}: {' '.join(argv)[:160]}")
        stack.extend(children.get(current, ()))
    return found


# --------------------------------------------------------------------------------------
# Which translation units, and with which command.

def _git(lane: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(lane), *args], capture_output=True, text=True,
                          timeout=60, check=True).stdout


def changed_files(lane: Path, base: str = "HEAD") -> list[str]:
    """Lane-relative paths that differ from `base` in the working tree, untracked
    files included (a new TU is reported, not silently skipped)."""
    tracked = _git(lane, "diff", "--name-only", base, "--").split("\n")
    untracked = _git(lane, "ls-files", "--others", "--exclude-standard").split("\n")
    return sorted({p for p in tracked + untracked if p.strip()})


def load_compile_db(path: Path) -> dict[str, dict]:
    entries = json.loads(Path(path).read_text())
    return {str(Path(e["directory"], e["file"]).resolve()) if not Path(e["file"]).is_absolute()
            else str(Path(e["file"])): e for e in entries}


def anchor_root_of(build_dir: Path) -> Path:
    build = Path(build_dir)
    return next((p for p in build.parents if (p / ".git").exists()), build.parent)


def _depfile(entry: Mapping, build_dir: Path) -> Path | None:
    output = entry.get("output")
    if output:
        candidate = Path(build_dir) / f"{output}.d"
        if candidate.is_file():
            return candidate
    argv = _argv(entry)
    if "-o" in argv:
        candidate = Path(entry["directory"]) / f"{argv[argv.index('-o') + 1]}.d"
        if candidate.is_file():
            return candidate
    return None


def _depends_on(entry: Mapping, build_dir: Path, header_abs: str) -> bool:
    dep = _depfile(entry, build_dir)
    if dep is None:
        return False
    try:
        text = dep.read_text(errors="replace")
    except OSError:
        return False
    return any(tok == header_abs for tok in text.replace("\\\n", " ").split())


def select_units(changed: Sequence[str], db: Mapping[str, dict], *, anchor_root: Path,
                 build_dir: Path, max_header_tus: int) -> tuple[list[tuple[str, dict]], list[str]]:
    """(lane-relative TU path, compile-db entry) pairs, and notes for what was skipped."""
    units: dict[str, dict] = {}
    notes: list[str] = []
    root = str(Path(anchor_root))
    for rel in changed:
        if rel.endswith(SOURCE_SUFFIXES):
            entry = db.get(f"{root}/{rel}")
            if entry is None:
                notes.append(f"{rel}: no compile command in the anchor build (a new or "
                             "unbuilt TU); not checked")
            else:
                units[rel] = entry
    for rel in changed:
        if not rel.endswith(HEADER_SUFFIXES):
            continue
        header = f"{root}/{rel}"
        users = sorted((path for path, entry in db.items()
                        if path.startswith(root + "/") and _depends_on(entry, build_dir, header)),
                       key=lambda p: (Path(p).parent != Path(header).parent, p))
        if not users:
            notes.append(f"{rel}: no anchor TU includes it; not checked")
            continue
        picked = [p for p in users if p[len(root) + 1:] not in units][:max_header_tus]
        for path in picked:
            units[path[len(root) + 1:]] = db[path]
        if len(users) > len(picked):
            notes.append(f"{rel}: included by {len(users)} TUs; checked "
                         f"{len(picked)} ({', '.join(p[len(root) + 1:] for p in picked)})")
    return sorted(units.items()), notes


def _argv(entry: Mapping) -> list[str]:
    return list(entry["arguments"]) if "arguments" in entry else shlex.split(entry["command"])


_DEP_FLAGS_WITH_ARG = {"-MF", "-MT", "-MQ"}
_DEP_FLAGS = {"-MD", "-MMD", "-M", "-MM", "-MP"}


def rewrite_command(entry: Mapping, *, anchor_root: Path, lane: Path,
                    output: Path | None) -> list[str]:
    """The anchor's compile command for one TU, re-pointed at the lane.

    Every path under the anchor source root moves to the lane (the TU itself and its
    `-I` dirs, so an edited header is the lane's), except paths under a `build*` dir of
    the root (generated headers). Dependency-file flags are dropped; `-o` becomes
    `output` (None: `-fsyntax-only`, no output at all)."""
    root = str(Path(anchor_root)).rstrip("/")
    pattern = re.compile(re.escape(root) + r"(?=/|$)(?!/build)")
    argv = _argv(entry)
    out: list[str] = []
    skip = False
    for i, token in enumerate(argv):
        if skip:
            skip = False
            continue
        if token == "-o":
            skip = True
            continue
        if token in _DEP_FLAGS_WITH_ARG:
            skip = True
            continue
        if token in _DEP_FLAGS or token.startswith(("-MF", "-MT", "-MQ")) and len(token) > 3:
            continue
        out.append(pattern.sub(str(lane), token))
    if output is None:
        out.append("-fsyntax-only")
    else:
        out += ["-o", str(output)]
    out += ["-fdiagnostics-color=never", "-fmax-errors=40"]
    return out


def touched_types(lane: Path, base: str, rels: Sequence[str]) -> tuple[str, ...]:
    """The quant types the diff names (`Q4_K`, `block_q5_K`, `DequantizerQ4K`, ...),
    else the family of the iqk file it touches, else DEFAULT_TYPES."""
    found: set[str] = set()
    try:
        diff = _git(lane, "diff", "-U0", base, "--", *rels) if rels else ""
    except (subprocess.SubprocessError, OSError):
        diff = ""
    for line in diff.splitlines():
        if line.startswith(("+", "-", "@@")) and not line.startswith(("+++", "---")):
            for number, kind in TYPE_TOKEN.findall(line):
                found.add(f"q{number}_{kind.upper() if kind in 'kK' else kind}")
    if found:
        return tuple(sorted(found))
    for rel in rels:
        for stem, types in FILE_TYPES.items():
            if stem in rel:
                return types
    return DEFAULT_TYPES


# --------------------------------------------------------------------------------------
# Running things: nice, pinned, bounded, own process group.

def parse_cpus(text: str) -> list[int]:
    cpus: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def default_cpus(count: int) -> list[int]:
    """`AK_CHECK_CPUS` if set, else the LAST `count` CPUs of this process's affinity
    (inherited from the loop, so inside the run's own claim when it confines itself)."""
    text = os.environ.get(ENV_CPUS)
    cpus = parse_cpus(text) if text else sorted(os.sched_getaffinity(0))
    return cpus[:count] if text else cpus[-count:]


def _cpu_text(cpus: Sequence[int]) -> str:
    return ",".join(str(c) for c in cpus)


def run_bounded(argv: Sequence[str], *, cpus: Sequence[int], timeout_s: float,
                cwd: Path, env: Mapping[str, str] | None = None) -> tuple[int, str, bool]:
    """(returncode, stdout+stderr, timed_out). The child is `nice -n 19 taskset -c`,
    in its own session so a timeout ends the whole group (compiler + cc1plus)."""
    command = ["nice", "-n", "19", "taskset", "-c", _cpu_text(cpus), *argv]
    proc = subprocess.Popen(command, cwd=str(cwd), env=dict(env) if env else None,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            errors="replace", start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=max(1.0, timeout_s))
        return proc.returncode, out, False
    except subprocess.TimeoutExpired:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                out, _ = proc.communicate(timeout=5)
                break
            except subprocess.TimeoutExpired:
                out = ""
        return -9, out or "", True


_DIAG = re.compile(r":\d+:\d+: (?:fatal error|error|warning): ")
_SNIPPET = re.compile(r"^\s+(?:\d+\s*)?\||^\s{2,}\S")


def diagnostics(text: str, lane: Path) -> str:
    """Compiler output, lane-relative, with each distinct diagnostic shown once (a
    template error repeats once per instantiation: `nrc_y` 1..8 x every dequantizer),
    truncated to MAX_DIAG_LINES / MAX_DIAG_CHARS."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text).replace(str(lane).rstrip("/") + "/", "")
    groups: list[list[str]] = [[]]
    keys: list[str | None] = [None]
    for line in text.splitlines():
        if not line.strip():
            continue
        if keys[-1] is not None and not _SNIPPET.match(line) and " note: " not in line:
            groups.append([])
            keys.append(None)
        groups[-1].append(line)
        if keys[-1] is None and _DIAG.search(line):
            keys[-1] = line
    seen: dict[str, int] = {}
    kept: list[list[str]] = []
    for group, key in zip(groups, keys):
        if key is not None and key in seen:
            seen[key] += 1
            continue
        if key is not None:
            seen[key] = 0
        kept.append(group)
    lines = [l for group in kept for l in group]
    repeats = sum(seen.values())
    errors = sum(1 for k in seen if "error: " in k)
    if seen:
        lines.insert(0, f"[{errors} distinct error(s), {len(seen) - errors} warning(s)"
                        + (f"; {repeats} repeat(s) from other template instantiations "
                           "omitted]" if repeats else "]"))
    if len(lines) > MAX_DIAG_LINES:
        lines = lines[:MAX_DIAG_LINES] + [f"... {len(lines) - MAX_DIAG_LINES} more lines"]
    out = "\n".join(lines)
    return out if len(out) <= MAX_DIAG_CHARS else out[:MAX_DIAG_CHARS] + "\n... truncated"


# --------------------------------------------------------------------------------------
# Scratch layout (inside the loop-allocated dir only).

def _sha(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def _object_path(scratch: Path, rel: str) -> Path:
    return scratch / "obj" / f"{rel}.o"


def _object_key(argv: Sequence[str], lane: Path, rel: str, changed: Sequence[str]) -> str:
    """Compiler argv (no -o) + the TU + every changed file's bytes: conservative (an
    edited header recompiles every cached TU) and correct."""
    parts = [json.dumps([a for a in argv], separators=(",", ":")).encode()]
    for path in [rel, *sorted(changed)]:
        try:
            parts.append(path.encode() + b"\0" + (Path(lane) / path).read_bytes())
        except OSError:
            parts.append(path.encode() + b"\0<absent>")
    return _sha(parts)


def tree_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def compile_units(units: Sequence[tuple[str, dict]], *, lane: Path, anchor_root: Path,
                  scratch: Path, changed: Sequence[str], cpus: Sequence[int],
                  timeout_s: float) -> list[dict]:
    """Compile each TU to its scratch object, reusing one whose key still matches."""
    def one(item):
        rel, entry = item
        obj = _object_path(scratch, rel)
        argv = rewrite_command(entry, anchor_root=anchor_root, lane=lane, output=obj)
        key = _object_key([a for a in argv if a != str(obj)], lane, rel, changed)
        stamp = obj.with_suffix(obj.suffix + ".key")
        if obj.is_file() and stamp.is_file() and stamp.read_text() == key:
            return {"tu": rel, "ok": True, "cached": True, "seconds": 0.0, "output": ""}
        obj.parent.mkdir(parents=True, exist_ok=True)
        for stale in (obj, stamp):
            stale.unlink(missing_ok=True)
        started = time.monotonic()
        code, out, timed_out = run_bounded(argv, cpus=cpus, timeout_s=timeout_s,
                                           cwd=Path(entry["directory"]))
        ok = code == 0 and not timed_out and obj.is_file()
        if ok:
            stamp.write_text(key)
        return {"tu": rel, "ok": ok, "cached": False, "timed_out": timed_out,
                "exit": code, "seconds": round(time.monotonic() - started, 1),
                "output": diagnostics(out, lane)}
    workers = max(1, min(len(units), len(cpus)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, units))


def _target_of(entry: Mapping, build_dir: Path) -> tuple[Path, str] | None:
    """(the dir CMake runs the target's link line from, target name)."""
    output = str(entry.get("output") or "")
    match = re.search(r"(?:^|/)CMakeFiles/([^/]+)\.dir/", output)
    if not match:
        return None
    return (Path(build_dir) / output[:match.start()].rstrip("/")
            if match.start() else Path(build_dir)), match.group(1)


def relink(target_dir: Path, target: str, *, objects: Mapping[str, Path], scratch: Path,
           cpus: Sequence[int], timeout_s: float) -> dict:
    """Re-run the anchor's `link.txt` for one target with recompiled objects swapped in,
    writing only under `scratch/bin`. Every relative input is made absolute against the
    anchor dir (read-only); the dependency-file side output is dropped."""
    link = target_dir / "CMakeFiles" / f"{target}.dir" / "link.txt"
    if not link.is_file():
        return {"target": target, "ok": False, "output": f"no link line at {link}"}
    argv = shlex.split(link.read_text().strip().splitlines()[0])
    out: list[str] = []
    produced: Path | None = None
    skip = False
    for i, token in enumerate(argv):
        if skip:
            skip = False
            continue
        if token.startswith("-Wl,--dependency-file="):
            continue
        if token == "-o":
            produced = scratch / "bin" / Path(argv[i + 1]).name
            out += ["-o", str(produced)]
            skip = True
            continue
        if token.endswith(".o") and not token.startswith("-"):
            key = str((target_dir / token).resolve()) if not Path(token).is_absolute() else token
            out.append(str(objects.get(key, key)))
            continue
        if not token.startswith("-") and not Path(token).is_absolute() and i > 0 \
                and (target_dir / token).exists():
            out.append(str((target_dir / token).resolve()))
            continue
        out.append(token)
    if produced is None:
        return {"target": target, "ok": False, "output": "link line names no -o output"}
    produced.parent.mkdir(parents=True, exist_ok=True)
    produced.unlink(missing_ok=True)
    started = time.monotonic()
    code, text, timed_out = run_bounded(out, cpus=cpus, timeout_s=timeout_s, cwd=scratch)
    return {"target": target, "ok": code == 0 and not timed_out and produced.is_file(),
            "produced": str(produced), "seconds": round(time.monotonic() - started, 1),
            "output": text[-2000:]}


def mirror_sonames(anchor_bin: Path, scratch_bin: Path) -> None:
    """The anchor's soname symlinks, for each library relinked into the scratch bin."""
    for link in Path(anchor_bin).iterdir():
        if not link.is_symlink():
            continue
        real = link.resolve().name
        if (scratch_bin / real).is_file() and not (scratch_bin / link.name).exists():
            (scratch_bin / link.name).symlink_to(real)


def verify_linkage(binary: Path, env: Mapping[str, str], scratch_bin: Path,
                   relinked: Sequence[str]) -> tuple[bool, str]:
    """Every relinked library resolves under the scratch bin for `binary` (ldd honours
    LD_LIBRARY_PATH ahead of the anchor's RUNPATH, which is what the loader does)."""
    done = subprocess.run(["ldd", str(binary)], capture_output=True, text=True,
                          env=dict(env), timeout=30)
    lines = {}
    for line in done.stdout.splitlines():
        name, _, rest = line.strip().partition(" => ")
        lines[name.strip()] = rest.split(" (")[0].strip()
    bad = []
    for lib in relinked:
        stem = lib.split(".so")[0] + ".so"
        hits = {name: path for name, path in lines.items() if name.startswith(stem)}
        if hits and not all(path.startswith(str(scratch_bin)) for path in hits.values()):
            bad.append(f"{lib} -> {', '.join(hits.values())}")
    return (not bad, "; ".join(bad) or "relinked libraries resolve to the scratch bin")


_COUNT = re.compile(r"^\s*(\d+)/(\d+) tests passed", re.M)


def parse_backend_ops(text: str) -> dict:
    plain = re.sub(r"\x1b\[[0-9;]*m", "", text)
    status = re.search(r"^\s*Backend CPU: (OK|FAIL)\b", plain, re.M)
    counts = [(int(a), int(b)) for a, b in _COUNT.findall(plain)]
    failing = [l.strip() for l in plain.splitlines()
               if re.search(r"test failed\s+FAIL\b|\bFAIL$", l.strip())
               and "Backend CPU" not in l and l.strip() != "FAIL"]
    engaged = sorted(set(re.findall(r"\[iqk\] ACTIVE: ([^\n(]*)", plain)))
    passed, total = (sum(c[0] for c in counts), sum(c[1] for c in counts)) if counts else (0, 0)
    return {"status": status.group(1) if status else None, "passed": passed, "total": total,
            "failed": failing[:12], "failed_count": len(failing),
            "iqk_engaged": [e.strip() for e in engaged]}


# --------------------------------------------------------------------------------------
# The two modes.

def compile_check(*, lane: Path, build_dir: Path, db_path: Path, scratch: Path, base: str,
                  cpus: Sequence[int], timeout_s: float = COMPILE_TIMEOUT_S,
                  syntax_only: bool = False) -> dict:
    anchor_root = anchor_root_of(build_dir)
    changed = changed_files(lane, base)
    db = load_compile_db(db_path)
    units, notes = select_units(changed, db, anchor_root=anchor_root, build_dir=build_dir,
                                max_header_tus=MAX_HEADER_TUS_COMPILE)
    if not units:
        return {"status": "nothing", "changed": changed, "units": [], "notes": notes}
    if syntax_only:
        results = []
        for rel, entry in units:
            argv = rewrite_command(entry, anchor_root=anchor_root, lane=lane, output=None)
            started = time.monotonic()
            code, out, timed_out = run_bounded(argv, cpus=cpus, timeout_s=timeout_s,
                                               cwd=Path(entry["directory"]))
            results.append({"tu": rel, "ok": code == 0 and not timed_out, "cached": False,
                            "timed_out": timed_out, "exit": code,
                            "seconds": round(time.monotonic() - started, 1),
                            "output": diagnostics(out, lane)})
    else:
        results = compile_units(units, lane=lane, anchor_root=anchor_root, scratch=scratch,
                                changed=changed, cpus=cpus, timeout_s=timeout_s)
    return {"status": "pass" if all(r["ok"] for r in results) else "fail",
            "changed": changed, "units": results, "notes": notes}


def build_commit(build_dir: Path, lane: Path) -> str | None:
    """The source commit the anchor build was made from (its IDENTITY.json `head`),
    when the lane knows it: diffing against it makes champion keeps that the anchor
    objects predate recompile too."""
    try:
        head = json.loads((Path(build_dir) / "IDENTITY.json").read_text()).get("head")
        if head:
            _git(lane, "cat-file", "-e", f"{head}^{{commit}}")
            return str(head)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def op_test(*, lane: Path, build_dir: Path, db_path: Path, scratch: Path, base: str,
            cpus: Sequence[int], deadline: float, types: Sequence[str] | None = None,
            ops: Sequence[str] | None = None) -> dict:
    anchor_root = anchor_root_of(build_dir)
    diff_base = build_commit(build_dir, lane) or base
    changed = changed_files(lane, diff_base)
    db = load_compile_db(db_path)
    units, notes = select_units(changed, db, anchor_root=anchor_root, build_dir=build_dir,
                                max_header_tus=MAX_OP_TEST_TUS)
    if not units:
        return {"status": "nothing", "changed": changed, "notes": notes, "diff_base": diff_base}
    if len(units) > MAX_OP_TEST_TUS:
        return {"status": "skipped", "changed": changed, "diff_base": diff_base,
                "notes": notes + [f"{len(units)} TUs to rebuild exceeds {MAX_OP_TEST_TUS}; "
                                  "the op test would not fit its budget (compile check only)"]}
    results = compile_units(units, lane=lane, anchor_root=anchor_root, scratch=scratch,
                            changed=changed, cpus=cpus,
                            timeout_s=min(COMPILE_TIMEOUT_S, deadline - time.monotonic()))
    report: dict = {"changed": changed, "units": results, "notes": notes,
                    "diff_base": diff_base}
    if not all(r["ok"] for r in results):
        return {**report, "status": "fail", "stage": "compile"}
    targets: dict[tuple[Path, str], dict[str, Path]] = {}
    for rel, entry in units:
        where = _target_of(entry, build_dir)
        if where is None:
            notes.append(f"{rel}: no CMake target in its compile entry; not relinked")
            continue
        argv = _argv(entry)
        anchor_obj = str((Path(entry["directory"]) / argv[argv.index("-o") + 1]).resolve())
        targets.setdefault(where, {})[anchor_obj] = _object_path(scratch, rel)
    bin_dir = scratch / "bin"
    if bin_dir.is_dir():
        for stale in bin_dir.iterdir():
            stale.unlink()
    links = [relink(d, t, objects=objs, scratch=scratch, cpus=cpus,
                    timeout_s=min(LINK_TIMEOUT_S, deadline - time.monotonic()))
             for (d, t), objs in sorted(targets.items())]
    report["links"] = [{k: v for k, v in row.items() if k != "output" or not row["ok"]}
                       for row in links]
    if not all(row["ok"] for row in links):
        return {**report, "status": "error", "stage": "link"}
    bin_dir.mkdir(parents=True, exist_ok=True)
    mirror_sonames(Path(build_dir) / "bin", bin_dir)
    exe = bin_dir / "test-backend-ops"
    if not exe.is_file():
        exe = Path(build_dir) / "bin" / "test-backend-ops"
    if not exe.is_file():
        return {**report, "status": "error", "stage": "oracle",
                "notes": notes + [f"no test-backend-ops in {build_dir}/bin"]}
    env = {k: v for k, v in os.environ.items() if not k.startswith("GGML_")}
    env.update({"LD_LIBRARY_PATH": str(bin_dir) + (os.pathsep + env["LD_LIBRARY_PATH"]
                                                  if env.get("LD_LIBRARY_PATH") else ""),
                "GGML_IQK": "1", "OMP_NUM_THREADS": str(max(1, len(cpus) // 2))})
    relinked = [Path(row["produced"]).name for row in links]
    ok, why = verify_linkage(exe, env, bin_dir, relinked)
    report["linkage"] = why
    if not ok:
        return {**report, "status": "error", "stage": "linkage"}
    types = tuple(types or touched_types(lane, diff_base, changed))
    ops = tuple(ops or (("MUL_MAT", "MUL_MAT_ID")
                        if any(p.startswith("ggml/src/ggml-cpu/") for p in changed)
                        else ("MUL_MAT",)))
    argv = [str(exe), "test", "-b", "CPU", "-o", ",".join(ops),
            "-p", "type_a=(" + "|".join(types) + "),", "-j", "2"]
    started = time.monotonic()
    code, out, timed_out = run_bounded(argv, cpus=cpus, timeout_s=deadline - time.monotonic(),
                                       cwd=scratch, env=env)
    parsed = parse_backend_ops(out)
    report.update({"oracle": {"argv": argv[1:], "types": list(types), "ops": list(ops),
                              "seconds": round(time.monotonic() - started, 1),
                              "exit": code, "timed_out": timed_out, **parsed}})
    if timed_out:
        return {**report, "status": "error", "stage": "oracle",
                "notes": notes + ["test-backend-ops ran out of the op-test budget"]}
    if parsed["status"] == "FAIL" or (parsed["total"] and parsed["passed"] != parsed["total"]):
        if not parsed["failed"]:
            report["oracle"]["tail"] = diagnostics(out[-3000:], lane)
        return {**report, "status": "fail", "stage": "oracle"}
    if parsed["status"] != "OK" or not parsed["total"] or code != 0:
        report["oracle"]["tail"] = diagnostics(out[-3000:], lane)
        return {**report, "status": "error", "stage": "oracle",
                "notes": notes + ["test-backend-ops did not prove a nonempty CPU suite; "
                                  "this is a harness fault, NOT evidence about the patch"]}
    return {**report, "status": "pass"}


# --------------------------------------------------------------------------------------
# Rendering, the calls log, the loop-side helpers.

def render(result: Mapping) -> str:
    mode, status = result["mode"], result["status"]
    head = f"ak-check {mode}: {status.upper()}"
    if result.get("seconds") is not None:
        head += f"  [{result['seconds']:.1f} s"
        head += f", cpus {result['cpus']}]" if result.get("cpus") else "]"
    lines = [head]
    if result.get("reason"):
        lines.append(result["reason"])
    elif status == "nothing":
        lines.append("no changed C/C++ translation unit in the lane: nothing to check")
    for unit in result.get("units") or ():
        state = "ok" if unit["ok"] else ("TIMEOUT" if unit.get("timed_out") else "FAIL")
        lines.append(f"--- {unit['tu']}: {state}"
                     + (" (cached)" if unit.get("cached") else f" ({unit['seconds']} s)"))
        if not unit["ok"] and unit.get("output"):
            lines.append(unit["output"])
    for link in result.get("links") or ():
        if not link["ok"]:
            lines.append(f"--- link {link['target']}: FAIL\n{link.get('output', '')}")
    oracle = result.get("oracle")
    if oracle:
        lines.append(f"--- test-backend-ops -o {','.join(oracle['ops'])} types "
                     f"{','.join(oracle['types'])}: {oracle['passed']}/{oracle['total']} "
                     f"passed vs the CPU reference ({oracle['seconds']} s); iqk engaged: "
                     f"{'yes' if oracle['iqk_engaged'] else 'NO'}")
        for line in oracle.get("failed") or ():
            lines.append("    " + line)
        more = (oracle.get("failed_count") or 0) - len(oracle.get("failed") or ())
        if more > 0:
            lines.append(f"    ... and {more} more failing cases")
        if oracle.get("tail") and status != "pass":
            lines.append(oracle["tail"])
    for note in result.get("notes") or ():
        lines.append(f"note: {note}")
    if status == "fail":
        lines.append("Fix every error above, then run ak-check again. Never reply with a "
                     "patch that fails ak-check.")
    elif status in ("refused", "error"):
        lines.append("The sandbox could not check the patch (this is not about your "
                     "code). Review the patch by reading it instead.")
    return "\n".join(lines)


def record_call(log: Path | None, row: Mapping) -> None:
    if log is None:
        return
    try:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        pass   # evidence, never a reason to fail the check


def usage_summary(log: Path | str | None, call_id: str | None) -> dict | None:
    """The `ak_check` block of one actor call's metrics row: calls, pass/fail/refused/
    error counts, seconds, scratch bytes created, by mode. None when the call ran none."""
    if not log or not call_id:
        return None
    try:
        rows = [json.loads(line) for line in Path(log).read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return None
    rows = [r for r in rows if r.get("call_id") == call_id]
    if not rows:
        return {"calls": 0}
    out: dict = {"calls": len(rows), "seconds": round(sum(r.get("seconds") or 0 for r in rows), 1),
                 "bytes_created": sum(max(0, r.get("scratch_bytes_delta") or 0) for r in rows),
                 "scratch_bytes": rows[-1].get("scratch_bytes_after"),
                 "last_status": rows[-1].get("status"), "by_mode": {}}
    for status in ("pass", "fail", "refused", "error", "nothing", "skipped"):
        out[status] = sum(1 for r in rows if r.get("status") == status)
    for mode in sorted({r.get("mode") for r in rows}):
        sub = [r for r in rows if r.get("mode") == mode]
        out["by_mode"][mode] = {"calls": len(sub),
                                "pass": sum(1 for r in sub if r.get("status") == "pass"),
                                "fail": sum(1 for r in sub if r.get("status") == "fail"),
                                "seconds": round(sum(r.get("seconds") or 0 for r in sub), 1)}
    return out


def scratch_provider(registry, lane_name: str, *, op_test_bytes: int = OP_TEST_SCRATCH_BYTES):
    """The loop side of the scratch contract, for `AgentPlanner.sandbox_scratch`.

    Returns a callable giving (this lane's ak-check build dir in the ITERATION scope open
    on the calling thread, or None; a reason to degrade `--op-test`, or None). The dir is
    allocated once per iteration scope (`scope.dir(SCRATCH_KIND, lane)`) and reused by
    every author call in it; the registry releases it when the scope closes, on every
    exit path. `registry.ensure_free` failing degrades the op test to the compile check."""
    allocated: dict[str, Path] = {}

    def provide() -> tuple[Path | None, str | None]:
        scope = registry.current()
        while scope is not None and getattr(scope, "level", None) != "iteration":
            scope = getattr(scope, "parent", None)
        if scope is None or getattr(scope, "closed", False):
            return None, None
        path = allocated.get(scope.id)
        if path is None:
            allocated.clear()          # an earlier iteration's scope is closed
            path = allocated[scope.id] = Path(scope.dir(SCRATCH_KIND, lane_name))
        degrade = None
        if not registry.ensure_free(op_test_bytes):
            floor = getattr(registry, "min_free_bytes", 0) / 10 ** 9
            degrade = f"free disk is below the scratch floor ({floor:.0f} GB)"
        return path, degrade
    return provide


def shim_text(*, python: str, lane: Path, build_dir: Path, compile_db: Path) -> str:
    args = [python, str(Path(__file__).resolve()), "--lane", str(lane),
            "--build-dir", str(build_dir), "--compile-db", str(compile_db)]
    return ("#!/bin/bash\n# ak-check shim for one author call (written by the autokernel loop)\n"
            "set -euo pipefail\nexec " + " ".join(shlex.quote(a) for a in args) + ' "$@"\n')


def author_env(lane: Path, build_dir: Path, *, scratch: Path | None, log: Path,
               call_id: str, python: str = sys.executable,
               op_test_off: str | None = None,
               base_path: str | None = None) -> dict[str, str]:
    """The env an author call needs for `ak-check`: the shim's dir first on PATH, the
    loop-allocated scratch dir, the calls log and this call's id. Writes the shim INSIDE
    the loop-allocated scratch dir (released with its iteration scope), or, with none,
    beside the lane (never inside it: the lane's diff is the candidate)."""
    lane, build_dir = Path(lane), Path(build_dir)
    shim_dir = (Path(scratch) / SHIM_DIR_NAME if scratch is not None
                else lane.parent / SHIM_DIR_NAME / lane.name)
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / COMMAND
    tmp = shim.with_name(f".{COMMAND}.{os.getpid()}.tmp")
    tmp.write_text(shim_text(python=python, lane=lane, build_dir=build_dir,
                             compile_db=build_dir / "compile_commands.json"))
    tmp.chmod(0o755)
    os.replace(tmp, shim)
    path = base_path if base_path is not None else os.environ.get("PATH", "")
    env = {"PATH": str(shim_dir) + (os.pathsep + path if path else ""),
           ENV_LOG: str(log), ENV_CALL_ID: call_id}
    if scratch is not None:
        env[ENV_SCRATCH] = str(scratch)
    if op_test_off:
        env[ENV_OP_TEST] = "off:" + op_test_off
    return env


# --------------------------------------------------------------------------------------
# The command.

def _scratch_dir(arg: str | None) -> Path:
    text = arg or os.environ.get(ENV_SCRATCH)
    if not text:
        raise Refused(f"no scratch dir: the loop allocates one per lane per iteration and "
                      f"passes it as {ENV_SCRATCH}; ak-check creates build dirs nowhere else")
    path = Path(text)
    if not path.is_dir():
        raise Refused(f"scratch dir {path} does not exist (it is allocated by the loop, "
                      "never by ak-check)")
    if not (path / SCRATCH_MARKER).is_file():
        raise Refused(f"{path} carries no {SCRATCH_MARKER}: not a loop-allocated scratch "
                      "dir, and ak-check builds nowhere else")
    return path


@contextmanager
def lane_lock(scratch: Path, wait_s: float = LANE_WAIT_S) -> Iterator[None]:
    """One check at a time per scratch dir (a second concurrent call waits, bounded)."""
    handle = _open_lock(scratch / LANE_LOCK)
    deadline = time.monotonic() + wait_s
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise Refused("another ak-check of this lane is still running") from None
                time.sleep(0.5)
        yield
    finally:
        handle.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=COMMAND, description=__doc__.split("\n\n")[0])
    parser.add_argument("--op-test", action="store_true",
                        help="also relink and run test-backend-ops against the CPU reference")
    parser.add_argument("--lane", type=Path, default=Path.cwd())
    parser.add_argument("--build-dir", type=Path, required=True,
                        help="the anchor build (read-only)")
    parser.add_argument("--compile-db", type=Path, default=None,
                        help="default: <build-dir>/compile_commands.json")
    parser.add_argument("--scratch", default=None, help=f"default: ${ENV_SCRATCH}")
    parser.add_argument("--base", default="HEAD", help="the lane base the diff is against")
    parser.add_argument("--cpus", default=None, help=f"CPU list (default: ${ENV_CPUS} or "
                        "the last CPUs of the inherited affinity)")
    parser.add_argument("--types", default=None, help="quant types for --op-test, comma list")
    parser.add_argument("--ops", default=None, help="ops for --op-test, comma list")
    parser.add_argument("--syntax-only", action="store_true",
                        help="-fsyntax-only instead of compiling to objects (faster, "
                             "misses immediate-operand and codegen errors)")
    parser.add_argument("--no-fence", action="store_true", help=argparse.SUPPRESS)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else EXIT_PASS
    lane = args.lane.resolve()
    mode = "op-test" if args.op_test else "compile"
    started, started_at = time.monotonic(), time.time()
    log = Path(os.environ[ENV_LOG]) if os.environ.get(ENV_LOG) else None
    row: dict = {"schema": SCHEMA, "mode": mode, "lane": str(lane),
                 "call_id": os.environ.get(ENV_CALL_ID), "pid": os.getpid(),
                 "started_at": started_at}
    scratch: Path | None = None
    before = 0
    result: dict
    try:
        scratch = _scratch_dir(args.scratch)
        before = tree_bytes(scratch)
        db_path = args.compile_db or args.build_dir / "compile_commands.json"
        if not db_path.is_file():
            raise Refused(
                f"no {db_path}. Get one WITHOUT rebuilding (never on a production or "
                "kernel-store tree): configure-only a scratch dir from the same source "
                "and cache flags, `cmake -S <tree> -B <tree>/build-cpu-cc "
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON <the anchor CMakeCache flags>` (no build "
                "step runs), then pass --compile-db")
        if not args.no_fence:
            slot = sandbox_slot(fence_dir(lane))
        else:
            from contextlib import nullcontext
            slot = nullcontext()
        with slot:
            busy = measuring_processes()
            if busy:
                raise Refused("the loop that launched this check is measuring: "
                              + "; ".join(busy[:3]))
            with lane_lock(scratch):
                off = os.environ.get(ENV_OP_TEST, "")
                if args.op_test and off.startswith("off"):
                    reason = off.partition(":")[2] or "disabled by the loop"
                    cpus = parse_cpus(args.cpus)[:COMPILE_CPUS] if args.cpus \
                        else default_cpus(COMPILE_CPUS)
                    result = compile_check(lane=lane, build_dir=args.build_dir,
                                           db_path=db_path, scratch=scratch, base=args.base,
                                           cpus=cpus)
                    result["notes"] = [f"op test SKIPPED ({reason}); compile check only"] \
                        + result.get("notes", [])
                    result["op_test_skipped"] = reason
                    mode = row["mode"] = "op-test-degraded"
                elif args.op_test:
                    cpus = parse_cpus(args.cpus)[:OP_TEST_CPUS] if args.cpus \
                        else default_cpus(OP_TEST_CPUS)
                    result = op_test(lane=lane, build_dir=args.build_dir, db_path=db_path,
                                     scratch=scratch, base=args.base, cpus=cpus,
                                     deadline=time.monotonic() + OP_TEST_TIMEOUT_S,
                                     types=args.types.split(",") if args.types else None,
                                     ops=args.ops.split(",") if args.ops else None)
                else:
                    cpus = parse_cpus(args.cpus)[:COMPILE_CPUS] if args.cpus \
                        else default_cpus(COMPILE_CPUS)
                    result = compile_check(lane=lane, build_dir=args.build_dir,
                                           db_path=db_path, scratch=scratch, base=args.base,
                                           cpus=cpus, syntax_only=args.syntax_only)
                result["cpus"] = _cpu_text(cpus)
    except Refused as exc:
        result = {"status": "refused", "reason": f"REFUSED: {exc}"}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        result = {"status": "error", "reason": f"sandbox error: {type(exc).__name__}: {exc}"}
    result["mode"] = mode
    result["seconds"] = round(time.monotonic() - started, 1)
    after = tree_bytes(scratch) if scratch is not None and scratch.is_dir() else 0
    row.update({"status": result["status"], "seconds": result["seconds"],
                "scratch": str(scratch) if scratch else None,
                "scratch_bytes_before": before, "scratch_bytes_after": after,
                "scratch_bytes_delta": after - before,
                "units": [{k: u[k] for k in ("tu", "ok", "cached", "seconds") if k in u}
                          for u in result.get("units") or ()],
                "oracle": ({k: result["oracle"][k] for k in
                            ("passed", "total", "types", "ops", "seconds", "iqk_engaged")}
                           if result.get("oracle") else None),
                "reason": result.get("reason")})
    record_call(log, row)
    print(render(result))
    return {"pass": EXIT_PASS, "nothing": EXIT_PASS, "fail": EXIT_FAIL}.get(
        result["status"], EXIT_REFUSED)


__all__ = ["OP_TEST_SCRATCH_BYTES", "SCRATCH_KIND", "SCRATCH_MARKER", "scratch_provider",
           "ALLOWED_COMMANDS", "COMMAND", "ENV_CALL_ID", "ENV_LOG", "ENV_OP_TEST",
           "ENV_SCRATCH", "FENCE_DIR_NAME", "Refused", "TailFenceTimeout", "author_env",
           "changed_files", "compile_check", "fence_dir", "measuring_processes", "op_test",
           "rewrite_command", "sandbox_slot", "select_units", "tail_fence",
           "touched_types", "usage_summary"]

if __name__ == "__main__":
    sys.exit(main())
