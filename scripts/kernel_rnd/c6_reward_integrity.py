#!/usr/bin/env python3
"""c6_reward_integrity.py — C6 anti-reward-hacking + provenance harness for the
MI210 auto-kernel authoring loop (Phase 2 of the kernel-R&D loop).

STANDALONE, importable, GPU-free. This module is the loop's owned
differentiator — **C6 (reward integrity)** — plus its provenance layer. It is
NOT wired into kernel_eval.sh / kernel_sweep.sh; the future Phase-2 loop imports
it. The logic is ported from the proven design in the MIT-licensed reference
repo github.com/MrSteeeve/OpenHyra (sandbox.py / provenance.py / stopping.py /
eb.py), adapted to the SOL-ExecBench kernel task/scoring contract.

Discipline it enforces (why each exists):
  * Anti-TOCTOU snapshot — a candidate cannot mutate its artifact after the
    evaluator has looked at it. We SIGKILL the candidate's whole process group
    BEFORE snapshotting, open O_NOFOLLOW, reject symlink / FIFO / non-regular /
    multiply-linked files, cap size, then chmod 0444 an immutable copy.
  * Trusted evaluator — the score is RECOMPUTED by parent-controlled code on the
    immutable snapshot; any self-reported number in the candidate output is
    ignored. A candidate can never grade its own homework.
  * Correctness-gate-BEFORE-latency (lexicographic, mirrors kernel_store's
    `_is_correct`) — `is_correct` MUST pass before any latency / sol_score is
    recorded or ranked. A fast-but-wrong kernel scores nothing.
  * Run-manifest provenance — a sha256 over {sources, task spec, evaluator,
    config}; a resume is REFUSED if any result-affecting input drifted. A
    flock-based single-writer RunLock stops two harnesses sharing one run dir.
  * Evidence-gated stop — an autonomous "stop" is only a REQUEST; it is honored
    only when deterministic guards computed from evaluator RECORDS agree.
    Malformed / empty input can never trigger a stop.
  * Linux sandbox backend — bwrap / unshare + resource.setrlimit (NOT macOS
    Seatbelt). If no sandbox tool works, we FAIL CLOSED (raise) — we never run a
    candidate unsandboxed silently. Availability is probed at import.

Every number this harness produces is an OBSERVATION (MEASUREMENT.md): it has no
protocol id and NEVER gates a keep/revert/deploy/promote decision. The operator
alone authorizes any production push.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shutil
import signal
import stat
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

# --- read-only reuse of kernel_store's correctness semantics -----------------
# We import kernel_store purely to reuse `_is_correct` (its lexicographic
# correctness definition for kernel_eval.sh JSONL records). The import has no
# side effects (kernel_store only defines functions/constants at module scope;
# any DB work is guarded behind functions and __main__). We NEVER call its
# mutating entry points and we do not modify any of its symbols.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:  # pragma: no cover - trivial import guard
    import kernel_store as _kernel_store
except Exception:  # pragma: no cover
    _kernel_store = None


# =============================================================================
# Exceptions
# =============================================================================
class SandboxUnavailable(RuntimeError):
    """Raised (fail-closed) when no sandbox backend is available and the caller
    did not explicitly authorize unsandboxed execution."""


class ArtifactRejected(ValueError):
    """Raised when a candidate artifact fails an anti-TOCTOU integrity check."""


class ProvenanceError(RuntimeError):
    """Raised on run-manifest checksum mismatch or resume-drift rejection."""


class RunLockError(RuntimeError):
    """Raised when a run directory is already owned by another writer."""


# =============================================================================
# SOL-ExecBench task / scoring contract
# =============================================================================
@dataclass(frozen=True)
class KernelTaskSpec:
    """The SOL-ExecBench C5/C6 task + scoring contract.

    The six scoring-core fields are the contract the prompt pins:
      entry_point      e.g. "kernel.py::run" — module file + callable.
      target_hardware  e.g. "MI210/gfx90a".
      dependencies     declared deps (tuple of package names).
      is_correct       HARD GATE — must be True before any score/latency ranks.
      sol_score        speed-of-light score; None until correctness passes.
      latency_ms       measured latency; None until correctness passes.

    is_correct / sol_score / latency_ms are RESULT fields: they are populated
    ONLY by the trusted evaluator (see `trusted_evaluate`), never read from the
    candidate's self-report. On a fresh task spec they are the un-evaluated
    defaults (False / None / None).

    The remaining fields are execution + provenance metadata used by the harness
    (limits, evaluator path, declared source files, free-form config).
    """

    entry_point: str
    target_hardware: str
    dependencies: tuple[str, ...] = ()
    # --- result contract (trusted-evaluator-owned) ---
    is_correct: bool = False
    sol_score: float | None = None
    latency_ms: float | None = None
    # --- execution / provenance metadata ---
    evaluator: str | None = None
    sources: tuple[str, ...] = ()
    config: dict = field(default_factory=dict)
    artifact_name: str = "solution.json"
    timeout_s: int = 60
    max_memory_mb: int = 1024
    max_output_mb: int = 16
    max_artifact_bytes: int = 1024 * 1024
    evaluator_timeout_s: int = 120
    evaluator_max_memory_mb: int = 512

    def entry_module(self) -> str:
        """Return the module-file half of ``entry_point`` (before '::')."""
        return self.entry_point.split("::", 1)[0]

    def entry_callable(self) -> str | None:
        """Return the callable half of ``entry_point`` (after '::'), or None."""
        parts = self.entry_point.split("::", 1)
        return parts[1] if len(parts) == 2 else None

    def scoring_core(self) -> dict:
        """The six-field SOL-ExecBench scoring core as a plain dict."""
        return {
            "entry_point": self.entry_point,
            "target_hardware": self.target_hardware,
            "dependencies": list(self.dependencies),
            "is_correct": self.is_correct,
            "sol_score": self.sol_score,
            "latency_ms": self.latency_ms,
        }

    def with_result(self, is_correct, sol_score, latency_ms) -> "KernelTaskSpec":
        """Return a copy carrying an evaluation result, correctness-gated:
        a non-correct result NEVER carries a score or latency."""
        if not is_correct:
            sol_score = None
            latency_ms = None
        return replace(
            self,
            is_correct=bool(is_correct),
            sol_score=sol_score,
            latency_ms=latency_ms,
        )


@dataclass(frozen=True)
class KernelEvaluation:
    """A trusted-evaluator verdict. Constructed only via `gated(...)`, which
    enforces the correctness-before-latency invariant at the type boundary."""

    is_correct: bool
    sol_score: float | None
    latency_ms: float | None
    status: str  # ok | crash | timeout | rejected | cancelled
    note: str = ""
    metrics: dict = field(default_factory=dict)
    candidate_artifact_sha256: str | None = None

    @classmethod
    def gated(cls, *, is_correct, sol_score, latency_ms, status, note="",
              metrics=None, candidate_artifact_sha256=None) -> "KernelEvaluation":
        """Build a verdict with the hard gate applied: if not is_correct, the
        score and latency are dropped to None no matter what was passed in."""
        if not is_correct:
            sol_score = None
            latency_ms = None
        return cls(
            is_correct=bool(is_correct),
            sol_score=sol_score,
            latency_ms=latency_ms,
            status=status,
            note=note,
            metrics=metrics or {},
            candidate_artifact_sha256=candidate_artifact_sha256,
        )

    def to_record(self, task: KernelTaskSpec | None = None) -> dict:
        rec = asdict(self)
        if task is not None:
            rec["entry_point"] = task.entry_point
            rec["target_hardware"] = task.target_hardware
        rec["observation"] = True
        return rec


# =============================================================================
# Correctness gate (consistent with kernel_store._is_correct)
# =============================================================================
def kernel_eval_is_correct(rec: dict) -> int:
    """Lexicographic correctness for a kernel_eval.sh JSONL record.

    Delegates to kernel_store._is_correct when available (status==OK + full
    test-backend-ops pass + coherent/byte-identical output); otherwise mirrors
    that exact semantics so this module is self-contained if kernel_store cannot
    be imported."""
    if _kernel_store is not None:
        return _kernel_store._is_correct(rec)
    if rec.get("status") != "OK":
        return 0
    corr = rec.get("correctness", {}) or {}
    tbo = corr.get("test_backend_ops", "")
    ok_tbo = False
    if "/" in tbo:
        a = tbo.split("/")[0].strip().split()[-1]
        b = tbo.split("/")[1].strip().split()[0]
        ok_tbo = a.isdigit() and b.isdigit() and a == b
    ok_coh = corr.get("coherence") in ("byte-identical", "coherent")
    return 1 if (ok_tbo and ok_coh) else 0


def is_correct(obj) -> bool:
    """Unified correctness gate for either record shape.

    * A kernel_eval.sh JSONL record (has a 'correctness' block) is judged by
      `kernel_eval_is_correct`.
    * A C6 evaluation (KernelEvaluation, KernelTaskSpec, or a dict with an
      'is_correct' key) is judged by its boolean gate.
    """
    if isinstance(obj, (KernelEvaluation, KernelTaskSpec)):
        return bool(obj.is_correct)
    if isinstance(obj, dict):
        if "correctness" in obj:
            return bool(kernel_eval_is_correct(obj))
        if "is_correct" in obj:
            return bool(obj["is_correct"])
    return False


def _score_of(obj):
    if isinstance(obj, (KernelEvaluation, KernelTaskSpec)):
        return obj.sol_score
    if isinstance(obj, dict):
        return obj.get("sol_score")
    return None


def _latency_of(obj):
    if isinstance(obj, (KernelEvaluation, KernelTaskSpec)):
        return obj.latency_ms
    if isinstance(obj, dict):
        return obj.get("latency_ms")
    return None


def rank_correct_first(evaluations):
    """Rank evaluations lexicographic-correctness-first.

    ONLY correct evaluations with a real sol_score are ever ranked; a
    fast-but-wrong candidate is dropped and can never place. Among the correct,
    higher sol_score wins, ties broken by lower latency_ms. Returns a new list;
    the input is never mutated."""
    ranked = [
        e for e in evaluations
        if is_correct(e) and _score_of(e) is not None
    ]
    ranked.sort(
        key=lambda e: (
            -float(_score_of(e)),
            float(_latency_of(e)) if _latency_of(e) is not None else float("inf"),
        )
    )
    return ranked


# =============================================================================
# Linux sandbox backend (bwrap / unshare + setrlimit) — fail-closed
# =============================================================================
# A tiny in-process wrapper that clamps address space, output-file size and CPU
# seconds via resource.setrlimit, then exec()s the real command. Limits are
# inherited across the subsequent exec of the sandbox tool and the candidate.
LIMIT_WRAPPER = r"""
import os, resource, sys
limits = (
    (resource.RLIMIT_AS, int(sys.argv[1])),
    (resource.RLIMIT_FSIZE, int(sys.argv[2])),
    (resource.RLIMIT_CPU, int(sys.argv[3])),
)
for key, value in limits:
    if value <= 0:
        continue
    try:
        _soft, hard = resource.getrlimit(key)
        target = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(key, (target, target))
    except (OSError, ValueError):
        pass
os.execvp(sys.argv[4], sys.argv[4:])
"""

_ALLOW_ENV = "EPYC_C6_ALLOW_UNSANDBOXED"


def _probe_bwrap():
    exe = shutil.which("bwrap")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "--ro-bind", "/", "/", "--dev", "/dev", "true"],
            capture_output=True, timeout=10,
        )
        return exe if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _probe_unshare():
    exe = shutil.which("unshare")
    if not exe:
        return None
    try:
        # Actually attempt the namespace op — a present binary is not enough;
        # user namespaces are frequently disabled (unprivileged containers).
        r = subprocess.run(
            [exe, "--user", "--map-root-user", "--net", "--", "true"],
            capture_output=True, timeout=10,
        )
        return exe if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def detect_sandbox_backend():
    """Probe (not just which()) for a working sandbox backend.

    Returns (name, tool_path) — ('bwrap', ...) preferred, else ('unshare', ...),
    else (None, None). Called once at import; re-callable to re-probe."""
    tool = _probe_bwrap()
    if tool:
        return "bwrap", tool
    tool = _probe_unshare()
    if tool:
        return "unshare", tool
    return None, None


SANDBOX_BACKEND, SANDBOX_TOOL = detect_sandbox_backend()
SANDBOX_AVAILABLE = SANDBOX_BACKEND is not None


def _allow_unsandboxed(explicit):
    if explicit is not None:
        return bool(explicit)
    return os.environ.get(_ALLOW_ENV) == "1"


def build_sandboxed_command(cmd, *, writable_dir, allow_unsandboxed=None):
    """Wrap ``cmd`` (a list) so the candidate runs isolated.

    Isolation strength by backend:
      bwrap    — read-only root, private /dev+/proc+/tmp, one writable bind on
                 ``writable_dir``, all namespaces unshared (network denied).
      unshare  — new user+net+pid namespaces (network denied), mount-proc.
    If NO backend is available this FAILS CLOSED (raises SandboxUnavailable)
    unless the caller passes allow_unsandboxed=True or sets
    EPYC_C6_ALLOW_UNSANDBOXED=1 — appropriate only inside an already-isolated
    container/VM (e.g. this devcontainer or CI)."""
    cmd = list(cmd)
    writable_dir = str(Path(writable_dir).resolve())
    if SANDBOX_BACKEND == "bwrap":
        return [
            SANDBOX_TOOL,
            "--ro-bind", "/", "/",
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--bind", writable_dir, writable_dir,
            "--chdir", writable_dir,
            "--unshare-all",
            "--die-with-parent",
            "--",
        ] + cmd
    if SANDBOX_BACKEND == "unshare":
        return [
            SANDBOX_TOOL,
            "--user", "--map-root-user",
            "--net", "--pid", "--fork", "--mount-proc",
            "--",
        ] + cmd
    if _allow_unsandboxed(allow_unsandboxed):
        return cmd
    raise SandboxUnavailable(
        "no working sandbox backend (bwrap/unshare) — refusing to run a "
        "candidate unsandboxed. Set EPYC_C6_ALLOW_UNSANDBOXED=1 ONLY inside an "
        "external container/VM, or install bwrap / enable user namespaces."
    )


def rlimit_wrapped_command(cmd, task: KernelTaskSpec):
    """Prepend the setrlimit exec-wrapper (address space / fsize / cpu)."""
    mem = int(task.max_memory_mb) * 1024 * 1024
    out = int(task.max_output_mb) * 1024 * 1024
    cpu = int(task.timeout_s) + 5
    return [sys.executable, "-c", LIMIT_WRAPPER, str(mem), str(out), str(cpu), *cmd]


# =============================================================================
# Anti-TOCTOU immutable snapshot
# =============================================================================
READ_CHUNK_BYTES = 64 * 1024


def _kill_process_group(proc):
    """SIGKILL the candidate's whole session/process group. Closes the artifact
    mutation race: even descendants deliberately left running after the parent
    exits are gone before we snapshot."""
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


def _read_regular_file(path, max_bytes):
    """Read ONE untrusted artifact without following links or blocking on FIFOs.

    Rejects (raising ArtifactRejected): a symbolic link, a non-regular file
    (FIFO / socket / device), a multiply-linked file (st_nlink != 1), and any
    file over ``max_bytes``. Opens O_NOFOLLOW|O_NONBLOCK — the O_NOFOLLOW is the
    anti-race second line of defense if a symlink is swapped in AFTER the lstat
    check but BEFORE the open."""
    path = Path(path)
    try:
        before = os.lstat(path)
    except FileNotFoundError as exc:
        raise ArtifactRejected(f"artifact not found: {path}") from exc
    if stat.S_ISLNK(before.st_mode):
        raise ArtifactRejected(f"artifact must not be a symbolic link: {path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        # ELOOP here == a symlink was swapped in after the lstat (TOCTOU).
        raise ArtifactRejected(f"could not safely open artifact: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactRejected(
                f"artifact must be a regular file (got mode {oct(info.st_mode)}): {path}"
            )
        if info.st_nlink != 1:
            raise ArtifactRejected(
                f"artifact must have exactly one hard link (st_nlink="
                f"{info.st_nlink}): {path}"
            )
        if info.st_size > max_bytes:
            raise ArtifactRejected(
                f"artifact exceeds the {max_bytes}-byte limit: {path}"
            )
        chunks = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(fd, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise ArtifactRejected(
                f"artifact exceeds the {max_bytes}-byte limit: {path}"
            )
        return data
    finally:
        os.close(fd)


def snapshot_candidate_artifact(artifact_path, trusted_dir, *, proc=None,
                                max_bytes=1024 * 1024):
    """Kill the candidate's process group, then copy its validated artifact into
    a fresh parent-controlled directory as an immutable 0444 snapshot.

    The kill happens BEFORE the read so the candidate cannot mutate the file
    between validation and snapshot. Returns (snapshot_path, sha256, data)."""
    _kill_process_group(proc)
    data = _read_regular_file(artifact_path, max_bytes)
    trusted_dir = Path(trusted_dir)
    if trusted_dir.exists():
        shutil.rmtree(trusted_dir)
    trusted_dir.mkdir(parents=True)
    snapshot = trusted_dir / "solution.snapshot.json"
    snapshot.write_bytes(data)
    snapshot.chmod(0o444)
    digest = hashlib.sha256(data).hexdigest()
    return snapshot, digest, data


def trusted_artifact_dir(sandbox_dir):
    """A parent-controlled trusted dir OUTSIDE the candidate's write root."""
    sandbox_dir = Path(sandbox_dir)
    return sandbox_dir.parent / ".c6_trusted" / sandbox_dir.name


# =============================================================================
# Trusted evaluator (recompute score on the snapshot; ignore self-report)
# =============================================================================
def _wait_process(proc, timeout_s):
    started = time.monotonic()
    while True:
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            return "timeout"
        try:
            proc.wait(timeout=min(0.2, remaining))
            return "completed"
        except subprocess.TimeoutExpired:
            pass


def trusted_evaluate(task: KernelTaskSpec, snapshot_path) -> KernelEvaluation:
    """Recompute the verdict with PARENT-controlled evaluator code on the
    immutable snapshot. The candidate's own self-reported score/latency (if any
    inside the artifact) is IGNORED — the harness never reads a number from the
    candidate output; only the evaluator's JSON verdict counts.

    The evaluator is trusted code, so it runs under resource limits only (no
    sandbox), matching OpenHyra's trusted-scoring pattern. Its last stdout line
    must be a JSON object: {"is_correct": bool, "sol_score": <num>?,
    "latency_ms": <num>?, "metrics": {...}?} or {"error": "..."}.

    Correctness gate: even if the evaluator emits a score, a non-correct verdict
    is returned with sol_score=None and latency_ms=None."""
    if not task.evaluator:
        raise ValueError("task.evaluator is required for trusted_evaluate")
    snapshot_path = str(snapshot_path)
    digest = hashlib.sha256(Path(snapshot_path).read_bytes()).hexdigest()

    command = [sys.executable, str(task.evaluator), snapshot_path]
    limited = [
        sys.executable, "-c", LIMIT_WRAPPER,
        str(int(task.evaluator_max_memory_mb) * 1024 * 1024),
        str(int(task.max_output_mb) * 1024 * 1024),
        str(int(task.evaluator_timeout_s) + 5),
        *command,
    ]
    started = time.perf_counter()
    proc = subprocess.Popen(
        limited, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    state = _wait_process(proc, task.evaluator_timeout_s)
    _kill_process_group(proc)  # trusted code must not leave descendants either
    stdout, stderr = proc.communicate()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if state == "timeout":
        return KernelEvaluation.gated(
            is_correct=False, sol_score=None, latency_ms=None, status="timeout",
            note="evaluator timed out", candidate_artifact_sha256=digest,
        )
    line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    try:
        verdict = json.loads(line)
    except ValueError:
        return KernelEvaluation.gated(
            is_correct=False, sol_score=None, latency_ms=None, status="crash",
            note=f"evaluator produced no verdict: {stderr.strip()[:300]}",
            candidate_artifact_sha256=digest,
        )
    if "error" in verdict:
        return KernelEvaluation.gated(
            is_correct=False, sol_score=None, latency_ms=None, status="rejected",
            note=f"evaluator rejected artifact: {verdict['error']}",
            candidate_artifact_sha256=digest,
        )

    correct = bool(verdict.get("is_correct"))
    score = verdict.get("sol_score")
    latency = verdict.get("latency_ms")
    metrics = dict(verdict.get("metrics", {}))
    metrics["evaluator_ms"] = round(elapsed_ms, 4)
    return KernelEvaluation.gated(
        is_correct=correct,
        sol_score=(float(score) if score is not None else None),
        latency_ms=(float(latency) if latency is not None else None),
        status="ok",
        note="",
        metrics=metrics,
        candidate_artifact_sha256=digest,
    )


def evaluate_candidate(candidate_cmd, work_dir, task: KernelTaskSpec, *,
                       env=None, allow_unsandboxed=None) -> KernelEvaluation:
    """End-to-end C6 candidate evaluation:

      1. run the (untrusted) candidate command sandboxed + rlimited in work_dir;
      2. SIGKILL its process group (close the mutation race);
      3. anti-TOCTOU snapshot of its artifact into a trusted dir (immutable);
      4. trusted-evaluator recomputes the verdict on the snapshot;
      5. correctness-gate the result (no score/latency unless is_correct).

    Returns a KernelEvaluation. Never raises for a merely-failing candidate
    (crash/timeout become a non-correct verdict); it DOES raise
    SandboxUnavailable if no sandbox backend and no explicit override."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    artifact = work_dir / task.artifact_name
    if artifact.exists():
        artifact.unlink()

    sandboxed = build_sandboxed_command(
        candidate_cmd, writable_dir=work_dir, allow_unsandboxed=allow_unsandboxed,
    )
    full = rlimit_wrapped_command(sandboxed, task)
    child_env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(work_dir),
        "TMPDIR": str(work_dir),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if env:
        child_env.update(env)

    log_path = work_dir / "run.log"
    with open(log_path, "w") as log_stream:
        proc = subprocess.Popen(
            full, cwd=str(work_dir), env=child_env,
            stdout=log_stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait_state = "completed"
        try:
            wait_state = _wait_process(proc, task.timeout_s)
        finally:
            _kill_process_group(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    log_tail = ""
    if log_path.exists():
        log_tail = "\n".join(
            log_path.read_text(errors="replace").splitlines()[-15:]
        )
    if wait_state == "timeout":
        return KernelEvaluation.gated(
            is_correct=False, sol_score=None, latency_ms=None, status="timeout",
            note=(f"killed candidate process group after {task.timeout_s}s\n"
                  f"{log_tail}").strip(),
        )
    if proc.returncode != 0:
        return KernelEvaluation.gated(
            is_correct=False, sol_score=None, latency_ms=None, status="crash",
            note=log_tail,
        )

    trusted_dir = trusted_artifact_dir(work_dir)
    try:
        snapshot, digest, _ = snapshot_candidate_artifact(
            artifact, trusted_dir, proc=proc, max_bytes=task.max_artifact_bytes,
        )
    except ArtifactRejected as exc:
        return KernelEvaluation.gated(
            is_correct=False, sol_score=None, latency_ms=None, status="rejected",
            note=(log_tail + f"\nartifact rejected: {exc}").strip(),
        )
    result = trusted_evaluate(task, snapshot)
    return replace(result, candidate_artifact_sha256=digest)


# =============================================================================
# Run-manifest provenance + single-writer RunLock
# =============================================================================
RUN_MANIFEST_SCHEMA = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload):
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def build_run_manifest(task: KernelTaskSpec, *, run_id, sources=None,
                       config=None):
    """Build an immutable run manifest: a sha256 over {sources, task spec,
    evaluator, config}. ``sources`` is a {logical_name: path} map of the loop's
    own source files whose content must not drift across a resume.

    A change to ANY hashed input flips manifest_sha256 (and the per-field
    blocks), so `validate_run_manifest` can refuse a poisoned resume."""
    sources = sources or {}
    if task.evaluator is None:
        raise ValueError("task.evaluator is required to build a run manifest")
    payload = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": run_id,
        "task": {
            "entry_point": task.entry_point,
            "target_hardware": task.target_hardware,
            "dependencies": list(task.dependencies),
            "artifact_name": task.artifact_name,
            "evaluator_sha256": sha256_file(task.evaluator),
        },
        "source_sha256": {
            name: sha256_file(path) for name, path in sorted(sources.items())
        },
        "config": config or dict(task.config),
        "limits": {
            "timeout_s": task.timeout_s,
            "max_memory_mb": task.max_memory_mb,
            "max_output_mb": task.max_output_mb,
            "max_artifact_bytes": task.max_artifact_bytes,
            "evaluator_timeout_s": task.evaluator_timeout_s,
            "evaluator_max_memory_mb": task.evaluator_max_memory_mb,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "sandbox_backend": SANDBOX_BACKEND,
        },
    }
    payload["manifest_sha256"] = sha256_json(payload)
    return payload


def write_run_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def load_run_manifest(path):
    path = Path(path)
    if not path.is_file():
        raise ProvenanceError(
            f"run provenance is missing: {path}; legacy runs cannot be resumed"
        )
    manifest = json.loads(path.read_text())
    expected = manifest.get("manifest_sha256")
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if expected != sha256_json(unsigned):
        raise ProvenanceError(f"run provenance checksum mismatch: {path}")
    return manifest


def validate_run_manifest(recorded, current):
    """Refuse to resume when any result-affecting input drifted.

    Compares the result-affecting blocks {task (incl. evaluator sha),
    source_sha256, config, limits, environment}. On drift raises
    ProvenanceError naming the drifted field(s)."""
    mismatches = [
        field_name
        for field_name in ("task", "source_sha256", "config", "limits",
                           "environment")
        if recorded.get(field_name) != current.get(field_name)
    ]
    if mismatches:
        raise ProvenanceError(
            "run provenance drift in " + ", ".join(mismatches) +
            "; start a new run_id instead of mixing experiments"
        )
    return recorded


class RunLock:
    """Non-blocking, single-writer flock over one run directory. A second
    holder (this process or another) fails fast rather than corrupting a run."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = open(self.path, "a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            self.stream.close()
            self.stream = None
            raise RunLockError(
                f"run {self.path.parent.name!r} is already owned by another "
                f"writer"
            ) from exc
        return self

    def release(self):
        if self.stream is None:
            return
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
        self.stream = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


# =============================================================================
# Append-only, fsync'd record store (eb.py port)
# =============================================================================
class RecordStore:
    """All-outcomes append-only JSONL store with per-write fsync — every
    proposed kernel's verdict is durably recorded, win or lose."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> dict:
        with open(self.path, "a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def records(self):
        if not self.path.exists():
            return []
        with open(self.path) as stream:
            return [json.loads(line) for line in stream if line.strip()]


# =============================================================================
# Evidence-gated stop controller (stopping.py port)
# =============================================================================
@dataclass(frozen=True)
class KernelStopPolicy:
    enabled: bool = False
    min_records: int = 3
    min_correct: int = 3
    stop_patience: int = 3          # correct evals since last strict improvement
    meaningful_delta: float = 1e-9

    def __post_init__(self):
        for name in ("min_records", "min_correct", "stop_patience"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.meaningful_delta < 0:
            raise ValueError("meaningful_delta must be >= 0")


@dataclass(frozen=True)
class StopRequest:
    action: str          # "stop" | "continue"
    reason: str = ""


@dataclass(frozen=True)
class StopReview:
    accepted: bool
    reasons: tuple
    evidence: dict

    def to_dict(self):
        return {"accepted": self.accepted, "reasons": list(self.reasons),
                "evidence": self.evidence}


def _record_well_formed(rec) -> bool:
    """A record must be a dict from which correctness is derivable, and if it
    claims correctness it must carry a numeric sol_score. Anything else is
    malformed and can never contribute to a stop."""
    if not isinstance(rec, dict):
        return False
    if "correctness" not in rec and "is_correct" not in rec:
        return False
    if is_correct(rec):
        score = _score_of(rec)
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            return False
    return True


def stopping_evidence(records, policy: KernelStopPolicy) -> dict:
    """Deterministic evidence derived SOLELY from evaluator records."""
    total = len(records)
    malformed = sum(0 if _record_well_formed(r) else 1 for r in records)
    correct_scores = [
        float(_score_of(r)) for r in records
        if _record_well_formed(r) and is_correct(r) and _score_of(r) is not None
    ]
    running_best = None
    since_improvement = 0
    for score in correct_scores:
        if running_best is None or score > running_best + policy.meaningful_delta:
            running_best = score
            since_improvement = 0
        else:
            since_improvement += 1
    return {
        "total_records": total,
        "malformed_records": malformed,
        "correct_records": len(correct_scores),
        "best_score": running_best,
        "evals_since_meaningful_improvement": since_improvement,
    }


class KernelStopController:
    """Treat an autonomous stop as a REQUEST gated by deterministic evidence.

    The controller never trusts the requester's self-assessment; every guard is
    recomputed from the evaluator RECORDS. Malformed or empty input yields at
    least one blocking reason, so it can never trigger a stop."""

    def __init__(self, policy: KernelStopPolicy):
        self.policy = policy

    def review(self, request: StopRequest, records) -> StopReview:
        evidence = stopping_evidence(records, self.policy)
        reasons = []
        if request.action != "stop":
            reasons.append("not_a_stop_request")
        if not self.policy.enabled:
            reasons.append("stops_disabled")
        if evidence["total_records"] == 0:
            reasons.append("no_records")
        if evidence["malformed_records"]:
            reasons.append("malformed_records")
        if evidence["total_records"] < self.policy.min_records:
            reasons.append("minimum_records_not_met")
        if evidence["correct_records"] < self.policy.min_correct:
            reasons.append("insufficient_correct_records")
        if evidence["evals_since_meaningful_improvement"] < self.policy.stop_patience:
            reasons.append("patience_not_met")
        accepted = request.action == "stop" and not reasons
        return StopReview(accepted, tuple(reasons), evidence)


__all__ = [
    "KernelTaskSpec", "KernelEvaluation",
    "SandboxUnavailable", "ArtifactRejected", "ProvenanceError", "RunLockError",
    "is_correct", "kernel_eval_is_correct", "rank_correct_first",
    "detect_sandbox_backend", "build_sandboxed_command", "rlimit_wrapped_command",
    "SANDBOX_BACKEND", "SANDBOX_TOOL", "SANDBOX_AVAILABLE",
    "snapshot_candidate_artifact", "trusted_artifact_dir",
    "trusted_evaluate", "evaluate_candidate",
    "build_run_manifest", "write_run_manifest", "load_run_manifest",
    "validate_run_manifest", "RunLock", "RecordStore",
    "KernelStopPolicy", "StopRequest", "StopReview", "KernelStopController",
    "stopping_evidence",
]
