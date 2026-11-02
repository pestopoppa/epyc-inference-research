"""Sandboxed executable-scoring for code-generation suites (LiveCodeBench/BigCodeBench).

The standardized code-correctness scorer: extract the model's code, run it against
the problem's test cases in an isolated subprocess, return pass/fail. This is Track 2
of handoffs/active/scoring-infra-standardization.md — the missing piece that lets a
code-gen suite measure real capability instead of the adapter's placeholder
`substring "def "` check.

Isolation (scaffold level): fresh temp cwd, RLIMIT_CPU / RLIMIT_AS (memory) /
RLIMIT_CORE=0, wall-clock timeout, minimal env, and a dedicated process group so a
timeout kills the child's whole descendant tree rather than just the child.

⚠ NOT bounded: process/thread COUNT. RLIMIT_NPROC is deliberately NOT set. It is
enforced per real UID, not per process tree, and this host runs ~9.5k threads under
the same uid (llama-server alone accounts for most of them). Any per-scorer cap
would therefore fail the child's *first* fork under normal fleet load, and would
fail NONDETERMINISTICALLY as that load varies — turning a scorer into an instrument
whose results track how busy the box is. That is a worse failure than the one it
would fix. The correct mechanism is cgroup v2 `pids.max`, which counts only the
scorer's own subtree and is unaffected by co-tenants; verified available on this
host (needs `+pids` in cgroup.subtree_control). Tracked in
handoffs/active/scoring-infra-standardization.md. Corrected 2026-07-29 — this
docstring previously claimed RLIMIT_NPROC was set; it never was.

The default remains the trusted-code scaffold. An explicit bubblewrap mode adds a
separate user, PID and network namespace, a restricted read-only system runtime,
and a private temporary filesystem. Optional Docker mode uses an explicit,
trusted local image ID and a no-network, read-only, non-root container with a
process count limit and an explicit subset of the caller's CPU affinity; the
caller must separately hold the relevant region claim. It changes the interpreter
environment. Candidate and test bytes are encoded in container argv and visible
to the host Docker daemon. Both fail closed if setup is unavailable. Neither
hides tests assembled into the candidate program or establishes a complete
hostile-code security boundary.
"""
from __future__ import annotations

import os
import base64
import json
import re
import resource
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


class SandboxUnavailable(RuntimeError):
    """Requested isolation could not be established; never score as a model failure."""


def _parse_cpuset(value: str | None) -> set[int]:
    cpus: set[int] = set()
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise SandboxUnavailable("a nonempty explicit Docker CPU set is required")
    for component in value.split(","):
        if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?", component):
            raise SandboxUnavailable("invalid Docker CPU set")
        parts = [int(x) for x in component.split("-")]
        lo, hi = parts[0], parts[-1]
        if lo > hi or hi > 65535:
            raise SandboxUnavailable("invalid Docker CPU range")
        cpus.update(range(lo, hi + 1))
    return cpus


def _run_docker(code: str, stdin: str, timeout: int, cpu_s: int, mem_mb: int,
                docker_image: str | None, docker_cpuset: str | None,
                broker_socket: str | None = None) -> tuple[bool, str]:
    """Run in an isolated, caller-pinned local Python image; never mount host paths."""
    if not docker_image or not re.fullmatch(r"sha256:[0-9a-f]{64}", docker_image):
        raise SandboxUnavailable("docker mode requires an immutable local image ID")
    cpus = _parse_cpuset(docker_cpuset)
    if not cpus.issubset(os.sched_getaffinity(0)):
        raise SandboxUnavailable("Docker CPU set exceeds caller process affinity")
    mount_args = []
    if broker_socket is not None:
        sock = Path(broker_socket)
        if (not sock.is_absolute() or ',' in broker_socket or not sock.exists()
                or str(sock.resolve()) != broker_socket
                or not stat.S_ISSOCK(sock.stat().st_mode) or sock.stat().st_uid != os.getuid()):
            raise SandboxUnavailable("broker must be an existing absolute Unix socket")
        mount_args = ["--mount", f"type=bind,src={broker_socket},dst=/broker/socket,readonly"]
    name = f"codeexec-{secrets.token_hex(16)}"
    marker = f"__docker_ready_{secrets.token_hex(16)}__"
    payload = base64.b64encode(code.encode()).decode()
    bootstrap = ("import base64,resource,sys; "
                 "resource.setrlimit(resource.RLIMIT_CPU,(int(sys.argv[3]),int(sys.argv[3]))); "
                 "resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
                 "sys.stdout.write(sys.argv[1]); sys.stdout.flush(); "
                 "exec(compile(base64.b64decode(sys.argv[2]),'<candidate>','exec'))")
    def docker(*args: str, timeout_s: int = 15) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(["docker", *args], capture_output=True, text=True,
                                  timeout=timeout_s, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable(f"docker command failed: {exc}") from exc
    info = docker("image", "inspect", docker_image)
    try:
        image = json.loads(info.stdout)[0]
        if info.returncode or image["Id"] != docker_image or image["Config"].get("Volumes"):
            raise ValueError("image unavailable, ID mismatch, or declared volumes")
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise SandboxUnavailable(f"docker image preflight failed: {exc}") from exc
    cid = None
    try:
        created = docker("create", "--pull=never", "--name", name,
                         "--network=none", "--read-only", "--cap-drop=ALL",
                         "--security-opt=no-new-privileges", "--pids-limit=64",
                         "--memory", f"{mem_mb}m", "--cpus=1", f"--cpuset-cpus={docker_cpuset}", "--user=65534:65534",
                         "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=64m", "--workdir=/tmp",
                         "--env=HOME=/tmp", "--env=PYTHONDONTWRITEBYTECODE=1",
                         "--env=OPENBLAS_NUM_THREADS=1", "--env=OMP_NUM_THREADS=1",
                         *mount_args,
                         "--entrypoint=python3", "-i", docker_image, "-c", bootstrap, marker, payload, str(cpu_s))
        cid = created.stdout.strip()
        if created.returncode or not re.fullmatch(r"[0-9a-f]{64}", cid):
            raise SandboxUnavailable(f"docker create failed: {created.stderr[-200:]}")
        inspected = docker("inspect", cid)
        try:
            item = json.loads(inspected.stdout)[0]
            host = item["HostConfig"]
            mounts = item["Mounts"]
            mount_valid = (not mounts if broker_socket is None else
                           len(mounts) == 1 and mounts[0].get("Type") == "bind"
                           and mounts[0].get("Source") == broker_socket
                           and mounts[0].get("Destination") == "/broker/socket"
                           and mounts[0].get("RW") is False)
            valid = (inspected.returncode == 0 and item["Id"] == cid
                     and host["NetworkMode"] == "none" and host["ReadonlyRootfs"] is True
                     and host["Privileged"] is False and mount_valid
                     and host["PidMode"] in ("", "private") and host["IpcMode"] != "host"
                     and item["Config"]["User"] == "65534:65534"
                     and "ALL" in host["CapDrop"] and host["PidsLimit"] == 64
                     and "no-new-privileges" in host["SecurityOpt"]
                     and host["Memory"] == mem_mb * 1024 * 1024
                     and host["NanoCpus"] == 1000000000
                     and _parse_cpuset(host["CpusetCpus"]) == cpus
                     and host["Tmpfs"].get("/tmp") == "rw,nosuid,nodev,noexec,size=64m")
            if not valid:
                raise ValueError("container does not match isolation contract")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise SandboxUnavailable(f"docker isolation preflight failed: {exc}") from exc
        try:
            run = subprocess.run(["docker", "start", "-ai", cid], input=stdin,
                                 capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            observed = exc.stdout or b""
            if not observed.startswith(marker.encode() if isinstance(observed, bytes) else marker):
                raise SandboxUnavailable("docker setup timed out") from exc
            return False, "__timeout__"
        except OSError as exc:
            raise SandboxUnavailable(f"docker start failed: {exc}") from exc
        if not run.stdout.startswith(marker):
            raise SandboxUnavailable(f"docker setup failed: {run.stderr[-200:]}")
        out = run.stdout[len(marker):]
        return ((False, f"__exit_{run.returncode}__:{run.stderr[-200:]}")
                if run.returncode else (True, out))
    finally:
        # Name is unique and ours, including when the create response was lost.
        target = cid if cid and re.fullmatch(r"[0-9a-f]{64}", cid) else name
        removed = docker("rm", "-f", target)
        remaining = docker("inspect", target)
        if remaining.returncode == 0 or "no such" not in remaining.stderr.lower():
            raise SandboxUnavailable(f"docker cleanup could not be verified: {removed.stderr[-200:]}")


def _bubblewrap_command(exe: str, src: Path, bwrap_exe: str, marker: str) -> list[str]:
    """Build an explicit mount allowlist; never expose the host root or workspace."""
    binary = Path(bwrap_exe)
    runtime = Path(exe)
    if not binary.is_absolute() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise SandboxUnavailable("bubblewrap executable must be an absolute executable file")
    # A venv outside /usr would need additional mounts and can expose arbitrary
    # host files. Fail explicitly until a separately reviewed runtime is supplied.
    if not runtime.is_absolute() or not runtime.is_relative_to('/usr') or not runtime.resolve().is_relative_to('/usr'):
        raise SandboxUnavailable("bubblewrap mode currently supports interpreters under /usr only")
    return [str(binary), '--unshare-user', '--unshare-pid', '--unshare-net',
            '--unshare-ipc', '--unshare-uts', '--die-with-parent',
            '--ro-bind', '/usr', '/usr',
            '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib64', '/lib64',
            '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
            '--dir', '/work', '--ro-bind', str(src), '/work/sol.py',
            '--chdir', '/work', '--', exe, '-c',
            'import os,sys; os.write(1,sys.argv[1].encode()); '
            'os.execv(sys.executable,[sys.executable,sys.argv[2]])',
            marker, '/work/sol.py']


def extract_code(response: str, language: str = "python") -> str:
    """Pull the model's code: last fenced block for the language, else last fenced
    block, else the raw text (a bare code answer)."""
    if not response:
        return ""
    fences = re.findall(r"```(?:\s*(\w+))?\s*\n(.*?)```", response, re.DOTALL)
    if fences:
        typed = [body for lang, body in fences if lang and lang.lower() in
                 (language, "py" if language == "python" else language)]
        return (typed[-1] if typed else fences[-1][1]).strip()
    return response.strip()


def _limits(cpu_s: int, mem_mb: int):
    def _apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        soft = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return _apply


def _kill_group(proc: subprocess.Popen | None) -> None:
    """SIGKILL the child's entire process group.

    subprocess's own timeout path kills ONLY the direct child, so anything the
    scored code spawned survives the timeout as an orphan and keeps consuming
    cores reserved for inference. `start_new_session=True` puts the child in its
    own process group precisely so the group can be killed as a unit here.
    """
    if proc is None or proc.pid is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass  # already reaped, or the group is gone
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001 — best-effort reap; never mask the real error
        pass


def _run_once(code: str, stdin: str, timeout: int, cpu_s: int, mem_mb: int,
              python_exe: str | None = None, *, isolation: str = "scaffold",
              bwrap_exe: str = "/usr/bin/bwrap", docker_image: str | None = None,
              docker_cpuset: str | None = None, broker_socket: str | None = None) -> tuple[bool, str]:
    """Run code with stdin, return (ok, stdout-or-error)."""
    # absolutize BEFORE the child runs with cwd=tempdir — a relative interpreter
    # path would otherwise resolve inside the tempdir and fail to spawn.
    # absolute(), NOT resolve(): resolve() dereferences the venv's bin/python
    # symlink to the BASE interpreter, silently dropping the venv site-packages.
    exe = str(Path(python_exe).absolute()) if python_exe else sys.executable
    if isolation == "docker":
        if python_exe is not None:
            raise SandboxUnavailable("docker uses its pinned image interpreter, not python_exe")
        return _run_docker(code, stdin, timeout, cpu_s, mem_mb, docker_image, docker_cpuset, broker_socket)
    if isolation not in ("scaffold", "bubblewrap"):
        raise ValueError(f"unsupported isolation mode: {isolation}")
    with tempfile.TemporaryDirectory(prefix="codeexec_") as d:
        src = Path(d) / "sol.py"
        src.write_text(code)
        proc: subprocess.Popen | None = None
        marker = f"__bubblewrap_ready_{secrets.token_hex(16)}__" if isolation == "bubblewrap" else None
        try:
            cmd = (_bubblewrap_command(exe, src, bwrap_exe, marker)
                   if marker is not None else [exe, str(src)])
            # Popen (not run) so the pid is available for a GROUP kill on timeout;
            # start_new_session=True gives the child its own process group so the
            # kill reaches descendants the scored code spawned.
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
                cwd=d, env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
                            "MPLBACKEND": "Agg", "HOME": "/tmp" if marker else d,
                            # single-threaded math libs: OpenBLAS sizes buffers by
                            # nproc (192 here) and blows RLIMIT_AS otherwise
                            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"},
                preexec_fn=_limits(cpu_s, mem_mb),
                start_new_session=True,
            )
            out, err = proc.communicate(input=stdin, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            _kill_group(proc)
            # drain the pipes the killed group left behind, else the fds leak
            try:
                proc.communicate(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            if marker is not None:
                observed = e.output or b""
                if not observed.startswith(marker.encode() if isinstance(observed, bytes) else marker):
                    raise SandboxUnavailable("bubblewrap setup timed out")
            return False, "__timeout__"
        except SandboxUnavailable:
            raise
        except Exception as e:  # noqa: BLE001
            _kill_group(proc)
            if marker is not None:
                raise SandboxUnavailable(f"bubblewrap spawn failed: {e}") from e
            return False, f"__spawn_error__:{e}"
        if marker is not None:
            if not out.startswith(marker):
                raise SandboxUnavailable(f"bubblewrap setup failed: {err[-200:]}")
            out = out[len(marker):]
        if proc.returncode != 0:
            return False, f"__exit_{proc.returncode}__:{err[-200:]}"
        return True, out


def score_code(
    response: str,
    test_cases: list[dict[str, Any]],
    language: str = "python",
    timeout: int = 10,
    cpu_s: int = 8,
    mem_mb: int = 1024,
    *, isolation: str = "scaffold", bwrap_exe: str = "/usr/bin/bwrap", docker_image: str | None = None,
    docker_cpuset: str | None = None,
) -> dict[str, Any]:
    """Score generated code against test_cases.

    Each test case is either stdin/stdout: {"input": str, "output": str}, or a
    self-checking snippet: {"assert": "<python that raises on failure>"}.
    Returns {passed, total, pass_rate, correct(bool: all passed), detail[]}.
    """
    code = extract_code(response, language)
    if not code or language != "python":
        return {"passed": 0, "total": len(test_cases), "pass_rate": 0.0,
                "correct": False, "detail": ["__no_code__" if not code else "__unsupported_lang__"]}
    passed, detail = 0, []
    for tc in test_cases:
        if "assert" in tc:
            ok, out = _run_once(code + "\n\n" + tc["assert"], "", timeout, cpu_s, mem_mb, isolation=isolation, bwrap_exe=bwrap_exe, docker_image=docker_image, docker_cpuset=docker_cpuset)
        else:
            ok, out = _run_once(code, tc.get("input", ""), timeout, cpu_s, mem_mb, isolation=isolation, bwrap_exe=bwrap_exe, docker_image=docker_image, docker_cpuset=docker_cpuset)
            if ok:
                ok = out.strip() == str(tc.get("output", "")).strip()
        passed += int(ok)
        detail.append("pass" if ok else (out[:60] if out.startswith("__") else "wrong_output"))
    n = len(test_cases)
    return {"passed": passed, "total": n, "pass_rate": passed / n if n else 0.0,
            "correct": n > 0 and passed == n, "detail": detail}


def score_functional(
    response: str,
    test_code: str,
    entry_point: str,
    prompt: str = "",
    timeout: int = 10,
    cpu_s: int = 8,
    mem_mb: int = 1024,
    *, isolation: str = "scaffold", bwrap_exe: str = "/usr/bin/bwrap", docker_image: str | None = None,
    docker_cpuset: str | None = None,
) -> bool:
    """HumanEval/MBPP-style: the model completes a function; `test_code` is a
    `check(candidate)` body of asserts. Pass iff the assembled program runs clean.

    Robust to instruct models that either reproduce the full function or emit only
    the body: if the extracted code does not define `entry_point`, we prepend the
    prompt (signature+docstring) so the completion attaches to it.
    """
    code = extract_code(response, "python")
    if not code:
        return False
    if not re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", code):
        code = (prompt or "") + "\n" + code
    program = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"
    ok, _ = _run_once(program, "", timeout, cpu_s, mem_mb, isolation=isolation, bwrap_exe=bwrap_exe, docker_image=docker_image, docker_cpuset=docker_cpuset)
    return ok


def score_unittest(
    response: str,
    test_code: str,
    entry_point: str,
    code_prompt: str = "",
    python_exe: str | None = None,
    timeout: int = 30,
    cpu_s: int = 25,
    mem_mb: int = 4096,
    *, isolation: str = "scaffold", bwrap_exe: str = "/usr/bin/bwrap", docker_image: str | None = None,
    docker_cpuset: str | None = None,
) -> bool:
    """BigCodeBench-style: the model implements `entry_point` (task_func); the
    suite ships a unittest.TestCase in `test_code`. Pass iff the whole unittest
    run exits 0. Requires the dep-rich interpreter via `python_exe`
    (matplotlib/pandas/sklearn tasks won't import under the default venv);
    MPLBACKEND=Agg + HOME=tmpdir are set by the runner env."""
    code = extract_code(response, "python")
    if not code:
        return False
    if not re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", code):
        code = (code_prompt or "") + "\n" + code
    program = (f"{code}\n\n{test_code}\n\n"
               "import unittest as _ut\n_ut.main(argv=['x'], exit=True)\n")
    ok, _ = _run_once(program, "", timeout, cpu_s, mem_mb, python_exe=python_exe, isolation=isolation, bwrap_exe=bwrap_exe, docker_image=docker_image, docker_cpuset=docker_cpuset)
    return ok


if __name__ == "__main__":
    # smoke test on TRUSTED code (no model involved) — proves the harness scores
    # correct vs wrong solutions and enforces timeout.
    good = "```python\nn=int(input())\nprint(n*n)\n```"
    bad = "```python\nn=int(input())\nprint(n+1)\n```"
    loop = "```python\nwhile True:\n    pass\n```"
    tests = [{"input": "3", "output": "9"}, {"input": "5", "output": "25"}]
    r_good = score_code(good, tests)
    r_bad = score_code(bad, tests)
    r_loop = score_code(loop, [{"input": "3", "output": "9"}], timeout=3, cpu_s=2)
    assert r_good["correct"] and r_good["passed"] == 2, r_good
    assert not r_bad["correct"] and r_bad["passed"] == 0, r_bad
    # runaway caught either by wall-timeout or the CPU rlimit (SIGKILL exit -9)
    assert not r_loop["correct"] and r_loop["detail"][0].startswith("__"), r_loop
    # functional/assert-style
    fn = "```python\ndef sq(x):\n    return x*x\n```"
    r_fn = score_code(fn, [{"assert": "assert sq(4)==16"}, {"assert": "assert sq(0)==0"}])
    assert r_fn["correct"], r_fn
    # HumanEval/MBPP functional path: model completes a function, check() asserts
    hprompt = "def add(a, b):\n    "
    hgood = "```python\ndef add(a, b):\n    return a + b\n```"
    hbad = "```python\ndef add(a, b):\n    return a - b\n```"
    htest = "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(0, 0) == 0"
    assert score_functional(hgood, htest, "add", hprompt)
    assert not score_functional(hbad, htest, "add", hprompt)
    print("PASS code_exec_scorer smoke:",
          {"good": r_good, "bad": r_bad["passed"], "loop": r_loop["detail"][0], "fn": r_fn["correct"]})
