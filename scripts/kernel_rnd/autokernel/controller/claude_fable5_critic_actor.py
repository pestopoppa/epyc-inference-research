#!/usr/bin/env python3
"""Launch one hash-bound Claude Fable 5 critic with no local capabilities.

The module is inert on import.  ``run_critic`` stages only the Claude OAuth
credential in an ephemeral private directory, supplies a scrubbed state file,
disables tools, MCP, setting sources, and session persistence, and accepts only
the exact structured critic result bound to the caller's content digests.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence


PROVIDER = "claude"
MODEL = "claude-fable-5"
EFFORT = "high"
RUNTIME_KIND = "claude_cli_structured_critic"
AUTH_STAGING_POLICY = "ephemeral_0600_copy_no_secret_receipt"
BINDING_KEYS = (
    "proposal_sha256",
    "source_manifest_sha256",
    "candidate_patch_sha256",
    "context_sha256",
    "template_catalog_sha256",
)
RESULT_KEYS = ("decision", "reason", *BINDING_KEYS)
DECISIONS = ("accept", "reject", "revise")
MAX_PROMPT_BYTES = 1024 * 1024
MAX_REASON_CHARS = 4000
MAX_CREDENTIAL_BYTES = 1024 * 1024
MAX_STDOUT_BYTES = 2 * 1024 * 1024
MAX_STDERR_BYTES = 1024 * 1024
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_SCRUBBED_STATE = b'{"hasCompletedOnboarding":true}\n'
_EMPTY_MCP = b'{"mcpServers":{}}\n'
_ARGV_POLICY = {
    "schema": "epyc.autokernel.claude_fable5_critic.argv.v1",
    "model": MODEL,
    "effort": EFFORT,
    "input": "stdin",
    "output": "json_schema_structured_output",
    "permission_mode": "plan",
    "safe_mode": True,
    "slash_commands": False,
    "tools": [],
    "disallowed_tools": ["Bash", "Edit", "Write", "NotebookEdit"],
    "mcp_servers": [],
    "setting_sources": [],
    "session_persistence": False,
    "process_group": "new_session_term_then_kill",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


ARGV_POLICY_SHA256 = _sha256_bytes(_canonical_bytes(_ARGV_POLICY))


class ClaudeFable5CriticError(RuntimeError):
    """The sealed Claude critic boundary or response was invalid."""


class ClaudeFable5CriticTimeout(ClaudeFable5CriticError):
    """The critic timed out and its captured process group was destroyed."""


@dataclass(frozen=True)
class CriticResult:
    decision: str
    reason: str
    proposal_sha256: str
    source_manifest_sha256: str
    candidate_patch_sha256: str
    context_sha256: str
    template_catalog_sha256: str
    wrapper_sha256: str
    argv_sha256: str
    stdout_sha256: str
    stderr_sha256: str

    def binding_map(self) -> dict[str, str]:
        return {key: str(getattr(self, key)) for key in BINDING_KEYS}


@dataclass(frozen=True)
class _Stage:
    root: Path
    config: Path
    runtime: Path
    mcp_config: Path
    wrapper: Path


def _read_regular(path: Path, *, executable: bool = False,
                  private: bool = False, maximum: int | None = None) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ClaudeFable5CriticError(f"unsafe or unavailable file: {path.name}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ClaudeFable5CriticError(f"file authority is not a single regular inode: {path.name}")
        if executable and before.st_mode & 0o111 == 0:
            raise ClaudeFable5CriticError(f"file is not executable: {path.name}")
        if private and (before.st_uid != os.getuid() or stat.S_IMODE(before.st_mode) != 0o600):
            raise ClaudeFable5CriticError(f"private file ownership or mode is unsafe: {path.name}")
        if maximum is not None and before.st_size > maximum:
            raise ClaudeFable5CriticError(f"file exceeds its bounded size: {path.name}")
        blocks: list[bytes] = []
        remaining = maximum
        while True:
            request = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining + 1)
            block = os.read(fd, request)
            if not block:
                break
            blocks.append(block)
            if remaining is not None:
                remaining -= len(block)
                if remaining < 0:
                    raise ClaudeFable5CriticError(f"file exceeds its bounded size: {path.name}")
        content = b"".join(blocks)
        after = os.fstat(fd)
        identity_before = (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        )
        if identity_before != identity_after or len(content) != after.st_size:
            raise ClaudeFable5CriticError(f"file changed while it was read: {path.name}")
        return content
    finally:
        os.close(fd)


def _wrapper_identity(wrapper: Path) -> tuple[Path, str]:
    if not wrapper.is_absolute() or wrapper.is_symlink():
        raise ClaudeFable5CriticError("Claude wrapper must be an absolute non-symlink path")
    content = _read_regular(wrapper, executable=True)
    return wrapper, _sha256_bytes(content)


def _wrapper_authority(wrapper: Path) -> tuple[Path, str, bytes]:
    if not wrapper.is_absolute() or wrapper.is_symlink():
        raise ClaudeFable5CriticError("Claude wrapper must be an absolute non-symlink path")
    content = _read_regular(wrapper, executable=True)
    return wrapper, _sha256_bytes(content), content


def runtime_identity(wrapper: Path) -> dict[str, object]:
    """Return the exact non-secret runtime identity expected by deployment."""
    path, digest = _wrapper_identity(wrapper)
    return {
        "kind": RUNTIME_KIND,
        "provider": PROVIDER,
        "model": MODEL,
        "effort": EFFORT,
        "wrapper_path": str(path),
        "wrapper_sha256": digest,
        "argv_policy_sha256": ARGV_POLICY_SHA256,
        "auth_staging_policy": AUTH_STAGING_POLICY,
    }


def _launcher_sha256() -> str:
    return _sha256_bytes(_read_regular(Path(__file__).resolve()))


def _validated_bindings(bindings: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(bindings, Mapping) or set(bindings) != set(BINDING_KEYS):
        raise ClaudeFable5CriticError(
            f"critic bindings must contain exactly {list(BINDING_KEYS)}")
    result: dict[str, str] = {}
    for key in BINDING_KEYS:
        value = bindings[key]
        if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
            raise ClaudeFable5CriticError(f"critic binding is not a canonical SHA-256: {key}")
        result[key] = value
    return result


def output_schema(bindings: Mapping[str, str]) -> dict[str, object]:
    """Build a schema whose digest fields are constants, not free-form strings."""
    exact = _validated_bindings(bindings)
    properties: dict[str, object] = {
        "decision": {"type": "string", "enum": list(DECISIONS)},
        "reason": {"type": "string", "minLength": 1, "maxLength": MAX_REASON_CHARS},
    }
    properties.update({key: {"const": exact[key]} for key in BINDING_KEYS})
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(RESULT_KEYS),
    }


def build_argv(*, wrapper: Path, config_dir: Path,
               bindings: Mapping[str, str]) -> tuple[str, ...]:
    """Return the exact pinned no-capability Claude invocation."""
    _wrapper_identity(wrapper)
    if (not config_dir.is_absolute() or config_dir.is_symlink()
            or not config_dir.is_dir()):
        raise ClaudeFable5CriticError("Claude config directory is unavailable or unsafe")
    mcp = config_dir / "empty-mcp.json"
    if _read_regular(mcp, private=True, maximum=4096) != _EMPTY_MCP:
        raise ClaudeFable5CriticError("Claude empty MCP authority changed")
    schema = _canonical_bytes(output_schema(bindings)).decode("ascii")
    return (
        str(wrapper),
        "--print",
        "--output-format", "json",
        "--json-schema", schema,
        "--model", MODEL,
        "--effort", EFFORT,
        "--permission-mode", "plan",
        "--safe-mode",
        "--disable-slash-commands",
        "--tools", "",
        "--disallowedTools", "Bash", "Edit", "Write", "NotebookEdit",
        "--strict-mcp-config",
        "--mcp-config", str(mcp),
        "--setting-sources", "",
        "--no-session-persistence",
    )


def _write_private(path: Path, content: bytes, *, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    fd = os.open(path, flags, mode)
    try:
        offset = 0
        while offset < len(content):
            offset += os.write(fd, content[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    if stat.S_IMODE(path.stat().st_mode) != mode:
        raise ClaudeFable5CriticError(f"could not seal private staged file: {path.name}")


def _credentials(auth_root: Path) -> bytes:
    if not auth_root.is_absolute() or auth_root.is_symlink() or not auth_root.is_dir():
        raise ClaudeFable5CriticError("Claude auth root is unavailable or unsafe")
    content = _read_regular(
        auth_root / ".credentials.json", private=True,
        maximum=MAX_CREDENTIAL_BYTES,
    )
    try:
        parsed = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClaudeFable5CriticError("Claude credential carrier is not valid JSON") from exc
    if (not isinstance(parsed, dict) or set(parsed) != {"claudeAiOauth"}
            or not isinstance(parsed["claudeAiOauth"], dict)
            or not parsed["claudeAiOauth"]):
        raise ClaudeFable5CriticError("Claude credential carrier has an unexpected shape")
    return content


def _safe_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ClaudeFable5CriticError(f"{label} must be an existing absolute directory")
    return path


@contextmanager
def _staged_auth(*, workspace: Path, auth_root: Path,
                 wrapper_content: bytes) -> Iterator[_Stage]:
    workspace = _safe_directory(workspace, label="critic workspace")
    credential = _credentials(auth_root)
    root = Path(tempfile.mkdtemp(prefix=".autokernel-fable5-", dir=workspace))
    root.chmod(0o700)
    try:
        config = root / "config"
        runtime = root / "runtime"
        config.mkdir(mode=0o700)
        runtime.mkdir(mode=0o700)
        _write_private(config / ".credentials.json", credential)
        _write_private(config / ".claude.json", _SCRUBBED_STATE)
        mcp = config / "empty-mcp.json"
        _write_private(mcp, _EMPTY_MCP)
        staged_wrapper = root / "claude"
        _write_private(staged_wrapper, wrapper_content, mode=0o500)
        yield _Stage(
            root=root, config=config, runtime=runtime, mcp_config=mcp,
            wrapper=staged_wrapper,
        )
    finally:
        if root.is_symlink():
            raise ClaudeFable5CriticError("Claude staging root became a symlink")
        shutil.rmtree(root)
        if root.exists() or root.is_symlink():
            raise ClaudeFable5CriticError("Claude staged credentials survived cleanup")


_ENV_ALLOWLIST = frozenset({
    "PATH", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "LANG", "LC_ALL",
})


def _scrubbed_environment(environment: Mapping[str, str], stage: _Stage) -> dict[str, str]:
    clean: dict[str, str] = {}
    for key in _ENV_ALLOWLIST:
        value = environment.get(key)
        if isinstance(value, str) and value and "\x00" not in value:
            clean[key] = value
    clean.update({
        "HOME": str(stage.runtime),
        "CLAUDE_CONFIG_DIR": str(stage.config),
        "TMPDIR": str(stage.runtime),
        "TMP": str(stage.runtime),
        "TEMP": str(stage.runtime),
        "XDG_CONFIG_HOME": str(stage.runtime),
        "XDG_CACHE_HOME": str(stage.runtime),
        "XDG_STATE_HOME": str(stage.runtime),
        "NO_COLOR": "1",
        "CLAUDE_CODE_DISABLE_AGENT_VIEW": "1",
        "CLAUDE_CODE_DISABLE_WORKFLOWS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    })
    return clean


def _process_group_members(group_id: int) -> tuple[int, ...]:
    members: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        raise ClaudeFable5CriticError("/proc is required for process-group death proof")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text(encoding="ascii")
            tail = text[text.rindex(")") + 2:].split()
            state = tail[0]
            process_group = int(tail[2])
        except (OSError, ValueError, IndexError):
            continue
        # A zombie has no executable process or open resources and cannot
        # survive a signal; its parent/init owns the remaining reaping record.
        if process_group == group_id and state != "Z":
            members.append(int(entry.name))
    return tuple(sorted(members))


def _signal_group(group_id: int, sig: signal.Signals) -> None:
    try:
        os.killpg(group_id, sig)
    except ProcessLookupError:
        return
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise


def _await_group_death(group_id: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        if not _process_group_members(group_id):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


def _destroy_group(process: subprocess.Popen[str], grace_seconds: float) -> None:
    group_id = process.pid
    if _process_group_members(group_id):
        _signal_group(group_id, signal.SIGTERM)
        if not _await_group_death(group_id, grace_seconds):
            _signal_group(group_id, signal.SIGKILL)
            if not _await_group_death(group_id, max(1.0, grace_seconds)):
                raise ClaudeFable5CriticError(
                    "captured Claude process group survived SIGTERM and SIGKILL")
    try:
        process.wait(timeout=max(1.0, grace_seconds))
    except subprocess.TimeoutExpired as exc:
        raise ClaudeFable5CriticError("captured Claude process could not be reaped") from exc


def _close_process_pipes(process: subprocess.Popen[str]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            stream.close()


def _limit_output_files() -> None:
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (max(MAX_STDOUT_BYTES, MAX_STDERR_BYTES),
         max(MAX_STDOUT_BYTES, MAX_STDERR_BYTES)),
    )


def _capture_text(handle: Any, maximum: int) -> str:
    handle.flush()
    handle.seek(0)
    content = handle.read(maximum + 1)
    if len(content) > maximum:
        raise ClaudeFable5CriticError("Claude output exceeded its bounded capture")
    return content.decode("utf-8", errors="replace")


def _run_process(*, argv: Sequence[str], cwd: Path, environment: Mapping[str, str],
                 prompt: str, timeout_seconds: float,
                 terminate_grace_seconds: float,
                 capture_root: Path,
                 expected_launcher_sha256: str | None = None) -> tuple[int, str, str]:
    with tempfile.TemporaryFile(mode="w+b", dir=capture_root) as stdout_file, \
            tempfile.TemporaryFile(mode="w+b", dir=capture_root) as stderr_file:
        if (expected_launcher_sha256 is not None
                and _launcher_sha256() != expected_launcher_sha256):
            raise ClaudeFable5CriticError(
                "Claude launcher bytes changed at the process spawn boundary")
        try:
            process = subprocess.Popen(
                tuple(argv), cwd=cwd, env=dict(environment), stdin=subprocess.PIPE,
                stdout=stdout_file, stderr=stderr_file,
                start_new_session=True, preexec_fn=_limit_output_files,
            )
        except OSError as exc:
            raise ClaudeFable5CriticError("could not start sealed Claude wrapper") from exc
        timed_out = False
        primary: BaseException | None = None
        try:
            try:
                process.communicate(
                    input=prompt.encode("utf-8"), timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
        except BaseException as exc:
            primary = exc
        cleanup_error: BaseException | None = None
        try:
            _destroy_group(process, terminate_grace_seconds)
        except BaseException as exc:
            cleanup_error = exc
        finally:
            _close_process_pipes(process)
        if cleanup_error is not None:
            if primary is not None:
                cleanup_error.add_note(
                    f"Claude process also raised before teardown: {primary!r}")
                raise cleanup_error from primary
            raise cleanup_error
        if primary is not None:
            raise primary
        stdout = _capture_text(stdout_file, MAX_STDOUT_BYTES)
        stderr = _capture_text(stderr_file, MAX_STDERR_BYTES)
        if timed_out:
            raise ClaudeFable5CriticTimeout(
                "Claude Fable 5 critic exceeded its bounded timeout; process group destroyed")
        if process.returncode is None:
            raise ClaudeFable5CriticError("Claude wrapper has no terminal return code")
        return process.returncode, stdout, stderr


def _parse_result(stdout: str, bindings: Mapping[str, str]) -> dict[str, str]:
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ClaudeFable5CriticError("Claude stdout is not one strict JSON document") from exc
    if not isinstance(envelope, dict) or envelope.get("is_error") is True:
        raise ClaudeFable5CriticError("Claude stdout is not a successful result envelope")
    payload = envelope.get("structured_output")
    if not isinstance(payload, dict) or set(payload) != set(RESULT_KEYS):
        raise ClaudeFable5CriticError("Claude structured output has the wrong exact fields")
    decision = payload.get("decision")
    reason = payload.get("reason")
    if decision not in DECISIONS:
        raise ClaudeFable5CriticError("Claude critic decision is invalid")
    if (not isinstance(reason, str) or not reason.strip()
            or len(reason) > MAX_REASON_CHARS):
        raise ClaudeFable5CriticError("Claude critic reason is empty or unbounded")
    exact = _validated_bindings(bindings)
    for key in BINDING_KEYS:
        if payload.get(key) != exact[key]:
            raise ClaudeFable5CriticError(f"Claude critic did not echo exact binding: {key}")
    return {key: str(payload[key]) for key in RESULT_KEYS}


def run_critic(
    *, wrapper: Path, workspace: Path, prompt: str,
    bindings: Mapping[str, str], environment: Mapping[str, str],
    auth_root: Path | None = None,
    expected_wrapper_sha256: str | None = None,
    expected_runtime_identity: Mapping[str, Any] | None = None,
    expected_launcher_sha256: str | None = None,
    timeout_seconds: float = 900.0,
    terminate_grace_seconds: float = 2.0,
) -> CriticResult:
    """Run exactly one Fable5/high critique and verify every returned binding."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ClaudeFable5CriticError("critic prompt must be non-empty")
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ClaudeFable5CriticError("critic prompt exceeds its bounded size")
    if (not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ClaudeFable5CriticError("critic timeout must be finite and positive")
    if (not isinstance(terminate_grace_seconds, (int, float))
            or isinstance(terminate_grace_seconds, bool)
            or not math.isfinite(terminate_grace_seconds)
            or terminate_grace_seconds <= 0):
        raise ClaudeFable5CriticError("critic terminate grace must be finite and positive")
    exact_bindings = _validated_bindings(bindings)
    workspace = _safe_directory(workspace, label="critic workspace")
    wrapper_path, wrapper_sha256, wrapper_content = _wrapper_authority(wrapper)
    current_runtime = {
        "kind": RUNTIME_KIND,
        "provider": PROVIDER,
        "model": MODEL,
        "effort": EFFORT,
        "wrapper_path": str(wrapper_path),
        "wrapper_sha256": wrapper_sha256,
        "argv_policy_sha256": ARGV_POLICY_SHA256,
        "auth_staging_policy": AUTH_STAGING_POLICY,
    }
    if (expected_wrapper_sha256 is not None
            and current_runtime["wrapper_sha256"] != expected_wrapper_sha256):
        raise ClaudeFable5CriticError("Claude wrapper bytes changed after deployment validation")
    if (expected_runtime_identity is not None
            and current_runtime != dict(expected_runtime_identity)):
        raise ClaudeFable5CriticError("Claude runtime identity changed before spawn")
    launcher_sha256 = _launcher_sha256()
    if (expected_launcher_sha256 is not None
            and launcher_sha256 != expected_launcher_sha256):
        raise ClaudeFable5CriticError("Claude launcher/argv policy bytes changed before spawn")
    if auth_root is None:
        home = environment.get("HOME")
        if not isinstance(home, str) or not Path(home).is_absolute():
            raise ClaudeFable5CriticError("absolute HOME is required to locate Claude auth")
        auth_root = Path(home) / ".claude"
    with _staged_auth(
            workspace=workspace, auth_root=auth_root,
            wrapper_content=wrapper_content) as stage:
        argv = build_argv(
            wrapper=stage.wrapper, config_dir=stage.config, bindings=exact_bindings)
        # Reopen every public byte authority immediately before process creation.
        if runtime_identity(wrapper) != current_runtime:
            raise ClaudeFable5CriticError("Claude runtime identity changed during staging")
        child_environment = _scrubbed_environment(environment, stage)
        if _launcher_sha256() != launcher_sha256:
            raise ClaudeFable5CriticError("Claude launcher bytes changed during staging")
        returncode, stdout, stderr = _run_process(
            argv=argv, cwd=workspace,
            environment=child_environment, prompt=prompt,
            timeout_seconds=float(timeout_seconds),
            terminate_grace_seconds=float(terminate_grace_seconds),
            capture_root=stage.runtime,
            expected_launcher_sha256=launcher_sha256,
        )
        if runtime_identity(wrapper) != current_runtime:
            raise ClaudeFable5CriticError("Claude runtime identity changed during execution")
        if _launcher_sha256() != launcher_sha256:
            raise ClaudeFable5CriticError("Claude launcher bytes changed during execution")
        if returncode != 0:
            tail = stderr[-400:].replace("\n", " ")
            raise ClaudeFable5CriticError(
                f"Claude Fable 5 critic failed with status {returncode}: {tail}")
        payload = _parse_result(stdout, exact_bindings)
        return CriticResult(
            **payload,
            wrapper_sha256=str(current_runtime["wrapper_sha256"]),
            argv_sha256=_sha256_bytes(_canonical_bytes(list(argv))),
            stdout_sha256=_sha256_bytes(stdout.encode("utf-8")),
            stderr_sha256=_sha256_bytes(stderr.encode("utf-8")),
        )
