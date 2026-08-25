#!/usr/bin/env python3
"""Launch one hash-bound DeepSeek V4 Flash critic through the opencode CLI.

Backup critic for the Claude Fable 5 critic: inert on import.  ``run_critic``
stages only the opencode auth and configuration into an ephemeral private
directory, supplies a scrubbed environment, disables tools and session
persistence, runs ``opencode run`` with the exact model id, parses the JSON
event stream, and accepts only the exact structured critique bound to the
caller's content digests.

The prompt travels as the positional message (``opencode run [message..]``)
and is bounded well below the kernel per-argument limit so the exec always
succeeds.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence


PROVIDER = "opencode"
MODEL = "deepseek/deepseek-v4-flash"
RUNTIME_KIND = "opencode_cli_structured_critic"
AUTH_STAGING_POLICY = "ephemeral_0600_copy_atomic_opencode_no_secret_receipt"
BINDING_KEYS = (
    "proposal_sha256",
    "source_manifest_sha256",
    "candidate_patch_sha256",
    "context_sha256",
    "template_catalog_sha256",
)
RESULT_KEYS = ("decision", "reason", *BINDING_KEYS)
DECISIONS = ("accept", "reject", "revise")
# The prompt is a positional argv element; stay well under the 128 KiB
# per-argument exec limit (MAX_ARG_STRLEN) with headroom for the flags.
MAX_PROMPT_BYTES = 100 * 1024
MAX_REASON_CHARS = 4000
MAX_AUTH_BYTES = 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
MAX_STDOUT_BYTES = 4 * 1024 * 1024
MAX_STDERR_BYTES = 1024 * 1024
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_OPENCODE_AUTH_RELATIVE = Path(".local/share/opencode/auth.json")
_OPENCODE_CONFIG_RELATIVE = Path(".config/opencode/opencode.jsonc")
# A --pure opencode run without external plugins; no tools; JSON event stream.
_ARGV_POLICY = {
    "schema": "epyc.autokernel.opencode_deepseek_critic.argv.v1",
    "model": MODEL,
    "input": "positional_message",
    "output": "json_event_stream",
    "pure": True,
    "tools": [],
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


class OpenCodeCriticError(RuntimeError):
    """The sealed opencode critic boundary or response was invalid."""


class OpenCodeCriticTimeout(OpenCodeCriticError):
    """The opencode critic timed out and its captured process group died."""


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


def _read_regular(path: Path, *, executable: bool = False,
                  private: bool = False, maximum: int | None = None) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise OpenCodeCriticError(f"unsafe or unavailable file: {path.name}") from exc
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_uid != os.geteuid()
                or before.st_mode & 0o022
                or (private and stat.S_IMODE(before.st_mode) != 0o600)
                or (executable and not before.st_mode & 0o111)
                or (maximum is not None and before.st_size > maximum)):
            raise OpenCodeCriticError(f"unsafe identity: {path.name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        pathname = os.lstat(path)
    except OSError as exc:
        raise OpenCodeCriticError("path disappeared during read") from exc
    def identity(row: os.stat_result) -> tuple[int, ...]:
        return (row.st_dev, row.st_ino, row.st_uid, stat.S_IFMT(row.st_mode),
                row.st_nlink, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(pathname):
        raise OpenCodeCriticError("file changed during stable read")
    return b"".join(chunks)


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
        raise OpenCodeCriticError(f"could not seal private staged file: {path.name}")


def _safe_directory(path: Path, *, label: str) -> Path:
    if (not path.is_absolute() or path.is_symlink() or not path.is_dir()
            or path != path.resolve(strict=False)):
        raise OpenCodeCriticError(f"{label} must be an absolute real directory")
    return path


def _validated_auth(content: bytes) -> bytes:
    if not content.strip():
        raise OpenCodeCriticError("opencode auth carrier is empty")
    return content


def _validated_config(content: bytes) -> bytes:
    try:
        parsed = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenCodeCriticError("opencode config carrier is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise OpenCodeCriticError("opencode config carrier must be an object")
    return content


def _home_carrier(relative: Path, *, label: str,
                  validator: Any) -> bytes:
    home = Path.home()
    if home.is_symlink() or not home.is_dir():
        raise OpenCodeCriticError(f"home directory is unsafe for {label}")
    path = home / relative
    if path.is_symlink() or not path.is_file():
        raise OpenCodeCriticError(f"{label} is unavailable at {path}")
    raw = _read_regular(path, private=True, maximum=MAX_AUTH_BYTES)
    return validator(raw)


def _stage_auth(*, auth: bytes, config: bytes | None, root: Path) -> None:
    """Stage the opencode auth/config into the ephemeral private dir.

    The stage mirrors the caller's XDG roots: XDG_DATA_HOME is pointed at the
    staged data dir (where opencode reads ``opencode/auth.json``) and
    XDG_CONFIG_HOME at the staged config dir (``opencode/opencode.jsonc``).
    """
    data_root = root / "data"
    config_root = root / "config"
    data_root.mkdir(mode=0o700)
    config_root.mkdir(mode=0o700)
    (data_root / "opencode").mkdir(mode=0o700)
    (config_root / "opencode").mkdir(mode=0o700)
    _write_private(data_root / "opencode" / "auth.json", auth)
    if config is not None:
        _write_private(config_root / "opencode" / "opencode.jsonc", config)


def runtime_identity(wrapper: Path) -> dict[str, object]:
    if wrapper.is_symlink() or not wrapper.is_file():
        raise OpenCodeCriticError("opencode wrapper is not a regular file")
    return {
        "kind": RUNTIME_KIND,
        "provider": PROVIDER,
        "model": MODEL,
        "wrapper_path": str(wrapper),
        "wrapper_sha256": _sha256_bytes(_read_regular(wrapper, executable=True)),
        "argv_policy_sha256": ARGV_POLICY_SHA256,
        "auth_staging_policy": AUTH_STAGING_POLICY,
    }


def output_schema(bindings: Mapping[str, str]) -> dict[str, object]:
    required = ["decision", "reason", *BINDING_KEYS]
    properties: dict[str, object] = {
        "decision": {"type": "string", "enum": list(DECISIONS)},
        "reason": {"type": "string", "maxLength": MAX_REASON_CHARS},
    }
    for key in BINDING_KEYS:
        _require_digest(bindings, key)
        properties[key] = {"type": "string", "const": bindings[key]}
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _require_digest(bindings: Mapping[str, str], key: str) -> None:
    value = bindings.get(key)
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise OpenCodeCriticError(f"binding {key} must be a SHA-256 digest")


def _validated_bindings(bindings: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(bindings, Mapping) or set(bindings) != set(BINDING_KEYS):
        raise OpenCodeCriticError("critic bindings are incomplete")
    for key in BINDING_KEYS:
        _require_digest(bindings, key)
    return dict(bindings)


def build_argv(*, wrapper: Path, prompt: str) -> tuple[str, ...]:
    if wrapper.is_symlink() or not wrapper.is_file():
        raise OpenCodeCriticError("opencode wrapper is not a regular file")
    if not prompt or "\x00" in prompt or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise OpenCodeCriticError("critic prompt is empty, NUL-laden, or over the argv bound")
    return (
        str(wrapper), "run", prompt, "--model", MODEL, "--format", "json",
        "--pure",
    )


def _parse_event_stream(raw: bytes, *, label: str) -> str:
    """Reduce the opencode JSON event stream to the assistant's final text."""
    if not raw:
        raise OpenCodeCriticError(f"{label} produced no output")
    texts: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenCodeCriticError(
                f"{label} event stream is not JSON") from exc
        if not isinstance(event, dict):
            continue
        message = event.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if (isinstance(part, dict)
                            and part.get("type") == "text"
                            and isinstance(part.get("text"), str)):
                        texts.append(part["text"])
            elif isinstance(content, str):
                texts.append(content)
    if not texts:
        raise OpenCodeCriticError(f"{label} stream carried no assistant text")
    return "".join(texts)


def _extract_critique(text: str, *, label: str) -> dict[str, Any]:
    """Parse the strict critique JSON, tolerating a bounded prose frame."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise OpenCodeCriticError(
                f"{label} assistant text carries no critique object") from None
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise OpenCodeCriticError(
                f"{label} critique object is not strict JSON") from exc
    if (not isinstance(value, dict) or set(value) != set(RESULT_KEYS)
            or value["decision"] not in DECISIONS
            or not isinstance(value["reason"], str)
            or not value["reason"].strip()
            or len(value["reason"]) > MAX_REASON_CHARS):
        raise OpenCodeCriticError(f"{label} critique shape changed")
    return value


def _destroy_group(process: subprocess.Popen[str], grace_seconds: float) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired as exc:
        raise OpenCodeCriticError(
            "opencode critic process group remained alive after teardown") from exc


def _close_process_pipes(process: subprocess.Popen[str]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            stream.close()


def _run_process(*, argv: Sequence[str], cwd: Path, environment: Mapping[str, str],
                 timeout_seconds: float, grace_seconds: float) -> tuple[bytes, bytes, int]:
    started_ns = time.monotonic_ns()
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            tuple(argv), cwd=cwd, env=dict(environment), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, close_fds=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _destroy_group(process, grace_seconds)
            raise OpenCodeCriticTimeout(
                "opencode critic timed out and its process group was destroyed")
        if len(stdout) > MAX_STDOUT_BYTES or len(stderr) > MAX_STDERR_BYTES:
            raise OpenCodeCriticError("opencode critic output exceeded its bound")
        return stdout, stderr, int(process.returncode)
    finally:
        if process is not None:
            _close_process_pipes(process)
        elapsed = (time.monotonic_ns() - started_ns) / 1e9
        if elapsed > timeout_seconds + grace_seconds + 1.0:
            raise OpenCodeCriticError("opencode critic teardown exceeded its bound")


def _stage_dir(*, workspace: Path) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(
            prefix=".opencode-critic-", dir=workspace) as directory:
        root = Path(directory)
        try:
            yield root
        finally:
            for path in root.rglob("*"):
                if path.is_file():
                    try:
                        path.chmod(0o600)
                    except OSError:
                        pass
            shutil.rmtree(root, ignore_errors=True)


@contextmanager
def _stage_dir_cm(*, workspace: Path) -> Iterator[Path]:
    yield from _stage_dir(workspace=workspace)


def run_critic(
    *, wrapper: Path, workspace: Path, prompt: str,
    bindings: Mapping[str, str], environment: Mapping[str, str],
    expected_wrapper_sha256: str | None = None,
    expected_runtime_identity: Mapping[str, Any] | None = None,
    expected_launcher_sha256: str | None = None,
    timeout_seconds: float = 900.0,
    terminate_grace_seconds: float = 2.0,
) -> CriticResult:
    """Run exactly one DeepSeek V4 Flash critique and verify every binding."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise OpenCodeCriticError("critic prompt must be non-empty")
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise OpenCodeCriticError("critic prompt exceeds its bounded size")
    if (not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise OpenCodeCriticError("critic timeout must be finite and positive")
    if (not isinstance(terminate_grace_seconds, (int, float))
            or isinstance(terminate_grace_seconds, bool)
            or not math.isfinite(terminate_grace_seconds)
            or terminate_grace_seconds <= 0):
        raise OpenCodeCriticError("critic terminate grace must be finite and positive")
    exact_bindings = _validated_bindings(bindings)
    workspace = _safe_directory(workspace, label="critic workspace")
    if wrapper.is_symlink() or not wrapper.is_file():
        raise OpenCodeCriticError("opencode wrapper is not a regular file")
    wrapper_sha256 = _sha256_bytes(_read_regular(wrapper, executable=True))
    if (expected_wrapper_sha256 is not None
            and wrapper_sha256 != expected_wrapper_sha256):
        raise OpenCodeCriticError("opencode wrapper bytes changed after deployment validation")
    current_runtime = {
        "kind": RUNTIME_KIND,
        "provider": PROVIDER,
        "model": MODEL,
        "wrapper_path": str(wrapper),
        "wrapper_sha256": wrapper_sha256,
        "argv_policy_sha256": ARGV_POLICY_SHA256,
        "auth_staging_policy": AUTH_STAGING_POLICY,
    }
    if (expected_runtime_identity is not None
            and current_runtime != dict(expected_runtime_identity)):
        raise OpenCodeCriticError("opencode critic runtime identity changed")
    if (expected_launcher_sha256 is not None
            and _sha256_bytes(Path(__file__).resolve().read_bytes())
            != expected_launcher_sha256):
        raise OpenCodeCriticError("opencode critic launcher/argv policy changed")
    auth = _home_carrier(
        _OPENCODE_AUTH_RELATIVE, label="opencode auth", validator=_validated_auth)
    config_path = Path.home() / _OPENCODE_CONFIG_RELATIVE
    config = None
    if config_path.is_file() and not config_path.is_symlink():
        config = _validated_config(_read_regular(
            config_path, maximum=MAX_CONFIG_BYTES))
    argv = build_argv(wrapper=wrapper, prompt=prompt)
    argv_sha256 = _sha256_bytes(_canonical_bytes(list(argv)))
    with _stage_dir_cm(workspace=workspace) as stage:
        _stage_auth(auth=auth, config=config, root=stage)
        staged_environment = {
            **dict(environment),
            "XDG_DATA_HOME": str(stage / "data"),
            "XDG_CONFIG_HOME": str(stage / "config"),
        }
        stdout, stderr, exit_code = _run_process(
            argv=argv, cwd=workspace, environment=staged_environment,
            timeout_seconds=timeout_seconds,
            grace_seconds=terminate_grace_seconds)
    if exit_code != 0:
        raise OpenCodeCriticError(
            f"opencode critic exited nonzero ({exit_code}): "
            + stderr.decode("utf-8", "replace")[-500:])
    text = _parse_event_stream(stdout, label="opencode critic")
    critique = _extract_critique(text, label="opencode critic")
    observed = {key: critique[key] for key in RESULT_KEYS}
    for key, expected in exact_bindings.items():
        if observed[key] != expected:
            raise OpenCodeCriticError(
                f"opencode critic binding {key} differs from the caller digest")
    return CriticResult(
        decision=observed["decision"], reason=observed["reason"],
        proposal_sha256=observed["proposal_sha256"],
        source_manifest_sha256=observed["source_manifest_sha256"],
        candidate_patch_sha256=observed["candidate_patch_sha256"],
        context_sha256=observed["context_sha256"],
        template_catalog_sha256=observed["template_catalog_sha256"],
        wrapper_sha256=wrapper_sha256, argv_sha256=argv_sha256,
        stdout_sha256=_sha256_bytes(stdout), stderr_sha256=_sha256_bytes(stderr))
