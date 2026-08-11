#!/usr/bin/env python3
"""Run the AutoKernel Codex actor in a workspace-only Docker boundary.

The host Codex ``workspace-write`` sandbox currently fails while configuring a
nested loopback interface.  Disabling that sandbox directly would expose the
measurement host.  This launcher instead gives the actor one writable host
bind (the copied Arena workspace), a read-only asset bind, an ephemeral root
filesystem, and ordinary bridge networking for the model API.  Codex's inner
sandbox is disabled only inside that container.

Importing this module performs no Docker, model, or filesystem work.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence


EXECUTABLE_MODULE = (
    "scripts.kernel_rnd.autokernel.controller.codex_container_actor")
CONTAINER_IMAGE_ID = (
    "sha256:3a2e92b4133d06d1287f96ec47bacd743717b377f4b9df6be1e3af626c35dbb0")
CONTAINER_WORKSPACE = "/workspace"
CONTAINER_ASSETS = "/codex-assets"
CONTAINER_CODEX_HOME = "/codex-home"
ASSET_TEMP_ROOT = Path("/mnt/raid0/llm/tmp")
CA_CERTIFICATE_PATH = Path("/etc/ssl/certs/ca-certificates.crt")
DOCKER_EXECUTABLE = "/usr/bin/docker"
CODEX_NATIVE_RELATIVE = Path(
    "node_modules/@openai/codex-linux-x64/vendor/"
    "x86_64-unknown-linux-musl/bin/codex")
CODE_MODE_HOST_NAME = "codex-code-mode-host"


class CodexContainerError(RuntimeError):
    """The workspace-only actor boundary cannot be established."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _codex_native_assets(wrapper: Path) -> tuple[Path, Path]:
    package_root = wrapper.resolve().parent.parent
    native = package_root / CODEX_NATIVE_RELATIVE
    code_mode_host = native.with_name(CODE_MODE_HOST_NAME)
    if not native.is_file() or not os.access(native, os.X_OK):
        raise CodexContainerError("installed Codex native binary is unavailable")
    if not code_mode_host.is_file() or not os.access(code_mode_host, os.X_OK):
        raise CodexContainerError("installed Codex code-mode host is unavailable")
    return native, code_mode_host


def _auth_file(environment: Mapping[str, str]) -> Path:
    configured = environment.get("CODEX_HOME")
    home = Path(configured) if configured else Path(environment.get("HOME", "")) / ".codex"
    auth = home / "auth.json"
    if not auth.is_file() or auth.is_symlink():
        raise CodexContainerError("Codex auth.json is unavailable or unsafe")
    return auth.resolve()


def build_docker_argv(
    *, workspace: Path, assets: Path, uid: int, gid: int,
    model: str, effort: str, container_name: str = "autokernel-codex-test",
) -> tuple[str, ...]:
    """Build the exact container command without executing Docker or a model."""
    if not workspace.is_absolute() or not workspace.is_dir() or workspace.is_symlink():
        raise CodexContainerError("workspace must be an existing absolute directory")
    if not assets.is_absolute() or not assets.is_dir() or assets.is_symlink():
        raise CodexContainerError("asset root must be an existing absolute directory")
    for path in (workspace, assets):
        if any(character in str(path) for character in (",", "\n", "\r")):
            raise CodexContainerError("Docker mount paths contain unsafe characters")
    if model != "gpt-5.6-sol" or effort != "high":
        raise CodexContainerError("actor model and effort must remain campaign-pinned")
    if not container_name.startswith("autokernel-codex-"):
        raise CodexContainerError("actor container name is outside its owned namespace")
    user = f"{uid}:{gid}"
    return (
        DOCKER_EXECUTABLE, "run", "--rm", "--name", container_name,
        "--init", "--read-only",
        "--network", "bridge", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--pids-limit", "256",
        "--user", user, "--workdir", CONTAINER_WORKSPACE,
        "--mount", f"type=bind,src={workspace},dst={CONTAINER_WORKSPACE}",
        "--mount", f"type=bind,src={assets},dst={CONTAINER_ASSETS},readonly",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=512m,uid=1000,gid=1000,mode=1777",
        "--tmpfs", "/codex-home:rw,nosuid,size=32m,uid=1000,gid=1000,mode=700",
        "--mount", (
            f"type=bind,src={assets / 'auth.json'},"
            f"dst={CONTAINER_CODEX_HOME}/auth.json,readonly"),
        "--env", "HOME=/tmp", "--env", f"CODEX_HOME={CONTAINER_CODEX_HOME}",
        "--env", f"SSL_CERT_FILE={CONTAINER_ASSETS}/ca-certificates.crt",
        CONTAINER_IMAGE_ID, f"{CONTAINER_ASSETS}/codex", "exec", "--json",
        "--ignore-user-config", "--ephemeral", "--model", model,
        "--config", f'model_reasoning_effort="{effort}"',
        "--config", 'approval_policy="never"',
        "--sandbox", "danger-full-access", "--skip-git-repo-check",
        "--cd", CONTAINER_WORKSPACE, "-",
    )


def run_actor(
    *, wrapper: Path, workspace: Path, model: str, effort: str,
    prompt: str, environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    """Stage read-only assets, run one container, then erase staged auth."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise CodexContainerError("actor prompt must be non-empty")
    if not Path(DOCKER_EXECUTABLE).is_file() or not os.access(DOCKER_EXECUTABLE, os.X_OK):
        raise CodexContainerError("Docker executable is unavailable")
    if not CA_CERTIFICATE_PATH.is_file() or CA_CERTIFICATE_PATH.is_symlink():
        raise CodexContainerError("host CA certificate bundle is unavailable or unsafe")
    if not ASSET_TEMP_ROOT.is_dir() or ASSET_TEMP_ROOT.is_symlink():
        raise CodexContainerError("container asset staging root is unavailable or unsafe")
    native, code_mode_host = _codex_native_assets(wrapper)
    auth = _auth_file(environment)
    with tempfile.TemporaryDirectory(
            prefix="autokernel-codex-assets-", dir=ASSET_TEMP_ROOT) as temporary:
        assets = Path(temporary)
        staged = {
            "codex": native,
            CODE_MODE_HOST_NAME: code_mode_host,
            "auth.json": auth,
            "ca-certificates.crt": CA_CERTIFICATE_PATH,
        }
        for name, source in staged.items():
            destination = assets / name
            shutil.copyfile(source, destination)
            destination.chmod(0o500 if name.startswith("codex") else 0o400)
        container_name = f"autokernel-codex-{os.getpid()}-{secrets.token_hex(4)}"
        argv = build_docker_argv(
            workspace=workspace, assets=assets, uid=os.getuid(), gid=os.getgid(),
            model=model, effort=effort, container_name=container_name)
        previous_handlers = {
            sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

        def terminate(_signum: int, _frame: object) -> None:
            raise CodexContainerError("actor container interrupted at checkpoint")

        for sig in previous_handlers:
            signal.signal(sig, terminate)
        result: subprocess.CompletedProcess[str] | None = None
        try:
            result = subprocess.run(
                argv, input=prompt, capture_output=True, text=True, check=False,
                env=dict(environment))
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            subprocess.run(
                (DOCKER_EXECUTABLE, "rm", "--force", container_name),
                capture_output=True, text=True, check=False,
                env=dict(environment), timeout=30)
            still_live = subprocess.run(
                (DOCKER_EXECUTABLE, "inspect", container_name),
                capture_output=True, text=True, check=False,
                env=dict(environment), timeout=30)
            if still_live.returncode == 0:
                raise CodexContainerError(
                    "captured actor container survived exact-name teardown")
        if result is None:
            raise CodexContainerError("actor container returned no process result")
        return result


def runtime_identity(wrapper: Path) -> dict[str, object]:
    """Return non-secret hashes describing the external actor boundary."""
    native, code_mode_host = _codex_native_assets(wrapper)
    return {
        "kind": "docker_workspace_bind_only",
        "docker_path": DOCKER_EXECUTABLE,
        "docker_sha256": _sha256_file(Path(DOCKER_EXECUTABLE)),
        "image_id": CONTAINER_IMAGE_ID,
        "codex_native_sha256": _sha256_file(native),
        "code_mode_host_sha256": _sha256_file(code_mode_host),
        "ca_certificate_sha256": _sha256_file(CA_CERTIFICATE_PATH),
        "writable_host_binds": [CONTAINER_WORKSPACE],
        "host_network_mode": "docker_bridge",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex-wrapper", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True)
    args = parser.parse_args(argv)
    prompt = sys.stdin.read()
    result = run_actor(
        wrapper=Path(args.codex_wrapper), workspace=Path(args.workspace).resolve(),
        model=args.model, effort=args.effort, prompt=prompt,
        environment=os.environ)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
