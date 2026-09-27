"""Declared-vs-live launch ENV attestation (realized-first, fabric axiom 2).

WHY THIS EXISTS
---------------
From b060dd56 (2026-07-31) to 2026-09-26 the launcher silently removed every GGML_* knob a
role declared in stack_env._ROLE_ENV_BLOCKS: a binary-override strip keyed on binary_dir
presence, which that commit made universal. runtime_attestation compared argv and a few
env-derived fields (LD_LIBRARY_PATH, KMP_BLOCKTIME) against the compiled priors, never the
declared env blocks, so nothing noticed for eight weeks. architect_critic's
GGML_NUMA_REPACK_INTERLEAVE was declared the whole time and never reached its process.

This check reads the env each live process ACTUALLY has (/proc/<pid>/environ, read-only)
and fails on any key the stack DECLARES for it that is missing or different:

  llama-server role   stack_env._CANONICAL_OMP_ENV + _role_env_overrides(role), and
                      LD_LIBRARY_PATH must contain the LLVM-20 libomp dir
  aux service         stack_manifest.AUX_SERVICES[name].env

A process with no declared env contract (the API, docker services) is listed as not
attested, never as passing. If no managed process can be read at all, the verdict is
COULD-NOT-CHECK, which is not a pass.

EXPECTED DEVIATIONS (UFH-12 arm A3, 2026-09-27). A one-shot experiment may relaunch the embedders
with a recorded env override (scripts/server/embedder_env_override.py). A mismatch is EXPECTED --
listed in ``expected``, surfaced as a warning naming the experiment, never an error and never
silent -- only when the record is unexpired, names that pid, and declares exactly the live value.
Anything else (an expired record, another pid, another key or value) stays an error.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from scripts.server.stack_env import _CANONICAL_OMP_ENV, _LLVM20_LIBDIR, _role_env_overrides


@dataclass
class EnvAttestation:
    compared: list[str] = field(default_factory=list)      # "name:port pid N (k keys)"
    not_attested: list[str] = field(default_factory=list)  # no declared env contract
    errors: list[str] = field(default_factory=list)        # declared key missing/different, or unreadable
    expected: list[str] = field(default_factory=list)      # deviations covered by a live override record
    override_record: dict | None = None                    # the embedder env override record read, if any

    @property
    def verdict(self) -> str:
        if self.errors:
            return "failed"
        return "ok" if self.compared else "could-not-check"


def read_environ(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    return dict(item.decode(errors="replace").split("=", 1) for item in raw.split(b"\0") if b"=" in item)


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def _alive(pid: int) -> bool:
    return pid > 0 and Path(f"/proc/{pid}").exists()


def declared_env_for(name: str, role: str, cmdline: str, aux_services: dict) -> tuple[dict[str, str], bool] | None:
    """(declared env, is_llama_server) for one managed process, or None when nothing is declared."""
    if name in aux_services or role in aux_services:
        spec = aux_services.get(name) or aux_services.get(role)
        return {k: str(v) for k, v in (getattr(spec, "env", None) or {}).items()}, False
    if "llama-server" in cmdline:
        declared = dict(_CANONICAL_OMP_ENV)
        declared.update(_role_env_overrides(role))
        return declared, True
    return None


_READ_OVERRIDE = object()


def _read_override_record() -> dict | None:
    from scripts.server.embedder_env_override import read_record

    return read_record()


def attest(state: dict, *, aux_services: dict | None = None, pids_on_port=None,
           override_record=_READ_OVERRIDE) -> EnvAttestation:
    """Compare every live managed process's environ with the env the stack declares for it.

    `state` maps name -> ProcessInfo-like (role, pid, port). Several names share one process
    (aliases, server_<port> rows); each PID is attested once, under its primary role.
    `override_record` defaults to the live embedder env override record (None = no record).
    """
    if aux_services is None:
        from scripts.server.stack_manifest import AUX_SERVICES as aux_services  # noqa: N811
    out = EnvAttestation()
    if override_record is _READ_OVERRIDE:
        try:
            override_record = _read_override_record()
        except Exception as exc:  # noqa: BLE001 — an unreadable record covers nothing, loudly
            out.errors.append(f"embedder env override record unreadable ({exc}); no deviation is expected")
            override_record = None
    out.override_record = override_record
    from scripts.server.embedder_env_override import covers, is_expired
    by_pid: dict[int, tuple[str, object]] = {}
    for name, info in sorted(state.items()):
        pid = int(getattr(info, "pid", -1))
        port = int(getattr(info, "port", 0))
        if not _alive(pid) and pids_on_port is not None and port:
            live = pids_on_port(port)
            pid = live[0] if live else pid
        if not _alive(pid):
            continue
        role = str(getattr(info, "role", name))
        # Prefer the row whose name IS the process's role (the primary), else the first seen.
        if pid not in by_pid or name == role:
            by_pid[pid] = (name, info)
    for pid, (name, info) in sorted(by_pid.items()):
        role = str(getattr(info, "role", name))
        port = int(getattr(info, "port", 0))
        tag = f"{name}:{port} pid {pid}"
        cmdline = _cmdline(pid)
        contract = declared_env_for(name, role, cmdline, aux_services)
        if contract is None:
            out.not_attested.append(tag)
            continue
        declared, is_llama = contract
        try:
            live = read_environ(pid)
        except OSError as exc:
            out.errors.append(f"{tag}: cannot read /proc/{pid}/environ ({exc.__class__.__name__}); could not check")
            continue
        for key, want in sorted(declared.items()):
            got = live.get(key)
            if got != want:
                drift = (f"{tag} ({role}): declared {key}={want!r} but live has "
                         + ("it MISSING" if got is None else f"{key}={got!r}"))
                if covers(override_record, pid=pid, key=key, live_value=got):
                    out.expected.append(
                        f"{drift} -- EXPECTED under experiment "
                        f"{override_record.get('experiment_id')!r} (expires {override_record.get('expires_at')})")
                    continue
                if override_record and is_expired(override_record) and \
                        (override_record.get("env") or {}).get(key) == got:
                    drift += (f" -- override record {override_record.get('experiment_id')!r} EXPIRED at "
                              f"{override_record.get('expires_at')}: restore with `reload embedders`")
                out.errors.append(drift)
        for key in sorted(set((override_record or {}).get("env") or {}) - set(declared)):
            # An override of a key the stack does not declare (e.g. KMP_LIBRARY) is still a deviation.
            got = live.get(key)
            if got is None:
                continue
            if covers(override_record, pid=pid, key=key, live_value=got):
                out.expected.append(f"{tag} ({role}): undeclared {key}={got!r} -- EXPECTED under "
                                    f"experiment {override_record.get('experiment_id')!r}")
        n = len(declared)
        if is_llama:
            n += 1
            if _LLVM20_LIBDIR not in live.get("LD_LIBRARY_PATH", "").split(os.pathsep):
                out.errors.append(f"{tag} ({role}): LD_LIBRARY_PATH lacks the declared {_LLVM20_LIBDIR}")
        out.compared.append(f"{tag} ({role}, {n} declared keys)")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    from scripts.server.stack_state import load_state_file

    ap = argparse.ArgumentParser(description="Declared-vs-live launch env attestation (read-only).")
    ap.add_argument("--state-file", type=Path, default=None,
                    help="orchestrator_state.json (default: this tree's logs/orchestrator_state.json)")
    args = ap.parse_args(argv)
    if args.state_file is None:
        from scripts.server.orchestrator_stack import STATE_FILE
        args.state_file = STATE_FILE
    from scripts.server.orchestrator_stack import _pids_on_port

    result = attest(load_state_file(args.state_file), pids_on_port=_pids_on_port)
    for line in result.compared:
        print(f"  compared      {line}")
    for line in result.not_attested:
        print(f"  not-attested  {line} (no declared env contract)")
    for line in result.expected:
        print(f"  EXPECTED      {line}")
    for line in result.errors:
        print(f"  ERROR         {line}")
    if result.override_record is not None:
        rec = result.override_record
        print(f"  override-record experiment={rec.get('experiment_id')!r} env={rec.get('env')} "
              f"expires_at={rec.get('expires_at')}")
    print(f"declared_env_attestation: {result.verdict} "
          f"({len(result.compared)} compared, {len(result.not_attested)} not attested, {len(result.errors)} errors, "
          f"{len(result.expected)} expected deviations)")
    return {"ok": 0, "failed": 1}.get(result.verdict, 2)


if __name__ == "__main__":
    raise SystemExit(main())
