"""CTC-2 part 2: carry stage admission between serial_run and its run.py children.

Coordinated mode exists only when serial_run is launched with
`--max-active-targets > 1`. It then hands each child one end of a socketpair
(`AUTOKERNEL_STAGE_FD`) and keeps the other end on a `ParentServer` that feeds the
parent-owned `stage_admission.Admission`. Requests are bound to the child identity
the PARENT captured after spawn and to the target identity the parent derived
from the child's own argv; the child never declares either.

In the child, `StageScope` replaces run.py's whole-batch CPU/GPU claims: every
heavy callback (gate = build+correctness, measure, confirm, reprofile, champion /
anchor build) runs inside a stage that asks for admission, then takes the real
claims in order CPU regions -> device through the existing claim owners, and
releases them after the callback returns. Nested callbacks on the same thread
reuse the current lease instead of re-flocking through a second descriptor.

Paths whose owners retain a run-long claim object (runtime admission/restore,
source LOO validation) are refused in coordinated mode rather than handed a
released claim; they keep running in the default single-target mode.

Default mode (no fd) is byte-for-byte the previous behaviour: `from_env()`
returns None and run.py takes its original whole-batch claims.
"""
from __future__ import annotations

import os
import socket
import threading
from contextlib import ExitStack, contextmanager
from typing import Callable, Optional

from . import stage_admission as sa
from . import stage_footprint as sf

ENV_FD = "AUTOKERNEL_STAGE_FD"


class CoordinationRefused(RuntimeError):
    pass


def _footprint_wire(fp: sf.StageFootprint) -> dict:
    return {"kind": fp.kind, "target": fp.target, "attempt": fp.attempt,
            "regions": list(fp.regions), "devices": list(fp.devices)}


def _footprint_from_wire(body: dict) -> sf.StageFootprint:
    kind = body.get("kind")
    if kind not in sf.STAGE_KINDS:
        raise sa.AdmissionRefused(f"unknown stage kind {kind!r}")
    regions = tuple(str(r) for r in body.get("regions", ()))
    devices = tuple(str(d) for d in body.get("devices", ()))
    adapter = sf.ExecutionAdapter("wire", None, None, None, None)
    return sf.StageFootprint(kind, str(body.get("target")), str(body.get("attempt")), adapter,
                             None, None, None, None, frozenset(), frozenset(), regions, None,
                             devices, None)


# ---- parent side ---------------------------------------------------------------

class ParentServer:
    """One reader thread per child; EOF or a protocol error cancels that child."""

    def __init__(self, admission: Optional[sa.Admission] = None) -> None:
        self.admission = admission or sa.Admission()
        self._threads: list[threading.Thread] = []

    def spawn_pair(self) -> tuple[socket.socket, socket.socket]:
        parent, child = socket.socketpair()
        child.set_inheritable(True)
        return parent, child

    def attach(self, sock: socket.socket, identity: sa.ChildIdentity) -> threading.Thread:
        self.admission.register(identity)
        thread = threading.Thread(target=self._serve, args=(sock, identity), daemon=True,
                                  name=f"stage-admission-{identity.pid}")
        thread.start()
        self._threads.append(thread)
        return thread

    def _serve(self, sock: socket.socket, identity: sa.ChildIdentity) -> None:
        reader = sock.makefile("rb")
        current = None
        try:
            for line in reader:
                msg = sa.decode(line)
                if msg["kind"] == "request" and current is None:
                    try:
                        current = self.admission.request(
                            identity, str(msg.get("attempt")),
                            _footprint_from_wire(msg.get("footprint") or {}))
                    except sa.AdmissionRefused as exc:
                        sock.sendall(sa.encode("refused", reason=str(exc)))
                        continue
                    current.event.wait()
                    sock.sendall(sa.encode("grant", seq=current.seq))
                elif msg["kind"] == "release" and current is not None:
                    self.admission.release(current)
                    current = None
                elif msg["kind"] == "status":
                    sock.sendall(sa.encode("status", queue=self.admission.status()))
                else:
                    sock.sendall(sa.encode("refused", reason=f"unexpected {msg['kind']}"))
        except (OSError, ValueError, sa.AdmissionRefused):
            pass
        finally:
            self.admission.cancel_child(identity)
            try:
                sock.close()
            except OSError:
                pass


def popen_kwargs(server: Optional[ParentServer], env: Optional[dict], pass_fds: tuple):
    """(env, pass_fds, parent_sock, child_sock) for one child; identity when no server."""
    if server is None:
        return env, pass_fds, None, None
    parent, child = server.spawn_pair()
    env = dict(os.environ if env is None else env, **{ENV_FD: str(child.fileno())})
    return env, (*pass_fds, child.fileno()), parent, child


def attach_child(server: Optional[ParentServer], parent_sock, child_sock, process,
                 target: str) -> None:
    if server is None:
        return
    from . import worker_lifecycle
    child_sock.close()
    ident = worker_lifecycle.process_identity(process.pid)
    server.attach(parent_sock, sa.ChildIdentity(ident.pid, ident.start_ticks, target))


def target_of_argv(child_argv) -> str:
    """Parent-derived target identity: the enrolled revision the child's argv selects."""
    from . import campaign_cli, legacy_targets
    from pathlib import Path
    argv = list(child_argv)
    resolved = target_id = None
    for i, arg in enumerate(argv[:-1]):
        if arg == "--resolved-campaign":
            resolved = argv[i + 1]
        elif arg == "--target-id":
            target_id = argv[i + 1]
    if resolved is None or target_id is None:
        raise CoordinationRefused("coordinated child argv names no resolved target")
    campaign = campaign_cli.load_previous(Path(resolved))
    matches = [t for t in campaign.targets if target_id in t.target_ids]
    if len(matches) != 1:
        raise legacy_targets.TargetSelectionRefused(f"target {target_id!r} is not unique")
    return sf.target_identity(matches[0])


# ---- child side ----------------------------------------------------------------

class ChildClient:
    """Serialises this child's requests: one outstanding stage per target."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._reader = sock.makefile("rb")
        self._lock = threading.Lock()

    @contextmanager
    def admitted(self, footprint: sf.StageFootprint):
        with self._lock:
            self._sock.sendall(sa.encode("request", attempt=footprint.attempt,
                                         footprint=_footprint_wire(footprint)))
            reply = sa.decode(self._reader.readline() or b'{"kind": "refused"}')
            if reply["kind"] != "grant":
                raise CoordinationRefused(f"stage admission refused: {reply.get('reason')}")
            try:
                yield reply
            finally:
                self._sock.sendall(sa.encode("release", seq=reply.get("seq")))


class StageScope:
    """run.py's per-stage claim owner in coordinated mode."""

    def __init__(self, client: ChildClient, *, runtime: sf.StageFootprint,
                 build: Optional[sf.StageFootprint], hold_cpu: Callable[[str], object],
                 hold_device: Optional[Callable[[], object]]) -> None:
        self.client, self.runtime, self.build = client, runtime, build
        self._hold_cpu, self._hold_device = hold_cpu, hold_device
        self._local = threading.local()
        self.windows: list[dict] = []

    def held(self) -> list:
        """The claim objects currently held on this thread (empty outside a stage)."""
        return list(getattr(self._local, "held", ()))

    def _footprint(self, kind: str) -> sf.StageFootprint:
        base = self.runtime
        if kind == "build" and self.build is not None:
            regions = tuple(sorted(set(self.build.regions) | set(base.regions)))
            return sf.StageFootprint("build", base.target, base.attempt, base.adapter,
                                     base.source, self.build.build, None, None,
                                     self.build.cpus | base.cpus, self.build.cores | base.cores,
                                     regions, None, base.devices, self.build.build_jobs)
        return sf.StageFootprint(kind, base.target, base.attempt, base.adapter, base.source,
                                 base.build, base.instrument, base.recipe, base.cpus,
                                 base.cores, base.regions, base.memory, base.devices, None)

    @contextmanager
    def stage(self, kind: str):
        if getattr(self._local, "depth", 0):
            self._local.depth += 1
            try:
                yield self.held()
            finally:
                self._local.depth -= 1
            return
        footprint = self._footprint(kind)
        with self.client.admitted(footprint), ExitStack() as claims:
            from ..execution.cpu_region_claim import render_cpu_list
            held = [claims.enter_context(self._hold_cpu(render_cpu_list(footprint.cpus)))]
            if footprint.devices and self._hold_device is not None:
                held.append(claims.enter_context(self._hold_device()))
            self._local.depth, self._local.held = 1, held
            self.windows.append({"kind": kind, "target": footprint.target})
            try:
                yield held
            finally:
                self._local.depth, self._local.held = 0, ()

    def wrap(self, kind: str, fn: Callable) -> Callable:
        def staged(*args, **kwargs):
            with self.stage(kind):
                return fn(*args, **kwargs)
        staged.__wrapped__ = fn
        return staged

    def wrap_factory(self, kind: str, factory: Callable) -> Callable:
        """`gate_for(worker)` style: wrap the callable the factory returns."""
        def staged_factory(*args, **kwargs):
            return self.wrap(kind, factory(*args, **kwargs))
        staged_factory.__wrapped__ = factory
        return staged_factory


def client_from_env(environ=None) -> Optional[ChildClient]:
    value = (os.environ if environ is None else environ).get(ENV_FD)
    if not value:
        return None
    if not value.isdigit():
        raise CoordinationRefused(f"{ENV_FD} is not a file descriptor")
    return ChildClient(socket.socket(fileno=int(value)))


def refuse_run_long_claim_owners(*, runtime_enabled: bool, source_loo: bool) -> None:
    if runtime_enabled:
        raise CoordinationRefused("runtime admission retains a run-long claim; "
                                  "not available under --max-active-targets > 1")
    if source_loo:
        raise CoordinationRefused("source LOO validation retains a run-long claim; "
                                  "not available under --max-active-targets > 1")


def install(client: ChildClient, *, args, target, launch, resources, hold_cpu, hold_device,
            runtime_enabled: bool) -> StageScope:
    """Build run.py's StageScope from CTC-1 footprints, refusing unsupported paths."""
    refuse_run_long_claim_owners(runtime_enabled=runtime_enabled,
                                 source_loo=bool(getattr(args, "validate_source_loo", False)))
    if getattr(args, "cpu_held_intervals_v2", "off") == "on":
        raise CoordinationRefused("interval v2 accounting of stage windows is CTC-4")
    if getattr(args, "gpu_window", "off") == "required":
        raise CoordinationRefused("--gpu-window required hands one window-long device flock")
    if target is None or launch is None or resources is None:
        raise CoordinationRefused("coordinated stages need an enrolled target and its launch")
    adapter = sf.execution_adapter(_argv(args))
    runtime = sf.runtime_footprint("measure", target, launch, resources, adapter=adapter,
                                   attempt=str(getattr(args, "store", "")),
                                   environment=os.environ)
    build = sf.build_footprint(target, resources, adapter=adapter, attempt=runtime.attempt,
                               build=str(launch.build_dir))
    return StageScope(client, runtime=runtime, build=build, hold_cpu=hold_cpu,
                      hold_device=hold_device)


def _argv(args) -> list:
    out = ["--model", str(getattr(args, "model", "") or "")]
    for flag, attr in (("--cpu-serving-launch", "cpu_serving_launch"),
                       ("--gpu-serving-launch", "gpu_serving_launch"),
                       ("--serving-instrument", "serving_instrument"),
                       ("--cpu-screen-scope", "cpu_screen_scope"), ("--pairs", "pairs"),
                       ("--serving-pairs", "serving_pairs")):
        value = getattr(args, attr, None)
        if value is not None:
            out += [flag, str(value)]
    return out
