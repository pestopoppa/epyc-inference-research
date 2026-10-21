"""AKR-SG-2 follow-up: launch an A/B llama-server PAIR on private ports and prove what is answering.

Two servers (arm A, arm B), each its own process group, its own port, its own ``LD_LIBRARY_PATH`` (the
three-ggml-generations hazard), explicit CPU list or GPU device. Nothing here runs unless a caller (the
driver ``run_paired_serving_ab.py``) is inside a confirmed window. Proofs, all recorded in the returned dict:

  * port ownership   ``ss`` pid on the arm's port == the pid this launcher spawned (a foreign server on a
                     port makes /health answer for the wrong process; INC-class lesson from job 06)
  * linkage          ``/proc/<pid>/maps`` loads libggml*/libllama* ONLY from the arm's own build dir
                     (``ldd`` cannot see dlopened ``libggml-hip.so``); for ``--device gpu`` libggml-hip must be
                     mapped; a CPU arm must NOT map it
  * identity         ``/props`` n_ctx / total_slots echoed; ``/metrics`` answers (``--metrics`` is forced on)
  * teardown         SIGTERM the arm's captured group, then SIGKILL, then ``kill -0`` verification; never a
                     name-pattern kill

Denied ports (the production stack) are refused before anything starts.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DENY_PORTS = frozenset({8000, 8070, 8071, 8072, 8080, 8081, 8082, 8083, 8090, 8100, 8180, 8181, 8182, 8199})


@dataclass
class ArmSpec:
    name: str                       # "A" | "B"
    build_dir: Path                 # dir holding llama-server and ITS libggml*.so
    port: int
    model: Path
    device: str = "cpu"             # cpu | gpu
    cpu_list: str | None = None     # taskset list; required for cpu
    ctx: int = 4096
    np: int = 2
    threads: int = 8
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


class PairError(RuntimeError):
    pass


def listening_pid(port: int) -> int | None:
    out = subprocess.run(["ss", "-H", "-ltnp", f"sport = :{port}"], capture_output=True, text=True).stdout
    m = re.search(r"pid=(\d+)", out)
    return int(m.group(1)) if m else None


def port_is_free(port: int) -> bool:
    return listening_pid(port) is None and not subprocess.run(["ss", "-H", "-ltn", f"sport = :{port}"], capture_output=True, text=True).stdout.strip()


def check_specs(a: ArmSpec, b: ArmSpec) -> None:
    for s in (a, b):
        if s.port in DENY_PORTS:
            raise PairError(f"arm {s.name}: port {s.port} is a production port")
        if not (s.build_dir / "llama-server").exists():
            raise PairError(f"arm {s.name}: {s.build_dir}/llama-server missing")
        if not s.model.exists():
            raise PairError(f"arm {s.name}: model {s.model} missing")
        if s.device == "cpu" and not s.cpu_list:
            raise PairError(f"arm {s.name}: cpu arm needs an explicit cpu_list")
        if s.device not in ("cpu", "gpu"):
            raise PairError(f"arm {s.name}: device {s.device!r}")
    if a.port == b.port:
        raise PairError("A and B need different ports")
    if a.cpu_list and a.cpu_list == b.cpu_list and a.device == b.device == "cpu":
        raise PairError("A and B on the SAME cpu_list would contend; give each its own region")


def argv_for(s: ArmSpec) -> list[str]:
    argv = [str(s.build_dir / "llama-server"), "-m", str(s.model), "--host", "127.0.0.1", "--port", str(s.port),
            "-c", str(s.ctx), "-np", str(s.np), "-t", str(s.threads), "--metrics", "--no-webui", *s.extra_args]
    if s.device == "cpu":
        argv = ["taskset", "-c", s.cpu_list, *argv, "-ngl", "0"]
    return argv


def env_for(s: ArmSpec) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    env["LD_LIBRARY_PATH"] = str(s.build_dir)            # ONLY this arm's own libs
    env.update(s.env)
    return env


def maps_linkage(pid: int, build_dir: Path, device: str) -> dict:
    real = os.path.realpath(build_dir)
    libs = set()
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        p = line.split(None, 5)[-1] if len(line.split(None, 5)) == 6 else ""
        if re.search(r"/lib(ggml|llama)[^/]*\.so", p):
            libs.add(os.path.realpath(p))
    outside = sorted(p for p in libs if not p.startswith(real + os.sep))
    hip = any("libggml-hip" in p for p in libs)
    ok = bool(libs) and not outside and (hip if device == "gpu" else not hip)
    return {"libs": sorted(libs), "outside_build_dir": outside, "hip_mapped": hip, "ok": ok}


def _get_json(url: str):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


class ServerPair:
    def __init__(self, a: ArmSpec, b: ArmSpec, log_dir: Path):
        check_specs(a, b)
        self.specs = {"A": a, "B": b}
        self.log_dir = log_dir
        self.procs: dict[str, subprocess.Popen] = {}
        self.proof: dict[str, dict] = {}

    def url(self, arm: str) -> str:
        return f"http://127.0.0.1:{self.specs[arm].port}"

    def start(self, ready_timeout_s: float = 600.0) -> dict:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for arm, s in self.specs.items():
            if not port_is_free(s.port):
                raise PairError(f"arm {arm}: port {s.port} is already listening (not ours)")
        try:
            for arm, s in self.specs.items():
                self.procs[arm] = subprocess.Popen(argv_for(s), env=env_for(s), stdout=open(self.log_dir / f"{arm}.log", "ab"),
                                                   stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.time() + ready_timeout_s
            for arm, s in self.specs.items():
                while True:
                    if self.procs[arm].poll() is not None:
                        raise PairError(f"arm {arm}: server exited rc={self.procs[arm].returncode} (see {self.log_dir}/{arm}.log)")
                    try:
                        if _get_json(self.url(arm) + "/health").get("status") == "ok":
                            break
                    except Exception:
                        pass
                    if time.time() > deadline:
                        raise PairError(f"arm {arm}: not ready in {ready_timeout_s}s")
                    time.sleep(2)
            for arm, s in self.specs.items():
                self.proof[arm] = self.prove(arm)
                if not self.proof[arm]["ok"]:
                    raise PairError(f"arm {arm}: proof failed {json.dumps(self.proof[arm])[:600]}")
        except BaseException:
            self.stop()
            raise
        return self.proof

    def _taskset_child(self, pid: int) -> int:
        """taskset exec()s in place, so the spawned pid IS llama-server; kept as a seam for wrappers."""
        return pid

    def prove(self, arm: str) -> dict:
        s, pid = self.specs[arm], self._taskset_child(self.procs[arm].pid)
        owner = listening_pid(s.port)
        link = maps_linkage(pid, s.build_dir, s.device)
        props = {}
        try:
            p = _get_json(self.url(arm) + "/props")
            props = {"total_slots": p.get("total_slots"), "n_ctx": (p.get("default_generation_settings") or {}).get("n_ctx")}
        except Exception as e:
            props = {"error": str(e)[:100]}
        try:
            urllib.request.urlopen(self.url(arm) + "/metrics", timeout=5).read(64)
            metrics_ok = True
        except Exception:
            metrics_ok = False
        return {"arm": arm, "port": s.port, "spawned_pid": pid, "port_owner_pid": owner, "port_owned": owner == pid,
                "linkage": link, "props": props, "metrics_ok": metrics_ok,
                "ok": owner == pid and link["ok"] and metrics_ok}

    def stop(self) -> dict:
        res = {}
        for arm, p in self.procs.items():
            pid, alive = p.pid, p.poll() is None
            if alive:
                for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
                    try:
                        os.killpg(pid, sig)
                    except ProcessLookupError:
                        break
                    t = time.time()
                    while time.time() - t < wait and p.poll() is None:
                        time.sleep(0.5)
                    if p.poll() is not None:
                        break
            try:
                p.wait(timeout=1)
            except Exception:
                pass
            try:
                os.kill(pid, 0)
                dead = False
            except ProcessLookupError:
                dead = True
            res[arm] = {"pid": pid, "dead": dead}
        self.procs.clear()
        return res
