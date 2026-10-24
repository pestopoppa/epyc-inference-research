#!/usr/bin/env python3
"""Run ONE arm of the AKX-ALL-LB3.2 attribution measurement: one llama-bench process, its stderr timestamped line by
line on CLOCK_MONOTONIC the moment each line is read, then `analyze.py` over the result.

  --arm G   champion store image under `perf record -e cpu-clock:u -k CLOCK_MONOTONIC -D -1 --control fifo:...`.
            Control commands follow phases.control_action (enable at a test's `starting` and at `prompt run`, disable
            only on a non-cached `depth run`, i.e. the long prefill). Commands are written without waiting; a thread
            records each ack, so a slow ack can never delay the timestamp of the next stderr line.
  --arm M   measurement build (GGML_CPU_PROF + LB3.2a per-eval record): no perf. The caller's environment must
            carry GGML_CPU_PROF=1 GGML_CPU_PROF_VERBOSE=1 GGML_CPU_PROF_EVAL_FILE=<prefix>.evals.tsv (checked).

Starts exactly one child process tree (perf ->) taskset -> numactl -> llama-bench; on SIGTERM/SIGHUP terminates that
captured child only and exits 143. Writes: <prefix>.jsonl .err .lines.tsv [.control.tsv .perf.data .perfscript.txt]
[.evals.tsv] .numa .environ .profile.json. Prepared by workspace-f8 2026-10-24; run with /usr/bin/python3 -B -I.

usage: lb32_run.py --arm G|M --prefix OUTPREFIX --place PLACE_SH [--freq 499] -- CMD...
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from akx_lb3_attribution import analyze, phases  # noqa: E402
    from akx_lb3_attribution.perfscript import PERF_FIELDS  # noqa: E402
else:
    from . import analyze, phases
    from .perfscript import PERF_FIELDS


def mono() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC)


class PerfControl:
    """Non-blocking writer of perf --control commands plus an ack-reader thread (acks come back in order)."""

    def __init__(self, ctl: str, ack: str, proc: subprocess.Popen):
        self.log = []                       # [t_sent, cmd, t_ack]
        self.lock = threading.Lock()
        self.ack_fd = os.open(ack, os.O_RDONLY | os.O_NONBLOCK)
        self.ctl_fd = None
        t0 = time.time()
        while self.ctl_fd is None and time.time() - t0 < 60 and proc.poll() is None:
            try:
                self.ctl_fd = os.open(ctl, os.O_WRONLY | os.O_NONBLOCK)
            except OSError:
                time.sleep(0.2)
        self.stop = False
        self.th = threading.Thread(target=self._reader, daemon=True)
        self.th.start()

    def _reader(self):
        buf = b""
        while not self.stop:
            try:
                chunk = os.read(self.ack_fd, 256)
            except BlockingIOError:
                chunk = b""
            except OSError:
                return
            if chunk:
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip() == b"ack":
                        now = mono()
                        with self.lock:
                            for e in self.log:
                                if e[2] is None:
                                    e[2] = now
                                    break
            else:
                time.sleep(0.0005)

    def send(self, cmd: str):
        with self.lock:
            self.log.append([mono(), cmd, None])
        os.write(self.ctl_fd, (cmd + "\n").encode())

    def close(self, path: str):
        time.sleep(0.2)
        self.stop = True
        self.th.join(2)
        with open(path, "w") as f:
            for t, c, a in self.log:
                f.write(f"{t:.6f}\t{c}\t{'' if a is None else f'{a:.6f}'}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("G", "M"), required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--place", required=True)
    ap.add_argument("--freq", type=int, default=499)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    P = a.prefix
    if a.arm == "M":
        need = {"GGML_CPU_PROF": None, "GGML_CPU_PROF_VERBOSE": None, "GGML_CPU_PROF_EVAL_FILE": P + ".evals.tsv"}
        for k, v in need.items():
            if k not in os.environ or (v is not None and os.environ[k] != v):
                print(f"M arm needs {k}{'=' + v if v else ''} in the environment", file=sys.stderr)
                return 6
        for k in ("GGML_CPU_PROF_SKIP", "GGML_CPU_PROF_NNODES_EQ", "GGML_CPU_PROF_NNODES_MIN", "GGML_CPU_PROF_NNODES_MAX"):
            if k in os.environ:
                print(f"M arm refuses {k}: every evaluation must be accumulated", file=sys.stderr)
                return 6
    ctl = ack = None
    if a.arm == "G":
        ctl, ack = P + ".ctl.fifo", P + ".ack.fifo"
        for f in (ctl, ack):
            if os.path.exists(f):
                os.unlink(f)
            os.mkfifo(f)
        full = ["perf", "record", "-q", "-e", "cpu-clock:u", "-F", str(a.freq), "-k", "CLOCK_MONOTONIC", "-D", "-1",
                "--control", f"fifo:{ctl},{ack}", "-o", P + ".perf.data", "--"] + cmd
    else:
        full = cmd
    out = open(P + ".jsonl", "w")
    err = open(P + ".err", "w")
    lt = open(P + ".lines.tsv", "w")
    # the receipt time of a stderr line is what phases.timed_reps measures from: read with a raw unbuffered pipe
    proc = subprocess.Popen(full, stdout=out, stderr=subprocess.PIPE, bufsize=0)
    ctrl = None

    def on_term(sig, frm):
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        sys.exit(143)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, on_term)
    if a.arm == "G":
        ctrl = PerfControl(ctl, ack, proc)
        if ctrl.ctl_fd is None:
            print("perf control fifo never opened", file=sys.stderr)
            proc.terminate()
            proc.wait()
            return 6
    enabled, placed = False, False
    rd = proc.stderr
    pending = b""
    while True:
        chunk = rd.read1(65536) if hasattr(rd, "read1") else os.read(rd.fileno(), 65536)
        if not chunk:
            break
        now = mono()
        pending += chunk
        while b"\n" in pending:
            raw, pending = pending.split(b"\n", 1)
            text = raw.decode(errors="replace")
            lt.write(f"{now:.6f}\t{text}\n")
            err.write(text + "\n")
            p = phases.parse_progress(text)
            if p is None:
                continue
            _idx, _n, kind, _rep = p
            if ctrl is not None:
                act = phases.control_action(kind)
                if act == "enable" and not enabled:
                    ctrl.send("enable")
                    enabled = True
                elif act == "disable" and enabled:
                    ctrl.send("disable")
                    enabled = False
            if not placed:                       # the model is loaded once the first progress line appears
                placed = True
                try:
                    kids = open(f"/proc/{proc.pid}/task/{proc.pid}/children").read().split()
                    lb = kids[0] if (a.arm == "G" and kids) else str(proc.pid)
                    env = open(f"/proc/{lb}/environ", "rb").read().split(b"\0")
                    Path(P + ".environ").write_text("\n".join(sorted(x.decode(errors="replace") for x in env
                                                                     if x.startswith((b"GGML_", b"OMP_", b"LLAMA_")))) + "\n")
                    r = subprocess.run(["bash", a.place, lb], capture_output=True, text=True)
                    Path(P + ".numa").write_text(r.stdout + r.stderr + f"\nRC={r.returncode}\n")
                except OSError as e:
                    Path(P + ".numa").write_text(f"placement read failed: {e}\nRC=99\n")
    if pending:
        lt.write(f"{mono():.6f}\t{pending.decode(errors='replace')}\n")
    rc = proc.wait()
    for f in (out, err, lt):
        f.close()
    if ctrl is not None:
        ctrl.close(P + ".control.tsv")
        for f in (ctl, ack):
            os.unlink(f)
        with open(P + ".perfscript.txt", "w") as f:
            subprocess.run(["perf", "script", "-i", P + ".perf.data", "-F", PERF_FIELDS], stdout=f,
                           stderr=subprocess.DEVNULL)
    if rc == 0:
        analyze.main(["--arm", a.arm, "--prefix", P])
    return rc


if __name__ == "__main__":
    sys.exit(main())
