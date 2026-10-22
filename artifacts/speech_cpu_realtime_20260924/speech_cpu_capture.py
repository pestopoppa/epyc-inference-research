#!/usr/bin/env python3
"""Belief-kernel write side for the CPU speech harnesses (VB-SPEECH-CPU-1 / VB-SPEECH-CPU-1a).

WHY: `speech_cpu_bench.py` and `live_llm_contention.py` wrote rows with RTF, wall time and the
core layout but no schema version, row hash, binary digest or co-tenant identity. Those
2026-09-24 rows produce ZERO belief tuples and are never backfilled: a field invented on read
would claim a warrant the run never captured. This module is what a re-measurement writes
through, so its rows are gradeable by `claim_tuple.grade()` (epyc-root adapter
`speech-cpu-measurement`, which imports `validate_row` from this file by pinned sha256).

The three cautions of the source row are enforced as REQUIRED fields, not prose:
  1. the core layout is part of the measurand (whisper collapses on some layouts at the same
     thread count) -> exact core list + thread count on every row;
  2. TTS thread count exists only via the get_nprocs preload shim -> `shim_nprocs` on every
     TTS row (and the shim's absence recorded as such, never assumed);
  3. co-tenant state decides the result -> at least one co-tenant sample taken DURING the row
     (sample time inside [t_start, t_end]); a sample taken after the row is not evidence.

A capped, timed-out or guard-aborted request is written with `completed: false` and
`value: null`, so it can never read as a completed measurement, and it is never dropped
(the adapter counts it in the run's censored denominator).

Opt-in: the harnesses write capture rows only when SPEECH_CPU_CAPTURE_DIR is set, and that
directory must NOT be this 2026-09-24 artifact's own `raw/` (its rows are hashed evidence).
Pure stdlib; it starts no process and signals nothing; /proc is read only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

CAPTURE_SCHEMA = "epyc.vidya.speech_cpu_capture.v1"
CAPTURE_PRODUCER = "epyc-inference-research/artifacts/speech_cpu_realtime_20260924"
HARNESSES = ("speech_cpu_bench", "live_llm_contention")
RETROSPECTIVE_RAW = Path(__file__).resolve().parent / "raw"

#: metric -> (unit, direction). Directions are explicit per metric (mixed directions are a caution).
CAPTURE_METRICS = {
    "stt_rtf": ("ratio", "lower_better"),
    "tts_rtf": ("ratio", "lower_better"),
    "tts_first_packet_s": ("s", "lower_better"),
    "cotenant_llm_tok_s": ("tok/s", "higher_better"),
}
REQUIRED_KEYS = (
    "schema", "producer", "harness", "run_id", "row_index", "measurement_id", "emitted_at", "date",
    "metric", "unit", "metric_direction", "value", "category", "reps", "reps_basis", "protocol_id",
    "binaries", "placement", "cotenant", "request", "native", "row_sha256",
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CaptureRefused(ValueError):
    """The row (or its destination) would violate the write-side contract."""


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def row_digest(row: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in row.items() if k != "row_sha256"}).encode()).hexdigest()


def binary_digest(path: str) -> dict:
    """sha256 + mtime of the binary that served the row (never a hand-typed digest)."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    st = p.stat()
    return {"path": str(p), "sha256": h.hexdigest(), "mtime": int(st.st_mtime), "size": st.st_size}


# ------------------------------------------------------------------ co-tenant sampling (/proc, read only)
def _listen_inodes(port: int) -> set:
    inodes = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            f = line.split()
            if len(f) > 9 and f[3] == "0A" and int(f[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(f[9])
    return inodes


def pid_listening_on(port: int) -> Optional[int]:
    """The pid owning a LISTEN socket on `port`, or None when it cannot be resolved."""
    inodes = _listen_inodes(port)
    if not inodes:
        return None
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            for fd in (d / "fd").iterdir():
                link = os.readlink(fd)
                if link.startswith("socket:[") and link[8:-1] in inodes:
                    return int(d.name)
        except OSError:
            continue
    return None


def _cpu_ticks(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12])  # utime + stime


def sample_process(pid: Optional[int], *, interval_s: float = 0.25) -> dict:
    """One co-tenant sample: argv, `-t`, allowed cpus, and CPU ticks over `interval_s`."""
    now = time.time()
    if pid is None:
        return {"t": now, "pid": None, "resolved": False}
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        argv = [a.decode(errors="replace") for a in argv if a]
        status = Path(f"/proc/{pid}/status").read_text()
        cpus = next((ln.split(":", 1)[1].strip() for ln in status.splitlines()
                     if ln.startswith("Cpus_allowed_list:")), None)
        a = _cpu_ticks(pid)
        time.sleep(interval_s)
        b = _cpu_ticks(pid)
    except (OSError, IndexError, ValueError):
        return {"t": now, "pid": pid, "resolved": False}
    threads = None
    for flag in ("-t", "--threads"):
        if flag in argv and argv.index(flag) + 1 < len(argv):
            threads = argv[argv.index(flag) + 1]
    hz = os.sysconf("SC_CLK_TCK")
    return {"t": now, "pid": pid, "resolved": True, "argv": argv, "threads_flag": threads,
            "cpus_allowed_list": cpus, "cpu_ticks_delta": b - a, "interval_s": interval_s,
            "busy_cores": round((b - a) / hz / interval_s, 2)}


class CotenantSampler:
    """Samples a co-tenant DURING a timed request (start before it, stop after it).

    `own_load_in_flight` is a callable the harness supplies (its own LLM thread alive?); the
    `generating` verdict is derived from it OR measured busy cores >= `busy_threshold`, and the
    basis is recorded so the verdict is never an unexplained boolean.
    """

    def __init__(self, port: Optional[int], *, own_load_in_flight: Callable[[], bool] = lambda: False,
                 interval_s: float = 0.25, busy_threshold: float = 1.0):
        self.port, self.own, self.interval_s, self.busy_threshold = port, own_load_in_flight, interval_s, busy_threshold
        self.samples: list = []
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    def _run(self):
        pid = pid_listening_on(self.port) if self.port else None
        while not self._stop.is_set():
            s = sample_process(pid, interval_s=self.interval_s)
            s["own_load_in_flight"] = bool(self.own())
            self.samples.append(s)

    def start(self):
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._th:
            self._th.join()
        return self.summary()

    def summary(self) -> dict:
        resolved = [s for s in self.samples if s.get("resolved")]
        own = any(s.get("own_load_in_flight") for s in self.samples)
        busy = max((s.get("busy_cores") or 0.0) for s in resolved) if resolved else None
        return {"port": self.port, "samples": self.samples, "n_samples": len(self.samples),
                "generating": bool(own or (busy is not None and busy >= self.busy_threshold)),
                "generating_basis": {"own_load_in_flight": own, "max_busy_cores": busy,
                                     "busy_threshold": self.busy_threshold}}


# ------------------------------------------------------------------ rows
def _metrics_for(native: dict) -> list:
    kind = native.get("kind")
    if kind == "stt":
        return [("stt_rtf", native.get("rtf"))]
    if kind == "tts":
        return [("tts_rtf", native.get("rtf")), ("tts_first_packet_s", native.get("first_packet_s"))]
    if kind == "llm":
        tps = native.get("tok_s")
        if tps is None:
            tps = (native.get("llm_timings") or {}).get("predicted_per_second")
        return [("cotenant_llm_tok_s", tps)]
    raise CaptureRefused(f"native row kind {kind!r} is not stt/tts/llm")


def request_outcome(native: dict, *, cap_s: Optional[float], guard_aborted: bool) -> dict:
    timed_out = bool(native.get("timed_out_after_s")) or bool(native.get("capped")) \
        or bool((native.get("llm") or {}).get("capped_at_s"))
    status = native.get("status")
    if status is None and isinstance(native.get("llm"), dict):
        status = native["llm"].get("status")
    # Some live rows (tts_capped, llm_once) carry no HTTP status; for them completion is decided
    # by the cap/abort flags here and by a finite value in make_rows, never assumed.
    completed = status in (200, None) and not timed_out and not guard_aborted
    return {"cap_s": cap_s, "timed_out": timed_out, "guard_aborted": bool(guard_aborted),
            "status": status, "completed": completed}


def make_rows(native: dict, *, harness: str, run_id: str, row_index: int, binaries: dict,
              cores: str, threads: int, shim_nprocs: Optional[int], cotenant: dict,
              cap_s: Optional[float], guard_aborted: bool = False, mode: str = "",
              emitted_at: Optional[str] = None) -> list:
    """Project ONE native harness row into one capture row per metric it carries."""
    emitted_at = emitted_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    outcome = request_outcome(native, cap_s=cap_s, guard_aborted=guard_aborted)
    contended = "llm" in (mode or "") or bool(cotenant.get("generating"))
    out = []
    for metric, raw in _metrics_for(native):
        unit, direction = CAPTURE_METRICS[metric]
        ok = outcome["completed"] and isinstance(raw, (int, float)) and not isinstance(raw, bool) \
            and math.isfinite(float(raw)) and float(raw) >= 0
        value = float(raw) if ok else None
        row_outcome = dict(outcome, completed=bool(ok))
        mid = hashlib.sha256(f"{run_id}|{harness}|{row_index}|{metric}".encode()).hexdigest()[:24]
        row = {
            "schema": CAPTURE_SCHEMA, "producer": CAPTURE_PRODUCER, "harness": harness,
            "run_id": run_id, "row_index": row_index, "measurement_id": f"speechcpu:{mid}",
            "emitted_at": emitted_at, "date": emitted_at[:10],
            "metric": metric, "unit": unit, "metric_direction": direction, "value": value,
            "category": "CANDIDATE" if contended else "BASELINE",
            "reps": 1, "reps_basis": "scored" if ok else "attempted", "protocol_id": "",
            "binaries": binaries,
            "placement": {"cores": cores, "threads": threads, "shim_nprocs": shim_nprocs, "mode": mode},
            "cotenant": cotenant, "request": row_outcome, "native": native,
        }
        row["row_sha256"] = row_digest(row)
        problems = validate_row(row)
        if problems:
            raise CaptureRefused("; ".join(problems))
        out.append(row)
    return out


def validate_row(row: Any) -> list:
    """Shared by the writer (refuse to emit) and the epyc-root adapter (refuse to project)."""
    if not isinstance(row, dict):
        return ["row is not an object"]
    missing = [k for k in REQUIRED_KEYS if k not in row]
    if missing:
        return [f"missing keys: {', '.join(missing)}"]
    p = []
    if row["schema"] != CAPTURE_SCHEMA:
        p.append(f"schema is {row['schema']!r}, not {CAPTURE_SCHEMA!r}")
    if row["harness"] not in HARNESSES:
        p.append(f"harness {row['harness']!r} unknown")
    if row["metric"] not in CAPTURE_METRICS:
        p.append(f"metric {row['metric']!r} is not one this producer emits")
    elif (row["unit"], row["metric_direction"]) != CAPTURE_METRICS[row["metric"]]:
        p.append(f"unit/direction for {row['metric']} must be {CAPTURE_METRICS[row['metric']]}")
    if row["category"] not in ("BASELINE", "CANDIDATE"):
        p.append("category must be BASELINE (solo) or CANDIDATE (contended)")
    req = row["request"] if isinstance(row["request"], dict) else {}
    if not {"cap_s", "timed_out", "guard_aborted", "completed"} <= set(req):
        p.append("request must record cap_s, timed_out, guard_aborted and completed")
    v = row["value"]
    if req.get("completed"):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            p.append("a completed row needs a finite value")
        if req.get("timed_out") or req.get("guard_aborted"):
            p.append("a timed-out or guard-aborted row cannot be completed")
    elif v is not None:
        p.append("a capped/aborted/failed row must carry value null, never a number")
    bins = row["binaries"] if isinstance(row["binaries"], dict) else {}
    if not bins or any(not isinstance(b, dict) or not _HEX64.match(str(b.get("sha256", ""))) for b in bins.values()):
        p.append("binaries must map each serving binary to a measured sha256")
    pl = row["placement"] if isinstance(row["placement"], dict) else {}
    if not pl.get("cores") or not isinstance(pl.get("threads"), int) or pl.get("threads", 0) < 1:
        p.append("placement must carry the exact core list and thread count (layout is part of the measurand)")
    if row["metric"].startswith("tts_") and not isinstance(pl.get("shim_nprocs"), int):
        p.append("TTS rows must record SHIM_NPROCS (the shim is the only TTS thread control)")
    ct = row["cotenant"] if isinstance(row["cotenant"], dict) else {}
    nat = row["native"] if isinstance(row["native"], dict) else {}
    t0, t1 = nat.get("t_start_epoch"), nat.get("t_end_epoch")
    samples = ct.get("samples") or []
    if not samples or "generating" not in ct:
        p.append("co-tenant state must be sampled, not assumed (no samples / no generating verdict)")
    elif isinstance(t0, (int, float)) and isinstance(t1, (int, float)):
        if not any(t0 <= s.get("t", -1) <= t1 for s in samples):
            p.append("no co-tenant sample falls inside the row's window; a sample after the row is not evidence")
    else:
        p.append("native row must carry t_start_epoch/t_end_epoch so the co-tenant window is checkable")
    if not _HEX64.match(str(row.get("row_sha256", ""))) or row["row_sha256"] != row_digest(row):
        p.append("row_sha256 does not match the row body")
    return p


def open_capture_sink(capture_dir: str):
    """Append-mode sink in a NEW directory; the 2026-09-24 raw/ is hashed evidence and refused."""
    d = Path(capture_dir).resolve()
    if d == RETROSPECTIVE_RAW or RETROSPECTIVE_RAW in d.parents:
        raise CaptureRefused("refusing to write capture rows into the 2026-09-24 retrospective raw/ "
                             "(SHA256SUMS-bound evidence); use a new dated directory")
    d.mkdir(parents=True, exist_ok=True)
    return (d / "speech_cpu_capture.jsonl").open("a", encoding="utf-8")


def write_rows(sink, rows: list) -> None:
    for r in rows:
        problems = validate_row(r)
        if problems:
            raise CaptureRefused("; ".join(problems))
        sink.write(canonical(r) + "\n")
    sink.flush()


# ------------------------------------------------------------------ harness instrumentation (opt-in)
class HarnessCapture:
    """Instruments a harness's request functions and writes capture rows beside its legacy rows.

    `wrap(fn)` brackets every request with a CotenantSampler and stamps epoch times on the native
    result; `emit(native, ...)` queues the capture rows; `flush(guard_aborted=...)` writes them once
    the harness's guard has decided, so the rows of the repetition that tripped the guard say so.
    """

    def __init__(self, harness: str, capture_dir: str, *, cotenant_port: Optional[int],
                 binaries: dict, run_id: Optional[str] = None, sample_interval_s: float = 0.25):
        if harness not in HARNESSES:
            raise CaptureRefused(f"unknown harness {harness!r}")
        self.harness, self.cotenant_port, self.sample_interval_s = harness, cotenant_port, sample_interval_s
        self.binary_paths = dict(binaries)
        self._binaries: Optional[dict] = None
        self.run_id = run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
        self.capture_dir = capture_dir
        self.sink = open_capture_sink(capture_dir)
        self.own_in_flight = 0
        self._lock = threading.Lock()
        self._pending: list = []
        self._row_index = 0

    @property
    def binaries(self) -> dict:
        if self._binaries is None:
            self._binaries = {k: binary_digest(p) for k, p in self.binary_paths.items()}
        return self._binaries

    def wrap(self, fn: Callable, *, own_load: bool = False) -> Callable:
        def wrapped(*a, **kw):
            if own_load:
                with self._lock:
                    self.own_in_flight += 1
            sampler = CotenantSampler(self.cotenant_port, own_load_in_flight=lambda: self.own_in_flight > 0,
                                      interval_s=self.sample_interval_s).start()
            t0 = time.time()
            try:
                res = fn(*a, **kw)
            finally:
                t1 = time.time()
                summary = sampler.stop()
                if own_load:
                    with self._lock:
                        self.own_in_flight -= 1
            target = res if isinstance(res, dict) else (a[0] if a and isinstance(a[0], dict) else None)
            if isinstance(target, dict):
                key = a[1] if len(a) > 1 and isinstance(a[1], str) and a[1] in target else None
                d = target[key] if key else target
                d["t_start_epoch"], d["t_end_epoch"] = t0, t1
                d["_cotenant"] = summary
            return res
        wrapped.__wrapped__ = fn
        return wrapped

    def emit(self, native: dict, *, cores: str, threads: int, shim_nprocs: Optional[int],
             cap_s: Optional[float], mode: str = "") -> None:
        native = dict(native)
        cot = native.pop("_cotenant", None)
        if isinstance(native.get("llm"), dict):  # speech_cpu_bench llm rows nest the timing dict
            inner = dict(native["llm"])
            cot = cot or inner.pop("_cotenant", None)
            native["llm"] = inner
            for k in ("t_start_epoch", "t_end_epoch"):
                if k in inner:
                    native[k] = inner[k]
        with self._lock:
            self._pending.append((self._row_index, native, cot or {}, cores, threads, shim_nprocs, cap_s, mode))
            self._row_index += 1

    def flush(self, *, guard_aborted: bool = False) -> int:
        with self._lock:
            pending, self._pending = self._pending, []
        rows = []
        for idx, native, cot, cores, threads, shim, cap_s, mode in pending:
            rows += make_rows(native, harness=self.harness, run_id=self.run_id, row_index=idx,
                              binaries=self.binaries, cores=cores, threads=threads, shim_nprocs=shim,
                              cotenant=cot, cap_s=cap_s, guard_aborted=guard_aborted, mode=mode)
        write_rows(self.sink, rows)
        return len(rows)


def strip_private(native: dict) -> dict:
    """The legacy row must stay byte-compatible: drop the capture-only `_cotenant` key."""
    out = {k: v for k, v in native.items() if k != "_cotenant"}
    if isinstance(out.get("llm"), dict):
        out["llm"] = {k: v for k, v in out["llm"].items() if k != "_cotenant"}
    return out


def server_placement(port: int) -> dict:
    """Placement of a LIVE server read from /proc (never assumed): exe, core list, -t, SHIM_NPROCS."""
    pid = pid_listening_on(port)
    if pid is None:
        raise CaptureRefused(f"no process resolvable on :{port}; placement cannot be captured")
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
        s = sample_process(pid, interval_s=0.0)
        env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError as exc:
        raise CaptureRefused(f"placement of :{port} pid {pid} unreadable: {exc}") from exc
    shim = next((int(e.split(b"=", 1)[1]) for e in env if e.startswith(b"SHIM_NPROCS=")), None)
    threads = int(s["threads_flag"]) if str(s.get("threads_flag") or "").isdigit() else (
        shim // 2 if shim else None)
    return {"pid": pid, "exe": exe, "cores": s.get("cpus_allowed_list"), "threads": threads,
            "shim_nprocs": shim, "threads_basis": "argv -t" if s.get("threads_flag") else
            ("SHIM_NPROCS/2" if shim else "unresolved")}
