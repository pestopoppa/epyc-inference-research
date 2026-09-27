#!/usr/bin/env python3
"""Belief-kernel write side for the UFH-12 embedder placement gate (VB-UFH12-PLACEMENT).

``embedder_placement_gate.py`` writes one JSON record per run (``--out``). This module writes the
belief sidecar next to it -- ``<record-stem>.belief_measurements.jsonl`` -- with one self-hashed row
per (gate x port x metric), and nothing else. It is imported by the gate and by any G3 speech driver;
the epyc-root adapter ``scripts/vidya/adapters/embedder_placement_gate.py`` loads THIS file (pinned by
sha256) and calls :func:`validate_row`, so the writer and the reader share one definition.

WHAT A ROW CARRIES (:data:`ROW_FIELDS`): the metric, its unit and direction, the value, the raw
samples the value re-derives from (:func:`derive`), the run label as category, the record file's
sha256 (the attestation), and ``extra.provenance`` -- identical on every row of a run:

    instrument      gate + capture file sha256, the method (ABA pairs, n_predict, load warm-up,
                    G2 window, prompt/text digests)
    load            load_mode (raw | scheduler), per-port and total in-flight, pool ports,
                    scheduler policy digest (scheduler mode)
    window          started_at / finished_at, plus per-gate windows
    topology_hash   the contention-matrix topology fingerprint (preflight_gate)
    processes       per port, snapshotted at window START and END and required equal: pid, argv
                    (+ sha256), Cpus_allowed_list (union over threads), exe path + sha256 (the kernel-store binary),
                    LD_LIBRARY_PATH, the libggml objects actually mapped
    frontdoor_props per frontdoor port: /props build_info and model_path
    placement_digest sha256 over the per-port argv/cpuset/exe/ggml facts

The A/A noise floor rides on every G1 row, re-derived from the Q samples; a G1 ratio is never
emitted without it.

CLAIMS ARE SEPARATE PER GATE. G1 (frontdoor decode, S/Q ratio and idle tok/s, per port), G2 (pool
scaling, per run) and G3 (speech non-regression, per run) never share a row. G0 (post idle vs pre
idle) is a cross-run comparison and is NOT emitted: its inputs are the G1 ``g1_q_decode_tps_median``
rows of the two runs.

REFUSALS (:class:`CaptureError`, nothing written): an unknown record schema or label; a run that
started before :data:`HOOK_SINCE` (the 2026-09-26 Phase-0 files are retrospective and never
backfilled); a capture more than :data:`MAX_EMIT_LAG_S` after the run finished; a process identity,
cpuset or binary that changed between the window's start and end snapshots; a port with no process
behind it; an empty topology hash; a G3 sample with a non-200 status.

NO GRADE IS WRITTEN. ``protocol_id`` is empty -- no protocol for this gate is codified under
``measurement/protocols/`` -- so ``claim_tuple.grade()`` caps every row at an observation. The
producer's own PASS/ROLLBACK verdict is not projected.

BEHAVIOUR-NEUTRAL. Snapshots read ``/proc`` and ``GET /props`` strictly before the first and after the
last timed sample; nothing here runs inside a measurement window, starts or signals a process.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CAPTURE_SCHEMA = "epyc.vidya.embedder_placement_capture.v1"
RECORD_SCHEMA = "epyc.embedder_placement_gate.v1"
G3_RECORD_SCHEMA = "epyc.embedder_placement_g3.v1"
SOURCE_KIND = "embedder-placement-gate"
SIDECAR_SUFFIX = ".belief_measurements.jsonl"
#: The hook landed 2026-09-27; every earlier record is retrospective (never backfilled).
HOOK_SINCE = "2026-09-27T00:00:00Z"
#: The hook runs at report time. A capture later than this after ``finished_at`` is a backfill.
MAX_EMIT_LAG_S = 3600
#: arm-baseline / arm-candidate (2026-09-27): the two sides of a pre-registered UFH-12 arm ABA
#: (A0 cap 0, A3 embedder OpenMP env); the arm's name, policy overrides and embedder env readback
#: ride in the record's params, hence in ``provenance.instrument.method``.
LABELS = {"pre": "BASELINE", "post": "CANDIDATE", "post-cap": "CANDIDATE",
          "arm-baseline": "BASELINE", "arm-candidate": "CANDIDATE"}
LOAD_MODES = ("raw", "scheduler")

#: metric -> (gate, unit, direction, per_port)
METRICS: dict[str, tuple[str, str, str, bool]] = {
    "g1_s_over_q_median": ("G1", "ratio of decode tok/s (saturated / idle)", "higher_better", True),
    "g1_q_decode_tps_median": ("G1", "decode tok/s (timings.predicted_per_second)", "higher_better", True),
    "g1_s_embed_texts_per_s_median": ("G1", "embedding texts/s", "higher_better", True),
    "g2_scaling_ratio": ("G2", "ratio of embedding texts/s (whole pool / one port)", "higher_better", False),
    "g2_one_port_texts_per_s": ("G2", "embedding texts/s", "higher_better", False),
    "g2_whole_pool_texts_per_s": ("G2", "embedding texts/s", "higher_better", False),
    "g3_saturated_stt_rtf_median": ("G3", "STT real-time factor", "lower_better", False),
    "g3_saturated_stt_rtf_worst": ("G3", "STT real-time factor", "lower_better", False),
    "g3_saturated_tts_first_packet_s_median": ("G3", "TTS first packet s", "lower_better", False),
    "g3_saturated_tts_first_packet_s_worst": ("G3", "TTS first packet s", "lower_better", False),
}

ROW_FIELDS = (
    "schema", "source_kind", "run_id", "producer", "emitted_at", "date", "measurement_id",
    "metric", "gate", "port", "value", "unit", "metric_direction", "category", "claim",
    "protocol_id", "reps", "reps_basis", "samples", "record_file", "record_sha256", "extra",
    "row_sha256",
)
EXTRA_FIELDS = ("locator", "label", "load_mode", "instrument_class", "provenance",
                "provenance_sha256")
PROVENANCE_FIELDS = ("instrument", "load", "window", "topology_hash", "processes",
                     "frontdoor_props", "placement_digest")
PROCESS_FIELDS = ("pid", "argv", "argv_sha256", "cpus_allowed_list", "exe", "exe_sha256",
                  "ld_library_path", "ggml_libs")


class CaptureError(RuntimeError):
    """The capture refuses; nothing is written."""


# ------------------------------------------------------------------ hashing


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_digest(row: Mapping[str, Any]) -> str:
    return content_hash({k: v for k, v in row.items() if k != "row_sha256"})


def measurement_identity(*, run_id: str, gate: str, port: int | None, metric: str,
                         record_sha256: str) -> str:
    return "embpl-" + content_hash([run_id, gate, port, metric, record_sha256])[:32]


def locator(run_id: str, gate: str, port: int | None, metric: str) -> str:
    return f"embpl:{run_id}:{gate}:{'pool' if port is None else port}:{metric}"


# ------------------------------------------------------------------ derivation (shared)


def _num_list(value: Any, *, min_len: int = 1) -> bool:
    return (isinstance(value, list) and len(value) >= min_len
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                    for x in value))


def _ratios(q: list[float], s: list[float]) -> list[float]:
    """Exactly the gate's G1 arithmetic: each S against the mean of its flanking Qs."""
    return [si / ((qa + qb) / 2) for si, qa, qb in zip(s, q, q[1:])]


def aa_floor(q: list[float]) -> float:
    """The gate's A/A noise floor: median relative change between consecutive Q samples."""
    return statistics.median(abs(b - a) / a for a, b in zip(q, q[1:]))


def derive(metric: str, samples: Mapping[str, Any]) -> float:
    """Re-derive a row's value from its samples. Raises ValueError on a malformed sample set."""
    if metric in ("g1_s_over_q_median", "g1_q_decode_tps_median"):
        q, s = samples.get("q_samples"), samples.get("s_samples")
        if not (_num_list(q, min_len=2) and _num_list(s) and len(q) == len(s) + 1):
            raise ValueError("G1 needs ABA samples: len(q) == len(s) + 1 >= 2")
        if any(x <= 0 for x in q):
            raise ValueError("G1 Q samples must be positive")
        if metric == "g1_q_decode_tps_median":
            return float(statistics.median(q))
        return float(statistics.median(_ratios(q, s)))
    if metric == "g1_s_embed_texts_per_s_median":
        tps = samples.get("texts_per_s")
        if not _num_list(tps):
            raise ValueError("G1 embedding throughput samples missing")
        return float(statistics.median(tps))
    if metric.startswith("g2_"):
        arms = {}
        for arm in ("one_port", "whole_pool"):
            got = samples.get(arm)
            if not (isinstance(got, Mapping) and isinstance(got.get("texts"), int)
                    and _num_list([got.get("seconds")]) and got["seconds"] > 0 and got["texts"] > 0):
                raise ValueError(f"G2 {arm} needs positive texts and seconds")
            arms[arm] = got["texts"] / got["seconds"]
        if metric == "g2_scaling_ratio":
            return float(arms["whole_pool"] / arms["one_port"])
        return float(arms[metric[len("g2_"):-len("_texts_per_s")]])
    if metric.startswith("g3_saturated_"):
        key = "stt_rtf" if "_stt_rtf_" in metric else "tts_first_packet_s"
        vals = samples.get("saturated", {}).get(key) if isinstance(samples.get("saturated"), Mapping) else None
        if not _num_list(vals):
            raise ValueError(f"G3 saturated {key} samples missing")
        return float(max(vals) if metric.endswith("_worst") else statistics.median(vals))
    raise ValueError(f"unknown metric {metric!r}")


# ------------------------------------------------------------------ validation (shared with the reader)


def _utc(text: Any) -> datetime | None:
    if not isinstance(text, str) or not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate_provenance(prov: Any, row: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if not isinstance(prov, Mapping) or set(prov) != set(PROVENANCE_FIELDS):
        return ["provenance must carry exactly " + ", ".join(PROVENANCE_FIELDS)]
    inst = prov["instrument"]
    if not (isinstance(inst, Mapping) and _hex64(inst.get("gate_sha256"))
            and _hex64(inst.get("capture_sha256")) and isinstance(inst.get("method"), Mapping)):
        problems.append("instrument must name the gate and capture sha256 and the method")
    load = prov["load"]
    if not (isinstance(load, Mapping) and load.get("load_mode") in LOAD_MODES + ("g3",)
            and isinstance(load.get("per_port_in_flight"), int) and load["per_port_in_flight"] > 0
            and isinstance(load.get("pool_ports"), list) and load["pool_ports"]
            and load.get("in_flight_total") == load["per_port_in_flight"] * len(load["pool_ports"])):
        problems.append("load shape (mode, per-port and total in-flight, pool ports) incomplete")
    window = prov["window"]
    start = _utc(window.get("started_at")) if isinstance(window, Mapping) else None
    end = _utc(window.get("finished_at")) if isinstance(window, Mapping) else None
    if start is None or end is None or end < start:
        problems.append("window needs UTC started_at <= finished_at")
    if not (isinstance(prov["topology_hash"], str) and prov["topology_hash"].strip()):
        problems.append("topology_hash is empty")
    procs = prov["processes"]
    if not isinstance(procs, Mapping) or not procs:
        problems.append("processes snapshot missing")
    else:
        for port, facts in procs.items():
            if not (isinstance(facts, Mapping) and set(facts) == set(PROCESS_FIELDS)
                    and isinstance(facts["pid"], int) and _hex64(facts["argv_sha256"])
                    and _hex64(facts["exe_sha256"]) and isinstance(facts["cpus_allowed_list"], str)
                    and facts["cpus_allowed_list"]
                    and facts["argv_sha256"] == content_hash(facts["argv"])):
                problems.append(f"process facts for :{port} incomplete or argv digest wrong")
        pool = {str(p) for p in load.get("pool_ports", [])} if isinstance(load, Mapping) else set()
        if not pool <= set(procs):
            problems.append("a pool port has no process snapshot")
        if prov["placement_digest"] != placement_digest(procs):
            problems.append("placement_digest does not re-derive")
    props = prov["frontdoor_props"]
    if not isinstance(props, Mapping):
        problems.append("frontdoor_props must be a mapping")
    elif row.get("gate") == "G1":
        facts = props.get(str(row.get("port")))
        if not (isinstance(facts, Mapping) and isinstance(facts.get("build_info"), str)
                and facts["build_info"]):
            problems.append(f"G1 :{row.get('port')} lacks frontdoor build_info")
        if str(row.get("port")) not in (procs if isinstance(procs, Mapping) else {}):
            problems.append(f"G1 :{row.get('port')} lacks a frontdoor process snapshot")
    return problems


def validate_row(row: Any) -> list[str]:
    """Every reason ``row`` is not a well-formed capture row. Empty list = valid."""
    if not isinstance(row, Mapping):
        return ["row is not an object"]
    problems: list[str] = []
    if set(row) != set(ROW_FIELDS):
        return [f"row fields differ from the capture contract: {sorted(set(row) ^ set(ROW_FIELDS))}"]
    if row["schema"] != CAPTURE_SCHEMA or row["source_kind"] != SOURCE_KIND:
        problems.append("schema or source_kind is foreign")
    if row["row_sha256"] != row_digest(row):
        problems.append("row_sha256 does not re-derive")
    metric = row["metric"]
    if metric not in METRICS:
        return problems + [f"unknown metric {metric!r}"]
    gate, unit, direction, per_port = METRICS[metric]
    if (row["gate"], row["unit"], row["metric_direction"]) != (gate, unit, direction):
        problems.append("gate/unit/direction differ from the metric's declaration")
    port = row["port"]
    if (per_port and not (isinstance(port, int) and not isinstance(port, bool))) \
            or (not per_port and port is not None):
        problems.append("port must be an int for per-port metrics and null otherwise")
    extra = row["extra"]
    if not isinstance(extra, Mapping) or set(extra) != set(EXTRA_FIELDS):
        return problems + ["extra must carry exactly " + ", ".join(EXTRA_FIELDS)]
    if extra["label"] not in LABELS or row["category"] != LABELS[extra["label"]]:
        problems.append("label/category mismatch")
    if extra["instrument_class"] != "live-http":
        problems.append("instrument_class must be live-http")
    if extra["locator"] != locator(row["run_id"], gate, port, metric):
        problems.append("locator does not bind run, gate, port and metric")
    if not _hex64(row["record_sha256"]):
        problems.append("record_sha256 must be a sha256")
    if row["measurement_id"] != measurement_identity(
            run_id=row["run_id"], gate=gate, port=port, metric=metric,
            record_sha256=row["record_sha256"]):
        problems.append("measurement_id does not re-derive")
    rf = row["record_file"]
    if not isinstance(rf, str) or not rf or "/" in rf or rf.startswith("."):
        problems.append("record_file must be a bare file name beside the sidecar")
    if row["protocol_id"] != "":
        problems.append("no protocol is codified for this gate; protocol_id must be empty")
    if not isinstance(row["samples"], Mapping):
        return problems + ["samples must be an object"]
    try:
        value = derive(metric, row["samples"])
    except ValueError as exc:
        return problems + [f"samples: {exc}"]
    if not isinstance(row["value"], (int, float)) or not math.isclose(row["value"], value, rel_tol=1e-12):
        problems.append("value does not re-derive from samples")
    if gate == "G1" and metric != "g1_s_embed_texts_per_s_median":
        floor = row["samples"].get("aa_noise_floor_rel_median")
        if not (isinstance(floor, (int, float)) and math.isclose(
                floor, aa_floor(row["samples"]["q_samples"]), rel_tol=1e-12, abs_tol=1e-15)):
            problems.append("G1 row lacks its re-derived A/A noise floor")
    if gate == "G3":
        for arm in ("quiet", "saturated"):
            got = row["samples"].get(arm)
            if not (isinstance(got, Mapping) and got.get("stt_status")
                    and all(s == 200 for s in got["stt_status"] + got.get("tts_status", [0]))):
                problems.append(f"G3 {arm} arm has a non-200 or missing status")
    if not (isinstance(row["reps"], int) and row["reps"] >= 1 and isinstance(row["reps_basis"], str)
            and row["reps_basis"]):
        problems.append("reps must be a positive int with a basis")
    if not (isinstance(row["claim"], str) and row["claim"].strip()):
        problems.append("claim text is empty")
    emitted = _utc(row["emitted_at"])
    if emitted is None or row["date"] != row["emitted_at"][:10]:
        problems.append("emitted_at/date malformed")
    if extra.get("provenance_sha256") != content_hash(extra.get("provenance")):
        problems.append("provenance_sha256 does not re-derive")
    problems += _validate_provenance(extra["provenance"], row)
    prov = extra["provenance"]
    if isinstance(prov, Mapping) and isinstance(prov.get("load"), Mapping) \
            and isinstance(prov.get("window"), Mapping):
        if prov["load"].get("load_mode") != extra["load_mode"]:
            problems.append("load_mode differs from provenance")
        started = _utc(prov["window"].get("started_at"))
        if started is not None and started < _utc(HOOK_SINCE):
            problems.append("run started before the hook existed (retrospective)")
    return problems


# ------------------------------------------------------------------ live snapshot (/proc + /props)


def _listen_inodes(ports: Sequence[int]) -> dict[int, str]:
    wanted, found = set(ports), {}
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10 or parts[3] != "0A":  # LISTEN
                continue
            port = int(parts[1].rsplit(":", 1)[1], 16)
            if port in wanted:
                found.setdefault(port, parts[9])
    return found


def _pids_for_ports(ports: Sequence[int]) -> dict[int, int]:
    inodes = _listen_inodes(ports)
    by_socket = {f"socket:[{ino}]": port for port, ino in inodes.items()}
    out: dict[int, int] = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or len(out) == len(by_socket):
            continue
        try:
            fds = os.listdir(proc / "fd")
        except OSError:
            continue
        for fd in fds:
            try:
                port = by_socket.get(os.readlink(proc / "fd" / fd))
            except OSError:
                continue
            if port is not None and port not in out:
                out[port] = int(proc.name)
    return out


_EXE_CACHE: dict[tuple[str, int, int], str] = {}


def _exe_sha256(proc_exe: Path, exe: str) -> str:
    """Hash the mapped binary through /proc/<pid>/exe, so a replaced file on disk cannot stand in."""
    st = os.stat(proc_exe)
    key = (exe, st.st_size, st.st_mtime_ns)
    if key not in _EXE_CACHE:
        _EXE_CACHE[key] = file_sha256(proc_exe)
    return _EXE_CACHE[key]


def _parse_cpus(text: str) -> set[int]:
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    return out


def _format_cpus(cpus: set[int]) -> str:
    runs, ordered = [], sorted(cpus)
    for cpu in ordered:
        if runs and cpu == runs[-1][1] + 1:
            runs[-1][1] = cpu
        else:
            runs.append([cpu, cpu])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _thread_cpus(root: Path) -> str:
    """Union of Cpus_allowed_list over every thread. The main thread's own mask is whatever the
    OpenMP runtime bound it to, not the placement the launcher declared."""
    union: set[int] = set()
    for task in (root / "task").iterdir():
        try:
            status = (task / "status").read_text().splitlines()
        except OSError:
            continue
        for line in status:
            if line.startswith("Cpus_allowed_list:"):
                union |= _parse_cpus(line.split(":", 1)[1])
    return _format_cpus(union)


def process_facts(pid: int) -> dict[str, Any]:
    root = Path(f"/proc/{pid}")
    argv = [a.decode(errors="replace") for a in (root / "cmdline").read_bytes().split(b"\0") if a]
    cpus = _thread_cpus(root)
    exe = os.readlink(root / "exe")
    try:
        env = dict(kv.split("=", 1) for kv in (root / "environ").read_bytes().decode(
            errors="replace").split("\0") if "=" in kv)
    except OSError:
        env = {}
    try:
        maps = (root / "maps").read_text().splitlines()
    except OSError:
        maps = []
    ggml = sorted({line.split()[-1] for line in maps if "libggml" in line.split()[-1]})
    return {"pid": pid, "argv": argv, "argv_sha256": content_hash(argv), "cpus_allowed_list": cpus,
            "exe": exe, "exe_sha256": _exe_sha256(root / "exe", exe), "ld_library_path": env.get("LD_LIBRARY_PATH"),
            "ggml_libs": ggml}


def frontdoor_props(port: int) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=10) as resp:  # noqa: S310
            body = json.loads(resp.read())
    except (OSError, ValueError):
        return {}
    return {"build_info": str(body.get("build_info") or ""),
            "model_path": str(body.get("model_path") or "")}


def live_topology_hash() -> str:
    repo = str(Path(__file__).resolve().parents[2])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from scripts.server.preflight_gate import _live_topology_hash
    except Exception:  # noqa: BLE001 — an empty hash is refused by the writer, never filled
        return ""
    return _live_topology_hash() or ""


def snapshot(process_ports: Sequence[int], props_ports: Sequence[int]) -> dict[str, Any]:
    """One read-only snapshot of every process behind ``process_ports`` plus /props."""
    pids = _pids_for_ports(process_ports)
    procs: dict[str, Any] = {}
    for port in process_ports:
        pid = pids.get(port)
        try:
            procs[str(port)] = process_facts(pid) if pid is not None else None
        except OSError:
            procs[str(port)] = None
    return {"processes": procs,
            "frontdoor_props": {str(p): frontdoor_props(p) for p in props_ports},
            "topology_hash": live_topology_hash()}


def placement_digest(processes: Mapping[str, Any]) -> str:
    return content_hash({port: {k: f[k] for k in ("argv_sha256", "cpus_allowed_list", "exe_sha256",
                                                   "ggml_libs")}
                         for port, f in processes.items() if isinstance(f, Mapping)})


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class CaptureWindow:
    """Snapshot the serving identity before the first and after the last timed sample.

    ``process_ports`` are every port whose process is part of the measurand (embedders, frontdoors,
    speech servers); ``props_ports`` are the llama-server frontdoors whose /props build_info is
    recorded. ``snapshot_fn`` is injectable for tests only.
    """

    def __init__(self, process_ports: Sequence[int], props_ports: Sequence[int] = (), *,
                 snapshot_fn: Callable[[Sequence[int], Sequence[int]], dict] | None = None) -> None:
        self.process_ports = list(process_ports)
        self.props_ports = list(props_ports)
        self._snap = snapshot_fn or snapshot
        self.start: dict | None = None
        self.end: dict | None = None
        self.gates: dict[str, dict[str, str]] = {}

    def begin(self) -> None:
        self.start = self._snap(self.process_ports, self.props_ports)

    def finish(self) -> None:
        self.end = self._snap(self.process_ports, self.props_ports)

    def mark(self, gate: str, started_at: str, finished_at: str) -> None:
        self.gates[gate] = {"started_at": started_at, "finished_at": finished_at}

    def stable_snapshot(self) -> dict:
        if self.start is None or self.end is None:
            raise CaptureError("capture window was not snapshotted at both ends")
        missing = sorted(p for p, f in self.start["processes"].items() if f is None)
        if missing:
            raise CaptureError(f"no process found behind ports {missing}")
        if self.start != self.end:
            changed = sorted(p for p in self.start["processes"]
                             if self.start["processes"][p] != self.end["processes"].get(p))
            raise CaptureError("serving identity changed during the window "
                               f"(processes {changed or 'unchanged'}; props/topology may differ)")
        if not self.start["topology_hash"]:
            raise CaptureError("topology hash unavailable")
        return self.start


# ------------------------------------------------------------------ the writer


def _claim(metric: str, port: int | None, value: float, samples: Mapping[str, Any], label: str,
           load_mode: str, in_flight: int) -> str:
    where = f":{port}" if port is not None else "pool"
    head = f"UFH-12 embedder placement ({label}, load {load_mode}, {in_flight} embeddings in flight)"
    if metric == "g1_s_over_q_median":
        return (f"{head}: frontdoor {where} decode S/Q median {value:.4f} over "
                f"{len(samples['s_samples'])} ABA pairs; A/A noise floor "
                f"{samples['aa_noise_floor_rel_median']:.4%} (same port, same unit).")
    if metric == "g1_q_decode_tps_median":
        return (f"{head}: frontdoor {where} idle-pool decode median {value:.2f} tok/s over "
                f"{len(samples['q_samples'])} samples; A/A noise floor "
                f"{samples['aa_noise_floor_rel_median']:.4%}.")
    if metric == "g1_s_embed_texts_per_s_median":
        return f"{head}: embedding throughput during {where} decode samples, median {value:.2f} texts/s."
    if metric == "g2_scaling_ratio":
        return f"{head}: whole-pool / one-port embedding throughput ratio {value:.3f}, frontdoor idle."
    if metric.startswith("g2_"):
        return f"{head}: {metric[3:].replace('_', ' ')} {value:.2f}, frontdoor idle."
    stat_name = "worst" if metric.endswith("_worst") else "median"
    what = "STT RTF" if "_stt_rtf_" in metric else "TTS first packet (s)"
    return f"{head}: speech {what} {stat_name} {value:.4f} with the pool saturated, frontdoor idle."


def _rows_for_record(record: Mapping[str, Any]) -> list[tuple[str, int | None, str, dict, int, str]]:
    """Yield (metric, port, gate, samples, reps, reps_basis) for every claim in the record."""
    out = []
    if record.get("gate") == "G3":
        method = record.get("method", {})
        samples = {"quiet": record["quiet"], "saturated": record["saturated"],
                   "warmup_discarded": bool(method.get("warmup_discarded"))}
        n = len(record["saturated"].get("stt_rtf", []))
        for metric in METRICS:
            if metric.startswith("g3_"):
                out.append((metric, None, "G3", samples, max(n, 1), "saturated-arm requests"))
        return out
    g2 = record["g2_pool_scaling"]
    g2_samples = {arm: {"texts": g2[arm]["texts"], "seconds": g2[arm]["seconds"]}
                  for arm in ("one_port", "whole_pool")}
    for metric in ("g2_scaling_ratio", "g2_one_port_texts_per_s", "g2_whole_pool_texts_per_s"):
        out.append((metric, None, "G2", g2_samples, 1, "one timed window per arm"))
    for port_text, row in sorted(record["g1_frontdoor_decode"].items()):
        port = int(port_text)
        samples = {"q_samples": row["q_samples"], "s_samples": row["s_samples"],
                   "aa_noise_floor_rel_median": row["aa_noise_floor_rel_median"]}
        pairs = len(row["s_samples"])
        out.append(("g1_s_over_q_median", port, "G1", samples, pairs, "ABA pairs scored"))
        out.append(("g1_q_decode_tps_median", port, "G1", samples, len(row["q_samples"]),
                    "idle-pool decode samples"))
        loads = row.get("s_embedding_load")
        if loads:
            out.append(("g1_s_embed_texts_per_s_median", port, "G1",
                        {**samples, "texts_per_s": [x["texts_per_s"] for x in loads],
                         "counters_delta": [x.get("counters_delta", {}) for x in loads],
                         "deferred": [x.get("deferred", 0) for x in loads]},
                        len(loads), "saturated S windows"))
    return out


def build_rows(record: Mapping[str, Any], *, record_path: str | Path, window: CaptureWindow,
               producer: str, gate_path: str | Path, emitted_at: str | None = None) -> list[dict]:
    """Project one gate (or G3) record into validated capture rows. Raises CaptureError."""
    is_g3 = record.get("gate") == "G3"
    if record.get("schema") != (G3_RECORD_SCHEMA if is_g3 else RECORD_SCHEMA):
        raise CaptureError(f"unknown record schema {record.get('schema')!r}")
    label = record.get("label")
    if label not in LABELS:
        raise CaptureError(f"unknown run label {label!r}")
    started, finished = _utc(record.get("started_at")), _utc(record.get("finished_at"))
    if started is None or finished is None or finished < started:
        raise CaptureError("record needs UTC started_at <= finished_at")
    if started < _utc(HOOK_SINCE):
        raise CaptureError("record predates the capture hook; retrospective records are not backfilled")
    when = emitted_at or utc_now()
    emitted = _utc(when)
    if emitted is None:
        raise CaptureError("emitted_at must be a UTC timestamp")
    lag = (emitted - finished).total_seconds()
    if lag < 0 or lag > MAX_EMIT_LAG_S:
        raise CaptureError(f"capture lag {lag:.0f}s outside [0, {MAX_EMIT_LAG_S}]: never a backfill")
    snap = window.stable_snapshot()
    params = record.get("params")
    if not isinstance(params, Mapping):
        raise CaptureError("record lacks its params block")
    load_mode = "g3" if is_g3 else record.get("load_mode")
    per_port = params.get("per_port_in_flight")
    pool_ports = list(params.get("pool_ports") or [])
    record_path = Path(record_path)
    capture_path = Path(__file__).resolve()
    provenance = {
        "instrument": {"gate_sha256": file_sha256(gate_path), "gate_file": Path(gate_path).name,
                       "capture_sha256": file_sha256(capture_path),
                       "method": {k: v for k, v in params.items() if k not in ("pool_ports",)}},
        "load": {"load_mode": load_mode, "per_port_in_flight": per_port, "pool_ports": pool_ports,
                 "in_flight_total": (per_port or 0) * len(pool_ports),
                 "scheduler_policy_sha256": (content_hash(record["scheduler_policy"])
                                             if "scheduler_policy" in record else None)},
        "window": {"started_at": record["started_at"], "finished_at": record["finished_at"],
                   "gates": dict(window.gates)},
        "topology_hash": snap["topology_hash"],
        "processes": snap["processes"],
        "frontdoor_props": snap["frontdoor_props"],
        "placement_digest": placement_digest(snap["processes"]),
    }
    provenance = json.loads(json.dumps(provenance))
    record_sha = file_sha256(record_path)
    run_id = f"{label}@{record['started_at']}"
    arm = params.get("arm")
    claim_label = f"{label} {arm}" if isinstance(arm, str) and arm.strip() else label
    rows = []
    for metric, port, gate, samples, reps, basis in _rows_for_record(record):
        try:
            value = derive(metric, samples)
        except ValueError as exc:
            raise CaptureError(f"{metric}: {exc}") from exc
        unit, direction = METRICS[metric][1], METRICS[metric][2]
        row = {
            "schema": CAPTURE_SCHEMA, "source_kind": SOURCE_KIND, "run_id": run_id,
            "producer": producer, "emitted_at": when, "date": when[:10],
            "measurement_id": measurement_identity(run_id=run_id, gate=gate, port=port,
                                                   metric=metric, record_sha256=record_sha),
            "metric": metric, "gate": gate, "port": port, "value": value, "unit": unit,
            "metric_direction": direction, "category": LABELS[label],
            "claim": _claim(metric, port, value, samples, claim_label, load_mode,
                            provenance["load"]["in_flight_total"]),
            "protocol_id": "", "reps": reps, "reps_basis": basis,
            "samples": json.loads(json.dumps(samples)),
            "record_file": record_path.name, "record_sha256": record_sha,
            "extra": {"locator": locator(run_id, gate, port, metric), "label": label,
                      "load_mode": load_mode, "instrument_class": "live-http",
                      "provenance": provenance, "provenance_sha256": content_hash(provenance)},
        }
        row["row_sha256"] = row_digest(row)
        problems = validate_row(row)
        if problems:
            raise CaptureError(f"{metric} :{port}: refusing an invalid row: " + "; ".join(problems))
        rows.append(row)
    return rows


def sidecar_path(record_path: str | Path) -> Path:
    path = Path(record_path)
    return path.with_name(path.stem + SIDECAR_SUFFIX)


def write_belief_measurements(record_path: str | Path, record: Mapping[str, Any], *,
                              window: CaptureWindow, producer: str, gate_path: str | Path,
                              emitted_at: str | None = None) -> Path:
    """Write ``<record-stem>.belief_measurements.jsonl`` beside the record. All rows or none."""
    rows = build_rows(record, record_path=record_path, window=window, producer=producer,
                      gate_path=gate_path, emitted_at=emitted_at)
    out = sidecar_path(record_path)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    os.replace(tmp, out)
    return out


__all__ = [
    "CAPTURE_SCHEMA", "EXTRA_FIELDS", "G3_RECORD_SCHEMA", "HOOK_SINCE", "LABELS", "MAX_EMIT_LAG_S",
    "METRICS", "PROCESS_FIELDS", "PROVENANCE_FIELDS", "RECORD_SCHEMA", "ROW_FIELDS", "SIDECAR_SUFFIX",
    "SOURCE_KIND", "CaptureError", "CaptureWindow", "aa_floor", "build_rows", "content_hash",
    "derive", "file_sha256", "locator", "measurement_identity", "placement_digest", "row_digest",
    "sidecar_path", "snapshot", "validate_row", "write_belief_measurements",
]
