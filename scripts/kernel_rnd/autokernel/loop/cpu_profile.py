"""Owned, request-scoped CPU perf observations; no correctness or promotion authority.

The installed producer is a worker, never a grant provider. It can profile only
the actual serving child launched below itself. The sealed adapter keeps its
original independent-full-request label; direct loop profiles retain their actual
v1/v2 requests separately, without manufacturing sealed preparation authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Mapping

from ..evaluator import c3_epyc_tensor_capture as tensor_capture
from . import lifecycle_observation as lo, measurement_capture as mc
from . import native_model_preparation as mp, native_server_response as ns
from . import observation_binding as ob, planned_serving as ps
from . import resolved_recipe as rr, worker_lifecycle as wl

CONFIG_SCHEMA = "epyc.autokernel.cpu_profile_config.v1"
CAPTURE_SCHEMA = "epyc.autokernel.cpu_profile_capture.v1"
SOURCE_SCHEMA = "epyc.autokernel.cpu_profile_source.v1"
LOOP_PROFILE_SCHEMA = "epyc.autokernel.loop_cpu_profile.v1"
LOOP_CAPTURE_SCHEMA = "epyc.autokernel.loop_cpu_profile_capture.v1"
LOOP_MODE = "original_serving_requests_observation"
MODE = "independent_full_request_v1"
MECHANISM_ID = "owned-cpu-perf-v1"
CONFIG_ENV = "EPYC_AUTOKERNEL_CPU_PROFILE_CONFIG"
PHASES = ("warmup", "measurement")
EVENTS = ("instructions:u", "cache-references:u", "cache-misses:u", "task-clock")
MAX_CONFIG_BYTES = 256 * 1024
MAX_STDOUT_BYTES = 48 * 1024
MAX_LINE_BYTES = 16384
MAX_DIAGNOSTIC_BYTES = 65536
MAX_TARGET_TIDS = 4096
LIMITS = {"max_stage_seconds": 14400, "teardown_seconds": 120,
          "control_seconds": 30, "reduce_seconds": 600,
          "max_raw_file_bytes": 4 * 1024**3, "max_total_raw_bytes": 10 * 1024**3,
          "max_parser_bytes": 2 * 1024**3, "max_rows": 10_000_000,
          "max_symbols": 4096, "max_metadata_bytes": 16 * 1024**2}
LIMITATIONS = ("sampled-period totals are estimated user-cycle attribution, not exact CPU cost",
               "worker self attribution is not wall-time share or an optimization gain",
               "totals are not a comparable performance objective across windows, exposure or unknown sample loss",
               "full-request warmup/measurement only; setup/load are not profiled",
               "counter totals cover their own enable/disable window, not the exact request or sample window; no cross-window IPC",
               "no exact generated-token, MTP, correctness, contention or GPU warrant",
               "no ratified measurement protocol or opportunity is supplied")
LOOP_BUDGETS = {"max_stage_seconds": 1800, "teardown_seconds": 5,
                "control_seconds": 30, "reduce_seconds": 120,
                "max_raw_file_bytes": 128 * 1024**2, "max_total_raw_bytes": 512 * 1024**2,
                "max_parser_bytes": 96 * 1024**2, "max_rows": 1_000_000,
                "max_symbols": 4096, "max_metadata_bytes": 16 * 1024**2}


class CpuProfileRefused(ValueError):
    pass


class CpuProfileCleanupUncertain(CpuProfileRefused):
    """An owned profiling child was not proven terminal; do not continue a measurement."""


def _plain(value):
    return ob._plain(value)


def _closed(value, fields, label):
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise CpuProfileRefused(f"{label} fields differ")
    return dict(value)


def _same(left, right, label):
    if left != right:
        raise CpuProfileRefused(f"{label} differs")


def _number(value, label, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise CpuProfileRefused(f"{label} is not finite/nonnegative")
    return float(value)


def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_file(path, limit, *, retain):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise CpuProfileRefused("artifact path must be absolute without symlinks")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise CpuProfileRefused("artifact is not a bounded regular file")
        digest = hashlib.sha256()
        count = 0
        retained = []
        while True:
            part = os.read(fd, min(1024 * 1024, limit + 1 - count))
            if not part:
                break
            count += len(part)
            if count > limit:
                raise CpuProfileRefused("artifact grew past byte bound")
            digest.update(part)
            if retain:
                retained.append(part)
        after = os.fstat(fd)
        keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in keys):
            raise CpuProfileRefused("artifact changed during read")
        identity = {"path": str(path), "sha256": digest.hexdigest(), "size": count,
                "dev": before.st_dev, "ino": before.st_ino,
                "mtime_ns": before.st_mtime_ns, "ctime_ns": before.st_ctime_ns}
        return identity, b"".join(retained) if retain else None
    finally:
        os.close(fd)


def _file(path, limit):
    return _read_file(path, limit, retain=False)[0]


def _read(path, limit):
    return _read_file(path, limit, retain=True)[1]


def _reduce_artifact(reference, *, kind, arguments):
    """Hash and reduce the same bounded regular fd, never reopen an unchecked path."""
    path = Path(reference["path"])
    if not path.is_absolute() or path.resolve() != path:
        raise CpuProfileRefused("reduction path is not canonical")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > arguments["max_bytes"]:
            raise CpuProfileRefused("reduction input is not bounded regular data")
        digest = hashlib.sha256()
        count = 0
        while part := stream.read(min(1024 * 1024, arguments["max_bytes"] + 1 - count)):
            count += len(part)
            if count > arguments["max_bytes"]:
                raise CpuProfileRefused("reduction input grew beyond bound")
            digest.update(part)
        actual = {"path": str(path), "sha256": digest.hexdigest(), "size": count,
            "dev": before.st_dev, "ino": before.st_ino,
            "mtime_ns": before.st_mtime_ns, "ctime_ns": before.st_ctime_ns}
        _same(actual, reference, "reduction original bytes")
        stream.seek(0)
        if kind == "samples":
            result = reduce_perf_script(stream, **arguments)
        elif kind == "counters":
            result = reduce_perf_stat(stream, **arguments)
        else:
            raise CpuProfileRefused("unknown concrete reducer")
        after = os.fstat(stream.fileno())
        if any(getattr(before, key) != getattr(after, key) for key in
               ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
            raise CpuProfileRefused("reduction input changed during reduction")
        return result


def _process(pid):
    probe = lo.FilesystemProbe()
    budget = {"max_read_bytes": 1024 * 1024}
    before = probe.process_identity(pid, budget)
    raw_stat = _read_proc(Path(f"/proc/{pid}/stat"), 65536).decode()
    ppid = int(raw_stat[raw_stat.rfind(")") + 1:].split()[1])
    cgroup = probe._cgroup_path(pid, budget)
    path = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    container = {**lo._stat_identity(path), "path": cgroup}
    argv = _read_proc(Path(f"/proc/{pid}/cmdline"), 65536).split(b"\0")
    after = probe.process_identity(pid, budget)
    _same((before["pid"], before["start_ticks"]), (after["pid"], after["start_ticks"]),
          "process changed during identity read")
    return {"pid": before["pid"], "start_ticks": before["start_ticks"], "ppid": ppid,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "container": container, "argv": [x.decode() for x in argv if x],
            "exe": str(Path(f"/proc/{pid}/exe").resolve(strict=True))}


def _read_proc(path, limit):
    with open(path, "rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise CpuProfileRefused("process read exceeds bound")
    return value


def _stable_stat(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise CpuProfileRefused("model member path is not canonical")
    value = path.stat()
    if not stat.S_ISREG(value.st_mode):
        raise CpuProfileRefused("model member is not a regular file")
    return {name: getattr(value, name) for name in
            ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


def _identity_key(value):
    return value["pid"], value["start_ticks"], value["boot_id"], value["container"]


def source_identity():
    from . import serving
    root = Path(__file__).resolve().parents[4]
    roles = {"capture_init": CpuProfileCapture.__init__,
             "capture_initialize": CpuProfileCapture._initialize,
             "capture_direct": CpuProfileCapture.for_loop.__func__,
             "capture_attach": CpuProfileCapture.attach_target,
             "capture_begin": CpuProfileCapture.begin_round,
             "capture_end": CpuProfileCapture.end_round,
             "capture_retain": CpuProfileCapture.retain_responses,
             "capture_finish": CpuProfileCapture.finish,
             "capture_abort": CpuProfileCapture.abort,
             "capture_check": CpuProfileCapture._check_target,
             "capture_spawn": CpuProfileCapture._spawn,
             "capture_control": CpuProfileCapture._control,
             "capture_stop": CpuProfileCapture._stop,
             "capture_script": CpuProfileCapture.script,
             "capture_version": CpuProfileCapture.verify_version,
             "capture_open": CpuProfileCapture._open,
             "capture_deadline": CpuProfileCapture._remaining,
             "file_reader": _file, "single_fd_reader": _read_file,
             "bounded_reader": _read, "proc_reader": _process,
             "bounded_proc_reader": _read_proc, "model_stat": _stable_stat,
             "pipe_pump": _pump, "owned_reader": _owned_reader,
             "exited_reader_observation": _exited_reader_observation,
             "artifact_reducer": _reduce_artifact,
             "completed_response": _completed_response,
             "entry": main, "signal_refusal": _interrupt,
             "original_replay": reopen_capture,
             "no_gpu_proof": CpuOnlySampler.proof.fget,
             "run": run_profile_request, "source": source_identity,
             "loop_profile": profile_loop, "phase_reduction": _reduce_phases,
             "loop_reopen": reopen_loop_profile,
             "phase_reopen": _reopen_phases,
             "loop_claim": _loop_claim,
             "config": CpuProfileConfig.from_dict.__func__,
             "script_reducer": reduce_perf_script, "counter_reducer": reduce_perf_stat,
             "serving": serving._measure_once,
             "model_validator": tensor_capture.CaptureModelIdentity.validate,
             "prompt_parser": ps.FrozenPrompt.from_dict.__func__}
    callables = []
    for role, value in roles.items():
        identity = lo.callable_identity(value)
        if (identity["implementation_status"] != "pinned"
                or identity["configuration_status"] != "pinned"):
            raise CpuProfileRefused(f"source identity incomplete: {role}")
        callables.append({"role": role, "identity": identity})
    paths = (Path(__file__).resolve(), Path(serving.__file__).resolve(),
             Path(tensor_capture.__file__).resolve(), Path(lo.__file__).resolve(),
             Path(ps.__file__).resolve(), Path(ns.__file__).resolve(),
             Path(ob.__file__).resolve(), Path(rr.__file__).resolve(),
             Path(mp.__file__).resolve(), Path(mc.__file__).resolve(), Path(wl.__file__).resolve(),
             root / "scripts/benchmark/run_autokernel_cpu_profile.py")
    return {"schema": SOURCE_SCHEMA, "callables": callables,
            "files": [{"relative_path": str(path.relative_to(root)),
                       "sha256": _file(path, MAX_CONFIG_BYTES * 8)["sha256"]} for path in paths],
            "python": {key: value for key, value in _file(
                Path(sys.executable).resolve(), 512 * 1024**2).items() if key in {"path", "sha256"}},
            "constants": {"mode": MODE, "phases": list(PHASES), "events": list(EVENTS),
                          "limits": dict(LIMITS), "stdout_bytes": MAX_STDOUT_BYTES,
                          "sample_frequency": 99, "clock": "CLOCK_MONOTONIC",
                          "target_tid_limit": MAX_TARGET_TIDS, "line_bytes": MAX_LINE_BYTES,
                          "diagnostic_bytes": MAX_DIAGNOSTIC_BYTES, "limitations": list(LIMITATIONS),
                          "request_bytes": ns.MAX_REQUEST_BYTES,
                          "response_bytes": ns.MAX_RESPONSE_BYTES,
                          "loop_mode": LOOP_MODE, "loop_budgets": dict(LOOP_BUDGETS)}}


@dataclass(frozen=True)
class CpuProfileConfig:
    body: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value):
        row = _closed(value, {"schema", "mode", "loaded_identity", "resolved_recipe",
            "prompt_manifest", "model_preparation", "profiler", "source_closure",
            "storage", "budgets"}, "CPU profile config")
        if row["schema"] != CONFIG_SCHEMA or row["mode"] != MODE:
            raise CpuProfileRefused("unsupported sealed profiling mode")
        recipe = rr.resolved_recipe_from_dict(row["resolved_recipe"])
        if recipe.backend != "cpu" or recipe.template.np != 1:
            raise CpuProfileRefused("CPU capture requires exact cpu backend and np=1")
        prompts = ps.FrozenPromptManifest.from_dict(row["prompt_manifest"])
        prompts.requests(tuple(item.prompt_id for item in prompts.prompts), recipe.template)
        if len(prompts.prompts) != 1:
            raise CpuProfileRefused("CPU profile must have exactly one frozen prompt")
        spec = mp.ScheduledModelPreparation.from_dict(row["model_preparation"])
        loaded = _closed(row["loaded_identity"], {"target_revision_digest", "model_digest",
            "quantization", "recipe_digest", "executable_digest", "dso_digest"}, "loaded identity")
        arm = ps.arm_identity(recipe.template, recipe)
        _same(loaded["model_digest"], arm["model_digest"], "model digest")
        _same(loaded["recipe_digest"], recipe.snapshot_digest, "recipe digest")
        _same(loaded["executable_digest"], arm["executable_digest"], "executable digest")
        _same(loaded["dso_digest"], arm["dso_set_digest"], "DSO digest")
        _same(spec.target_revision_digest, loaded["target_revision_digest"], "model target")
        _same(spec.recipe_execution_digest, recipe.execution_digest, "model recipe")
        _same((spec.entry_path, spec.entry_sha256), (recipe.model.path, recipe.model.sha256),
              "model entry bridge")
        if type(loaded["quantization"]) is not str or not loaded["quantization"]:
            raise CpuProfileRefused("quantization must be recorded")
        profiler = _closed(row["profiler"], {"path", "sha256", "version", "server_interpreter"}, "profiler")
        for key in ("path", "sha256", "version"):
            if not isinstance(profiler[key], str) or not profiler[key]:
                raise CpuProfileRefused("profiler identity is missing")
        for path in (profiler["path"], row["storage"]):
            if type(path) is not str or not Path(path).is_absolute() or Path(path).resolve() != Path(path):
                raise CpuProfileRefused("deployment path is not absolute and canonical")
        if profiler["server_interpreter"] is not None:
            _closed(profiler["server_interpreter"], {"path", "sha256"}, "server interpreter")
        budgets = _closed(row["budgets"], LIMITS, "budgets")
        for name, maximum in LIMITS.items():
            amount = _number(budgets[name], name, positive=True)
            if amount > maximum or (not name.endswith("seconds") and type(budgets[name]) is not int):
                raise CpuProfileRefused(f"{name} exceeds supported budget")
        if budgets["max_total_raw_bytes"] < (2 * budgets["max_raw_file_bytes"]
                + 2 * budgets["max_parser_bytes"]
                + 2 * (ns.MAX_REQUEST_BYTES + ns.MAX_RESPONSE_BYTES)
                + 9 * MAX_DIAGNOSTIC_BYTES + budgets["max_metadata_bytes"]):
            raise CpuProfileRefused("aggregate raw capacity is insufficient before launch")
        _same(row["source_closure"], source_identity(), "installed source closure")
        if len(_canonical(row)) > MAX_CONFIG_BYTES:
            raise CpuProfileRefused("config exceeds bound")
        return cls(ob._freeze(_plain(row)))

    def to_dict(self):
        return _plain(self.body)


def build_installed_cpu_profile_binding(config_ref, *, valid_for_seconds):
    from . import profile_preparation as pp, target_profile_execution as tp
    ref = _closed(config_ref, {"path", "sha256"}, "config reference")
    raw = _read(Path(ref["path"]), MAX_CONFIG_BYTES)
    _same(hashlib.sha256(raw).hexdigest(), ref["sha256"], "config source")
    config = CpuProfileConfig.from_dict(json.loads(raw)).to_dict()
    root = Path(__file__).resolve().parents[4]
    entry = root / "scripts/benchmark/run_autokernel_cpu_profile.py"
    environment = {CONFIG_ENV: json.dumps(ref, sort_keys=True, separators=(",", ":")),
                   "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
                   "PYTHONDONTWRITEBYTECODE": "1"}
    mechanism = tp.ProfileMechanism(MECHANISM_ID, entry, _file(entry, MAX_CONFIG_BYTES)["sha256"],
        root, environment, config["loaded_identity"], config["budgets"]["max_stage_seconds"],
        config["budgets"]["teardown_seconds"], MAX_STDOUT_BYTES)
    return pp.InstalledProfileMechanismBinding(mechanism, valid_for_seconds)


def _lines(stream, *, max_bytes, max_rows):
    count = 0
    for index in range(max_rows + 1):
        raw = stream.readline(MAX_LINE_BYTES + 1)
        if not raw:
            return
        count += len(raw)
        if index == max_rows or count > max_bytes or len(raw) > MAX_LINE_BYTES:
            raise CpuProfileRefused("parser byte/line/row budget exhausted")
        try:
            yield raw.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise CpuProfileRefused("perf output is not UTF-8") from exc


def reduce_perf_script(stream, *, pid, tids, interval, max_bytes, max_rows, max_symbols):
    """Reduce exact perf -F pid,tid,time,period,event,ip,sym,dso output, never infer lost=0."""
    pattern = re.compile(r"^(\d+)(?:/|\s+)(\d+)\s+(\d+\.\d{9}):\s+(\d+)\s+cycles:u:\s+([0-9a-fA-F]+)\s+(.+)\s+\((.+)\)$")
    groups, by_tid = {}, {}
    samples = outside = total = 0
    for line in _lines(stream, max_bytes=max_bytes, max_rows=max_rows):
        if not line or line.startswith("#"):
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise CpuProfileRefused("unsupported/lost/malformed perf sample record")
        got_pid, tid, stamp, period, _ip, symbol, dso = match.groups()
        tid, period, stamp = int(tid), int(period), float(stamp)
        if int(got_pid) != pid or tid not in tids or period < 1:
            raise CpuProfileRefused("sample target/TID/period differs")
        if not interval[0] <= stamp <= interval[1]:
            outside += 1
            continue
        key = (dso, symbol)
        if key not in groups and len(groups) == max_symbols:
            raise CpuProfileRefused("symbol table budget exhausted")
        groups[key] = groups.get(key, 0) + period
        by_tid[tid] = by_tid.get(tid, 0) + period
        samples += 1
        total += period
    if not samples:
        raise CpuProfileRefused("no target samples overlap the request")
    return {"samples": samples, "sampled_period_total": total,
            "outside_request_samples": outside, "lost_records": "not_independently_quantified",
            "symbol_periods": [{"dso": key[0], "symbol": key[1], "period": value}
                               for key, value in sorted(groups.items())],
            "tid_periods": {str(key): value for key, value in sorted(by_tid.items())}}


def reduce_perf_stat(stream, *, max_bytes, max_rows):
    """Retain observed JSON counter values/running percentages, including unavailable states."""
    result = {}
    required = {"counter-value", "unit", "event", "event-runtime", "pcnt-running"}
    for line in _lines(stream, max_bytes=max_bytes, max_rows=max_rows):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CpuProfileRefused("malformed perf stat JSON") from exc
        if (not isinstance(row, dict) or not required <= set(row)
                or set(row) - required - {"metric-value", "metric-unit", "metric-threshold"}
                or row["event"] not in EVENTS or row["event"] in result):
            raise CpuProfileRefused("counter fields/event cardinality differs")
        value = row["counter-value"]
        unavailable = value in ("<not supported>", "<not counted>")
        try:
            value = value if unavailable else _number(float(value), "counter value")
            runtime = _number(float(row["event-runtime"]), "counter runtime")
            running = _number(float(row["pcnt-running"]), "counter running percent")
        except (ValueError, TypeError) as exc:
            raise CpuProfileRefused("counter numeric fields invalid") from exc
        if running > 100:
            raise CpuProfileRefused("counter running percentage exceeds 100")
        result[row["event"]] = {"value": value, "unit": row["unit"], "runtime": runtime,
            "running_percent": running, "status": "unavailable" if unavailable else
                "multiplexed" if running < 100 else "reported_full_running"}
    if set(result) != set(EVENTS):
        raise CpuProfileRefused("counter set is incomplete")
    return result


class CpuOnlySampler:
    """No GPU read, no invented samples; the owning CPU branch says not_applicable."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    @property
    def proof(self):
        return {"samples": 0, "vram_reads": 0, "resident": False,
                "collection": "not_requested_cpu_profile", "cpu_placement": "unproven",
                "contention": "unproven"}


def _pump(item, deadline, *, expect_ack=False, output=None, limit=0):
    """Drain all owned pipes concurrently; missing EOF is a bounded refusal."""
    process = item["process"]
    streams = {}
    if not item.get("stderr_eof"):
        streams[process.stderr.fileno()] = "stderr"
    if output is not None and not item.get("stdout_eof"):
        streams[process.stdout.fileno()] = "stdout"
    if expect_ack:
        streams[item["ack"]] = "ack"
    acknowledgment = bytearray()
    while streams:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CpuProfileRefused("owned tool pipe deadline exhausted")
        ready, _, _ = select.select(list(streams), [], [], remaining)
        if not ready:
            raise CpuProfileRefused("owned tool pipe timeout")
        for fd in ready:
            kind = streams[fd]
            raw = os.read(fd, 65536 if kind != "ack" else 5)
            if not raw:
                del streams[fd]
                item[kind + "_eof"] = True
                if kind == "ack":
                    raise CpuProfileRefused("perf acknowledgement pipe closed")
                continue
            if kind == "stderr":
                item["diagnostic"].extend(raw)
                if len(item["diagnostic"]) > MAX_DIAGNOSTIC_BYTES:
                    raise CpuProfileRefused("tool diagnostic byte bound exceeded")
            elif kind == "stdout":
                item["output_bytes"] += len(raw)
                if item["output_bytes"] > limit:
                    raise CpuProfileRefused("tool output byte bound exceeded")
                output.write(raw)
            else:
                acknowledgment.extend(raw)
                if (acknowledgment != b"a" and acknowledgment != b"ac"
                        and acknowledgment != b"ack" and acknowledgment != b"ack\n"
                        and acknowledgment != b"ack\n\x00"):
                    raise CpuProfileRefused("malformed perf acknowledgement")
                if acknowledgment == b"ack\n" or acknowledgment == b"ack\n\x00":
                    return
        if expect_ack and process.poll() is not None:
            raise CpuProfileRefused("perf exited before acknowledgement")
    process.wait(timeout=max(0.001, deadline - time.monotonic()))


def _exited_reader_observation(process):
    """Only an original unreaped child may lack its image after a proven exit."""
    try:
        proof = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError as exc:
        raise CpuProfileRefused("reader is already reaped or is not our child") from exc
    if proof is None or proof.si_pid != process.pid:
        raise CpuProfileRefused("missing reader image without exact owned-child exit proof")
    if proof.si_code == os.CLD_EXITED:
        returncode, semantics = proof.si_status, "exit_code"
    elif proof.si_code in (os.CLD_KILLED, os.CLD_DUMPED):
        returncode, semantics = -proof.si_status, "terminating_signal"
    else:
        raise CpuProfileRefused("reader waitid status is not a terminal exit")
    identity = {"pid": process.pid, "pid_basis": "original_Popen_child",
        "start_ticks": None, "boot_id": None, "ppid": None, "container": None,
        "argv": None, "exe": None, "image_readback": "unavailable_after_proven_exit"}
    # WNOWAIT preserves the zombie and prevents PID reuse. Retain only facts
    # actually readable from it; never copy the requested command into argv/exe.
    try:
        identity.update(wl.process_identity(process.pid).to_dict())
        raw_stat = _read_proc(Path(f"/proc/{process.pid}/stat"), 65536).decode()
        _same(int(raw_stat.split(" ", 1)[0]), process.pid, "exited reader stat PID")
        identity["ppid"] = int(raw_stat[raw_stat.rfind(")") + 1:].split()[1])
        _same(identity["ppid"], os.getpid(), "exited reader ancestry")
    except (FileNotFoundError, ProcessLookupError):
        pass
    try:
        cgroup = lo.FilesystemProbe()._cgroup_path(process.pid, {"max_read_bytes": 1024 * 1024})
        identity["container"] = {**lo._stat_identity(Path("/sys/fs/cgroup") / cgroup.lstrip("/")),
                                  "path": cgroup}
    except (FileNotFoundError, ProcessLookupError):
        pass
    return identity, {"pid": proof.si_pid, "si_code": proof.si_code,
        "si_status": proof.si_status, "semantics": semantics, "returncode": returncode,
        "wait_flags": "WEXITED|WNOHANG|WNOWAIT"}


def _owned_reader(command, *, output, limit, deadline, teardown_seconds):
    """One owned reader process; concurrently drain and bound both pipes and EOF."""
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, close_fds=True,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "DEBUGINFOD_URLS": ""})
    item = {"process": process, "diagnostic": bytearray(), "output_bytes": 0}
    try:
        exit_proof = None
        try:
            identity = _process(process.pid)
            _same(identity["ppid"], os.getpid(), "reader child ancestry")
        except (FileNotFoundError, ProcessLookupError):
            identity, exit_proof = _exited_reader_observation(process)
        _pump(item, deadline, output=output, limit=limit)
        if exit_proof is not None:
            _same(process.returncode, exit_proof["returncode"], "reader waitid/Popen terminal status")
        if process.returncode:
            raise CpuProfileRefused("owned reader returned nonzero")
        return {"command": command, "identity": identity, "returncode": process.returncode,
                "popen_parent_pid": os.getpid(), "pre_readback_exit_proof": exit_proof,
                "stderr": bytes(item["diagnostic"]).decode("utf-8", errors="replace")}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=teardown_seconds)
        process.stdout.close()
        process.stderr.close()


def _completed_response(raw, requested_n):
    try:
        result = json.loads(raw)
        if type(result) is not dict or type(result.get("timings")) is not dict:
            raise CpuProfileRefused("response/timings is not an object")
        observed_n = result["timings"].get("predicted_n")
        if (result.get("stop") is not True or type(observed_n) is not int
                or not 0 < observed_n <= requested_n):
            raise CpuProfileRefused("request lacks successful terminal completion")
        return observed_n
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CpuProfileRefused("request response is malformed") from exc


class CpuProfileCapture:
    def __init__(self, *, config, request, store):
        if type(config) is not CpuProfileConfig or type(store) is not mc.ArtifactStore:
            raise CpuProfileRefused("concrete config/store required")
        self._initialize(config.to_dict(), request, store)

    @classmethod
    def for_loop(cls, recipe, prompts, *, store, perf_path, timeout_s, server_interpreter=None):
        if type(store) is not mc.ArtifactStore or cls is not CpuProfileCapture:
            raise CpuProfileRefused("direct CPU profile requires the original concrete capture/store")
        recipe = rr.resolved_recipe_from_dict(recipe.to_dict())
        prompts = ps.FrozenPromptManifest.from_dict(prompts.to_dict())
        if recipe.backend != "cpu" or recipe.template.np != 1 or len(prompts.prompts) != 1:
            raise CpuProfileRefused("direct CPU profiling currently supports one original CPU request")
        prompts.requests(tuple(item.prompt_id for item in prompts.prompts), recipe.template)
        timeout_s = _number(timeout_s, "profile timeout", positive=True)
        if timeout_s > LIMITS["max_stage_seconds"]:
            raise CpuProfileRefused("profile timeout exceeds supported capture bound")
        perf = _file(Path(perf_path).resolve(), 512 * 1024**2)
        settings = {"resolved_recipe": recipe.to_dict(), "prompt_manifest": prompts.to_dict(),
                    "profiler": {"path": perf["path"], "sha256": perf["sha256"],
                                 "version": None, "server_interpreter": server_interpreter},
                    "source_closure": source_identity(),
                    "budgets": {**LOOP_BUDGETS, "max_stage_seconds": timeout_s}}
        request = {"mode": LOOP_MODE, "execution_digest": recipe.execution_digest,
                   "prompt_manifest_digest": prompts.digest,
                   "producer_pid": os.getpid(), "started_monotonic_ns": time.monotonic_ns()}
        capture = object.__new__(cls)
        capture._initialize(settings, request, store)
        return capture

    def _initialize(self, settings, request, store):
        self.config = _plain(settings)
        self.request = _plain(request)
        self.store = store
        self.budgets = self.config["budgets"]
        self.started = time.monotonic()
        self.deadline = self.started + self.budgets["max_stage_seconds"]
        self.owner = _process(os.getpid())
        self.ancestor = _process(os.getppid())
        # The controller may be outside the allocated worker cgroup. Descendants
        # must match the worker's actual container, not the controller's container.
        _same(self.owner["ppid"], self.ancestor["pid"], "worker ancestor")
        self.target = None
        self.phases = []
        self.active = []
        self.raw_responses = None
        self.failed = None
        self.cleanup_uncertain = None
        self.finished = False
        self.directory = store.root / ("cpu-raw-" + _digest(request)[:32])
        self.directory.mkdir(mode=0o700)
        self.directory_identity = lo._stat_identity(self.directory)
        self.perf = self.config["profiler"]["path"]
        _same(_file(Path(self.perf), 512 * 1024**2)["sha256"],
              self.config["profiler"]["sha256"], "perf executable")

    def _remaining(self, maximum):
        amount = min(maximum, self.deadline - time.monotonic())
        if amount <= 0:
            raise CpuProfileRefused("profile stage deadline exhausted")
        return amount

    def verify_version(self):
        path = self.directory / "perf-version.txt"
        with os.fdopen(self._open(path.name), "wb") as stream:
            result = _owned_reader([self.perf, "--version"], output=stream,
                limit=MAX_DIAGNOSTIC_BYTES,
                deadline=time.monotonic() + self._remaining(self.budgets["control_seconds"]),
                teardown_seconds=self.budgets["teardown_seconds"])
            stream.flush()
            os.fsync(stream.fileno())
        version = _read(path, MAX_DIAGNOSTIC_BYTES).decode().strip()
        if not version:
            raise CpuProfileRefused("perf reported no version")
        if self.config["profiler"]["version"] is not None:
            _same(version, self.config["profiler"]["version"], "perf reported version")
        return {**result, "artifact": _file(path, MAX_DIAGNOSTIC_BYTES)}

    def validate_launch(self, recipe, requests):
        expected = rr.resolved_recipe_from_dict(self.config["resolved_recipe"])
        _same(recipe.to_dict(), expected.to_dict(), "profile selected launch")
        prompts = ps.FrozenPromptManifest.from_dict(self.config["prompt_manifest"])
        _same(tuple(requests), prompts.requests(tuple(x.prompt_id for x in prompts.prompts),
              recipe.template), "profile frozen requests")
        _same(source_identity(), self.config["source_closure"], "capture selected source")

    def attach_target(self, pid):
        if self.target is not None:
            raise CpuProfileRefused("target already attached")
        observed = _process(pid)
        _same(observed["ppid"], self.owner["pid"], "server is not producer child")
        _same(observed["container"], self.owner["container"], "server container")
        self.target = observed

    def _check_target(self):
        if self.target is None:
            raise CpuProfileRefused("target is absent")
        _same(_identity_key(_process(os.getpid())), _identity_key(self.owner), "producer identity")
        current = _process(self.target["pid"])
        _same(_identity_key(current), _identity_key(self.target), "server identity")
        _same(current["ppid"], self.owner["pid"], "server ancestry")
        recipe = rr.resolved_recipe_from_dict(self.config["resolved_recipe"])
        executable = Path(recipe.executable.path).resolve()
        _same(_file(executable, 512 * 1024**2)["sha256"], recipe.executable.sha256, "server bytes")
        interpreter = self.config["profiler"]["server_interpreter"]
        if interpreter is None:
            _same(current["exe"], str(executable), "loaded executable")
            expected_argv = list(recipe.command_argv)
        else:
            _same(current["exe"], interpreter["path"], "loaded script interpreter")
            _same(_file(Path(current["exe"]), 512 * 1024**2)["sha256"],
                  interpreter["sha256"], "interpreter bytes")
            expected_argv = [interpreter["path"], *recipe.command_argv]
        _same(current["argv"], expected_argv, "actual server argv")
        maps = _read_proc(Path(f"/proc/{current['pid']}/maps"), 1024 * 1024).decode()
        mappings = []
        for dso in recipe.dsos:
            fact = _file(Path(dso.path).resolve(), 512 * 1024**2)
            _same(fact["sha256"], dso.sha256, "loaded DSO bytes")
            matched = False
            for line in maps.splitlines():
                row = line.split(maxsplit=5)
                if len(row) != 6 or row[5].endswith(" (deleted)"):
                    continue
                major, minor = row[3].split(":")
                if "x" not in row[1]:
                    continue
                mapped = None
                if (int(row[4]) == fact["ino"]
                        and os.makedev(int(major, 16), int(minor, 16)) == fact["dev"]):
                    mapped = fact
                else:
                    # CMake may embed a RUNPATH to another build directory whose
                    # DSO is a byte-identical copy, not a hardlink.  The recipe
                    # pins bytes, so verify the file actually mapped by the
                    # process instead of requiring filesystem identity with the
                    # recipe's archival copy.
                    try:
                        mapped = _file(Path(row[5]).resolve(), 512 * 1024**2)
                    except (FileNotFoundError, OSError, CpuProfileRefused):
                        continue
                    if mapped["sha256"] != dso.sha256:
                        continue
                if mapped is not None:
                    matched = True
                    mappings.append({"artifact": mapped, "declared_artifact": fact,
                                     "mapping": line,
                                     "observed_path": row[5]})
                    break
            if not matched:
                raise CpuProfileRefused("required DSO executable mapping missing")
        return {**current, "loaded_dso_mappings": mappings}

    def _open(self, name):
        _same(lo._stat_identity(self.directory), self.directory_identity, "raw directory")
        return os.open(self.directory / name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)

    def _spawn(self, kind, phase):
        suffix = "data" if kind == "record" else "jsonl"
        name = f"{phase}-{kind}.{suffix}"
        output = self._open(name)
        ctl_read, ctl_write = os.pipe()
        ack_read, ack_write = os.pipe()
        command = [self.perf, kind, "--delay=-1", f"--control=fd:{ctl_read},{ack_write}",
                   "-p", str(self.target["pid"]), "-o", f"/proc/self/fd/{output}"]
        if kind == "record":
            command += ["-F", "99", "-e", "cycles:u", "--clockid", "mono", "-P", "-T",
                        "--no-buildid", "--no-buildid-cache", "--mmap-pages=8",
                        "--max-size", f'{self.budgets["max_raw_file_bytes"]}B']
        else:
            command += ["--json-output", "-e", ",".join(EVENTS)]
        process = None
        item = None
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C",
                                             "DEBUGINFOD_URLS": ""},
                pass_fds=(output, ctl_read, ack_write), close_fds=True)
            item = {"process": process, "identity": None, "ctl": ctl_write, "ack": ack_read,
                    "path": self.directory / name, "kind": kind, "command": command,
                    "diagnostic": bytearray(), "output_bytes": 0, "controls": []}
            # Enroll before any fallible proc read. An unreaped Popen child remains
            # ours to kill even when the optional identity read itself fails.
            self.active.append(item)
            item["identity"] = wl.process_identity(process.pid)
            observed = _process(process.pid)
            _same(observed["ppid"], os.getpid(), "perf child ancestry")
            _same(observed["container"], self.owner["container"], "perf child container")
            item["process_observation"] = observed
            self._control(item, b"enable\n")
            return item
        except BaseException:
            if item is not None:
                self._stop(item, normal=False)
            else:
                os.close(ctl_write)
                os.close(ack_read)
            raise
        finally:
            os.close(ctl_read)
            os.close(ack_write)
            os.close(output)

    def _control(self, item, command):
        if item["process"].poll() is not None or not wl.same_process(item["identity"]):
            raise CpuProfileRefused("perf exited before control acknowledgement")
        sent = time.monotonic()
        if os.write(item["ctl"], command) != len(command):
            raise CpuProfileRefused("short perf control write")
        _pump(item, time.monotonic() + self._remaining(self.budgets["control_seconds"]),
              expect_ack=True)
        item["controls"].append({"command": command.decode().strip(),
                                 "sent": sent, "acknowledged": time.monotonic()})

    def begin_round(self, phase):
        _same(phase, PHASES[len(self.phases)] if len(self.phases) < 2 else None, "phase order")
        if self.active:
            raise CpuProfileRefused("perf round overlaps")
        self.round_target_before = self._check_target()
        self.round_tids = set()
        for path in Path(f"/proc/{self.target['pid']}/task").iterdir():
            if len(self.round_tids) == MAX_TARGET_TIDS:
                raise CpuProfileRefused("target TID budget exhausted")
            self.round_tids.add(int(path.name))
        self._spawn("record", phase)
        self._spawn("stat", phase)
        self.round_started = time.monotonic()

    def _stop(self, item, *, normal):
        process = item["process"]
        failure = None
        try:
            if normal:
                self._control(item, b"disable\n")
        except BaseException as exc:
            failure = exc
        finally:
            try:
                if process.poll() is None:
                    if item["identity"] is not None and not wl.same_process(item["identity"]):
                        raise CpuProfileRefused("perf PID identity lost during cleanup")
                    process.send_signal(signal.SIGINT)
                _pump(item, time.monotonic() + min(5.0, self.budgets["teardown_seconds"]))
            except BaseException as exc:
                failure = failure or exc
                if process.poll() is None:
                    process.kill()
                try:
                    process.wait(timeout=min(5.0, self.budgets["teardown_seconds"]))
                except subprocess.TimeoutExpired as exc:
                    self.cleanup_uncertain = "owned perf child did not terminate"
                    raise CpuProfileCleanupUncertain(self.cleanup_uncertain) from exc
            finally:
                if process.returncode is None:
                    self.cleanup_uncertain = "owned perf child has no terminal wait result"
                process.stderr.close()
                for field in ("ctl", "ack"):
                    os.close(item[field])
                self.active.remove(item)
        item["stderr"] = bytes(item["diagnostic"][:MAX_DIAGNOSTIC_BYTES]).decode("utf-8", errors="replace")
        if failure is not None:
            raise CpuProfileRefused(f"perf cleanup/control failed: {failure}") from failure
        # `perf record` flushes a valid capture when its owning wrapper ends it
        # with SIGINT; Popen reports that normal tool shutdown as -SIGINT.
        if normal and process.returncode not in (0, -signal.SIGINT):
            raise CpuProfileRefused(f"perf {item['kind']} failed: {item['stderr'][:300]}")
        return item

    def script(self, record, path):
        with os.fdopen(self._open(path.name), "wb") as stream:
            result = _owned_reader([self.perf, "script", "-i", record["path"],
                "--ns", "--show-lost-events", "-F", "pid,tid,time,period,event,ip,sym,dso"],
                output=stream, limit=self.budgets["max_parser_bytes"],
                deadline=time.monotonic() + self._remaining(self.budgets["reduce_seconds"]),
                teardown_seconds=self.budgets["teardown_seconds"])
            stream.flush()
            os.fsync(stream.fileno())
            return result

    def end_round(self, phase):
        ended = time.monotonic()
        target_after = self._check_target()
        items = [self._stop(item, normal=True) for item in list(self.active)]
        self.phases.append({"phase": phase, "enabled_interval": [self.round_started, ended],
            "target_before": self.round_target_before, "target_after": target_after,
            "tids": sorted(self.round_tids), "tools": [{"kind": item["kind"],
                "identity": item["identity"].to_dict(), "returncode": item["process"].returncode,
                "process_observation": item["process_observation"], "command": item["command"],
                "controls": item["controls"],
                "stderr": item["stderr"], "artifact": _file(item["path"],
                    self.budgets["max_raw_file_bytes"] if item["kind"] == "record"
                    else MAX_DIAGNOSTIC_BYTES)} for item in items]})

    def retain_responses(self, rows):
        if self.raw_responses is not None or len(rows) != 2:
            raise CpuProfileRefused("response cardinality/repeated completion differs")
        serialized = []
        for phase, raw in zip(PHASES, rows):
            if type(raw) is not ns.RawServerResponse or raw.phase != phase or raw.slot_index != 0:
                raise CpuProfileRefused("response phase/slot differs")
            serialized.append({"phase": raw.phase, "slot": raw.slot_index,
                "prompt_id": raw.prompt_id, "request_hex": raw.request.hex(),
                "response_hex": None if raw.response is None else raw.response.hex(),
                "start": raw.started_monotonic_s, "end": raw.ended_monotonic_s, "error": raw.error})
        self.raw_responses = serialized

    def abort(self, reason):
        self.failed = str(reason)
        failures = []
        for item in list(self.active):
            try:
                self._stop(item, normal=False)
            except Exception as exc:
                failures.append(str(exc))
        if failures:
            if self.cleanup_uncertain:
                raise CpuProfileCleanupUncertain(self.cleanup_uncertain)
            raise CpuProfileRefused("perf cleanup/control failed: " + ";".join(failures))

    def finish(self):
        if self.cleanup_uncertain:
            raise CpuProfileCleanupUncertain(self.cleanup_uncertain)
        if self.active:
            self.abort("capture did not finish both request rounds")
        self.finished = True


def reopen_capture(reference, *, store, config, request):
    """Recompute original factual reductions; no live grant or new observations."""
    expected = config.to_dict() if type(config) is CpuProfileConfig else _plain(config)
    body = _plain(store.read(reference["locator"], reference["sha256"]))
    _closed(body, {"schema", "request_digest", "target_revision_digest", "source_closure",
        "loaded_identity", "model_verification", "profiler", "processes", "phases", "completion", "limitations"}, "capture")
    _same(body["schema"], CAPTURE_SCHEMA, "capture schema")
    _same(body["request_digest"], _digest(request), "original request")
    _same(body["source_closure"], expected["source_closure"], "original source")
    _same(body["loaded_identity"], expected["loaded_identity"], "original loaded identities")
    _same(body["target_revision_digest"], request["target_revision_digest"], "original target")
    _same(body["completion"], "complete", "capture completion")
    _same(body["limitations"], list(LIMITATIONS), "capture limitations")
    if [item.get("phase") for item in body["phases"]] != list(PHASES):
        raise CpuProfileRefused("original phase cardinality/order differs")
    _same(body["model_verification"]["spec"], expected["model_preparation"], "original model preparation")
    model = mp.ScheduledModelPreparation.from_dict(expected["model_preparation"])
    model_raw = body["model_verification"]["manifest_raw"].encode()
    _same(hashlib.sha256(model_raw).hexdigest(), model.inventory_identity["model_manifest_sha256"],
          "original verified model manifest")
    return _reopen_phases(body, expected)


def _reopen_phases(body, expected):
    """One raw-evidence replay for both original sealed and direct observations."""
    prompts = ps.FrozenPromptManifest.from_dict(expected["prompt_manifest"])
    recipe = rr.resolved_recipe_from_dict(expected["resolved_recipe"])
    frozen = prompts.requests(tuple(x.prompt_id for x in prompts.prompts), recipe.template)
    owner, ancestor, target = (body["processes"][key] for key in ("producer", "ancestor", "server"))
    _same(owner["ppid"], ancestor["pid"], "original worker ancestor")
    _same(target["ppid"], owner["pid"], "original server parent")
    _same(target["container"], owner["container"], "original server container")
    reported_version = _read(Path(body["profiler"]["artifact"]["path"]), MAX_DIAGNOSTIC_BYTES).decode().strip()
    if not reported_version:
        raise CpuProfileRefused("original profiler version is empty")
    if expected["profiler"]["version"] is not None:
        _same(reported_version, expected["profiler"]["version"], "original profiler version")
    version = _file(Path(body["profiler"]["artifact"]["path"]), MAX_DIAGNOSTIC_BYTES)
    _same(version, body["profiler"]["artifact"], "original profiler version artifact")
    total = version["size"]
    for phase in body["phases"]:
        response = phase["response"]
        _same(response["phase"], phase["phase"], "original response phase")
        _same((response["slot"], response["prompt_id"]), (0, frozen[0][0]), "original response slot")
        _same(bytes.fromhex(response["request_hex"]), frozen[0][1], "original request bytes")
        if response["error"] or response["response_hex"] is None:
            raise CpuProfileRefused("original response has error")
        _same(_completed_response(bytes.fromhex(response["response_hex"]), recipe.template.n_predict),
              phase["observed_predicted_n"], "original completed length")
        interval = [response["start"], response["end"]]
        if not phase["enabled_interval"][0] <= interval[0] <= interval[1] <= phase["enabled_interval"][1]:
            raise CpuProfileRefused("original request window differs")
        if [item["kind"] for item in phase["tools"]] != ["record", "stat"]:
            raise CpuProfileRefused("original tool membership differs")
        for item in phase["tools"]:
            _same(item["process_observation"]["ppid"], owner["pid"], "original perf parent")
            _same(item["process_observation"]["container"], owner["container"], "original perf container")
            _same(item["identity"]["pid"], item["process_observation"]["pid"], "original perf PID")
            controls = item["controls"]
            if ([value["command"] for value in controls] != ["enable", "disable"]
                    or item["returncode"] not in (0, -signal.SIGINT)):
                raise CpuProfileRefused("original perf control completion differs")
            enable, disable = controls
            times = [enable["sent"], enable["acknowledged"], phase["enabled_interval"][0],
                     interval[0], interval[1], phase["enabled_interval"][1],
                     disable["sent"], disable["acknowledged"]]
            if any(_number(value, "control timestamp") != value for value in times) or times != sorted(times):
                raise CpuProfileRefused("original control/request chronology differs")
        _same(phase["counter_scope"], {"enable_transition": phase["tools"][1]["controls"][0],
              "disable_transition": phase["tools"][1]["controls"][1],
              "scope": "cumulative own enable-to-disable window; not exact request/sample interval"},
              "original counter scope")
        for readback in (phase["target_before"], phase["target_after"]):
            _same(_identity_key(readback), _identity_key(target), "original loaded server identity")
            _same(readback["ppid"], owner["pid"], "original loaded server parent")
            interpreter = expected["profiler"]["server_interpreter"]
            argv = list(recipe.command_argv) if interpreter is None else [interpreter["path"], *recipe.command_argv]
            _same(readback["argv"], argv, "original loaded server argv")
            _same(readback["exe"], recipe.executable.path if interpreter is None else interpreter["path"],
                  "original loaded executable")
            _same([value["artifact"]["sha256"]
                   for value in readback["loaded_dso_mappings"]],
                  [dso.sha256 for dso in recipe.dsos], "original loaded DSO set")
            for value in readback["loaded_dso_mappings"]:
                row = value["mapping"].split(maxsplit=5)
                major, minor = row[3].split(":")
                if (len(row) != 6 or "x" not in row[1] or row[5].endswith(" (deleted)")
                        or int(row[4]) != value["artifact"]["ino"]
                        or os.makedev(int(major, 16), int(minor, 16)) != value["artifact"]["dev"]):
                    raise CpuProfileRefused("original DSO mapping differs")
        for ref in [item["artifact"] for item in phase["tools"]] + [phase["script"]]:
            actual = _file(Path(ref["path"]), expected["budgets"]["max_total_raw_bytes"])
            _same(actual, ref, "original raw artifact")
            total += actual["size"]
        samples = _reduce_artifact(phase["script"], kind="samples", arguments={
            "pid": target["pid"], "tids": phase["tids"], "interval": interval,
            "max_bytes": expected["budgets"]["max_parser_bytes"],
            "max_rows": expected["budgets"]["max_rows"], "max_symbols": expected["budgets"]["max_symbols"]})
        counters = _reduce_artifact(phase["tools"][1]["artifact"], kind="counters",
            arguments={"max_bytes": MAX_DIAGNOSTIC_BYTES, "max_rows": 16})
        _same(samples, phase["samples"], "original sample reduction")
        _same(counters, phase["counters"], "original counter reduction")
    if total + len(_canonical(body)) > expected["budgets"]["max_total_raw_bytes"]:
        raise CpuProfileRefused("actual aggregate raw bytes exceed budget")
    return ob._freeze(body)


def _reduce_phases(capture, recipe, frozen, budgets):
    if capture.failed or not capture.finished or len(capture.phases) != 2 or capture.raw_responses is None:
        raise CpuProfileRefused("profile capture is incomplete")
    phase_results = []
    for phase, response in zip(capture.phases, capture.raw_responses):
        if response["error"] or response["response_hex"] is None:
            raise CpuProfileRefused("request response failed")
        request_raw = bytes.fromhex(response["request_hex"])
        _same(request_raw, frozen[0][1], "original request bytes")
        observed_n = _completed_response(bytes.fromhex(response["response_hex"]), recipe.template.n_predict)
        interval = [response["start"], response["end"]]
        if not phase["enabled_interval"][0] <= interval[0] <= interval[1] <= phase["enabled_interval"][1]:
            raise CpuProfileRefused("request not enclosed by perf window")
        record = next(x["artifact"] for x in phase["tools"] if x["kind"] == "record")
        counter = next(x["artifact"] for x in phase["tools"] if x["kind"] == "stat")
        script_path = capture.directory / (phase["phase"] + "-script.txt")
        parser = capture.script(record, script_path)
        script_ref = _file(script_path, budgets["max_parser_bytes"])
        samples = _reduce_artifact(script_ref, kind="samples", arguments={
            "pid": capture.target["pid"], "tids": phase["tids"], "interval": interval,
            "max_bytes": budgets["max_parser_bytes"], "max_rows": budgets["max_rows"],
            "max_symbols": budgets["max_symbols"]})
        counters = _reduce_artifact(counter, kind="counters",
            arguments={"max_bytes": MAX_DIAGNOSTIC_BYTES, "max_rows": 16})
        phase_results.append({**phase, "response": response, "samples": samples,
            "observed_predicted_n": observed_n, "parser": parser,
            "counter_scope": {"enable_transition": phase["tools"][1]["controls"][0],
                "disable_transition": phase["tools"][1]["controls"][1],
                "scope": "cumulative own enable-to-disable window; not exact request/sample interval"},
            "counters": counters, "script": script_ref})
    return phase_results



def reopen_loop_profile(reference, *, store):
    """Reopen the original direct record; never create a sealed profile or verifier."""
    record = _plain(store.read(reference["locator"], reference["sha256"]))
    _closed(record, {"schema", "mode", "capture", "source_id", "run_id", "profile_claim_tuple"},
            "loop profile")
    _same(record["schema"], LOOP_PROFILE_SCHEMA, "loop profile schema")
    _same(record["mode"], LOOP_MODE, "loop profile scope")
    _same(record["source_id"], "VB-AK-UNIFIED-PROFILE", "original measurement source")
    body = _plain(store.read(record["capture"]["locator"], record["capture"]["sha256"]))
    _closed(body, {"schema", "request", "settings", "profiler", "processes", "phases",
                   "completion", "limitations"}, "direct capture")
    _same(body["schema"], LOOP_CAPTURE_SCHEMA, "direct capture schema")
    _same(body["completion"], "complete", "direct capture completion")
    _same(body["limitations"], list(LIMITATIONS), "direct capture limitations")
    settings = body["settings"]
    recipe = rr.resolved_recipe_from_dict(settings["resolved_recipe"])
    prompts = ps.FrozenPromptManifest.from_dict(settings["prompt_manifest"])
    _same(body["request"]["execution_digest"], recipe.execution_digest, "original execution")
    _same(body["request"]["prompt_manifest_digest"], prompts.digest, "original prompts")
    _same(body["request"]["mode"], LOOP_MODE, "original mode")
    _same(body["request"]["producer_pid"], body["processes"]["producer"]["pid"], "original producer")
    _same([item["phase"] for item in body["phases"]], list(PHASES), "original phases")
    _reopen_phases(body, settings)
    _same(record["run_id"], "loop-cpu-profile:" + _digest(body["request"]), "original run")
    _same(record["profile_claim_tuple"], _loop_claim(body, record["capture"],
          record["profile_claim_tuple"]["date"]), "original measurement claim")
    return ob._freeze(body)


def _loop_claim(body, artifact, observed_date):
    return {"date": observed_date, "category": "CANDIDATE", "protocol_id": "", "reps": 1,
        "reps_basis": "one completed profiled request; samples are not repetitions",
        "attestation_locator": artifact["locator"], "attestation_sha256": artifact["sha256"],
        "attestation_present": True, "attestation_verified": True,
        "measurement_id": "loop-cpu-profile:" + _digest(body["request"]) + ":profile",
        "metric": "request_sampled_period_total",
        "value": body["phases"][1]["samples"]["sampled_period_total"],
        "metric_direction": "lower_better", "unit": "sampled user-cycle periods",
        "claim": "Recorded sampled-period total for this exact full request; estimated attribution, not exact CPU cost",
        "source_class": "measurement", "extra": {"limitations": list(LIMITATIONS),
            "mode": LOOP_MODE, "not_performance_comparison": True,
            "model_inventory_verification": "not performed by direct profiler"}}


def profile_loop(recipe, prompts, *, store_root, perf_path="/usr/bin/perf",
                 timeout_s=1800, server_interpreter=None):
    """Separate observational launch using the loop's actual current binary and requests."""
    from . import serving
    capture = None
    with closing(mc.ArtifactStore(Path(store_root) / "cpu-profiles")) as store:
        try:
            capture = CpuProfileCapture.for_loop(recipe, prompts, store=store,
                perf_path=perf_path, timeout_s=timeout_s, server_interpreter=server_interpreter)
            profiler = capture.verify_version()
            frozen = prompts.requests(tuple(x.prompt_id for x in prompts.prompts), recipe.template)
            serving._measure_once(recipe.template, recipe.build_dir, recipe.port,
                boot_timeout_s=math.ceil(capture._remaining(timeout_s)),
                resolved_recipe=recipe, frozen_requests=frozen, cpu_profile_capture=capture)
            phases = _reduce_phases(capture, recipe, frozen, capture.budgets)
            _same(source_identity(), capture.config["source_closure"], "original direct source")
            body = {"schema": LOOP_CAPTURE_SCHEMA, "request": capture.request,
                "settings": capture.config, "profiler": profiler,
                "processes": {"producer": capture.owner, "ancestor": capture.ancestor,
                              "server": capture.target}, "phases": phases,
                "completion": "complete", "limitations": list(LIMITATIONS)}
            if len(_canonical(body)) > capture.budgets["max_metadata_bytes"]:
                raise CpuProfileRefused("direct capture metadata exceeds byte budget")
            artifact = store.write("loop-cpu-capture:" + _digest(capture.request), body)
            record = {"schema": LOOP_PROFILE_SCHEMA, "mode": LOOP_MODE,
                "capture": artifact.to_dict(), "source_id": "VB-AK-UNIFIED-PROFILE",
                "run_id": "loop-cpu-profile:" + _digest(capture.request),
                "profile_claim_tuple": _loop_claim(body, artifact.to_dict(),
                    datetime.now(timezone.utc).date().isoformat())}
            exported = store.write(record["run_id"], record)
            original = reopen_loop_profile(exported.to_dict(), store=store)
            measured = original["phases"][1]["samples"]
            total = measured["sampled_period_total"]
            return {"status": "observed", "record": str(store.root / exported.locator),
                "record_sha256": exported.sha256, "execution_digest": recipe.execution_digest,
                "prompt_manifest_digest": prompts.digest, "sampled_period_total": total,
                "samples": measured["samples"], "limitations": list(LIMITATIONS),
                "hotspots": [{**_plain(row), "sampled_period_fraction": row["period"] / total}
                    for row in sorted(measured["symbol_periods"],
                        key=lambda row: (-row["period"], row["dso"], row["symbol"]))[:12]]}
        except BaseException as exc:
            if capture is not None:
                capture.abort(f"{type(exc).__name__}: {exc}")
                if capture.cleanup_uncertain:
                    raise CpuProfileCleanupUncertain(capture.cleanup_uncertain) from exc
            raise


def run_profile_request(request, config):
    """Run only inside the owning profile worker; successful JSON is not grant authority."""
    from . import serving, target_profile_execution as tp, unified_driver as ud, unified_planner as up
    config = CpuProfileConfig.from_dict(config.to_dict() if type(config) is CpuProfileConfig else config)
    body = config.to_dict()
    selected = ud.ProfilePreparationRequest.from_dict(request)
    _same(selected.target_revision_digest, body["loaded_identity"]["target_revision_digest"], "selected target")
    _same(selected.profile_contract["adapter_id"], MECHANISM_ID, "selected mechanism")
    spec = mp.ScheduledModelPreparation.from_dict(body["model_preparation"])
    recipe = rr.resolved_recipe_from_dict(body["resolved_recipe"])
    prompts = ps.FrozenPromptManifest.from_dict(body["prompt_manifest"])
    capture = None
    with closing(mc.ArtifactStore(Path(body["storage"]))) as store:
        try:
            # This is real preparation inside the held worker stage, before serving.
            capture = CpuProfileCapture(config=config, request=request, store=store)
            profiler = capture.verify_version()
            manifest_raw = _read(Path(spec.inventory_identity["model_manifest"]), tensor_capture.MAX_JSON_BYTES)
            _same(hashlib.sha256(manifest_raw).hexdigest(),
                  spec.inventory_identity["model_manifest_sha256"], "original model manifest bytes")
            manifest = json.loads(manifest_raw)
            root = Path(spec.inventory_identity["model_id"])
            _closed(manifest, {"schema", "model_path", "files"}, "model manifest")
            if type(manifest["files"]) is not list or not 1 <= len(manifest["files"]) <= 4096:
                raise CpuProfileRefused("model manifest member count unsupported")
            paths = []
            for row in manifest["files"]:
                row = _closed(row, {"path", "sha256"}, "model manifest member")
                member = Path(row["path"])
                if member.is_absolute() or ".." in member.parts:
                    raise CpuProfileRefused("model member escapes declared inventory")
                paths.append(root if row["path"] == "." else root / member)
            before = {str(path): _stable_stat(path) for path in paths}
            spec.identity().validate()
            _same(before, {str(path): _stable_stat(path) for path in paths}, "verified model continuity")
            members = {str(root if row["path"] == "." else root / row["path"]): row["sha256"]
                       for row in manifest["files"]}
            _same(members.get(spec.entry_path), spec.entry_sha256, "verified entry member")
            model = {"spec": spec.to_dict(), "manifest_raw": manifest_raw.decode(), "file_stats": before,
                     "authority": "original scheduled full-inventory byte verification; no T0 claim"}
            frozen = prompts.requests(tuple(x.prompt_id for x in prompts.prompts), recipe.template)
            observation = []
            serving._measure_once(recipe.template, recipe.build_dir, recipe.port,
                boot_timeout_s=math.ceil(capture._remaining(body["budgets"]["max_stage_seconds"])),
                resolved_recipe=recipe, frozen_requests=frozen,
                observation=observation, cpu_profile_capture=capture)
            phase_results = _reduce_phases(capture, recipe, frozen, body["budgets"])
            _same(before, {str(path): _stable_stat(path) for path in paths}, "post-serving model continuity")
            receipt = {"schema": CAPTURE_SCHEMA, "request_digest": _digest(request),
                "target_revision_digest": selected.target_revision_digest,
                "source_closure": body["source_closure"], "loaded_identity": body["loaded_identity"],
                "model_verification": model, "profiler": profiler, "processes": {"producer": capture.owner,
                    "ancestor": capture.ancestor, "server": capture.target}, "phases": phase_results,
                "completion": "complete", "limitations": list(LIMITATIONS)}
            _same(source_identity(), body["source_closure"], "final selected source")
            if len(_canonical(receipt)) > body["budgets"]["max_metadata_bytes"]:
                raise CpuProfileRefused("capture metadata exceeds byte budget")
            artifact = store.write("owned-cpu-profile:" + _digest(request), receipt)
            reopen_capture(artifact.to_dict(), store=store, config=config, request=request)
            measured = phase_results[1]["samples"]
            profile = up.TargetProfile.from_dict({"schema": up.PROFILE_SCHEMA,
                "target_revision_digest": selected.target_revision_digest, "freshness": "fresh",
                "quant": body["loaded_identity"]["quantization"],
                "hotspots": [x["dso"] + ":" + x["symbol"] for x in sorted(
                    measured["symbol_periods"], key=lambda x: (-x["period"], x["dso"], x["symbol"]))],
                "observation_states": ["unknown"], "kept_scope": [MODE, *LIMITATIONS],
                "resource_cost": selected.stage_proposal.estimated_claims.to_dict(), "opportunities": []})
            run_id = "cpu-profile:" + _digest(request)
            common = {"date": datetime.now(timezone.utc).date().isoformat(), "category": "CANDIDATE",
                "protocol_id": "", "reps": 1, "reps_basis": "one completed profiled request; samples are not repetitions",
                "attestation_locator": artifact.locator, "attestation_sha256": artifact.sha256,
                "attestation_present": True, "attestation_verified": True}
            profile_tuple = {**common, "measurement_id": run_id + ":profile",
                "metric": "request_sampled_period_total", "value": measured["sampled_period_total"],
                "metric_direction": "lower_better", "unit": "sampled user-cycle periods",
                "claim": "Recorded sampled-period total for this exact full request; estimated attribution, not exact CPU cost",
                "source_class": "measurement", "extra": {"limitations": list(LIMITATIONS)}}
            proposition = "This original CPU profile receipt reopens with the recorded request, artifact, source and window joins"
            validation_tuple = {**common, "measurement_id": run_id + ":integrity",
                "metric": "original_profile_receipt_integrity", "value": 1, "metric_direction": "higher_better",
                "unit": "checked proposition", "claim": proposition, "decided_proposition": proposition,
                "source_class": "verifier", "binding_kind": "identity",
                "extra": {"not_model_correctness": True, "not_trial_validation": True}}
            output = {"schema": tp.PROFILE_OUTPUT_SCHEMA, "profile_content": profile.to_dict(),
                "loaded_identity": body["loaded_identity"], "artifact_identity": artifact.to_dict(),
                "measurement_carrier": {"schema": "epyc.autokernel.profile_measurement_carrier.v1",
                    "profile_source_id": tp.PROFILE_SOURCE_ID, "validation_source_id": tp.VALIDATION_SOURCE_ID,
                    "run_id": run_id, "profile_claim_tuple": profile_tuple, "validation_claim_tuple": validation_tuple}}
            if len(_canonical(output)) > MAX_STDOUT_BYTES:
                raise CpuProfileRefused("compact profile output exceeds bound")
            return output
        except BaseException as exc:
            if capture is not None:
                capture.abort(f"{type(exc).__name__}: {exc}")
            store.write("cpu-profile-failure:" + _digest(request), {"schema": CAPTURE_SCHEMA,
                "request_digest": _digest(request), "completion": "failed",
                "failure": f"{type(exc).__name__}: {exc}"[:4096],
                "phases": [] if capture is None else capture.phases})
            raise


def _interrupt(signum, frame):
    raise CpuProfileRefused(f"profile worker interrupted by signal {signum}")


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1 or len(arguments[0].encode()) > MAX_CONFIG_BYTES:
        raise CpuProfileRefused("producer expects one bounded JSON request")
    reference = _closed(json.loads(os.environ[CONFIG_ENV]), {"path", "sha256"}, "config reference")
    raw = _read(Path(reference["path"]), MAX_CONFIG_BYTES)
    _same(hashlib.sha256(raw).hexdigest(), reference["sha256"], "original config")
    original = {signum: signal.signal(signum, _interrupt) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        output = run_profile_request(json.loads(arguments[0]), json.loads(raw))
    finally:
        for signum, handler in original.items():
            signal.signal(signum, handler)
    sys.stdout.buffer.write(_canonical(output) + b"\n")
    return 0
